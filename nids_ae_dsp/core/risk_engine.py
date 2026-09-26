"""
risk_engine.py

Decision Layer: Risk Engine.
Not an ML model. A deterministic security rule layer that combines:
  - Classifier prediction and confidence
  - Anomaly score and anomaly decision
  - Known vs unknown attack indication
and produces the final security decision:
  - NORMAL
  - SUSPICIOUS
  - MALICIOUS
  - UNKNOWN ATTACK
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .classifier import ClassifierResult
from .anomaly_detector import AnomalyResult


class SecurityDecision(str, Enum):
    """The five canonical security decisions in the multi-tier policy."""
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    MALICIOUS = "MALICIOUS"
    UNKNOWN_ATTACK = "UNKNOWN ATTACK"
    CRITICAL = "CRITICAL"


@dataclass
class RiskDecision:
    """Detailed evaluation output for a single network flow or sample."""
    decision: SecurityDecision
    risk_score: float  # Range: 0.0 to 100.0
    action: str  # e.g., "ALLOW", "INSPECT", "BLOCK", "QUARANTINE"
    reason: str  # Transparent natural language audit explanation
    predicted_class: str
    confidence: float
    anomaly_score: float
    threshold: float
    is_anomaly: bool
    is_known_attack: bool
    is_unknown_attack: bool
    nearest_proto_id: int
    nearest_proto_dist: float
    is_critical: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "risk_score": round(self.risk_score, 2),
            "action": self.action,
            "reason": self.reason,
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "anomaly_score": round(self.anomaly_score, 4),
            "threshold": round(self.threshold, 4),
            "is_anomaly": self.is_anomaly,
            "is_known_attack": self.is_known_attack,
            "is_unknown_attack": self.is_unknown_attack,
            "is_critical": self.is_critical,
            "nearest_proto_id": self.nearest_proto_id,
            "nearest_proto_dist": round(self.nearest_proto_dist, 4),
        }



class RiskEngine:
    """
    Deterministic security decision engine.
    Combines Classifier and Anomaly Detector outputs without any ML model inside.
    """

    def __init__(
        self,
        normal_class_names: Optional[List[str]] = None,
        high_confidence_threshold: float = 0.55,
        low_confidence_threshold: float = 0.40,
        borderline_margin_ratio: float = 0.15,
    ) -> None:
        self.normal_class_names = set(
            [n.lower() for n in (normal_class_names or ["normal", "benign"])]
        )
        self.high_conf_th = float(high_confidence_threshold)
        self.low_conf_th = float(low_confidence_threshold)
        self.borderline_ratio = float(borderline_margin_ratio)

    def is_normal_label(self, label: str) -> bool:
        """Check if label corresponds to a benign/normal traffic type."""
        return str(label).strip().lower() in self.normal_class_names

    def evaluate_sample(
        self,
        pred_class: str,
        confidence: float,
        anomaly_score: float,
        threshold: float,
        is_anomaly: bool,
        ae_z: float,
        dsp_z: float,
        nearest_proto_id: int,
        nearest_proto_dist: float,
        is_novel_pattern: bool,
        has_behavioral_flag: bool = False,
        behavioral_reason: Optional[str] = None,
        threshold_med: Optional[float] = None,
    ) -> RiskDecision:
        """
        Evaluate a single flow using deterministic security decision logic.
        """
        is_pred_normal = self.is_normal_label(pred_class)
        th_med = float(threshold_med) if threshold_med is not None else 0.8111
        is_medium_anomaly = (anomaly_score >= th_med)
        borderline_margin = max(0.2, abs(threshold) * self.borderline_ratio)
        is_borderline_anomaly = (threshold - borderline_margin) <= anomaly_score < threshold

        # ---------------------------------------------------------------------
        # Case A: Check for High-Severity Critical Exploits
        # ---------------------------------------------------------------------
        b_reason_str = str(behavioral_reason or "").lower()
        is_exploit = "exploit" in b_reason_str
        is_priv = "privilege" in b_reason_str and ae_z >= 1.5

        if is_exploit or is_priv:
            return RiskDecision(
                decision=SecurityDecision.CRITICAL,
                risk_score=98.0,
                action="EMERGENCY BLOCK & ISOLATE (CRITICAL THREAT)",
                reason=(
                    f"Critical security violation detected! Threat signature: "
                    f"{behavioral_reason or 'Exploit payload / privilege escalation'} "
                    f"(Score={anomaly_score:.3f}, AE_z={ae_z:.2f}, Prototype Dist={nearest_proto_dist:.2f})."
                ),
                predicted_class=pred_class,
                confidence=confidence,
                anomaly_score=anomaly_score,
                threshold=threshold,
                is_anomaly=True,
                is_known_attack=False,
                is_unknown_attack=True,
                is_critical=True,
                nearest_proto_id=nearest_proto_id,
                nearest_proto_dist=nearest_proto_dist,
            )


        # ---------------------------------------------------------------------
        # Count Independent Corroborating Signals
        # ---------------------------------------------------------------------
        corroborating_signals = 0
        if ae_z >= 1.5:
            corroborating_signals += 1
        if is_novel_pattern or nearest_proto_dist >= 12.0 or dsp_z < -1.0:
            corroborating_signals += 1
        if has_behavioral_flag:
            corroborating_signals += 1
        if (not is_pred_normal) or confidence < self.high_conf_th:
            corroborating_signals += 1

        # ---------------------------------------------------------------------
        # Case B: Traffic is flagged as Anomalous by AE-DSP (is_anomaly == True)
        # ---------------------------------------------------------------------
        if is_anomaly:
            # Subcase B0: Extreme Prototype Rejection / Severe Manifold Rupture (Zero-Day)
            # When DSP prototype similarity is severely rejected (DSP_z < -2.5) with high AE deviation,
            # or extreme novelty, traffic is mathematically out-of-distribution even if linear classifier has a leaning.
            if (dsp_z < -3.0) or (is_novel_pattern and dsp_z < -2.5 and ae_z >= 2.5):
                risk_score = min(100.0, 92.0 + max(0.0, (anomaly_score - threshold) * 2.0))
                return RiskDecision(
                    decision=SecurityDecision.UNKNOWN_ATTACK,
                    risk_score=risk_score,
                    action="QUARANTINE & DISSECT (OUT-OF-DISTRIBUTION ATTACK)",
                    reason=(
                        f"Novel zero-day attack detected: extreme prototype rejection (DSP_z={dsp_z:.2f}) "
                        f"and manifold deviation (AE_z={ae_z:.2f}) confirm flow is completely "
                        f"out-of-distribution from known attack prototypes."
                    ),
                    predicted_class=pred_class,
                    confidence=confidence,
                    anomaly_score=anomaly_score,
                    threshold=threshold,
                    is_anomaly=True,
                    is_known_attack=False,
                    is_unknown_attack=True,
                    nearest_proto_id=nearest_proto_id,
                    nearest_proto_dist=nearest_proto_dist,
                )

            # Subcase B1: Classifier confidently predicts a known attack class
            if (not is_pred_normal) and (confidence >= self.high_conf_th):
                risk_score = min(100.0, 75.0 + (confidence * 20.0) + max(0.0, (anomaly_score - threshold) * 1.5))
                return RiskDecision(
                    decision=SecurityDecision.MALICIOUS,
                    risk_score=risk_score,
                    action="BLOCK & ALERT",
                    reason=(
                        f"Known attack confirmed: '{pred_class}' with {confidence:.2%} confidence. "
                        f"AE-DSP confirms signature match (Anomaly Score={anomaly_score:.3f}, "
                        f"nearest DSP prototype #{nearest_proto_id}, DSP_z={dsp_z:.2f})."
                    ),
                    predicted_class=pred_class,
                    confidence=confidence,
                    anomaly_score=anomaly_score,
                    threshold=threshold,
                    is_anomaly=True,
                    is_known_attack=True,
                    is_unknown_attack=False,
                    nearest_proto_id=nearest_proto_id,
                    nearest_proto_dist=nearest_proto_dist,
                )

            # Subcase B2: Classifier predicts 'Normal' (Evasion / Zero-Day Candidate)
            if is_pred_normal:
                # Require multiple independent pieces of evidence before escalating to UNKNOWN ATTACK
                if corroborating_signals >= 2 or anomaly_score >= 4.0 or has_behavioral_flag:
                    risk_score = min(100.0, 85.0 + max(0.0, (anomaly_score - threshold) * 3.0))
                    return RiskDecision(
                        decision=SecurityDecision.UNKNOWN_ATTACK,
                        risk_score=risk_score,
                        action="QUARANTINE & DISSECT (ZERO-DAY ALERT)",
                        reason=(
                            f"Zero-day attack detected. Classifier predicted '{pred_class}' ({confidence:.2%}), "
                            f"corroborated by multi-source evidence: AE deviation (AE_z={ae_z:.2f}), "
                            f"prototype novelty (DSP_z={dsp_z:.2f}), and {corroborating_signals} corroborating signals."
                        ),
                        predicted_class=pred_class,
                        confidence=confidence,
                        anomaly_score=anomaly_score,
                        threshold=threshold,
                        is_anomaly=True,
                        is_known_attack=False,
                        is_unknown_attack=True,
                        nearest_proto_id=nearest_proto_id,
                        nearest_proto_dist=nearest_proto_dist,
                    )
                else:
                    # Single-signal borderline anomaly without corroborating evidence -> SUSPICIOUS
                    return RiskDecision(
                        decision=SecurityDecision.SUSPICIOUS,
                        risk_score=60.0,
                        action="FLAG FOR DEEP INSPECTION & MONITOR",
                        reason=(
                            f"Borderline anomaly score ({anomaly_score:.3f} >= {threshold:.3f}) "
                            f"without corroborating secondary evidence (signals={corroborating_signals}); "
                            f"held at SUSPICIOUS per evidence policy."
                        ),
                        predicted_class=pred_class,
                        confidence=confidence,
                        anomaly_score=anomaly_score,
                        threshold=threshold,
                        is_anomaly=False,
                        is_known_attack=False,
                        is_unknown_attack=False,
                        nearest_proto_id=nearest_proto_id,
                        nearest_proto_dist=nearest_proto_dist,
                    )

            # Subcase B3: Moderate confidence known attack prediction
            if dsp_z > 0.0 or confidence >= self.low_conf_th:
                if is_novel_pattern:
                    risk_score = min(100.0, 90.0 + max(0.0, (anomaly_score - threshold) * 2.0))
                    return RiskDecision(
                        decision=SecurityDecision.UNKNOWN_ATTACK,
                        risk_score=risk_score,
                        action="QUARANTINE & DISSECT (OUT-OF-DISTRIBUTION ATTACK)",
                        reason=(
                            f"Novel attack pattern detected. AE-DSP manifold deviation is high (AE_z={ae_z:.2f}), "
                            f"and distance to nearest attack prototype #{nearest_proto_id} ({nearest_proto_dist:.2f}) "
                            f"exceeds known prototype boundaries."
                        ),
                        predicted_class=pred_class,
                        confidence=confidence,
                        anomaly_score=anomaly_score,
                        threshold=threshold,
                        is_anomaly=True,
                        is_known_attack=False,
                        is_unknown_attack=True,
                        nearest_proto_id=nearest_proto_id,
                        nearest_proto_dist=nearest_proto_dist,
                    )
                return RiskDecision(
                    decision=SecurityDecision.MALICIOUS,
                    risk_score=min(100.0, 70.0 + (confidence * 20.0)),
                    action="BLOCK & ALERT",
                    reason=(
                        f"Known attack confirmed: '{pred_class}' ({confidence:.2%}) "
                        f"with matching DSP prototype #{nearest_proto_id} (DSP_z={dsp_z:.2f})."
                    ),
                    predicted_class=pred_class,
                    confidence=confidence,
                    anomaly_score=anomaly_score,
                    threshold=threshold,
                    is_anomaly=True,
                    is_known_attack=True,
                    is_unknown_attack=False,
                    nearest_proto_id=nearest_proto_id,
                    nearest_proto_dist=nearest_proto_dist,
                )

            # Subcase B4: Out-of-distribution attack pattern
            if is_novel_pattern:
                risk_score = min(100.0, 90.0 + max(0.0, (anomaly_score - threshold) * 2.0))
                return RiskDecision(
                    decision=SecurityDecision.UNKNOWN_ATTACK,
                    risk_score=risk_score,
                    action="QUARANTINE & DISSECT (OUT-OF-DISTRIBUTION ATTACK)",
                    reason=(
                        f"Novel attack pattern detected. AE-DSP manifold deviation is high (AE_z={ae_z:.2f}), "
                        f"and distance to nearest attack prototype #{nearest_proto_id} ({nearest_proto_dist:.2f}) "
                        f"exceeds known prototype boundaries."
                    ),
                    predicted_class=pred_class,
                    confidence=confidence,
                    anomaly_score=anomaly_score,
                    threshold=threshold,
                    is_anomaly=True,
                    is_known_attack=False,
                    is_unknown_attack=True,
                    nearest_proto_id=nearest_proto_id,
                    nearest_proto_dist=nearest_proto_dist,
                )


            return RiskDecision(
                decision=SecurityDecision.SUSPICIOUS,
                risk_score=65.0,
                action="FLAG FOR DEEP INSPECTION & MONITOR",
                reason=(
                    f"Moderate confidence attack prediction '{pred_class}' ({confidence:.2%}), "
                    f"flagged by AE-DSP anomaly score ({anomaly_score:.3f})."
                ),
                predicted_class=pred_class,
                confidence=confidence,
                anomaly_score=anomaly_score,
                threshold=threshold,
                is_anomaly=True,
                is_known_attack=False,
                is_unknown_attack=False,
                nearest_proto_id=nearest_proto_id,
                nearest_proto_dist=nearest_proto_dist,
            )

        # ---------------------------------------------------------------------
        # Case C: Traffic is NOT flagged as High Anomaly (is_anomaly == False)
        # ---------------------------------------------------------------------
        # Subcase C0: Behavioral indicator combined with AE-DSP deviation
        if has_behavioral_flag and (is_medium_anomaly or ae_z > 0.5):
            if is_pred_normal:
                risk_score = min(100.0, 85.0 + max(0.0, ae_z * 2.5))
                return RiskDecision(
                    decision=SecurityDecision.UNKNOWN_ATTACK,
                    risk_score=risk_score,
                    action="QUARANTINE & DISSECT (EVASIVE ZERO-DAY ALERT)",
                    reason=(
                        f"Unknown evasive attack detected. AE-DSP manifold deviation (AE_z={ae_z:.2f}, "
                        f"Score={anomaly_score:.3f}) combined with behavioral indicator: {behavioral_reason or 'Suspicious protocol pattern'}."
                    ),
                    predicted_class=pred_class,
                    confidence=confidence,
                    anomaly_score=anomaly_score,
                    threshold=threshold,
                    is_anomaly=True,
                    is_known_attack=False,
                    is_unknown_attack=True,
                    nearest_proto_id=nearest_proto_id,
                    nearest_proto_dist=nearest_proto_dist,
                )
            else:
                risk_score = min(100.0, 80.0 + (confidence * 15.0))
                return RiskDecision(
                    decision=SecurityDecision.MALICIOUS,
                    risk_score=risk_score,
                    action="BLOCK & ALERT",
                    reason=(
                        f"Attack behavior confirmed: '{pred_class}' ({confidence:.2%}) with "
                        f"behavioral indicator: {behavioral_reason or 'Suspicious protocol pattern'}."
                    ),
                    predicted_class=pred_class,
                    confidence=confidence,
                    anomaly_score=anomaly_score,
                    threshold=threshold,
                    is_anomaly=True,
                    is_known_attack=True,
                    is_unknown_attack=False,
                    nearest_proto_id=nearest_proto_id,
                    nearest_proto_dist=nearest_proto_dist,
                )

        # Subcase C1: Medium Anomaly / Borderline anomaly score
        if is_medium_anomaly or is_borderline_anomaly:
            return RiskDecision(
                decision=SecurityDecision.SUSPICIOUS,
                risk_score=50.0 + (confidence * 15.0 if not is_pred_normal else 0.0),
                action="FLAG FOR DEEP INSPECTION & MONITOR",
                reason=(
                    f"Medium anomaly score ({anomaly_score:.3f}) in suspicious range [{th_med:.3f}, {threshold:.3f}); "
                    f"predicted class '{pred_class}' ({confidence:.2%})."
                ),
                predicted_class=pred_class,
                confidence=confidence,
                anomaly_score=anomaly_score,
                threshold=threshold,
                is_anomaly=False,
                is_known_attack=False,
                is_unknown_attack=False,
                nearest_proto_id=nearest_proto_id,
                nearest_proto_dist=nearest_proto_dist,
            )

        # Subcase C2: Classifier predicts an attack class even though AE-DSP did not flag anomaly
        if not is_pred_normal:
            if confidence >= self.high_conf_th:
                return RiskDecision(
                    decision=SecurityDecision.MALICIOUS,
                    risk_score=68.0,
                    action="BLOCK & ALERT",
                    reason=(
                        f"Classifier confidently identified '{pred_class}' ({confidence:.2%}), "
                        f"despite low anomaly score ({anomaly_score:.3f})."
                    ),
                    predicted_class=pred_class,
                    confidence=confidence,
                    anomaly_score=anomaly_score,
                    threshold=threshold,
                    is_anomaly=False,
                    is_known_attack=True,
                    is_unknown_attack=False,
                    nearest_proto_id=nearest_proto_id,
                    nearest_proto_dist=nearest_proto_dist,
                )
            else:
                return RiskDecision(
                    decision=SecurityDecision.SUSPICIOUS,
                    risk_score=55.0,
                    action="FLAG FOR DEEP INSPECTION & MONITOR",
                    reason=(
                        f"Classifier signaled '{pred_class}' ({confidence:.2%}), but AE-DSP anomaly score "
                        f"({anomaly_score:.3f}) is below threshold ({threshold:.3f})."
                    ),
                    predicted_class=pred_class,
                    confidence=confidence,
                    anomaly_score=anomaly_score,
                    threshold=threshold,
                    is_anomaly=False,
                    is_known_attack=False,
                    is_unknown_attack=False,
                    nearest_proto_id=nearest_proto_id,
                    nearest_proto_dist=nearest_proto_dist,
                )

        # Subcase C3: Normal traffic verified (Low Anomaly score < th_med)
        norm_risk = max(0.0, 15.0 * (1.0 - confidence) + max(0.0, (anomaly_score / (threshold + 1e-6)) * 10.0))
        return RiskDecision(
            decision=SecurityDecision.NORMAL,
            risk_score=min(25.0, norm_risk),
            action="ALLOW",
            reason=(
                f"Normal traffic verified. Conforms to AE-DSP normal contribution manifold "
                f"(Anomaly Score={anomaly_score:.3f} < {th_med:.3f}). "
                f"Classifier predicted '{pred_class}' with {confidence:.2%} confidence."
            ),
            predicted_class=pred_class,
            confidence=confidence,
            anomaly_score=anomaly_score,
            threshold=threshold,
            is_anomaly=False,
            is_known_attack=False,
            is_unknown_attack=False,
            nearest_proto_id=nearest_proto_id,
            nearest_proto_dist=nearest_proto_dist,
        )


    def evaluate_batch(
        self,
        classifier_result: ClassifierResult,
        anomaly_result: AnomalyResult,
        behavioral_indicators: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[RiskDecision]:
        """
        Evaluate a batch of predictions from Classifier and Anomaly Detector.
        """
        n = len(classifier_result)
        decisions: List[RiskDecision] = []

        has_flags = behavioral_indicators.get("has_behavioral_flag") if behavioral_indicators else None
        reasons = behavioral_indicators.get("behavioral_reason") if behavioral_indicators else None
        th_med = getattr(anomaly_result, "threshold_med", 0.8111)

        for i in range(n):
            h_flag = bool(has_flags[i]) if has_flags is not None else False
            b_reason = str(reasons[i]) if reasons is not None else None
            d = self.evaluate_sample(
                pred_class=classifier_result.predicted_class[i],
                confidence=float(classifier_result.confidence[i]),
                anomaly_score=float(anomaly_result.anomaly_score[i]),
                threshold=float(anomaly_result.threshold),
                is_anomaly=bool(anomaly_result.is_anomaly[i]),
                ae_z=float(anomaly_result.ae_z[i]),
                dsp_z=float(anomaly_result.dsp_z[i]),
                nearest_proto_id=int(anomaly_result.nearest_proto_id[i]),
                nearest_proto_dist=float(anomaly_result.nearest_proto_dist[i]),
                is_novel_pattern=bool(anomaly_result.is_novel_pattern[i]),
                has_behavioral_flag=h_flag,
                behavioral_reason=b_reason,
                threshold_med=th_med,
            )
            decisions.append(d)

        return decisions
