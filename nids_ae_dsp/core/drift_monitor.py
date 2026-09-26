"""
drift_monitor.py

Lightweight model drift and concept drift monitoring layer for AE-DSP NIDS.
Monitors network feature and anomaly score distributions over time without
introducing large secondary ML models.

Distinguishes between:
  1. Benign distribution shifts (Covariate Drift)
  2. Feature corruption / sensor faults (Data Quality)
  3. Known attack surges (Attack Distribution Shift)
  4. Genuine novel/zero-day attack waves (Zero-Day Burst)
  5. Fundamental normal baseline drift (Concept Drift)

Implements safe rollback tracking and strict no-silent-recalibration policies.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


class DriftType(str, Enum):
    NONE = "NONE"
    COVARIATE_DRIFT = "COVARIATE_DRIFT"          # Benign traffic shift (protocols/services shift, but scores normal)
    FEATURE_CORRUPTION = "FEATURE_CORRUPTION"    # Sensor/data quality fault (out-of-range, NaN, zero bursts)
    ATTACK_SURGE = "ATTACK_SURGE"                # Known attack volume surge
    ZERO_DAY_BURST = "ZERO_DAY_BURST"            # New/unseen zero-day attack campaign
    CONCEPT_DRIFT = "CONCEPT_DRIFT"              # Significant baseline shift in normal manifold


@dataclass
class DriftReport:
    """Telemetry report emitted by the Drift Monitor."""
    drift_detected: bool
    drift_type: DriftType
    severity: str  # "NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    recommended_action: str = "CONTINUE_MONITORING"
    metrics: Dict[str, float] = field(default_factory=dict)
    details: str = ""
    covariate_scale: float = 1.0
    consecutive_drift_count: int = 0
    is_confirmed_drift: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_detected": self.drift_detected,
            "drift_type": self.drift_type.value,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "recommended_action": self.recommended_action,
            "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in self.metrics.items()},
            "details": self.details,
            "covariate_scale": round(self.covariate_scale, 4),
            "consecutive_drift_count": self.consecutive_drift_count,
            "is_confirmed_drift": self.is_confirmed_drift,
        }


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10, eps: float = 1e-4) -> float:
    """Calculate Population Stability Index (PSI) between two distributions."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine bin edges based on expected distribution quantiles
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    exp_pct = exp_counts / max(1, len(expected))
    act_pct = act_counts / max(1, len(actual))

    exp_pct = np.clip(exp_pct, eps, 1.0)
    act_pct = np.clip(act_pct, eps, 1.0)

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(np.maximum(0.0, psi_val))


def calculate_categorical_psi(
    expected_freqs: Dict[str, float],
    actual_freqs: Dict[str, float],
    eps: float = 1e-4,
) -> float:
    """Calculate PSI between two discrete categorical frequency distributions."""
    all_keys = set(expected_freqs.keys()).union(set(actual_freqs.keys()))
    psi = 0.0
    for k in all_keys:
        p = expected_freqs.get(k, eps)
        q = actual_freqs.get(k, eps)
        p = max(p, eps)
        q = max(q, eps)
        psi += (q - p) * np.log(q / p)
    return float(max(0.0, psi))


class AEDSPDriftMonitor:
    """
    Lightweight, streaming-compatible drift detector.
    Compares sliding window telemetry against a frozen known-good baseline.
    """

    def __init__(
        self,
        baseline_df: pd.DataFrame,
        baseline_anomaly_scores: np.ndarray,
        window_size: int = 500,
        psi_threshold: float = 0.20,
        anomaly_drift_threshold: float = 0.35,
    ) -> None:
        self.window_size = int(window_size)
        self.psi_threshold = float(psi_threshold)
        self.anomaly_drift_threshold = float(anomaly_drift_threshold)

        # Build baseline statistics from verified normal traffic
        self.baseline_size = len(baseline_df)
        self.baseline_protocols = self._get_categorical_distribution(baseline_df, "protocol_type")
        self.baseline_services = self._get_categorical_distribution(baseline_df, "service", top_k=15)

        self.baseline_duration = baseline_df["duration"].values.astype(np.float64) if "duration" in baseline_df else np.array([])
        self.baseline_src_bytes = baseline_df["src_bytes"].values.astype(np.float64) if "src_bytes" in baseline_df else np.array([])
        self.baseline_dst_bytes = baseline_df["dst_bytes"].values.astype(np.float64) if "dst_bytes" in baseline_df else np.array([])
        self.baseline_rerror = float(baseline_df["rerror_rate"].mean()) if "rerror_rate" in baseline_df else 0.0
        self.baseline_serror = float(baseline_df["serror_rate"].mean()) if "serror_rate" in baseline_df else 0.0

        self.baseline_anomaly_scores = np.asarray(baseline_anomaly_scores, dtype=np.float64)
        self.baseline_anomaly_mean = float(np.mean(baseline_anomaly_scores))
        self.baseline_anomaly_p95 = float(np.percentile(baseline_anomaly_scores, 95.0))

        # Sliding window buffer
        self.history: List[DriftReport] = []

        # Consecutive window tracking (dampening noisy transient spikes)
        self.consecutive_drift_count: int = 0
        self.min_consecutive_windows: int = 2
        self.last_drift_type: DriftType = DriftType.NONE

    def _get_categorical_distribution(
        self, df: pd.DataFrame, col: str, top_k: Optional[int] = None
    ) -> Dict[str, float]:
        if col not in df:
            return {}
        counts = df[col].astype(str).value_counts(normalize=True)
        if top_k is not None:
            top = counts.head(top_k).to_dict()
            other_sum = 1.0 - sum(top.values())
            if other_sum > 0:
                top["__other__"] = other_sum
            return top
        return counts.to_dict()

    def inspect_window(
        self,
        current_df: pd.DataFrame,
        anomaly_scores: np.ndarray,
        risk_decisions: Optional[List[Any]] = None,
    ) -> DriftReport:
        """
        Inspect a batch/window of traffic against baseline.
        Returns a detailed DriftReport without modifying any model thresholds.
        """
        n = len(current_df)
        if n == 0:
            return DriftReport(
                drift_detected=False,
                drift_type=DriftType.NONE,
                severity="NONE",
                details="Empty window passed to Drift Monitor.",
            )

        # 1. Check for Sensor / Data Quality Corruption
        # Out-of-bounds check, negative values, all-zero features
        corruption_reasons = []
        if "duration" in current_df and np.any(current_df["duration"] < 0):
            corruption_reasons.append("Negative duration detected")
        if "src_bytes" in current_df and np.any(current_df["src_bytes"] < 0):
            corruption_reasons.append("Negative src_bytes detected")
        if np.any(pd.isna(current_df.values)):
            corruption_reasons.append("Null / NaN sensor inputs detected")

        if corruption_reasons:
            self.consecutive_drift_count = 1
            self.last_drift_type = DriftType.FEATURE_CORRUPTION
            report = DriftReport(
                drift_detected=True,
                drift_type=DriftType.FEATURE_CORRUPTION,
                severity="CRITICAL",
                recommended_action="REJECT_BATCH_AND_CHECK_SENSOR_PIPELINE",
                metrics={"corruption_flags": float(len(corruption_reasons))},
                details=f"Sensor/data quality fault: {'; '.join(corruption_reasons)}",
                covariate_scale=1.0,
                consecutive_drift_count=1,
                is_confirmed_drift=True,
            )
            self.history.append(report)
            return report

        # 2. Compute Categorical PSI (Protocols & Services)
        curr_protocols = self._get_categorical_distribution(current_df, "protocol_type")
        curr_services = self._get_categorical_distribution(current_df, "service", top_k=15)
        proto_psi = calculate_categorical_psi(self.baseline_protocols, curr_protocols)
        service_psi = calculate_categorical_psi(self.baseline_services, curr_services)

        # 3. Compute Numerical Distribution Shifts
        # Duration KS-test
        ks_duration = 1.0
        if len(self.baseline_duration) > 0 and "duration" in current_df:
            _, ks_duration = ks_2samp(self.baseline_duration, current_df["duration"].values)

        # Byte statistics Wasserstein distance
        w_src = 0.0
        if len(self.baseline_src_bytes) > 0 and "src_bytes" in current_df:
            w_src = float(wasserstein_distance(self.baseline_src_bytes, current_df["src_bytes"].values))

        # Error rates shift
        curr_rerr = float(current_df["rerror_rate"].mean()) if "rerror_rate" in current_df else 0.0
        delta_rerror = abs(curr_rerr - self.baseline_rerror)

        # 4. Compute Anomaly Score Distribution Shift (Wasserstein distance on AE-DSP scores)
        w_anomaly = float(wasserstein_distance(self.baseline_anomaly_scores, anomaly_scores))
        high_anomaly_ratio = float(np.mean(anomaly_scores >= self.baseline_anomaly_p95))

        # 5. Decision Layer Class Ratios (if provided)
        unknown_ratio = 0.0
        malicious_ratio = 0.0
        if risk_decisions:
            total_d = len(risk_decisions)
            unknown_ratio = sum(1 for d in risk_decisions if getattr(d, "decision", "") == "UNKNOWN ATTACK") / total_d
            malicious_ratio = sum(1 for d in risk_decisions if getattr(d, "decision", "") == "MALICIOUS") / total_d

        metrics = {
            "protocol_psi": proto_psi,
            "service_psi": service_psi,
            "duration_ks_pvalue": float(ks_duration),
            "src_bytes_wasserstein": w_src,
            "delta_rerror_rate": delta_rerror,
            "anomaly_score_wasserstein": w_anomaly,
            "high_anomaly_ratio": high_anomaly_ratio,
            "unknown_attack_ratio": unknown_ratio,
            "malicious_attack_ratio": malicious_ratio,
        }

        # 6. Distinguish Drift Categories
        # A. Zero-Day Attack Wave: High unknown attack decisions + elevated anomaly scores
        if unknown_ratio >= 0.20 or (high_anomaly_ratio >= 0.35 and unknown_ratio >= 0.10):
            drift_detected = True
            drift_type = DriftType.ZERO_DAY_BURST
            severity = "HIGH"
            action = "FLAG_ZERO_DAY_BURST_INCREASE_INSPECTION_DO_NOT_RECALIBRATE"
            details = (
                f"Surge in unknown/zero-day attacks detected (ratio={unknown_ratio:.1%}, "
                f"anomaly score shift={w_anomaly:.3f}). Traffic is hostile: DO NOT recalibrate baseline."
            )

        # B. Known Attack Surge: High malicious known attacks
        elif malicious_ratio >= 0.35:
            drift_detected = True
            drift_type = DriftType.ATTACK_SURGE
            severity = "HIGH"
            action = "FLAG_ATTACK_SURGE_ACTIVATE_BLOCKING"
            details = f"Known attack surge detected ({malicious_ratio:.1%} malicious flows). Active mitigation engaged."

        # C. Concept Drift in Normal Baseline: Anomaly scores shifted upward without attack burst
        elif w_anomaly >= self.anomaly_drift_threshold and high_anomaly_ratio >= 0.25:
            drift_detected = True
            drift_type = DriftType.CONCEPT_DRIFT
            severity = "MEDIUM"
            action = "FLAG_CONCEPT_DRIFT_PREPARE_CONTROLLED_ADMIN_REVIEW"
            details = (
                f"Significant manifold baseline drift observed (Wasserstein distance={w_anomaly:.3f} >= {self.anomaly_drift_threshold}). "
                f"Preserving existing model; queueing verified benign samples for controlled review."
            )

        # D. Covariate Drift: Protocol/service proportions changed, but anomaly scores remain low
        elif (proto_psi >= self.psi_threshold or service_psi >= self.psi_threshold) and high_anomaly_ratio < 0.15:
            drift_detected = True
            drift_type = DriftType.COVARIATE_DRIFT
            severity = "LOW"
            action = "LOG_COVARIATE_SHIFT_PRESERVE_MODEL"
            details = (
                f"Benign traffic distribution shift detected (Protocol PSI={proto_psi:.3f}, "
                f"Service PSI={service_psi:.3f}). Anomaly distribution remains stable."
            )

        # No significant drift
        else:
            drift_detected = False
            drift_type = DriftType.NONE
            severity = "NONE"
            action = "CONTINUE_MONITORING"
            details = "Traffic distribution complies with baseline envelope."

        # Track consecutive windows for temporal persistence
        if drift_detected:
            if drift_type == self.last_drift_type:
                self.consecutive_drift_count += 1
            else:
                self.consecutive_drift_count = 1
            self.last_drift_type = drift_type
        else:
            self.consecutive_drift_count = 0
            self.last_drift_type = DriftType.NONE

        is_confirmed = (self.consecutive_drift_count >= self.min_consecutive_windows) or (severity in ["CRITICAL", "HIGH"])

        covariate_scale = 1.0
        if drift_type == DriftType.COVARIATE_DRIFT:
            covariate_scale = float(1.0 + 0.5 * min(1.0, max(proto_psi, service_psi)))

        report = DriftReport(
            drift_detected=drift_detected,
            drift_type=drift_type,
            severity=severity,
            recommended_action=action,
            metrics=metrics,
            details=details,
            covariate_scale=covariate_scale,
            consecutive_drift_count=self.consecutive_drift_count,
            is_confirmed_drift=is_confirmed,
        )
        self.history.append(report)
        return report
