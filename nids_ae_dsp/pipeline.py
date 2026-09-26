"""
pipeline.py

Unified NIDS Architecture built on Pretrained AE-DSP.

Conceptual Pipeline:
    Pretrained AE-DSP
           │
           ▼
    AE-DSP representation/features
           │
    ┌──────┴──────────────┐
    │                     │
    ▼                     ▼
Classifier         Anomaly Detector
(Known Classes)     (Zero-Day / Novel)
    │                     │
    └──────┬──────────────┘
           ▼
      Risk Engine
    (Decision Layer)
           │
           ▼
NORMAL / SUSPICIOUS / MALICIOUS / UNKNOWN ATTACK
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

from .core.ae_dsp_model import AutoEncoder, AEDSPRepresentation, AEDSPFoundation, PretrainedAEDSP
from .core.classifier import AEDSPClassifier, ClassifierResult
from .core.anomaly_detector import AEDSPAnomalyDetector, AnomalyResult
from .core.risk_engine import RiskEngine, RiskDecision, SecurityDecision
from .core.behavioral import extract_behavioral_indicators
from .bundle_io import load_aedsp_bundle, save_aedsp_bundle


class AEDSP_NIDS:
    """
    End-to-End Network Intrusion Detection System built strictly on the
    AE-DSP framework (https://github.com/Feiyu0Li/AE-DSP).
    """

    def __init__(
        self,
        ae_dsp: AEDSPFoundation,
        classifier: AEDSPClassifier,
        anomaly_detector: AEDSPAnomalyDetector,
        risk_engine: RiskEngine,
    ) -> None:
        self.ae_dsp = ae_dsp
        self.classifier = classifier
        self.anomaly_detector = anomaly_detector
        self.risk_engine = risk_engine

    @classmethod
    def from_aedsp_bundle(
        cls,
        bundle_dir: str,
        class_names: List[str],
        device: str = "cpu",
        hidden_dim: int = 64,
        latent_dim: int = 8,
        novelty_sensitivity: float = 1.25,
    ) -> "AEDSP_NIDS":
        """
        Initialize the complete NIDS architecture from a locally generated AE-DSP model bundle
        (produced using the official AE-DSP experiment pipeline).
        """
        ae_dsp = load_aedsp_bundle(
            bundle_dir=bundle_dir,
            device=device,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )

        classifier = AEDSPClassifier(
            representation_dim=ae_dsp.representation_dim,
            class_names=class_names,
            hidden_dim=hidden_dim,
            device=device,
        )

        anomaly_detector = AEDSPAnomalyDetector(
            threshold=ae_dsp.threshold,
            threshold_med=getattr(ae_dsp, "threshold_med", 0.8111),
            alpha=ae_dsp.alpha,
            gamma=ae_dsp.gamma,
            novelty_sensitivity=novelty_sensitivity,
        )

        risk_engine = RiskEngine()

        return cls(
            ae_dsp=ae_dsp,
            classifier=classifier,
            anomaly_detector=anomaly_detector,
            risk_engine=risk_engine,
        )

    # Backward-compatibility alias
    from_pretrained_bundle = from_aedsp_bundle

    def extract_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        is_already_shap: bool = False,
    ) -> AEDSPRepresentation:
        """Step 1: Extract rich representation from Pretrained AE-DSP foundation."""
        return self.ae_dsp.extract_representation(X, is_already_shap=is_already_shap)

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        is_already_shap: bool = False,
    ) -> List[RiskDecision]:
        """
        Run end-to-end NIDS inference and return final security decisions.
        """
        detailed = self.predict_detailed(X, is_already_shap=is_already_shap)
        return detailed["decisions"]

    def predict_detailed(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        is_already_shap: bool = False,
    ) -> Dict[str, Any]:
        """
        Run complete architecture and expose full telemetry from all stages:
          1. AE-DSP representation
          2. Classifier results
          3. Anomaly Detector results
          4. Risk Engine decisions
        """
        # Step 1: Pretrained AE-DSP Representation
        representation = self.extract_features(X, is_already_shap=is_already_shap)

        # Step 2: Classifier for Known Classes
        classifier_result = self.classifier.predict(representation)

        # Step 3: Anomaly Detector for Unknown / Zero-Day Attacks (3-level)
        anomaly_result = self.anomaly_detector.detect(representation)

        # Step 4: Extract Behavioral Indicators (if raw DataFrame provided)
        behavioral_indicators = None
        if isinstance(X, pd.DataFrame):
            try:
                behavioral_indicators = extract_behavioral_indicators(X)
            except Exception:
                behavioral_indicators = None

        # Step 5: Deterministic Risk Engine Decision Layer
        decisions = self.risk_engine.evaluate_batch(
            classifier_result,
            anomaly_result,
            behavioral_indicators=behavioral_indicators,
        )

        # Compute summary counts
        counts = {
            SecurityDecision.NORMAL.value: 0,
            SecurityDecision.SUSPICIOUS.value: 0,
            SecurityDecision.MALICIOUS.value: 0,
            SecurityDecision.UNKNOWN_ATTACK.value: 0,
            SecurityDecision.CRITICAL.value: 0,
        }
        for d in decisions:
            counts[d.decision.value] += 1


        return {
            "representation": representation,
            "classifier_result": classifier_result,
            "anomaly_result": anomaly_result,
            "decisions": decisions,
            "summary_counts": counts,
        }

    def save(self, output_dir: str) -> None:
        """Save the entire NIDS pipeline (AE-DSP core + components)."""
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)

        # 1. Save AE-DSP bundle
        core_dir = p / "aedsp_core"
        save_aedsp_bundle(self.ae_dsp, str(core_dir))

        # 2. Save Classifier
        classifier_dir = p / "classifier"
        classifier_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.classifier.state_dict(), classifier_dir / "classifier.pt")

        # 3. Save Anomaly Detector and Risk Engine configs
        config = {
            "threshold": self.anomaly_detector.threshold,
            "alpha": self.anomaly_detector.alpha,
            "gamma": self.anomaly_detector.gamma,
            "proto_dist_p95": self.anomaly_detector.proto_dist_p95,
            "novelty_sensitivity": self.anomaly_detector.novelty_sensitivity,
            "class_names": self.classifier.class_names,
            "representation_dim": self.classifier.representation_dim,
        }
        with open(p / "pipeline_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, model_dir: str, device: str = "cpu") -> "AEDSP_NIDS":
        """Load a saved AEDSP_NIDS pipeline."""
        p = Path(model_dir)
        with open(p / "pipeline_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        ae_dsp = load_aedsp_bundle(str(p / "aedsp_core"), device=device)

        classifier = AEDSPClassifier(
            representation_dim=config["representation_dim"],
            class_names=config["class_names"],
            device=device,
        )
        classifier_state = torch.load(p / "classifier" / "classifier.pt", map_location=device)
        classifier.load_state_dict(classifier_state)

        anomaly_detector = AEDSPAnomalyDetector(
            threshold=config["threshold"],
            alpha=config["alpha"],
            gamma=config["gamma"],
            proto_dist_p95=config.get("proto_dist_p95"),
            novelty_sensitivity=config.get("novelty_sensitivity", 1.25),
        )

        risk_engine = RiskEngine()

        return cls(
            ae_dsp=ae_dsp,
            classifier=classifier,
            anomaly_detector=anomaly_detector,
            risk_engine=risk_engine,
        )

    def summary(self) -> str:
        """Generate human-readable architectural summary."""
        lines = [
            "==========================================================================",
            "                   AE-DSP NIDS ARCHITECTURE SUMMARY                       ",
            "==========================================================================",
            "AE-DSP Foundation: AutoEncoder + Deviation Semantic Prototypes (AE-DSP)",
            f"  - Input Top-k Features : {self.ae_dsp.topk}",
            f"  - Latent Dimension     : {self.ae_dsp.latent_dim}",
            f"  - DSP Prototypes (K)   : {self.ae_dsp.proto_k}",
            f"  - Fusion Weights       : alpha={self.ae_dsp.alpha}, gamma={self.ae_dsp.gamma}",
            f"  - Calibrated Threshold : tau={self.ae_dsp.threshold:.4f}",
            "--------------------------------------------------------------------------",
            "AE-DSP Representation Vector:",
            f"  - Total Dimension      : {self.ae_dsp.representation_dim}",
            "    [Latent (z) | DSP Similarities (K) | AE_z | DSP_z | Proto_Dist | Fusion]",
            "--------------------------------------------------------------------------",
            "Component 1: Classifier (Known Classes Head)",
            f"  - Target Classes       : {self.classifier.class_names}",
            f"  - Architecture         : Linear({self.ae_dsp.representation_dim} -> 64 -> {len(self.classifier.class_names)})",
            "  - Output Metrics       : Predicted Class, Confidence (Probability), Entropy",
            "--------------------------------------------------------------------------",
            "Component 2: Anomaly Detector (Zero-Day / Novel Attacks)",
            "  - Mechanics            : Reconstruction Error (AE_z) + DSP Prototype Deviation",
            f"  - Decision Rule        : Score >= {self.anomaly_detector.threshold:.4f} -> Anomaly",
            "  - Novelty Detection    : Prototype distance boundary breach & high AE deviation",
            "--------------------------------------------------------------------------",
            "Decision Layer: Risk Engine (Deterministic Security Rule Layer)",
            "  - Inputs               : Classifier (Class, Conf) + Anomaly Detector (Score, Novelty)",
            "  - Output Categories    : NORMAL | SUSPICIOUS | MALICIOUS | UNKNOWN ATTACK",
            "==========================================================================",
        ]
        return "\n".join(lines)
