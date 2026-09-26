"""
nids_ae_dsp: NIDS Model built on AE-DSP Framework.
Official Repository: https://github.com/Feiyu0Li/AE-DSP
"""

from .core.ae_dsp_model import AutoEncoder, AEDSPRepresentation, AEDSPFoundation, PretrainedAEDSP
from .core.classifier import AEDSPClassifier, ClassifierResult
from .core.anomaly_detector import AEDSPAnomalyDetector, AnomalyResult
from .core.risk_engine import RiskEngine, SecurityDecision, RiskDecision
from .pipeline import AEDSP_NIDS
from .bundle_io import load_aedsp_bundle, save_aedsp_bundle, load_official_aedsp_bundle, save_official_aedsp_bundle
from .trainer import build_aedsp_nids

__version__ = "1.0.0"

__all__ = [
    "AutoEncoder",
    "AEDSPRepresentation",
    "AEDSPFoundation",
    "PretrainedAEDSP",
    "AEDSPClassifier",
    "ClassifierResult",
    "AEDSPAnomalyDetector",
    "AnomalyResult",
    "RiskEngine",
    "SecurityDecision",
    "RiskDecision",
    "AEDSP_NIDS",
    "load_aedsp_bundle",
    "save_aedsp_bundle",
    "load_official_aedsp_bundle",
    "save_official_aedsp_bundle",
    "build_aedsp_nids",
]
