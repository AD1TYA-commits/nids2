"""
anomaly_detector.py

Component 2: Anomaly Detector built on Pretrained AE-DSP Representations.
Detects abnormal traffic and identifies unknown/novel/zero-day attacks
using AE-DSP manifold deviation and prototype distance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .ae_dsp_model import AEDSPRepresentation


@dataclass
class AnomalyResult:
    """Output for a batch or single prediction of the Anomaly Detector."""
    anomaly_score: np.ndarray  # AE-DSP fusion score [batch_size]
    is_anomaly: np.ndarray  # boolean decision (score >= threshold) [batch_size]
    threshold: float  # calibrated high decision threshold tau_high (95.5th percentile of normal)
    ae_z: np.ndarray  # AE reconstruction deviation z-score [batch_size]
    dsp_z: np.ndarray  # DSP prototype similarity z-score [batch_size]
    nearest_proto_id: np.ndarray  # closest attack prototype ID [batch_size]
    nearest_proto_dist: np.ndarray  # distance to closest prototype [batch_size]
    novelty_score: np.ndarray  # deviation indicating novel/zero-day behavior [batch_size]
    is_novel_pattern: np.ndarray  # flag for unknown/zero-day pattern [batch_size]
    threshold_med: float = 0.8111  # calibrated medium threshold tau_med (75th percentile of normal)
    anomaly_level: Optional[np.ndarray] = None  # 3-level decision: 'LOW', 'MEDIUM', 'HIGH' [batch_size]

    def __len__(self) -> int:
        return len(self.anomaly_score)

    def get_sample(self, idx: int) -> Dict[str, Any]:
        """Get result for a single sample."""
        level = str(self.anomaly_level[idx]) if self.anomaly_level is not None else ("HIGH" if self.is_anomaly[idx] else "LOW")
        return {
            "anomaly_score": float(self.anomaly_score[idx]),
            "anomaly_level": level,
            "is_anomaly": bool(self.is_anomaly[idx]),
            "threshold": float(self.threshold),
            "threshold_med": float(self.threshold_med),
            "ae_z": float(self.ae_z[idx]),
            "dsp_z": float(self.dsp_z[idx]),
            "nearest_proto_id": int(self.nearest_proto_id[idx]),
            "nearest_proto_dist": float(self.nearest_proto_dist[idx]),
            "novelty_score": float(self.novelty_score[idx]),
            "is_novel_pattern": bool(self.is_novel_pattern[idx]),
        }


class AEDSPAnomalyDetector:
    """
    Anomaly Detector grounded directly in the AE-DSP framework:
      - Uses AE reconstruction error z-score (ae_z)
      - Uses DSP prototype similarity z-score (dsp_z)
      - Computes AE-DSP fusion score: alpha * ae_z + gamma * dsp_z
      - Evaluates novelty in prototype space to identify unknown/zero-day attacks
    """

    def __init__(
        self,
        threshold: float,
        threshold_med: Optional[float] = None,
        alpha: float = 0.8,
        gamma: float = 1.8,
        proto_dist_p95: Optional[float] = None,
        novelty_sensitivity: float = 1.25,
    ) -> None:
        self.threshold = float(threshold)
        self.threshold_med = float(threshold_med) if threshold_med is not None else 0.8111
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.proto_dist_p95 = proto_dist_p95
        self.novelty_sensitivity = novelty_sensitivity

    def calibrate_novelty_threshold(
        self,
        train_attack_proto_dists: np.ndarray,
        percentile: float = 95.0,
    ) -> None:
        """
        Calibrate the distance boundary for known attack prototypes.
        Samples exceeding this boundary in prototype space indicate novel/zero-day attacks.
        """
        dists = np.asarray(train_attack_proto_dists, dtype=np.float64)
        self.proto_dist_p95 = float(np.percentile(dists, percentile))

    def detect(
        self,
        representation: AEDSPRepresentation,
    ) -> AnomalyResult:
        """
        Run anomaly detection and unknown/zero-day attack assessment
        using the pretrained AE-DSP representation.
        """
        ae_z = representation.ae_z
        dsp_z = representation.dsp_z
        fusion_score = representation.fusion_score
        nearest_proto_dist = representation.nearest_proto_dist
        nearest_proto_id = representation.nearest_proto_id

        # 3-level decision system:
        # High anomaly (fusion >= threshold): UNKNOWN ATTACK candidate / Anomaly
        # Medium anomaly (threshold_med <= fusion < threshold): SUSPICIOUS candidate
        # Low anomaly (fusion < threshold_med): NORMAL candidate
        is_anomaly = (fusion_score >= self.threshold)
        anomaly_level = np.where(
            fusion_score >= self.threshold,
            "HIGH",
            np.where(fusion_score >= self.threshold_med, "MEDIUM", "LOW")
        )

        # Novelty / Zero-day assessment:
        # A zero-day attack deviates strongly from normal traffic (high AE reconstruction error),
        # but fails to match known attack prototypes (high prototype distance, low/negative DSP similarity).
        if self.proto_dist_p95 is not None and self.proto_dist_p95 > 0:
            dist_ratio = nearest_proto_dist / (self.proto_dist_p95 + 1e-8)
        else:
            median_dist = float(np.median(nearest_proto_dist)) + 1e-8
            dist_ratio = nearest_proto_dist / median_dist

        # Novelty score combines high manifold deviation with large prototype distance
        # ae_z positive & large, plus dist_ratio > 1
        novelty_score = np.maximum(0.0, ae_z) * np.maximum(0.5, dist_ratio)

        # Novel pattern flag:
        # High anomaly, significant AE violation, and prototype distance exceeding known envelope
        is_novel_pattern = is_anomaly & (dist_ratio >= self.novelty_sensitivity) & (ae_z > 1.5)

        return AnomalyResult(
            anomaly_score=fusion_score,
            is_anomaly=is_anomaly,
            threshold=self.threshold,
            threshold_med=self.threshold_med,
            ae_z=ae_z,
            dsp_z=dsp_z,
            nearest_proto_id=nearest_proto_id,
            nearest_proto_dist=nearest_proto_dist,
            novelty_score=novelty_score,
            is_novel_pattern=is_novel_pattern,
            anomaly_level=anomaly_level,
        )
