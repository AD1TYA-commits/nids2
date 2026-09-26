"""
demo_live_inspection.py

Demonstration of Live Traffic Inspection with Pretrained AE-DSP NIDS Architecture.
Illustrates:
  Pretrained AE-DSP
        │
        ▼
  AE-DSP representation/features
        │
  ┌─────┴───────────────┐
  │                     │
  ▼                     ▼
Classifier       Anomaly Detector
  │                     │
  └─────┬───────────────┘
        ▼
   Risk Engine
        ▼
NORMAL / SUSPICIOUS / MALICIOUS / UNKNOWN ATTACK
"""

import sys
from pathlib import Path

# Add project root to Python search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from nids_ae_dsp.core.risk_engine import SecurityDecision
from nids_ae_dsp.trainer import build_aedsp_nids


def main():
    print("=" * 80)
    print("       NIDS INSPECTION ENGINE ON PRETRAINED AE-DSP FRAMEWORK       ")
    print("       Repository: https://github.com/Feiyu0Li/AE-DSP             ")
    print("=" * 80)

    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Setup synthetic network flow dataset with known & unseen zero-day attacks
    n_normal = 400
    n_dos = 120
    n_probe = 100
    num_features = 40
    topk = 30
    proto_k = 8

    # Normal traffic centered around origin
    X_normal = np.random.randn(n_normal, num_features).astype(np.float32)

    # Known attacks in training set
    X_dos = np.random.randn(n_dos, num_features).astype(np.float32) + 2.8
    X_probe = np.random.randn(n_probe, num_features).astype(np.float32) - 2.8

    X_train = np.vstack([X_normal, X_dos, X_probe])
    y_binary = np.array([0] * n_normal + [1] * (n_dos + n_probe), dtype=np.int64)
    y_labels = ["Normal"] * n_normal + ["DoS"] * n_dos + ["Probe"] * n_probe

    print("\n[+] Initializing Pretrained AE-DSP Core & Components...")
    pipeline = build_aedsp_nids(
        X_train_pre=X_train,
        y_train_binary=y_binary,
        y_train_labels=y_labels,
        shap_train=X_train,  # SHAP space
        topk=topk,
        proto_k=proto_k,
        latent_dim=8,
        epochs=20,
        classifier_epochs=20,
        alpha=0.8,
        gamma=1.8,
        target_fpr=0.045,
        verbose=False,
    )

    print("[+] Model initialized successfully.")
    print(f"    - AE-DSP Input Dim     : {pipeline.ae_dsp.topk}")
    print(f"    - AE Latent Bottleneck : {pipeline.ae_dsp.latent_dim}")
    print(f"    - DSP Prototypes (K)   : {pipeline.ae_dsp.proto_k}")
    print(f"    - Calibrated Threshold : {pipeline.ae_dsp.threshold:.4f} (Target FPR=4.5%)")
    print(f"    - Known Classes        : {pipeline.classifier.class_names}")

    # 2. Prepare test cases representing the 4 required security scenarios
    print("\n" + "=" * 80)
    print("                    LIVE TRAFFIC STREAM INSPECTION                    ")
    print("=" * 80)

    test_cases = [
        {
            "name": "Flow #1: Standard HTTPS Web Browsing",
            "data": np.random.randn(1, num_features).astype(np.float32) * 0.8,
            "expected": SecurityDecision.NORMAL,
        },
        {
            "name": "Flow #2: SYN-Flood DoS Signature Attack",
            "data": np.random.randn(1, num_features).astype(np.float32) + 2.9,
            "expected": SecurityDecision.MALICIOUS,
        },
        {
            "name": "Flow #3: PortScan / Reconnaissance Probe",
            "data": np.random.randn(1, num_features).astype(np.float32) - 2.9,
            "expected": SecurityDecision.MALICIOUS,
        },
        {
            "name": "Flow #4: Ambiguous / High Jitter Session",
            # Borderline features around 1.3
            "data": np.random.randn(1, num_features).astype(np.float32) + 1.2,
            "expected": SecurityDecision.SUSPICIOUS,
        },
        {
            "name": "Flow #5: ZERO-DAY Polymorphic Infiltration (Novel Attack)",
            # Severe orthogonal deviation never seen in training
            "data": np.random.randn(1, num_features).astype(np.float32) * 2.5 + 7.5,
            "expected": SecurityDecision.UNKNOWN_ATTACK,
        },
    ]

    for tc in test_cases:
        print(f"\n>>> INCOMING: {tc['name']}")
        detailed = pipeline.predict_detailed(tc["data"], is_already_shap=True)

        rep = detailed["representation"]
        cls_res = detailed["classifier_result"]
        anom_res = detailed["anomaly_result"]
        decision = detailed["decisions"][0]

        print(f"  [1] Pretrained AE-DSP Representation:")
        print(f"      - Latent Embedding z   : {np.round(rep.latent[0, :4], 3)} ... (dim={rep.latent.shape[1]})")
        print(f"      - AE Reconstruction res: AE_score={rep.ae_score[0]:.4f} | AE_z={rep.ae_z[0]:.3f}")
        print(f"      - Nearest Attack Proto : #{rep.nearest_proto_id[0]} (Distance={rep.nearest_proto_dist[0]:.3f})")
        print(f"      - DSP Similarity Score : DSP_z={rep.dsp_z[0]:.3f}")
        print(f"      - AE-DSP Fusion Score  : {rep.fusion_score[0]:.4f} (Threshold={anom_res.threshold:.4f})")

        print(f"  [2] Component 1: Classifier (Known Classes):")
        print(f"      - Predicted Class      : {cls_res.predicted_class[0]}")
        print(f"      - Confidence           : {cls_res.confidence[0]:.2%}")
        probs_str = ", ".join([f"{k}: {v:.2%}" for k, v in cls_res.get_sample(0)["probabilities"].items()])
        print(f"      - Class Probabilities  : [{probs_str}]")
        print(f"      - Predictive Entropy   : {cls_res.entropy[0]:.4f}")

        print(f"  [3] Component 2: Anomaly Detector (Zero-Day / Novel):")
        print(f"      - Anomaly Status       : {'ANOMALY DETECTED' if anom_res.is_anomaly[0] else 'NORMAL PROFILE'}")
        print(f"      - Novelty / Zero-day   : {'OUT-OF-DISTRIBUTION NOVEL PATTERN' if anom_res.is_novel_pattern[0] else 'KNOWN MANIFOLD'}")
        print(f"      - Novelty Score        : {anom_res.novelty_score[0]:.3f}")

        # Render colored/highlighted decision badge
        badge = decision.decision.value
        print(f"  [4] Decision Layer: Risk Engine:")
        print(f"      =======================================================")
        print(f"      FINAL SECURITY DECISION: >>> [ {badge} ] <<<")
        print(f"      Risk Score             : {decision.risk_score:.1f} / 100")
        print(f"      Security Action        : {decision.action}")
        print(f"      Audit Explanation      : {decision.reason}")
        print(f"      =======================================================")

    print("\n[+] Live inspection demo completed successfully.")


if __name__ == "__main__":
    main()
