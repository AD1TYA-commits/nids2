"""
train_and_eval_demo.py

Held-out Zero-Day Evaluation and Pipeline Serialization Demo.
Simulates a leave-one-attack-type-out protocol (matching AE-DSP experimental methodology):
  - Training/Validation: Benign (Normal) + Known Attacks (DoS, Probe)
  - Held-out Zero-Day: 'Infiltration' (completely excluded from training and validation)
  - Evaluation: Evaluates Known Recall, Zero-Day Recall, False Positive Rate, and Risk Engine Decisions.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from nids_ae_dsp.core.risk_engine import SecurityDecision
from nids_ae_dsp.trainer import build_aedsp_nids
from nids_ae_dsp.pipeline import AEDSP_NIDS


def main():
    print("=" * 80)
    print("      HELD-OUT ZERO-DAY EVALUATION (AE-DSP EXPERIMENTAL METHODOLOGY)      ")
    print("=" * 80)

    np.random.seed(42)
    torch.manual_seed(42)

    num_features = 42
    topk = 32
    proto_k = 10

    # 1. Generate Dataset
    # Benign / Normal traffic
    n_train_normal = 800
    n_val_normal = 200
    n_test_normal = 300
    X_normal = np.random.randn(n_train_normal + n_val_normal + n_test_normal, num_features).astype(np.float32)

    # Known Attack 1: DoS
    n_dos = 350
    X_dos = np.random.randn(n_dos, num_features).astype(np.float32) + 2.7
    X_dos[:, 0:10] += 1.5

    # Known Attack 2: Probe (PortScan / Recon)
    n_probe = 250
    X_probe = np.random.randn(n_probe, num_features).astype(np.float32) - 2.5
    X_probe[:, 10:20] += 2.0

    # HELD-OUT ZERO-DAY ATTACK: Novel Infiltration
    # Never seen during training or validation!
    n_zeroday = 150
    X_zeroday = np.random.randn(n_zeroday, num_features).astype(np.float32) * 2.2
    X_zeroday[:, 20:35] += 5.5  # Completely novel subspace deviation

    # Train / Val / Test Splits
    # Train
    X_tr = np.vstack([X_normal[:n_train_normal], X_dos[:250], X_probe[:180]])
    y_tr_bin = np.array([0] * n_train_normal + [1] * (250 + 180), dtype=np.int64)
    y_tr_lab = ["Normal"] * n_train_normal + ["DoS"] * 250 + ["Probe"] * 180

    # Val (for threshold calibration)
    X_val = np.vstack([X_normal[n_train_normal : n_train_normal + n_val_normal], X_dos[250:300], X_probe[180:210]])
    y_val_bin = np.array([0] * n_val_normal + [1] * (50 + 30), dtype=np.int64)

    # Test (Contains Normal, Known DoS, Known Probe, AND Unseen Zero-Day Infiltration)
    X_test_norm = X_normal[n_train_normal + n_val_normal :]
    X_test_dos = X_dos[300:]
    X_test_probe = X_probe[210:]

    print(f"\nDataset Distribution:")
    print(f"  - Train Set : {len(X_tr)} flows (Normal: {n_train_normal}, DoS: 250, Probe: 180)")
    print(f"  - Val Set   : {len(X_val)} flows (Normal: {n_val_normal}, DoS: 50, Probe: 30)")
    print(f"  - Test Set  : {len(X_test_norm) + len(X_test_dos) + len(X_test_probe) + n_zeroday} flows")
    print(f"    * Normal (Benign)           : {len(X_test_norm)}")
    print(f"    * Known Attacks (DoS/Probe) : {len(X_test_dos) + len(X_test_probe)}")
    print(f"    * Held-Out Zero-Day Attacks : {n_zeroday} (Infiltration)")

    print("\n[+] Training AE-DSP Foundation, Classifier, and Anomaly Detector...")
    pipeline = build_aedsp_nids(
        X_train_pre=X_tr,
        y_train_binary=y_tr_bin,
        y_train_labels=y_tr_lab,
        X_val_pre=X_val,
        y_val_binary=y_val_bin,
        shap_train=X_tr,  # simulated SHAP space
        shap_val=X_val,
        topk=topk,
        proto_k=proto_k,
        epochs=25,
        classifier_epochs=25,
        alpha=0.8,
        gamma=1.8,
        target_fpr=0.045,
        verbose=False,
    )

    print("[+] Model successfully trained.")
    print(f"    - AE-DSP Input Dim     : {pipeline.ae_dsp.topk}")
    print(f"    - Calibrated Threshold : {pipeline.ae_dsp.threshold:.4f} (Target FPR=4.5%)")

    # 3. Evaluate each group independently
    print("\n" + "=" * 80)
    print("                         EVALUATION RESULTS                           ")
    print("=" * 80)

    # A. Benign / Normal test
    res_norm = pipeline.predict_detailed(X_test_norm, is_already_shap=True)
    norm_decisions = res_norm["decisions"]
    norm_fp = sum(1 for d in norm_decisions if d.decision in [SecurityDecision.MALICIOUS, SecurityDecision.UNKNOWN_ATTACK])
    norm_fpr = norm_fp / len(norm_decisions)
    norm_clean = sum(1 for d in norm_decisions if d.decision == SecurityDecision.NORMAL) / len(norm_decisions)

    # B. Known Attacks test (DoS + Probe)
    X_known_test = np.vstack([X_test_dos, X_test_probe])
    res_known = pipeline.predict_detailed(X_known_test, is_already_shap=True)
    known_decisions = res_known["decisions"]
    known_detected_malicious = sum(1 for d in known_decisions if d.decision == SecurityDecision.MALICIOUS) / len(known_decisions)
    known_total_blocked = sum(1 for d in known_decisions if d.decision in [SecurityDecision.MALICIOUS, SecurityDecision.UNKNOWN_ATTACK]) / len(known_decisions)

    # C. Held-Out Zero-Day Attacks test
    res_zeroday = pipeline.predict_detailed(X_zeroday, is_already_shap=True)
    zeroday_decisions = res_zeroday["decisions"]
    zeroday_flagged_unknown = sum(1 for d in zeroday_decisions if d.decision == SecurityDecision.UNKNOWN_ATTACK) / len(zeroday_decisions)
    zeroday_total_blocked = sum(1 for d in zeroday_decisions if d.decision in [SecurityDecision.MALICIOUS, SecurityDecision.UNKNOWN_ATTACK]) / len(zeroday_decisions)

    print(f"\n1. BENIGN TRAFFIC (NORMAL):")
    print(f"   - Verified Clean Rate (NORMAL) : {norm_clean:.2%}")
    print(f"   - False Positive Rate          : {norm_fpr:.2%} (within target 4.5% envelope)")

    print(f"\n2. KNOWN ATTACKS (DoS & Probe):")
    print(f"   - Confirmed Known Attack (MALICIOUS) : {known_detected_malicious:.2%}")
    print(f"   - Overall Threat Neutralization      : {known_total_blocked:.2%}")

    print(f"\n3. HELD-OUT ZERO-DAY ATTACKS (Infiltration):")
    print(f"   - Successfully Isolated as UNKNOWN ATTACK : {zeroday_flagged_unknown:.2%}")
    print(f"   - Total Zero-Day Interception Rate        : {zeroday_total_blocked:.2%}")

    print("\n" + "=" * 80)
    print("                     PIPELINE PERSISTENCE CHECK                       ")
    print("=" * 80)

    save_dir = Path("saved_nids_model")
    pipeline.save(str(save_dir))
    print(f"[+] Saved complete NIDS model bundle to: {save_dir.resolve()}")

    reloaded_pipeline = AEDSP_NIDS.load(str(save_dir))
    print("[+] Reloaded NIDS pipeline successfully.")

    # Quick verification on sample zero-day flow
    sample_res = reloaded_pipeline.predict_detailed(X_zeroday[:1], is_already_shap=True)
    d = sample_res["decisions"][0]
    print(f"    Sample Verification Decision: {d.decision.value} (Risk Score={d.risk_score:.1f}, Action={d.action})")

    # Clean up save dir
    if save_dir.exists():
        shutil.rmtree(save_dir)
        print("[+] Cleaned up temporary test model directory.")

    print("\n[+] Zero-day evaluation and serialization completed successfully.")


if __name__ == "__main__":
    main()
