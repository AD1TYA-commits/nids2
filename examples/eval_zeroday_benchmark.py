"""
eval_zeroday_benchmark.py

Reproducible evaluation script comparing:
1. Baseline AE-DSP NIDS (single binary threshold tau = 2.1198)
2. Improved AE-DSP Decision Layer (3-level anomaly thresholding + lightweight behavioral rules)

Evaluated on the REAL NSL-KDD benchmark (KDDTest+, 22,544 samples).
"""

import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from nids_ae_dsp.bundle_io import load_aedsp_bundle
from nids_ae_dsp.core.behavioral import extract_behavioral_indicators


def run_evaluation():
    print("=" * 78)
    print("        AE-DSP ZERO-DAY INTRUSION DETECTION BENCHMARK EVALUATION")
    print("=" * 78)
    print("[1/5] Loading locally generated AE-DSP bundle...")
    bundle_path = "AE-DSP/results/nsl_kdd/model_bundle_seed42"
    bundle = load_aedsp_bundle(bundle_path)
    print(f"      Bundle loaded successfully from: {bundle_path}")
    print(f"      Top-k SHAP features: {bundle.topk}, DSP Prototypes: {bundle.proto_k}")
    print(f"      Baseline calibrated threshold (tau_high): {bundle.threshold:.4f}")

    print("\n[2/5] Loading real NSL-KDD benchmark datasets...")
    test_path = "AE-DSP/data/nsl_kdd/KDDTest+.txt"
    train_path = "AE-DSP/data/nsl_kdd/KDDTrain+.txt"

    cols = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
        "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
        "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
    ]

    test_df = pd.read_csv(test_path, header=None, names=cols)
    test_df["label"] = test_df["label"].astype(str).str.strip().str.rstrip(".")

    train_df = pd.read_csv(train_path, header=None)
    train_labels = set(train_df[41].astype(str).str.strip().str.rstrip(".")) - {"normal"}

    labels = test_df["label"].values
    y_true = (labels != "normal").astype(int)

    known_mask = np.isin(labels, list(train_labels)) & (y_true == 1)
    unseen_mask = (~np.isin(labels, list(train_labels))) & (y_true == 1)
    normal_mask = (y_true == 0)

    print(f"      Total Test Samples     : {len(test_df):,}")
    print(f"      Normal Samples         : {int(np.sum(normal_mask)):,}")
    print(f"      Known Attack Samples   : {int(np.sum(known_mask)):,}")
    print(f"      Unseen Zero-Day Samples: {int(np.sum(unseen_mask)):,}")

    print("\n[3/5] Extracting AE-DSP foundation representations...")
    feature_cols = [c for c in cols if c not in ["label", "difficulty"]]
    X_raw = test_df[feature_cols]
    X_pre = bundle.preprocessor.transform(X_raw)
    if hasattr(X_pre, "toarray"):
        X_pre = X_pre.toarray()

    shap_vals = bundle.base_tree_model.predict(X_pre, pred_contrib=True)[:, :-1]
    X_topk = shap_vals[:, bundle.top_idx]

    # AutoEncoder reconstruction
    X_ae = bundle.ae_scaler.transform(X_topk).astype(np.float32)
    with torch.no_grad():
        x_t = torch.tensor(X_ae, dtype=torch.float32)
        recon, _ = bundle.autoencoder(x_t)
        res2 = ((recon - x_t) ** 2).cpu().numpy()
        ae_scores = np.mean(res2, axis=1)

    ae_z = (ae_scores - bundle.ae_med) / bundle.ae_scale

    # DSP prototype similarity
    delta = (X_topk - bundle.mu_normal) / (bundle.std_normal + 1e-8)
    centers = bundle.dsp_kmeans.cluster_centers_
    d2 = np.sum((delta[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    dist = np.sqrt(np.min(d2, axis=1))
    dp_raw = np.exp(-dist / (bundle.dsp_sigma + 1e-8))
    dsp_z = (dp_raw - bundle.dsp_med) / bundle.dsp_scale

    # AE-DSP Fusion score
    fusion = bundle.alpha * ae_z + bundle.gamma * dsp_z

    print("\n[4/5] Evaluating Decision Layers...")
    # Baseline: single threshold tau = 2.1198
    th_high = bundle.threshold
    th_med = 0.8111  # 75th percentile of normal validation data
    baseline_pred = (fusion >= th_high).astype(int)

    # Improved: 3-level anomaly thresholding + lightweight behavioral indicators
    behavioral_dict = extract_behavioral_indicators(test_df)
    behavioral_flag = behavioral_dict["has_behavioral_flag"]

    # Decision rule:
    # 1. High anomaly score (fusion >= th_high)
    # 2. Medium anomaly score (th_med <= fusion < th_high) with behavioral flag
    # 3. Behavioral flag with AE manifold deviation (ae_z > 0.5)
    improved_pred = (
        (fusion >= th_high)
        | ((fusion >= th_med) & behavioral_flag)
        | (behavioral_flag & (ae_z > 0.5))
    ).astype(int)

    # Compute Metrics
    # Baseline metrics
    prec_base, rec_base, f1_base, _ = precision_recall_fscore_support(y_true, baseline_pred, average="binary")
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true, baseline_pred).ravel()
    fpr_base = fp_b / (fp_b + tn_b)
    zd_rec_base = np.mean(baseline_pred[unseen_mask])
    known_rec_base = np.mean(baseline_pred[known_mask])

    # Improved metrics
    prec_new, rec_new, f1_new, _ = precision_recall_fscore_support(y_true, improved_pred, average="binary")
    tn_n, fp_n, fn_n, tp_n = confusion_matrix(y_true, improved_pred).ravel()
    fpr_new = fp_n / (fp_n + tn_n)
    zd_rec_new = np.mean(improved_pred[unseen_mask])
    known_rec_new = np.mean(improved_pred[known_mask])

    print("\n[5/5] Comparative Evaluation Summary:")
    print("=" * 78)
    print(f"{'Metric':<25} | {'Baseline (AE-DSP)':<18} | {'Improved Decision Layer':<24} | {'Gain'}")
    print("-" * 78)
    print(f"{'Overall Zero-Day Recall':<25} | {zd_rec_base * 100:>16.2f}% | {zd_rec_new * 100:>22.2f}% | {+(zd_rec_new - zd_rec_base) * 100:>+5.2f}%")
    print(f"{'Known Attack Recall':<25} | {known_rec_base * 100:>16.2f}% | {known_rec_new * 100:>22.2f}% | {+(known_rec_new - known_rec_base) * 100:>+5.2f}%")
    print(f"{'Precision':<25} | {prec_base * 100:>16.2f}% | {prec_new * 100:>22.2f}% | {+(prec_new - prec_base) * 100:>+5.2f}%")
    print(f"{'Overall Recall':<25} | {rec_base * 100:>16.2f}% | {rec_new * 100:>22.2f}% | {+(rec_new - rec_base) * 100:>+5.2f}%")
    print(f"{'F1-Score':<25} | {f1_base * 100:>16.2f}% | {f1_new * 100:>22.2f}% | {+(f1_new - f1_base) * 100:>+5.2f}%")
    print(f"{'False Positive Rate':<25} | {fpr_base * 100:>16.2f}% | {fpr_new * 100:>22.2f}% | {+(fpr_new - fpr_base) * 100:>+5.2f}%")
    print("=" * 78)

    print("\n" + "=" * 78)
    print("             DETAILED PER-UNSEEN-ATTACK RECALL BREAKDOWN")
    print("=" * 78)
    target_unseen = sorted(list(set(labels[unseen_mask])))
    total_prev_missed = 0
    total_newly_caught = 0

    for att in target_unseen:
        m = (labels == att)
        sup = int(np.sum(m))
        b_det = int(np.sum(baseline_pred[m]))
        n_det = int(np.sum(improved_pred[m]))
        b_rec = b_det / sup * 100
        n_rec = n_det / sup * 100
        newly = n_det - b_det
        if b_det < sup:
            total_prev_missed += (sup - b_det)
            total_newly_caught += max(0, newly)
        status_marker = "  [RECOVERED]" if newly > 0 else "             "
        print(f"  {att:<15} | Support: {sup:>4} | Base: {b_det:>4}/{sup:>4} ({b_rec:>6.2f}%) -> New: {n_det:>4}/{sup:>4} ({n_rec:>6.2f}%) | +{newly:>3} caught{status_marker}")

    print("-" * 78)
    print(f"Total Previously Missed Zero-Day Samples : {total_prev_missed}")
    print(f"Newly Detected Zero-Day Samples          : {total_newly_caught}")
    recovery_rate = total_newly_caught / total_prev_missed * 100 if total_prev_missed > 0 else 0
    print(f"Zero-Day Recovery Rate                   : {recovery_rate:.2f}% of missed attacks recovered")
    print("=" * 78)


if __name__ == "__main__":
    run_evaluation()
