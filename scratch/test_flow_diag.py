import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/adityabohra/nids")
sys.path.insert(0, "/home/adityabohra/nids2")
sys.path.insert(0, "/home/adityabohra/nids2/nids_m/nids2")

from backend.adapter import IntegratedNIDS, rust_flow_to_raw_77, rust_flow_to_model_features, PRODUCTION_90_FEATURES
from ml.experiments.feature_engineer import RAW_77_FEATURES, add_engineered_features

with open("/home/adityabohra/nids2/scratch/http_completed_flow.json") as f:
    flow = json.load(f)

adapter_inst = IntegratedNIDS()
res = adapter_inst.process_flow(flow)
print("Pipeline Result:")
print(json.dumps(res, indent=2))

detector = adapter_inst.detector
df_raw, feat_90 = rust_flow_to_model_features(flow)
df_eng = add_engineered_features(df_raw)

# Breakdown of anomaly components
X_size = detector.scaler_size.transform(df_eng[detector.size_cols].values.astype(np.float32))
recon_size = detector.pca_size.inverse_transform(detector.pca_size.transform(X_size))
err_size = np.mean((X_size - recon_size) ** 2, axis=1)[0]

X_all = detector.scaler_all.transform(df_eng[detector.feature_names].values.astype(np.float32))
with torch.no_grad():
    x_t = torch.tensor(X_all, dtype=torch.float32)
    recon_ae = detector.autoencoder(x_t).numpy()
    err_ae = np.mean((X_all - recon_ae) ** 2, axis=1)[0]
    sq_errs = (X_all[0] - recon_ae[0]) ** 2

err_iso = -detector.iso.score_samples(X_all)[0]
err_unc = detector.classifier.get_uncertainty_scores(df_eng[detector.feature_names].values.astype(np.float32))[0]

w_s = detector.weights["size_pca"]
w_ae = detector.weights["global_ae"]
w_iso = detector.weights["isolation_forest"]
w_unc = detector.weights["uncertainty"]

zs = (err_size - detector.norm_stats["size_pca"][0]) / detector.norm_stats["size_pca"][1]
zae = (err_ae - detector.norm_stats["global_ae"][0]) / detector.norm_stats["global_ae"][1]
ziso = (err_iso - detector.norm_stats["isolation_forest"][0]) / detector.norm_stats["isolation_forest"][1]
zunc = (err_unc - detector.norm_stats["uncertainty"][0]) / detector.norm_stats["uncertainty"][1]

print("\nZ-Scores:")
print(f"zs   (Size PCA):         {zs:.4f}  (weight: {w_s}) -> contribution: {w_s * zs:.4f}")
print(f"zae  (Global AE):        {zae:.4f}  (weight: {w_ae}) -> contribution: {w_ae * zae:.4f}")
print(f"ziso (Isolation Forest): {ziso:.4f}  (weight: {w_iso}) -> contribution: {w_iso * ziso:.4f}")
print(f"zunc (Uncertainty):      {zunc:.4f}  (weight: {w_unc}) -> contribution: {w_unc * zunc:.4f}")
total_score = w_s * zs + w_ae * zae + w_iso * ziso + w_unc * zunc
print(f"Total anomaly score:     {total_score:.4f}")
print(f"Threshold:               {detector.threshold:.4f}")

# Top AE error contributors
top_idx = np.argsort(sq_errs)[::-1][:20]
print("\nTop 20 AE Squared Error contributors:")
print(f"{'Feature':<30} | {'Orig':<10} | {'Scaled':<8} | {'Recon':<8} | {'SqErr':<8} | {'% of Total':<8}")
for idx in top_idx:
    fname = detector.feature_names[idx]
    orig_val = df_eng[fname].values[0]
    scaled_val = X_all[0, idx]
    rec_val = recon_ae[0, idx]
    se = sq_errs[idx]
    pct = (se / (err_ae * 90.0)) * 100.0
    print(f"{fname:<30} | {orig_val:<10.3f} | {scaled_val:<8.4f} | {rec_val:<8.4f} | {se:<8.5f} | {pct:<8.2f}%")
