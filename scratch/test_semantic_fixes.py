import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/adityabohra/nids")
sys.path.insert(0, "/home/adityabohra/nids2")
sys.path.insert(0, "/home/adityabohra/nids2/nids_m/nids2")

from backend.adapter import IntegratedNIDS, rust_flow_to_raw_77, rust_flow_to_model_features
from ml.experiments.feature_engineer import RAW_77_FEATURES, add_engineered_features

with open("/home/adityabohra/nids2/scratch/http_completed_flow.json") as f:
    flow = json.load(f)

adapter_inst = IntegratedNIDS()
detector = adapter_inst.detector

raw_77 = rust_flow_to_raw_77(flow)

# Test 1: baseline
df_raw1 = pd.DataFrame([raw_77])[RAW_77_FEATURES]
m_res1 = detector.predict(df_raw1)[0]
print("Baseline:")
print("  Score:", m_res1["anomaly_score"], "Pred:", m_res1["prediction"], "Anomaly:", m_res1["anomaly"])

# Test 2: Bwd PSH Flags = 0.0
raw_77_2 = raw_77.copy()
raw_77_2["Bwd PSH Flags"] = 0.0
df_raw2 = pd.DataFrame([raw_77_2])[RAW_77_FEATURES]
m_res2 = detector.predict(df_raw2)[0]
print("\n+ Bwd PSH = 0.0:")
print("  Score:", m_res2["anomaly_score"], "Pred:", m_res2["prediction"], "Anomaly:", m_res2["anomaly"])

# Test 3: Bwd PSH = 0.0 & FIN = 0.0
raw_77_3 = raw_77.copy()
raw_77_3["Bwd PSH Flags"] = 0.0
raw_77_3["FIN Flag Count"] = 0.0
df_raw3 = pd.DataFrame([raw_77_3])[RAW_77_FEATURES]
m_res3 = detector.predict(df_raw3)[0]
print("\n+ Bwd PSH = 0.0 & FIN = 0.0:")
print("  Score:", m_res3["anomaly_score"], "Pred:", m_res3["prediction"], "Anomaly:", m_res3["anomaly"])

# Test 4: Check other flags: Fwd PSH Flags vs SYN, Packet Length Min
raw_77_4 = raw_77_3.copy()
# In CICFlowMeter, what is Packet Length Min?
# If packet lengths include Ethernet headers in Rust (66 bytes min for TCP), whereas CICFlowMeter was IP packets (min 52 bytes or payload 0 bytes)
print("\nPacket Length Min in raw_77:", raw_77["Packet Length Min"])
print("Fwd Packet Length Min in raw_77:", raw_77["Fwd Packet Length Min"])
print("Bwd Packet Length Min in raw_77:", raw_77["Bwd Packet Length Min"])
print("PSH Flag Count in raw_77:", raw_77["PSH Flag Count"])

df_eng3 = add_engineered_features(df_raw3)
X_size = detector.scaler_size.transform(df_eng3[detector.size_cols].values.astype(np.float32))
recon_size = detector.pca_size.inverse_transform(detector.pca_size.transform(X_size))
err_size = np.mean((X_size - recon_size) ** 2, axis=1)[0]
X_all = detector.scaler_all.transform(df_eng3[detector.feature_names].values.astype(np.float32))
with torch.no_grad():
    x_t = torch.tensor(X_all, dtype=torch.float32)
    recon_ae = detector.autoencoder(x_t).numpy()
    err_ae = np.mean((X_all - recon_ae) ** 2, axis=1)[0]
    sq_errs = (X_all[0] - recon_ae[0]) ** 2

err_iso = -detector.iso.score_samples(X_all)[0]
err_unc = detector.classifier.get_uncertainty_scores(df_eng3[detector.feature_names].values.astype(np.float32))[0]

zs = (err_size - detector.norm_stats["size_pca"][0]) / detector.norm_stats["size_pca"][1]
zae = (err_ae - detector.norm_stats["global_ae"][0]) / detector.norm_stats["global_ae"][1]
ziso = (err_iso - detector.norm_stats["isolation_forest"][0]) / detector.norm_stats["isolation_forest"][1]
zunc = (err_unc - detector.norm_stats["uncertainty"][0]) / detector.norm_stats["uncertainty"][1]
print("\nZ-scores Test 3:")
print(f"zs={zs:.4f}, zae={zae:.4f}, ziso={ziso:.4f}, zunc={zunc:.4f}")
print("Total score:", detector.weights["size_pca"]*zs + detector.weights["global_ae"]*zae + detector.weights["isolation_forest"]*ziso + detector.weights["uncertainty"]*zunc)

top_idx = np.argsort(sq_errs)[::-1][:15]
print("\nRemaining Top 15 AE error contributors in Test 3:")
for idx in top_idx:
    fname = detector.feature_names[idx]
    orig_val = df_eng3[fname].values[0]
    scaled_val = X_all[0, idx]
    rec_val = recon_ae[0, idx]
    se = sq_errs[idx]
    pct = (se / (err_ae * 90.0)) * 100.0
    print(f"  {fname:<30}: orig={orig_val:<10.3f} scaled={scaled_val:<8.4f} recon={rec_val:<8.4f} sq_err={se:.5f} ({pct:.1f}%)")
