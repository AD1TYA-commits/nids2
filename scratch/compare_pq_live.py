import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/adityabohra/nids")
sys.path.insert(0, "/home/adityabohra/nids2")

from backend.adapter import IntegratedNIDS, rust_flow_to_raw_77, rust_flow_to_model_features, PRODUCTION_90_FEATURES
from ml.experiments.feature_engineer import RAW_77_FEATURES, add_engineered_features

detector = IntegratedNIDS().detector

# Load parquet normal sample
df_all = pd.read_parquet("/home/adityabohra/nids/ml/data/working_dataset_all77.parquet")
match = df_all[(df_all["StandardLabel"] == "Normal") & (df_all["Protocol"] == 6.0) & (df_all["Total Fwd Packets"] == 5) & (df_all["Total Backward Packets"] == 5)]
sample_pq = match.iloc[0][RAW_77_FEATURES].to_dict()

# Load live flow
with open("/home/adityabohra/nids2/scratch/http_completed_flow.json") as f:
    live_flow = json.load(f)
sample_live = rust_flow_to_raw_77(live_flow)

df_pq = pd.DataFrame([sample_pq])[RAW_77_FEATURES]
df_live = pd.DataFrame([sample_live])[RAW_77_FEATURES]

eng_pq = add_engineered_features(df_pq)
eng_live = add_engineered_features(df_live)

X_pq = detector.scaler_all.transform(eng_pq[detector.feature_names].values.astype(np.float32))
X_live = detector.scaler_all.transform(eng_live[detector.feature_names].values.astype(np.float32))

with torch.no_grad():
    x_pq_t = torch.tensor(X_pq, dtype=torch.float32)
    x_live_t = torch.tensor(X_live, dtype=torch.float32)
    rec_pq = detector.autoencoder(x_pq_t).numpy()
    rec_live = detector.autoencoder(x_live_t).numpy()
    se_pq = (X_pq[0] - rec_pq[0]) ** 2
    se_live = (X_live[0] - rec_live[0]) ** 2

print(f"{'Feature Name':<28} | {'PQ Raw':<10} | {'Live Raw':<10} | {'PQ Scale':<8} | {'Live Scl':<8} | {'Live SqErr':<10}")
print("-" * 88)
# Sort by highest live squared error
diff_indices = np.argsort(se_live)[::-1]
for idx in diff_indices:
    fname = detector.feature_names[idx]
    v_pq = eng_pq[fname].values[0]
    v_live = eng_live[fname].values[0]
    s_pq = X_pq[0, idx]
    s_live = X_live[0, idx]
    sqe = se_live[idx]
    print(f"{fname:<28} | {v_pq:<10.2f} | {v_live:<10.2f} | {s_pq:<8.3f} | {s_live:<8.3f} | {sqe:<10.5f}")
