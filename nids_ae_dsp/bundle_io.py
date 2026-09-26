"""
bundle_io.py

I/O utilities for loading and saving AE-DSP model bundles generated using
the official AE-DSP repository methodology (https://github.com/Feiyu0Li/AE-DSP).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .core.ae_dsp_model import AutoEncoder, AEDSPFoundation, PretrainedAEDSP
from .core.classifier import AEDSPClassifier
from .core.anomaly_detector import AEDSPAnomalyDetector
from .core.risk_engine import RiskEngine


def load_aedsp_bundle(
    bundle_dir: str,
    device: str = "cpu",
    latent_dim: int = 8,
    hidden_dim: int = 64,
) -> AEDSPFoundation:
    """
    Load an AE-DSP model bundle generated locally using the official AE-DSP
    experiment pipeline (e.g., experiments/nsl_kdd/run_p6_final_zeroday_sensitive_system.py).
    """
    p = Path(bundle_dir)
    if not p.exists():
        raise FileNotFoundError(f"Bundle directory does not exist: {bundle_dir}")

    # 1. Load numpy arrays and scalers
    arrays_path = p / "final_arrays_and_scalers.npz"
    if not arrays_path.exists():
        raise FileNotFoundError(f"Missing final_arrays_and_scalers.npz in {bundle_dir}")
    data = np.load(arrays_path)

    top_idx = data["top_idx"]
    mu_normal = data["mu_normal"]
    std_normal = data["std_normal"]
    ae_med = float(data["ae_med"][0])
    ae_scale = float(data["ae_scale"][0])
    dsp_med = float(data["dsp_med"][0])
    dsp_scale = float(data["dsp_scale"][0])
    dsp_sigma = float(data["dsp_sigma"][0])
    threshold = float(data["threshold"][0])
    alpha = float(data["alpha"][0]) if "alpha" in data else 0.8
    gamma = float(data["gamma"][0]) if "gamma" in data else 1.8

    # 2. Load sklearn / joblib models
    ae_scaler = joblib.load(p / "ae_scaler.joblib")
    dsp_kmeans = joblib.load(p / "dsp_kmeans.joblib")

    preprocessor = None
    prep_path = p / "preprocessor.joblib"
    if prep_path.exists():
        preprocessor = joblib.load(prep_path)

    base_tree_model = None
    lgbm_path = p / "lightgbm.joblib"
    if lgbm_path.exists():
        base_tree_model = joblib.load(lgbm_path)

    # 3. Load feature names
    feature_names = []
    feat_path = p / "selected_feature_names.json"
    if feat_path.exists():
        with open(feat_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

    # 4. Load AutoEncoder weights
    input_dim = len(top_idx)
    autoencoder = AutoEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    )
    ae_state_path = p / "ae_state_dict.pt"
    if ae_state_path.exists() and torch is not None:
        state_dict = torch.load(ae_state_path, map_location=device)
        autoencoder.load_state_dict(state_dict)

    return AEDSPFoundation(
        autoencoder=autoencoder,
        ae_scaler=ae_scaler,
        top_idx=top_idx,
        mu_normal=mu_normal,
        std_normal=std_normal,
        dsp_kmeans=dsp_kmeans,
        dsp_sigma=dsp_sigma,
        ae_med=ae_med,
        ae_scale=ae_scale,
        dsp_med=dsp_med,
        dsp_scale=dsp_scale,
        threshold=threshold,
        alpha=alpha,
        gamma=gamma,
        preprocessor=preprocessor,
        base_tree_model=base_tree_model,
        feature_names=feature_names,
        device=device,
    )


# Backward-compatibility alias
load_official_aedsp_bundle = load_aedsp_bundle


def save_aedsp_bundle(
    aedsp: AEDSPFoundation,
    output_dir: str,
) -> None:
    """
    Save an AEDSPFoundation model bundle following the artifact format defined in
    the official AE-DSP repository.
    """
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)

    # Save sklearn models
    joblib.dump(aedsp.ae_scaler, p / "ae_scaler.joblib")
    joblib.dump(aedsp.dsp_kmeans, p / "dsp_kmeans.joblib")
    if aedsp.preprocessor is not None:
        joblib.dump(aedsp.preprocessor, p / "preprocessor.joblib")
    if aedsp.base_tree_model is not None:
        joblib.dump(aedsp.base_tree_model, p / "lightgbm.joblib")

    # Save AutoEncoder
    torch.save(aedsp.autoencoder.state_dict(), p / "ae_state_dict.pt")

    # Save arrays and scalers
    np.savez(
        p / "final_arrays_and_scalers.npz",
        top_idx=aedsp.top_idx,
        mu_normal=aedsp.mu_normal,
        std_normal=aedsp.std_normal,
        ae_med=np.array([aedsp.ae_med]),
        ae_scale=np.array([aedsp.ae_scale]),
        dsp_med=np.array([aedsp.dsp_med]),
        dsp_scale=np.array([aedsp.dsp_scale]),
        dsp_sigma=np.array([aedsp.dsp_sigma]),
        threshold=np.array([aedsp.threshold]),
        alpha=np.array([aedsp.alpha]),
        gamma=np.array([aedsp.gamma]),
    )

    # Save feature names
    with open(p / "selected_feature_names.json", "w", encoding="utf-8") as f:
        json.dump(aedsp.feature_names, f, ensure_ascii=False, indent=2)


# Backward-compatibility alias
save_official_aedsp_bundle = save_aedsp_bundle
