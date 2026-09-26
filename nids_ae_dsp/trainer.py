"""
trainer.py

Training and initialization engine strictly adhering to the AE-DSP methodology
defined in the official AE-DSP repository:
  - NSL-KDD: run_p6_final_zeroday_sensitive_system.py
  - CICIDS2017: run_p8a_cicids2017_heldout_validation.py
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    import shap
except ImportError:
    shap = None

from .core.ae_dsp_model import AutoEncoder, PretrainedAEDSP
from .core.classifier import AEDSPClassifier
from .core.anomaly_detector import AEDSPAnomalyDetector
from .core.risk_engine import RiskEngine
from .pipeline import AEDSP_NIDS


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def robust_fit(x: np.ndarray) -> Tuple[float, float]:
    """Official AE-DSP robust median and IQR scaling."""
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x))
    q25, q75 = np.percentile(x, [25, 75])
    iqr = float(q75 - q25)
    if iqr < 1e-12:
        iqr = float(np.std(x) + 1e-12)
    return med, iqr


def robust_apply(x: np.ndarray, med: float, scale: float) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - med) / (scale + 1e-12)


def select_global_topk(shap_train: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Official AE-DSP Global Top-k selection by mean(abs(SHAP))."""
    importance = np.mean(np.abs(shap_train), axis=0)
    k = min(k, shap_train.shape[1])
    idx = np.argsort(importance)[::-1][:k]
    return idx, importance


def train_ae_model(
    X_train_normal: np.ndarray,
    input_dim: int,
    seed: int = 42,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    latent_dim: int = 8,
    hidden_dim: int = 64,
    dropout: float = 0.05,
    device: str = "cpu",
    verbose: bool = False,
) -> Tuple[AutoEncoder, Dict[str, Any]]:
    """
    Train AutoEncoder on normal traffic contribution space following AE-DSP.
    """
    set_seed(seed)
    model = AutoEncoder(input_dim, latent_dim, hidden_dim, dropout).to(device)
    x = torch.tensor(X_train_normal, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    mse = nn.MSELoss()
    steps_per_epoch = max(1, int(np.ceil(len(X_train_normal) / batch_size)))
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            idx = torch.randint(0, x.shape[0], (min(batch_size, x.shape[0]),), device=device)
            xb = x[idx]
            recon, _ = model(xb)
            loss = mse(recon, xb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        loss_mean = float(np.mean(losses))
        history.append({"epoch": epoch, "loss": loss_mean})
        if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
            print(f"      AE epoch {epoch:03d}/{epochs} | recon_loss={loss_mean:.6f}")

    return model, {"train_sec": time.perf_counter() - start_time, "last_loss": history[-1]["loss"]}


def fit_dsp_prototypes(
    delta_attack: np.ndarray,
    K: int = 12,
    seed: int = 42,
) -> Tuple[KMeans, float]:
    """
    Fit K-means prototypes on attack DeltaSHAP representations and compute kernel sigma.
    """
    effective_K = int(min(K, max(1, len(delta_attack))))
    km = KMeans(n_clusters=effective_K, random_state=seed, n_init=10)
    km.fit(delta_attack)

    # Nearest prototype distances on attack training data
    centers = km.cluster_centers_
    d2 = np.sum((delta_attack[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    dist = np.sqrt(np.min(d2, axis=1))

    sigma = float(np.median(dist))
    if sigma < 1e-8:
        sigma = float(np.mean(dist) + 1e-8)
    return km, sigma


def build_aedsp_nids(
    X_train_pre: np.ndarray,
    y_train_binary: np.ndarray,
    y_train_labels: List[str],
    X_val_pre: Optional[np.ndarray] = None,
    y_val_binary: Optional[np.ndarray] = None,
    preprocessor: Optional[Any] = None,
    base_tree_model: Optional[Any] = None,
    shap_train: Optional[np.ndarray] = None,
    shap_val: Optional[np.ndarray] = None,
    topk: int = 50,
    proto_k: int = 12,
    latent_dim: int = 8,
    hidden_dim: int = 64,
    epochs: int = 50,
    classifier_epochs: int = 30,
    alpha: float = 0.8,
    gamma: float = 1.8,
    target_fpr: float = 0.045,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
) -> AEDSP_NIDS:
    """
    Complete end-to-end builder to train/initialize the AE-DSP NIDS framework:
      1. Compute SHAP space and select Global Top-k.
      2. Train AE on normal traffic.
      3. Fit DSP prototypes on known attack traffic.
      4. Calibrate robust normal parameters and validation threshold.
      5. Initialize PretrainedAEDSP core.
      6. Train Classifier Head on Pretrained AE-DSP representations.
      7. Calibrate Anomaly Detector.
      8. Assemble unified AEDSP_NIDS with Risk Engine.
    """
    set_seed(seed)
    if verbose:
        print("[1/6] Extracting SHAP contribution space...")

    # If SHAP matrices not provided, compute with LightGBM + TreeExplainer
    if shap_train is None:
        if base_tree_model is None:
            if verbose:
                print("      Training LightGBM base tree model on known traffic...")
            base_tree_model = LGBMClassifier(
                objective="binary",
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            )
            base_tree_model.fit(X_train_pre, y_train_binary)

        if verbose:
            print("      Computing TreeSHAP values...")
        explainer = shap.TreeExplainer(base_tree_model)
        sv_tr = explainer.shap_values(X_train_pre)
        if isinstance(sv_tr, list):
            shap_train = sv_tr[1] if len(sv_tr) == 2 else sv_tr[0]
        else:
            shap_train = sv_tr[:, :, 1] if sv_tr.ndim == 3 else sv_tr

        if X_val_pre is not None:
            sv_val = explainer.shap_values(X_val_pre)
            shap_val = sv_val[1] if isinstance(sv_val, list) and len(sv_val) == 2 else sv_val
            if not isinstance(sv_val, list) and shap_val.ndim == 3:
                shap_val = shap_val[:, :, 1]
        else:
            shap_val = shap_train

    shap_train = np.asarray(shap_train, dtype=np.float32)
    shap_val = np.asarray(shap_val, dtype=np.float32) if shap_val is not None else shap_train

    # [2/6] Select Global Top-k
    if verbose:
        print(f"[2/6] Selecting Global Top-{topk} SHAP features...")
    top_idx, _ = select_global_topk(shap_train, topk)
    X_tr_top = shap_train[:, top_idx]
    X_val_top = shap_val[:, top_idx]

    # [3/6] Train AutoEncoder on Normal traffic
    if verbose:
        print("[3/6] Training AutoEncoder on normal traffic contribution manifold...")
    normal_mask_tr = (y_train_binary == 0)
    ae_scaler = StandardScaler()
    ae_scaler.fit(X_tr_top[normal_mask_tr])
    X_tr_ae = ae_scaler.transform(X_tr_top).astype(np.float32)
    X_val_ae = ae_scaler.transform(X_val_top).astype(np.float32)

    ae_model, _ = train_ae_model(
        X_train_normal=X_tr_ae[normal_mask_tr],
        input_dim=len(top_idx),
        seed=seed,
        epochs=epochs,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        device=device,
        verbose=verbose,
    )

    # Compute AE reconstruction residuals and robust z-score parameters
    ae_model.eval()
    with torch.no_grad():
        x_tr_ae_t = torch.tensor(X_tr_ae, dtype=torch.float32, device=device)
        recon_tr, _ = ae_model(x_tr_ae_t)
        res2_tr = ((recon_tr - x_tr_ae_t) ** 2).cpu().numpy()
        ae_tr_scores = np.mean(res2_tr, axis=1)

    ae_med, ae_scale = robust_fit(ae_tr_scores[normal_mask_tr])
    ae_tr_z = robust_apply(ae_tr_scores, ae_med, ae_scale)

    # [4/6] Build DSP Prototypes on Attack traffic
    if verbose:
        print(f"[4/6] Constructing DSP attack prototypes (K={proto_k})...")
    mu_n = X_tr_top[normal_mask_tr].mean(axis=0)
    std_n = X_tr_top[normal_mask_tr].std(axis=0)

    delta_tr = ((X_tr_top - mu_n) / (std_n + 1e-8)).astype(np.float32)
    attack_mask_tr = (y_train_binary == 1)

    kmeans, sigma = fit_dsp_prototypes(delta_tr[attack_mask_tr], K=proto_k, seed=seed)

    # Compute DSP similarity and robust parameters
    centers = kmeans.cluster_centers_
    d2 = np.sum((delta_tr[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    dist = np.sqrt(np.min(d2, axis=1))
    dp_train_raw = np.exp(-dist / (sigma + 1e-8))

    dp_med, dp_scale = robust_fit(dp_train_raw[normal_mask_tr])
    dsp_tr_z = robust_apply(dp_train_raw, dp_med, dp_scale)

    # Calibrate normal threshold on validation normal scores
    if X_val_pre is not None and y_val_binary is not None:
        normal_mask_val = (y_val_binary == 0)
        with torch.no_grad():
            x_val_ae_t = torch.tensor(X_val_ae, dtype=torch.float32, device=device)
            recon_val, _ = ae_model(x_val_ae_t)
            res2_val = ((recon_val - x_val_ae_t) ** 2).cpu().numpy()
            ae_val_scores = np.mean(res2_val, axis=1)

        ae_val_z = robust_apply(ae_val_scores, ae_med, ae_scale)
        delta_val = ((X_val_top - mu_n) / (std_n + 1e-8)).astype(np.float32)
        d2_val = np.sum((delta_val[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        dist_val = np.sqrt(np.min(d2_val, axis=1))
        dp_val_raw = np.exp(-dist_val / (sigma + 1e-8))
        dsp_val_z = robust_apply(dp_val_raw, dp_med, dp_scale)
        fusion_val = alpha * ae_val_z + gamma * dsp_val_z
        norm_val_scores = fusion_val[normal_mask_val]
        threshold = float(np.percentile(norm_val_scores, 100.0 * (1.0 - target_fpr)))
    else:
        fusion_tr = alpha * ae_tr_z + gamma * dsp_tr_z
        threshold = float(np.percentile(fusion_tr[normal_mask_tr], 100.0 * (1.0 - target_fpr)))

    if verbose:
        print(f"      Calibrated AE-DSP Threshold: {threshold:.4f} (Target FPR={target_fpr:.3f})")

    # Instantiate Pretrained AE-DSP Foundation
    ae_dsp = PretrainedAEDSP(
        autoencoder=ae_model,
        ae_scaler=ae_scaler,
        top_idx=top_idx,
        mu_normal=mu_n,
        std_normal=std_n,
        dsp_kmeans=kmeans,
        dsp_sigma=sigma,
        ae_med=ae_med,
        ae_scale=ae_scale,
        dsp_med=dp_med,
        dsp_scale=dp_scale,
        threshold=threshold,
        alpha=alpha,
        gamma=gamma,
        preprocessor=preprocessor,
        base_tree_model=base_tree_model,
        device=device,
    )

    # [5/6] Extract representations and train Classifier Head for Known Classes
    if verbose:
        print("[5/6] Training Classifier head on Pretrained AE-DSP representations...")
    rep_train = ae_dsp.extract_representation(shap_train, is_already_shap=True)
    known_classes = sorted(list(set(y_train_labels)))

    classifier = AEDSPClassifier(
        representation_dim=ae_dsp.representation_dim,
        class_names=known_classes,
        hidden_dim=hidden_dim,
        device=device,
    )
    classifier.fit(
        representation=rep_train,
        y_labels=y_train_labels,
        epochs=classifier_epochs,
        verbose=verbose,
    )

    # [6/6] Calibrate Anomaly Detector
    if verbose:
        print("[6/6] Initializing Anomaly Detector and Risk Engine...")
    attack_dists = rep_train.nearest_proto_dist[attack_mask_tr]
    anomaly_detector = AEDSPAnomalyDetector(
        threshold=threshold,
        alpha=alpha,
        gamma=gamma,
    )
    anomaly_detector.calibrate_novelty_threshold(attack_dists, percentile=95.0)

    risk_engine = RiskEngine()

    pipeline = AEDSP_NIDS(
        ae_dsp=ae_dsp,
        classifier=classifier,
        anomaly_detector=anomaly_detector,
        risk_engine=risk_engine,
    )

    if verbose:
        print("\n" + pipeline.summary())

    return pipeline
