"""
classifier.py

Component 1: Classifier built on Pretrained AE-DSP Representations.
Classifies network traffic into known classes/attack types and outputs
the predicted class, confidence, and class probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as e:
    raise ImportError("PyTorch is required for AEDSPClassifier.") from e

from .ae_dsp_model import AEDSPRepresentation


@dataclass
class ClassifierResult:
    """Output for a batch or single prediction of the Classifier."""
    predicted_class: List[str]  # e.g., ["Normal", "DoS", ...]
    predicted_class_id: np.ndarray  # class indices [batch_size]
    confidence: np.ndarray  # max probability [batch_size]
    probabilities: np.ndarray  # full probability distribution [batch_size, num_classes]
    entropy: np.ndarray  # predictive entropy [batch_size]
    class_names: List[str]

    def __len__(self) -> int:
        return len(self.predicted_class)

    def get_sample(self, idx: int) -> Dict[str, Any]:
        """Get result for a single sample."""
        probs_dict = {
            self.class_names[c]: float(self.probabilities[idx, c])
            for c in range(len(self.class_names))
        }
        return {
            "predicted_class": self.predicted_class[idx],
            "predicted_class_id": int(self.predicted_class_id[idx]),
            "confidence": float(self.confidence[idx]),
            "probabilities": probs_dict,
            "entropy": float(self.entropy[idx]),
        }


class ClassificationHead(nn.Module):
    """
    Lightweight neural classification head operating directly on AE-DSP representation.
    Adheres strictly to the constraint: no extraneous models (no Transformer, VAE, XGBoost, etc.).
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AEDSPClassifier:
    """
    Classifier built directly on top of the Pretrained AE-DSP representation.
    """

    def __init__(
        self,
        representation_dim: int,
        class_names: List[str],
        hidden_dim: int = 64,
        dropout: float = 0.1,
        device: str = "cpu",
    ) -> None:
        self.representation_dim = representation_dim
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.device = device

        self.head = ClassificationHead(
            input_dim=representation_dim,
            num_classes=self.num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(self.device)

        self._is_trained = False

    def fit(
        self,
        representation: Union[AEDSPRepresentation, np.ndarray],
        y_labels: Union[List[str], np.ndarray],
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        use_class_weight: bool = True,
        class_weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Train the classification head on representations produced by AE-DSP.
        Applies class weighting / cost-sensitive learning to handle severe class imbalance.
        """
        if isinstance(representation, AEDSPRepresentation):
            X = representation.fused_features
        else:
            X = np.asarray(representation, dtype=np.float32)

        # Convert label strings to class indices
        if isinstance(y_labels[0], (str, np.str_)):
            label_to_idx = {name: i for i, name in enumerate(self.class_names)}
            y = np.array([label_to_idx.get(lab, 0) for lab in y_labels], dtype=np.int64)
        else:
            y = np.asarray(y_labels, dtype=np.int64)

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)

        # Handle class imbalance via cost-sensitive class weights
        if class_weights is not None:
            if isinstance(class_weights, torch.Tensor):
                weights_t = class_weights.to(self.device).float()
            else:
                weights_t = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            criterion = nn.CrossEntropyLoss(weight=weights_t)
        elif use_class_weight:
            classes, counts = np.unique(y, return_counts=True)
            total = len(y)
            weights = np.ones(self.num_classes, dtype=np.float32)
            for c, cnt in zip(classes, counts):
                weights[c] = total / (len(classes) * max(1, cnt))
            weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device)
            criterion = nn.CrossEntropyLoss(weight=weights_t)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(self.head.parameters(), lr=lr, weight_decay=weight_decay)

        self.head.train()
        n_samples = X.shape[0]
        history = []

        for epoch in range(1, epochs + 1):
            perm = torch.randperm(n_samples, device=self.device)
            total_loss = 0.0
            total_batches = 0

            for i in range(0, n_samples, batch_size):
                idx = perm[i : i + batch_size]
                xb, yb = X_t[idx], y_t[idx]

                optimizer.zero_grad()
                logits = self.head(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                total_batches += 1

            avg_loss = total_loss / max(1, total_batches)
            history.append({"epoch": epoch, "loss": avg_loss})
            if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
                print(f"  Classifier Head Epoch {epoch:03d}/{epochs} | Loss: {avg_loss:.5f}")

        self._is_trained = True
        return {"history": history, "final_loss": history[-1]["loss"]}

    @torch.no_grad()
    def predict(
        self,
        representation: Union[AEDSPRepresentation, np.ndarray],
    ) -> ClassifierResult:
        """
        Compute predicted class, confidence (probability), full probabilities, and entropy
        using the pretrained AE-DSP representation.
        """
        if isinstance(representation, AEDSPRepresentation):
            X = representation.fused_features
        else:
            X = np.asarray(representation, dtype=np.float32)

        self.head.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        logits = self.head(X_t)
        probs_t = torch.softmax(logits, dim=-1)

        probs = probs_t.cpu().numpy()
        pred_ids = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)

        # Compute predictive entropy: -sum(p * log(p + eps))
        eps = 1e-12
        entropy = -np.sum(probs * np.log(probs + eps), axis=1)

        predicted_classes = [self.class_names[idx] for idx in pred_ids]

        return ClassifierResult(
            predicted_class=predicted_classes,
            predicted_class_id=pred_ids,
            confidence=confidences,
            probabilities=probs,
            entropy=entropy,
            class_names=self.class_names,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "head": self.head.state_dict(),
            "class_names": self.class_names,
            "representation_dim": self.representation_dim,
            "is_trained": self._is_trained,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.head.load_state_dict(state["head"])
        self.class_names = state["class_names"]
        self.num_classes = len(self.class_names)
        self.representation_dim = state["representation_dim"]
        self._is_trained = state.get("is_trained", True)
