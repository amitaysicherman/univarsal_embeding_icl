"""Standard baseline classifiers for foundation representation evaluation."""

from __future__ import annotations

import logging
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class LinearBaseline:
    """Logistic Regression baseline with feature standardization and L2 regularization."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42) -> None:
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state, solver="lbfgs")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> LinearBaseline:
        X_scaled = self.scaler.fit_transform(X_train)
        self.clf.fit(X_scaled, y_train)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_test)
        return self.clf.predict_proba(X_scaled)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_test)
        return self.clf.predict(X_scaled)


class KNNBaseline:
    """k-Nearest Neighbors baseline with feature standardization."""

    def __init__(self, n_neighbors: int = 5, metric: str = "cosine") -> None:
        self.scaler = StandardScaler()
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights="distance")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> KNNBaseline:
        X_scaled = self.scaler.fit_transform(X_train)
        # Cap neighbors at number of samples - 1
        k = min(self.clf.n_neighbors, max(1, len(X_train) - 1))
        self.clf.n_neighbors = k
        self.clf.fit(X_scaled, y_train)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_test)
        return self.clf.predict_proba(X_scaled)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_test)
        return self.clf.predict(X_scaled)


class XGBoostBaseline:
    """XGBoost gradient-boosted tree baseline (CPU by default)."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        import xgboost as xgb

        self._xgb = xgb
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.clf = None
        self.classes_: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> XGBoostBaseline:
        self.classes_ = np.unique(y_train)
        n_classes = len(self.classes_)
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([label_map[y] for y in y_train])

        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        kwargs = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            objective=objective,
            eval_metric="logloss" if n_classes == 2 else "mlogloss",
        )
        if n_classes > 2:
            kwargs["num_class"] = n_classes
        self.clf = self._xgb.XGBClassifier(**kwargs)
        self.clf.fit(X_train, y_idx)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X_test)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        pred_idx = self.clf.predict(X_test)
        return self.classes_[pred_idx.astype(int)]


class _TorchMLP(nn.Module):
    """MLP with exactly one hidden layer (as specified for this benchmark's baseline suite)."""

    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPBaseline:
    """2-Layer PyTorch MLP with Early Stopping on validation fold."""

    def __init__(
        self,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 150,
        patience: int = 15,
        device: str = "cpu",
    ) -> None:
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.device = device
        self.scaler = StandardScaler()
        self.model: _TorchMLP | None = None
        self.classes_: np.ndarray | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> MLPBaseline:
        self.classes_ = np.unique(y_train)
        n_classes = len(self.classes_)
        in_dim = X_train.shape[1]

        # Map labels to 0..C-1
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_train_idx = np.array([label_map[y] for y in y_train])

        X_train_scaled = self.scaler.fit_transform(X_train)
        x_tr = torch.tensor(X_train_scaled, dtype=torch.float32, device=self.device)
        y_tr = torch.tensor(y_train_idx, dtype=torch.long, device=self.device)

        if X_val is not None and y_val is not None and len(X_val) > 0:
            X_val_scaled = self.scaler.transform(X_val)
            x_vl = torch.tensor(X_val_scaled, dtype=torch.float32, device=self.device)
            y_val_idx = np.array([label_map.get(y, 0) for y in y_val])
            y_vl = torch.tensor(y_val_idx, dtype=torch.long, device=self.device)
        else:
            x_vl, y_vl = None, None

        self.model = _TorchMLP(in_dim, self.hidden_dim, n_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        self.model.train()
        batch_size = min(128, len(X_train))

        for epoch in range(self.epochs):
            perm = torch.randperm(len(x_tr))
            for b in range(0, len(x_tr), batch_size):
                idx = perm[b : b + batch_size]
                optimizer.zero_grad()
                logits = self.model(x_tr[idx])
                loss = criterion(logits, y_tr[idx])
                loss.backward()
                optimizer.step()

            if x_vl is not None:
                self.model.eval()
                with torch.inference_mode():
                    val_logits = self.model(x_vl)
                    val_loss = criterion(val_logits, y_vl).item()
                self.model.train()

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MLPBaseline not fitted")
        X_scaled = self.scaler.transform(X_test)
        x_te = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(x_te)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_test)
        pred_idx = np.argmax(probs, axis=1)
        return self.classes_[pred_idx]


class TabICLZeroShotBaseline:
    """Vanilla Pretrained TabICL In-Context Classifier (Zero-Shot, No Fine-Tuning)."""

    def __init__(
        self,
        checkpoint_repo: str = "jingang/TabICL",
        checkpoint_file: str = "tabicl-classifier-v2-20260212.ckpt",
        context_size: int = 512,
        device: str = "cpu",
    ) -> None:
        self.context_size = context_size
        self.device = device
        self._model = None
        self._checkpoint_repo = checkpoint_repo
        self._checkpoint_file = checkpoint_file

    def _load_model(self):
        if self._model is None:
            from huggingface_hub import hf_hub_download
            from tabicl import TabICL
            # Apply the same tabicl indexless-cuda-device mem_get_info
            # workaround used by UniversalTabICL (see that module for details).
            from .tabicl_universal import _patch_mem_get_info_for_indexless_cuda_device
            _patch_mem_get_info_for_indexless_cuda_device()
            ckpt_path = hf_hub_download(repo_id=self._checkpoint_repo, filename=self._checkpoint_file)
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            config = dict(checkpoint["config"])
            # NOTE: recompute (gradient checkpointing) is intentionally left
            # off here - this baseline is inference-only (runs under
            # torch.inference_mode() below) and TabICL's checkpoint
            # implementation is incompatible with no-grad/inference-mode
            # contexts (raises a device-index error). bf16 autocast alone
            # (applied at the call site) keeps memory low enough for the
            # full unclipped training set without needing checkpointing.
            config["recompute"] = False
            self._model = TabICL(**config).to(self.device)
            self._model.load_state_dict(checkpoint["state_dict"])
            self._model.eval()
        return self._model

    def predict_proba(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        subsample_context: bool = False,
    ) -> np.ndarray:
        """In-context classification: X_train/y_train is context, X_test is query.

        By default uses the FULL training set as context (matching official
        TabICL usage - it never subsamples training rows, only batches the
        ensemble/query dimension for memory). Pass subsample_context=True to
        opt into the old class-stratified subsampling behavior if a task's
        training set is too large to fit in memory as full context.
        """
        model = self._load_model()
        classes = np.unique(y_train)
        n_classes = len(classes)
        label_map = {c: i for i, c in enumerate(classes)}
        y_ctx_idx = np.array([label_map[y] for y in y_train])

        # Optional class-stratified subsampling, off by default (see docstring).
        if subsample_context and len(X_train) > self.context_size:
            indices = []
            rng = np.random.default_rng(42)
            per_class = max(1, self.context_size // n_classes)
            for c in range(n_classes):
                c_idx = np.flatnonzero(y_ctx_idx == c)
                if len(c_idx) > 0:
                    chosen = rng.choice(c_idx, size=min(len(c_idx), per_class), replace=False)
                    indices.extend(chosen)
            ctx_idx = np.array(indices)
        else:
            ctx_idx = np.arange(len(X_train))

        X_ctx = X_train[ctx_idx]
        y_ctx = y_ctx_idx[ctx_idx]

        # Standardize features using context support set
        mean = X_ctx.mean(axis=0, keepdims=True)
        std = np.clip(X_ctx.std(axis=0, keepdims=True), 1e-6, None)
        X_ctx_std = (X_ctx - mean) / std
        X_test_std = (X_test - mean) / std

        # TabICL forward in batches of query points
        n_test = len(X_test)
        query_chunk = 64
        all_probs = []

        with torch.inference_mode():
            t_ctx_x = torch.tensor(X_ctx_std, dtype=torch.float32, device=self.device)
            t_ctx_y = torch.tensor(y_ctx, dtype=torch.long, device=self.device).unsqueeze(0)

            for b in range(0, n_test, query_chunk):
                q_chunk_x = torch.tensor(X_test_std[b : b + query_chunk], dtype=torch.float32, device=self.device)
                # Concatenate context + query along row dimension: (1, N_ctx + N_q, D)
                combined_x = torch.cat([t_ctx_x, q_chunk_x], dim=0).unsqueeze(0)
                try:
                    if combined_x.device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits = model(combined_x, t_ctx_y, return_logits=True)[0, :, :n_classes]
                    else:
                        logits = model(combined_x, t_ctx_y, return_logits=True)[0, :, :n_classes]
                    # Softmax in float32, not bf16: normalizing in low precision
                    # leaves row sums off by up to ~0.2% (0.998-1.002), which
                    # sklearn's multi-class AUROC validator rejects outright
                    # (binary AUROC never hit this since it only reads one
                    # column, not the full normalized row).
                    probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                except Exception as e:
                    logger.warning(f"TabICL inference chunk failed: {e}. Falling back to uniform.")
                    probs = np.ones((len(q_chunk_x), n_classes)) / n_classes
                all_probs.append(probs)

        return np.vstack(all_probs)
