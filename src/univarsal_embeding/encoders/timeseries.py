"""Time Series Foundation Encoders (MiniRocket, Rocket, Chronos-T5, ResNet1D)."""

from __future__ import annotations

import logging
from typing import Any, List
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

try:
    from aeon.transformations.collection.convolution_based import MiniRocket, Rocket
except ImportError:
    MiniRocket = None
    Rocket = None

from .base import BaseEncoder

logger = logging.getLogger(__name__)


class MiniRocketEncoder(BaseEncoder):
    """MiniRocket transform encoder for time series representations."""

    def __init__(self, n_kernels: int = 512, device: str = "cpu"):
        super().__init__(encoder_id="minirocket", domain="timeseries", device=device); self.embedding_dim = 999
        self.n_kernels = n_kernels

    def encode(self, inputs: List[Any], batch_size: int = 256) -> np.ndarray:
        # Standardize inputs to (N, 1, T) numpy array with zero-padding to max length
        max_len = max(len(s) for s in inputs)
        padded = np.zeros((len(inputs), 1, max_len), dtype=np.float64)
        for i, s in enumerate(inputs):
            padded[i, 0, :len(s)] = np.nan_to_num(np.array(s, dtype=np.float64), 0.0)

        mr = MiniRocket(n_kernels=self.n_kernels, random_state=42)
        embs = mr.fit_transform(padded)
        embs_np = np.nan_to_num(np.asarray(embs, dtype=np.float32), 0.0)
        self.embedding_dim = embs_np.shape[1]
        return embs_np


class RocketEncoder(BaseEncoder):
    """Full ROCKET transform encoder for time series representations."""

    def __init__(self, n_kernels: int = 512, device: str = "cpu"):
        super().__init__(encoder_id="rocket", domain="timeseries", device=device); self.embedding_dim = 1024
        self.n_kernels = n_kernels

    def encode(self, inputs: List[Any], batch_size: int = 256) -> np.ndarray:
        max_len = max(len(s) for s in inputs)
        padded = np.zeros((len(inputs), 1, max_len), dtype=np.float64)
        for i, s in enumerate(inputs):
            padded[i, 0, :len(s)] = np.nan_to_num(np.array(s, dtype=np.float64), 0.0)

        r = Rocket(n_kernels=self.n_kernels, random_state=42)
        embs = r.fit_transform(padded)
        embs_np = np.nan_to_num(np.asarray(embs, dtype=np.float32), 0.0)
        self.embedding_dim = embs_np.shape[1]
        return embs_np


class ChronosT5Encoder(BaseEncoder):
    """Amazon Chronos-T5 time series representation encoder."""

    def __init__(self, model_id: str = "amazon/chronos-t5-small", device: str = "cpu"):
        super().__init__(encoder_id="chronos_t5_small", domain="timeseries", device=device); self.embedding_dim = 512
        self.model_id = model_id
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
        self.model.eval()

    def encode(self, inputs: List[Any], batch_size: int = 64) -> np.ndarray:
        all_embs = []
        dev = torch.device(self.device)

        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                chunk = inputs[i : i + batch_size]
                max_len = min(512, max(len(s) for s in chunk))
                batch_tensor = torch.zeros((len(chunk), max_len), dtype=torch.float32, device=dev)
                for j, s in enumerate(chunk):
                    vals = torch.tensor(s[:max_len], dtype=torch.float32, device=dev)
                    batch_tensor[j, :len(vals)] = vals

                # Pass through Chronos encoder
                # Chronos quantizes or normalizes series; as continuous embedding, pass through shared encoder
                # Chronos uses model.model.encoder (T5Stack)
                encoder = self.model.get_encoder()
                # T5 encoder expects input_ids (int) or inputs_embeds (float). Project 1D to d_model:
                d_model = self.model.config.d_model
                # Map continuous values to token space via linear projection or embedder
                x_scaled = (batch_tensor - batch_tensor.mean(dim=-1, keepdim=True)) / (batch_tensor.std(dim=-1, keepdim=True) + 1e-6)
                # Map into Chronos vocabulary range (0..4095) for Chronos-T5
                token_ids = torch.clamp(((x_scaled + 3.0) / 6.0 * 4000).long(), 0, 4095)
                out = encoder(input_ids=token_ids)
                hidden = out.last_hidden_state  # (B, T, D)
                pooled = hidden.mean(dim=1)     # (B, D)
                all_embs.append(pooled.cpu().numpy())

        return np.vstack(all_embs).astype(np.float32)


class ResNet1DEncoder(BaseEncoder):
    """Deep 1D Convolutional ResNet for time series representations."""

    def __init__(self, embedding_dim: int = 256, device: str = "cpu"):
        super().__init__(encoder_id="resnet1d", domain="timeseries", device=device); self.embedding_dim = embedding_dim
        torch.manual_seed(42)
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, embedding_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        ).to(device)
        self.net.eval()

    def encode(self, inputs: List[Any], batch_size: int = 64) -> np.ndarray:
        all_embs = []
        dev = torch.device(self.device)

        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                chunk = inputs[i : i + batch_size]
                max_len = max(len(s) for s in chunk)
                batch_tensor = torch.zeros((len(chunk), 1, max_len), dtype=torch.float32, device=dev)
                for j, s in enumerate(chunk):
                    vals = torch.tensor(s, dtype=torch.float32, device=dev)
                    batch_tensor[j, 0, :len(vals)] = vals

                out = self.net(batch_tensor).squeeze(-1)
                all_embs.append(out.cpu().numpy())

        return np.vstack(all_embs).astype(np.float32)
