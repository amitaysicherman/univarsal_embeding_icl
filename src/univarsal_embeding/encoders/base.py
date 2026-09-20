"""Abstract base encoder interface for multimodal representations with robust batching."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Sequence
import numpy as np


class BaseEncoder(ABC):
    """Abstract base class for all foundation model encoders."""

    def __init__(self, encoder_id: str, domain: str, device: str = "cpu") -> None:
        self.encoder_id = encoder_id
        self.domain = domain
        self.device = device
        self.embedding_dim: int = 768

    @abstractmethod
    def encode(self, inputs: Sequence[Any], batch_size: int = 32) -> np.ndarray:
        """Extract continuous representation embeddings as an (N, d) float32 numpy array."""
        pass


class DummyEncoder(BaseEncoder):
    """Deterministic pseudo-encoder for offline unit testing, smoke tests, and local demos."""

    def __init__(self, encoder_id: str, domain: str, embedding_dim: int = 256, device: str = "cpu") -> None:
        super().__init__(encoder_id=encoder_id, domain=domain, device=device)
        self.embedding_dim = embedding_dim

    def encode(self, inputs: Sequence[Any], batch_size: int = 32) -> np.ndarray:
        embs = []
        for item in inputs:
            s_hash = abs(hash(str(item))) % 1000000
            val_rng = np.random.default_rng(s_hash)
            v = val_rng.normal(loc=0.0, scale=1.0, size=(self.embedding_dim,)).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            embs.append(v)
        return np.vstack(embs)
