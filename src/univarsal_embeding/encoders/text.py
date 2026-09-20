"""Text foundation encoders (ModernBERT, BGE) with batching."""

from __future__ import annotations

import logging
from typing import Sequence
import numpy as np
import torch
from .base import BaseEncoder

logger = logging.getLogger(__name__)


class HFTextEncoder(BaseEncoder):
    """HuggingFace text encoder with batched mean pooling over attention mask."""

    def __init__(
        self,
        checkpoint: str,
        encoder_id: str,
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        super().__init__(encoder_id=encoder_id, domain="text", device=device)
        from transformers import AutoModel, AutoTokenizer

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device).eval()
        self.embedding_dim = getattr(self.model.config, "hidden_size", 768)

    def encode(self, inputs: Sequence[str], batch_size: int = 64) -> np.ndarray:
        all_embeddings = []
        n_samples = len(inputs)

        with torch.inference_mode():
            for i in range(0, n_samples, batch_size):
                batch_texts = list(inputs[i : i + batch_size])
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**encoded)
                last_hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                all_embeddings.append(pooled.detach().cpu().to(torch.float32).numpy())

        return np.vstack(all_embeddings)


class ModernBERTEncoder(HFTextEncoder):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(
            checkpoint="nomic-ai/modernbert-embed-base",
            encoder_id="modernbert_embed",
            device=device,
        )


class BGELargeEncoder(HFTextEncoder):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(
            checkpoint="BAAI/bge-large-en-v1.5",
            encoder_id="bge_large_en_v1_5",
            device=device,
        )
