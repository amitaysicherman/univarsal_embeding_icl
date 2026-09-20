"""Protein foundation encoders (ESM-2, ProtBERT, ESM-Cambrian, Ankh)."""

from __future__ import annotations

import logging
from typing import Sequence
import numpy as np
import torch
from .base import BaseEncoder

logger = logging.getLogger(__name__)


class HFProteinEncoder(BaseEncoder):
    """HuggingFace protein language model encoder with batched sequence length pooling."""

    def __init__(
        self,
        checkpoint: str,
        encoder_id: str,
        device: str = "cpu",
        spaced_sequence: bool = False,
        max_length: int = 1024,
    ) -> None:
        super().__init__(encoder_id=encoder_id, domain="proteins", device=device)
        from transformers import AutoModel, AutoTokenizer

        self.spaced_sequence = spaced_sequence
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device).eval()
        self.embedding_dim = getattr(self.model.config, "hidden_size", 1024)
        # Encoder-decoder checkpoints (e.g. T5-based Ankh) load as the full
        # seq2seq model under AutoModel; run only the encoder stack so no
        # decoder_input_ids are required to obtain per-token hidden states.
        self._forward_module = (
            self.model.encoder if hasattr(self.model, "encoder") and hasattr(self.model, "decoder") else self.model
        )

    def encode(self, inputs: Sequence[str], batch_size: int = 16) -> np.ndarray:
        all_embeddings = []
        n_samples = len(inputs)

        with torch.inference_mode():
            for i in range(0, n_samples, batch_size):
                batch_raw = ["".join(s.strip().split()).upper() for s in inputs[i : i + batch_size]]
                if self.spaced_sequence:
                    batch_seqs = [" ".join(list(s)) for s in batch_raw]
                else:
                    batch_seqs = batch_raw

                encoded = self.tokenizer(
                    batch_seqs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self._forward_module(**encoded)
                last_hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                all_embeddings.append(pooled.detach().cpu().to(torch.float32).numpy())

        return np.vstack(all_embeddings)


class ESM2_650MEncoder(HFProteinEncoder):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(
            checkpoint="facebook/esm2_t33_650M_UR50D",
            encoder_id="esm2_650m",
            device=device,
            spaced_sequence=False,
        )


class ProtBERTEncoder(HFProteinEncoder):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(
            checkpoint="Rostlab/prot_bert",
            encoder_id="prot_bert",
            device=device,
            spaced_sequence=True,
        )
