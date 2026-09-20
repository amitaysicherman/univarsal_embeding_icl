"""Molecular foundation encoders (ChemBERTa, MoLFormer-XL) with batching."""

from __future__ import annotations

import logging
from typing import Sequence
import numpy as np
import torch
from .base import BaseEncoder

logger = logging.getLogger(__name__)


class HFSmilesEncoder(BaseEncoder):
    """General HuggingFace SMILES sequence encoder with batched mean pooling."""

    def __init__(
        self,
        checkpoint: str,
        encoder_id: str,
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        super().__init__(encoder_id=encoder_id, domain="molecules", device=device)
        from transformers import AutoModel, AutoTokenizer

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<pad>"
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device).eval()
        self.embedding_dim = getattr(self.model.config, "hidden_size", 768)

    def encode(self, inputs: Sequence[str], batch_size: int = 64) -> np.ndarray:
        all_embeddings = []
        n_samples = len(inputs)

        with torch.inference_mode():
            for i in range(0, n_samples, batch_size):
                batch_smiles = list(inputs[i : i + batch_size])
                encoded = self.tokenizer(
                    batch_smiles,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                encoded.pop("token_type_ids", None)
                outputs = self.model(**encoded)
                last_hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                all_embeddings.append(pooled.detach().cpu().to(torch.float32).numpy())

        return np.vstack(all_embeddings)


class ChemBERTaEncoder(HFSmilesEncoder):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(
            checkpoint="DeepChem/ChemBERTa-77M-MTR",
            encoder_id="chemberta_77m_mtr",
            device=device,
        )


class SmilesBERTEncoder(HFSmilesEncoder):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(
            checkpoint="unikei/bert-base-smiles",
            encoder_id="smiles_bert",
            device=device,
        )


class MoLFormerXLEncoder(HFSmilesEncoder):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(
            checkpoint="ibm/MoLFormer-XL-both-10pct",
            encoder_id="molformer_xl",
            device=device,
        )
