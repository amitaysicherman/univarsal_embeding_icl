"""Vision foundation encoders (DINOv2, SigLIP-2) with batching."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence
import numpy as np
from PIL import Image
import torch
from .base import BaseEncoder

logger = logging.getLogger(__name__)


class HFVisionEncoder(BaseEncoder):
    """General HuggingFace vision encoder with batched AutoImageProcessor and AutoModel."""

    def __init__(
        self,
        checkpoint: str,
        encoder_id: str,
        device: str = "cpu",
    ) -> None:
        super().__init__(encoder_id=encoder_id, domain="vision", device=device)
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device).eval()
        cfg = self.model.config
        if hasattr(cfg, "vision_config") and hasattr(cfg.vision_config, "hidden_size"):
            self.embedding_dim = cfg.vision_config.hidden_size
        else:
            self.embedding_dim = getattr(cfg, "hidden_size", 768)

    def encode(self, inputs: Sequence[str], batch_size: int = 32) -> np.ndarray:
        all_embeddings = []
        n_samples = len(inputs)

        with torch.inference_mode():
            for i in range(0, n_samples, batch_size):
                batch_paths = inputs[i : i + batch_size]
                images = [Image.open(Path(p)).convert("RGB") for p in batch_paths]
                inputs_tensor = self.processor(images=images, return_tensors="pt").to(self.device)
                if hasattr(self.model, "get_image_features"):
                    feat = self.model.get_image_features(**inputs_tensor)
                    pooled = feat if isinstance(feat, torch.Tensor) else feat[0]
                else:
                    outputs = self.model(**inputs_tensor)
                    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                        pooled = outputs.pooler_output
                    elif hasattr(outputs, "last_hidden_state"):
                        pooled = outputs.last_hidden_state[:, 0]
                    else:
                        pooled = outputs[0][:, 0]
                if isinstance(pooled, torch.Tensor) and pooled.ndim == 3:
                    pooled = pooled.mean(dim=1)
                all_embeddings.append(pooled.detach().cpu().to(torch.float32).numpy())

        return np.vstack(all_embeddings)


class DINOv2RegistersEncoder(HFVisionEncoder):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(
            checkpoint="facebook/dinov2-with-registers-base",
            encoder_id="dinov2_registers_base",
            device=device,
        )


class SigLIP2BaseEncoder(HFVisionEncoder):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(
            checkpoint="google/siglip2-base-patch16-224",
            encoder_id="siglip2_base_patch16_224",
            device=device,
        )


class ViTBaseEncoder(HFVisionEncoder):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(
            checkpoint="google/vit-base-patch16-224-in21k",
            encoder_id="vit_base_patch16_224",
            device=device,
        )


class BEiTBaseEncoder(HFVisionEncoder):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(
            checkpoint="microsoft/beit-base-patch16-224",
            encoder_id="beit_base_patch16_224",
            device=device,
        )
