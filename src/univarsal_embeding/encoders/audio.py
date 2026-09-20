"""Audio foundation encoders (Whisper-turbo, WavLM, CLAP)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from .base import BaseEncoder

logger = logging.getLogger(__name__)


def load_wav(path_str: str, target_sr: int = 16000) -> np.ndarray:
    import wave
    with wave.open(path_str, "r") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)
        sig = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if wf.getnchannels() > 1:
            sig = sig.reshape(-1, wf.getnchannels()).mean(axis=1)
    if sr != target_sr:
        new_len = int(len(sig) * float(target_sr) / sr)
        sig = np.interp(np.linspace(0, 1, new_len), np.linspace(0, 1, len(sig)), sig).astype(np.float32)
    return sig


class HFAudioEncoder(BaseEncoder):
    """General HuggingFace audio encoder (WavLM, AST, Seamless)."""

    def __init__(
        self,
        checkpoint: str,
        encoder_id: str,
        device: str = "cpu",
        target_sr: int = 16000,
    ) -> None:
        super().__init__(encoder_id=encoder_id, domain="audio", device=device)
        from transformers import AutoFeatureExtractor, AutoModel

        self.processor = AutoFeatureExtractor.from_pretrained(checkpoint, trust_remote_code=True)
        sr = getattr(self.processor, "sampling_rate", None)
        self.target_sr = sr if sr is not None else target_sr
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device).eval()
        self.embedding_dim = getattr(self.model.config, "hidden_size", 768)

    def encode(self, inputs: Sequence[str], batch_size: int = 16, **kwargs) -> np.ndarray:
        all_embeddings = []
        n_samples = len(inputs)

        with torch.inference_mode():
            for i in range(0, n_samples, batch_size):
                batch_inputs = inputs[i : i + batch_size]
                waveforms = [load_wav(p, self.target_sr) for p in batch_inputs]

                if "whisper" in self.encoder_id:
                    inputs_tensor = self.processor(
                        waveforms,
                        sampling_rate=self.target_sr,
                        return_tensors="pt",
                    ).to(self.device)
                    outputs = self.model.encoder(**inputs_tensor)
                    pooled = outputs.last_hidden_state.mean(dim=1)
                elif hasattr(self.model, "get_audio_features"):
                    inputs_tensor = self.processor(
                        waveforms,
                        sampling_rate=self.target_sr,
                        return_tensors="pt",
                        padding=True,
                    ).to(self.device)
                    pooled = self.model.get_audio_features(**inputs_tensor)
                else:
                    inputs_tensor = self.processor(
                        waveforms,
                        sampling_rate=self.target_sr,
                        return_tensors="pt",
                        padding=True,
                    ).to(self.device)
                    # Encoder-decoder checkpoints (e.g. SpeechT5) load as the full
                    # seq2seq model under AutoModel; calling them without
                    # decoder_input_ids triggers pathological default decoding
                    # behavior, so run only the encoder stack instead.
                    if hasattr(self.model, "encoder") and hasattr(self.model, "decoder"):
                        outputs = self.model.encoder(**inputs_tensor)
                    else:
                        outputs = self.model(**inputs_tensor)
                    if hasattr(outputs, "last_hidden_state"):
                        pooled = outputs.last_hidden_state.mean(dim=1)
                    elif hasattr(outputs, "pooler_output"):
                        pooled = outputs.pooler_output
                    else:
                        pooled = outputs[0].mean(dim=1)

                all_embeddings.append(pooled.detach().cpu().to(torch.float32).numpy())

        return np.vstack(all_embeddings)


class WavLMBaseEncoder(HFAudioEncoder):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(
            checkpoint="microsoft/wavlm-base-plus",
            encoder_id="wavlm_base_plus",
            device=device,
        )
