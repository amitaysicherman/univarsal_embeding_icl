"""Central registry and factory for all foundation encoders (4 SOTA per domain)."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from .base import BaseEncoder, DummyEncoder
from .molecules import ChemBERTaEncoder, MoLFormerXLEncoder, SmilesBERTEncoder, HFSmilesEncoder
from .proteins import ESM2_650MEncoder, ProtBERTEncoder, HFProteinEncoder
from .vision import DINOv2RegistersEncoder, SigLIP2BaseEncoder, ViTBaseEncoder, BEiTBaseEncoder, HFVisionEncoder
from .text import ModernBERTEncoder, BGELargeEncoder, HFTextEncoder
from .audio import WavLMBaseEncoder, HFAudioEncoder

from .graphs import PyGGINEncoder, PyGGCNEncoder, PyGGATEncoder, PyGSAGEEncoder
from .timeseries import MiniRocketEncoder, RocketEncoder, ChronosT5Encoder, ResNet1DEncoder

logger = logging.getLogger(__name__)

# 4 SOTA Foundation Encoders per Domain
ENCODER_FACTORIES = {
    # --- Molecules ---
    "chemberta_77m_mtr": lambda dev: ChemBERTaEncoder(device=dev),
    "molformer_xl": lambda dev: MoLFormerXLEncoder(device=dev),
    "chemberta_zinc_base": lambda dev: HFSmilesEncoder("seyonec/ChemBERTa-zinc-base-v1", "chemberta_zinc_base", device=dev),
    "pubchem10m_bpe": lambda dev: HFSmilesEncoder("DeepChem/ChemBERTa-10M-MTR", "pubchem10m_bpe", device=dev),
    "chemberta_v2": lambda dev: HFSmilesEncoder("seyonec/ChemBERTa_zinc250k_v2_40k", "chemberta_v2", device=dev),
    "smiles_bert": lambda dev: SmilesBERTEncoder(device=dev),

    # --- Proteins ---
    "esm2_650m": lambda dev: ESM2_650MEncoder(device=dev),
    "prot_bert": lambda dev: ProtBERTEncoder(device=dev),
    "esmc_600m": lambda dev: HFProteinEncoder("EvolutionaryScale/esmc-600m", "esmc_600m", device=dev),
    "ankh_base": lambda dev: HFProteinEncoder("ElnaggarLab/ankh-base", "ankh_base", device=dev),
    "esm2_150m": lambda dev: HFProteinEncoder("facebook/esm2_t30_150M_UR50D", "esm2_150m", device=dev),
    "prost_t5": lambda dev: HFProteinEncoder("Rostlab/ProstT5", "prost_t5", device=dev),

    # --- Vision ---
    "dinov2_registers_base": lambda dev: DINOv2RegistersEncoder(device=dev),
    "siglip2_base_patch16_224": lambda dev: SigLIP2BaseEncoder(device=dev),
    "aimv2_large_patch14_224": lambda dev: HFVisionEncoder("apple/aimv2-large-patch14-224", "aimv2_large_patch14_224", device=dev),
    "dinov2_small": lambda dev: HFVisionEncoder("facebook/dinov2-small", "dinov2_small", device=dev),
    "siglip_base": lambda dev: HFVisionEncoder("google/siglip-base-patch16-224", "siglip_base", device=dev),
    "metaclip_b16": lambda dev: HFVisionEncoder("facebook/metaclip-b16-400m", "metaclip_b16", device=dev),
    "vit_base_patch16_224": lambda dev: ViTBaseEncoder(device=dev),
    "beit_base_patch16_224": lambda dev: BEiTBaseEncoder(device=dev),

    # --- Text ---
    "modernbert_embed": lambda dev: ModernBERTEncoder(device=dev),
    "bge_large_en_v1_5": lambda dev: BGELargeEncoder(device=dev),
    "gte_qwen2_1_5b": lambda dev: HFTextEncoder("Alibaba-NLP/gte-Qwen2-1.5B-instruct", "gte_qwen2_1_5b", device=dev),
    "arctic_embed_m_v2": lambda dev: HFTextEncoder("Snowflake/snowflake-arctic-embed-m-v1.5", "arctic_embed_m_v2", device=dev),
    "nomic_embed_v1_5": lambda dev: HFTextEncoder("nomic-ai/nomic-embed-text-v1.5", "nomic_embed_v1_5", device=dev),
    "bge_m3": lambda dev: HFTextEncoder("BAAI/bge-m3", "bge_m3", device=dev),

    # --- Audio ---
    "wavlm_base_plus": lambda dev: WavLMBaseEncoder(device=dev),
    "whisper_large_v3_turbo": lambda dev: HFAudioEncoder("openai/whisper-large-v3-turbo", "whisper_large_v3_turbo", device=dev),
    "clap_general": lambda dev: HFAudioEncoder("laion/larger_clap_general", "clap_general", device=dev),
    "ast_audioset": lambda dev: HFAudioEncoder("MIT/ast-finetuned-audioset-10-10-0.4593", "ast_audioset", device=dev),
    "speecht5_asr": lambda dev: HFAudioEncoder("microsoft/speecht5_asr", "speecht5_asr", device=dev),
    "w2v_bert_2": lambda dev: HFAudioEncoder("facebook/w2v-bert-2.0", "w2v_bert_2", device=dev),

    # --- Graphs ---
    "pyg_gin": lambda dev: PyGGINEncoder(device=dev),
    "pyg_gcn": lambda dev: PyGGCNEncoder(device=dev),
    "pyg_gat": lambda dev: PyGGATEncoder(device=dev),
    "pyg_sage": lambda dev: PyGSAGEEncoder(device=dev),

    # --- Time Series ---
    "minirocket": lambda dev: MiniRocketEncoder(device=dev),
    "rocket": lambda dev: RocketEncoder(device=dev),
    "chronos_t5_small": lambda dev: ChronosT5Encoder(device=dev),
    "resnet1d": lambda dev: ResNet1DEncoder(device=dev),
}

DOMAIN_TO_ENCODERS: Dict[str, List[str]] = {
    # Capped at 5 encoders per domain. Encoders dropped below are known-broken
    # upstream/version incompatibilities that could not be fixed without
    # regressing other encoders in the same domain:
    #   molecules: molformer_xl (MoLFormer-XL's own remote code calls
    #     transformers.masking_utils.create_bidirectional_mask, which no
    #     public transformers release ships) -> replaced with smiles_bert.
    #   proteins: esmc_600m (needs an EsmcTokenizer class not present in any
    #     transformers version compatible with our other checkpoints) -> dropped.
    #   vision: dinov2_registers_base, siglip2_base_patch16_224,
    #     aimv2_large_patch14_224 (all need transformers versions newer than
    #     what our CVE-safe torch.load pin allows) -> replaced with
    #     vit_base_patch16_224 and beit_base_patch16_224.
    #   text: gte_qwen2_1_5b (requires the flash_attn package, not installed) -> dropped.
    #   audio: w2v_bert_2 (redundant wav2vec-family SSL encoder, dropped to
    #     make room under the cap; wavlm_base_plus covers that niche).
    #     speecht5_asr also dropped: its encoder submodule cannot be called
    #     standalone on raw audio (SpeechT5's speech "prenet" preprocessing
    #     only runs inside the full model's forward()), which on GPU manifests
    #     as a broken broadcast attempting a 100+GB allocation. whisper_large_v3_turbo
    #     already covers the ASR-style-encoder niche cleanly.
    "molecules": ["chemberta_77m_mtr", "smiles_bert", "chemberta_zinc_base", "pubchem10m_bpe", "chemberta_v2"],
    "proteins": ["esm2_650m", "prot_bert", "ankh_base", "esm2_150m", "prost_t5"],
    "vision": ["dinov2_small", "siglip_base", "metaclip_b16", "vit_base_patch16_224", "beit_base_patch16_224"],
    "text": ["modernbert_embed", "bge_large_en_v1_5", "arctic_embed_m_v2", "nomic_embed_v1_5", "bge_m3"],
    "audio": ["wavlm_base_plus", "whisper_large_v3_turbo", "clap_general", "ast_audioset"],
    "graphs": ["pyg_gin", "pyg_gcn", "pyg_gat", "pyg_sage"],
    "timeseries": ["minirocket", "rocket", "chronos_t5_small", "resnet1d"],
}


def get_encoder(
    encoder_id: str,
    domain: str,
    device: str = "cpu",
    use_dummy: bool = False,
    dummy_dim: int = 768,
) -> BaseEncoder:
    """Returns an initialized foundation encoder instance (or dummy pseudo-encoder for fast testing).

    CRITICAL: This project requires 100% authentic embeddings. Dummy encoders are ONLY allowed when
    use_dummy=True is explicitly passed (for testing/demo purposes). Any encoder loading failure will
    raise an exception rather than silently falling back to synthetic data.
    """
    if use_dummy:
        return DummyEncoder(encoder_id=encoder_id, domain=domain, embedding_dim=dummy_dim, device=device)

    if encoder_id not in ENCODER_FACTORIES:
        raise ValueError(
            f"FATAL: Encoder '{encoder_id}' is not registered in ENCODER_FACTORIES. "
            f"Valid encoders for domain '{domain}': {DOMAIN_TO_ENCODERS.get(domain, [])}"
        )

    try:
        return ENCODER_FACTORIES[encoder_id](device)
    except Exception as e:
        raise RuntimeError(
            f"FATAL: Could not load encoder '{encoder_id}' on device '{device}'. "
            f"This project requires 100% authentic embeddings—no synthetic fallbacks allowed. "
            f"Please fix the underlying issue and retry. Original error: {type(e).__name__}: {e}"
        ) from e
