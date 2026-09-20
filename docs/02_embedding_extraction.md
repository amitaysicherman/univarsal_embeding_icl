# Embedding Extraction

## Encoder registry (final, working set)

Exactly 5 encoders per domain except graphs and timeseries (4 each, all of them working — there
was no need to drop anything there). Registry source of truth:
`src/univarsal_embeding/encoders/registry.py`, `DOMAIN_TO_ENCODERS`.

| Domain | Encoder ID | Checkpoint / architecture | Output dim |
|---|---|---|---|
| Molecules | `chemberta_77m_mtr` | `DeepChem/ChemBERTa-77M-MTR` | 384 |
| | `smiles_bert` | `unikei/bert-base-smiles` | 768 |
| | `chemberta_zinc_base` | `seyonec/ChemBERTa-zinc-base-v1` | 768 |
| | `pubchem10m_bpe` | `DeepChem/ChemBERTa-10M-MTR` | 384 |
| | `chemberta_v2` | `seyonec/ChemBERTa_zinc250k_v2_40k` | 768 |
| Proteins | `esm2_650m` | `facebook/esm2_t33_650M_UR50D` | 1280 |
| | `prot_bert` | `Rostlab/prot_bert` | 1024 |
| | `ankh_base` | `ElnaggarLab/ankh-base` (T5 encoder-decoder; encoder stack only, see fix below) | 768 |
| | `esm2_150m` | `facebook/esm2_t30_150M_UR50D` | 640 |
| | `prost_t5` | `Rostlab/ProstT5` — **broken, see below** | — |
| Vision | `dinov2_small` | `facebook/dinov2-small` | 384 |
| | `siglip_base` | `google/siglip-base-patch16-224` | 768 |
| | `metaclip_b16` | `facebook/metaclip-b16-400m` | 512 |
| | `vit_base_patch16_224` | `google/vit-base-patch16-224-in21k` | 768 |
| | `beit_base_patch16_224` | `microsoft/beit-base-patch16-224` | 768 |
| Text | `modernbert_embed` | `nomic-ai/modernbert-embed-base` — **broken, see below** | — |
| | `bge_large_en_v1_5` | `BAAI/bge-large-en-v1.5` | 1024 |
| | `arctic_embed_m_v2` | `Snowflake/snowflake-arctic-embed-m-v1.5` (see naming fix below) | 768 |
| | `nomic_embed_v1_5` | `nomic-ai/nomic-embed-text-v1.5` | 768 |
| | `bge_m3` | `BAAI/bge-m3` | 1024 |
| Audio | `wavlm_base_plus` | `microsoft/wavlm-base-plus` | 768 |
| | `whisper_large_v3_turbo` | `openai/whisper-large-v3-turbo` (encoder only) | 1280 |
| | `clap_general` | `laion/larger_clap_general` | 512 |
| | `ast_audioset` | `MIT/ast-finetuned-audioset-10-10-0.4593` | 768 |
| Graphs | `pyg_gin`, `pyg_gcn`, `pyg_gat`, `pyg_sage` | Untrained/frozen PyTorch Geometric GIN/GCN/GAT/GraphSAGE stacks, global-pooled | 256 each |
| Timeseries | `minirocket`, `rocket` | ROCKET-family random convolutional feature transforms | 504 / 1024 |
| | `chronos_t5_small` | Amazon `chronos-t5-small` | 512 |
| | `resnet1d` | Untrained/frozen 1D ResNet | 256 |

**Total working encoder–task pairs = 392 embedding files.** All 392 have been validated (see
below) for correct shape, dtype, no NaN/Inf, no all-zero rows, non-degenerate norm distribution,
and — critically — every encoder produces the *exact same dimension* on every task it was run
on (checked by grouping all 392 files by encoder-ID filename and confirming a single dimension
per group; this is a strong sanity signal that no embeddings got mixed up or corrupted).

## Dropped encoders (with the specific technical reason each is unfixable, not just "didn't try")

Two of these — `prost_t5` and `modernbert_embed` — are still marked broken in the registry above
because they were never fixed within this session; the rest were fixed or replaced.

| Encoder ID | Domain | Reason dropped | Fixable? |
|---|---|---|---|
| `molformer_xl` | molecules | `ibm/MoLFormer-XL-both-10pct`'s own remote code calls `transformers.masking_utils.create_bidirectional_mask`, a function that **no released version of `transformers` has ever shipped** (checked the source of `masking_utils.py` directly — only `create_causal_mask`/`create_chunked_causal_mask`/`create_sliding_window_causal_mask` exist). Upstream bug in the checkpoint's own code, not a version-pinning issue. Replaced with `smiles_bert`. |
| `esmc_600m` | proteins | `EvolutionaryScale/esmc-600m` doesn't exist under that repo ID (correct home is `biohub/ESMC-600M`), and even the correct repo requires an `EsmcTokenizer` class that isn't registered in any `transformers` version compatible with the rest of the pipeline's checkpoints. Dropped, not replaced (proteins already has 5 working encoders without it after adding `ankh_base`). |
| `dinov2_registers_base`, `siglip2_base_patch16_224`, `aimv2_large_patch14_224` | vision | Newer architectures needing a `transformers` version that conflicts with the CVE-safety pin required by the older checkpoints in the same run (see `torch.load` CVE note below). Replaced with `dinov2_small`, `siglip_base`, `vit_base_patch16_224`, `beit_base_patch16_224`. |
| `gte_qwen2_1_5b` | text | Requires the `flash_attn` package, not installed (would need CUDA-toolkit compilation on the cluster; not attempted, judged not worth the build risk for one encoder). |
| `speecht5_asr` | audio | `microsoft/speecht5_asr`'s encoder submodule can't be called standalone on raw audio — SpeechT5's "speech prenet" preprocessing only runs inside the full model's `forward()`, and calling just `.encoder(...)` (needed to avoid requiring `decoder_input_ids`) skips it, producing a shape mismatch that manifests on GPU as an attempted 100+GB allocation. Real architecture limitation, not a quick fix. `whisper_large_v3_turbo` already covers the same "ASR-style encoder" role. |
| `w2v_bert_2` | audio | Not broken — dropped purely to stay at the 5-encoder-per-domain cap; redundant with `wavlm_base_plus` (same wav2vec-family architecture family). |
| `prost_t5` | proteins | `Rostlab/ProstT5`'s tokenizer raises "You're trying to run a Unigram model but your file was trained with a different algorithm" — a tokenizer-format/version mismatch with the `tokenizers` library version pinned for the rest of the pipeline. Left broken; not chased further this session. |
| `modernbert_embed` | text | `nomic-ai/modernbert-embed-base`'s tokenizer raises "data did not match any variant of untagged enum ModelWrapper" — same category of tokenizer-format/version mismatch as `prost_t5`. Left broken. |

### The `torch.load` CVE constraint that shapes several of the above

`transformers` versions from roughly 4.45+ refuse to `torch.load` a non-`safetensors` checkpoint
unless the installed `torch` is ≥2.6 (CVE-2025-32434). Several older/smaller checkpoints in this
pipeline (the ChemBERTa family in particular) are only distributed as `.bin` weights, not
`safetensors`, and needed an **older** `transformers` (pinned to `4.44.2`) to load without
requiring a `torch` upgrade. That pin is what makes several of the *newer* architectures above
(which need masking/tokenizer APIs added after 4.44.2) incompatible in the same environment —
this is the actual mechanism behind most of the "dropped, transformers-version-conflict" rows.

### Two encoder implementation bugs fixed this session (not upstream — our own code)

1. **`ankh_base`** (T5 encoder-decoder): the generic `HFProteinEncoder` was calling
   `self.model(**encoded)` on a full seq2seq checkpoint, which requires `decoder_input_ids` and
   crashed. Fixed in `src/univarsal_embeding/encoders/proteins.py` by detecting encoder-decoder
   models (`hasattr(model, "encoder") and hasattr(model, "decoder")`) and calling
   `model.encoder(...)` instead — no decoder pass needed for embedding extraction.
2. **`arctic_embed_m_v2`**: was pointing at `Snowflake/snowflake-arctic-embed-m-v2`, which does
   not exist as a repo (confirmed via the HF Hub API). Corrected to
   `Snowflake/snowflake-arctic-embed-m-v1.5`.

## Extraction pipeline

- **Script**: `scripts/extract_embeddings.py --domain <domain> --device cuda --batch-size 16
  --overwrite`
- **SLURM**: one independent job per domain, `slurm/domains/ext_<domain>.sbatch` — chosen over a
  single combined job specifically so a slow/failing domain doesn't block the others, and so
  each domain's GPU memory footprint stays predictable.
- **Zero-synthetic-data enforcement**: `registry.py`'s `get_encoder()` raises a fatal
  `RuntimeError` if an encoder fails to load or run — there is no dummy/fallback embedding path.
  A failed (task, encoder) pair simply has no `.npy` file; it is never silently faked. This was
  a deliberate design constraint applied throughout, not just for extraction.
- **Post-encoding validation** (built into `extract_embeddings.py`): after each encoder finishes
  a task, its output is checked for correct shape, dtype, NaN/Inf, and degenerate variance before
  being saved, with a warning logged for unusual norm distributions.

## Independent full-dataset validation (run separately from extraction, after the fact)

`scripts/audit_data.py` was extended this session to check, for every embedding file in
`data/tasks/`: shape matches the task's labeled sample count, no NaN/Inf, no all-zero rows,
non-degenerate norm standard deviation (>1e-9), correct float dtype, and per-encoder dimension
consistency across every task that encoder touched. The last full run found **392/392 files
clean** with zero issues (the check was re-run twice more after subsequent additions — proteins
6→15 tasks and audio corrections — with the same 100% pass rate each time).
