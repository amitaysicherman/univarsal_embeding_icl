# Model Architecture

Implementation: `src/univarsal_embeding/models/tabicl_universal.py`, class `UniversalTabICL`.

## Base model

**TabICL-v2**, a pretrained tabular in-context-learning transformer, loaded from
`jingang/TabICL` on HuggingFace Hub, checkpoint file `tabicl-classifier-v2-20260212.ckpt`. TabICL
takes a single tensor of shape `(1, N_context + N_query, D)` plus the context labels, and
produces per-class logits for every row (context and query) via in-context learning — no
gradient-based adaptation is required to use it "as is" (this is the `tabicl_zeroshot` baseline
in `03_baselines.md`).

## Why a random projection layer

Every domain's foundation encoders produce a different embedding width (256–1536 dimensions
across the 32 working encoders — see `02_embedding_extraction.md`), but TabICL expects a fixed
input width. `sample_orthogonal_matrix(in_dim, target_dim=256, seed=None)` draws a random
orthonormal matrix via QR decomposition of a Gaussian random matrix (`W` such that
`Wᵀ W = I`), so any encoder's embedding can be linearly projected down to TabICL's 256-d input
space while approximately preserving relative distances (a Johnson–Lindenstrauss-style
guarantee). When `in_dim == target_dim` exactly, the identity matrix is used (no-op).

**Crucially, a fresh random projection is drawn every time** (unless an explicit seed is
passed) — during training, every meta-step gets a new projection; during evaluation, 16
different explicit seeds are used and their predictions averaged (below). This is a form of
regularization: the LoRA weights are optimized to work well *across* random projections, not to
overfit to one specific projection.

An alternative **learnable** projection (`UniversalRandomProjection` with `learnable=True`) also
exists in the codebase but was not used for the main results — see `07_results_and_ablations.md`
for its status as a planned ablation.

## LoRA adaptation

Low-rank adapters (`LoRALinear`) are injected into TabICL's attention linear projections
(`linear1`, `linear2`, `out_proj`, `in_linear` — matched by substring on submodule name), while
the base TabICL weights are frozen (`requires_grad = False`). Each `LoRALinear` adds
`B @ A` (scaled by `lora_alpha / r`) to the frozen layer's output; `A` is Kaiming-initialized,
**`B` is zero-initialized**, so an untouched/freshly-constructed model with LoRA injected but no
trained checkpoint loaded is mathematically identical to the pure base model (this is exploited
directly to implement the fair "zero-shot + ensemble" ablation, see `07_results_and_ablations.md`
— no special-casing needed, just don't load a checkpoint).

`lora_r` (rank) and `lora_alpha` are the two LoRA hyperparameters swept in
`05_training_methodology.md`.

## Forward pass

`forward_episodic(X_ctx, y_ctx, X_query, dynamic_rp, projection_seed, learnable_rp)`:
1. Project both context and query embeddings through the (random or learned) projection.
2. Standardize using the **context/support set's** mean and std only (never the query's).
3. Concatenate context + query along the row dimension, run through TabICL, slice logits down
   to the actual number of classes present in this episode's context labels.

`forward_episodic_ensemble(..., num_projections=16, base_seed=42)`: runs `forward_episodic`
`num_projections` times, each with a distinct seed (`base_seed + i * 10007` for
`i in 0..num_projections-1`), softmaxes each, and averages the probabilities. Used for all
evaluation (never for training — see below). Internally wrapped in `torch.no_grad()` since it's
inference-only (see the memory bug below for why this specifically matters).

## Three GPU memory/precision bugs found and fixed while building this

These were discovered because the project enforces "use the **full** training set as context,
never subsample it" (see `05_training_methodology.md`) — which pushes context lengths up to
6254 rows for the largest task (`ts_electric_devices`), well beyond what a naive forward pass
fits in GPU memory. In order of discovery:

1. **Plain fp32 forward pass OOMs.** A single forward pass with the full 6254-row context used
   41.89 GB on a 47 GB A40 GPU — leaving no room for a backward pass. **Fix**: enable TabICL's
   built-in `recompute=True` (gradient checkpointing — trades compute for memory by not caching
   activations for backward, a pure engineering tradeoff, not a data or correctness compromise)
   combined with `bf16` autocast (`torch.autocast(device_type="cuda", dtype=torch.bfloat16)`)
   around the TabICL forward call. This brought a full forward+backward training step down to
   **13.11 GB**.

2. **The 16-projection evaluation ensemble still OOMs even with the above fix.** Root cause:
   each of the 16 sequential `forward_episodic` calls inside `forward_episodic_ensemble` was
   building its own autograd graph without releasing it, multiplying memory ~16×. **Fix**: wrap
   the ensemble loop in `torch.no_grad()` — inference never needs gradients, so nothing should
   be tracked in the first place. This also revealed that `recompute=True` (gradient
   checkpointing) is actively **incompatible** with running under `no_grad`/`inference_mode` in
   this specific TabICL implementation (a `ValueError` about device indices, traced below) —
   checkpointing exists specifically to save memory across a backward pass, so it is now set to
   `recompute=False` for any purely-inference model instance (`evaluate_universal.py`'s model,
   `TabICLZeroShotBaseline`), and `recompute=True` only for the training model.

3. **A genuine upstream bug in the third-party `tabicl` library itself**, surfaced only once
   (1) and (2) above were fixed and the ensemble ran under `no_grad` for the first time. TabICL's
   internal `InferenceManager` (`tabicl/model/inference.py`) estimates a safe batch size via
   `torch.cuda.mem_get_info(self.exe_device)`, but falls back to a bare, **index-less**
   `torch.device("cuda")` in one code path (line ~725) rather than an indexed
   `torch.device("cuda", 0)`. Newer PyTorch's `mem_get_info` rejects index-less devices with
   `ValueError: Expected a torch.device with a specified index or an integer`. This is a bug in
   the pretrained model's packaging, not anything under this project's control — traced by
   reading the full traceback down to `tabicl/model/inference.py:783`. **Fix**: a small, targeted
   monkey-patch of `torch.cuda.mem_get_info` (applied once at import time in
   `tabicl_universal.py`, also invoked from `TabICLZeroShotBaseline`) that normalizes any
   index-less CUDA device to an explicit indexed one before delegating to the real function.
   Transparent, doesn't touch model weights or data, purely a device-handling correctness fix.

**End-to-end confirmed memory usage after all three fixes**: 13.11 GB for a full forward+backward
training step, 6.53 GB for a full 16-projection evaluation ensemble pass — both on the largest
(6254-row-context) task, comfortably within a single 47 GB A40.

A related but separate bug (bf16 logits reaching `.numpy()` before being cast back to float32,
since NumPy has no bfloat16 dtype) is documented in `03_baselines.md`, since it manifested there
first — the same `.float()`-before-`.numpy()` fix was applied everywhere the same pattern
occurred (`tabicl_universal.py`, `baselines.py`, `evaluate_universal.py`).
