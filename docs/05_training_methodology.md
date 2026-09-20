# Training Methodology

Implementation: `scripts/train_tabicl_lora.py`.

## Episodic meta-training

`EpisodicMetaDataset` pre-loads every (domain, task, encoder) triple assigned to the "seen"
training split (see `06_evaluation_methodology.md` for how that split is chosen) into memory —
embeddings, labels, and fold assignments. A triple is **excluded** if it has fewer than 10
training rows, fewer than 5 validation rows, or fewer than 2 classes present in its training
rows (too small to form a meaningful episode).

Each meta-training step (`sample_episode`) picks one random (task, encoder) episode from that
pool and constructs:

- **Context = every training-fold row for that task/encoder, never subsampled.** This is a
  deliberate, explicit design decision (`max_context: int | None = None` — the default is
  unclipped), matching how the official TabICL library itself is used: it never subsamples
  training rows for context, only ever batches the *ensemble/query* dimension for memory. A bug
  that silently violated this (defaulting to a 256-row cap) was found and fixed mid-session —
  see `08_known_issues_and_fixes.md` for the full story, since it took three separate rounds of
  GPU-memory engineering (documented in `04_model_architecture.md`) to make full-context
  actually fit in memory once the cap was removed.
- **Query = a random minibatch of up to `max_query` (default 64) validation-fold rows**, with
  labels remapped locally to `0..C-1` for that episode.

The forward pass, loss (`F.cross_entropy`), and optimizer step all happen per-step exactly once
per episode — this is standard episodic/meta-learning, not batched across multiple tasks per
step. Gradients are clipped to norm 1.0. Only LoRA parameters (and, optionally, a learnable
projection) are trainable; the base TabICL weights are frozen throughout.

**A fresh random projection matrix is drawn every single meta-step** (`dynamic_rp=True,
projection_seed=None` — with no explicit seed, `sample_orthogonal_matrix` draws from the global
PyTorch RNG, which advances on every call). Verified directly against the running code, not
assumed.

## Hyperparameters and the search that picked them

Swept: LoRA rank `r ∈ {8, 16, 32}` and learning rate `∈ {3e-4, 1e-4}` (4 configs: r8/lr3e-4,
r16/lr3e-4, r16/lr1e-4, r32/lr3e-4), each trained for a short 1500 meta-steps as a quick search
pass (`target_dim=256`, `lora_alpha=32.0`, `weight_decay=1e-4`, AdamW, cosine LR schedule with
`eta_min = lr * 0.05`).

**Finding**: differences between configs were mostly small (mean spread across configs on 86
shared completed pairs: 2.3 percentage points accuracy, median 1.3pp) — the handful of pairs
with a large spread (>10pp) were concentrated in small, noisy graph datasets (few hundred
samples), not a real hyperparameter signal. **`r=8, lr=3e-4` won the most pairs outright (40/86,
47%)** despite having the fewest trainable parameters (467,992) — evidence that, at this training
budget, extra LoRA capacity isn't being used, or risks mild overfitting on the smaller tasks.
**`r=8, lr=3e-4` was selected as the config for all subsequent longer training runs.**

## Longer training with early stopping

For the real (non-search) runs, meta-steps was raised to up to 8000, with **early stopping**
added directly into `train_meta_lora()`:

- Every `--eval-interval` steps (1000 in the final runs; a smoke test first confirmed 500 was
  needlessly expensive — see below), run a **fast** approximate evaluation
  (`--eval-num-projections=4`, not the full 16) on a held-out regime — `unseen_tasks` by default,
  or `unseen_domain` (the held-out group) automatically for LODO runs.
- Track the best held-out **balanced accuracy** seen so far; save it separately as
  `tabicl_lora_best.pt` whenever it improves.
- Stop early if `--patience` (3) consecutive checks show no improvement.
- **At the end of training (whether by early stop or hitting the step limit), automatically
  reload the best checkpoint and run the full final evaluation** (all 4 regimes, full 16-way
  ensemble) in the same script run, writing `<output-dir>/final_evaluation_summary.json`. This
  means training and evaluation are one pipeline/one job — no separate script invocation
  required to get a checkpoint's final numbers.

A 30-step smoke test (`--meta-steps 30 --eval-interval 10 --patience 5`) was run first to
confirm the full mechanism (checkpoint tracking, no-improvement counting, final-eval chaining)
worked correctly before committing real GPU time — it did, and also produced the empirical
timing (~250 seconds per fast early-stop check on ~57 pairs) used to choose `--eval-interval
1000` over `500` for the real runs (500 would have added ~67 minutes of pure early-stopping
overhead per run over an 8000-step budget; 1000 halves that to ~33 minutes).

## The 4 seeds

Four independent runs, `--seed 42/123/456/789`, all with `r=8, lr=3e-4, meta-steps=8000,
eval-interval=1000, patience=3`. **The seed controls both the task/encoder train-test split
(`UniversalBenchmarkSplits`) and the episodic sampling RNG** — so each seed run trains on and is
evaluated against a genuinely different partition of tasks/encoders into "seen" vs "unseen",
not just a different training trajectory on the same split. This is why a given (task, encoder)
pair can fall under a *different* generalization regime in different seed runs (see
`07_results_and_ablations.md`'s combined results table, which records the regime per seed per
pair for exactly this reason).

Outcomes: `seed=42` ran the full 8000 steps without triggering early stopping (best at step
6000); `seed=123/456/789` all stopped early (steps 5000/4000/4000 respectively, best checkpoints
at steps 2000/1000/1000).

## Checkpoints

Each run's directory (`checkpoints/<run_name>/`) contains: `tabicl_lora_best.pt` (the one to
use — reloaded automatically before final evaluation), periodic `tabicl_lora_step<N>.pt` saves
(every `--save-interval`, default 250 — these are just training-progress snapshots, not
meaningfully different from the best checkpoint for most purposes), and
`final_evaluation_summary.json` (see `07_results_and_ablations.md`).
