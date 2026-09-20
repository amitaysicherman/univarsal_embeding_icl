# Baselines

## Models

Five baseline classifiers are run on every (task, encoder) embedding pair, defined in
`src/univarsal_embeding/models/baselines.py` and orchestrated by `scripts/run_baselines.py`:

| Name | Implementation | Notes |
|---|---|---|
| `linear` | scikit-learn `LogisticRegression` (C=1.0, L2) on standardized features | |
| `knn` | scikit-learn `KNeighborsClassifier` (k=5, cosine metric, distance-weighted), standardized features | k is capped at `n_train - 1` for tiny tasks |
| `mlp` | PyTorch MLP, **exactly one hidden layer** (input → hidden(256) → BatchNorm → ReLU → Dropout → output), trained with AdamW, early-stopped on a held-out val fold | Deliberately kept to one hidden layer per an explicit spec — an earlier 2-hidden-layer version was corrected |
| `xgboost` | `xgboost.XGBClassifier` (300 estimators, depth 6, lr 0.1) | Binary `logistic` or multi-class `softprob` objective chosen automatically from the label count |
| `tabicl_zeroshot` | The frozen pretrained TabICL-v2 model used **without any LoRA fine-tuning** — the training set is the in-context "support set", the test set is the query | See caveat below: as originally run, this used a single random projection per call, not the 16-way ensemble used everywhere else — a fairer, ensemble-matched version was added later as an ablation (`07_results_and_ablations.md`) |

All five use the **full training set as context/support** with no subsampling — see
`05_training_methodology.md` for why "full context, never clipped" is a deliberate,
session-wide design decision (and the bug that once violated it).

## Metrics

Computed by `src/univarsal_embeding/eval/evaluator.py`, `compute_classification_metrics()`:
accuracy, balanced accuracy, macro F1, **MCC** (Matthews correlation coefficient — robust to
class imbalance, several molecule tasks have <1% positive rate), **ROC-AUC** (binary: single
positive-class probability column; multi-class: one-vs-rest macro), **AUPRC** (average
precision — more informative than ROC-AUC under severe imbalance), and log loss.

## Where baselines run: CPU vs GPU split

`linear`/`knn`/`mlp`/`xgboost` run on a CPU-only cluster partition
(`slurm/baselines/run_baselines_darwin.sbatch`) — the GPU cluster partition explicitly rejects
CPU-only job submissions ("CPU-only jobs are not allowed on the public GPU infrastructure"), and
these four are fast enough on CPU regardless.

`tabicl_zeroshot` runs separately on the GPU cluster partition
(`slurm/baselines/run_baselines_tabicl_gpu.sbatch`) — running it on CPU was tried first and
found to take **577–1183 seconds per single (task, encoder) pair** with a full unclipped
context (would have been ~90+ hours for all 392 pairs); on GPU with the memory/precision fixes
described in `04_model_architecture.md`, the same computation takes single-digit seconds per
pair.

## A precision bug found and fixed mid-session (important for interpreting older result files)

The GPU `tabicl_zeroshot` baseline runs bf16 autocast for speed. Early runs computed the
softmax **on the bf16 logits**, which left predicted-probability rows summing to 0.998–1.002
instead of exactly 1.0. scikit-learn's multi-class `roc_auc_score` validator rejects this
outright and silently returned `NaN` — but **only for multi-class tasks**, since binary AUROC
only reads a single probability column and never checks the row sum. This was caught by
noticing that 114/392 baseline results had `AUROC=NaN`, and that the affected set was *exactly*
every multi-class task and *never* a binary one — too clean a pattern to be random noise.

**Fix**: cast logits to `float32` *before* the softmax, not after (`logits.float()` prior to
`torch.softmax`, not `torch.softmax(logits).float()` afterward) — normalizing in full precision
fixes the row sums to within normal float32 tolerance (~1e-7). Applied in
`src/univarsal_embeding/models/baselines.py` and `src/univarsal_embeding/models/tabicl_universal.py`.
**Accuracy itself was never affected** by this bug (argmax is invariant to the tiny
normalization error) — only AUROC, AUPRC, and log loss for multi-class tasks in the affected
runs. If working from an older `baselines_gpu_tabicl/*.json`, check its file mtime against when
this fix landed before trusting its non-accuracy metrics.

## Results location

- `results/baselines/*.json` — CPU baselines (`linear`/`knn`/`mlp`/`xgboost`), one file per
  domain, keyed by `"<task>::<encoder>"`.
- `results/baselines_gpu_tabicl/*.json` — `tabicl_zeroshot`, same key format, post-precision-fix.
- `results/baseline_scores_accuracy.csv` — all of the above merged into one wide table: one row
  per (domain, task, encoder), one accuracy+AUROC column pair per model, plus task-context
  columns (`num_samples`, `train_size`/`val_size`/`test_size`, `num_classes`, `min_class_frac`,
  `task_type`, `source`).
- `results/baseline_scores_with_ours.csv` — the same table with our fine-tuned model's results
  added (see `07_results_and_ablations.md`).

## A sanity check worth repeating for the paper

A spot-check against published numbers found the baselines behave sensibly: MNIST/CIFAR-10
linear-probe accuracies land in the expected 88–98% range for strong embeddings (SigLIP,
DINOv2), SST-2 with BGE-large lands at 92–94% (in line with published sentence-embedding
linear-probe results), and audio encoders show the expected specialization — CLAP/AST (general
audio) score 95–100% on ESC-10 while WavLM/wav2vec-family (speech-specialized) score only
73–88%, exactly as domain expertise would predict. The one place raw accuracy is *not* a
reliable signal is the low-prevalence TDC assays (see `01_data_collection.md`) — `tdc_hiv` shows
98% accuracy but only 0.52-0.59 AUROC for most baselines (a near-trivial majority-class
prediction), which matches why the official TDC leaderboard reports AUROC, not accuracy, for
this exact task.
