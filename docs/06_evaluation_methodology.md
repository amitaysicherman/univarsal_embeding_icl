# Evaluation Methodology

Implementation: `src/univarsal_embeding/eval/splits.py` (`UniversalBenchmarkSplits`) and
`scripts/evaluate_universal.py`.

## The four generalization regimes

For each domain, tasks are split 80/20 into `train_tasks`/`test_tasks` (deterministic hash of
`f"{seed}:{task_id}"`, so the same seed always reproduces the same split without storing it),
and encoders are split similarly into `train_encoders`/`test_encoders` (the last 1–2 of each
domain's encoder list, by count-dependent rule). Four regimes fall out of the 2×2 combination:

| Regime | Tasks | Encoders | Meaning |
|---|---|---|---|
| `seen` | train | train | Both seen during meta-training |
| `unseen_tasks` | test | train | New task, familiar encoder |
| `unseen_encoders` | train | test | Familiar task, new encoder/embedding space |
| `unseen_both` | test | test | **The real generalization test** — neither seen |

A fifth regime, `unseen_domain`, is used only for leave-domain(s)-out runs (below): every task
and encoder belonging to the held-out domain(s), regardless of the 80/20 split (since the whole
domain was excluded from training).

**Current pair counts** (with 89 tasks, all working encoders): `seen`=268, `unseen_tasks`=59,
`unseen_encoders`=73, `unseen_both`=16 — not degenerate at any regime, though `unseen_both` is
noticeably the smallest (worth keeping in mind when reading its variance across seeds in
`07_results_and_ablations.md`).

## Leave-domain(s)-out (LODO)

Rather than 7 separate single-domain-held-out runs, the 7 domains are partitioned into **4
groups of sizes (2, 2, 2, 1)** — every domain held out exactly once, across 4 runs instead of 7,
for efficiency:

| Group | Held-out domains | `unseen_domain` pair count |
|---|---|---|
| g1 | audio + graphs | 68 |
| g2 | molecules + proteins | 195 |
| g3 | text + timeseries | 103 |
| g4 | vision | 50 |

`--holdout-domain` accepts a comma-separated list (e.g. `audio,graphs`) — this multi-domain
support was added specifically for this grouping (the original code only supported one domain
at a time) and was verified with an offline pair-count check (`68 = 68`, i.e. the combined
2-domain pair count exactly equals the sum of each domain's pairs alone) before being trusted
in a real GPU run. Each LODO run trains on the remaining 5–6 domains' `seen`-style pairs and
early-stops/evaluates against its own held-out group's `unseen_domain` regime — see
`05_training_methodology.md` for the shared training mechanics.

## Inference: the 16-projection ensemble

Every evaluation (final-evaluation-after-training and any standalone `evaluate_universal.py`
run) uses `forward_episodic_ensemble` with **16 distinct random projection seeds**
(`base_seed + i*10007` for `i in 0..15`), softmax-averaged. See `04_model_architecture.md` for
why this exists and the three memory/precision bugs that had to be fixed to make it run
correctly at all. `--num-projections` is the knob (default 16 for "real" results, reduced to 4
during training for cheap early-stopping checks, and swept down to 1 as a deliberate ablation —
`07_results_and_ablations.md`).

## Context handling at evaluation time

Same rule as training (`05_training_methodology.md`): the **entire training-fold set** for a
given task is used as context, never subsampled (`context_size: int | None = None` in
`evaluate_pair`, default unclipped). Verified directly against the deployed code, not assumed —
both the training-time and evaluation-time context-construction code paths were re-checked
line-by-line after the mid-session clipping bug was found.

## Metrics

Same set as the baselines (`03_baselines.md`): accuracy, balanced accuracy, macro F1, MCC,
ROC-AUC, AUPRC, log loss — computed per (domain, task, encoder) pair, then averaged (unweighted
mean across pairs) per regime to produce `final_evaluation_summary.json`'s `mean_<metric>`
fields.
