# Data Collection

## Overview

89 real classification tasks across 7 modalities. Every task is a genuine dataset from an
established source (TDC, HuggingFace Datasets, PyTorch Geometric's TUDataset collection, the
UCR/UEA time-series archive via `aeon`) — nothing is synthetically generated. See
`data_manifest.csv` for the full per-task table (domain, N, classes, class balance, source,
description).

| Domain | Tasks | Typical N per task | Classes | Primary sources |
|---|---|---|---|---|
| Molecules | 24 | 500–6000 | 2 (binary ADMET/tox assays) | Therapeutics Data Commons (TDC) |
| Proteins | 15 | 500–8000 | 2–10 | HuggingFace Datasets, TDC-adjacent peptide/protein sets |
| Vision | 10 | 200–10000 | 2–10 | HuggingFace Datasets (CIFAR-10, Fashion-MNIST, SVHN, Beans, MNIST, MedMNIST family) |
| Text | 11 | 500–3000 | 2–4 | HuggingFace Datasets (AG News, SST-2, TweetEval, IMDB, Rotten Tomatoes, etc.) |
| Audio | 7 | 200–2000 | 2–10 | HuggingFace Datasets (ESC-10, GTZAN, CREMA-D, EMO-DB, RAVDESS, UrbanSound8K, VoxForge) |
| Graphs | 10 | 188–4110 | 2–6 | PyTorch Geometric TUDataset (MUTAG, ENZYMES, NCI1, PROTEINS, etc.) |
| Time series | 12 | 50–8926 | 2–7 | UCR/UEA archive via `aeon` |

## Acceptance criteria (applied to every task, old and new)

1. **Single label per sample**, not per-residue/per-frame/per-position. This tripped up several
   candidate datasets during this session (TDC Epitope/Paratope, a SaProtHub signal-peptide set)
   that looked like ordinary classification datasets but were actually per-position tagging
   tasks wrapped in a string column — rejected after empirical inspection, not just reading the
   dataset card.
2. **2–10 classes.**
3. **≤10,000 samples** per task (larger source datasets are fine if easy to subsample).
4. **Freely downloadable**, no gating/manual steps, no streaming-only sources that are painful
   to slice.
5. **Real, non-degenerate label distribution** — every class must have genuine support (a
   dataset with one class effectively empty was rejected: `proteinea/malate_dehydrogenase`,
   100% single label).

## How new tasks were found

Proteins grew from 6 → 15 tasks, and audio grew from 4 (later corrected to 6, see below) → 7,
via a general-purpose search-and-verify pass: candidates were discovered via the HuggingFace Hub
API and TDC's dataset catalog, then *empirically* loaded and checked against the criteria above
(not trusted from the dataset card/description alone) before being fetched and saved. See
`scripts/fetch_extra_tasks.py` for the `save_task()` helper used for all new tasks — it
handles class remapping to `0..C-1`, generates deterministic 5-fold splits, and writes the
standard per-task file layout (see `06_evaluation_methodology.md` for what "fold" means here).

### Rejected candidates (documented so they aren't re-tried)

- **Per-residue/per-frame labels disguised as classification**: TDC `Epitope` (iedb_jespersen,
  pdb_jespersen), TDC `Paratope` (sabdab_liberis), `SaProtHub/Dataset-Signal-Peptides`,
  `SeprotHub/Dataset-Signal-Peptides`.
- **Degenerate labels**: `proteinea/malate_dehydrogenase` (100% one class).
- **Multi-label, not single-label**: `AI4Protein/EC`, `AI4Protein/GO_BP/CC/MF`,
  `nikolayvV/protein-function-prediction-preprocessed`.
- **Regression, not classification**: `proteinglm/optimal_ph`, `optimal_temperature`,
  `enzyme_catalytic_efficiency`, `fitness_prediction`, `fluorescence_prediction`,
  `stability_prediction`, TAPE Stability/Fluorescence, FLIP AAV/GB1.
- **Class count out of range**: `proteinglm/antibiotic_resistance` (19 classes),
  `proteinglm/fold_prediction` (1195 classes), speech_commands v0.02 (~35 word classes).
- **No real audio/data array, metadata only**: `monster-monash/WhaleSounds`,
  `monster-monash/InsectSound`, `alvgaona/heart-sounds`, `stdt1/mimii_pump_datasets`,
  `luyangliuable/circor-heart-sound`, `Fhrozen/FSD50k` (label column was literally
  `dev`/`eval`, not real classes).
- **Ethical/policy fit**: `garystafford/deepfake-audio-detection` was schema-valid but rejected
  since its "fake" class is itself synthesized audio, which sits awkwardly against this
  project's zero-synthetic-data standard even though the *labels* would have been real.

## A data-quality correction made during this session

Two originally-included audio tasks were **removed** after a full validation pass found they
violated the 2–10 class rule that had been applied to every other task:

- `audio_esc50` — 50 classes (the full ESC-50 taxonomy).
- `audio_superb_ks` — 12 classes, and severely imbalanced (one class had 1 sample out of 2000).

Both were deleted (task data and any extracted embeddings) rather than kept as a documented
exception, per an explicit decision to hold every task to the same standard.

## Known real-data quirks worth being aware of

- **Split encoding inconsistency (fixed in code, not data)**: 23 of the tasks store a genuine
  5-fold CV assignment (`folds.npy` values 0–4) rather than a plain 3-way test/val/train
  encoding (values 0/1/2). The original loader only recognized values 0/1/2 and silently
  dropped ~40% of samples for those 23 tasks before this was caught and fixed — see
  `08_known_issues_and_fixes.md`. The convention now used everywhere: fold 0 = test, fold 1 =
  val, fold ≥2 = train, which is correct for both encodings.
- **Two input-file formats beyond JSON**: graph tasks store PyG `Data` objects in `inputs.pt`
  (can't be JSON-serialized), and some time-series tasks store a plain float array in
  `inputs.npy`. The loader (`src/univarsal_embeding/core/store.py`, `TaskStore.load_task`)
  checks for `inputs.json`, then `inputs.npy`, then `inputs.pt`.
- **Class imbalance is real and not corrected for**: several molecule TDC assays have very low
  positive rates (e.g. `tdc_hiv`: 125/6000 = 2.1% positive; `tdc_kcnq2_potassium_channel_butkiewicz`:
  2/6000 = 0.03%). This is genuine biological screening data, not a bug — but it means raw
  accuracy is a misleading metric for these tasks specifically; balanced accuracy and AUROC/AUPRC
  (all reported, see `03_baselines.md` and `06_evaluation_methodology.md`) should be preferred
  when characterizing performance on them. `min_class_frac` in `data_manifest.csv` flags this
  per task.
