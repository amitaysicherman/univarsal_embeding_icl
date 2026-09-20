# Known Issues and Fixes

A consolidated log of every real bug found and fixed across this project's development, kept
here so (a) a methods/limitations section can cite them accurately, and (b) nobody re-discovers
the same issue from scratch. Organized by area; each entry has the actual failure mode,
diagnosis method, and fix — not just "there was a bug."

## Data pipeline

| Bug | Symptom | Diagnosis | Fix |
|---|---|---|---|
| Fold-encoding mismatch | `TaskData.__post_init__` raised `Split count 621 does not equal total samples 1034` for `vision_beans` (and 22 other tasks) | Inspected `folds.npy` directly — found values `{0,1,2,3,4}` (genuine 5-fold CV), not the `{0,1,2}` the loader expected | `store.py`: fold 0=test, fold 1=val, **fold ≥2 = train** (was `fold == 2` only) — handles both encodings, no data loss |
| Multi-format inputs | `FileNotFoundError: inputs.json` for 12 tasks (6 graph, 6 timeseries) | Checked disk — those tasks had `inputs.pt` (PyG objects, can't be JSON-serialized) or `inputs.npy` (plain float arrays) instead | `store.py`: check `inputs.json`, then `.npy`, then `.pt` |
| Missing `split_type` field | `TypeError: TaskMetadata.__init__() missing 1 required positional argument: 'split_type'` for vision/graphs/timeseries | Some tasks' `metadata.json` simply lacked this field | `schema.py`: default it from `task_type` if absent, matching the pattern already used for other optional fields |
| Graph edge_index double-transpose | 6/10 graph tasks failed batch construction with tensor-size-mismatch errors | Traced to `_build_batch` unconditionally transposing `edge_index` — correct for JSON-stored edge-pair-lists, but PyG `Data` objects (loaded from `.pt`) already have `edge_index` in the correct `[2,E]` layout, so transposing corrupted it to `[E,2]` | `graphs.py`: branch on `isinstance(g, Data)` — use PyG tensors as-is, only transpose the JSON-list format |
| Non-2-10-class tasks slipped through | `audio_esc50` (50 classes) and `audio_superb_ks` (12 classes, severely imbalanced) were in the dataset despite the stated 2–10 class rule | Full validation pass explicitly checking class count against policy | Both removed (task data + any embeddings) |

## Embedding extraction environment

| Bug | Symptom | Diagnosis | Fix |
|---|---|---|---|
| Broken conda env | `ModuleNotFoundError: No module named 'torch.nn'` inside jobs that reported success (exit 0) in seconds | Per-domain try/except in `extract_embeddings.py` swallowed the crash and moved on, making 0 real output look like a fast success | Reinstalled a consistent torch/torchvision/numpy/transformers version set (final: torch 2.4.1+cu121, torchvision 0.19.1+cu121, transformers 4.44.2, numpy 1.26.4) |
| `sbatch --wrap` shell mismatch | `source: not found` — jobs failed instantly | `--wrap` runs under `/bin/sh`, which has no `source` builtin | Switched to real `.sbatch` files with `#!/bin/bash` shebangs |
| GPU quota confusion | Jobs stuck `PD` with reason `QOSMaxGRESPerUser` | Checked `sacctmgr show qos` for the GPU partition | Cluster caps at 4 GPUs/user — expected, not a bug; jobs queue and cycle automatically |
| `ankh_base` / `arctic_embed_m_v2` | See `02_embedding_extraction.md` | | |

## `tabicl_zeroshot` baseline and evaluation

| Bug | Symptom | Diagnosis | Fix |
|---|---|---|---|
| Wrong inference mode | Baseline called `.train()`, not `.eval()`, before inference | Checked the official `tabicl` sklearn wrapper's own source (`base.py`, `classifier.py`) — it always calls `.eval()` | Changed `.train()` → `.eval()` in `TabICLZeroShotBaseline._load_model` |
| CPU too slow | 577–1183 seconds per single (task, encoder) pair on CPU (would be ~90h for all 392) | Timed directly from job logs | Moved to a dedicated GPU job (`03_baselines.md`) |
| bf16-to-NumPy | `Got unsupported ScalarType BFloat16`, silently caught and replaced with a uniform/random-guess prediction (AUROC exactly 0.5000 on every affected pair) | NumPy has no bfloat16 dtype; `.cpu().numpy()` on a bf16 tensor raises, and the surrounding `except` block masked it as a graceful fallback | `.float()` cast before `.cpu().numpy()` everywhere this pattern occurred |
| bf16 softmax precision | 114/392 multi-class results had `AUROC=NaN`; row sums off by up to 0.2% (0.998–1.002) | Reproduced directly: printed row sums, found them just outside sklearn's tolerance; binary AUROC never hit it since it only reads one column | Cast logits to float32 **before** softmax, not after — normalize in full precision |
| Missing metric fields silently | `AUPRC`/`MCC` showed `NaN` in every evaluation log line | Root cause: `evaluator.py` had been *edited locally but never actually synced to the cluster* — the deployed copy simply didn't have the AUPRC/MCC code at all (confirmed by testing `compute_classification_metrics` on the cluster directly and seeing the keys missing from the returned dict) | Synced the file for the first time; re-ran affected evaluation jobs |
| The 3 GPU memory/precision bugs (recompute+bf16, no_grad ensemble, `mem_get_info` monkeypatch) | See `04_model_architecture.md` for full detail | | |

## Training pipeline

| Bug | Symptom | Diagnosis | Fix |
|---|---|---|---|
| Context silently clipped | `EpisodicMetaDataset.sample_episode` and `evaluate_pair` both defaulted to a 256-row context cap, contradicting the explicit "use full training set as context" requirement | Read the deployed code directly, found `max_context: int = 256` / `context_size: int = 384/512` defaults still present in multiple places despite earlier fixes | Changed all defaults to `None` (unclipped); verified by re-reading the deployed code line-by-line after the fix, not just trusting the diff |
| Disk-quota checkpoint-save crash | 3/4 resubmitted LODO jobs failed with `RuntimeError: File ... tabicl_lora_best.pt cannot be opened` during `torch.save` | Same failure signature as an earlier "Disk quota exceeded" error from the data-fetching phase; `df -h` showed 49TB free (not a real space issue), pointing to a per-user quota not exposed by standard tooling | Deleted ~2.2GB of confirmed-stale files (old single-domain LODO checkpoints, abandoned hyperparameter-search checkpoints) and resubmitted |

## General lesson for whoever continues this work

Several of the worst bugs here (the fold-encoding mismatch, the context-clipping default, the
missing `evaluator.py` sync, the bf16→NaN masking) were only caught by **directly inspecting
values and re-reading the actually-deployed code**, not by trusting logs, prior fixes, or
"should be fine" reasoning. A job reporting success, or a metric being present in a JSON file,
was repeatedly not sufficient evidence that the underlying computation was correct. When
extending this work, re-verify assumptions against real output before trusting a result,
especially after any environment or dependency change (several of these bugs were introduced by
seemingly-unrelated version pins needed for a completely different encoder).
