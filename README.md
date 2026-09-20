# Universal In-Context Learning Across Multimodal Foundation Embeddings via Dynamic Projections

Code for the paper *"Universal In-Context Learning Across Multimodal Foundation Embeddings via
Dynamic Projections"* (ICLR 2027, under double-blind review).

We adapt a tabular in-context-learning transformer (TabICL) via LoRA and a dynamically resampled
random orthogonal projection into a single, training-free classification head that works across
7 data modalities (molecules, proteins, vision, text, audio, graphs, time series), 31 foundation
encoders, and 89 real classification tasks — without any per-task or per-encoder gradient
updates. See the paper for the method, benchmark design, and results.

This repository is anonymized for double-blind review: no author names, institutions, or private
infrastructure identifiers appear anywhere in the code or configs.

## Repository layout

```
├── src/univarsal_embeding/  <- library code: data fetching, encoders, TabICL+LoRA model, eval
├── scripts/                 <- entry points: fetch data, extract embeddings, train, evaluate, baselines
├── data_manifest.csv         <- the 89-task benchmark inventory (domain, N, classes, source)
└── tests/                   <- smoke tests
```

## Zero-synthetic-data policy

No dummy, random, or fallback data/embeddings are ever substituted for real ones. If an encoder
fails to load or run, the pipeline raises and skips that (task, encoder) pair rather than faking
a result — see `get_encoder` in `src/univarsal_embeding/encoders/registry.py`.

## Installation

```bash
pip install -e .
pip install -r requirements.txt
```

## Quickstart

```bash
# Fetches a tiny smoke sample across all 7 modalities and runs the pipeline end to end
python scripts/demo_local.py

# Unit/smoke tests
pytest tests/
```

## Getting the full data

The complete embedding set behind every result in the paper (392 verified embedding files across
all 89 tasks, with labels, 5-fold splits, and metadata; ~2.6GB) **will be released as a CC-BY-4.0
archive on Zenodo upon paper acceptance** — withheld during review solely to avoid
deanonymization, not for any proprietary reason. A DOI will be added here once it's published.
Once available, unpack it so the layout matches what every script expects (`--data-root data`,
the default):

```bash
unzip universal_embeddings.zip -d data/
# → data/tasks/<domain>/<task_id>/{embeddings/<encoder>.npy, labels.npy, folds.npy,
#                                   metadata.json, split_indices.json}
```

Raw inputs (SMILES strings, protein sequences, images, audio, etc.) are not redistributed — they
come fresh from their original public sources (Therapeutics Data Commons, HuggingFace Datasets,
PyTorch Geometric's TUDataset, the UCR/UEA archive via `aeon`) via `scripts/fetch_data.py` and
friends, per each source's own license.

## Training and evaluating

**Baselines** (linear probe, $k$-NN, MLP, XGBoost, zero-shot TabICL):
```bash
python scripts/run_baselines.py --data-root data --models linear,knn,mlp,xgboost,tabicl_zeroshot \
    --device cuda --output-dir results/baselines
```

**Train the TabICL+LoRA adapter** (one meta-training run, one seed):
```bash
python scripts/train_tabicl_lora.py --data-root data --seed 42 --lora-r 8 --lora-alpha 32 \
    --target-dim 256 --meta-steps 8000 --output-dir checkpoints/universal_tabicl_seed42
```

**Evaluate across the 4 generalization regimes** (+ leave-domain-out via `--holdout-domain`):
```bash
python scripts/evaluate_universal.py --data-root data \
    --checkpoint checkpoints/universal_tabicl_seed42/tabicl_lora_best.pt \
    --seed 42 --output-dir results/universal_eval
```

## License

MIT. See `LICENSE`. The Zenodo data release (once published) will be CC-BY-4.0.

## A note on anonymity for reviewers

This repository is kept free of author names, institutional affiliation, and internal
infrastructure details for the duration of double-blind review. For the same reason, the full
embedding archive is not published on Zenodo yet. Both will be released under the authors' names
immediately upon acceptance.
