# Universal In-Context Learning Across Multimodal Foundation Embeddings via Dynamic Projections

Code and data for the paper *"Universal In-Context Learning Across Multimodal Foundation
Embeddings via Dynamic Projections"* (ICLR 2027, under double-blind review).

We adapt a tabular in-context-learning transformer (TabICL) via LoRA and a dynamically resampled
random orthogonal projection into a single, training-free classification head that works across
7 data modalities (molecules, proteins, vision, text, audio, graphs, time series), 31 foundation
encoders, and 89 real classification tasks — without any per-task or per-encoder gradient
updates.

This repository is anonymized for double-blind review: no author names, institutions, or private
infrastructure identifiers appear anywhere in the code or configs.

## Repository layout

```
├── src/univarsal_embeding/  <- library code: data fetching, encoders, TabICL+LoRA model, eval
├── scripts/                 <- entry points: fetch data, extract embeddings, train, evaluate, baselines
├── slurm/                   <- SLURM job templates used for every run reported in the paper
├── results/                 <- raw result files the paper's tables/figures are computed from
├── data/demo/                <- small (12MB, 14-task) sample so the pipeline runs with no download
├── data_manifest.csv         <- full 89-task inventory (domain, N, classes, class balance, source)
└── tests/                   <- smoke tests
```

## Reproducing the paper's results

`results/` has the raw evaluation output every number in the paper is computed from:

| Paper table | Source |
|---|---|
| Table 1 (headline, by regime) & Table 2 (by domain) | `results/seed_evals/seed_{42,123,456,789}.json` |
| Table 3 (leave-domain(s)-out) | `results/lodo_evals/lodo_{audio_graphs, molecules_proteins, text_timeseries, vision}.json` |
| Table 4 (ablations: ensemble size, zero-shot-matched, full-context-vs-clipped) | `results/ablations/*.json` |
| Baselines (linear probe, $k$-NN, MLP, XGBoost, zero-shot TabICL) | `results/baselines/`, `results/baselines_gpu_tabicl/` |

`scripts/summarize_results.py` and `scripts/aggregate_5fold_results.py` aggregate these raw files
into the summary numbers reported in the paper.

## Zero-synthetic-data policy

No dummy, random, or fallback data/embeddings are ever substituted for real ones. If an encoder
fails to load or run, the pipeline raises and skips that (task, encoder) pair rather than faking
a result — see `get_encoder` in `src/univarsal_embeding/encoders/registry.py`.

## Quickstart (no download required)

```bash
pip install -e .
pip install -r requirements.txt

# End-to-end sanity check on the bundled 14-task demo sample (data/demo/)
python scripts/demo_local.py

# Unit/smoke tests
pytest tests/
```

## Full reproduction: getting the embeddings

The complete embedding set — **392 verified embedding files across all 89 tasks** (labels,
5-fold splits, and metadata included; everything downstream of raw-input encoding needs, ~2.6GB)
— **will be released as a CC-BY-4.0 archive on Zenodo upon paper acceptance.** It is withheld
during double-blind review solely to avoid deanonymization (see note below), not for any
proprietary reason; a DOI will be added here as soon as it's published.

In the meantime, this repository includes a **14-task sample spanning all 7 modalities**
(`data/demo/`, 12MB) so every command below can be exercised end-to-end without any download —
just point `--data-root` at it. Once the Zenodo archive is released, unpack it so the layout
matches what every script expects (`--data-root data`, the default):

```bash
# after the Zenodo archive is available:
unzip universal_embeddings.zip -d data/
# → data/tasks/<domain>/<task_id>/{embeddings/<encoder>.npy, labels.npy, folds.npy,
#                                   metadata.json, split_indices.json}
```

Raw inputs (SMILES strings, protein sequences, images, audio, etc.) are **not** redistributed —
they're pulled fresh from their original public sources (Therapeutics Data Commons, HuggingFace
Datasets, PyTorch Geometric's TUDataset, the UCR/UEA archive via `aeon`) by `scripts/fetch_data.py`
and friends, per each source's own license. Model checkpoints are likewise not redistributed;
train your own with the commands below (a full run takes a few hours on a single 40GB+ GPU).

## Training and evaluating

All scripts default to `--data-root data`; swap in `data/demo` for a quick dry run.

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

`slurm/` has the exact job templates used for every run in the paper (extraction in
`slurm/domains/`, training/eval in `slurm/train/`, baselines in `slurm/baselines/`) — adapt the
`#SBATCH` directives to your own cluster and set `UNIVARSAL_EMBEDING_ROOT` to your clone's path
before submitting.

## License

Code: MIT. The released embeddings/results archive: CC-BY-4.0. See `LICENSE`.

## A note on anonymity for reviewers

This repository is kept free of author names, institutional affiliation, and internal
infrastructure details (cluster hostnames, usernames) for the duration of double-blind review.
For the same reason, the full embedding archive is **not** published on Zenodo yet — publishing
it now would attach an identifiable account/DOI record before review ends. The `data/demo/`
sample above is provided instead so reviewers can still run and inspect the full pipeline. The
complete archive and this code repository will both be released under the authors' names
immediately upon acceptance.
