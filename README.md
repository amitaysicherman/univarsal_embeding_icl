# Universal In-Context Learning Across Multimodal Foundation Embeddings via Dynamic Projections

Code and data for the paper *"Universal In-Context Learning Across Multimodal Foundation
Embeddings via Dynamic Projections"* (under double-blind review).

We adapt a tabular in-context-learning transformer (TabICL) via LoRA and a dynamically resampled
random orthogonal projection into a single, training-free classification head that works across
7 data modalities (molecules, proteins, vision, text, audio, graphs, time series), 31 foundation
encoders, and 89 real classification tasks — without any per-task or per-encoder gradient
updates. The paper itself (LaTeX source and PDF) is submitted and managed separately from this
code repository — this repo covers the code, methodology docs, and results only.

This repository is anonymized for double-blind review: no author names, institutions, or private
infrastructure identifiers appear anywhere in the code, docs, or configs.

## Repository layout

```
├── src/univarsal_embeding/  <- library code: data fetching, encoders, TabICL+LoRA model, eval
├── scripts/                 <- entry points: fetch data, extract embeddings, train, evaluate, baselines
├── slurm/                   <- SLURM job templates (extraction, training, baselines)
├── docs/                    <- methodology write-up, read in order (see below)
├── results/                 <- all result files underlying every table/figure in the paper
├── data/demo/                <- small (12MB, 14-task) sample so the pipeline runs with no download
└── tests/                   <- smoke tests
```

## Methodology docs

`docs/` documents the full pipeline end to end, one short self-contained file per stage — read
in order:

1. [`01_data_collection.md`](docs/01_data_collection.md) — where the 89 tasks came from and how
   authenticity was verified
2. [`02_embedding_extraction.md`](docs/02_embedding_extraction.md) — the encoder registry and
   extraction/validation pipeline
3. [`03_baselines.md`](docs/03_baselines.md) — the 5 baseline models and metrics
4. [`04_model_architecture.md`](docs/04_model_architecture.md) — TabICL + LoRA + dynamic random
   projection
5. [`05_training_methodology.md`](docs/05_training_methodology.md) — the meta-training loop and
   hyperparameter search
6. [`06_evaluation_methodology.md`](docs/06_evaluation_methodology.md) — the 4 generalization
   regimes, splits, and leave-domain(s)-out design
7. [`07_results_and_ablations.md`](docs/07_results_and_ablations.md) — headline numbers, with
   pointers to the exact result files
8. [`08_known_issues_and_fixes.md`](docs/08_known_issues_and_fixes.md) — every bug found and
   fixed during the project, and how

[`docs/data_manifest.csv`](docs/data_manifest.csv) has the full 89-task inventory (domain, task,
N, classes, class balance, source, description).

## Zero-synthetic-data policy

No dummy, random, or fallback data/embeddings are ever substituted for real ones. If an encoder
fails to load or run, the pipeline raises and skips that (task, encoder) pair rather than faking
a result (`src/univarsal_embeding/encoders/registry.py`, `get_encoder`). See
`docs/02_embedding_extraction.md` and `docs/01_data_collection.md` for how this is enforced and
verified.

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
(`data/demo/`, 12MB) so every script below can be exercised end-to-end without any download —
just swap `--data-root data/demo` for `--data-root data` once the full archive is unpacked. Once
released, unpack the Zenodo archive so the layout matches what every script expects
(`--data-root data`, default):

```bash
# after the Zenodo archive is available:
unzip universal_embeddings.zip -d data/
# → data/tasks/<domain>/<task_id>/{embeddings/<encoder>.npy, labels.npy, folds.npy,
#                                   metadata.json, split_indices.json}
```

Raw inputs (SMILES strings, protein sequences, images, audio, etc.) are **not** redistributed —
they're pulled fresh from their original public sources (Therapeutics Data Commons, HuggingFace
Datasets, PyTorch Geometric's TUDataset, the UCR/UEA archive via `aeon`) by `scripts/fetch_data.py`
and friends, per each source's own license. Model checkpoints are likewise not redistributed here;
train your own with the commands below (a full run takes a few hours on a single 40GB+ GPU).

## Running the experiments

All scripts default to `--data-root data`, so once the Zenodo archive is unpacked (or against
`data/demo/` for a quick dry run) they work unmodified.

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

## Results

`results/` contains every result file the paper's tables and figures are built from: per-seed
evaluations (`seed_evals/`), leave-domain(s)-out runs (`lodo_evals/`), ensemble-size and
zero-shot-isolation ablations (`ablations/`), and all 5 baselines per domain (`baselines/`,
`baselines_gpu_tabicl/`).

## License

Code: MIT. The released embeddings/results archive: CC-BY-4.0. See `LICENSE`.

## A note on anonymity for reviewers

This repository and the paper's Reproducibility Statement are kept free of author names,
institutional affiliation, and internal infrastructure details (cluster hostnames, usernames) for
the duration of double-blind review. For the same reason, the full embedding archive is **not**
published on Zenodo yet — publishing it now would attach an identifiable account/DOI record
before review ends. The `data/demo/` sample above is provided instead so reviewers can still run
and inspect the full pipeline. The complete archive, code repository, and paper will all be
released under the authors' names immediately upon acceptance.
