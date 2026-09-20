#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --output=slurm_logs/baselines_%j.log
#SBATCH --partition=public
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mol_tabicl
cd "${UNIVARSAL_EMBEDING_ROOT:?set this to your clone of this repo, e.g. export UNIVARSAL_EMBEDING_ROOT=$HOME/univarsal_embeding}"
mkdir -p slurm_logs results/baselines

python scripts/run_baselines.py --models linear,knn,mlp,tabicl_zeroshot --device cpu --output-dir results/baselines
