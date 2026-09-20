#!/bin/bash
#SBATCH --job-name=lora_train
#SBATCH --output=slurm_logs/lora_%j.log
#SBATCH --partition=public
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mol_tabicl
cd "${UNIVARSAL_EMBEDING_ROOT:?set this to your clone of this repo, e.g. export UNIVARSAL_EMBEDING_ROOT=$HOME/univarsal_embeding}"
mkdir -p slurm_logs

python scripts/train_tabicl_lora.py "$@"
