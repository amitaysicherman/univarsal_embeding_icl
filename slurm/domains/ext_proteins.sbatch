#!/bin/bash
#SBATCH --job-name=ext_proteins
#SBATCH --output=slurm_logs/ext_proteins_%j.log
#SBATCH --partition=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --exclude=bruno[1-5]

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mol_tabicl
cd "${UNIVARSAL_EMBEDING_ROOT:?set this to your clone of this repo, e.g. export UNIVARSAL_EMBEDING_ROOT=$HOME/univarsal_embeding}"

echo "Python: $(python --version)"
echo "Torch: $(python -c \"import torch; print(torch.__version__)\")"
echo "CUDA available: $(python -c \"import torch; print(torch.cuda.is_available())\")"
echo ""

python scripts/extract_embeddings.py --domain proteins --device cuda --batch-size 16 --overwrite
echo "Done: proteins"
