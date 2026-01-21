#!/bin/bash
#SBATCH --job-name=bp_p2_00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:01:00

set -euo pipefail
mkdir -p logs

module purge
module use /projects/community/modulefiles
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate binpat

export HF_HOME=/scratch/$USER/hf
export HF_HUB_CACHE=/scratch/$USER/hf/hub
export TORCH_HOME=/scratch/$USER/torch

python /scratch/dsk129/vik/binpat_dev/binpat_esm2_analysis/scripts/phase2/00_preflight.py \
	--require-gpu \
	--print-env

