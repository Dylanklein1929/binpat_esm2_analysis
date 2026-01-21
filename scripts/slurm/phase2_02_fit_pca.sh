#!/bin/bash
#SBATCH --job-name=bp_p2_pca
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

module purge
module use /projects/community/modulefiles
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate binpat

OUTDIR="/scratch/dsk129/vik/binpat_dev/phase1_test"
EMBDIR="$OUTDIR/phase2/embeddings"

python /scratch/dsk129/vik/binpat_dev/binpat_esm2_analysis/scripts/phase2/02_fit_pca.py \
  --outdir "$OUTDIR" \
  --embeddings-dir "$EMBDIR" \
  --model facebook/esm2_t33_650M_UR50D \
  --n-components 5 \
  --write-scores

