#!/bin/bash
#SBATCH --job-name=binpat_pc_pdbs
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=binpat_pc_pdbs.%j.out
#SBATCH --error=binpat_pc_pdbs.%j.err
#SBATCH --requeue

OUTDIR="${OUTDIR:-results/run001}"
WHICH_PC="${WHICH_PC:-1}"
USE_SCORES="${USE_SCORES:-1}"   # 1/0
IDS_FILE="${IDS_FILE:-}"
LIMIT="${LIMIT:-}"
OVERWRITE="${OVERWRITE:-0}"

module purge
module use /projects/community/modulefiles
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate binpat-esm2

CMD=(python scripts/phase2/03_write_pc_structures.py
  --outdir "$OUTDIR"
  --which-pc "$WHICH_PC"
)

if [ "$USE_SCORES" -eq 1 ]; then
  CMD+=(--use-precomputed-scores)
fi
if [ -n "$IDS_FILE" ]; then
  CMD+=(--ids-file "$IDS_FILE")
fi
if [ -n "$LIMIT" ]; then
  CMD+=(--limit "$LIMIT")
fi
if [ "$OVERWRITE" -eq 1 ]; then
  CMD+=(--overwrite)
fi

echo "[RUN] ${CMD[@]}"
"${CMD[@]}"
