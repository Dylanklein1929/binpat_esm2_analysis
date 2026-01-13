#!/bin/bash
#SBATCH --job-name=binpat_pca_fit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=binpat_pca_fit.%j.out
#SBATCH --error=binpat_pca_fit.%j.err
#SBATCH --requeue
#SBATCH --constraint="ampere|volta|adalovelace"

# -------- user vars (override with --export=ALL,VAR=...) ----------
OUTDIR="${OUTDIR:-results/run001}"
EMBDIR="${EMBDIR:-$OUTDIR/phase2/embeddings}"
MODEL="${MODEL:-facebook/esm2_t33_650M_UR50D}"
NCOMP="${NCOMP:-10}"
BATCH_RES="${BATCH_RES:-200000}"
DTYPE="${DTYPE:-float32}"
MAX_RES_TOTAL="${MAX_RES_TOTAL:-}"   # e.g. 5000000 to cap
WRITE_SCORES="${WRITE_SCORES:-1}"     # 1/0
IDS_FILE="${IDS_FILE:-}"             # optional
LIMIT="${LIMIT:-}"                   # optional

module purge
module use /projects/community/modulefiles
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate binpat-esm2

python -V

# Caches to node-local
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export XDG_CACHE_HOME="$TMPDIR/xdg_${SLURM_JOB_ID:-$$}"
export TORCH_HOME="$TMPDIR/torch_${SLURM_JOB_ID:-$$}"
export HF_HOME="$TMPDIR/hf_${SLURM_JOB_ID:-$$}"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME"

CMD=(python scripts/phase2/02_fit_pca.py
  --outdir "$OUTDIR"
  --embeddings-dir "$EMBDIR"
  --model "$MODEL"
  --n-components "$NCOMP"
  --batch-residues "$BATCH_RES"
  --dtype "$DTYPE"
)

if [ -n "$MAX_RES_TOTAL" ]; then
  CMD+=(--max-residues-total "$MAX_RES_TOTAL")
fi
if [ -n "$IDS_FILE" ]; then
  CMD+=(--ids-file "$IDS_FILE")
fi
if [ -n "$LIMIT" ]; then
  CMD+=(--limit "$LIMIT")
fi
if [ "$WRITE_SCORES" -eq 1 ]; then
  CMD+=(--write-scores)
fi

echo "[RUN] ${CMD[@]}"
"${CMD[@]}"
