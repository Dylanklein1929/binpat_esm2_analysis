#!/bin/bash
#SBATCH --job-name=bp_p2_pcstruct
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

# --- Modules / environment ---
module purge
module use /projects/community/modulefiles
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate binpat

# no downloads but keep for consistency
export HF_HOME=/scratch/$USER/hf
export HF_HUB_CACHE=/scratch/$USER/hf/hub
export TORCH_HOME=/scratch/$USER/torch

# --- User-set parameters ---
OUTDIR="$(pwd)"
WHICH_PC="1"                  # 1=PC1, 2=PC2, ...
DTYPE="float32"               # float32 is safest
USE_PRECOMPUTED="1"           # 1 to pass --use-precomputed-scores
OVERWRITE="0"                 # 1 to pass --overwrite
IDS_FILE=""                   # optional: path to ids file; leave blank to infer from NPZ dir
LIMIT=""                      # optional: integer

# --- Derived paths (change if necessary) ---
PCA_MODEL="$OUTDIR/phase2/pca/pca_model.joblib"
PDB_DIR="$OUTDIR/pdbs"
EMB_DIR="$OUTDIR/phase2/embeddings"
SCORES_DIR="$OUTDIR/phase2/pca_scores"

echo "[slurm] job: ${SLURM_JOB_ID:-NA}  host: $(hostname)"
echo "[slurm] python: $(which python)"
python -V

echo "[inputs] OUTDIR      = $OUTDIR"
echo "[inputs] PCA_MODEL   = $PCA_MODEL"
echo "[inputs] PDB_DIR     = $PDB_DIR"
echo "[inputs] EMB_DIR     = $EMB_DIR"
echo "[inputs] SCORES_DIR  = $SCORES_DIR"
echo "[params] WHICH_PC    = $WHICH_PC"
echo "[params] DTYPE       = $DTYPE"
echo "[params] USE_PRECOMP = $USE_PRECOMPUTED"
echo "[params] OVERWRITE   = $OVERWRITE"
echo "[params] IDS_FILE    = ${IDS_FILE:-<none>}"
echo "[params] LIMIT       = ${LIMIT:-<none>}"

# --- Hard checks that should fail fast ---
if [[ ! -d "$OUTDIR" ]]; then
  echo "ERROR: OUTDIR not found: $OUTDIR" >&2
  exit 2
fi

if [[ ! -f "$PCA_MODEL" ]]; then
  echo "ERROR: PCA model not found: $PCA_MODEL" >&2
  echo "       Did you run phase2 step02 (02_fit_pca.py)?" >&2
  exit 3
fi

if [[ ! -d "$PDB_DIR" ]]; then
  echo "ERROR: PDB dir not found: $PDB_DIR" >&2
  echo "       Step03 expects <outdir>/pdbs/<variant_id>.pdb by default." >&2
  exit 4
fi

# Decide where IDs will be inferred from if IDS_FILE not given
DEFAULT_NPZ_DIR="$EMB_DIR"
EXTRA_FLAGS=()

if [[ "$USE_PRECOMPUTED" == "1" ]]; then
  EXTRA_FLAGS+=(--use-precomputed-scores)
  if [[ -d "$SCORES_DIR" ]]; then
    DEFAULT_NPZ_DIR="$SCORES_DIR"
  else
    echo "WARNING: SCORES_DIR not found ($SCORES_DIR). Will infer IDs from EMB_DIR and project on the fly." >&2
  fi
fi

if [[ ! -d "$DEFAULT_NPZ_DIR" ]]; then
  echo "ERROR: No NPZ directory found to infer IDs from: $DEFAULT_NPZ_DIR" >&2
  echo "       Expected embeddings at $EMB_DIR or scores at $SCORES_DIR." >&2
  exit 5
fi

# Optional flags
if [[ -n "$IDS_FILE" ]]; then
  if [[ ! -f "$IDS_FILE" ]]; then
    echo "ERROR: IDS_FILE not found: $IDS_FILE" >&2
    exit 6
  fi
  EXTRA_FLAGS+=(--ids-file "$IDS_FILE")
fi

if [[ -n "$LIMIT" ]]; then
  EXTRA_FLAGS+=(--limit "$LIMIT")
fi

if [[ "$OVERWRITE" == "1" ]]; then
  EXTRA_FLAGS+=(--overwrite)
fi

echo "[sanity] counting files..."
echo "[sanity] n_pdb   = $(ls -1 "$PDB_DIR"/*.pdb 2>/dev/null | wc -l || true)"
echo "[sanity] n_emb   = $(ls -1 "$EMB_DIR"/*.npz 2>/dev/null | wc -l || true)"
echo "[sanity] n_score = $(ls -1 "$SCORES_DIR"/*.npz 2>/dev/null | wc -l || true)"

# --- Run ---
set -x
python /scratch/dsk129/vik/binpat_dev/binpat_esm2_analysis/scripts/phase2/03_write_pc_structures.py \
  --outdir "$OUTDIR" \
  --which-pc "$WHICH_PC" \
  --dtype "$DTYPE" \
  "${EXTRA_FLAGS[@]}"
set +x

echo "[done] wrote: $OUTDIR/pdbs_pc/pc${WHICH_PC}"
echo "[done] report: $OUTDIR/pdbs_pc/pc${WHICH_PC}/pc_write_report.csv"