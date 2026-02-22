#!/bin/bash
#SBATCH --job-name=bp_p2_01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=binpat_p2_embed.%x.%j.out
#SBATCH --error=binpat_p2_embed.%x.%j.err
#SBATCH --requeue
#SBATCH --constraint="ampere|volta|adalovelace"

# ---------------- USER VARS (override with --export=ALL,...) ----------------
OUTDIR="${OUTDIR:-$(pwd)}"
VARIANTS_FASTA="${VARIANTS_FASTA:-$OUTDIR/variants.fasta}"

# selection
MODE="${MODE:-cluster_sample_k}"              # all | ids_file | cluster_sample_k
CLUSTER_CSV="${CLUSTER_CSV:-$OUTDIR/cluster_assignments.csv}"
K_PER_CLUSTER="${K_PER_CLUSTER:-5}"
IDS_FILE="${IDS_FILE:-}"

# model
MODEL="${MODEL:-facebook/esm2_t33_650M_UR50D}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DTYPE="${DTYPE:-float16}"                    # float16 | float32

# ---------------- Environment ----------------
module purge
module use /projects/community/modulefiles
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate ESM2-ready conda env
conda activate binpat || { echo "[ERROR] conda env 'binpat' not found"; exit 2; }

python -V
nvidia-smi || true

# ---------------- Caches to node-local scratch ----------------
export MPLBACKEND=Agg
export TMPDIR="${SLURM_TMPDIR:-/tmp}"

export XDG_CACHE_HOME="$TMPDIR/xdg_${SLURM_JOB_ID:-$$}"
export TORCH_HOME="$TMPDIR/torch_${SLURM_JOB_ID:-$$}"
export HF_HOME="$TMPDIR/hf_${SLURM_JOB_ID:-$$}"
export HF_DATASETS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME"
# ---------------- Run ----------------

# Ensure editable install exists
python -c "import binpat; print('binpat import OK')"

GPU_FLAG="--device cuda"

SEL_FLAGS="--mode $MODE"
if [ "$MODE" = "cluster_sample_k" ]; then
  SEL_FLAGS="$SEL_FLAGS --cluster-assignments $CLUSTER_CSV --k-per-cluster $K_PER_CLUSTER"
fi
if [ "$MODE" = "ids_file" ]; then
  SEL_FLAGS="$SEL_FLAGS --ids-file $IDS_FILE"
fi

python /scratch/dsk129/vik/binpat_dev/binpat_esm2_analysis/scripts/phase2/01_embed_sequences.py \
  --outdir "$OUTDIR" \
  --variants-fasta "$VARIANTS_FASTA" \
  $SEL_FLAGS \
  --model "$MODEL" \
  --batch-size "$BATCH_SIZE" \
  $GPU_FLAG \
  --dtype "$DTYPE"

echo "[INFO] Done."