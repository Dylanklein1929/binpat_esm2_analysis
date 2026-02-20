#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=bp_p1_test
#SBATCH --output=slurm.%x.%j.out
#SBATCH --error=slurm.%x.%j.err


module purge
module use /projects/community/modulefiles
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate binpat


# outdir root (edit these for your own setup)
out_dir="/scratch/dsk129/vik/binpat_dev/phase1_test"
binpat_dir="/scratch/dsk129/vik/binpat_dev/binpat_esm2_analysis"


# step 1
python ${binpat_dir}/scripts/phase1/01_generate_library.py \
	--input-fasta ${out_dir}/binpat_phase1_inputs_master.fasta \
	--config ${out_dir}/binpat_phase1_4hcc_config.yaml \
	--outdir ${out_dir} \

# step 2
python ${binpat_dir}/scripts/phase1/02_predict_structures_esmatlas.py \
	--variants-fasta ${out_dir}/binpat_phase1_4hcc_variants.fasta \
	--outdir ${out_dir} \

# step 3
python ${binpat_dir}/scripts/phase1/03_compute_structural_metrics.py \
	--outdir ${out_dir} \
	--variants-metadata ${out_dir}/binpat_phase1_4hcc_metadata.csv \
	--rasa-threshold 0.35 \
	--variants-fasta ${out_dir}/binpat_phase1_4hcc_variants.fasta \
	--skip-motif GGGGGG \ 

# step 4
python ${binpat_dir}/scripts/phase1/04_cluster_structures.py \
	--outdir ${out_dir} \
	--pdb-dir ${out_dir}/pdbs \
	--variants-metadata ${out_dir}/binpat_phase1_4hcc_metadata.csv \
	--linkage single \
	--dendrogram \
	--dendrogram-format png \


# step 5
python ${binpat_dir}/scripts/phase1/05_average_structures.py \
	--outdir ${out_dir} \
