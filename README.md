# Binary-Patterned Protein Design Pipeline
----------------------------------------
This is a modular pipeline for generating combinatorial libraries of sequence variants, predicting their structures
with ESMFold/ESM atlas, and assessing the results via structural metrics and clustering. The pipeline also includes
the ability to embed the sequences with the ESM2 family of protein language models and map PCA projections of those
embeddings back onto structure predictions for visualization.


Overview
--------
Phase 1:  
  01 -> generate combinatorial libraries  
  02 -> predict the structures  
  03 -> compute structural metric averages (helicity, rASA, B factor)  
  04 -> cluster the structure predictions (RMSD between atomic coordinates as similarity metric)  
  05 -> compute average structures (backbone atoms only) for each cluster  

Phase 2:  
  01 -> embed the sequences with user-selected ESM2 model  
  02 -> fit to PCA  
  03 -> generate PCA structures  


Installation
------------
Supported platforms:  
  Phase 1 -> any Linux/macOS with conda and mkdssp  
  Phase 2 -> Linux with NVIDIA GPU + CONDA drivers (HPC modules or local CUDA)  

1. clone repository  
`git clone https://github.com/DylanKlein1929/binpat_esm2_analysis.git
cd binpat_esm2_analysis`  

**Phase 1 setup only (CPU)**  
2. create and activate a conda environment (python≥3.9 recommended)
`conda create -n binpat python=3.9 -y
conda activate binpat`  

2. install python dependencies  
`pip install -e .`  

3. install external dependency  
`brew install dssp (or mkdssp)`  

------------ Phase 1 scripts ready to run -------------


*** Phase 2 setup included (GPU recommended)***

1) On a computing cluster:
### load system modules (example)
module purge  
module use /projects/community/modulefiles  
module load miniconda  
module load cuda/12.1  
source "$(conda info --base)/etc/profile.d/conda.sh"  

### create conda environment
conda create -n binpat python=3.9 -y  
conda activate binpat

### install pytorch
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121

### install binpat (in the root of the cloned repo)
pip install -e .  
pip install -e .[gpu]  


2) On local NVIDIA GPU
### install PyTorch using official guide

### install binpat
pip install -e .  
pip install -e .[gpu]  

### run gpu preflight script before running Phase 2
python scripts/phase2/00_preflight.py  
(this reports versions/availability for python, pytorch, cuda, gpu device)  
