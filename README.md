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

**Phase 1 setup only (CPU)**  
1. create and activate a conda environment (python≥3.9 recommended)
```bash
conda create -n binpat python=3.10 -y
conda activate binpat
```
2. clone repo and install dependencies  
```bash
git clone https://github.com/DylanKlein1929/binpat_esm2_analysis.git
cd binpat_esm2_analysis
pip install -e . --only-binary=:all:
```  

3. install external dependency  
```bash
brew install dssp (or mkdssp)
```  

*Phase 1 scripts ready to run*


**Phase 1 + Phase 2 (GPU recommended)**  
On a computing cluster example:  
1. load system modules  
```bash
module purge  
module use /projects/community/modulefiles  
module load miniconda  
module load python/3.8.2  
source "$(conda info --base)/etc/profile.d/conda.sh"
```

2. create conda environment  
```bash
conda create -n binpat python=3.10 -y  
conda activate binpat
```

3. clone the repo and install dependencies  
```bash
git clone https://DylanKlein1929/binpat_esm2_analysis.git
cd binpat_esm2_analysis
pip install -e ".[gpu]" --only-binary=:all:
```

On local NVIDIA GPU  
1. install PyTorch using official guide  

2. install binpat  
```bash
git clone https://github.com/DylanKlein1929/binpat_esm2_analysis.git
cd binpat_esm2_analysis
pip install -e ".[gpu]" --only-binary=:all:
```

### run gpu preflight script before running Phase 2
```bash
python scripts/phase2/00_preflight.py  
```
(this reports versions/availability for python, pytorch, cuda, gpu device)  
