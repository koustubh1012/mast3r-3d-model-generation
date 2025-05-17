#!/bin/bash

echo "Initializing Conda..."

# Initialize Conda for non-interactive shells
eval "$(~/miniconda3/bin/conda shell.bash hook)"  # or change to ~/anaconda3 if you're using Anaconda

echo "Building the Conda environment..."
conda create -n mast3r python=3.11 cmake=3.14.0 -y
conda activate mast3r

conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install -c conda-forge faiss-gpu -y

cd mast3r

pip install -r requirements.txt
pip install -r dust3r/requirements.txt
pip install -r dust3r/requirements_optional.txt

pip install cython
git clone https://github.com/jenicek/asmk
cd asmk/cython/
cythonize *.pyx
cd ..
pip install . # python3 setup.py build_ext --inplace
cd ..

cd dust3r/croco/models/curope/
python3 setup.py build_ext --inplace
cd ../../../../

echo "Build complete."