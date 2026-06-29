#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "========================================="
echo "Setting up environment for CRISPR-BERT..."
echo "========================================="

# Miniconda directory
MINICONDA_DIR="$HOME/miniconda"
export PATH="$MINICONDA_DIR/bin:$PATH"
eval "$($MINICONDA_DIR/bin/conda shell.bash hook)"

# Create environment if it does not exist
if conda info --envs | grep -q "crispr_bert"; then
    echo "Conda environment 'crispr_bert' already exists."
else
    echo "Creating Conda environment 'crispr_bert' with Python 3.9..."
    conda create -n crispr_bert python=3.9 -y
fi

# Activate and install dependencies
echo "Activating 'crispr_bert' environment..."
conda activate crispr_bert

echo "Installing requirements from /workspace/crispr-bert/requirements.txt..."
cd /workspace/crispr-bert
pip install --upgrade pip
pip install -r requirements.txt

echo "========================================="
echo "CRISPR-BERT VM Setup Successful!"
echo "========================================="
