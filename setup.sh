#!/bin/bash
# SATTC — Quick environment setup
# Usage: bash setup.sh

set -e

echo ">>> Creating conda environment 'sattc' ..."
conda create -n sattc python=3.10 -y
conda activate sattc

echo ">>> Installing PyTorch (CUDA 11.8) ..."
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cu118

echo ">>> Installing project dependencies ..."
pip install -r requirements.txt

echo ">>> Done. Activate with: conda activate sattc"