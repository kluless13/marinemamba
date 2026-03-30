#!/bin/bash
set -e
echo "============================================================"
echo "STEP 1: ENVIRONMENT SETUP"
echo "============================================================"

# Upgrade setuptools (vtx needs license-files support)
pip install --upgrade setuptools 2>&1 | tail -1

# Install in correct order — DO NOT CHANGE
echo "Installing flash-attn..."
pip install flash-attn --no-build-isolation 2>&1 | tail -1
echo "Installing vtx..."
pip install vtx --no-build-isolation 2>&1 | tail -1
echo "Installing evo2..."
pip install evo2 --no-build-isolation 2>&1 | tail -1
echo "Installing mamba-ssm..."
pip install mamba-ssm --no-build-isolation 2>&1 | tail -1
echo "Installing causal-conv1d..."
pip install causal-conv1d --no-build-isolation 2>&1 | tail -1
echo "Installing other deps..."
pip install pytorch_lightning einops timm torchtext transformers huggingface_hub tqdm scikit-learn 2>&1 | tail -1

# Install BLAST
apt-get update -qq && apt-get install -qq -y ncbi-blast+ > /dev/null 2>&1 && echo "BLAST installed" || echo "BLAST not available (skipping)"

# Clone repo
cd /workspace
git clone https://github.com/kluless13/marinemamba.git 2>/dev/null || (cd marinemamba && git pull)
cd /workspace/marinemamba

# Verify everything
python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')
print(f'PyTorch: {torch.__version__}')
from mamba_ssm.modules.mamba2 import Mamba2
from causal_conv1d import causal_conv1d_fn
from evo2 import Evo2
print('mamba-ssm: OK')
print('causal-conv1d: OK')
print('evo2: OK')
"

echo ""
echo "============================================================"
echo "SETUP COMPLETE — run 02_fetch_data.sh next"
echo "============================================================"
