#!/bin/bash
# =============================================================================
# Vast.ai Setup Script for Causal Biosignal Foundation Model
# Target: RTX 5090 32GB GPU on Vast.ai
# =============================================================================
set -e

echo "============================================"
echo "  Causal Biosignal FM — Vast.ai Setup"
echo "============================================"

# --- System packages ---
apt-get update && apt-get install -y \
    git wget curl unzip htop tmux \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Create workspace structure ---
mkdir -p /workspace/{data,outputs,checkpoints}
mkdir -p /workspace/data/{sleep_edf,chbmit,ptbxl}

# --- Clone project (update with your repo URL) ---
cd /workspace
if [ ! -d "causal-biosignal-fm" ]; then
    echo "Clone your project repo here:"
    echo "  git clone https://github.com/YOUR_USERNAME/causal-biosignal-fm.git"
    echo "Or upload files via scp/rsync."
fi

# --- Python environment ---
echo "Setting up Python environment..."

# Vast.ai images typically have conda pre-installed
# If not, use system Python with venv
if command -v conda &> /dev/null; then
    conda create -n biosignal python=3.11 -y
    conda activate biosignal
else
    python3 -m venv /workspace/venv
    source /workspace/venv/bin/activate
fi

# --- Install PyTorch (CUDA 12.8 — supports 3090/4090/5090/A100) ---
pip install --upgrade pip
pip install torch>=2.7.0 torchaudio>=2.7.0 --index-url https://download.pytorch.org/whl/cu128

# --- Install project dependencies ---
cd /workspace/causal-biosignal-fm
pip install -r requirements.txt

# --- Install flash-attention v2 (optional, for perf) ---
echo "Installing flash-attention..."
pip install flash-attn --no-build-isolation 2>/dev/null || \
    echo "flash-attn install failed — will use standard attention. This is fine for initial development."

# --- Verify GPU ---
echo ""
echo "============================================"
echo "  GPU Verification"
echo "============================================"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:             {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory:      {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    print(f'CUDA version:    {torch.version.cuda}')
    print(f'cuDNN version:   {torch.backends.cudnn.version()}')
    # Quick bf16 test
    x = torch.randn(2, 2, device='cuda', dtype=torch.bfloat16)
    print(f'BF16 support:    OK')
print()
print('All checks passed!')
"

# --- WandB login ---
echo ""
echo "============================================"
echo "  Weights & Biases Setup"
echo "============================================"
echo "Run: wandb login"
echo "Or set WANDB_API_KEY environment variable"

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Download datasets (run scripts/download_*.sh)"
echo "  2. Preprocess data (run scripts/preprocess.py)"
echo "  3. Train: python -m src.train"
echo ""
