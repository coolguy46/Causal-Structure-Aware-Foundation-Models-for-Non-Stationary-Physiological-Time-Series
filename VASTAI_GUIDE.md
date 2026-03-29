# Vast.ai Setup & Training Guide

Step-by-step instructions to train this model on Vast.ai with an RTX 5090.

---

## 1. Create a Vast.ai Instance

1. Go to [cloud.vast.ai](https://cloud.vast.ai/) → **Search** for `RTX 5090`
2. Filter: **On-Demand** (stable) or **Interruptible** (~50% cheaper, needs checkpointing)
3. Select a machine with **32GB VRAM**, **35GB+ RAM**, **100GB+ disk**
4. Docker Image: `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel`
5. Click **Rent** → wait for instance to start
6. Connect via **SSH** (Vast.ai provides the SSH command in the dashboard)

## 2. Initial Instance Setup

```bash
# SSH into your instance (copy the command from Vast.ai dashboard)
# ssh -p PORT root@HOST -L 8080:localhost:8080

# Clone/upload your project
cd /workspace
git clone https://github.com/YOUR_USERNAME/causal-biosignal-fm.git
cd causal-biosignal-fm

# Run the setup script (installs everything)
bash scripts/setup_vastai.sh
```

The setup script installs:
- PyTorch 2.7+ with CUDA 12.8
- All project dependencies (mne, wfdb, h5py, hydra, wandb, etc.)
- Verifies GPU, fp16, and CUDA

## 3. Download & Preprocess Data

All three datasets are **open access** on PhysioNet — no credentials or applications needed!

### Sleep-EDF Expanded (sleep staging EEG)
```bash
bash scripts/download_sleep_edf.sh
```

### CHB-MIT (seizure detection EEG)
```bash
bash scripts/download_chbmit.sh
```

### PTB-XL (arrhythmia classification ECG)
```bash
bash scripts/download_ptbxl.sh
```

### Preprocess all datasets to HDF5
```bash
python scripts/preprocess.py --dataset all --data_root /workspace/data
```

### Quick test with synthetic data (no downloads needed)
```bash
# Generate fake data to verify the pipeline works end-to-end
python scripts/generate_synthetic.py --data_root /workspace/data
python -m src.train dataset=sleep_edf train.epochs=3 wandb.mode=disabled
```

## 4. Login to Weights & Biases

```bash
pip install wandb
wandb login
# Paste your API key from wandb.ai/authorize
```

## 5. Training

### Single training run
```bash
cd /workspace/causal-biosignal-fm

# Train on Sleep-EDF
python -m src.train dataset=sleep_edf

# Train on CHB-MIT
python -m src.train dataset=chbmit

# Train on PTB-XL
python -m src.train dataset=ptbxl
```

### Full experiment suite (3 seeds × 3 datasets)
```bash
# Use tmux so training survives terminal disconnect
tmux new -s train

# Multirun across seeds and datasets
python -m src.train --multirun seed=42,123,7 dataset=sleep_edf,chbmit,ptbxl
```

### Monitor training
```bash
# In another tmux pane:
watch -n 5 nvidia-smi  # GPU usage
# Or check W&B dashboard in your browser
```

### Resume from checkpoint (if pod restarts)
The checkpoints are saved to `/workspace/checkpoints/` (on your persistent volume), so you won't lose progress. Training automatically picks up the best checkpoint.

## 6. Run Baselines

```bash
# You'll need to add baseline training to your training script
# or create a simple wrapper. Example for each baseline:

# PatchTST
python -m src.train --multirun seed=42,123,7 dataset=sleep_edf,chbmit,ptbxl \
    model=patchtst  # (after adding baseline configs)

# Run the full ablation sweep
python -m src.sweep
```

## 7. Evaluate

```bash
# Transfer evaluation
python -m src.eval.transfer --checkpoint /workspace/checkpoints/sleep_edf_best.pt \
    --datasets chbmit ptbxl

# Synthetic causal graph validation
python -m scripts.validate_causal_graph --n_nodes 10 --n_samples 10000

# Theory verification (in Python)
python -c "
from src.theory import verify_bound_empirically
# ... load model and data ...
"
```

## 8. Important Vast.ai Tips

### Don't lose your work
- **Always use `/workspace/`** for data and checkpoints
- Vast.ai on-demand instances have persistent disk — your data survives reboots
- For **interruptible** instances, push checkpoints to cloud storage (W&B artifacts or `rsync` to a backup)
- Commit code changes to git regularly

### Save money
- **Destroy the instance** when not training (you pay per-second while it exists)
- Use **Interruptible** instances for ~50% savings on long runs (save checkpoints frequently!)
- Prices fluctuate — check the marketplace for the best current deals
- Per-second billing means no wasted time

### Performance tips
- RTX 5090 32GB can handle `batch_size=32` with this model easily
- Enable `torch.compile` (already in config: `hardware.compile: true`)
- fp16 mixed precision is enabled by default
- If you hit OOM, reduce batch_size or use `accumulate_grad_batches: 2`

### Debugging
```bash
# Quick sanity check (3 epochs, no W&B)
python -m src.train train.epochs=3 wandb.mode=disabled

# Check GPU utilization
nvidia-smi -l 1

# If CUDA OOM:
# Reduce batch size
python -m src.train train.batch_size=16

# Or enable gradient accumulation
python -m src.train train.batch_size=16 train.accumulate_grad_batches=4
```

## 9. Estimated Costs

| Instance Type | Rate | Est. Total Training Time | Est. Cost |
|---------------|------|--------------------------|-----------|
| RTX 5090 32GB (On-Demand) | ~$0.37/hr | ~295 GPU-hrs | ~$109 |
| RTX 5090 32GB (Interruptible) | ~$0.19/hr | ~295 GPU-hrs | ~$56 |
| A100 80GB SXM (On-Demand) | ~$0.77/hr | ~350 GPU-hrs | ~$270 |

**Recommendation**: Use Interruptible instances for baselines/ablations (which can be restarted easily), On-Demand for final runs.

## 10. File Structure on Vast.ai

```
/workspace/
├── causal-biosignal-fm/      # Your project code
│   ├── src/
│   ├── scripts/
│   ├── tests/
│   └── ...
├── data/
│   ├── sleep_edf/             # HDF5 preprocessed data
│   ├── chbmit/
│   └── ptbxl/
├── checkpoints/              # Model checkpoints
├── outputs/                  # Hydra output dirs
└── results/                  # Evaluation results
```
