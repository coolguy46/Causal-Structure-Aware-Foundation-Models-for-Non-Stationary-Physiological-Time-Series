# Causal Structure-Aware Foundation Model for Physiological Time Series

A foundation model that learns causal graph structure over physiological signals (EEG, ECG), enabling zero-shot generalization across subjects, devices, and conditions.

**Paper**: *Causal Structure-Aware Foundation Models for Non-Stationary Physiological Time Series*

## Architecture

```
Raw Signal (B, C, T)
    │
    ▼
┌─────────────────────┐
│  Spectral Tokenizer  │  Per-band STFT → frequency tokens
│  (delta/theta/alpha/  │  + channel ID + band ID embeddings
│   beta/gamma)         │
└─────────┬───────────┘
          │ tokens (B, C*n_bands, d_token)
          ▼
┌─────────────────────┐
│  Causal Graph        │  Pairwise edge scoring + straight-through
│  Inferencer          │  → sparse directed adjacency matrix
└─────────┬───────────┘
          │ adj (B, N, N)
          ▼
┌─────────────────────┐
│  Graph-Conditioned   │  Attention masked by causal graph
│  Transformer         │  (key theoretical contribution)
└─────────┬───────────┘
          │ embeddings (B, N, d)
          ▼
┌─────────────────────┐
│  Classification Head  │  Global pool → MLP → task logits
│  + Subject Adapter    │  (optional lightweight per-subject adapter)
└─────────────────────┘
```

## Quick Start on Vast.ai

### 1. Launch an Instance
- **GPU**: RTX 5090 32GB (or A100 80GB)
- **Image**: `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel`
- **Disk**: 100GB+

### 2. Setup
```bash
# Upload project files to /workspace/causal-biosignal-fm/
cd /workspace/causal-biosignal-fm

# Run setup script
bash scripts/setup_vastai.sh
```

### 3. Download Data
```bash
# All three datasets are open access — no credentials needed!
bash scripts/download_sleep_edf.sh
bash scripts/download_chbmit.sh
bash scripts/download_ptbxl.sh
```

### 4. Preprocess
```bash
python scripts/preprocess.py --dataset all
```

### 5. Train
```bash
# Default config (Sleep-EDF, sleep staging)
python -m src.train

# Override config via Hydra
python -m src.train dataset=chbmit train.batch_size=16 train.lr=1e-4

# Disable wandb for debugging
python -m src.train wandb.mode=disabled
```

### 6. Evaluate
```bash
# Zero-shot transfer
python -m src.eval.transfer --checkpoint /workspace/checkpoints/best_model.pt --datasets sleep_edf chbmit ptbxl
```

### 7. Run Ablations
```bash
# Loss term ablation (7 configs × 3 seeds)
python -m src.sweep --sweeps loss_ablation --seeds 42 123 456

# Lambda sensitivity
python -m src.sweep --sweeps lambda_sweep

# Token dimension sweep
python -m src.sweep --sweeps token_dim
```

## Project Structure

```
src/
├── config/                 # Hydra YAML configs
│   ├── base.yaml          # Main config with defaults
│   ├── dataset/           # Sleep-EDF, CHB-MIT, PTB-XL
│   ├── model/             # Architecture hyperparameters
│   └── loss/              # Loss weights
├── data/                  # Data loading & preprocessing
│   ├── eeg_dataset.py     # HDF5 lazy-loading EEG dataset
│   ├── ecg_dataset.py     # HDF5 lazy-loading ECG dataset
│   ├── transforms.py      # Z-score, bandpass, artifact rejection
│   └── splits.py          # Subject-stratified splitting
├── model/                 # Architecture
│   ├── tokenizer.py       # SpectralTokenizer + TokenDecoder
│   ├── causal_graph.py    # CausalGraphInferencer (NOTEARS + STE)
│   ├── transformer.py     # GraphConditionedTransformer
│   ├── adapter.py         # SubjectAdapter (~16k params)
│   └── full_model.py      # CausalBiosignalModel (end-to-end)
├── loss/                  # Loss functions
│   ├── spectral_loss.py   # MSE in STFT domain
│   ├── causal_loss.py     # do-calculus consistency
│   └── task_loss.py       # Joint objective
├── eval/                  # Evaluation
│   ├── benchmark.py       # F1, AUROC, ECE
│   ├── transfer.py        # Zero-shot cross-dataset
│   └── interpret.py       # Graph visualization
├── train.py               # Main training loop
└── sweep.py               # Ablation sweep runner
scripts/
├── setup_vastai.sh        # Vast.ai environment setup
├── download_sleep_edf.sh  # Sleep-EDF download
├── download_chbmit.sh     # CHB-MIT download
├── download_ptbxl.sh      # PTB-XL download
└── preprocess.py          # Raw → HDF5 preprocessing
tests/                     # pytest test suite
```

## Key Datasets

| Dataset | Signal | Channels | Task | Access |
|---------|--------|----------|------|--------|
| [Sleep-EDF](https://physionet.org/content/sleep-edfx/) | EEG | 2 | Sleep staging (5-class) | Open (instant) |
| [CHB-MIT](https://physionet.org/content/chbmit/) | EEG | 23 | Seizure detection (binary) | Open (instant) |
| [PTB-XL](https://physionet.org/content/ptb-xl/) | ECG | 12 | Arrhythmia (5-class) | Open (instant) |

## Tests
```bash
pytest tests/ -v
```

## Requirements
- Python 3.11+
- PyTorch 2.2+ with CUDA 12
- RTX 5090 32GB GPU (recommended) or A100 80GB
- ~50GB storage for raw data + preprocessed cache
