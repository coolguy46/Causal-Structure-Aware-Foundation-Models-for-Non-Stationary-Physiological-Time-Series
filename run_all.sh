#!/bin/bash
# =============================================================================
#  MASTER SCRIPT — Causal Biosignal Foundation Model
#  Single script to run after SSH into Vast.ai (RTX 5090 32GB)
#
#  Usage:
#    bash run_all.sh                     # full pipeline from scratch
#    bash run_all.sh --step 5            # resume from step 5
#    bash run_all.sh --synthetic         # use synthetic data (no downloads)
#    bash run_all.sh --step 5 --dry-run  # print commands without executing
#
#  Steps:
#    0  Environment setup (apt, pip, CUDA verification)
#    1  Install dependencies (pip)
#    2  Data acquisition (download or synthetic)
#    3  Preprocessing (raw → HDF5)
#    4  Sanity check (3-epoch smoke test)
#    5  W&B login
#    6  Train main model (causal × 3 datasets × 3 seeds = 9 runs)
#    7  Train baselines (5 baselines × 3 datasets × 3 seeds = 45 runs)
#    8  Ablation sweeps (loss components + λ sensitivity)
#    9  Transfer evaluations (6 cross-dataset pairs)
#   10  Synthetic causal graph validation
#   11  Run analysis notebook (all figures, tables, stats)
#
#  Estimated total GPU time: ~37 hrs on RTX 5090
#  Estimated cost: ~$14 interruptible / ~$27 on-demand
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/workspace/data"
CKPT_DIR="/workspace/checkpoints"
OUTPUT_DIR="/workspace/outputs"
RESULTS_DIR="/workspace/results"
SEEDS="42,123,7"
DATASETS="sleep_edf,chbmit,ptbxl"

# Parse arguments
START_STEP=0
USE_SYNTHETIC=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)   START_STEP="$2"; shift 2 ;;
        --step=*) START_STEP="${1#*=}"; shift ;;
        --synthetic) USE_SYNTHETIC=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
step_header() {
    echo ""
    echo "============================================================"
    echo "  STEP $1: $2"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo ""
}

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "[dry-run] $*"
    else
        "$@"
    fi
}

elapsed() {
    local start=$1
    local end
    end=$(date +%s)
    local diff=$((end - start))
    printf '%02dh:%02dm:%02ds' $((diff/3600)) $((diff%3600/60)) $((diff%60))
}

SCRIPT_START=$(date +%s)

echo "============================================================"
echo "  Causal Biosignal FM — Vast.ai Master Script"
echo "  Starting from step $START_STEP"
echo "  Synthetic data: $USE_SYNTHETIC"
echo "  Dry run: $DRY_RUN"
echo "  $(date)"
echo "============================================================"

# ============================= STEP 0 =====================================
if [ "$START_STEP" -le 0 ]; then
    step_header 0 "ENVIRONMENT SETUP"
    STEP_START=$(date +%s)

    # System packages
    run_cmd apt-get update
    run_cmd apt-get install -y git wget curl unzip htop tmux libhdf5-dev
    rm -rf /var/lib/apt/lists/* 2>/dev/null || true

    # Directories
    mkdir -p "$DATA_DIR"/{sleep_edf,chbmit,ptbxl}
    mkdir -p "$OUTPUT_DIR" "$CKPT_DIR" "$RESULTS_DIR"

    # Python packages
    run_cmd pip install --upgrade pip
    run_cmd pip install torch>=2.7.0 torchaudio>=2.7.0 --index-url https://download.pytorch.org/whl/cu128

    # Verify GPU
    python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU:   {gpu}')
print(f'VRAM:  {vram:.1f} GB')
print(f'CUDA:  {torch.version.cuda}')
print(f'cuDNN: {torch.backends.cudnn.version()}')
x = torch.randn(2, 2, device='cuda', dtype=torch.bfloat16)
print('BF16:  OK')
"
    echo "[STEP 0] Done ($(elapsed $STEP_START))"
fi

# ============================= STEP 1 =====================================
if [ "$START_STEP" -le 1 ]; then
    step_header 1 "INSTALL DEPENDENCIES"

    cd "$REPO_DIR"
    run_cmd pip install -r requirements.txt
    # torch_scatter/torch_sparse need torch already installed — install separately
    run_cmd pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-$(python3 -c "import torch; print(torch.__version__.split('+')[0])")+cu128.html
    run_cmd pip install jupyterlab nbconvert

    # Quick import check
    python3 -c "from src.train import build_model; print('Project imports: OK')"

    echo "[STEP 1] Done"
fi

# ============================= STEP 2 =====================================
if [ "$START_STEP" -le 2 ]; then
    step_header 2 "DATA ACQUISITION"
    cd "$REPO_DIR"

    if [ "$USE_SYNTHETIC" = true ]; then
        echo "Generating synthetic data (no downloads)..."
        run_cmd python scripts/generate_synthetic.py \
            --data_root "$DATA_DIR" --dataset all \
            --n_subjects 50 --n_windows 100
    else
        echo "Downloading public datasets (all open-access, no credentials)..."
        echo ""
        echo "--- Sleep-EDF Expanded (2-ch EEG, ~1 GB) ---"
        run_cmd bash scripts/download_sleep_edf.sh "$DATA_DIR/sleep_edf"

        echo "--- CHB-MIT Scalp EEG (23-ch EEG, ~3 GB) ---"
        run_cmd bash scripts/download_chbmit.sh "$DATA_DIR/chbmit"

        echo "--- PTB-XL (12-lead ECG, ~2 GB) ---"
        run_cmd bash scripts/download_ptbxl.sh "$DATA_DIR/ptbxl"
    fi
    echo "[STEP 2] Done"
fi

# ============================= STEP 3 =====================================
if [ "$START_STEP" -le 3 ]; then
    step_header 3 "PREPROCESSING"
    cd "$REPO_DIR"

    if [ "$USE_SYNTHETIC" = true ]; then
        echo "Synthetic data is already HDF5. Skipping."
    else
        run_cmd python scripts/preprocess.py --dataset all --data_root "$DATA_DIR"
    fi
    echo "[STEP 3] Done"
fi

# ============================= STEP 4 =====================================
if [ "$START_STEP" -le 4 ]; then
    step_header 4 "SANITY CHECK (3-epoch smoke test)"
    cd "$REPO_DIR"

    run_cmd python -m src.train \
        dataset=sleep_edf \
        train.epochs=3 \
        train.batch_size=32 \
        wandb.mode=disabled \
        paths.data_root="$DATA_DIR" \
        paths.checkpoint_dir="$CKPT_DIR" \
        paths.output_dir="$OUTPUT_DIR"

    echo ""
    echo "Pipeline runs end-to-end. Proceeding to full training."
    echo "[STEP 4] Done"
fi

# ============================= STEP 5 =====================================
if [ "$START_STEP" -le 5 ]; then
    step_header 5 "WEIGHTS & BIASES LOGIN"

    if [ -n "${WANDB_API_KEY:-}" ]; then
        run_cmd wandb login "$WANDB_API_KEY"
        echo "Logged in via WANDB_API_KEY env var."
    else
        echo "========================================"
        echo "  ACTION REQUIRED: Log in to W&B"
        echo "========================================"
        echo ""
        echo "  Option A: export WANDB_API_KEY=<your-key>"
        echo "  Option B: wandb login"
        echo "  Get key at: https://wandb.ai/authorize"
        echo ""
        wandb login || echo "[warn] W&B login skipped — training will still work with wandb.mode=disabled"
    fi
    echo "[STEP 5] Done"
fi

# ============================= STEP 6 =====================================
if [ "$START_STEP" -le 6 ]; then
    step_header 6 "TRAIN MAIN MODEL (causal × 3 datasets × 3 seeds)"
    STEP_START=$(date +%s)
    cd "$REPO_DIR"

    # Hydra multirun: 9 runs total
    # Checkpoints → /workspace/checkpoints/causal_{dataset}_seed{seed}/best_model.pt
    run_cmd python -m src.train --multirun \
        seed=$SEEDS \
        dataset=$DATASETS \
        model=default \
        paths.data_root="$DATA_DIR" \
        paths.checkpoint_dir="$CKPT_DIR" \
        paths.output_dir="$OUTPUT_DIR"

    echo "[STEP 6] Done ($(elapsed $STEP_START))"
fi

# ============================= STEP 7 =====================================
if [ "$START_STEP" -le 7 ]; then
    step_header 7 "TRAIN BASELINES (5 × 3 datasets × 3 seeds = 45 runs)"
    STEP_START=$(date +%s)
    cd "$REPO_DIR"

    # All baselines save to the SAME checkpoint_dir — the run_name inside
    # train.py already prefixes with model_class, so no collisions:
    #   /workspace/checkpoints/patchtst_sleep_edf_seed42/best_model.pt
    #   /workspace/checkpoints/vanilla_tf_sleep_edf_seed42/best_model.pt
    #   etc.
    BASELINES="patchtst vanilla_tf static_gnn corr_graph raw_waveform"

    for BL in $BASELINES; do
        echo ""
        echo ">>> Training baseline: $BL"
        echo ""
        run_cmd python -m src.train --multirun \
            seed=$SEEDS \
            dataset=$DATASETS \
            model=$BL \
            paths.data_root="$DATA_DIR" \
            paths.checkpoint_dir="$CKPT_DIR" \
            paths.output_dir="$OUTPUT_DIR" \
        || echo "[warn] $BL training had issues — continuing"
    done

    echo "[STEP 7] Done ($(elapsed $STEP_START))"
fi

# ============================= STEP 8 =====================================
if [ "$START_STEP" -le 8 ]; then
    step_header 8 "ABLATION SWEEPS"
    STEP_START=$(date +%s)
    cd "$REPO_DIR"

    echo ">>> Loss component ablation (7 configs × 3 seeds)..."
    run_cmd python -m src.sweep \
        --sweeps loss_ablation \
        --seeds 42 123 7 \
    || echo "[warn] loss ablation had issues"

    echo ""
    echo ">>> Lambda sensitivity sweep (4 values × 3 seeds)..."
    run_cmd python -m src.sweep \
        --sweeps lambda_sweep \
        --seeds 42 123 7 \
    || echo "[warn] lambda sweep had issues"

    echo ""
    echo ">>> Band ablation (8 configs × 3 seeds)..."
    run_cmd python -m src.sweep \
        --sweeps band_ablation \
        --seeds 42 123 7 \
    || echo "[warn] band ablation had issues"

    echo ""
    echo ">>> Graph sparsity sweep (4 configs × 3 seeds)..."
    run_cmd python -m src.sweep \
        --sweeps graph_sparsity \
        --seeds 42 123 7 \
    || echo "[warn] graph sparsity sweep had issues"

    echo "[STEP 8] Done ($(elapsed $STEP_START))"
fi

# ============================= STEP 9 =====================================
if [ "$START_STEP" -le 9 ]; then
    step_header 9 "TRANSFER EVALUATION (6 cross-dataset pairs)"
    cd "$REPO_DIR"

    for SOURCE in sleep_edf chbmit ptbxl; do
        # Build target list (all except source)
        TARGETS=""
        for T in sleep_edf chbmit ptbxl; do
            [ "$T" != "$SOURCE" ] && TARGETS="$TARGETS $T"
        done

        # Find checkpoint (primary path pattern from train.py)
        CKPT_PATH="$CKPT_DIR/causal_${SOURCE}_seed42/best_model.pt"
        if [ ! -f "$CKPT_PATH" ]; then
            CKPT_PATH=$(find "$CKPT_DIR" "$OUTPUT_DIR" \
                -path "*causal*${SOURCE}*seed*42*best_model.pt" 2>/dev/null \
                | head -1 || true)
        fi

        if [ -n "${CKPT_PATH:-}" ] && [ -f "$CKPT_PATH" ]; then
            echo ">>> Transfer: $SOURCE →$TARGETS"
            run_cmd python -m src.eval.transfer \
                --checkpoint "$CKPT_PATH" \
                --datasets $TARGETS
        else
            echo "[skip] No checkpoint for causal/$SOURCE/seed42"
        fi
    done

    echo "[STEP 9] Done"
fi

# ============================= STEP 10 ====================================
if [ "$START_STEP" -le 10 ]; then
    step_header 10 "SYNTHETIC CAUSAL GRAPH VALIDATION"
    cd "$REPO_DIR"

    run_cmd python -m scripts.validate_causal_graph \
        --n_nodes 10 \
        --n_samples 10000 \
        --n_epochs 200 \
        --output "$RESULTS_DIR/causal_validation.json"

    echo "[STEP 10] Done"
fi

# ============================= STEP 11 ====================================
if [ "$START_STEP" -le 11 ]; then
    step_header 11 "RUN ANALYSIS NOTEBOOK (figures, tables, stats)"
    cd "$REPO_DIR"

    mkdir -p figures results

    # Execute notebook non-interactively (1hr timeout per cell)
    run_cmd jupyter nbconvert --to notebook --execute analysis.ipynb \
        --output analysis_executed.ipynb \
        --ExecutePreprocessor.timeout=3600 \
    || echo "[warn] Notebook had errors — open in JupyterLab to debug interactively"

    echo ""
    echo "Outputs:"
    echo "  figures/*.pdf        — all paper figures"
    echo "  results/*.csv|json   — evaluation data / LaTeX tables"
    echo "  analysis_executed.ipynb — notebook with outputs"
    echo ""
    echo "[STEP 11] Done"
fi

# ===========================================================================
TOTAL_ELAPSED=$(elapsed $SCRIPT_START)
echo ""
echo "============================================================"
echo "  ALL STEPS COMPLETE  ($TOTAL_ELAPSED)"
echo "============================================================"
echo ""
echo "Output locations:"
echo "  Checkpoints:  $CKPT_DIR/"
echo "  Results:      $REPO_DIR/results/"
echo "  Figures:      $REPO_DIR/figures/"
echo "  Notebook:     $REPO_DIR/analysis_executed.ipynb"
echo ""
echo "To browse results interactively:"
echo "  jupyter lab --ip=0.0.0.0 --port=8888 --allow-root"
echo ""
echo "To resume from a specific step:"
echo "  bash run_all.sh --step N"
echo ""
echo "Quick reference:"
echo "  --step 4   resume from sanity check"
echo "  --step 6   resume from main model training"
echo "  --step 7   resume from baseline training"
echo "  --step 8   resume from ablation sweeps"
echo "  --step 11  re-run analysis notebook only"
echo ""
