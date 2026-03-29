#!/bin/bash
# =============================================================================
#  MASTER SCRIPT — Causal Biosignal Foundation Model
#  Run every step from setup to final analysis on Vast.ai (RTX 5090 32GB)
#
#  Usage:
#    bash run_all.sh              # run everything (interactive prompts)
#    bash run_all.sh --step 3     # jump to step 3
#    bash run_all.sh --synthetic  # use synthetic data (no dataset downloads)
# =============================================================================
set -e

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR="/workspace/causal-biosignal-fm"
DATA_DIR="/workspace/data"
CKPT_DIR="/workspace/checkpoints"
OUTPUT_DIR="/workspace/outputs"
RESULTS_DIR="/workspace/results"
SEEDS="42,123,7"

# Parse args
START_STEP=0
USE_SYNTHETIC=false
for arg in "$@"; do
    case $arg in
        --step)   shift; START_STEP=$1; shift ;;
        --step=*) START_STEP="${arg#*=}" ;;
        --synthetic) USE_SYNTHETIC=true ;;
    esac
done

step_header() {
    echo ""
    echo "============================================================"
    echo "  STEP $1: $2"
    echo "============================================================"
    echo ""
}

# ---------------------------------------------------------------------------
# STEP 0: Environment setup
# ---------------------------------------------------------------------------
if [ "$START_STEP" -le 0 ]; then
    step_header 0 "ENVIRONMENT SETUP"

    # System packages
    apt-get update && apt-get install -y \
        git wget curl unzip htop tmux libhdf5-dev \
        && rm -rf /var/lib/apt/lists/*

    # Directories
    mkdir -p "$DATA_DIR"/{sleep_edf,chbmit,ptbxl}
    mkdir -p "$OUTPUT_DIR" "$CKPT_DIR" "$RESULTS_DIR"

    # Python deps
    pip install --upgrade pip
    pip install torch>=2.7.0 torchaudio>=2.7.0 --index-url https://download.pytorch.org/whl/cu128
    cd "$PROJECT_DIR"
    pip install -r requirements.txt
    pip install jupyterlab
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "[warn] flash-attn skipped"

    # Verify GPU
    python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'GPU:  {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
x = torch.randn(2, 2, device='cuda', dtype=torch.bfloat16)
print('BF16: OK')
print('Environment ready.')
"
    echo "[STEP 0] DONE"
fi

# ---------------------------------------------------------------------------
# STEP 1: Data — download or generate synthetic
# ---------------------------------------------------------------------------
if [ "$START_STEP" -le 1 ]; then
    step_header 1 "DATA ACQUISITION"
    cd "$PROJECT_DIR"

    if [ "$USE_SYNTHETIC" = true ]; then
        echo ">>> Generating synthetic data (no downloads required)..."
        python scripts/generate_synthetic.py \
            --data_root "$DATA_DIR" --dataset all \
            --n_subjects 50 --n_windows 100
    else
        echo ">>> Downloading public datasets (no credentials needed)..."
        echo ""

        # Sleep-EDF (open access)
        echo "--- Downloading Sleep-EDF Expanded ---"
        bash scripts/download_sleep_edf.sh "$DATA_DIR/sleep_edf"

        # CHB-MIT (open access)
        echo "--- Downloading CHB-MIT ---"
        bash scripts/download_chbmit.sh "$DATA_DIR/chbmit"

        # PTB-XL (open access)
        echo "--- Downloading PTB-XL ---"
        bash scripts/download_ptbxl.sh "$DATA_DIR/ptbxl"
    fi
    echo "[STEP 1] DONE"
fi

# ---------------------------------------------------------------------------
# STEP 2: Preprocess raw data to HDF5
# ---------------------------------------------------------------------------
if [ "$START_STEP" -le 2 ]; then
    step_header 2 "PREPROCESSING"
    cd "$PROJECT_DIR"

    if [ "$USE_SYNTHETIC" = false ]; then
        python scripts/preprocess.py --dataset all --data_root "$DATA_DIR"
    else
        echo "Synthetic data is already in HDF5 format. Skipping."
    fi
    echo "[STEP 2] DONE"
fi

# ---------------------------------------------------------------------------
# STEP 3: Sanity check — quick 3-epoch run
# ---------------------------------------------------------------------------
if [ "$START_STEP" -le 3 ]; then
    step_header 3 "SANITY CHECK (3 epochs, no W&B)"
    cd "$PROJECT_DIR"

    python -m src.train \
        dataset=sleep_edf \
        train.epochs=3 \
        train.batch_size=32 \
        wandb.mode=disabled \
        paths.data_root="$DATA_DIR" \
        paths.checkpoint_dir="$CKPT_DIR" \
        paths.output_dir="$OUTPUT_DIR"

    echo "[STEP 3] DONE — pipeline works end-to-end"
fi

# ---------------------------------------------------------------------------
# STEP 4: W&B login
# ---------------------------------------------------------------------------
if [ "$START_STEP" -le 4 ]; then
    step_header 4 "WEIGHTS & BIASES LOGIN"

    if [ -n "$WANDB_API_KEY" ]; then
        wandb login "$WANDB_API_KEY"
    else
        echo "Run 'wandb login' and paste your API key from https://wandb.ai/authorize"
        echo "Or: export WANDB_API_KEY=your_key"
        wandb login || echo "[warn] wandb login skipped — set WANDB_API_KEY or run manually"
    fi
    echo "[STEP 4] DONE"
fi

# ---------------------------------------------------------------------------
# STEP 5: Train main model (3 seeds x 3 datasets = 9 runs)
# ---------------------------------------------------------------------------
if [ "$START_STEP" -le 5 ]; then
    step_header 5 "TRAIN MAIN MODEL (3 seeds × 3 datasets)"
    cd "$PROJECT_DIR"

    python -m src.train --multirun \
        seed=$SEEDS \
        dataset=sleep_edf,chbmit,ptbxl \
        paths.data_root="$DATA_DIR" \
        paths.checkpoint_dir="$CKPT_DIR" \
        paths.output_dir="$OUTPUT_DIR"

    echo "[STEP 5] DONE"
fi

# ---------------------------------------------------------------------------
# STEP 6: Train baselines (same seeds)
#   You will need baseline training configs or a wrapper.
#   Below are the commands assuming you've wired baseline model selection
#   into Hydra config. Alternatively, train each manually.
# ---------------------------------------------------------------------------
if [ "$START_STEP" -le 6 ]; then
    step_header 6 "TRAIN BASELINES"
    cd "$PROJECT_DIR"

    echo ">>> Training PatchTST baseline..."
    python -m src.train --multirun \
        seed=$SEEDS \
        dataset=sleep_edf,chbmit,ptbxl \
        model=patchtst \
        paths.data_root="$DATA_DIR" \
        paths.checkpoint_dir="$CKPT_DIR/patchtst" \
        paths.output_dir="$OUTPUT_DIR/patchtst" \
    || echo "[warn] patchtst training had issues"

    echo ">>> Training Vanilla Transformer baseline..."
    python -m src.train --multirun \
        seed=$SEEDS \
        dataset=sleep_edf,chbmit,ptbxl \
        model=vanilla_tf \
        paths.data_root="$DATA_DIR" \
        paths.checkpoint_dir="$CKPT_DIR/vanilla_tf" \
        paths.output_dir="$OUTPUT_DIR/vanilla_tf" \
    || echo "[warn] vanilla_tf training had issues"

    echo ">>> Training Static GNN baseline..."
    python -m src.train --multirun \
        seed=$SEEDS \
        dataset=sleep_edf,chbmit,ptbxl \
        model=static_gnn \
        paths.data_root="$DATA_DIR" \
        paths.checkpoint_dir="$CKPT_DIR/static_gnn" \
        paths.output_dir="$OUTPUT_DIR/static_gnn" \
    || echo "[warn] static_gnn training had issues"

    echo ">>> Training Correlation Graph baseline..."
    python -m src.train --multirun \
        seed=$SEEDS \
        dataset=sleep_edf,chbmit,ptbxl \
        model=corr_graph \
        paths.data_root="$DATA_DIR" \
        paths.checkpoint_dir="$CKPT_DIR/corr_graph" \
        paths.output_dir="$OUTPUT_DIR/corr_graph" \
    || echo "[warn] corr_graph training had issues"

    echo ">>> Training Raw Waveform baseline..."
    python -m src.train --multirun \
        seed=$SEEDS \
        dataset=sleep_edf,chbmit,ptbxl \
        model=raw_waveform \
        paths.data_root="$DATA_DIR" \
        paths.checkpoint_dir="$CKPT_DIR/raw_waveform" \
        paths.output_dir="$OUTPUT_DIR/raw_waveform" \
    || echo "[warn] raw_waveform training had issues"

    echo "[STEP 6] DONE"
fi

# ---------------------------------------------------------------------------
# STEP 7: Ablation sweep
# ---------------------------------------------------------------------------
if [ "$START_STEP" -le 7 ]; then
    step_header 7 "ABLATION SWEEP"
    cd "$PROJECT_DIR"

    echo ">>> Loss component ablation..."
    python -m src.sweep \
        --sweeps loss_ablation \
        --seeds 42 123 7

    echo ">>> Lambda sweep..."
    python -m src.sweep \
        --sweeps lambda_sweep \
        --seeds 42 123 7

    echo ">>> Token dimension sweep..."
    python -m src.sweep \
        --sweeps token_dim \
        --seeds 42 123 7

    echo ">>> Window size sweep..."
    python -m src.sweep \
        --sweeps window_size \
        --seeds 42 123 7

    echo "[STEP 7] DONE"
fi

# ---------------------------------------------------------------------------
# STEP 8: Transfer evaluations (6 source→target pairs)
# ---------------------------------------------------------------------------
if [ "$START_STEP" -le 8 ]; then
    step_header 8 "TRANSFER EVALUATION"
    cd "$PROJECT_DIR"

    for SOURCE in sleep_edf chbmit ptbxl; do
        # Build target list (all datasets except source)
        TARGETS=""
        for T in sleep_edf chbmit ptbxl; do
            if [ "$T" != "$SOURCE" ]; then
                TARGETS="$TARGETS $T"
            fi
        done

        CKPT_PATH="$CKPT_DIR/causal_${SOURCE}_seed42/best_model.pt"
        if [ ! -f "$CKPT_PATH" ]; then
            # Try alternative paths
            CKPT_PATH=$(find "$CKPT_DIR" "$OUTPUT_DIR" -path "*${SOURCE}*seed*42*best_model.pt" 2>/dev/null | head -1)
        fi

        if [ -n "$CKPT_PATH" ] && [ -f "$CKPT_PATH" ]; then
            echo ">>> Transfer: $SOURCE → $TARGETS"
            python -m src.eval.transfer \
                --checkpoint "$CKPT_PATH" \
                --datasets $TARGETS
        else
            echo "[skip] No checkpoint found for $SOURCE"
        fi
    done

    echo "[STEP 8] DONE"
fi

# ---------------------------------------------------------------------------
# STEP 9: Synthetic causal graph validation
# ---------------------------------------------------------------------------
if [ "$START_STEP" -le 9 ]; then
    step_header 9 "CAUSAL GRAPH VALIDATION"
    cd "$PROJECT_DIR"

    python -m scripts.validate_causal_graph \
        --n_nodes 10 \
        --n_samples 10000 \
        --n_epochs 200 \
        --output "$RESULTS_DIR/causal_validation.json"

    echo "[STEP 9] DONE"
fi

# ---------------------------------------------------------------------------
# STEP 10: Run analysis notebook (bound verification, stats, figures)
# ---------------------------------------------------------------------------
if [ "$START_STEP" -le 10 ]; then
    step_header 10 "RUN ANALYSIS NOTEBOOK"
    cd "$PROJECT_DIR"

    # Execute notebook non-interactively
    pip install nbconvert 2>/dev/null
    jupyter nbconvert --to notebook --execute analysis.ipynb \
        --output analysis_executed.ipynb \
        --ExecutePreprocessor.timeout=3600 \
    || echo "[warn] Notebook execution had errors — open in JupyterLab to debug"

    echo ""
    echo "Results saved to:"
    echo "  figures/           — all paper figures (PDF)"
    echo "  results/           — JSON/CSV data files"
    echo "  analysis_executed.ipynb — executed notebook with outputs"

    echo "[STEP 10] DONE"
fi

# ---------------------------------------------------------------------------
# DONE
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  ALL STEPS COMPLETE"
echo "============================================================"
echo ""
echo "Output locations:"
echo "  Checkpoints:  $CKPT_DIR/"
echo "  Outputs:      $OUTPUT_DIR/"
echo "  Results:      $RESULTS_DIR/"
echo "  Figures:      $PROJECT_DIR/figures/"
echo "  Notebook:     $PROJECT_DIR/analysis_executed.ipynb"
echo ""
echo "To view figures interactively:"
echo "  jupyter lab --ip=0.0.0.0 --port=8888 --allow-root"
echo ""
echo "Quick reference — run individual steps:"
echo "  bash run_all.sh --step 5    # start from training"
echo "  bash run_all.sh --step 8    # start from transfer eval"
echo "  bash run_all.sh --step 10   # just run analysis notebook"
echo ""
