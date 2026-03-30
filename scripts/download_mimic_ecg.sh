#!/bin/bash
# =============================================================================
# Open-access replacement for MIMIC-IV-ECG download.
# Redirects to PTB-XL.
# =============================================================================
set -e

DATA_DIR="${1:-/workspace/data/ptbxl}"
mkdir -p "$DATA_DIR"

echo "============================================"
echo "  Open Replacement for MIMIC-IV-ECG"
echo "============================================"

echo "Restricted MIMIC-IV-ECG was replaced with open PTB-XL for this repo."
echo "Calling download_ptbxl.sh with target: $DATA_DIR"
bash "$(dirname "$0")/download_ptbxl.sh" "$DATA_DIR"
