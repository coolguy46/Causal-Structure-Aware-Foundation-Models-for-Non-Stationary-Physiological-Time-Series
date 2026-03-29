#!/bin/bash
# =============================================================================
# Download SHHS (Sleep Heart Health Study) dataset
# Requires NSRR account and token: https://sleepdata.org/
# =============================================================================
set -e

DATA_DIR="${1:-/workspace/data/shhs}"
mkdir -p "$DATA_DIR"

echo "============================================"
echo "  SHHS Dataset Download"
echo "============================================"

# Check for NSRR token
if [ -z "$NSRR_TOKEN" ]; then
    echo "ERROR: Set NSRR_TOKEN environment variable."
    echo ""
    echo "Steps to get access:"
    echo "  1. Create account at https://sleepdata.org/"
    echo "  2. Request access to SHHS dataset"
    echo "  3. Get your API token from your profile"
    echo "  4. export NSRR_TOKEN=your_token_here"
    exit 1
fi

# Install NSRR gem if not present
if ! command -v nsrr &> /dev/null; then
    echo "Installing NSRR downloader..."
    pip install nsrr
fi

echo "Downloading SHHS-1 EDF files..."
cd "$DATA_DIR"

# Download polysomnography EDFs (C3-A2, C4-A1 channels for sleep staging)
nsrr download shhs/polysomnography/edfs/shhs1 --token "$NSRR_TOKEN" || {
    echo "nsrr CLI failed. Trying direct download..."
    echo "Please download manually from: https://sleepdata.org/datasets/shhs"
    exit 1
}

echo "Downloading annotations..."
nsrr download shhs/polysomnography/annotations-events-nsrr/shhs1 --token "$NSRR_TOKEN" || true

echo ""
echo "SHHS download complete. Files in: $DATA_DIR"
echo "Next: run python scripts/preprocess.py --dataset shhs"
