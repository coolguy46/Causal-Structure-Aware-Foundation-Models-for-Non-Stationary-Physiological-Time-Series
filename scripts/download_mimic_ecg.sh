#!/bin/bash
# =============================================================================
# Download MIMIC-IV-ECG dataset from PhysioNet
# Requires PhysioNet credentialed account
# =============================================================================
set -e

DATA_DIR="${1:-/workspace/data/mimic_ecg}"
mkdir -p "$DATA_DIR"

echo "============================================"
echo "  MIMIC-IV-ECG Download"
echo "============================================"

if [ -z "$PHYSIONET_USER" ] || [ -z "$PHYSIONET_PASS" ]; then
    echo "ERROR: Set PHYSIONET_USER and PHYSIONET_PASS environment variables."
    echo ""
    echo "Steps to get access:"
    echo "  1. Create PhysioNet account: https://physionet.org/"
    echo "  2. Complete required training (CITI)"
    echo "  3. Request access to MIMIC-IV-ECG"
    echo "  4. export PHYSIONET_USER=your_username"
    echo "  5. export PHYSIONET_PASS=your_password"
    exit 1
fi

echo "Downloading MIMIC-IV-ECG..."
cd "$DATA_DIR"

wget -r -N -c -np \
    --user="$PHYSIONET_USER" --password="$PHYSIONET_PASS" \
    "https://physionet.org/files/mimic-iv-ecg/1.0/" \
    -P "$DATA_DIR/" || {
    echo "wget failed. Please download manually from PhysioNet."
    exit 1
}

echo ""
echo "MIMIC-IV-ECG download complete. Files in: $DATA_DIR"
echo "Next: run python scripts/preprocess.py --dataset mimic_ecg"
