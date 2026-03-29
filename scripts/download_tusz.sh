#!/bin/bash
# =============================================================================
# Download TUSZ (TUH Seizure Corpus) dataset
# Requires IISP account: https://isip.piconepress.com/projects/tuh_eeg/
# =============================================================================
set -e

DATA_DIR="${1:-/workspace/data/tusz}"
mkdir -p "$DATA_DIR"

echo "============================================"
echo "  TUSZ Dataset Download"
echo "============================================"

if [ -z "$TUH_USER" ] || [ -z "$TUH_PASS" ]; then
    echo "ERROR: Set TUH_USER and TUH_PASS environment variables."
    echo ""
    echo "Steps to get access:"
    echo "  1. Request access at https://isip.piconepress.com/projects/tuh_eeg/"
    echo "  2. You'll receive credentials via email"
    echo "  3. export TUH_USER=your_username"
    echo "  4. export TUH_PASS=your_password"
    exit 1
fi

echo "Downloading TUSZ v2.0.1..."
cd "$DATA_DIR"

# TUSZ is hosted on an rsync server
rsync -auxvL --progress \
    "nedc@www.isip.piconepress.com:data/tuh_eeg_seizure/v2.0.1/" \
    "$DATA_DIR/" \
    --password-file=<(echo "$TUH_PASS") || {
    echo "rsync failed. Try wget approach..."
    wget -r -np -nH --cut-dirs=3 \
        --user="$TUH_USER" --password="$TUH_PASS" \
        "https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_seizure/v2.0.1/" \
        -P "$DATA_DIR/" || {
        echo "Download failed. Please download manually."
        exit 1
    }
}

echo ""
echo "TUSZ download complete. Files in: $DATA_DIR"
echo "Next: run python scripts/preprocess.py --dataset tusz"
