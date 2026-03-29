#!/bin/bash
# =============================================================================
# Download CHB-MIT Scalp EEG Database from PhysioNet (OPEN ACCESS)
# No credentials required — instant download.
# https://physionet.org/content/chbmit/1.0.0/
# =============================================================================
set -e

DATA_DIR="${1:-/workspace/data/chbmit}"
mkdir -p "$DATA_DIR"

echo "============================================"
echo "  CHB-MIT Scalp EEG Download (Open Access)"
echo "============================================"

cd "$DATA_DIR"

echo "Downloading CHB-MIT (23 pediatric subjects with seizure annotations)..."
wget -r -N -c -np -nH --cut-dirs=3 \
    "https://physionet.org/files/chbmit/1.0.0/" \
    -P "$DATA_DIR/" \
    --reject="index.html*" \
    || {
    echo "wget failed. Trying alternative..."
    pip install wfdb 2>/dev/null
    python3 -c "
import wfdb
wfdb.dl_database('chbmit/1.0.0', '$DATA_DIR')
"
}

echo ""
echo "CHB-MIT download complete. Files in: $DATA_DIR"
echo "Next: run python scripts/preprocess.py --dataset chbmit"
