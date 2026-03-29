#!/bin/bash
# =============================================================================
# Download PTB-XL ECG dataset from PhysioNet (OPEN ACCESS)
# No credentials required — instant download.
# https://physionet.org/content/ptb-xl/1.0.3/
# =============================================================================
set -e

DATA_DIR="${1:-/workspace/data/ptbxl}"
mkdir -p "$DATA_DIR"

echo "============================================"
echo "  PTB-XL ECG Download (Open Access)"
echo "============================================"

cd "$DATA_DIR"

echo "Downloading PTB-XL (21,799 12-lead ECG records, 10-second, 500Hz)..."
wget -r -N -c -np -nH --cut-dirs=3 \
    "https://physionet.org/files/ptb-xl/1.0.3/" \
    -P "$DATA_DIR/" \
    --reject="index.html*" \
    || {
    echo "wget failed. Trying alternative..."
    pip install wfdb 2>/dev/null
    python3 -c "
import wfdb
wfdb.dl_database('ptb-xl/1.0.3', '$DATA_DIR')
"
}

echo ""
echo "PTB-XL download complete. Files in: $DATA_DIR"
echo "Next: run python scripts/preprocess.py --dataset ptbxl"
