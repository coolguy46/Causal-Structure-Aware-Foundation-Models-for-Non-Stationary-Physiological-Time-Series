#!/bin/bash
# =============================================================================
# Download Sleep-EDF Expanded dataset from PhysioNet (OPEN ACCESS)
# No credentials required — instant download.
# https://physionet.org/content/sleep-edfx/1.0.0/
# =============================================================================
set -e

DATA_DIR="${1:-/workspace/data/sleep_edf}"
mkdir -p "$DATA_DIR"

echo "============================================"
echo "  Sleep-EDF Expanded Download (Open Access)"
echo "============================================"

cd "$DATA_DIR"

# Download the Sleep Cassette subset (SC subjects) — 153 PSG recordings from 78 subjects
# These are the commonly used recordings for 5-class sleep staging benchmarks
echo "Downloading Sleep-EDF Expanded (Sleep Cassette subset)..."
wget -r -N -c -np -nH --cut-dirs=3 \
    "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/" \
    -P "$DATA_DIR/sleep-cassette/" \
    --reject="index.html*" \
    || {
    echo "wget failed. Trying alternative..."
    pip install wfdb 2>/dev/null
    python3 -c "
import wfdb
import os
os.makedirs('$DATA_DIR/sleep-cassette', exist_ok=True)
wfdb.dl_database('sleep-edfx/1.0.0/sleep-cassette', '$DATA_DIR/sleep-cassette')
"
}

echo ""
echo "Sleep-EDF download complete. Files in: $DATA_DIR"
echo "Next: run python scripts/preprocess.py --dataset sleep_edf"
