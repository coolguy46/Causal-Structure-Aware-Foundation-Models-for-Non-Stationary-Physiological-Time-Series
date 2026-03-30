#!/bin/bash
# =============================================================================
# Download Sleep-EDF Expanded dataset from PhysioNet (OPEN ACCESS)
# No credentials required — instant download.
# https://physionet.org/content/sleep-edfx/1.0.0/
# =============================================================================
set -euo pipefail

DATA_DIR="${1:-/workspace/data/sleep_edf}"
MODE="${2:-full}"  # full | quick
mkdir -p "$DATA_DIR"

echo "============================================"
echo "  Sleep-EDF Expanded Download (Open Access)"
echo "  Mode: $MODE"
echo "============================================"

cd "$DATA_DIR"

if [ "$MODE" = "quick" ]; then
    echo "Quick mode: downloading 2 PSG/Hypnogram pairs for smoke tests."
    mkdir -p "$DATA_DIR/sleep-cassette"
    URLS=(
        "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001E0-PSG.edf"
        "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001EC-Hypnogram.edf"
        "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4002E0-PSG.edf"
        "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4002EC-Hypnogram.edf"
    )

    if command -v aria2c >/dev/null 2>&1; then
        printf '%s\n' "${URLS[@]}" > "$DATA_DIR/sleep_edf_quick_urls.txt"
        aria2c -x 16 -s 16 -j 4 -i "$DATA_DIR/sleep_edf_quick_urls.txt" -d "$DATA_DIR/sleep-cassette"
        rm -f "$DATA_DIR/sleep_edf_quick_urls.txt"
    else
        for u in "${URLS[@]}"; do
            wget -c "$u" -P "$DATA_DIR/sleep-cassette"
        done
    fi
else
    echo "Full mode: downloading complete Sleep Cassette subset."

    if command -v aws >/dev/null 2>&1; then
        aws s3 sync --no-sign-request \
            "s3://physionet-open/sleep-edfx/1.0.0/sleep-cassette/" \
            "$DATA_DIR/sleep-cassette/" \
            --exclude "index.html*" \
            || true
    fi

    EDF_COUNT=$(find "$DATA_DIR/sleep-cassette" -type f -name "*.edf" | wc -l || true)
    if [ "${EDF_COUNT:-0}" -eq 0 ]; then
        wget -r -N -c -np -nH --cut-dirs=3 \
            "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/" \
            -P "$DATA_DIR/sleep-cassette/" \
            --reject="index.html*" \
            || {
            echo "wget failed. Trying wfdb client fallback..."
            pip install wfdb 2>/dev/null
            python3 -c "
import wfdb
import os
os.makedirs('$DATA_DIR/sleep-cassette', exist_ok=True)
wfdb.dl_database('sleep-edfx/1.0.0/sleep-cassette', '$DATA_DIR/sleep-cassette')
"
        }
    fi
fi

echo ""
echo "Sleep-EDF download complete. Files in: $DATA_DIR"
echo "Next: run python scripts/preprocess.py --dataset sleep_edf"
