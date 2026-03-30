#!/bin/bash
# =============================================================================
# Download CHB-MIT Scalp EEG Database from PhysioNet (OPEN ACCESS)
# No credentials required — instant download.
# https://physionet.org/content/chbmit/1.0.0/
# =============================================================================
set -euo pipefail

DATA_DIR="${1:-/workspace/data/chbmit}"
MODE="${2:-full}"  # full | quick
mkdir -p "$DATA_DIR"

echo "============================================"
echo "  CHB-MIT Scalp EEG Download (Open Access)"
echo "  Mode: $MODE"
echo "============================================"

cd "$DATA_DIR"

if [ "$MODE" = "quick" ]; then
    echo "Quick mode: downloading only one subject subset for smoke tests."
    mkdir -p "$DATA_DIR/chb01"

    # Includes summary + a few EDFs with at least one seizure example.
    URLS=(
        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01-summary.txt"
        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_01.edf"
        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_02.edf"
        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_03.edf"
        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_03.edf.seizures"
        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_04.edf"
    )

    if command -v aria2c >/dev/null 2>&1; then
        printf '%s\n' "${URLS[@]}" > "$DATA_DIR/chbmit_quick_urls.txt"
        aria2c -x 16 -s 16 -j 6 -i "$DATA_DIR/chbmit_quick_urls.txt" -d "$DATA_DIR/chb01"
        rm -f "$DATA_DIR/chbmit_quick_urls.txt"
    else
        for u in "${URLS[@]}"; do
            wget -c "$u" -P "$DATA_DIR/chb01"
        done
    fi
else
    echo "Full mode: downloading complete CHB-MIT dataset."

    # Fastest public path when aws cli is available.
    if command -v aws >/dev/null 2>&1; then
        aws s3 sync --no-sign-request \
            "s3://physionet-open/chbmit/1.0.0/" \
            "$DATA_DIR/" \
            --exclude "index.html*" \
            || true
    fi

    # Verify we downloaded some EDFs; otherwise fallback to recursive HTTP.
    EDF_COUNT=$(find "$DATA_DIR" -type f -name "*.edf" | wc -l || true)
    if [ "${EDF_COUNT:-0}" -eq 0 ]; then
        wget -r -N -c -np -nH --cut-dirs=3 \
            "https://physionet.org/files/chbmit/1.0.0/" \
            -P "$DATA_DIR/" \
            --reject="index.html*" \
            || {
            echo "wget failed. Trying wfdb client fallback..."
            pip install wfdb 2>/dev/null
            python3 -c "
import wfdb
wfdb.dl_database('chbmit/1.0.0', '$DATA_DIR')
"
        }
    fi
fi

echo ""
echo "CHB-MIT download complete. Files in: $DATA_DIR"
echo "Next: run python scripts/preprocess.py --dataset chbmit"
