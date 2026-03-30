#!/bin/bash
# =============================================================================
# Download PTB-XL ECG dataset from PhysioNet (OPEN ACCESS)
# No credentials required — instant download.
# https://physionet.org/content/ptb-xl/1.0.3/
# =============================================================================
set -euo pipefail

DATA_DIR="${1:-/workspace/data/ptbxl}"
MODE="${2:-full}"  # full | quick
mkdir -p "$DATA_DIR"

echo "============================================"
echo "  PTB-XL ECG Download (Open Access)"
echo "  Mode: $MODE"
echo "============================================"

cd "$DATA_DIR"

if [ "$MODE" = "quick" ]; then
    echo "Quick mode: downloading PTB-XL metadata + a small record subset."

    # Metadata needed by downstream preprocessing.
    wget -c "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv" -P "$DATA_DIR/"
    wget -c "https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv" -P "$DATA_DIR/"

    # Download first 100 records listed in metadata for smoke tests.
    pip install wfdb pandas 2>/dev/null || true
    export DATA_DIR
    python3 - <<'PY'
import os
import pandas as pd
import wfdb

data_dir = os.environ.get("DATA_DIR", "")
csv_path = os.path.join(data_dir, "ptbxl_database.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(csv_path)

df = pd.read_csv(csv_path)
records = df["filename_hr"].dropna().head(100).tolist()
wfdb.dl_database("ptb-xl/1.0.3", data_dir, records=records)
print(f"Downloaded {len(records)} PTB-XL records (quick mode)")
PY
else
    echo "Full mode: downloading complete PTB-XL dataset."

    if command -v aws >/dev/null 2>&1; then
        aws s3 sync --no-sign-request \
            "s3://physionet-open/ptb-xl/1.0.3/" \
            "$DATA_DIR/" \
            --exclude "index.html*" \
            || true
    fi

    HEA_COUNT=$(find "$DATA_DIR" -type f -name "*.hea" | wc -l || true)
    if [ "${HEA_COUNT:-0}" -eq 0 ]; then
        wget -r -N -c -np -nH --cut-dirs=3 \
            "https://physionet.org/files/ptb-xl/1.0.3/" \
            -P "$DATA_DIR/" \
            --reject="index.html*" \
            || {
            echo "wget failed. Trying wfdb client fallback..."
            pip install wfdb 2>/dev/null
            python3 -c "
import wfdb
wfdb.dl_database('ptb-xl/1.0.3', '$DATA_DIR')
"
        }
    fi
fi

echo ""
echo "PTB-XL download complete. Files in: $DATA_DIR"
echo "Next: run python scripts/preprocess.py --dataset ptbxl"
