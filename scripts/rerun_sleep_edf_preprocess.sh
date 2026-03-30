#!/bin/bash
# Rebuild Sleep-EDF preprocessed outputs only.
# Usage:
#   bash scripts/rerun_sleep_edf_preprocess.sh
#   bash scripts/rerun_sleep_edf_preprocess.sh /workspace/data

set -euo pipefail

DATA_ROOT="${1:-/workspace/data}"
SLEEP_DIR="$DATA_ROOT/sleep_edf"

echo "Cleaning old Sleep-EDF preprocessed files..."
rm -f "$SLEEP_DIR/data.h5" "$SLEEP_DIR/splits.json"

echo "Re-running Sleep-EDF preprocessing..."
python scripts/preprocess.py --dataset sleep_edf --data_root "$DATA_ROOT"

echo "Done. Rebuilt Sleep-EDF outputs in: $SLEEP_DIR"
