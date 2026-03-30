#!/bin/bash
# =============================================================================
# Open-access replacement for TUSZ download.
# Redirects to CHB-MIT.
# =============================================================================
set -e

DATA_DIR="${1:-/workspace/data/chbmit}"
mkdir -p "$DATA_DIR"

echo "============================================"
echo "  Open Replacement for TUSZ"
echo "============================================"

echo "Restricted TUSZ dataset was replaced with open CHB-MIT for this repo."
echo "Calling download_chbmit.sh with target: $DATA_DIR"
bash "$(dirname "$0")/download_chbmit.sh" "$DATA_DIR"
