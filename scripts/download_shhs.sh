#!/bin/bash
# =============================================================================
# Open-access replacement for SHHS download.
# Redirects to Sleep-EDF Expanded.
# =============================================================================
set -e

DATA_DIR="${1:-/workspace/data/sleep_edf}"
mkdir -p "$DATA_DIR"

echo "============================================"
echo "  Open Replacement for SHHS"
echo "============================================"

echo "Restricted SHHS dataset was replaced with open Sleep-EDF for this repo."
echo "Calling download_sleep_edf.sh with target: $DATA_DIR"
bash "$(dirname "$0")/download_sleep_edf.sh" "$DATA_DIR"
