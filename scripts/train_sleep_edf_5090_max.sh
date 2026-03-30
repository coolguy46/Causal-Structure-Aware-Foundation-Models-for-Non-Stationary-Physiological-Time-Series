#!/bin/bash
# Run a high-throughput, high-capacity Sleep-EDF training job on RTX 5090.
# Usage:
#   bash scripts/train_sleep_edf_5090_max.sh
#   bash scripts/train_sleep_edf_5090_max.sh /workspace/data

set -euo pipefail

DATA_ROOT="${1:-/workspace/data}"

python -m src.train \
  dataset=sleep_edf \
  model=large_5090 \
  num_workers=16 \
  train.batch_size=256 \
  eval.batch_size=256 \
  train.epochs=60 \
  hardware.precision=bf16-mixed \
  hardware.compile=true \
  hardware.compile_mode=reduce-overhead \
  paths.data_root="$DATA_ROOT"
