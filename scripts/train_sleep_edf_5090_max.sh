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
  num_workers=2 \
  train.batch_size=128 \
  eval.batch_size=256 \
  train.epochs=100 \
  train.accumulate_grad_batches=2 \
  train.lr=2e-4 \
  train.weight_decay=1e-4 \
  train.warmup_epochs=5 \
  hardware.precision=bf16-mixed \
  hardware.compile=false \
  paths.data_root="$DATA_ROOT"
