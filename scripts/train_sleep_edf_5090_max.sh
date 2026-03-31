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
  num_workers=24 \
  train.batch_size=64 \
  eval.batch_size=64 \
  train.epochs=100 \
  train.accumulate_grad_batches=4 \
  train.lr=1e-4 \
  train.weight_decay=1e-4 \
  train.warmup_epochs=5 \
  hardware.precision=bf16-mixed \
  hardware.compile=true \
  hardware.compile_mode=reduce-overhead \
  paths.data_root="$DATA_ROOT"
