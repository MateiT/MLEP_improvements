#!/usr/bin/env bash
# MLEP window / robustness sweep on an NVIDIA RTX 4090.
#
# On Linux the dataloader uses 'fork', so num_threads > 0 is safe (the "keep 0"
# note in experiment_windows.py is a macOS-only spawn/pickling caveat). Bump
# num_threads so the 4090 is not starved by single-process data loading.
set -euo pipefail

# ---------------------------------------------------------------------------
# Quick smoke test (a few minutes) -- confirms the GPU path works end-to-end.
# ---------------------------------------------------------------------------
conda run -n MLEP python experiment_windows.py \
  --dataroot datasets/TrainDatasets \
  --classes car \
  --arch resnet18 \
  --batch_size 64 \
  --cropSize 224 \
  --num_threads 8 \
  --lr 2e-4 \
  --max_train_steps 200 \
  --max_val_batches 20 \
  --log_freq 50 \
  --device cuda \
  --configs baseline_2x2,w3x3_multiscale,baseline_train_blur,baseline_train_jpeg \
  --out experiment_results.txt 2>&1 | tail -40

# ---------------------------------------------------------------------------
# Full sweep (all configs, longer training). Uncomment to run once the smoke
# test looks good. Scale batch_size / max_train_steps to taste; the 4090's
# 24 GB comfortably handles batch_size 128 at cropSize 224 on resnet18.
# ---------------------------------------------------------------------------
# conda run -n MLEP python experiment_windows.py \
#   --dataroot datasets/TrainDatasets \
#   --classes car,cat,chair,horse \
#   --arch resnet18 \
#   --batch_size 128 \
#   --cropSize 224 \
#   --num_threads 12 \
#   --lr 2e-4 \
#   --max_train_steps 2000 \
#   --max_val_batches 50 \
#   --log_freq 100 \
#   --device cuda \
#   --out experiment_results.txt 2>&1 | tail -60
