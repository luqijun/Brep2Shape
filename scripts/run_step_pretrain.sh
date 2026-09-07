#!/usr/bin/env bash
# Self-supervised pretraining on preprocessed STEP data.
#
# Usage:
#   DATASET_DIR=/path/to/preprocessed_dataset bash scripts/run_step_pretrain.sh
#
# 1) Preprocess STEP files first:
#   python step_preprocess.py --input_dir /path/to/step_files --output_dir /path/to/preprocessed_dataset
set -euo pipefail

cd "$(dirname "$0")/.."

: "${DATASET_DIR:?Set DATASET_DIR to the output of step_preprocess.py}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DATASET_DIR
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-step_pretraining}"
export EPOCHS="${EPOCHS:-50}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
bash scripts/pretrain.sh
