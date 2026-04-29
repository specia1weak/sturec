#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-inspect}"

# -----------------------------------------------------------------------------
# Local path configuration.
# Edit these paths to match your machine before running.
# -----------------------------------------------------------------------------

# Official-like training env vars.
export TRAIN_DATA_PATH="/abs/path/to/train_or_full_data"
export TRAIN_CKPT_PATH="/abs/path/to/output/checkpoints"
export TRAIN_LOG_PATH="/abs/path/to/output/logs"
export TRAIN_TF_EVENTS_PATH="/abs/path/to/output/tf_events"

# Official-like evaluation env vars.
export MODEL_OUTPUT_PATH="/abs/path/to/output/checkpoints/model_output"
export EVAL_DATA_PATH="/abs/path/to/eval_data"
export EVAL_RESULT_PATH="/abs/path/to/output/eval_result"

# Local simulation extras.
export SIM_WORKDIR="/abs/path/to/output/schema_manager_workdir"
export SIM_INSPECT_OUTPUT="/abs/path/to/output/inspect_report.json"

# Simple model / data options.
export SIM_DEVICE="cuda"
export SIM_BATCH_SIZE="4096"
export SIM_MAX_EPOCHS="3"
export SIM_ID_EMB_DIM="32"
export SIM_SPARSE_EMB_DIM="16"
export SIM_MIN_FREQ="10"
export SIM_HIDDEN_DIMS="256,128"

# Data split / label handling.
export SIM_SPLIT_STRATEGY="sequential_ratio"
export SIM_TRAIN_RATIO="0.8"
export SIM_VALID_RATIO="0.1"
export SIM_LABEL_SHIFT="1"

uv run python @examples/@myprojs/taac_official_sim.py --mode "${MODE}"
