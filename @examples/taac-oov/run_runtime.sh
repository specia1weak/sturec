#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

DATA_DIR="${TRAIN_DATA_PATH:-}"
if [ -z "${DATA_DIR}" ]; then
  echo "TRAIN_DATA_PATH is required"
  exit 1
fi

RUNTIME_DIR="${TRAIN_RUNTIME_DIR:-${SCRIPT_DIR}/runtime_artifacts}"
PROBE_JSON="${TRAIN_PROBE_JSON:-${RUNTIME_DIR}/probe.json}"
RUNTIME_SCHEMA="${TRAIN_RUNTIME_SCHEMA_PATH:-${RUNTIME_DIR}/schema_runtime.json}"
KEEP_NPZ="${TRAIN_KEEP_NPZ:-${RUNTIME_DIR}/keep_ids.npz}"

MIN_FREQ="${TRAIN_MIN_FREQ:-10}"
SEQ_MAX_LENS="${TRAIN_SEQ_MAX_LENS:-seq_a:50,seq_b:50,seq_c:50,seq_d:50}"
OOV_BATCH_SIZE="${TRAIN_OOV_BATCH_SIZE:-256}"
OOV_MAX_CHUNK_COST="${TRAIN_OOV_MAX_CHUNK_COST:-200}"
OOV_DENSE_SAMPLE_SIZE="${TRAIN_OOV_DENSE_SAMPLE_SIZE:-200000}"
SEQ_TOP_K="${TRAIN_SEQ_TOP_K:-50}"

mkdir -p "${RUNTIME_DIR}"

echo "=== Runtime Pipeline ==="
echo "DATA_DIR=${DATA_DIR}"
echo "RUNTIME_DIR=${RUNTIME_DIR}"
echo "PROBE_JSON=${PROBE_JSON}"
echo "RUNTIME_SCHEMA=${RUNTIME_SCHEMA}"
echo "KEEP_NPZ=${KEEP_NPZ}"
echo "MIN_FREQ=${MIN_FREQ}"
echo "SEQ_MAX_LENS=${SEQ_MAX_LENS}"
echo "OOV_BATCH_SIZE=${OOV_BATCH_SIZE}"
echo "OOV_MAX_CHUNK_COST=${OOV_MAX_CHUNK_COST}"
echo "OOV_DENSE_SAMPLE_SIZE=${OOV_DENSE_SAMPLE_SIZE}"
echo "SEQ_TOP_K=${SEQ_TOP_K}"

python3 -u "${SCRIPT_DIR}/oov_builder.py" \
  --parquet_path "${DATA_DIR}" \
  --schema_path "${DATA_DIR}/schema.json" \
  --min_freq "${MIN_FREQ}" \
  --seq_max_lens "${SEQ_MAX_LENS}" \
  --batch_size "${OOV_BATCH_SIZE}" \
  --max_chunk_cost "${OOV_MAX_CHUNK_COST}" \
  --dense_sample_size "${OOV_DENSE_SAMPLE_SIZE}" \
  --output_json "${PROBE_JSON}"

python3 -u "${SCRIPT_DIR}/build_runtime_schema.py" \
  --schema_path "${DATA_DIR}/schema.json" \
  --probe_json "${PROBE_JSON}" \
  --output_schema "${RUNTIME_SCHEMA}" \
  --output_keep_npz "${KEEP_NPZ}"

python3 -u "${SCRIPT_DIR}/train.py" \
  --data_dir "${DATA_DIR}" \
  --schema_path "${RUNTIME_SCHEMA}" \
  --seq_max_lens "${SEQ_MAX_LENS}" \
  --seq_top_k "${SEQ_TOP_K}" \
  "$@"
