#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASETS="${1:-0,1,2}"
BACKBONES="${2:-0,1,2,3,4,5,6,7}"
CUSTOM_SUFFIX="${3:-msr}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[run-msr-grid] root=${ROOT_DIR}"
echo "[run-msr-grid] datasets=${DATASETS}"
echo "[run-msr-grid] backbones=${BACKBONES}"
echo "[run-msr-grid] customsuffix=${CUSTOM_SUFFIX}"

cd "${ROOT_DIR}" || exit 1

IFS=',' read -r -a DATASET_ARRAY <<< "${DATASETS}"
IFS=',' read -r -a BACKBONE_ARRAY <<< "${BACKBONES}"

for dataset_index in "${DATASET_ARRAY[@]}"; do
  for backbone_index in "${BACKBONE_ARRAY[@]}"; do
    echo "================================================================================"
    echo "[run-msr-grid] dataset_index=${dataset_index} backbone_index=${backbone_index}"
    echo "================================================================================"
    if ! PYTHONPATH=src "${PYTHON_BIN}" "@examples/diff-msr/msr-sumup.py" \
      --datasetindex "${dataset_index}" \
      --backboneindex "${backbone_index}" \
      --customsuffix "${CUSTOM_SUFFIX}"; then
      echo "[run-msr-grid][skip] dataset_index=${dataset_index} backbone_index=${backbone_index}"
    fi
  done
done
