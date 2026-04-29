#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODE="${1:-inspect}"

set -a
source "${ROOT_DIR}/.env"
set +a

uv run python "${SCRIPT_DIR}/local_official_sim.py" --mode "${MODE}"
