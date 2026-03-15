#!/bin/bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

# You can override these by exporting env vars before running this script.
# 从 196 到 1024
MODEL_PATH=${MODEL_PATH:-/gz-data/qwen3vl_2b}
DATA_FILE=${DATA_FILE:-/gz-data/dataset/data/testmini-00000-of-00001-725687bf7a18d64b.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-logs/baseline_qwen3vl_2b_testmini_full/max_new_token_2048_version_1}
MAX_SAMPLES=${MAX_SAMPLES:--1000}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
PREFETCH_WORKERS=${PREFETCH_WORKERS:-8}
PREFETCH_SIZE=${PREFETCH_SIZE:-32}

python3 -u examples/mllm/eval_baseline.py \
    --model-path "${MODEL_PATH}" \
    --data-file "${DATA_FILE}" \
    --max-samples "${MAX_SAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --prefetch-workers "${PREFETCH_WORKERS}" \
    --prefetch-size "${PREFETCH_SIZE}" \
    --pin-memory \
    --non-blocking-to-device \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
