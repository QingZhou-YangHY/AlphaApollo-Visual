#!/bin/bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

# You can override these by exporting env vars before running this script.
MODEL_PATH=${MODEL_PATH:-/gz-data/qwen3vl_4b}
DATA_FILE=${DATA_FILE:-/gz-data/dataset/data/testmini-00000-of-00001-725687bf7a18d64b.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-logs/vllm_qwen3vl_4b_testmini_full_parallel/max_new_token_1024_version_bs1}
MAX_SAMPLES=${MAX_SAMPLES:-1000}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-1}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-8}
DATA_NUM_WORKERS=${DATA_NUM_WORKERS:-4}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-true}
FREE_CACHE_ENGINE_FLAG=""
if [ "${FREE_CACHE_ENGINE}" = "true" ]; then
    FREE_CACHE_ENGINE_FLAG="--free-cache-engine"
fi
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8172}

# cd /gz-data/qwen3vl_2b
# sed -i 's/Qwen3VLForConditionalGeneration/Qwen2VLForConditionalGeneration/g' config.json
# sed -i 's/"model_type": "qwen3_vl"/"model_type": "qwen2_vl"/g' config.json

# export PYTHONPATH=/gz-data/AlphaApollo-Visual/alphaapollo/core/generation:$PYTHONPATH
# cd /gz-data/AlphaApollo-Visual/alphaapollo
python3 -u examples/mllm/eval_vllm.py \
    --model-path "${MODEL_PATH}" \
    --data-file "${DATA_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-samples "${MAX_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --n-gpus-per-node "${N_GPUS_PER_NODE}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --ray-num-cpus "${RAY_NUM_CPUS}" \
    --data-num-workers "${DATA_NUM_WORKERS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    $([ "${FREE_CACHE_ENGINE}" = "true" ] && echo "--free-cache-engine") \
    "$@"
