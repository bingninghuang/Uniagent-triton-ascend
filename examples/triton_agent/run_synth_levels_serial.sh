#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/l00515014/opgen/train/sft/code/sft_test/Agentic_0425/Triton-Training}"
# DATASET="${DATASET:-${REPO_ROOT}/examples/triton_agent/benchmarks/NPUKernelBench}"
DATASET="${DATASET:-/home/l00515014/opgen/train/sft/code/sft_test/Agentic_0425/drkernel/data_process/distill/dataset_l1_l2/npu_py_level1_2_v2_modify_v2_split}"
OUT_ROOT="${OUT_ROOT:-/home/l00515014/opgen/train/sft/code/sft_test/Agentic_0425/uniagent_output}"

EVAL_DEVICE_IDS="${EVAL_DEVICE_IDS:-0}"
TRITON_ATTACH_PORT="${TRITON_ATTACH_PORT:-18000}"
TRITON_WORKSPACE_DIR="${TRITON_WORKSPACE_DIR:-/opt/workspace_serial_card0/agent_workdir}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"

PORT="${PORT:-7778}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-triton-synth}"
MAX_ROWS="${MAX_ROWS:-1024}"
FILTER_MODE="${FILTER_MODE:-all}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

SYNTH_CONTAINER="${SYNTH_CONTAINER:-uni-agent}"
TRITON_ATTACH_HOST="${TRITON_ATTACH_HOST:-http://127.0.0.1}"

TRITON_ATTACH_AUTH_TOKEN="${TRITON_ATTACH_AUTH_TOKEN:-mytoken12}"


export no_proxy="${no_proxy:-127.0.0.1,localhost,0.0.0.0,80.5.5.48}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"

# 串行执行的 level 顺序，可用 LEVELS 覆盖
# LEVELS="${LEVELS:-level0 level1 level2 level3 level4}"
LEVELS="${LEVELS:-level3 level1}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${OUT_ROOT}/synth_levels_serial_perf_${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
RESULT_DIR="${RUN_DIR}/results"

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

run_level() {
  local level="$1"

  local log_file="${LOG_DIR}/${level}.perf.log"
  local result_file="${RESULT_DIR}/results.perf.${level}.jsonl"

  echo "[serial] start level=${level}"
  echo "[serial] result=${result_file}"
  echo "[serial] log=${log_file}"
  echo "[serial] workspace=${TRITON_WORKSPACE_DIR}"
  echo "[serial] attach_port=${TRITON_ATTACH_PORT}"

  docker exec "${SYNTH_CONTAINER}" bash -lc "
    cd '${REPO_ROOT}' &&
    PYTHONUNBUFFERED=1 \
    TRITON_ATTACH_HOST='${TRITON_ATTACH_HOST}' \
    TRITON_ATTACH_PORT='${TRITON_ATTACH_PORT}' \
    TRITON_ATTACH_AUTH_TOKEN='${TRITON_ATTACH_AUTH_TOKEN}' \
    TRITON_WORKSPACE_DIR='${TRITON_WORKSPACE_DIR}' \
    TRITON_REWARD_TARGET_SPEEDUP='${TRITON_REWARD_TARGET_SPEEDUP:-2.0}' \
    no_proxy='${no_proxy}' \
    NO_PROXY='${NO_PROXY}' \
    python3 examples/triton_agent/synth_triton_local.py \
      --dataset '${DATASET}' \
      --levels '${level}' \
      --max-rows '${MAX_ROWS}' \
      --filter-mode '${FILTER_MODE}' \
      --port '${PORT}' \
      --host '${VLLM_HOST}' \
      --served-model-name '${SERVED_MODEL_NAME}' \
      --eval-device-ids '${EVAL_DEVICE_IDS}' \
      --output '${result_file}' \
      ${EXTRA_ARGS}
  " > "${log_file}" 2>&1

  echo "[serial] done level=${level} -> ${result_file}"
}

echo "[serial] dataset=${DATASET}"
echo "[serial] run_dir=${RUN_DIR}"
echo "[serial] logs=${LOG_DIR}"
echo "[serial] results=${RESULT_DIR}"
echo "[serial] attach=${TRITON_ATTACH_HOST}:${TRITON_ATTACH_PORT} container=${SYNTH_CONTAINER}"
echo "[serial] workspace=${TRITON_WORKSPACE_DIR}"
echo "[serial] synth_container=${SYNTH_CONTAINER}"
echo "[serial] levels=${LEVELS}"

status=0
for level in ${LEVELS}; do
  if ! run_level "${level}"; then
    echo "[serial] level=${level} failed, continuing to next level"
    status=1
  fi
done

echo "[serial] done status=${status}"
echo "[serial] results in ${RESULT_DIR}"
echo "[serial] logs in ${LOG_DIR}"

exit "${status}"