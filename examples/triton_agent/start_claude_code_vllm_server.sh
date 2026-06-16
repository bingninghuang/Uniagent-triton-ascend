#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------------------------------------------------------
# Model / vLLM settings
# ------------------------------------------------------------------------------
export MODEL_PATH="${MODEL_PATH:-/home/p00938733/Qwen3-Coder-30B-A3B-Instruct}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-triton-claude-code-model}"
export PROXY_PORT="${PROXY_PORT:-5000}"

nic_name="${NIC_NAME:-ens1f3}"
export HCCL_IF_IP="${HCCL_IF_IP:-80.48.5.63}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${nic_name}}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-${nic_name}}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-${nic_name}}"

export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_INTRA_PCIE_ENABLE="${HCCL_INTRA_PCIE_ENABLE:-0}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}"

export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARN}"

export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-3600000}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-30000}"
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT="${VLLM_NIXL_ABORT_REQUEST_TIMEOUT:-30000}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-204}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-120}"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:512}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-100000000000}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

if [[ "${MODEL_PATH}" == *"Qwen3-Coder"* ]]; then
    TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
else
    TOOL_PARSER="${TOOL_PARSER:-hermes}"
fi

ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-{\"enable_cpu_binding\":true,\"fuse_muls_add\":true,\"multistream_overlap_shared_expert\":false}}"
VLLM_LOG_PATH="${VLLM_LOG_PATH:-${SCRIPT_DIR}/vllm_server.log}"

echo "[vllm] model          : ${MODEL_PATH}"
echo "[vllm] served name    : ${SERVED_MODEL_NAME}"
echo "[vllm] endpoint       : http://0.0.0.0:${PROXY_PORT}/v1"
echo "[vllm] visible devices: ${ASCEND_RT_VISIBLE_DEVICES}"
echo "[vllm] TP/PP          : ${VLLM_TP_SIZE:-8}/${VLLM_PP_SIZE:-1}"
echo "[vllm] tool parser    : ${TOOL_PARSER}"
echo "[vllm] log            : ${VLLM_LOG_PATH}"

vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PROXY_PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --data-parallel-size "${VLLM_DP_SIZE:-1}" \
  --tensor-parallel-size "${VLLM_TP_SIZE:-8}" \
  --pipeline-parallel-size "${VLLM_PP_SIZE:-1}" \
  --prefill-context-parallel-size "${VLLM_PREFILL_CP_SIZE:-1}" \
  --decode-context-parallel-size "${VLLM_DECODE_CP_SIZE:-1}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-2}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-262144}" \
  --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.80}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER}" \
  --trust-remote-code \
  --dtype "${VLLM_DTYPE:-bfloat16}" \
  --additional-config "${ADDITIONAL_CONFIG}" \
  2>&1 | tee "${VLLM_LOG_PATH}"
