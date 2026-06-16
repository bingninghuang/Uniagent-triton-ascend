#!/usr/bin/env bash
set -euo pipefail
set -x

pkill -9 python || true
pkill -9 torchrun || true

# export MODEL_PATH=/home/p00938733/Qwen3-8B
export MODEL_PATH="${MODEL_PATH:-/home/p00938733/Qwen3-Coder-30B-A3B-Instruct}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-openhands-model}"
export PROXY_PORT="${PROXY_PORT:-5000}"

nic_name="ens1f3"
export HCCL_IF_IP=80.48.5.51
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export HCCL_INTRA_ROCE_ENABLE=1 # TODO
export HCCL_INTRA_PCIE_ENABLE=0 # TODO
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
# export HCCL_CONNECT_TIMEOUT=300

export VLLM_ASCEND_ENABLE_NZ=0
export TOKENIZERS_PARALLELISM=true
export VLLM_LOGGING_LEVEL=WARN

# Timeouts
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120

export OPENHANDS_IMAGE=openhands-triton-env:v1

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# ------------------------------------------------------------------------------
# vLLM / CUDA
# ------------------------------------------------------------------------------
export VLLM_ATTENTION_BACKEND="TORCH_SDPA"
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:512"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

export TOKENIZERS_PARALLELISM=true
export VLLM_LOGGING_LEVEL=WARN
export HYDRA_FULL_ERROR=1

# ------------------------------------------------------------------------------
# OpenHands container settings
# (read by openhands_agent.py → forwarded into each OpenHands container)
# ------------------------------------------------------------------------------
# Custom rllm-openhands image (built from workspace/Dockerfile).
# This image extends the official OpenHands image with workspace/entrypoint.py
# which uses the new OpenHands SDK (LLM, Agent, Conversation, Tool).
export OPENHANDS_IMAGE="${OPENHANDS_IMAGE:-openhands-triton-env:v1}"
# export OPENHANDS_MODEL_NAME="${OPENHANDS_MODEL_NAME:-/home/p00938733/Qwen3-8B}"
export OPENHANDS_MODEL_NAME="${OPENHANDS_MODEL_NAME:-$SERVED_MODEL_NAME}"
export OPENHANDS_BASE_URL_PORT=${PROXY_PORT:-4000}
export OPENHANDS_MAX_ITERATIONS="${OPENHANDS_MAX_ITERATIONS:-15}"
export OPENHANDS_CONTAINER_TIMEOUT="${OPENHANDS_CONTAINER_TIMEOUT:-1800}"
export OPENHANDS_ARTIFACT_DIR="${OPENHANDS_ARTIFACT_DIR:-/home/p00938733/openhands_results}"

# ------------------------------------------------------------------------------
# Training parameters
# ------------------------------------------------------------------------------
N_GPUS="${N_GPUS:-16}"

if [[ "$MODEL_PATH" == *"Qwen3-Coder"* ]]; then
    TOOL_PARSER=qwen3_coder
else
    TOOL_PARSER=hermes
fi

ADDITIONAL_CONFIG='{"enable_cpu_binding":true,"fuse_muls_add":true,"multistream_overlap_shared_expert":false}'

# =========================
# Start vLLM
# =========================
vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PROXY_PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --data-parallel-size 1 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --prefill-context-parallel-size 1 \
  --decode-context-parallel-size 1 \
  --max-num-seqs 2 \
  --max-model-len 262144 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.80 \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_PARSER" \
  --trust-remote-code \
  --dtype bfloat16 \
  --additional-config "$ADDITIONAL_CONFIG" \
  2>&1 | tee /home/p00938733/vllm.log
