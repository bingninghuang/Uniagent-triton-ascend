#!/usr/bin/env bash
# Parallel inference for the blackbox SWE-agent recipe.
#
# Usage:
#   bash examples/swe_agent_blackbox/scripts/run_infer.sh

set -euo pipefail

# ── Model & data ─────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3.5-9B}"
DATA_PATH="${DATA_PATH:-$HOME/data/swe_agent/swe_bench_verified.parquet}"

# ── Inference parameters ─────────────────────────────────────────────────
MAX_SAMPLES="${MAX_SAMPLES:--1}"
PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-65536}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
N="${N:-8}"
ENGINE="${ENGINE:-vllm}"
TP="${TP:-4}"

# ── Agent parameters ─────────────────────────────────────────────────────
RUNNER="${RUNNER:-uniagent}"
AGENT_CONFIG_PATH="${AGENT_CONFIG_PATH:-examples/swe_agent_blackbox/config/agent_config.yaml}"
export SWE_AGENT_MAX_TURNS="${SWE_AGENT_MAX_TURNS:-100}"
export SWE_AGENT_ACTION_TIMEOUT="${SWE_AGENT_ACTION_TIMEOUT:-300}"
export SWE_AGENT_EVAL_TIMEOUT="${SWE_AGENT_EVAL_TIMEOUT:-600}"

# ── Logging ──────────────────────────────────────────────────────────────
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"

echo "=== SWE-Agent Blackbox Inference ==="
echo "Model: ${MODEL_PATH}"
echo "Data:  ${DATA_PATH}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Engine: ${ENGINE} (TP=${TP})"
echo "====================================="

python examples/swe_agent_blackbox/parallel_infer.py \
    --model-path "${MODEL_PATH}" \
    --data-path "${DATA_PATH}" \
    --max-samples "${MAX_SAMPLES}" \
    --prompt-length "${PROMPT_LENGTH}" \
    --response-length "${RESPONSE_LENGTH}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --n "${N}" \
    --engine "${ENGINE}" \
    --tensor-parallel-size "${TP}" \
    --max-turns "${SWE_AGENT_MAX_TURNS}" \
    --runner "${RUNNER}" \
    --agent-config-path "${AGENT_CONFIG_PATH}"
