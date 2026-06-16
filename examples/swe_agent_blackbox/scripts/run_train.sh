#!/usr/bin/env bash
# Training launch script for the blackbox SWE-agent recipe.
#
# Uses GRPO + AgentFrameworkRolloutAdapter with reward computed in-process
# by the agent runner, then passed through the reward worker's compute_score.
#
# Usage:
#   bash examples/swe_agent_blackbox/scripts/run_train.sh
#
# All configurable via environment variables (see defaults below).

set -euo pipefail

# ── Model & data ─────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-Coder-30B-A3B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-$HOME/data/swe_agent/swe_bench_verified.parquet}"
VAL_DATA="${VAL_DATA:-$HOME/data/swe_agent/swe_bench_verified.parquet}"

# ── Hardware ─────────────────────────────────────────────────────────────
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"

# ── Training parameters ─────────────────────────────────────────────────
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-4096}"
ACTOR_LR="${ACTOR_LR:-3e-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-30}"
SAVE_FREQ="${SAVE_FREQ:-5}"
TEST_FREQ="${TEST_FREQ:-5}"

# ── Rollout parameters ──────────────────────────────────────────────────
ENGINE="${ENGINE:-vllm}"
TP="${TP:-4}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.5}"
N="${N:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"

# ── Agent parameters ─────────────────────────────────────────────────────
MAX_TURNS="${MAX_TURNS:-100}"
AGENT_CONFIG_PATH="${AGENT_CONFIG_PATH:-examples/swe_agent_blackbox/config/agent_config.yaml}"
COMPLETION_TIMEOUT="${COMPLETION_TIMEOUT:-600}"

# ── Logging ──────────────────────────────────────────────────────────────
PROJECT_NAME="${PROJECT_NAME:-swe_agent_blackbox}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-swe_agent_$(date +%Y%m%d_%H%M)}"
VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"

export SWE_AGENT_MAX_TURNS="${MAX_TURNS}"
export SWE_AGENT_EVAL_TIMEOUT="${SWE_AGENT_EVAL_TIMEOUT:-600}"
export VERL_LOGGING_LEVEL

# ── Environment for NCCL ─────────────────────────────────────────────────
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export TRANSFORMERS_ATTN_IMPLEMENTATION="${TRANSFORMERS_ATTN_IMPLEMENTATION:-eager}"

echo "=== SWE-Agent Blackbox Training ==="
echo "Model:       ${MODEL_PATH}"
echo "Train data:  ${TRAIN_DATA}"
echo "Val data:    ${VAL_DATA}"
echo "Engine:      ${ENGINE} (TP=${TP})"
echo "Batch size:  ${TRAIN_BATCH_SIZE}, N=${N}"
echo "Epochs:      ${TOTAL_EPOCHS}"
echo "====================================="

python3 -m verl.trainer.main_ppo_sync \
    --config-name=swe_agent_blackbox \
    --config-path="$(pwd)/examples/swe_agent_blackbox/config" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    data.train_files="['${TRAIN_DATA}']" \
    data.val_files="['${VAL_DATA}']" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${PROMPT_LENGTH} \
    data.max_response_length=${RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${N} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH} \
    actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.max_model_len=$((PROMPT_LENGTH + RESPONSE_LENGTH + 1024)) \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS} \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.rollout.nnodes=${NNODES} \
    actor_rollout_ref.rollout.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    actor_rollout_ref.rollout.custom.agent_framework.agent_runner_kwargs.agent_config_path="${AGENT_CONFIG_PATH}" \
    actor_rollout_ref.rollout.custom.agent_framework.completion_timeout_seconds=${COMPLETION_TIMEOUT} \
    "$@"
