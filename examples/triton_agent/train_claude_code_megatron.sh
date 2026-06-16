#!/usr/bin/env bash
# ==============================================================================
# Train Triton/Ascend agent with verl + uni-agent blackbox gateway + Claude Code.
#
# This script trains Triton/Ascend rollouts through Claude Code:
#
#   Ray trainer
#     -> uni-agent blackbox AgentFramework gateway
#       -> Claude Code CLI in a local Docker sandbox
#         -> Anthropic/OpenAI shim
#           -> verl rollout actor
#
# Key knobs:
#   MODEL_PATH, N_GPUS, BATCH_SIZE, ROLLOUT_N, VAL_BATCH_SIZE, VAL_ROLLOUT_N
# ==============================================================================
pkill -9 python || true
pkill -9 torchrun || true
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# First time
# export MODEL_PATH=/home/p00938733/Qwen3-8B
export MODEL_PATH="${MODEL_PATH:-/home/p00938733/Qwen3-Coder-30B-A3B-Instruct}"
# export MODEL_PATH=/home/p00938733/cszhou_sft_weight/global_step_100
export PROXY_PORT="${PROXY_PORT:-5000}"  # kept for compatibility; blackbox gateway does not use LiteLLM proxy
export LLM_MAX_OUTPUT_TOKENS="${LLM_MAX_OUTPUT_TOKENS:-4096}"

export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"

nic_name="${nic_name:-ens1f3}"
export HCCL_IF_IP="${HCCL_IF_IP:-80.48.5.63}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$nic_name}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-$nic_name}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-$nic_name}"

export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_INTRA_PCIE_ENABLE="${HCCL_INTRA_PCIE_ENABLE:-0}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}"
# export HCCL_CONNECT_TIMEOUT=300
# export HCCL_BUFFSIZE=64

export RAY_DEBUG_POST_MORTEM="${RAY_DEBUG_POST_MORTEM:-0}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}"

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}:${REPO_ROOT}/verl"
export HYDRA_FULL_ERROR=1

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

# ------------------------------------------------------------------------------
# vLLM / CUDA / NPU
# ------------------------------------------------------------------------------
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:1024"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-100000000000}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export VLLM_CONFIGURE_LOGGING="${VLLM_CONFIGURE_LOGGING:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export TRITON_SHIM_LOG_REQUESTS=1
export TRITON_VERBOSE_ROLLOUT_LOGS=1

# ------------------------------------------------------------------------------
# Claude Code sandbox settings
# ------------------------------------------------------------------------------
# The image should already contain Claude Code, Node.js, swerex, and the
# Triton/CANN validation stack.
export TRITON_CLAUDE_IMAGE="${TRITON_CLAUDE_IMAGE:-triton-claude-code-env:latest}"
export TRITON_CLAUDE_MODEL="${TRITON_CLAUDE_MODEL:-uni-agent-actor}"

export TRITON_CLAUDE_MAX_TURNS="${TRITON_CLAUDE_MAX_TURNS:-35}"
export TRITON_CLAUDE_TIME_BUDGET_SEC="${TRITON_CLAUDE_TIME_BUDGET_SEC:-1800}"
export TRITON_CLAUDE_REPAIR_ROUNDS="${TRITON_CLAUDE_REPAIR_ROUNDS:-1}"
export TRITON_EVAL_TIMEOUT="${TRITON_EVAL_TIMEOUT:-600}"
export TRITON_CLAUDE_ARTIFACT_DIR="${TRITON_CLAUDE_ARTIFACT_DIR:-/home/p00938733/claude_code_results}"
export TRITON_CLAUDE_RUN_USER="${TRITON_CLAUDE_RUN_USER:-claude}"
if [[ "${TRITON_CLAUDE_RUN_USER}" == "root" ]]; then
    export TRITON_CLAUDE_RUN_UID="${TRITON_CLAUDE_RUN_UID:-0}"
    export TRITON_CLAUDE_RUN_GID="${TRITON_CLAUDE_RUN_GID:-0}"
    export TRITON_CLAUDE_RUN_HOME="${TRITON_CLAUDE_RUN_HOME:-/root}"
else
    export TRITON_CLAUDE_RUN_UID="${TRITON_CLAUDE_RUN_UID:-1000}"
    export TRITON_CLAUDE_RUN_GID="${TRITON_CLAUDE_RUN_GID:-1000}"
    export TRITON_CLAUDE_RUN_HOME="${TRITON_CLAUDE_RUN_HOME:-/home/${TRITON_CLAUDE_RUN_USER}}"
fi
export TRITON_PROGRESS_RUN_ID="${TRITON_PROGRESS_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export TRITON_PROGRESS_COUNTER_FILE="${TRITON_PROGRESS_COUNTER_FILE:-/tmp/triton_claude_code_progress_${TRITON_PROGRESS_RUN_ID}.jsonl}"
if [[ -z "${TRITON_CLAUDE_EXTRA_ARGS:-}" ]]; then
    export TRITON_CLAUDE_EXTRA_ARGS="--max-turns ${TRITON_CLAUDE_MAX_TURNS}"
fi

# Remote sandbox mode for the A-train/B-eval topology.
#
# Leave TRITON_SANDBOX_DEPLOYMENT=local to start sandboxes on the runner host.
# Set TRITON_SANDBOX_DEPLOYMENT=local_attach and provide either:
#   TRITON_REMOTE_SANDBOX_HOST=http://<B_IP>
#   TRITON_REMOTE_SANDBOX_PORTS=18000,18001,18002,18003
#   TRITON_REMOTE_SANDBOX_AUTH_TOKEN=<token>
# or TRITON_REMOTE_SANDBOX_POOL_JSON='[{"host":"http://<B_IP>","port":18000,"auth_token":"..."}]'.
export TRITON_SANDBOX_DEPLOYMENT="${TRITON_SANDBOX_DEPLOYMENT:-local_attach}"
export TRITON_REMOTE_SANDBOX_HOST="${TRITON_REMOTE_SANDBOX_HOST:-80.48.5.51}"
export TRITON_REMOTE_SANDBOX_PORTS="${TRITON_REMOTE_SANDBOX_PORTS:-18000,18001,18002,18003}"
export TRITON_REMOTE_SANDBOX_AUTH_TOKEN="${TRITON_REMOTE_SANDBOX_AUTH_TOKEN:-tok}"
export TRITON_REMOTE_SANDBOX_AUTH_TOKENS="${TRITON_REMOTE_SANDBOX_AUTH_TOKENS:-${TRITON_REMOTE_SANDBOX_AUTH_TOKEN}}"
export TRITON_REMOTE_SANDBOX_POOL="${TRITON_REMOTE_SANDBOX_POOL:-}"
export TRITON_REMOTE_SANDBOX_POOL_JSON="${TRITON_REMOTE_SANDBOX_POOL_JSON:-}"
export TRITON_REMOTE_SANDBOX_WAIT_TIMEOUT="${TRITON_REMOTE_SANDBOX_WAIT_TIMEOUT:-3600}"

# Optional: reserve a separate NPU pool for operator validation.
export TRITON_DEFAULT_EVAL_DEVICE_IDS="${TRITON_DEFAULT_EVAL_DEVICE_IDS:-0,1,2,3}"
if [[ -z "${TRITON_EVAL_DEVICE_IDS+x}" ]]; then
    if [[ "${TRITON_SANDBOX_DEPLOYMENT}" == "local_attach" ]]; then
        unset TRITON_EVAL_DEVICE_IDS
    else
        export TRITON_EVAL_DEVICE_IDS="${TRITON_DEFAULT_EVAL_DEVICE_IDS}"
    fi
elif [[ -n "${TRITON_EVAL_DEVICE_IDS}" ]]; then
    export TRITON_EVAL_DEVICE_IDS
else
    unset TRITON_EVAL_DEVICE_IDS
fi
if [[ -z "${TRITON_EVAL_DEVICE_COUNT+x}" ]]; then
    if [[ "${TRITON_SANDBOX_DEPLOYMENT}" == "local_attach" ]]; then
        unset TRITON_EVAL_DEVICE_COUNT
    else
        export TRITON_EVAL_DEVICE_COUNT=1
    fi
elif [[ -n "${TRITON_EVAL_DEVICE_COUNT}" ]]; then
    export TRITON_EVAL_DEVICE_COUNT
else
    unset TRITON_EVAL_DEVICE_COUNT
fi

# Address that Docker sandboxes use to reach the host-side Claude shim.
export TRITON_CONTAINER_HOST_ALIAS="${TRITON_CONTAINER_HOST_ALIAS:-80.48.5.63}"
export TRITON_SHIM_PUBLIC_HOST="${TRITON_SHIM_PUBLIC_HOST:-${TRITON_CONTAINER_HOST_ALIAS}}"
export TRITON_SHIM_BIND_HOST="${TRITON_SHIM_BIND_HOST:-0.0.0.0}"
export TRITON_SHIM_REQUEST_TIMEOUT="${TRITON_SHIM_REQUEST_TIMEOUT:-600}"

# ------------------------------------------------------------------------------
# Dataset preprocessing
# ------------------------------------------------------------------------------
export TRITON_KERNELBENCH_DATASET="${TRITON_KERNELBENCH_DATASET:-/home/p00938733/uni-agent-claudecode/examples/triton_agent/benchmarks/NPUKernelBench}"
export TRITON_KERNELBENCH_LEVELS="${TRITON_KERNELBENCH_LEVELS:-level_1}"
# export TRITON_KERNELBENCH_LEVELS=level_1,level_2
export TRITON_KERNELBENCH_PARQUET="${TRITON_KERNELBENCH_PARQUET:-${SCRIPT_DIR}/kernelbench_claude_code.parquet}"
export TRITON_KERNELBENCH_MAX_ROWS="${TRITON_KERNELBENCH_MAX_ROWS:-128}"
export TRITON_KERNELBENCH_VAL_MAX_ROWS="${TRITON_KERNELBENCH_VAL_MAX_ROWS:-16}"
export TRITON_KERNELBENCH_VAL_START="${TRITON_KERNELBENCH_VAL_START:-0}"
export TRITON_KERNELBENCH_ARCH="${TRITON_KERNELBENCH_ARCH:-ascend910b1}"
export TRITON_KERNELBENCH_OPERATOR_BACKEND="${TRITON_KERNELBENCH_OPERATOR_BACKEND:-triton}"

export TRITON_REWARD_TARGET_SPEEDUP="${TRITON_REWARD_TARGET_SPEEDUP:-2.0}"
export TRITON_REWARD_AST_OK="${TRITON_REWARD_AST_OK:-0.0}"
export TRITON_REWARD_COMPILE_OK="${TRITON_REWARD_COMPILE_OK:-0.10}"
export TRITON_REWARD_CORRECTNESS_OK="${TRITON_REWARD_CORRECTNESS_OK:-0.55}"
export TRITON_REWARD_ALL_CORRECT_BONUS="${TRITON_REWARD_ALL_CORRECT_BONUS:-0.10}"
export TRITON_REWARD_SPEEDUP_MAX="${TRITON_REWARD_SPEEDUP_MAX:-0.40}"

export TRITON_KERNELBENCH_FILTER_MODE="${TRITON_KERNELBENCH_FILTER_MODE:-warmup}"
# export TRITON_KERNELBENCH_FILTER_MODE=all
# export TRITON_KERNELBENCH_INCLUDE_KEYWORDS=relu,add,mul,sigmoid,tanh,clamp
# export TRITON_KERNELBENCH_EXCLUDE_KEYWORDS=conv_transpose,attention,lstm
# export TRITON_KERNELBENCH_MAX_CODE_CHARS=10000
# export TRITON_KERNELBENCH_MAX_INPUT_ELEMENTS=67108864
# export TRITON_KERNELBENCH_MAX_OUTPUT_ELEMENTS=67108864
export TRITON_PIPELINE_ERROR_PREVIEW_CHARS="${TRITON_PIPELINE_ERROR_PREVIEW_CHARS:-1000}"

# ------------------------------------------------------------------------------
# Training parameters
# ------------------------------------------------------------------------------
N_GPUS="${N_GPUS:-16}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ROLLOUT_N="${ROLLOUT_N:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
TEST_FREQ="${TEST_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-10}"
ACTOR_LR="${ACTOR_LR:-2e-6}"
PROJECT_NAME="${PROJECT_NAME:-triton-claude-code}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-triton-claude-code}"
logs="${logs:-/home/p00938733/verl-claude-code.log}"

CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config}"
CONFIG_NAME="${CONFIG_NAME:-triton_claude_code_blackbox}"
AGENT_CONFIG_PATH="${AGENT_CONFIG_PATH:-${SCRIPT_DIR}/agent_config_claude_code.yaml}"
TRAIN_FILE="${TRAIN_FILE:-${TRITON_KERNELBENCH_PARQUET}}"
VAL_FILE="${VAL_FILE:-${TRAIN_FILE%.parquet}.val_fixed_s${TRITON_KERNELBENCH_VAL_START}_n${TRITON_KERNELBENCH_VAL_MAX_ROWS}.parquet}"
CKPTS_DIR="${CKPTS_DIR:-./ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
mkdir -p "${CKPTS_DIR}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-36864}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-${LLM_MAX_OUTPUT_TOKENS}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"

AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-8}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
if [[ -z "${MAX_CONCURRENT_SESSIONS:-}" ]]; then
    if [[ -n "${TRITON_REMOTE_SANDBOX_PORTS}" ]]; then
        MAX_CONCURRENT_SESSIONS="$(python3 -c "s='${TRITON_REMOTE_SANDBOX_PORTS}'; print(len([x for x in s.split(',') if x.strip()]) or 1)")"
    else
        MAX_CONCURRENT_SESSIONS=16
    fi
fi
COMPLETION_TIMEOUT="${COMPLETION_TIMEOUT:-${TRITON_CLAUDE_TIME_BUDGET_SEC}}"

# profiling configuration
PROFILE_STEPS="[1]"
PROFILE_RANKS_ALL=False
RANKS="[0]"
DISCRETE=False
# PROFILE_CONTINUOUS_STEPS=True

# profiling NPU options
SAVE_PATH="/home/p00938733/profile_data/all"
LEVEL="level0"
CONTENTS=['npu','cpu','memory']
#CONTENTS=['npu','cpu','memory','module','stack']
ANALYSIS=True

if [[ -z "${TOOL_PARSER:-}" ]]; then
    if [[ "$MODEL_PATH" == *"Qwen3-Coder"* || "$MODEL_PATH" == *"cszhou_sft_weight"* ]]; then
        TOOL_PARSER=qwen3_coder
        EMP_size=8
        ETP_size=1
    else
        TOOL_PARSER=hermes
        EMP_size=1
        ETP_size=1
    fi
fi

echo "=== verl + uni-agent + Claude Code (container-based) ==="
echo "  Model             : ${MODEL_PATH}"
echo "  N_GPUS (trainer)  : ${N_GPUS}"
echo "  Claude image      : ${TRITON_CLAUDE_IMAGE}"
echo "  Max turns         : ${TRITON_CLAUDE_MAX_TURNS}"
echo "  Time budget       : ${TRITON_CLAUDE_TIME_BUDGET_SEC}"
echo "  Tool parser       : ${TOOL_PARSER}"
echo "  Shim public host  : ${TRITON_SHIM_PUBLIC_HOST}"
echo "  Sandbox mode      : ${TRITON_SANDBOX_DEPLOYMENT}"
echo "  Remote sandbox    : ${TRITON_REMOTE_SANDBOX_HOST} ${TRITON_REMOTE_SANDBOX_PORTS}"
echo "  Eval device ids   : ${TRITON_EVAL_DEVICE_IDS:-<remote-sandbox-env>} (count=${TRITON_EVAL_DEVICE_COUNT:-<remote-sandbox-env>})"
echo "  Progress file     : ${TRITON_PROGRESS_COUNTER_FILE}"
echo "  Train parquet     : ${TRAIN_FILE}"

python3 "${SCRIPT_DIR}/prepare_kernelbench_claude_code_data.py" \
    --output-path "${TRAIN_FILE}" \
    --levels "${TRITON_KERNELBENCH_LEVELS}" \
    --max-rows "${TRITON_KERNELBENCH_MAX_ROWS}"

python3 - <<PY
import os
import pandas as pd

src = "${TRAIN_FILE}"
dst = "${VAL_FILE}"
start = max(0, int("${TRITON_KERNELBENCH_VAL_START}"))
n = max(1, int("${TRITON_KERNELBENCH_VAL_MAX_ROWS}"))
df = pd.read_parquet(src)
if start >= len(df):
    start = 0
df.iloc[start:start + n].to_parquet(dst, index=False)
print(f"Wrote {dst}")
PY

echo "Restarting Ray cluster..."
ray stop --force || true
rm -rf /tmp/ray/*
sleep 2
ray start --head \
    --port 6379 \
    --dashboard-host 0.0.0.0 \
    --dashboard-port 8265 \
    --disable-usage-stats \
    --node-ip-address "${MASTER_ADDR}" \
    --object-store-memory=$((32 * 1024 * 1024 * 1024))

ARGS=(
  # =========================
  # algorithm
  # =========================
  algorithm.adv_estimator=grpo
  algorithm.kl_ctrl.kl_coef=0.001

  # =========================
  # data
  # =========================
  "data.train_files=['${TRAIN_FILE}']"
  "data.val_files=['${VAL_FILE}']"
  data.train_batch_size=${BATCH_SIZE}
  data.val_batch_size=${VAL_BATCH_SIZE}
  data.return_raw_chat=True

  # =========================
  # actor_rollout_ref - common
  # =========================
  actor_rollout_ref.hybrid_engine=True
  # actor_rollout_ref.model.use_shm=True
  actor_rollout_ref.model.path=${MODEL_PATH}
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.trust_remote_code=True
  # actor_rollout_ref.model.enable_gradient_checkpointing=True
  # actor_rollout_ref.model.use_fused_kernels=True

  # =========================
  # actor - optimization / PPO
  # =========================
  actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
  actor_rollout_ref.actor.ppo_mini_batch_size=4
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960
  # actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0.0
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.28

  # =========================
  # actor.megatron
  # =========================
  actor_rollout_ref.actor.strategy=megatron
  actor_rollout_ref.actor.megatron.use_mbridge=True
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
  # actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH
  
  actor_rollout_ref.actor.megatron.param_offload=True
  actor_rollout_ref.actor.megatron.grad_offload=True
  actor_rollout_ref.actor.megatron.optimizer_offload=False

  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
  actor_rollout_ref.actor.megatron.context_parallel_size=2
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EMP_size}
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP_size}
  +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=2
  +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
  +actor_rollout_ref.actor.checkpoint.save_contents="['model']"

  # =========================
  # actor optimizer override
  # =========================
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
  # # +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
  +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True

  # =========================
  # ref
  # =========================
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1
  actor_rollout_ref.ref.megatron.context_parallel_size=2
  actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EMP_size}
  actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP_size}
  actor_rollout_ref.ref.megatron.param_offload=True
  actor_rollout_ref.ref.megatron.use_mbridge=True
  actor_rollout_ref.ref.megatron.use_dist_checkpointing=False

  # =========================
  # rollout
  # =========================
  actor_rollout_ref.rollout.tensor_model_parallel_size=4
  actor_rollout_ref.rollout.calculate_log_probs=True
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7
  actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}
  actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}
  actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
  actor_rollout_ref.rollout.max_num_seqs=1
  actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
  actor_rollout_ref.rollout.n=${ROLLOUT_N}
  actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUT_N}
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  ++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=True
  +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=${TOOL_PARSER}
  
  # actor_rollout_ref.rollout.enable_chunked_prefill=False
  # actor_rollout_ref.rollout.enable_prefix_caching=False
  actor_rollout_ref.rollout.free_cache_engine=False
  # +actor_rollout_ref.rollout.engine_kwargs.vllm.swap_space=0
  # +actor_rollout_ref.rollout.engine_kwargs.vllm.cpu_offload_gb=0

  # =========================
  # uni-agent blackbox gateway
  # =========================
  actor_rollout_ref.rollout.multi_turn.enable=True
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=1
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
  actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS}
  actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.trainer.framework.entry.AgentFrameworkRolloutAdapter
  actor_rollout_ref.rollout.custom.agent_framework.framework_class_fqn=examples.triton_agent.framework.TritonClaudeCodeFramework
  actor_rollout_ref.rollout.custom.agent_framework.agent_runner_fqn=examples.triton_agent.claude_code_agent_runner.triton_claude_code_runner
  actor_rollout_ref.rollout.custom.agent_framework.gateway_tool_parser=null
  actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT}
  actor_rollout_ref.rollout.custom.agent_framework.completion_timeout_seconds=${COMPLETION_TIMEOUT}
  actor_rollout_ref.rollout.custom.agent_framework.max_concurrent_sessions=${MAX_CONCURRENT_SESSIONS}
  actor_rollout_ref.rollout.custom.agent_framework.agent_runner_kwargs.agent_config_path=${AGENT_CONFIG_PATH}

  # =========================
  # reward
  # =========================
  reward.custom_reward_function.path=pkg://examples/triton_agent.reward
  reward.custom_reward_function.name=compute_score

  # =========================
  # trainer
  # =========================
  trainer.critic_warmup=0
  #trainer.logger='["console","wandb"]'
  trainer.logger='["console"]'
  trainer.project_name=${PROJECT_NAME}
  trainer.experiment_name=${EXPERIMENT_NAME}
  trainer.val_before_train=${VAL_BEFORE_TRAIN}
  trainer.n_gpus_per_node=${N_GPUS}
  trainer.nnodes=1
  trainer.device=npu
  trainer.save_freq=${SAVE_FREQ}
  trainer.test_freq=${TEST_FREQ}
  trainer.default_hdfs_dir=null
  trainer.default_local_dir=${CKPTS_DIR}
  trainer.total_epochs=100

  # =========================
  # profiler
  # =========================
#   actor_rollout_ref.rollout.profiler.enable=True
#   actor_rollout_ref.rollout.profiler.all_ranks=$PROFILE_RANKS_ALL
#   actor_rollout_ref.rollout.profiler.ranks=$RANKS
#   actor_rollout_ref.rollout.profiler.tool_config.npu.discrete=True
#   actor_rollout_ref.rollout.profiler.tool_config.npu.contents=$CONTENTS
#   actor_rollout_ref.rollout.profiler.tool_config.npu.level=$LEVEL
#   actor_rollout_ref.rollout.profiler.tool_config.npu.analysis=$ANALYSIS
#   actor_rollout_ref.ref.profiler.enable=True
#   actor_rollout_ref.ref.profiler.all_ranks=$PROFILE_RANKS_ALL
#   actor_rollout_ref.ref.profiler.ranks=$RANKS
#   actor_rollout_ref.ref.profiler.tool_config.npu.discrete=$DISCRETE
#   actor_rollout_ref.ref.profiler.tool_config.npu.contents=$CONTENTS
#   actor_rollout_ref.ref.profiler.tool_config.npu.level=$LEVEL
#   actor_rollout_ref.ref.profiler.tool_config.npu.analysis=$ANALYSIS
#   actor_rollout_ref.actor.profiler.enable=True
#   actor_rollout_ref.actor.profiler.all_ranks=$PROFILE_RANKS_ALL
#   actor_rollout_ref.actor.profiler.ranks=$RANKS
#   actor_rollout_ref.actor.profiler.tool_config.npu.discrete=$DISCRETE
#   actor_rollout_ref.actor.profiler.tool_config.npu.contents=$CONTENTS
#   actor_rollout_ref.actor.profiler.tool_config.npu.level=$LEVEL
#   actor_rollout_ref.actor.profiler.tool_config.npu.analysis=$ANALYSIS
#   global_profiler.tool=npu
#   global_profiler.steps=$PROFILE_STEPS
#   global_profiler.save_path=$SAVE_PATH
)

RUNTIME_ENV_JSON="$(python3 - <<PY
import json
import os

env_vars = {
    "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    "PYTHONUNBUFFERED": os.environ.get("PYTHONUNBUFFERED", "1"),
    "TRITON_PROGRESS_RUN_ID": os.environ["TRITON_PROGRESS_RUN_ID"],
    "TRITON_PROGRESS_COUNTER_FILE": os.environ["TRITON_PROGRESS_COUNTER_FILE"],
    "TRITON_VERBOSE_ROLLOUT_LOGS": os.environ.get("TRITON_VERBOSE_ROLLOUT_LOGS", "0"),
    "TRITON_SHIM_LOG_REQUESTS": os.environ.get("TRITON_SHIM_LOG_REQUESTS", "0"),
    "TRITON_CLAUDE_HEARTBEAT_SEC": os.environ.get("TRITON_CLAUDE_HEARTBEAT_SEC", "0"),
    "TRITON_PROGRESS_STDOUT": os.environ.get("TRITON_PROGRESS_STDOUT", "1"),
}
for name in (
    "TRITON_SHIM_PUBLIC_HOST",
    "TRITON_SHIM_BIND_HOST",
    "TRITON_SHIM_REQUEST_TIMEOUT",
    "TRITON_CLAUDE_MODEL",
    "TRITON_CLAUDE_TIME_BUDGET_SEC",
    "TRITON_CLAUDE_EXTRA_ARGS",
    "TRITON_CLAUDE_ARTIFACT_DIR",
    "TRITON_CLAUDE_SKIP_INSTALL",
    "TRITON_CLAUDE_NODE_TARBALL",
    "TRITON_CLAUDE_CODE_TARBALL",
    "TRITON_CLAUDE_RUN_USER",
    "TRITON_CLAUDE_RUN_UID",
    "TRITON_CLAUDE_RUN_GID",
    "TRITON_CLAUDE_RUN_HOME",
):
    if name in os.environ:
        env_vars[name] = os.environ[name]
print(json.dumps({"env_vars": env_vars}))
PY
)"

ray job submit \
  --address="http://${MASTER_ADDR}:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- \
  python3 -m verl.trainer.main_ppo_sync \
    --config-name="${CONFIG_NAME}" \
    --config-path="${CONFIG_PATH}" \
    "${ARGS[@]}" \
    "$@" \
  2>&1 | tee -i "${logs}"
