#!/usr/bin/env bash
# =============================================================================
# Concurrent sandbox verification launcher (per-card dynamic op dispatch).
#
# - Each NPU card binds to one swerex sandbox (own attach port + workspace +
#   log + result file). With 8 cards, 8 ops run concurrently, each in its own
#   isolated sandbox.
# - Ops are fed dynamically: a worker grabs the next pending op from a shared
#   queue (file-locked), runs it, then loops for the next one until the queue
#   is empty. This is true "free card -> next op" dynamic scheduling, not a
#   static round-robin split.
# - File/log isolation: every card writes to card{i}.log and
#   results.card{i}.jsonl; each op's artifacts go to artifacts/<op_name>/.
# - Resume: re-running with RESUME_DIR pointing at a previous run dir skips
#   ops already present in that run's merged results.
# =============================================================================
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/Uniagent-triton-ascend-compact}"
DATASET="${DATASET:-/home/ascendc-kernelgen-data/npu_benchmark}"
OUT_ROOT="${OUT_ROOT:-/home/Uniagent-triton-ascend-compact/eval/output}"

# --- vLLM server (shared by all workers) -------------------------------------
VLLM_HOST="${VLLM_HOST:-192.169.0.176}"
PORT="${PORT:-7777}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-triton-synth}"

# --- Task selection ----------------------------------------------------------
MAX_ROWS="${MAX_ROWS:-1024}"
FILTER_MODE="${FILTER_MODE:-all}"
NUMERIC_SORT="${NUMERIC_SORT:-1}"
LEVELS="${LEVELS:-level3,level4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RESUME_DIR="${RESUME_DIR:-}"

# --- Agent interaction (eval overrides training defaults) --------------------
# Training agent_config_synth.yaml uses action_timeout=1800; eval often needs a
# larger per-tool budget (slow compile/verify). Override at launch time, e.g.:
#   ACTION_TIMEOUT=7200 ./eval/run_synth_levels_concurrent.sh
ACTION_TIMEOUT="${ACTION_TIMEOUT:-3600}"
MAX_TURNS="${MAX_TURNS:-150}"

# --- Concurrency: cards / sandboxes ------------------------------------------
# Comma-separated NPU device ids, e.g. "0,1,2,3,4,5,6,7". Defaults to 8 cards.
EVAL_DEVICE_IDS="${EVAL_DEVICE_IDS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
# Base swerex attach port; card i uses PORT_BASE + i.
TRITON_ATTACH_PORT_BASE="${TRITON_ATTACH_PORT_BASE:-18000}"
# Base workspace dir; card i uses /opt/workspace_card{i}/agent_workdir.
TRITON_WORKSPACE_DIR_BASE="${TRITON_WORKSPACE_DIR_BASE:-/opt/workspace_card}"
# Single auth token reused across all swerex servers on this host.
TRITON_ATTACH_AUTH_TOKEN="${TRITON_ATTACH_AUTH_TOKEN:-mytoken1}"

# --- Containers --------------------------------------------------------------
SYNTH_CONTAINER="${SYNTH_CONTAINER:-uni-agent}"
SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-cc}"
START_SANDBOX="${START_SANDBOX:-1}"   # set 0 if swerex servers are already up
SANDBOX_LOG_DIR_IN_CONTAINER="${SANDBOX_LOG_DIR_IN_CONTAINER:-/tmp}"
TRITON_KERNELBENCH_ARCH="${TRITON_KERNELBENCH_ARCH:-ascend910b1}"

TRITON_ATTACH_HOST="${TRITON_ATTACH_HOST:-http://127.0.0.1}"

export no_proxy="${no_proxy:-127.0.0.1,localhost,0.0.0.0,192.169.0.176}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"
export TRITON_TRAIN_BEST_FIRST="${TRITON_TRAIN_BEST_FIRST:-1}"
# -----------------------------------------------------------------------------
# Parse device list
# -----------------------------------------------------------------------------
IFS=',' read -ra DEVICE_ARRAY <<< "${EVAL_DEVICE_IDS}"
NUM_CARDS="${#DEVICE_ARRAY[@]}"
if [[ "${NUM_CARDS}" -eq 0 ]]; then
  echo "[concurrent] ERROR: EVAL_DEVICE_IDS is empty"; exit 2
fi
echo "[concurrent] cards=${NUM_CARDS} devices=[${EVAL_DEVICE_IDS}]"

port_for_idx() {
  # idx (0-based) -> attach port
  echo $(( TRITON_ATTACH_PORT_BASE + $1 ))
}
workspace_for_idx() {
  # idx -> workspace dir inside sandbox container
  echo "${TRITON_WORKSPACE_DIR_BASE}${1}/agent_workdir"
}

# -----------------------------------------------------------------------------
# Output directories (resume-aware)
# -----------------------------------------------------------------------------
if [[ -n "${RESUME_DIR}" && -d "${RESUME_DIR}" ]]; then
  RUN_DIR="${RESUME_DIR}"
  echo "[concurrent] resume mode: reusing ${RUN_DIR}"
else
  RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
  RUN_DIR="${OUT_ROOT}/synth_levels_concurrent_${RUN_ID}"
fi
LOG_DIR="${RUN_DIR}/logs"
RESULT_DIR="${RUN_DIR}/results"
ARTIFACT_DIR="${RUN_DIR}/artifacts"
QUEUE_FILE="${RUN_DIR}/op_queue.txt"
DONE_FILE="${RUN_DIR}/op_done.txt"
LOCK_FILE="${RUN_DIR}/dispatch.lock"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}" "${ARTIFACT_DIR}"

# -----------------------------------------------------------------------------
# Enumerate ops for the requested levels (inside the synth container, which has
# the dataset + repo). We reuse load_tasks by printing op_names one per line.
# -----------------------------------------------------------------------------
enumerate_ops() {
  docker exec "${SYNTH_CONTAINER}" bash -lc "
    cd '${REPO_ROOT}' &&
    PYTHONUNBUFFERED=1 \
    python3 - <<'PY'
import os, sys
sys.path.insert(0, os.path.abspath('.'))
# Redirect load_tasks' own logging to stderr so only op_names reach stdout.
from contextlib import redirect_stdout
import io
from examples.triton_agent.synth_common import load_tasks
buf = io.StringIO()
with redirect_stdout(buf):
    tasks = load_tasks(
        dataset_path='${DATASET}',
        levels='${LEVELS}',
        max_rows=${MAX_ROWS:-'None'},
        filter_mode='${FILTER_MODE}',
    )
sys.stderr.write(buf.getvalue())
for t in tasks:
    print(t['op_name'])
PY
  "
}

echo "[concurrent] enumerating ops for levels=${LEVELS} ..."
mapfile -t OP_LIST < <(enumerate_ops)
# Defensive: drop empty lines and anything that doesn't look like an op name
# (op names are kernelbench_l{n}_... or task_..., never log lines).
cleaned=()
for line in "${OP_LIST[@]}"; do
  [[ -z "${line}" ]] && continue
  case "${line}" in
    kernelbench_*|task_*) cleaned+=("${line}") ;;
    *) echo "[concurrent] WARNING: dropping non-op line from queue: ${line}" >&2 ;;
  esac
done
OP_LIST=("${cleaned[@]}")
TOTAL_OPS="${#OP_LIST[@]}"
echo "[concurrent] total ops=${TOTAL_OPS}"

if [[ "${TOTAL_OPS}" -eq 0 ]]; then
  echo "[concurrent] no ops to run. abort."; exit 1
fi

# -----------------------------------------------------------------------------
# Build the pending queue: all ops minus already-done (resume support)
# -----------------------------------------------------------------------------
DONE_OPS_SET=()
if [[ -f "${DONE_FILE}" ]]; then
  mapfile -t DONE_OPS_SET < <(grep -v '^[[:space:]]*$' "${DONE_FILE}" || true)
fi

> "${QUEUE_FILE}"
skipped=0
for op in "${OP_LIST[@]}"; do
  if [[ -n "${DONE_OPS_SET[*]}" ]] && printf '%s\n' "${DONE_OPS_SET[@]}" | grep -qxF "${op}"; then
    skipped=$((skipped + 1)); continue
  fi
  echo "${op}" >> "${QUEUE_FILE}"
done
PENDING=$((TOTAL_OPS - skipped))
echo "[concurrent] pending=${PENDING} skipped(done)=${skipped}"

if [[ "${PENDING}" -eq 0 ]]; then
  echo "[concurrent] all ops already done. nothing to do."; exit 0
fi

: > "${DONE_FILE}"  # ensure exists (append-mode below)
touch "${LOCK_FILE}"

# -----------------------------------------------------------------------------
# Start one swerex sandbox server per card (inside the sandbox container).
# -----------------------------------------------------------------------------
start_sandbox_server() {
  local idx="$1" port="$2"
  local card="${DEVICE_ARRAY[$idx]}"
  local log="${SANDBOX_LOG_DIR_IN_CONTAINER}/swerex_card${card}_${port}.log"

  if [[ "${START_SANDBOX}" != "1" ]]; then
    echo "[concurrent] skip sandbox start idx=${idx} card=${card} (START_SANDBOX=0)"
    return 0
  fi
  echo "[concurrent] start swerex idx=${idx} card=${card} port=${port}"
  docker exec -d "${SANDBOX_CONTAINER}" bash -lc \
    "python -u -m swerex.server --host 0.0.0.0 --port ${port} --auth-token ${TRITON_ATTACH_AUTH_TOKEN} > ${log} 2>&1"
}

for ((i=0; i<NUM_CARDS; i++)); do
  start_sandbox_server "${i}" "$(port_for_idx "${i}")"
done

# Give swerex a moment to bind.
sleep 60

# -----------------------------------------------------------------------------
# Worker: loops pulling the next op from the shared queue (file-locked) and
# running it on this card's sandbox. Returns when the queue is empty.
# -----------------------------------------------------------------------------
claim_next_op() {
  # Atomically pop the first line from QUEUE_FILE. Returns op name on stdout,
  # or empty string if the queue is empty. Uses flock on LOCK_FILE.
  local op=""
  {
    flock -x 200
    op="$(head -n1 "${QUEUE_FILE}" 2>/dev/null || true)"
    if [[ -n "${op}" ]]; then
      # drop the first line
      tail -n +2 "${QUEUE_FILE}" > "${QUEUE_FILE}.tmp" && mv "${QUEUE_FILE}.tmp" "${QUEUE_FILE}"
    fi
  } 200>"${LOCK_FILE}"
  echo "${op}"
}

mark_done() {
  local op="$1"
  {
    flock -x 200
    echo "${op}" >> "${DONE_FILE}"
  } 200>"${LOCK_FILE}"
}

run_worker() {
  local idx="$1"
  local card="${DEVICE_ARRAY[$idx]}"
  local port; port="$(port_for_idx "${idx}")"
  local ws; ws="$(workspace_for_idx "${idx}")"
  # Per-card tool install dir: avoids concurrent uploads racing on the shared
  # /usr/local/bin when 8 swerex servers live in one sandbox container.
  local tool_dir="/opt/uni-agent-tools/card${idx}"
  local log_file="${LOG_DIR}/card${idx}.log"
  local result_file="${RESULT_DIR}/results.card${idx}.jsonl"
  local numeric_arg=""
  [[ "${NUMERIC_SORT}" == "1" ]] && numeric_arg="--numeric-sort"

  echo "[worker idx=${idx} card=${card}] online port=${port} ws=${ws} tools=${tool_dir}" | tee -a "${log_file}"

  while :; do
    local op; op="$(claim_next_op)"
    if [[ -z "${op}" ]]; then
      echo "[worker idx=${idx} card=${card}] queue empty, exiting" | tee -a "${log_file}"
      break
    fi

    local remaining; remaining="$(grep -c . "${QUEUE_FILE}" 2>/dev/null || echo 0)"
    echo "[worker idx=${idx} card=${card}] START op=${op} remaining_after=${remaining}" | tee -a "${log_file}"

    # Per-op isolated artifact dir.
    local op_art="${ARTIFACT_DIR}/${op}"
    mkdir -p "${op_art}"

    # Run exactly this one op via --op-names. Output appended to the card's
    # result file; full stdout/stderr appended to the card's log.
    if ! docker exec "${SYNTH_CONTAINER}" bash -lc "
      cd '${REPO_ROOT}' &&
      PYTHONUNBUFFERED=1 \
      TRITON_ATTACH_HOST='${TRITON_ATTACH_HOST}' \
      TRITON_ATTACH_PORT='${port}' \
      TRITON_ATTACH_AUTH_TOKEN='${TRITON_ATTACH_AUTH_TOKEN}' \
      TRITON_WORKSPACE_DIR='${ws}' \
      TRITON_TOOL_INSTALL_DIR='${tool_dir}' \
      TRITON_KERNELBENCH_ARCH='${TRITON_KERNELBENCH_ARCH}' \
      no_proxy='${no_proxy}' \
      NO_PROXY='${NO_PROXY}' \
      TRITON_REWARD_TARGET_SPEEDUP='${TRITON_REWARD_TARGET_SPEEDUP:-2.0}' \
      TRITON_REWARD_AST_OK='${TRITON_REWARD_AST_OK:-0.02}' \
      TRITON_REWARD_COMPILE_OK='${TRITON_REWARD_COMPILE_OK:-0.15}' \
      TRITON_REWARD_CORRECTNESS_OK='${TRITON_REWARD_CORRECTNESS_OK:-0.60}' \
      TRITON_REWARD_ALL_CORRECT_BONUS='${TRITON_REWARD_ALL_CORRECT_BONUS:-0.13}' \
      TRITON_REWARD_SPEEDUP_MAX='${TRITON_REWARD_SPEEDUP_MAX:-0.40}' \
      TRITON_TRAIN_BEST_FIRST='${TRITON_TRAIN_BEST_FIRST:-1}' \
      TRITON_COMPACTION_ENABLED='${TRITON_COMPACTION_ENABLED:-1}' \
      TRITON_COMPACTION_THRESHOLD='${TRITON_COMPACTION_THRESHOLD:-0.3}' \
      TRITON_COMPACTION_MAX_CONTEXT='${TRITON_COMPACTION_MAX_CONTEXT:-128000}' \
      python3 examples/triton_agent/synth_triton_local.py \
        --dataset '${DATASET}' \
        --levels '${LEVELS}' \
        --max-rows '${MAX_ROWS}' \
        --filter-mode '${FILTER_MODE}' \
        --op-names '${op}' \
        --port '${PORT}' \
        --host '${VLLM_HOST}' \
        --served-model-name '${SERVED_MODEL_NAME}' \
        --eval-device-ids '${card}' \
        --output '${result_file}' \
        --artifacts-dir '${ARTIFACT_DIR}' \
        --action-timeout '${ACTION_TIMEOUT}' \
        --max-turns '${MAX_TURNS}' \
        ${numeric_arg} \
        ${EXTRA_ARGS}
    " >> "${log_file}" 2>&1; then
      echo "[worker idx=${idx} card=${card}] FAIL op=${op} (non-zero exit)" | tee -a "${log_file}"
      # Still mark done so we don't loop forever on a broken op.
    fi

    mark_done "${op}"
    echo "[worker idx=${idx} card=${card}] DONE op=${op}" | tee -a "${log_file}"
  done
}

# -----------------------------------------------------------------------------
# Launch workers (one per card) and wait.
# -----------------------------------------------------------------------------
echo "[concurrent] dataset=${DATASET}"
echo "[concurrent] action_timeout=${ACTION_TIMEOUT}s max_turns=${MAX_TURNS}"
echo "[concurrent] TRITON_TRAIN_BEST_FIRST=${TRITON_TRAIN_BEST_FIRST}"
echo "[concurrent] run_dir=${RUN_DIR}"
echo "[concurrent] logs=${LOG_DIR}"
echo "[concurrent] results=${RESULT_DIR}"
echo "[concurrent] artifacts=${ARTIFACT_DIR}"
echo "[concurrent] synth_container=${SYNTH_CONTAINER} sandbox_container=${SANDBOX_CONTAINER}"
echo "[concurrent] ports=$(for ((i=0;i<NUM_CARDS;i++)); do printf '%s ' "$(port_for_idx "${i}")"; done)"
echo "[concurrent] workspaces=$(for ((i=0;i<NUM_CARDS;i++)); do printf '%s ' "$(workspace_for_idx "${i}")"; done)"
echo "[concurrent] queue=${QUEUE_FILE} done=${DONE_FILE}"
echo "[concurrent] tail logs: tail -f ${LOG_DIR}/card*.log"

PIDS=()
for ((i=0; i<NUM_CARDS; i++)); do
  run_worker "${i}" &
  PIDS+=("$!")
done

echo "[concurrent] worker pids: ${PIDS[*]}"

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

# -----------------------------------------------------------------------------
# Merge per-card results into one combined file for convenience.
# -----------------------------------------------------------------------------
MERGED="${RESULT_DIR}/results.all.jsonl"
: > "${MERGED}"
for ((i=0; i<NUM_CARDS; i++)); do
  rf="${RESULT_DIR}/results.card${i}.jsonl"
  [[ -f "${rf}" ]] && cat "${rf}" >> "${MERGED}"
done

echo "[concurrent] done status=${status}"
echo "[concurrent] results(per-card) in ${RESULT_DIR}/results.card{0..$((NUM_CARDS-1))}.jsonl"
echo "[concurrent] results(merged)  in ${MERGED}"
echo "[concurrent] logs in ${LOG_DIR}"
echo "[concurrent] done ops: $(grep -c . "${DONE_FILE}" 2>/dev/null || echo 0)/${TOTAL_OPS}"

exit "${status}"