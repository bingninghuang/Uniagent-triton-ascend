#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"
cd "${ROOT}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
_TORCH_NPU_WARNING_FILTERS="ignore::UserWarning:torch_npu.utils.collect_env,ignore::UserWarning:torch_npu.utils._path_manager"
if [[ -n "${PYTHONWARNINGS:-}" ]]; then
    export PYTHONWARNINGS="${_TORCH_NPU_WARNING_FILTERS},${PYTHONWARNINGS}"
else
    export PYTHONWARNINGS="${_TORCH_NPU_WARNING_FILTERS}"
fi

CALLER_UID="$(id -u)"
CALLER_GID="$(id -g)"
ACTIVE_COMMAND_PGID=""

command_group_alive() {
    local pgid="$1"
    kill -0 -- "-${pgid}" 2>/dev/null && return 0
    [[ "$(id -u)" != "0" && -x /usr/bin/sudo ]] \
        && sudo -n kill -0 -- "-${pgid}" 2>/dev/null
}

signal_command_group() {
    local signal="$1"
    local pgid="$2"
    kill "-${signal}" -- "-${pgid}" 2>/dev/null && return 0
    if [[ "$(id -u)" != "0" && -x /usr/bin/sudo ]]; then
        sudo -n kill "-${signal}" -- "-${pgid}" 2>/dev/null || true
    fi
}

cleanup_active_command_group() {
    local pgid="${ACTIVE_COMMAND_PGID:-}"
    [[ "${pgid}" =~ ^[0-9]+$ ]] || return 0
    ACTIVE_COMMAND_PGID=""
    command_group_alive "${pgid}" || return 0

    signal_command_group TERM "${pgid}"
    for _ in {1..30}; do
        command_group_alive "${pgid}" || return 0
        sleep 0.1
    done
    signal_command_group KILL "${pgid}"
}

handle_signal() {
    local status="$1"
    cleanup_active_command_group
    exit "${status}"
}

trap cleanup_active_command_group EXIT
trap 'handle_signal 143' TERM HUP
trap 'handle_signal 130' INT

first_csv_value() {
    local value="${1:-}"
    value="${value%%,*}"
    printf '%s' "${value}"
}

run_cmd=("$@")
if [[ "${#run_cmd[@]}" -eq 0 ]]; then
    echo "usage: tools/run_npu_command.sh <command> [args...]" >&2
    exit 2
fi

restore_artifact_permissions() {
    if [[ "$(id -u)" == "0" || ! -x /usr/bin/sudo ]]; then
        return 0
    fi
    local targets=()
    for path in output/verify output/perf_result.json perf_result.json profiling_results.json summary.json; do
        if [[ -e "${path}" ]]; then
            targets+=("${path}")
        fi
    done
    if [[ "${#targets[@]}" -gt 0 ]]; then
        sudo chown -R "${CALLER_UID}:${CALLER_GID}" "${targets[@]}" 2>/dev/null || true
        sudo chmod -R u+rwX "${targets[@]}" 2>/dev/null || true
    fi
}

if [[ "${#run_cmd[@]}" -ge 2 ]] \
    && [[ "${run_cmd[0]}" == "python" || "${run_cmd[0]}" == "python3" || "${run_cmd[0]}" == */python || "${run_cmd[0]}" == */python3 ]] \
    && [[ "${run_cmd[1]}" == */verify.py || "${run_cmd[1]}" == "verify.py" || "${run_cmd[1]}" == */benchmark.py || "${run_cmd[1]}" == "benchmark.py" ]] \
    && [[ -n "${OPERATOR_PYTHON:-}" && -x "${OPERATOR_PYTHON}" ]]; then
    echo "[env.sh] using OPERATOR_PYTHON=${OPERATOR_PYTHON} for ${run_cmd[1]} instead of ${run_cmd[0]}" >&2
    run_cmd[0]="${OPERATOR_PYTHON}"
fi

is_verify_command=0
is_snapshot_command=0
pipeline_script=""
verify_output=""
verify_op_name=""
verify_dir="output/verify"
verify_impl_name="triton_ascend_impl"
for ((i = 0; i < ${#run_cmd[@]}; i++)); do
    case "${run_cmd[$i]}" in
        */verify.py|verify.py)
            is_verify_command=1
            is_snapshot_command=1
            pipeline_script="verify"
            ;;
        */benchmark.py|benchmark.py)
            is_snapshot_command=1
            pipeline_script="benchmark"
            ;;
        --op_name)
            if (( i + 1 < ${#run_cmd[@]} )); then
                verify_op_name="${run_cmd[$((i + 1))]}"
            fi
            ;;
        --op_name=*)
            verify_op_name="${run_cmd[$i]#--op_name=}"
            ;;
        --verify_dir)
            if (( i + 1 < ${#run_cmd[@]} )); then
                verify_dir="${run_cmd[$((i + 1))]}"
            fi
            ;;
        --verify_dir=*)
            verify_dir="${run_cmd[$i]#--verify_dir=}"
            ;;
        --triton_impl_name)
            if (( i + 1 < ${#run_cmd[@]} )); then
                verify_impl_name="${run_cmd[$((i + 1))]}"
            fi
            ;;
        --triton_impl_name=*)
            verify_impl_name="${run_cmd[$i]#--triton_impl_name=}"
            ;;
        --output)
            if [[ "${pipeline_script}" == "verify" ]] && (( i + 1 < ${#run_cmd[@]} )); then
                verify_output="${run_cmd[$((i + 1))]}"
            fi
            ;;
        --output=*)
            if [[ "${pipeline_script}" == "verify" ]]; then
                verify_output="${run_cmd[$i]#--output=}"
            fi
            ;;
    esac
done

verify_timing_file=".triton_verify_timing.jsonl"

now_ms() {
    date +%s%3N
}

record_verify_timing() {
    [[ "${is_verify_command}" == "1" ]] || return 0
    local lock_wait_ms="$1"
    local execute_ms="$2"
    local status="$3"
    local device_id="${4:-}"
    mkdir -p "$(dirname "${verify_timing_file}")" 2>/dev/null || true
    printf '{"lock_wait_ms":%s,"execute_ms":%s,"status":%s,"device_id":"%s"}\n' \
        "${lock_wait_ms}" "${execute_ms}" "${status}" "${device_id}" >>"${verify_timing_file}"
}

run_isolated() {
    if ! command -v setsid >/dev/null 2>&1; then
        "$@"
        return
    fi

    local pid status=0
    setsid "$@" &
    pid=$!
    ACTIVE_COMMAND_PGID="${pid}"
    wait "${pid}" || status=$?
    cleanup_active_command_group
    return "${status}"
}

run_direct() {
    local -a direct_cmd
    if [[ "$(id -u)" != "0" && -x /usr/bin/sudo ]]; then
        direct_cmd=(
            sudo -H -E env
            HOME=/root
            "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-}"
            "ALLOCATED_DEVICE_ID=${ALLOCATED_DEVICE_ID:-}"
            "EVAL_DEVICE_IDS=${EVAL_DEVICE_IDS:-}"
            "EVAL_DEVICE_COUNT=${EVAL_DEVICE_COUNT:-}"
            "EVAL_ENV_NAME=${EVAL_ENV_NAME:-}"
            "OPERATOR_PYTHON=${OPERATOR_PYTHON}"
            "AST_CHECK_PYTHON=${AST_CHECK_PYTHON}"
            "WORKSPACE_BASE=${WORKSPACE_BASE}"
            "PATH=${PATH}"
            "PYTHONPATH=${PYTHONPATH:-}"
            "PYTHONWARNINGS=${PYTHONWARNINGS:-}"
            "PYTHONIOENCODING=${PYTHONIOENCODING:-utf-8}"
            "LANG=${LANG:-C.UTF-8}"
            "LC_ALL=${LC_ALL:-C.UTF-8}"
            "TRITON_REWARD_TARGET_SPEEDUP=${TRITON_REWARD_TARGET_SPEEDUP:-}"
            "TRITON_REWARD_AST_OK=${TRITON_REWARD_AST_OK:-}"
            "TRITON_REWARD_COMPILE_OK=${TRITON_REWARD_COMPILE_OK:-}"
            "TRITON_REWARD_CORRECTNESS_OK=${TRITON_REWARD_CORRECTNESS_OK:-}"
            "TRITON_REWARD_ALL_CORRECT_BONUS=${TRITON_REWARD_ALL_CORRECT_BONUS:-}"
            "TRITON_REWARD_SPEEDUP_MAX=${TRITON_REWARD_SPEEDUP_MAX:-}"
            "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
            "PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE}"
            "${run_cmd[@]}"
        )
    else
        direct_cmd=("${run_cmd[@]}")
    fi

    if [[ "${is_verify_command}" == "1" && "${TRITON_VERIFY_COMPACT_OUTPUT:-1}" != "0" ]]; then
        local raw_log raw_dir status
        if [[ -n "${verify_output}" ]]; then
            raw_log="${verify_output%.*}.raw.log"
        else
            raw_log="output/verify/verify_raw.log"
        fi
        raw_dir="$(dirname "${raw_log}")"
        mkdir -p "${raw_dir}" 2>/dev/null || true
        set +e
        run_isolated "${direct_cmd[@]}" >"${raw_log}" 2>&1
        status=$?
        set -e
        echo "[verifier-log] raw output saved to ${raw_log}" >&2
        return "${status}"
    fi

    run_isolated "${direct_cmd[@]}"
}

run_with_device() {
    local device_id="$1"
    export ASCEND_RT_VISIBLE_DEVICES="${device_id}"
    export ALLOCATED_DEVICE_ID="${device_id}"
    run_direct
}

acquire_and_run() {
    local lock_dir="${EVAL_LOCK_DIR:-/shared/device-locks}"
    local device_prefix="${EVAL_DEVICE_PREFIX:-npu}"
    local retry_interval="${EVAL_RETRY_INTERVAL:-1.0}"
    local timeout="${EVAL_TIMEOUT:-}"
    local start_ts wait_started_ms now elapsed
    start_ts="$(date +%s)"
    wait_started_ms="$(now_ms)"
    mkdir -p "${lock_dir}" 2>/dev/null || true

    local probe_file="${lock_dir}/.write-test.$$"
    if ! ( : > "${probe_file}" ) 2>/dev/null; then
        echo "[npu-lock] FAILED: cannot write lock directory ${lock_dir}. Check TRITON_EVAL_LOCK_DIR permissions." >&2
        return 125
    fi
    rm -f "${probe_file}" 2>/dev/null || true

    local devices=()
    IFS=',' read -r -a devices <<< "${EVAL_DEVICE_IDS}"
    while true; do
        local device_id lock_file lock_fd
        for device_id in "${devices[@]}"; do
            device_id="$(first_csv_value "${device_id}")"
            device_id="${device_id//[[:space:]]/}"
            [[ -n "${device_id}" ]] || continue
            lock_file="${lock_dir}/${device_prefix}${device_id}.lock"
            exec {lock_fd}>"${lock_file}" || continue
            if flock -n "${lock_fd}"; then
                local acquired_ms execute_finished_ms
                acquired_ms="$(now_ms)"
                if [[ "${EVAL_VERBOSE:-0}" == "1" || "${EVAL_VERBOSE:-}" == "true" || "${EVAL_VERBOSE:-}" == "True" ]]; then
                    echo "[npu-lock] acquired ${device_prefix}${device_id} (${lock_file})" >&2
                fi
                run_with_device "${device_id}"
                local status=$?
                execute_finished_ms="$(now_ms)"
                record_verify_timing \
                    "$((acquired_ms - wait_started_ms))" \
                    "$((execute_finished_ms - acquired_ms))" \
                    "${status}" \
                    "${device_id}"
                flock -u "${lock_fd}" || true
                exec {lock_fd}>&-
                return "${status}"
            fi
            exec {lock_fd}>&-
        done

        if [[ -n "${timeout}" && "${timeout}" != "None" && "${timeout}" != "none" ]]; then
            now="$(date +%s)"
            elapsed=$((now - start_ts))
            if (( elapsed >= ${timeout%.*} )); then
                local timed_out_ms
                timed_out_ms="$(now_ms)"
                echo "[npu-lock] timed out waiting for one of EVAL_DEVICE_IDS=${EVAL_DEVICE_IDS}" >&2
                record_verify_timing "$((timed_out_ms - wait_started_ms))" 0 124
                return 124
            fi
        fi
        sleep "${retry_interval}"
    done
}

run_status=0
if [[ "${EVAL_USE_DEVICE_LOCK:-1}" != "0" && -n "${EVAL_DEVICE_IDS:-}" && -x "$(command -v flock 2>/dev/null || true)" ]]; then
    acquire_and_run || run_status=$?
else
    if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
        if [[ -n "${ALLOCATED_DEVICE_ID:-}" ]]; then
            export ASCEND_RT_VISIBLE_DEVICES="$(first_csv_value "${ALLOCATED_DEVICE_ID}")"
        elif [[ -n "${EVAL_DEVICE_IDS:-}" ]]; then
            export ASCEND_RT_VISIBLE_DEVICES="$(first_csv_value "${EVAL_DEVICE_IDS}")"
        fi
    fi
    if [[ -z "${ALLOCATED_DEVICE_ID:-}" && -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
        export ALLOCATED_DEVICE_ID="$(first_csv_value "${ASCEND_RT_VISIBLE_DEVICES}")"
    fi
    execute_started_ms="$(now_ms)"
    run_direct || run_status=$?
    execute_finished_ms="$(now_ms)"
    record_verify_timing 0 "$((execute_finished_ms - execute_started_ms))" "${run_status}" "${ASCEND_RT_VISIBLE_DEVICES:-}"
fi

restore_artifact_permissions

if [[ "${is_verify_command}" == "1" ]]; then
    summary_output=""
    if [[ -n "${verify_output}" ]]; then
        summary_output="${verify_output%.*}_summary.json"
    fi
    if [[ -x "${SCRIPT_DIR}/summarize_verify_result.py" || -f "${SCRIPT_DIR}/summarize_verify_result.py" ]]; then
        if [[ -n "${summary_output}" ]]; then
            python3 "${SCRIPT_DIR}/summarize_verify_result.py" "${verify_output}" --exit-code "${run_status}" --write-json "${summary_output}" || true
        else
            python3 "${SCRIPT_DIR}/summarize_verify_result.py" "${verify_output}" --exit-code "${run_status}" || true
        fi
    else
        echo "[verifier-summary] FAILED: summarizer is missing. Do not claim success." >&2
    fi
fi

if [[ "${is_snapshot_command}" == "1" ]]; then
    if [[ -z "${verify_output}" ]]; then
        verify_output="${verify_dir%/}/verify_result.json"
    fi
    if [[ "${TRITON_VERIFY_BEST_SNAPSHOT:-1}" != "0" && -n "${verify_op_name}" \
        && -f "${SCRIPT_DIR}/snapshot_verify_best.py" ]]; then
        summary_for_snapshot="${verify_output%.*}_summary.json"
        snapshot_cmd=(
            python3 "${SCRIPT_DIR}/snapshot_verify_best.py"
            --op-name "${verify_op_name}"
            --workspace-dir "${ROOT}"
            --verify-dir "${verify_dir}"
            --verify-result "${verify_output}"
            --summary "${summary_for_snapshot}"
            --triton-impl-name "${verify_impl_name}"
        )
        "${snapshot_cmd[@]}" || true
    fi
fi

exit "${run_status}"
