#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"
cd "${ROOT}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

CALLER_UID="$(id -u)"
CALLER_GID="$(id -g)"

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
verify_output=""
for ((i = 0; i < ${#run_cmd[@]}; i++)); do
    case "${run_cmd[$i]}" in
        */verify.py|verify.py)
            is_verify_command=1
            ;;
        --output)
            if (( i + 1 < ${#run_cmd[@]} )); then
                verify_output="${run_cmd[$((i + 1))]}"
            fi
            ;;
        --output=*)
            verify_output="${run_cmd[$i]#--output=}"
            ;;
    esac
done

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
            "CONDA_BASE=${CONDA_BASE}"
            "OPERATOR_CONDA_ENV=${OPERATOR_CONDA_ENV}"
            "OPERATOR_PYTHON=${OPERATOR_PYTHON}"
            "AST_CHECK_PYTHON=${AST_CHECK_PYTHON}"
            "WORKSPACE_BASE=${WORKSPACE_BASE}"
            "PATH=${PATH}"
            "PYTHONPATH=${PYTHONPATH:-}"
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
        "${direct_cmd[@]}" >"${raw_log}" 2>&1
        status=$?
        set -e
        echo "[verifier-log] raw output saved to ${raw_log}" >&2
        return "${status}"
    fi

    "${direct_cmd[@]}"
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
    local start_ts now elapsed
    start_ts="$(date +%s)"
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
                if [[ "${EVAL_VERBOSE:-0}" == "1" || "${EVAL_VERBOSE:-}" == "true" || "${EVAL_VERBOSE:-}" == "True" ]]; then
                    echo "[npu-lock] acquired ${device_prefix}${device_id} (${lock_file})" >&2
                fi
                run_with_device "${device_id}"
                local status=$?
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
                echo "[npu-lock] timed out waiting for one of EVAL_DEVICE_IDS=${EVAL_DEVICE_IDS}" >&2
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
    run_direct || run_status=$?
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

exit "${run_status}"
