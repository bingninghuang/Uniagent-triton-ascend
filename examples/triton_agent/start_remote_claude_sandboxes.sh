#!/usr/bin/env bash
set -euo pipefail

IMAGE="${TRITON_CLAUDE_IMAGE:-triton-claude-code-env:latest}"
NUM_SANDBOXES="${TRITON_REMOTE_SANDBOX_NUM:-4}"
BASE_PORT="${TRITON_REMOTE_SANDBOX_BASE_PORT:-18000}"
TOKEN="${TRITON_REMOTE_SANDBOX_AUTH_TOKEN:-tok}"
PREFIX="${TRITON_REMOTE_SANDBOX_PREFIX:-triton-cc-sandbox}"
EVAL_DEVICE_IDS="${TRITON_EVAL_DEVICE_IDS:-0,1,2,3}"
LOCK_DIR="${TRITON_EVAL_LOCK_DIR:-/tmp/shared_npu_lock}"
CLAUDE_RUN_USER="${TRITON_CLAUDE_RUN_USER:-claude}"
if [[ "${CLAUDE_RUN_USER}" == "root" ]]; then
    echo "WARN: Claude Code cannot use bypassPermissions as root; using non-root user 'claude'. NPU validation still runs as root through sudo." >&2
    CLAUDE_RUN_USER="claude"
fi
CLAUDE_RUN_UID="${TRITON_CLAUDE_RUN_UID:-1000}"
CLAUDE_RUN_GID="${TRITON_CLAUDE_RUN_GID:-1000}"
CLAUDE_RUN_HOME="${TRITON_CLAUDE_RUN_HOME:-/home/${CLAUDE_RUN_USER}}"
SANDBOX_STYLE="${TRITON_REMOTE_SANDBOX_STYLE:-remote}"
NETWORK_MODE="${TRITON_REMOTE_SANDBOX_NETWORK:-host}"
IPC_MODE="${TRITON_REMOTE_SANDBOX_IPC:-host}"
SHM_SIZE="${TRITON_REMOTE_SANDBOX_SHM_SIZE:-500g}"

if [[ "${IPC_MODE}" == "host" ]]; then
    DOCKER_IPC_ARGS=(--ipc=host)
elif [[ "${IPC_MODE}" == "private" || "${IPC_MODE}" == "none" || -z "${IPC_MODE}" ]]; then
    DOCKER_IPC_ARGS=()
else
    DOCKER_IPC_ARGS=(--ipc="${IPC_MODE}")
fi

DOCKER_SHM_ARGS=()
if [[ -n "${SHM_SIZE}" ]]; then
    DOCKER_SHM_ARGS=(--shm-size "${SHM_SIZE}")
fi

DOCKER_COMPAT_MOUNTS=()
add_mount_if_exists() {
    local src="$1"
    local dst="$2"
    local mode="${3:-}"
    if [[ -e "${src}" ]]; then
        if [[ -n "${mode}" ]]; then
            DOCKER_COMPAT_MOUNTS+=(-v "${src}:${dst}:${mode}")
        else
            DOCKER_COMPAT_MOUNTS+=(-v "${src}:${dst}")
        fi
    fi
}

add_mount_if_exists /usr/local/sbin /usr/local/sbin ro
add_mount_if_exists /etc/localtime /etc/localtime ro
add_mount_if_exists /etc/timezone /etc/timezone ro
add_mount_if_exists /mnt/pipeline-data /mnt/pipeline-data

IFS=',' read -r -a EVAL_DEVICE_ID_LIST <<< "${EVAL_DEVICE_IDS}"
if [[ "${#EVAL_DEVICE_ID_LIST[@]}" -eq 0 ]]; then
    echo "TRITON_EVAL_DEVICE_IDS must contain at least one device id" >&2
    exit 1
fi

mkdir -p "${LOCK_DIR}"
chmod 1777 "${LOCK_DIR}" 2>/dev/null || true
for device_id in "${EVAL_DEVICE_ID_LIST[@]}"; do
    device_id="${device_id//[[:space:]]/}"
    [[ -n "${device_id}" ]] || continue
    touch "${LOCK_DIR}/npu${device_id}.lock" 2>/dev/null || true
    chmod 0666 "${LOCK_DIR}/npu${device_id}.lock" 2>/dev/null || true
done

for i in $(seq 0 $((NUM_SANDBOXES - 1))); do
    name="${PREFIX}-${i}"
    port=$((BASE_PORT + i))
    EVAL_ENV_ARGS=(
        -e "EVAL_LOCK_DIR=/shared/device-locks"
        -e "EVAL_DEVICE_PREFIX=npu"
        -e "EVAL_DEVICE_IDS=${EVAL_DEVICE_IDS}"
        -e "EVAL_DEVICE_COUNT=1"
        -e "EVAL_ENV_NAME=ASCEND_RT_VISIBLE_DEVICES"
        -e "EVAL_RETRY_INTERVAL=1.0"
        -e "EVAL_TIMEOUT=None"
        -e "CONDA_BASE=${CONDA_BASE:-/opt/conda}"
        -e "OPERATOR_CONDA_ENV=${OPERATOR_CONDA_ENV:-evaluator-py311}"
        -e "OPERATOR_PYTHON=${OPERATOR_PYTHON:-/opt/conda/envs/evaluator-py311/bin/python}"
    )
    if [[ "${NETWORK_MODE}" == "host" ]]; then
        DOCKER_NETWORK_ARGS=(--network host)
        server_port="${port}"
    else
        DOCKER_NETWORK_ARGS=(-p "${port}:8000")
        server_port=8000
    fi

    docker rm -f "${name}" >/dev/null 2>&1 || true
    docker run -d \
        --name "${name}" \
        --privileged \
        "${DOCKER_IPC_ARGS[@]}" \
        "${DOCKER_SHM_ARGS[@]}" \
        --entrypoint /usr/bin/env \
        "${DOCKER_NETWORK_ARGS[@]}" \
        "${EVAL_ENV_ARGS[@]}" \
        -e SWEREX_AUTH_TOKEN="${TOKEN}" \
        -e SWEREX_PORT="${server_port}" \
        -e TRITON_CLAUDE_RUN_USER="${CLAUDE_RUN_USER}" \
        -e TRITON_CLAUDE_RUN_UID="${CLAUDE_RUN_UID}" \
        -e TRITON_CLAUDE_RUN_GID="${CLAUDE_RUN_GID}" \
        -e TRITON_CLAUDE_RUN_HOME="${CLAUDE_RUN_HOME}" \
        -v /dev:/dev \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
        -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
        -v /usr/local/dcmi:/usr/local/dcmi:ro \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
        -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
        -v "${LOCK_DIR}:/shared/device-locks" \
        "${DOCKER_COMPAT_MOUNTS[@]}" \
        "${IMAGE}" \
        bash -lc '
set -euo pipefail
user="${TRITON_CLAUDE_RUN_USER:-claude}"
uid="${TRITON_CLAUDE_RUN_UID:-1000}"
gid="${TRITON_CLAUDE_RUN_GID:-1000}"
home="${TRITON_CLAUDE_RUN_HOME:-/home/${user}}"

if [[ "${user}" != "root" ]]; then
    if ! getent group "${gid}" >/dev/null 2>&1; then
        groupadd -g "${gid}" "${user}" 2>/dev/null || true
    fi

    if ! id -u "${user}" >/dev/null 2>&1; then
        if command -v useradd >/dev/null 2>&1; then
            useradd -m -u "${uid}" -g "${gid}" -s /bin/bash "${user}"
        elif command -v adduser >/dev/null 2>&1; then
            adduser --disabled-password --gecos "" --home "${home}" --uid "${uid}" --gid "${gid}" "${user}"
        else
            echo "Neither useradd nor adduser is available; cannot create ${user}" >&2
            exit 1
        fi
    fi
fi

mkdir -p "${home}"
if [[ "${user}" != "root" ]]; then
    chown -R "${uid}:${gid}" "${home}"
fi

mkdir -p /shared/device-locks
chmod 1777 /shared/device-locks 2>/dev/null || true
IFS=',' read -r -a eval_devices <<< "${EVAL_DEVICE_IDS:-}"
for device_id in "${eval_devices[@]}"; do
    device_id="${device_id//[[:space:]]/}"
    [[ -n "${device_id}" ]] || continue
    touch "/shared/device-locks/npu${device_id}.lock" 2>/dev/null || true
    chmod 0666 "/shared/device-locks/npu${device_id}.lock" 2>/dev/null || true
done

container_hostname="$(hostname 2>/dev/null || true)"
if [[ -n "${container_hostname}" ]] && ! getent hosts "${container_hostname}" >/dev/null 2>&1; then
    echo "127.0.1.1 ${container_hostname}" >> /etc/hosts 2>/dev/null || true
fi

add_device_group() {
    local path="$1"
    [[ -e "${path}" ]] || return 0
    local device_gid
    device_gid="$(stat -c "%g" "${path}" 2>/dev/null || true)"
    [[ -n "${device_gid}" ]] || return 0
    local group_name
    group_name="$(getent group "${device_gid}" | cut -d: -f1 || true)"
    if [[ -z "${group_name}" ]]; then
        group_name="ascenddev${device_gid}"
        groupadd -g "${device_gid}" "${group_name}" 2>/dev/null || true
    fi
    group_name="$(getent group "${device_gid}" | cut -d: -f1 || true)"
    [[ -n "${group_name}" ]] || return 0
    usermod -aG "${group_name}" "${user}" 2>/dev/null || true
}

for dev in /dev/davinci* /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
    add_device_group "${dev}"
done

if [[ "${user}" != "root" ]]; then
    if command -v sudo >/dev/null 2>&1; then
        {
            echo "Defaults:${user} env_keep += \"ASCEND_RT_VISIBLE_DEVICES ALLOCATED_DEVICE_ID ALLOCATED_DEVICE_PREFIX\""
            echo "Defaults:${user} env_keep += \"EVAL_LOCK_DIR EVAL_DEVICE_PREFIX EVAL_DEVICE_COUNT EVAL_DEVICE_IDS EVAL_ENV_NAME EVAL_RETRY_INTERVAL EVAL_TIMEOUT EVAL_VERBOSE\""
            echo "Defaults:${user} env_keep += \"CONDA_BASE OPERATOR_CONDA_ENV OPERATOR_PYTHON AST_CHECK_PYTHON WORKSPACE_BASE\""
            echo "Defaults:${user} env_keep += \"ASCEND_HOME ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_CUSTOM_PATH ASCEND_INSTALL_PATH TOOLCHAIN_HOME\""
            echo "Defaults:${user} env_keep += \"LD_LIBRARY_PATH PYTHONPATH PATH TORCH_DEVICE_BACKEND_AUTOLOAD\""
            echo "${user} ALL=(root) NOPASSWD:SETENV: ALL"
        } > /etc/sudoers.d/triton-eval
        chmod 0440 /etc/sudoers.d/triton-eval
    else
        echo "WARN: sudo is not installed; non-root Claude Code sessions cannot elevate verifier scripts if the image requires root for NPU validation." >&2
    fi
fi

if [[ -z "${OPERATOR_PYTHON:-}" && -x /opt/conda/envs/evaluator-py311/bin/python ]]; then
    export OPERATOR_PYTHON=/opt/conda/envs/evaluator-py311/bin/python
fi
export AST_CHECK_PYTHON="${AST_CHECK_PYTHON:-python3}"

exec python3 -m swerex.server --host 0.0.0.0 --port "${SWEREX_PORT:-8000}" --auth-token "$SWEREX_AUTH_TOKEN"
'

    echo "started ${name} on port ${port} devices=${EVAL_DEVICE_IDS} (shared-lock, style=${SANDBOX_STYLE}, network=${NETWORK_MODE}, ipc=${IPC_MODE}, shm=${SHM_SIZE:-none}, run_user=${CLAUDE_RUN_USER})"
done

ports="$(python3 - <<PY
base = int("${BASE_PORT}")
n = int("${NUM_SANDBOXES}")
print(",".join(str(base + i) for i in range(n)))
PY
)"

echo
echo "Use these on the training node:"
echo "export TRITON_SANDBOX_DEPLOYMENT=local_attach"
echo "export TRITON_REMOTE_SANDBOX_HOST=http://<B_NODE_IP>"
echo "export TRITON_REMOTE_SANDBOX_PORTS=${ports}"
echo "export TRITON_REMOTE_SANDBOX_AUTH_TOKEN=${TOKEN}"
echo "# Remote containers default to Claude Code remote launch style:"
echo "#   TRITON_REMOTE_SANDBOX_STYLE=remote, TRITON_REMOTE_SANDBOX_NETWORK=host,"
echo "#   TRITON_REMOTE_SANDBOX_IPC=host, TRITON_REMOTE_SANDBOX_SHM_SIZE=500g."
echo "# Remote sandbox containers share EVAL_DEVICE_IDS=${EVAL_DEVICE_IDS}; tools/run_npu_command.sh locks one NPU per validation."
echo "# Claude Code runs as non-root ${CLAUDE_RUN_USER}; NPU verify/benchmark runs as root via sudo."
echo "# Leave these unset on the training node unless you intentionally override sandbox-side allocation."
echo "unset TRITON_EVAL_DEVICE_IDS"
echo "unset TRITON_EVAL_DEVICE_COUNT"
echo "unset TRITON_EVAL_ENV_NAME"
echo "export TRITON_CLAUDE_RUN_USER=${CLAUDE_RUN_USER}"
echo "export TRITON_CLAUDE_RUN_HOME=${CLAUDE_RUN_HOME}"
