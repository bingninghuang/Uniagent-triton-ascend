#!/usr/bin/env bash
set -euo pipefail

# Minimal environment used by the operator validation wrapper. Keep this close
# to the old operator_pipeline.sh behavior: choose the evaluator Python, but do
# not rewrite CANN/Ascend library paths that are already prepared by the image.

: "${CONDA_BASE:=/opt/conda}"
: "${OPERATOR_CONDA_ENV:=evaluator-py311}"

_CONDA_PYTHON="${CONDA_BASE}/envs/${OPERATOR_CONDA_ENV}/bin/python"
if [[ -x "${_CONDA_PYTHON}" ]]; then
    export OPERATOR_PYTHON="${_CONDA_PYTHON}"
elif [[ -n "${OPERATOR_PYTHON:-}" ]]; then
    echo "[env.sh] WARN: conda python not found at ${_CONDA_PYTHON}, using OPERATOR_PYTHON=${OPERATOR_PYTHON}" >&2
else
    echo "[env.sh] ERROR: conda python not found at ${_CONDA_PYTHON} and OPERATOR_PYTHON is not set." >&2
    exit 1
fi

export AST_CHECK_PYTHON="${AST_CHECK_PYTHON:-python3}"
: "${WORKSPACE_BASE:=/opt/workspace/agent_workdir}"
export WORKSPACE_BASE
