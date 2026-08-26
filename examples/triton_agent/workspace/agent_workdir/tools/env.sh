#!/usr/bin/env bash
set -euo pipefail

# Minimal environment used by the operator validation wrapper. Use the
# configured operator Python only when it exists; otherwise use the image's
# default Python without rewriting CANN/Ascend library paths.

if [[ -n "${OPERATOR_PYTHON:-}" && -x "${OPERATOR_PYTHON}" ]]; then
    export OPERATOR_PYTHON
elif [[ -x /usr/local/bin/python3 ]]; then
    export OPERATOR_PYTHON=/usr/local/bin/python3
else
    export OPERATOR_PYTHON=python3
fi

export AST_CHECK_PYTHON="${AST_CHECK_PYTHON:-${OPERATOR_PYTHON}}"
: "${WORKSPACE_BASE:=/opt/workspace/agent_workdir}"
export WORKSPACE_BASE
