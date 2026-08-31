#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"
cd "${ROOT}"

op_name="${1:-${OPERATOR_NAME:-${TRITON_CURRENT_OP_NAME:-}}}"
verify_timeout="${2:-${TRITON_EVAL_TIMEOUT:-900}}"
if [[ -z "${op_name}" && -f "${ROOT}/.triton_current_op_name" ]]; then
    op_name="$(tr -d '[:space:]' < "${ROOT}/.triton_current_op_name" || true)"
fi
if [[ -z "${op_name}" ]]; then
    echo "usage: tools/verify_once.sh <op_name> [timeout_sec]" >&2
    exit 2
fi

ref="src/${op_name}.py"
impl="src/${op_name}_triton_ascend_impl.py"
verify_dir="output/verify"
case_baseline_dir=".triton_case_sidecars"
if [[ ! -f "${ref}" ]]; then
    echo "[verify-once] missing reference file: ${ref}" >&2
    exit 2
fi
if [[ ! -f "${impl}" ]]; then
    echo "[verify-once] missing implementation file: ${impl}" >&2
    exit 2
fi

# Agent-generated case files must never affect correctness or reward.
case_source="${case_baseline_dir}/${op_name}.json"
case_path="src/${op_name}.json"
if [[ ! -f "${case_source}" ]]; then
    echo "[verify-once] missing case file: ${case_source}" >&2
    exit 2
fi
rm -f src/*.json
cp "${case_source}" "${case_path}"
chmod 0444 "${case_path}"

rm -rf "${verify_dir}"
mkdir -p "${verify_dir}"
cp "${ref}" "${verify_dir}/${op_name}_torch.py"
cp "${impl}" "${verify_dir}/${op_name}_triton_ascend_impl.py"
cp "${case_path}" "${verify_dir}/${op_name}_torch.json"

"${AST_CHECK_PYTHON}" .claude/skills/triton-op-verifier/scripts/validate_triton_impl.py "${impl}" --json

PY="${OPERATOR_PYTHON:-/usr/local/bin/python3}"
exec bash tools/run_npu_command.sh \
    "${PY}" .claude/skills/triton-op-verifier/scripts/verify.py \
    --op_name "${op_name}" \
    --verify_dir "${verify_dir}" \
    --triton_impl_name triton_ascend_impl \
    --timeout "${verify_timeout}" \
    --output "${verify_dir}/verify_result.json"