#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"

# Existing OpenAI-compatible vLLM server. Start it separately with
# examples/triton_agent/start_claude_code_vllm_server.sh.
export PROXY_PORT="${PROXY_PORT:-5000}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:${PROXY_PORT}/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-triton-claude-code-model}"

# Sample selection.
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SCRIPT_DIR}/benchmarks/NPUKernelBench}"
BENCHMARK_LEVEL="${BENCHMARK_LEVEL:-level1}"
BENCHMARK_START="${BENCHMARK_START:-1}"
BENCHMARK_NUM="${BENCHMARK_NUM:-50}"
BENCHMARK_ARCH="${BENCHMARK_ARCH:-ascend910b1}"

# Output and Claude Code settings.
BENCHMARK_RUN_ROOT="${BENCHMARK_RUN_ROOT:-${SCRIPT_DIR}/vllm_benchmark_runs/$(date +%Y%m%d_%H%M%S)}"
BENCHMARK_CLAUDE_MODEL="${BENCHMARK_CLAUDE_MODEL:-${TRITON_CLAUDE_MODEL:-${SERVED_MODEL_NAME}}}"
BENCHMARK_CLAUDE_EXTRA_ARGS="${BENCHMARK_CLAUDE_EXTRA_ARGS:-${TRITON_CLAUDE_EXTRA_ARGS:---max-turns ${TRITON_CLAUDE_MAX_TURNS:-100}}}"
BENCHMARK_CLAUDE_PERMISSION_ARGS="${BENCHMARK_CLAUDE_PERMISSION_ARGS:---permission-mode bypassPermissions}"
export TRITON_CLAUDE_REPAIR_ROUNDS="${TRITON_CLAUDE_REPAIR_ROUNDS:-1}"
TRITON_CLAUDE_DISALLOWED_TOOLS="${TRITON_CLAUDE_DISALLOWED_TOOLS:-Task,TaskCreate,TaskUpdate,TaskList,TaskGet,TaskOutput,TaskStop,Workflow,AskUserQuestion,EnterPlanMode,ExitPlanMode}"
if [[ -z "${BENCHMARK_PROMPT:-}" ]]; then
    BENCHMARK_PROMPT="$(python3 - <<'PY'
from examples.triton_agent.claude_code_agent_runner import DEFAULT_CLAUDE_PROMPT
print(DEFAULT_CLAUDE_PROMPT)
PY
)"
fi
BENCHMARK_SANDBOX_MODE="${BENCHMARK_SANDBOX_MODE:-remote}"
BENCHMARK_AGENT_CONFIG="${BENCHMARK_AGENT_CONFIG:-${SCRIPT_DIR}/agent_config_claude_code.yaml}"
BENCHMARK_TIME_BUDGET_SEC="${BENCHMARK_TIME_BUDGET_SEC:-${TRITON_CLAUDE_TIME_BUDGET_SEC:-1800}}"
BENCHMARK_ARTIFACT_DIR="${BENCHMARK_ARTIFACT_DIR:-${TRITON_CLAUDE_ARTIFACT_DIR:-${BENCHMARK_RUN_ROOT}/artifacts}}"
export TRITON_CLAUDE_DISALLOWED_TOOLS

if [[ -z "${OPERATOR_PYTHON:-}" && -x /opt/conda/envs/evaluator-py311/bin/python ]]; then
    export OPERATOR_PYTHON=/opt/conda/envs/evaluator-py311/bin/python
fi

export TRITON_REMOTE_SANDBOX_HOST="${TRITON_REMOTE_SANDBOX_HOST:-80.48.5.51}"
export TRITON_REMOTE_SANDBOX_PORTS="${TRITON_REMOTE_SANDBOX_PORTS:-18000,18001,18002,18003}"
export TRITON_REMOTE_SANDBOX_AUTH_TOKEN="${TRITON_REMOTE_SANDBOX_AUTH_TOKEN:-tok}"
export TRITON_REMOTE_SANDBOX_AUTH_TOKENS="${TRITON_REMOTE_SANDBOX_AUTH_TOKENS:-${TRITON_REMOTE_SANDBOX_AUTH_TOKEN}}"

# Local Anthropic -> OpenAI shim used by Claude Code.
if [[ "${BENCHMARK_SANDBOX_MODE}" == "remote" ]]; then
    DEFAULT_SHIM_BIND_HOST="0.0.0.0"
    DEFAULT_SHIM_PUBLIC_HOST="80.48.5.63"
else
    DEFAULT_SHIM_BIND_HOST="127.0.0.1"
    DEFAULT_SHIM_PUBLIC_HOST="127.0.0.1"
fi
SHIM_BIND_HOST="${SHIM_BIND_HOST:-${TRITON_SHIM_BIND_HOST:-${DEFAULT_SHIM_BIND_HOST}}}"
SHIM_PUBLIC_HOST="${SHIM_PUBLIC_HOST:-${TRITON_SHIM_PUBLIC_HOST:-${DEFAULT_SHIM_PUBLIC_HOST}}}"
SHIM_PORT="${SHIM_PORT:-0}"
SHIM_REQUEST_TIMEOUT="${SHIM_REQUEST_TIMEOUT:-${TRITON_SHIM_REQUEST_TIMEOUT:-600}}"
export BENCHMARK_CLAUDE_MODEL SHIM_BIND_HOST SHIM_PUBLIC_HOST SHIM_PORT SHIM_REQUEST_TIMEOUT

SHIM_PID=""

cleanup() {
    if [[ -n "${SHIM_PID}" ]]; then
        kill "${SHIM_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

wait_for_http() {
    local url="$1"
    local timeout="${2:-300}"
    local started
    started="$(date +%s)"
    until curl -fsS "${url}" >/dev/null 2>&1; do
        if (( "$(date +%s)" - started > timeout )); then
            echo "Timed out waiting for ${url}" >&2
            return 1
        fi
        sleep 2
    done
}

if [[ "${BENCHMARK_SANDBOX_MODE}" == "local" ]] && ! command -v claude >/dev/null 2>&1; then
    echo "Claude Code CLI not found. Install it first or run inside an image that contains 'claude'." >&2
    exit 1
fi

mkdir -p "${BENCHMARK_RUN_ROOT}"
MANIFEST="${BENCHMARK_RUN_ROOT}/manifest.tsv"
SUMMARY="${BENCHMARK_RUN_ROOT}/summary.jsonl"
SHIM_INFO="${BENCHMARK_RUN_ROOT}/shim_info.json"
ENDPOINTS="${BENCHMARK_RUN_ROOT}/remote_endpoints.tsv"

echo "[benchmark] checking vLLM: ${OPENAI_BASE_URL}"
wait_for_http "${OPENAI_BASE_URL%/}/models" "${VLLM_WAIT_TIMEOUT:-300}"

if [[ "${BENCHMARK_SANDBOX_MODE}" == "local" ]]; then
python3 - "${SHIM_INFO}" <<'PY' &
import json
import os
import signal
import sys
import time
from pathlib import Path

from examples.triton_agent.anthropic_openai_shim import AnthropicOpenAIShim

info_path = Path(sys.argv[1])
with AnthropicOpenAIShim(
    openai_base_url=os.environ["OPENAI_BASE_URL"],
    openai_api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    model_name=os.environ["BENCHMARK_CLAUDE_MODEL"],
    host=os.environ.get("SHIM_BIND_HOST", "127.0.0.1"),
    port=int(os.environ.get("SHIM_PORT", "0") or "0"),
    request_timeout=float(os.environ.get("SHIM_REQUEST_TIMEOUT", "600")),
) as shim:
    url = f"http://{os.environ.get('SHIM_PUBLIC_HOST', '127.0.0.1')}:{shim.port}"
    info_path.write_text(json.dumps({"url": url, "port": shim.port}), encoding="utf-8")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        time.sleep(3600)
PY
SHIM_PID="$!"

    for _ in $(seq 1 100); do
        [[ -s "${SHIM_INFO}" ]] && break
        sleep 0.1
    done
    if [[ ! -s "${SHIM_INFO}" ]]; then
        echo "Shim failed to start" >&2
        exit 1
    fi

    SHIM_URL="$(python3 - "${SHIM_INFO}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["url"])
PY
)"
    wait_for_http "${SHIM_URL}/healthz" 30
    echo "[benchmark] local shim: ${SHIM_URL} -> ${OPENAI_BASE_URL}"
else
    python3 - "${ENDPOINTS}" <<'PY'
import sys
from pathlib import Path

from examples.triton_agent.claude_code_agent_runner import _pool_from_env

out = Path(sys.argv[1])
pool = _pool_from_env()
if not pool:
    raise SystemExit(
        "Remote mode requires TRITON_REMOTE_SANDBOX_POOL_JSON/TRITON_REMOTE_SANDBOX_POOL "
        "or TRITON_REMOTE_SANDBOX_HOST + TRITON_REMOTE_SANDBOX_PORTS + TRITON_REMOTE_SANDBOX_AUTH_TOKEN(S)."
    )
lines = []
for endpoint in pool:
    lines.append(
        "\t".join(
            [
                str(endpoint["host"]),
                str(endpoint["port"]),
                str(endpoint["auth_token"]),
            ]
        )
    )
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"remote_endpoints={len(lines)}")
PY
    echo "[benchmark] remote endpoints: ${ENDPOINTS}"
fi

python3 - \
  "${REPO_ROOT}" \
  "${BENCHMARK_ROOT}" \
  "${BENCHMARK_LEVEL}" \
  "${BENCHMARK_START}" \
  "${BENCHMARK_NUM}" \
  "${BENCHMARK_ARCH}" \
  "${BENCHMARK_RUN_ROOT}" \
  "${MANIFEST}" <<'PY'
import re
import shutil
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
benchmark_root = Path(sys.argv[2]).resolve()
level_arg = sys.argv[3]
start = int(sys.argv[4])
num = int(sys.argv[5])
arch = sys.argv[6]
run_root = Path(sys.argv[7]).resolve()
manifest_path = Path(sys.argv[8])

from examples.triton_agent.claude_code_agent_runner import (  # noqa: E402
    KERNELBENCH_INSTRUCTION_TEMPLATE,
    _prepare_claude_project_skills,
)

def level_dir(root: Path, value: str) -> Path:
    candidates = []
    raw = value.strip()
    candidates.append(raw)
    if raw.isdigit():
        candidates.extend([f"level{raw}", f"level_{raw}"])
    elif raw.startswith("level_"):
        candidates.append("level" + raw.split("_", 1)[1])
    elif raw.startswith("level"):
        suffix = raw.removeprefix("level")
        if suffix:
            candidates.append(f"level_{suffix}")
    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Cannot find level directory for {value!r} under {root}")

def sort_key(path: Path):
    parts = re.split(r"(\d+)", path.stem)
    return [int(part) if part.isdigit() else part.lower() for part in parts]

level_path = level_dir(benchmark_root, level_arg)
tasks = sorted((p for p in level_path.glob("*.py") if not p.name.startswith("__")), key=sort_key)
selected = tasks[start : start + num if num >= 0 else None]
if not selected:
    raise RuntimeError(f"No tasks selected from {level_path}; start={start}, num={num}, total={len(tasks)}")

template = repo_root / "examples" / "triton_agent" / "workspace" / "agent_workdir"
rows = []
for offset, task_file in enumerate(selected):
    position = start + offset
    stem = task_file.stem
    level_match = re.search(r"level_?(\d+)", level_path.name)
    level = level_match.group(1) if level_match else "1"
    op_name = f"kernelbench_l{level}_{stem}"
    sample_dir = run_root / f"{position:04d}_{op_name}"
    agent_dir = sample_dir / "workspace" / "agent_workdir"
    if agent_dir.exists():
        shutil.rmtree(agent_dir)
    shutil.copytree(template, agent_dir)
    _prepare_claude_project_skills(agent_dir)

    src_dir = agent_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    task_code = task_file.read_text(encoding="utf-8")
    (src_dir / f"{op_name}.py").write_text(task_code, encoding="utf-8")
    for sidecar in task_file.parent.glob(f"{stem}.*"):
        if sidecar.resolve() != task_file:
            shutil.copy2(sidecar, src_dir / sidecar.name)

    instruction = f"Implement KernelBench problem {stem} as an Ascend NPU Triton operator."
    (agent_dir / "INSTRUCTIONS.md").write_text(
        KERNELBENCH_INSTRUCTION_TEMPLATE.format(
            op_name=op_name,
            arch=arch,
            task_code=task_code,
            instruction=instruction,
        ),
        encoding="utf-8",
    )
    rows.append((str(position), op_name, str(task_file), str(sample_dir), str(agent_dir)))

manifest_path.write_text("\n".join("\t".join(row) for row in rows) + "\n", encoding="utf-8")
print(f"selected={len(rows)} level={level_path} start={start} num={num}")
PY

if [[ "${BENCHMARK_SANDBOX_MODE}" == "local" ]]; then
    export ANTHROPIC_BASE_URL="${SHIM_URL}"
    export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-benchmark-session}"
    export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-${ANTHROPIC_AUTH_TOKEN}}"
    export ANTHROPIC_MODEL="${BENCHMARK_CLAUDE_MODEL}"
    export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
    export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
    export CLAUDE_CODE_ATTRIBUTION_HEADER=0
    read -r -a CLAUDE_PERMISSION_ARGS_ARRAY <<< "${BENCHMARK_CLAUDE_PERMISSION_ARGS}"
    read -r -a CLAUDE_EXTRA_ARGS_ARRAY <<< "${BENCHMARK_CLAUDE_EXTRA_ARGS}"
    CLAUDE_DISALLOWED_ARGS_ARRAY=()
    if [[ -n "${TRITON_CLAUDE_DISALLOWED_TOOLS}" ]] \
        && [[ "${TRITON_CLAUDE_DISALLOWED_TOOLS}" != "0" ]] \
        && [[ "${TRITON_CLAUDE_DISALLOWED_TOOLS}" != "none" ]] \
        && [[ "${TRITON_CLAUDE_DISALLOWED_TOOLS}" != "false" ]]; then
        IFS=', ' read -r -a CLAUDE_DISALLOWED_TOOL_ARRAY <<< "${TRITON_CLAUDE_DISALLOWED_TOOLS}"
        CLAUDE_DISALLOWED_ARGS_ARRAY=(--disallowedTools "${CLAUDE_DISALLOWED_TOOL_ARRAY[@]}")
    fi
else
    mapfile -t REMOTE_ENDPOINT_LINES < "${ENDPOINTS}"
    if [[ "${#REMOTE_ENDPOINT_LINES[@]}" -eq 0 ]]; then
        echo "No remote sandbox endpoints loaded from ${ENDPOINTS}" >&2
        exit 1
    fi
fi

echo "[benchmark] manifest: ${MANIFEST}"
: > "${SUMMARY}"

SAMPLE_COUNTER=0
while IFS=$'\t' read -r SAMPLE_POSITION OP_NAME TASK_FILE SAMPLE_DIR AGENT_DIR; do
    [[ -z "${SAMPLE_POSITION}" ]] && continue
    TRAJECTORY="${AGENT_DIR}/claude_code_trajectory.jsonl"
    STDOUT_LOG="${AGENT_DIR}/claude_code_stdout.log"
    CONVERSATION_LOG="${AGENT_DIR}/conversation.log"

    echo "[benchmark] sample=${SAMPLE_POSITION} op=${OP_NAME} mode=${BENCHMARK_SANDBOX_MODE}"
    if [[ "${BENCHMARK_SANDBOX_MODE}" == "local" ]]; then
        set +e
        pushd "${AGENT_DIR}" >/dev/null
        claude -p "${BENCHMARK_PROMPT}" \
          "${CLAUDE_PERMISSION_ARGS_ARRAY[@]}" \
          "${CLAUDE_DISALLOWED_ARGS_ARRAY[@]}" \
          --output-format stream-json \
          --include-partial-messages \
          --include-hook-events \
          --verbose \
          "${CLAUDE_EXTRA_ARGS_ARRAY[@]}" \
          2>&1 | tee "${TRAJECTORY}" "${STDOUT_LOG}"
        CLAUDE_EXIT="${PIPESTATUS[0]}"
        popd >/dev/null
        set -e

        if [[ "${BENCHMARK_REQUIRE_CORRECTNESS:-1}" != "0" ]] \
            && [[ "${BENCHMARK_REQUIRE_CORRECTNESS:-1}" != "false" ]] \
            && [[ "${BENCHMARK_REQUIRE_CORRECTNESS:-1}" != "none" ]]; then
            if ! python3 - "${AGENT_DIR}" <<'PY'
import glob
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])

def load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None

metrics = load(root / "metrics.json")
if isinstance(metrics, dict) and (metrics.get("success") or metrics.get("correctness_ok")):
    raise SystemExit(0)

for pattern in ("verify_result.json", "**/verify_result.json", "**/verify_result_*.json"):
    for path in glob.glob(str(root / pattern), recursive=True):
        data = load(path)
        if not isinstance(data, dict):
            continue
        total = int(data.get("total_cases") or 0)
        passed = int(data.get("passed_cases") or 0)
        if total > 0 and passed == total:
            raise SystemExit(0)

summary = load(root / "summary.json")
if isinstance(summary, dict) and summary.get("success"):
    raise SystemExit(0)

raise SystemExit(1)
PY
            then
                if [[ "${CLAUDE_EXIT}" == "0" ]]; then
                    CLAUDE_EXIT=1
                fi
            fi
        fi

        python3 - "${TRAJECTORY}" "${CONVERSATION_LOG}" "${SUMMARY}" "${SAMPLE_POSITION}" "${OP_NAME}" "${TASK_FILE}" "${AGENT_DIR}" "${CLAUDE_EXIT}" <<'PY'
import json
import sys
from pathlib import Path

from examples.triton_agent.reward import _format_claude_trajectory

trajectory = Path(sys.argv[1])
conversation = Path(sys.argv[2])
summary = Path(sys.argv[3])
sample_position = int(sys.argv[4])
op_name = sys.argv[5]
task_file = sys.argv[6]
agent_dir = Path(sys.argv[7])
claude_exit = int(sys.argv[8])

if trajectory.exists():
    conversation.write_text(
        _format_claude_trajectory(trajectory.read_text(encoding="utf-8", errors="replace")),
        encoding="utf-8",
    )

metrics_path = agent_dir / "metrics.json"
metrics = None
if metrics_path.exists():
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        metrics = {"metrics_parse_error": str(exc)}
if metrics is None:
    summary_path = agent_dir / "summary.json"
    if summary_path.exists():
        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = {
                "success": bool(summary_data.get("success")),
                "perf_data": summary_data.get("perf_data", {}),
                "source": "upstream_summary_json",
            }
        except Exception as exc:
            metrics = {"summary_parse_error": str(exc)}

event = {
    "sample_position": sample_position,
    "op_name": op_name,
    "task_file": task_file,
    "agent_dir": str(agent_dir),
    "claude_exit": claude_exit,
    "conversation_log": str(conversation),
    "metrics_json": str(metrics_path) if metrics_path.exists() else "",
    "ast_check_ok": bool(metrics.get("ast_check_ok")) if isinstance(metrics, dict) else False,
    "compile_ok": bool(metrics.get("compile_ok")) if isinstance(metrics, dict) else False,
    "correctness_ok": bool(metrics.get("correctness_ok")) if isinstance(metrics, dict) else False,
    "passed_cases": metrics.get("passed_cases") if isinstance(metrics, dict) else None,
    "total_cases": metrics.get("total_cases") if isinstance(metrics, dict) else None,
    "pass_rate": metrics.get("pass_rate") if isinstance(metrics, dict) else None,
    "success": bool(metrics.get("success")) if isinstance(metrics, dict) else False,
    "reward": metrics.get("reward") if isinstance(metrics, dict) else None,
    "error_type": metrics.get("error_type") if isinstance(metrics, dict) else None,
}
with summary.open("a", encoding="utf-8") as f:
    f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
PY
    else
        ENDPOINT_INDEX=$((SAMPLE_COUNTER % ${#REMOTE_ENDPOINT_LINES[@]}))
        ENDPOINT_LINE="${REMOTE_ENDPOINT_LINES[${ENDPOINT_INDEX}]}"
        IFS=$'\t' read -r REMOTE_HOST REMOTE_PORT REMOTE_AUTH_TOKEN <<< "${ENDPOINT_LINE}"
        REMOTE_OUTPUT="${AGENT_DIR}/remote_result.json"
        mkdir -p "${BENCHMARK_ARTIFACT_DIR}"
        echo "[benchmark] remote endpoint[${ENDPOINT_INDEX}/${#REMOTE_ENDPOINT_LINES[@]}]: ${REMOTE_HOST}:${REMOTE_PORT} output=${REMOTE_OUTPUT}"

        set +e
        python3 "${SCRIPT_DIR}/run_claude_code_vllm_benchmark_remote.py" \
          --workspace "${SAMPLE_DIR}" \
          --op-name "${OP_NAME}" \
          --arch "${BENCHMARK_ARCH}" \
          --openai-base-url "${OPENAI_BASE_URL}" \
          --openai-api-key "${OPENAI_API_KEY}" \
          --model "${BENCHMARK_CLAUDE_MODEL}" \
          --prompt "${BENCHMARK_PROMPT}" \
          --extra-args "${BENCHMARK_CLAUDE_EXTRA_ARGS}" \
          --time-budget-sec "${BENCHMARK_TIME_BUDGET_SEC}" \
          --artifact-dir "${BENCHMARK_ARTIFACT_DIR}" \
          --agent-config "${BENCHMARK_AGENT_CONFIG}" \
          --remote-host "${REMOTE_HOST}" \
          --remote-port "${REMOTE_PORT}" \
          --remote-auth-token "${REMOTE_AUTH_TOKEN}" \
          --session-id "benchmark-${SAMPLE_POSITION}-${OP_NAME}" \
          --output "${REMOTE_OUTPUT}"
        REMOTE_EXIT="$?"
        set -e
        echo "[benchmark] remote helper exit=${REMOTE_EXIT} output=${REMOTE_OUTPUT}"

        python3 - "${SUMMARY}" "${SAMPLE_POSITION}" "${OP_NAME}" "${TASK_FILE}" "${SAMPLE_DIR}" "${REMOTE_OUTPUT}" "${REMOTE_EXIT}" <<'PY'
import json
import sys
from pathlib import Path

summary = Path(sys.argv[1])
sample_position = int(sys.argv[2])
op_name = sys.argv[3]
task_file = sys.argv[4]
sample_dir = Path(sys.argv[5])
remote_output = Path(sys.argv[6])
remote_exit = int(sys.argv[7])

result = {}
if remote_output.exists():
    try:
        result = json.loads(remote_output.read_text(encoding="utf-8"))
    except Exception as exc:
        result = {"result_parse_error": str(exc)}

reward_info = result.get("reward_info") if isinstance(result.get("reward_info"), dict) else {}
metrics = reward_info.get("metrics") if isinstance(reward_info.get("metrics"), dict) else {}
archived_dir = result.get("archived_dir") or ""
conversation_log = ""
if archived_dir:
    candidate = Path(archived_dir) / "conversation.log"
    if candidate.exists():
        conversation_log = str(candidate)

event = {
    "sample_position": sample_position,
    "op_name": op_name,
    "task_file": task_file,
    "sample_dir": str(sample_dir),
    "remote_exit": remote_exit,
    "claude_exit": result.get("claude_exit"),
    "archived_dir": archived_dir,
    "conversation_log": conversation_log,
    "ast_check_ok": bool(metrics.get("ast_check_ok")),
    "compile_ok": bool(metrics.get("compile_ok")),
    "correctness_ok": bool(metrics.get("correctness_ok")),
    "passed_cases": metrics.get("passed_cases"),
    "total_cases": metrics.get("total_cases"),
    "pass_rate": metrics.get("pass_rate"),
    "success": bool(metrics.get("success")),
    "reward": reward_info.get("reward_score"),
    "reason": reward_info.get("reason"),
    "error_type": metrics.get("error_type"),
}
with summary.open("a", encoding="utf-8") as f:
    f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
PY
    fi
    SAMPLE_COUNTER=$((SAMPLE_COUNTER + 1))
done < "${MANIFEST}"

AGGREGATE="${BENCHMARK_RUN_ROOT}/aggregate_summary.json"
python3 - "${SUMMARY}" "${AGGREGATE}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
aggregate_path = Path(sys.argv[2])

rows = []
if summary_path.exists():
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"parse_error": line})

def count_true(key):
    return sum(1 for row in rows if bool(row.get(key)))

total = len(rows)
aggregate = {
    "total": total,
    "ast_success": count_true("ast_check_ok"),
    "compile_success": count_true("compile_ok"),
    "correctness_success": count_true("correctness_ok"),
    "final_success": count_true("success"),
    "ast_success_rate": round(count_true("ast_check_ok") / total, 6) if total else 0.0,
    "compile_success_rate": round(count_true("compile_ok") / total, 6) if total else 0.0,
    "correctness_success_rate": round(count_true("correctness_ok") / total, 6) if total else 0.0,
    "final_success_rate": round(count_true("success") / total, 6) if total else 0.0,
    "avg_reward": round(
        sum(float(row.get("reward") or 0.0) for row in rows) / total,
        6,
    ) if total else 0.0,
}
aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(
    "[benchmark] aggregate: "
    f"total={aggregate['total']} "
    f"ast={aggregate['ast_success']}/{total}({aggregate['ast_success_rate']:.2%}) "
    f"compile={aggregate['compile_success']}/{total}({aggregate['compile_success_rate']:.2%}) "
    f"correctness={aggregate['correctness_success']}/{total}({aggregate['correctness_success_rate']:.2%}) "
    f"success={aggregate['final_success']}/{total}({aggregate['final_success_rate']:.2%}) "
    f"avg_reward={aggregate['avg_reward']:.4f}"
)
PY

echo "[benchmark] done"
echo "[benchmark] run root : ${BENCHMARK_RUN_ROOT}"
echo "[benchmark] summary  : ${SUMMARY}"
echo "[benchmark] aggregate: ${AGGREGATE}"
