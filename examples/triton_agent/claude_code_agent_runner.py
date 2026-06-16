"""Claude Code blackbox runner for Triton/Ascend KernelBench RL.

This runner is designed for uni-agent's blackbox/gateway framework:

* the framework creates a gateway-backed session and passes ``session.base_url``;
* this runner starts an Anthropic->OpenAI shim that forwards Claude Code calls
  into that gateway;
* Claude Code runs inside the Triton sandbox and edits the task workspace;
* reward is computed in-process from verifier artifacts and sent back through
  ``session_runtime.complete_session(..., reward_info=...)``.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import lzma
import os
import shlex
import shutil
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import yaml

from uni_agent.interaction.env import AgentEnv, AgentEnvConfig

from examples.triton_agent.anthropic_openai_shim import AnthropicOpenAIShim
from examples.triton_agent.reward import (
    DEFAULT_WORKSPACE_DIR,
    archive_text_artifacts,
    evaluate_triton_workspace,
)

logger = logging.getLogger(__name__)

_PORT_COUNTER = 18000
_PORT_LOCK = threading.Lock()
_REMOTE_SANDBOX_LOCK = threading.Lock()
_REMOTE_SANDBOX_IN_USE: set[int] = set()
_PROGRESS_LOCK = threading.Lock()
_PROCESS_COMPLETED_ROLLOUTS = 0


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False", "no", "")

WORKSPACE_ROOT = Path(__file__).resolve().parent / "workspace"
WORKSPACE_TEMP_ROOT = Path(__file__).resolve().parent / "workspace_temp"
CANNBOT_SKILLS_LABEL = "bundled local CANNBot Triton/NPU skills snapshot"


KERNELBENCH_INSTRUCTION_TEMPLATE = """# KernelBench Operator Task

Operator: {op_name}
Target: {arch}
Create: `src/{op_name}_triton_ascend_impl.py`

This is a concrete operator task. Do not ask which operator to implement.
Use only the local files bundled in this workspace; do not fetch, clone, or
search remote repositories for skills or references.

Reference behavior is defined by `src/{op_name}.py`. Read that file and the
sidecar case files in `src/` only as needed.

Workflow:
- Read `INSTRUCTIONS.md` and `CLAUDE.md`.
- Phase 1 task extraction is already complete. Do not call `triton-task-extractor`.
- Follow the bundled upstream `CLAUDE.md`, `.claude/skills`, and `.claude/refs`
  for implementation and verifier rules.
- Use `$OPERATOR_PYTHON` through `bash tools/run_npu_command.sh` for NPU
  verify/benchmark when it is set. A plain `python3` missing torch is not task
  success or proof that validation is impossible.
- Do not use task-management tools such as `TaskCreate` or `TaskUpdate`.
- Use the local `Skill` tool only when it helps find bundled guidance; do not
  treat reading a skill as validation.
- Do not write `IMPLEMENTATION_SUMMARY.md` or `IMPLEMENTATION_STATUS.md`.
- Leave the upstream artifacts such as `verify_result.json`, `perf_result.json`,
  `summary.json`, and the final generated implementation in the workspace.

Validation commands:
- AST check:
  `python3 .claude/skills/triton-op-verifier/scripts/validate_triton_impl.py src/{op_name}_triton_ascend_impl.py --json`
- Stage files:
  `mkdir -p output/verify && cp src/{op_name}.py output/verify/{op_name}_torch.py && cp src/{op_name}_triton_ascend_impl.py output/verify/{op_name}_triton_ascend_impl.py && {{ cp src/*.json src/*.jsonl output/verify/ 2>/dev/null || true; }}`
- Verify:
  `PY="${{OPERATOR_PYTHON:-/opt/conda/envs/evaluator-py311/bin/python}}" && bash tools/run_npu_command.sh "$PY" .claude/skills/triton-op-verifier/scripts/verify.py --op_name {op_name} --verify_dir output/verify --triton_impl_name triton_ascend_impl --timeout 900 --output output/verify/verify_result.json`
- After verifier failure, use `output/verify/verify_result_summary.json` or the
  compact printed summary for repair. Do not read full `verify_result.json`,
  full `*.raw.log`, or verifier source unless the compact summary is missing.
- Stop only after verifier reports `passed_cases == total_cases`. A passing AST
  check is not completion.

Common pitfalls:
- Launch Triton kernels by passing tensors directly, for example
  `kernel[grid](x, out, ...)`. Do not pass `x.data_ptr()`.
- `Unsupported ptr type ... in tl.load` means an integer pointer was passed to
  the kernel, usually via `.data_ptr()`. This is an implementation bug, not an
  environment issue.
- Prefer module-level `@triton.jit` kernels launched as `kernel[grid](...)`;
  avoid `self._kernel[grid](...)`.
- AST check is only a precheck. Compilation or correctness failures from
  `verify.py` are implementation failures to repair.
- Do not assume same-shape flattened elementwise behavior when the reference or
  JSON cases include broadcasting, `dim`, reductions, or shape changes.

Final response: one concise status sentence only. No summary markdown.

Task context: {instruction}
"""


DEFAULT_CLAUDE_PROMPT = """Concrete Triton Ascend operator task.
Read `INSTRUCTIONS.md` and `CLAUDE.md`, then implement `ModelNew`.
Use Read/Edit/Write/Bash for edits and validation; use local Skill only when
needed to find bundled guidance, and do not use Task/Workflow tools. Follow bundled
`CLAUDE.md`, local skill files, and refs for implementation rules. Use only the
compact verifier summary after failures; do not read full verifier JSON, raw
logs, or verifier source unless the compact summary is missing. Do not write
implementation summary/status markdown files. Keep the final response to one
short status sentence. Do not stop after AST success; only verifier
passed_cases == total_cases is success."""

DEFAULT_CLAUDE_REPAIR_PROMPT = """Repair the existing Triton Ascend implementation.
The previous attempt did not pass verifier correctness. Read the current
implementation, `INSTRUCTIONS.md`, and the compact verifier summary if present.
Fix the implementation, rerun the validation commands from `INSTRUCTIONS.md`,
and stop only when verifier reports `passed_cases == total_cases`. Do not write
summary/status markdown files or explain away verifier failures."""


def _allocate_port() -> int:
    global _PORT_COUNTER
    import socket

    with _PORT_LOCK:
        while True:
            port = _PORT_COUNTER
            _PORT_COUNTER += 1
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue


def _deep_merge(base: dict, overrides: dict) -> dict:
    if not isinstance(base, dict) or not isinstance(overrides, dict):
        return overrides
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _merge_env_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge env config, replacing deployment when its discriminator changes."""
    result = _deep_merge(dict(base), dict(overrides))
    base_deployment = base.get("deployment") if isinstance(base.get("deployment"), dict) else {}
    override_deployment = overrides.get("deployment") if isinstance(overrides.get("deployment"), dict) else {}
    if override_deployment:
        base_type = base_deployment.get("type")
        override_type = override_deployment.get("type")
        if override_type and override_type != base_type:
            result["deployment"] = dict(override_deployment)
    return result


def load_agent_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if isinstance(loaded, list):
        return loaded[0] if loaded else {}
    return loaded


def _normalize_remote_host(host: str) -> str:
    host = str(host).strip().rstrip("/")
    if not host:
        raise ValueError("Remote sandbox host is empty")
    if "://" not in host:
        host = f"http://{host}"
    parsed = urlparse(host)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError(f"Invalid remote sandbox host: {host!r}")
    hostname = parsed.hostname
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    return f"{parsed.scheme}://{hostname}"


def _remote_endpoint_from_url(text: str, auth_token: str | None = None) -> dict[str, Any]:
    raw = text.strip()
    token = auth_token
    if token is None:
        for sep in ("|", "="):
            if sep in raw:
                raw, token = raw.rsplit(sep, 1)
                break
    if token is None:
        raw, port, token = raw.rsplit(":", 2)
        raw = f"{raw}:{port}"
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    if parsed.port is None:
        raise ValueError(f"Remote sandbox endpoint must include a port: {text!r}")
    return {
        "type": "local_attach",
        "host": _normalize_remote_host(f"{parsed.scheme}://{parsed.hostname}"),
        "port": int(parsed.port),
        "auth_token": str(token),
    }


def _coerce_remote_endpoint(endpoint: Any) -> dict[str, Any]:
    if isinstance(endpoint, str):
        return _remote_endpoint_from_url(endpoint)
    if not isinstance(endpoint, dict):
        raise TypeError(f"Remote sandbox endpoint must be dict or str, got {type(endpoint)!r}")

    if "url" in endpoint:
        result = _remote_endpoint_from_url(str(endpoint["url"]), endpoint.get("auth_token"))
    else:
        result = {
            "type": "local_attach",
            "host": _normalize_remote_host(str(endpoint["host"])),
            "port": int(endpoint["port"]),
            "auth_token": str(endpoint["auth_token"]),
        }
    for key in ("timeout", "startup_timeout", "proxy"):
        if key in endpoint and endpoint[key] not in (None, ""):
            result[key] = endpoint[key]
    return result


def _pool_from_env() -> list[dict[str, Any]]:
    raw_pool = os.environ.get("TRITON_REMOTE_SANDBOX_POOL_JSON") or os.environ.get("TRITON_REMOTE_SANDBOX_POOL")
    if raw_pool:
        try:
            loaded = json.loads(raw_pool)
            if isinstance(loaded, dict):
                loaded = loaded.get("endpoints", [loaded])
            return [_coerce_remote_endpoint(item) for item in loaded]
        except json.JSONDecodeError:
            entries = [entry.strip() for entry in raw_pool.replace("\n", ",").split(",") if entry.strip()]
            return [_coerce_remote_endpoint(entry) for entry in entries]

    host = os.environ.get("TRITON_REMOTE_SANDBOX_HOST")
    ports = os.environ.get("TRITON_REMOTE_SANDBOX_PORTS")
    if not host or not ports:
        return []
    tokens_raw = os.environ.get("TRITON_REMOTE_SANDBOX_AUTH_TOKENS") or os.environ.get("TRITON_REMOTE_SANDBOX_AUTH_TOKEN")
    if not tokens_raw:
        raise ValueError("TRITON_REMOTE_SANDBOX_AUTH_TOKEN(S) is required when TRITON_REMOTE_SANDBOX_PORTS is set")
    port_list = [int(port.strip()) for port in ports.split(",") if port.strip()]
    token_list = [token.strip() for token in tokens_raw.split(",") if token.strip()]
    if len(token_list) == 1:
        token_list = token_list * len(port_list)
    if len(port_list) != len(token_list):
        raise ValueError("TRITON_REMOTE_SANDBOX_PORTS and TRITON_REMOTE_SANDBOX_AUTH_TOKENS lengths differ")
    return [
        {
            "type": "local_attach",
            "host": _normalize_remote_host(host),
            "port": port,
            "auth_token": token,
        }
        for port, token in zip(port_list, token_list, strict=True)
    ]


def _remote_sandbox_pool(tools_kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    cc_cfg = tools_kwargs.get("claude_code") if isinstance(tools_kwargs.get("claude_code"), dict) else {}
    pool = cc_cfg.get("remote_sandbox_pool") or (tools_kwargs.get("env") or {}).get("remote_sandbox_pool")
    if pool:
        return [_coerce_remote_endpoint(endpoint) for endpoint in pool]
    return _pool_from_env()


async def _acquire_remote_sandbox(pool: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    wait_timeout = float(os.environ.get("TRITON_REMOTE_SANDBOX_WAIT_TIMEOUT", "3600"))
    deadline = time.time() + wait_timeout
    while True:
        with _REMOTE_SANDBOX_LOCK:
            for idx, endpoint in enumerate(pool):
                if idx not in _REMOTE_SANDBOX_IN_USE:
                    _REMOTE_SANDBOX_IN_USE.add(idx)
                    return idx, endpoint
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for a free remote sandbox endpoint after {wait_timeout}s")
        await asyncio.sleep(1.0)


def _release_remote_sandbox(index: int | None) -> None:
    if index is None:
        return
    with _REMOTE_SANDBOX_LOCK:
        _REMOTE_SANDBOX_IN_USE.discard(index)


def _create_agent_env(run_id: str, tools_kwargs: dict[str, Any], agent_config: dict[str, Any]) -> AgentEnv:
    env_config = _merge_env_config(dict(agent_config.get("env", {})), dict(tools_kwargs.get("env", {})))
    deployment = dict(env_config.get("deployment", {}))
    deployment.setdefault("type", "local")
    if deployment.get("type") == "local" and "published_port" not in deployment:
        deployment["published_port"] = _allocate_port()
    env_config["deployment"] = deployment
    return AgentEnv(run_id=run_id, env_config=AgentEnvConfig(**env_config))


def _task_metadata(raw_prompt: Any, tools_kwargs: dict[str, Any]) -> dict[str, Any]:
    reward_cfg = tools_kwargs.get("reward") or {}
    metadata = dict(reward_cfg.get("metadata") or {})
    if metadata:
        return metadata

    extra_info = tools_kwargs.get("extra_info")
    if isinstance(extra_info, dict):
        return dict(extra_info)

    if isinstance(raw_prompt, list):
        text = "\n".join(str(message.get("content", "")) for message in raw_prompt if isinstance(message, dict))
    else:
        text = str(raw_prompt)
    return {"instruction": text, "op_name": "operator", "task_code": "", "arch": "ascend910b1"}


def _setup_host_workspace(task: dict[str, Any], trace_label: str) -> Path:
    WORKSPACE_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix=f"trajectory-{trace_label}-", dir=WORKSPACE_TEMP_ROOT))
    shutil.copytree(WORKSPACE_ROOT, workspace, dirs_exist_ok=True)

    op_name = task.get("op_name", "operator")
    arch = task.get("arch", "ascend910b1")
    instruction = task.get("instruction", "Implement the requested operator.")
    task_code = task.get("task_code", "")

    agent_dir = workspace / "agent_workdir"
    _prepare_claude_project_skills(agent_dir)
    src_dir = agent_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "INSTRUCTIONS.md").write_text(
        KERNELBENCH_INSTRUCTION_TEMPLATE.format(
            op_name=op_name,
            arch=arch,
            task_code=task_code,
            instruction=instruction,
        ),
        encoding="utf-8",
    )
    if task_code:
        (src_dir / f"{op_name}.py").write_text(task_code, encoding="utf-8")

    support_files = task.get("support_files")
    if isinstance(support_files, dict):
        for name, content in support_files.items():
            safe_name = Path(str(name)).name
            (src_dir / safe_name).write_text(str(content), encoding="utf-8")

    return workspace


def _prepare_claude_project_skills(agent_dir: Path) -> None:
    """Expose CANNBot skills through Claude Code's native project path."""

    claude_dir = agent_dir / ".claude"
    claude_skills = claude_dir / "skills"
    claude_dir.mkdir(parents=True, exist_ok=True)

    if not claude_skills.is_dir():
        raise FileNotFoundError(f"Claude skills directory not found: {claude_skills}")

    source_note = (
        "# Local CANNBot Skills\n\n"
        f"These skills are provided as a {CANNBOT_SKILLS_LABEL}.\n"
        "Use only the local files in this workspace. Do not fetch, clone, or search for remote skills.\n"
        "Do not edit skill or reference files during rollout.\n"
    )
    (claude_dir / "CANNBOT_SKILLS_SOURCE.md").write_text(source_note, encoding="utf-8")

    for stale_doc in ("AGENTS.md", "RUNTIME_VALIDATION.md"):
        path = agent_dir / stale_doc
        if path.is_file() or path.is_symlink():
            path.unlink()


def _tar_directory(source_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(source_dir, arcname="workspace")
    return buffer.getvalue()


async def _upload_workspace(env: AgentEnv, source_dir: Path, target_root: str = "/opt") -> str:
    archive_name = f"triton_workspace_{uuid4().hex}.tar.gz"
    container_archive = f"/tmp/{archive_name}"
    with tempfile.TemporaryDirectory() as tmp_dir:
        archive_path = Path(tmp_dir) / archive_name
        archive_path.write_bytes(_tar_directory(source_dir))
        await env.copy_to_container(archive_path, Path(container_archive))
    await env.communicate(
        f"rm -rf {shlex.quote(target_root)}/workspace && "
        f"tar -xzf {shlex.quote(container_archive)} -C {shlex.quote(target_root)} && "
        f"rm -f {shlex.quote(container_archive)}",
        check="raise",
        error_msg="Failed to extract Triton workspace",
    )
    return f"{target_root}/workspace/agent_workdir"


async def _command_ok(env: AgentEnv, command: str, timeout: int = 30) -> bool:
    output = await env.communicate(f"{command} >/tmp/command_check.out 2>&1; echo $?", timeout=timeout, check="ignore")
    return output.strip().splitlines()[-1:] == ["0"]


def _plain_tarball(host_tarball: Path) -> Path:
    host_tarball = host_tarball.expanduser()
    if host_tarball.suffix != ".xz":
        return host_tarball
    plain = Path(tempfile.gettempdir()) / f"triton_claude_code.{host_tarball.stem}.tar"
    if plain.exists() and plain.stat().st_mtime >= host_tarball.stat().st_mtime:
        return plain
    tmp = plain.with_suffix(".tar.partial")
    with lzma.open(host_tarball, "rb") as src, tmp.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(tmp, plain)
    return plain


async def _copy_host_file(env: AgentEnv, host_path: Path, remote_path: str) -> None:
    if not host_path.is_file():
        raise FileNotFoundError(f"Required host file not found: {host_path}")
    await env.copy_to_container(host_path, Path(remote_path))


async def _install_claude_code(env: AgentEnv) -> None:
    if os.environ.get("TRITON_CLAUDE_SKIP_INSTALL", "0") in ("1", "true", "True", "yes"):
        return
    if await _command_ok(env, "command -v claude && claude --version"):
        return

    node_tarball = os.environ.get("TRITON_CLAUDE_NODE_TARBALL") or os.environ.get("SWE_HOST_NODE_TARBALL")
    claude_tarball = os.environ.get("TRITON_CLAUDE_CODE_TARBALL") or os.environ.get("SWE_HOST_CC_TARBALL")
    if not node_tarball or not claude_tarball:
        raise RuntimeError(
            "Claude Code is not installed in the sandbox. Set TRITON_CLAUDE_NODE_TARBALL "
            "and TRITON_CLAUDE_CODE_TARBALL, or build an image that already has `claude`."
        )

    node_host = _plain_tarball(Path(node_tarball))
    claude_host = Path(claude_tarball).expanduser()
    await _copy_host_file(env, node_host, "/tmp/node22.tar")
    await _copy_host_file(env, claude_host, "/tmp/claude-code.tgz")
    await env.communicate(
        "set -e && mkdir -p /opt/node22 && "
        "tar xf /tmp/node22.tar -C /opt/node22 --strip-components=1 && "
        "ln -sf /opt/node22/bin/node /usr/local/bin/node && "
        "ln -sf /opt/node22/bin/npm /usr/local/bin/npm && "
        "ln -sf /opt/node22/bin/npx /usr/local/bin/npx && "
        "npm install -g --prefix=/usr/local --no-audit --no-fund /tmp/claude-code.tgz && "
        "claude --version",
        timeout=600,
        check="raise",
        error_msg="Failed to install Claude Code",
    )


def _claude_run_user(tools_kwargs: dict[str, Any]) -> str:
    cc_cfg = tools_kwargs.get("claude_code") if isinstance(tools_kwargs.get("claude_code"), dict) else {}
    return str(cc_cfg.get("run_user") or os.environ.get("TRITON_CLAUDE_RUN_USER", "")).strip()


def _claude_run_home(run_user: str, tools_kwargs: dict[str, Any]) -> str | None:
    cc_cfg = tools_kwargs.get("claude_code") if isinstance(tools_kwargs.get("claude_code"), dict) else {}
    configured = cc_cfg.get("run_home") or os.environ.get("TRITON_CLAUDE_RUN_HOME")
    if configured:
        return str(configured)
    if run_user == "root":
        return "/root"
    if run_user:
        return f"/home/{run_user}"
    return None


async def _ensure_claude_run_user(env: AgentEnv, run_user: str, run_home: str | None) -> None:
    if not run_user:
        return
    if run_user == "root":
        await env.communicate("mkdir -p /root", timeout=60, check="ignore")
        return
    run_uid = str(os.environ.get("TRITON_CLAUDE_RUN_UID", "1000"))
    run_gid = str(os.environ.get("TRITON_CLAUDE_RUN_GID", "1000"))
    run_home = run_home or f"/home/{run_user}"
    script = f"""
set -e
user={shlex.quote(run_user)}
uid={shlex.quote(run_uid)}
gid={shlex.quote(run_gid)}
home={shlex.quote(run_home)}
if [ "$(id -u)" = "0" ]; then
    if ! getent group "$gid" >/dev/null 2>&1; then
        groupadd -g "$gid" "$user" 2>/dev/null || true
    fi
    if ! id -u "$user" >/dev/null 2>&1; then
        if command -v useradd >/dev/null 2>&1; then
            useradd -m -u "$uid" -g "$gid" -s /bin/bash "$user"
        elif command -v adduser >/dev/null 2>&1; then
            adduser --disabled-password --gecos "" --home "$home" --uid "$uid" --gid "$gid" "$user"
        else
            echo "Neither useradd nor adduser is available; cannot create $user" >&2
            exit 1
        fi
    fi
    mkdir -p "$home"
    chown -R "$user" "$home"
else
    mkdir -p "$home" 2>/dev/null || true
fi
"""
    await env.communicate(script, timeout=60, check="raise", error_msg=f"Failed to prepare Claude run user {run_user}")


async def _prepare_claude_home(env: AgentEnv, run_user: str = "", run_home: str | None = None) -> None:
    settings = json.dumps({"hasCompletedOnboarding": True, "bypassPermissionsModeAccepted": True})
    home = run_home or ("${HOME:-/root}" if not run_user else f"/home/{run_user}")
    quoted_home = shlex.quote(home)
    chown_cmd = ""
    if run_user:
        chown_cmd = f" && chown -R {shlex.quote(run_user)} {quoted_home}/.claude {quoted_home}/.claude.json"
    await env.communicate(
        f"mkdir -p {quoted_home}/.claude && "
        f"printf %s {shlex.quote(settings)} > {quoted_home}/.claude.json && "
        f"printf %s {shlex.quote(settings)} > {quoted_home}/.claude/settings.json"
        f"{chown_cmd}",
        check="ignore",
    )


def _shell_exports(env_vars: dict[str, str]) -> str:
    return "\n".join(f"export {key}={shlex.quote(value)}" for key, value in env_vars.items())


def _ascend_env_shell_snippet() -> str:
    return r"""
setup_triton_ascend_env() {
    : "${CONDA_BASE:=/opt/conda}"
    : "${OPERATOR_CONDA_ENV:=evaluator-py311}"
    : "${WORKSPACE_BASE:=/opt/workspace/agent_workdir}"
    local conda_env="${CONDA_BASE}/envs/${OPERATOR_CONDA_ENV}"

    local conda_python="${conda_env}/bin/python"
    if [ -x "$conda_python" ]; then
        export OPERATOR_PYTHON="$conda_python"
    elif [ -z "${OPERATOR_PYTHON:-}" ]; then
        export OPERATOR_PYTHON=python3
    fi
    export AST_CHECK_PYTHON="${AST_CHECK_PYTHON:-python3}"
    export WORKSPACE_BASE

    first_csv_value() {
        local value="${1:-}"
        value="${value%%,*}"
        printf '%s' "$value"
    }

    if [ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
        if [ -n "${ALLOCATED_DEVICE_ID:-}" ]; then
            export ASCEND_RT_VISIBLE_DEVICES="$(first_csv_value "$ALLOCATED_DEVICE_ID")"
        elif [ -n "${EVAL_DEVICE_IDS:-}" ]; then
            export ASCEND_RT_VISIBLE_DEVICES="$(first_csv_value "$EVAL_DEVICE_IDS")"
        fi
    fi
    if [ -z "${ALLOCATED_DEVICE_ID:-}" ] && [ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
        export ALLOCATED_DEVICE_ID="$(first_csv_value "$ASCEND_RT_VISIBLE_DEVICES")"
    fi
}
setup_triton_ascend_env
"""


def _claude_disallowed_tools(cc_cfg: dict[str, Any]) -> list[str]:
    default_tools = [
        "Task",
        "TaskCreate",
        "TaskUpdate",
        "TaskList",
        "TaskGet",
        "TaskOutput",
        "TaskStop",
        "Workflow",
        "AskUserQuestion",
        "EnterPlanMode",
        "ExitPlanMode",
    ]
    raw = cc_cfg.get("disallowed_tools")
    if raw is None:
        raw = os.environ.get("TRITON_CLAUDE_DISALLOWED_TOOLS")
    if raw is None:
        return default_tools
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    text = str(raw).strip()
    if text in ("", "0", "none", "None", "false", "False"):
        return []
    return [item.strip() for item in text.replace(",", " ").split() if item.strip()]


def _progress_run_id() -> str:
    return os.environ.get("TRITON_PROGRESS_RUN_ID") or os.environ.get("RAY_JOB_ID") or "default"


def _progress_counter_file() -> str:
    return os.environ.get("TRITON_PROGRESS_COUNTER_FILE", "/tmp/triton_claude_code_progress.jsonl")


def _gate_exit_code(exit_code: int, eval_result: dict[str, Any]) -> int:
    if not _truthy_env("TRITON_REQUIRE_CORRECTNESS", "1"):
        return exit_code
    metrics = eval_result.get("metrics") if isinstance(eval_result.get("metrics"), dict) else {}
    correctness_ok = bool(metrics.get("correctness_ok") or metrics.get("success"))
    resolved = bool(eval_result.get("resolved"))
    if not (correctness_ok and resolved):
        return exit_code if exit_code not in (0, None) else 1
    return exit_code


def _claude_repair_rounds(tools_kwargs: dict[str, Any]) -> int:
    cc_cfg = tools_kwargs.get("claude_code") if isinstance(tools_kwargs.get("claude_code"), dict) else {}
    value = cc_cfg.get("repair_rounds", os.environ.get("TRITON_CLAUDE_REPAIR_ROUNDS", "1"))
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _eval_result_resolved(eval_result: dict[str, Any]) -> bool:
    metrics = eval_result.get("metrics") if isinstance(eval_result.get("metrics"), dict) else {}
    return bool(eval_result.get("resolved") or metrics.get("success") or metrics.get("correctness_ok"))


def _compact_eval_feedback(eval_result: dict[str, Any]) -> str:
    metrics = eval_result.get("metrics") if isinstance(eval_result.get("metrics"), dict) else {}
    fields = {
        "reason": eval_result.get("reason"),
        "reward": eval_result.get("reward"),
        "error_type": metrics.get("error_type"),
        "ast_check_ok": metrics.get("ast_check_ok"),
        "compile_ok": metrics.get("compile_ok"),
        "correctness_ok": metrics.get("correctness_ok"),
        "passed_cases": metrics.get("passed_cases"),
        "total_cases": metrics.get("total_cases"),
        "error": metrics.get("error"),
    }
    text = json.dumps(fields, ensure_ascii=False, sort_keys=True)
    max_chars = int(os.environ.get("TRITON_CLAUDE_REPAIR_FEEDBACK_CHARS", "2000"))
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars] + "...[truncated]"
    return text


def _with_repair_prompt(tools_kwargs: dict[str, Any], eval_result: dict[str, Any], repair_round: int) -> dict[str, Any]:
    prompt = os.environ.get("TRITON_CLAUDE_REPAIR_PROMPT") or DEFAULT_CLAUDE_REPAIR_PROMPT
    prompt = (
        f"{prompt.rstrip()}\n\n"
        f"Repair round: {repair_round}\n"
        f"Compact evaluator result:\n{_compact_eval_feedback(eval_result)}"
    )
    cc_cfg = dict(tools_kwargs.get("claude_code") or {})
    cc_cfg["prompt"] = prompt
    return {**tools_kwargs, "claude_code": cc_cfg}


def _count_progress_lines(file_obj, run_id: str) -> int:
    file_obj.seek(0)
    count = 0
    for line in file_obj:
        line = line.strip()
        if not line:
            continue
        try:
            if json.loads(line).get("run_id") == run_id:
                count += 1
        except json.JSONDecodeError:
            continue
    return count


def _record_rollout_progress(
    *,
    sample_index: int,
    metadata: dict[str, Any],
    reward_info: dict[str, Any],
    exit_code: int,
    archived_dir: str | None,
) -> tuple[int, int | None]:
    global _PROCESS_COMPLETED_ROLLOUTS

    with _PROGRESS_LOCK:
        _PROCESS_COMPLETED_ROLLOUTS += 1
        process_completed = _PROCESS_COMPLETED_ROLLOUTS

    run_id = _progress_run_id()
    counter_path = _progress_counter_file()
    if not counter_path:
        return process_completed, None

    metrics = reward_info.get("metrics") if isinstance(reward_info.get("metrics"), dict) else {}
    event = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_id": run_id,
        "sample_index": sample_index,
        "uid": metadata.get("uid"),
        "op_name": metadata.get("op_name"),
        "reward": reward_info.get("reward_score"),
        "claude_code_exit_code": exit_code,
        "passed": reward_info.get("passed"),
        "resolved": reward_info.get("resolved"),
        "accuracy": reward_info.get("accuracy"),
        "latency": metrics.get("latency") or metrics.get("avg_latency") or metrics.get("median_latency"),
        "reason": reward_info.get("reason"),
        "archived_dir": archived_dir,
    }
    path = Path(counter_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, sort_keys=True)

    try:
        with path.open("a+", encoding="utf-8") as f:
            if os.name == "posix":
                try:
                    import fcntl

                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except Exception:
                    pass
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
            global_completed = _count_progress_lines(f, run_id)
            if os.name == "posix":
                try:
                    import fcntl

                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
        return process_completed, global_completed
    except Exception:
        logger.debug("Failed to record rollout progress", exc_info=True)
        return process_completed, None


async def _run_claude_code(
    env: AgentEnv,
    *,
    workspace_dir: str,
    shim_url: str,
    session_id: str,
    tools_kwargs: dict[str, Any],
    metadata: dict[str, Any],
) -> int:
    cc_cfg = dict(tools_kwargs.get("claude_code") or {})
    prompt = cc_cfg.get("prompt") or os.environ.get("TRITON_CLAUDE_PROMPT") or DEFAULT_CLAUDE_PROMPT
    time_budget = int(cc_cfg.get("time_budget_sec") or os.environ.get("TRITON_CLAUDE_TIME_BUDGET_SEC", "1800"))
    heartbeat_sec = int(cc_cfg.get("heartbeat_sec") or os.environ.get("TRITON_CLAUDE_HEARTBEAT_SEC", "0"))
    extra_args = str(cc_cfg.get("extra_args") or os.environ.get("TRITON_CLAUDE_EXTRA_ARGS", "")).strip()
    model_name = str(cc_cfg.get("model") or os.environ.get("TRITON_CLAUDE_MODEL", "uni-agent-actor"))
    disallowed_tools = _claude_disallowed_tools(cc_cfg)
    disallowed_args = " ".join(shlex.quote(tool) for tool in disallowed_tools)
    run_user = _claude_run_user(tools_kwargs)
    run_home = _claude_run_home(run_user, tools_kwargs)
    trace_id = str(metadata.get("uid") or metadata.get("op_name") or "unknown")
    verbose_logs = _truthy_env("TRITON_VERBOSE_ROLLOUT_LOGS")
    op_name = str(metadata.get("op_name", "operator"))
    arch = str(metadata.get("arch", "ascend910b1"))
    prompt = (
        str(prompt).rstrip()
        + "\n\nRuntime task context:\n"
        + f"- Operator name: `{op_name}`\n"
        + f"- Target arch: `{arch}`\n"
        + f"- Reference task file: `src/{op_name}.py`\n"
        + f"- Implementation target: `src/{op_name}_triton_ascend_impl.py`\n"
        + "- Use only bundled local skill/reference files; do not fetch or search remote repositories.\n"
        + "- Never emit literal `<tool_call>` tags or JSON/XML tool-call text; use real Claude Code tools.\n"
        + "- Phase 1 task extraction is already complete; do not invoke `triton-task-extractor`.\n"
        + "- Do not use Task, TaskCreate, TaskUpdate, TaskList, TaskGet, TaskOutput, TaskStop, Workflow, AskUserQuestion, EnterPlanMode, or ExitPlanMode; use Skill only for local guidance when needed.\n"
        + "- Follow bundled `CLAUDE.md`, local PR205 skills, and refs for implementation and verifier rules.\n"
        + "- Use the compact validation commands in `INSTRUCTIONS.md`; do not read full verifier JSON, raw logs, or verifier source unless the compact summary is missing.\n"
        + "- Use `$OPERATOR_PYTHON` through `bash tools/run_npu_command.sh` for NPU verify/benchmark when set; plain `python3` missing torch is not success.\n"
        + "- Do not use `src` directly as verify_dir; stage the torch reference, implementation, and any src/*.json or src/*.jsonl sidecars into `output/verify` first.\n"
        + "- Pass tensors directly to Triton kernels, not `.data_ptr()` values; prefer module-level `kernel[grid](...)` launches.\n"
        + "- Treat `Unsupported ptr type ... in tl.load` as an implementation bug caused by integer pointers, not as an environment issue.\n"
        + "- Treat upstream verifier outputs as authoritative; only passed_cases == total_cases is success.\n"
        + "- Do not create IMPLEMENTATION_SUMMARY.md or IMPLEMENTATION_STATUS.md.\n"
        + "- Keep the final response to one short status sentence.\n"
        + "- Start with tool actions, not a clarification question: read `INSTRUCTIONS.md` and `CLAUDE.md`.\n"
    )

    done = f"{workspace_dir}/.claude_code_done"
    launcher = f"{workspace_dir}/.claude_code_run.sh"
    trajectory = f"{workspace_dir}/claude_code_trajectory.jsonl"
    stdout_log = f"{workspace_dir}/claude_code_stdout.log"

    env_vars = {
        "ANTHROPIC_BASE_URL": shim_url,
        "ANTHROPIC_AUTH_TOKEN": session_id,
        "ANTHROPIC_API_KEY": session_id,
        "ANTHROPIC_MODEL": model_name,
        "OPERATOR_BACKEND": str(metadata.get("operator_backend", "triton")),
        "OPERATOR_ARCH": str(metadata.get("arch", "ascend910b1")),
        "OPERATOR_NAME": op_name,
        "CONDA_BASE": os.environ.get("CONDA_BASE", "/opt/conda"),
        "OPERATOR_CONDA_ENV": os.environ.get("OPERATOR_CONDA_ENV", "evaluator-py311"),
        "OPERATOR_PYTHON": os.environ.get("OPERATOR_PYTHON", "/opt/conda/envs/evaluator-py311/bin/python"),
        "WORKSPACE_BASE": workspace_dir,
        "TRITON_PIPELINE_ERROR_PREVIEW_CHARS": os.environ.get("TRITON_PIPELINE_ERROR_PREVIEW_CHARS", "2000"),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    }
    if run_home:
        env_vars["HOME"] = run_home
    launcher_body = (
        "#!/usr/bin/env bash\n"
        "set -o pipefail\n"
        f"cd {shlex.quote(workspace_dir)}\n"
        f"{_shell_exports(env_vars)}\n"
        f"{_ascend_env_shell_snippet()}\n"
        f"claude -p {shlex.quote(prompt)} "
        "--permission-mode bypassPermissions "
        f"{('--disallowedTools ' + disallowed_args + ' ') if disallowed_args else ''}"
        "--output-format stream-json --include-partial-messages "
        "--include-hook-events --verbose "
        f"{extra_args} "
        f"2>&1 | tee -a {shlex.quote(trajectory)} {shlex.quote(stdout_log)}\n"
        "ec=${PIPESTATUS[0]}\n"
        f"echo $ec > {shlex.quote(done)}\n"
    )
    await env.write_file(launcher, launcher_body)
    await env.communicate(f"chmod +x {shlex.quote(launcher)}", check="raise")
    if run_user and run_user != "root":
        await env.communicate(
            f"chown -R {shlex.quote(run_user)} {shlex.quote(workspace_dir)}",
            timeout=60,
            check="raise",
            error_msg=f"Failed to chown Claude workspace to {run_user}",
        )
    if run_user and run_user != "root":
        run_user_env = (
            'env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-}" '
            'ALLOCATED_DEVICE_ID="${ALLOCATED_DEVICE_ID:-}" '
            'EVAL_DEVICE_IDS="${EVAL_DEVICE_IDS:-}" '
            'EVAL_DEVICE_COUNT="${EVAL_DEVICE_COUNT:-}" '
            'EVAL_ENV_NAME="${EVAL_ENV_NAME:-}" '
        )
        su_inner = f"{run_user_env} bash {shlex.quote(launcher)}"
        start_cmd = (
            f"rm -f {shlex.quote(done)} && "
            'if [ "$(id -u)" = "0" ]; then '
            f"if command -v runuser >/dev/null 2>&1; then "
            f"(nohup runuser -u {shlex.quote(run_user)} -- {run_user_env} bash {shlex.quote(launcher)} >/dev/null 2>&1 &); "
            f"else (nohup su -s /bin/bash {shlex.quote(run_user)} -c {shlex.quote(su_inner)} >/dev/null 2>&1 &); "
            "fi; "
            "else "
            f"(nohup bash {shlex.quote(launcher)} >/dev/null 2>&1 &); "
            "fi"
        )
    else:
        start_cmd = (
            f"rm -f {shlex.quote(done)} && "
            f"(nohup bash {shlex.quote(launcher)} >/dev/null 2>&1 &)"
        )
    log_stage = logger.info if verbose_logs else logger.debug
    log_stage(
        "Starting Claude Code rollout: trace=%s workspace=%s shim_url=%s run_user=%s time_budget=%ss",
        trace_id,
        workspace_dir,
        shim_url,
        run_user or "<current>",
        time_budget,
    )
    await env.communicate(
        start_cmd,
        timeout=30,
        check="ignore",
    )
    log_stage("Claude Code process launched: trace=%s done_file=%s", trace_id, done)

    started_at = time.time()
    deadline = started_at + time_budget
    last_heartbeat = started_at
    exit_code = -2
    while time.time() < deadline:
        await asyncio.sleep(5)
        output = await env.communicate(
            f"if [ -f {shlex.quote(done)} ]; then cat {shlex.quote(done)}; fi",
            timeout=20,
            check="ignore",
        )
        text = output.strip().splitlines()[-1] if output.strip() else ""
        if text:
            try:
                exit_code = int(text)
            except ValueError:
                exit_code = -1
            break
        now = time.time()
        if heartbeat_sec > 0 and now - last_heartbeat >= heartbeat_sec:
            log_stage(
                "Claude Code still running: trace=%s elapsed=%.0fs remaining=%.0fs",
                trace_id,
                now - started_at,
                max(0.0, deadline - now),
            )
            last_heartbeat = now

    if exit_code == -2:
        logger.warning("Claude Code timed out: trace=%s time_budget=%ss", trace_id, time_budget)
        await env.communicate("pkill -f '/usr/local/bin/claude|claude -p' || true", timeout=20, check="ignore")
    else:
        log_stage("Claude Code exited: trace=%s exit_code=%s elapsed=%.0fs", trace_id, exit_code, time.time() - started_at)
    return exit_code


async def _run_claude_code_with_repairs(
    env: AgentEnv,
    *,
    workspace_dir: str,
    shim_url: str,
    session_id: str,
    tools_kwargs: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[int, float, dict[str, Any]]:
    exit_code = await _run_claude_code(
        env,
        workspace_dir=workspace_dir,
        shim_url=shim_url,
        session_id=session_id,
        tools_kwargs=tools_kwargs,
        metadata=metadata,
    )
    score, eval_result = await evaluate_triton_workspace(env, metadata, workspace_dir=workspace_dir)

    for repair_round in range(1, _claude_repair_rounds(tools_kwargs) + 1):
        if _eval_result_resolved(eval_result):
            break
        logger.info(
            "Starting Claude Code repair round: round=%s op=%s reason=%s",
            repair_round,
            metadata.get("op_name"),
            eval_result.get("reason"),
        )
        await env.communicate(
            f"rm -f {shlex.quote(workspace_dir)}/metrics.json {shlex.quote(workspace_dir)}/metrics_error.log",
            timeout=20,
            check="ignore",
        )
        repair_tools_kwargs = _with_repair_prompt(tools_kwargs, eval_result, repair_round)
        exit_code = await _run_claude_code(
            env,
            workspace_dir=workspace_dir,
            shim_url=shim_url,
            session_id=session_id,
            tools_kwargs=repair_tools_kwargs,
            metadata=metadata,
        )
        score, eval_result = await evaluate_triton_workspace(env, metadata, workspace_dir=workspace_dir)
        eval_result["repair_rounds_used"] = repair_round

    return exit_code, score, eval_result


def _shim_public_host(agent_config: dict[str, Any], tools_kwargs: dict[str, Any]) -> str:
    configured = (
        (tools_kwargs.get("claude_code") or {}).get("shim_public_host")
        or os.environ.get("TRITON_SHIM_PUBLIC_HOST")
    )
    if configured:
        return str(configured)
    deployment_type = (
        (tools_kwargs.get("env") or {}).get("deployment", {}).get("type")
        or (agent_config.get("env") or {}).get("deployment", {}).get("type")
    )
    if deployment_type in ("host", "local_native"):
        return "127.0.0.1"
    return "host.docker.internal"


async def triton_claude_code_runner(
    *,
    raw_prompt,
    session,
    sample_index: int,
    session_runtime,
    tools_kwargs: dict | None = None,
    agent_config_path: str | None = None,
    **kwargs,
) -> None:
    """Run one Claude Code Triton rollout through a gateway session."""
    del kwargs
    tools_kwargs = tools_kwargs or {}
    if getattr(session, "base_url", None) is None:
        raise ValueError("session.base_url is required for triton_claude_code_runner")

    config_path = agent_config_path or tools_kwargs.get("agent_config_path")
    agent_config = load_agent_config(config_path)
    metadata = _task_metadata(raw_prompt, tools_kwargs)
    trace_label = str(metadata.get("uid") or metadata.get("op_name") or sample_index).replace("/", "_")[:96]
    run_id = f"triton_cc_{sample_index}_{uuid4().hex[:8]}"
    host_workspace: Path | None = None
    env: AgentEnv | None = None
    remote_sandbox_index: int | None = None
    reward_info: dict[str, Any] = {"reward_score": 0.0}

    try:
        host_workspace = _setup_host_workspace(metadata, trace_label)
        remote_pool = _remote_sandbox_pool(tools_kwargs)
        if remote_pool:
            remote_sandbox_index, endpoint = await _acquire_remote_sandbox(remote_pool)
            (logger.info if _truthy_env("TRITON_VERBOSE_ROLLOUT_LOGS") else logger.debug)(
                "Acquired remote sandbox: sample=%s index=%s endpoint=%s:%s",
                sample_index,
                remote_sandbox_index,
                endpoint.get("host"),
                endpoint.get("port"),
            )
            tools_kwargs = {
                **tools_kwargs,
                "env": _merge_env_config(
                    dict(tools_kwargs.get("env", {})),
                    {"deployment": endpoint},
                ),
            }
        env = _create_agent_env(run_id, tools_kwargs, agent_config)
        await env.start()
        workspace_dir = await _upload_workspace(env, host_workspace)
        await _install_claude_code(env)
        run_user = _claude_run_user(tools_kwargs)
        run_home = _claude_run_home(run_user, tools_kwargs)
        await _ensure_claude_run_user(env, run_user, run_home)
        await _prepare_claude_home(env, run_user=run_user, run_home=run_home)

        bind_host = os.environ.get("TRITON_SHIM_BIND_HOST", "0.0.0.0")
        public_host = _shim_public_host(agent_config, tools_kwargs)
        with AnthropicOpenAIShim(
            openai_base_url=session.base_url,
            openai_api_key=os.environ.get("TRITON_GATEWAY_API_KEY", "EMPTY"),
            model_name=os.environ.get("TRITON_GATEWAY_MODEL_NAME", "default"),
            host=bind_host,
            port=None,
            request_timeout=float(os.environ.get("TRITON_SHIM_REQUEST_TIMEOUT", "600")),
        ) as shim:
            shim_url = f"http://{public_host}:{shim.port}"
            exit_code, score, eval_result = await _run_claude_code_with_repairs(
                env,
                workspace_dir=workspace_dir,
                shim_url=shim_url,
                session_id=session.session_id,
                tools_kwargs=tools_kwargs,
                metadata=metadata,
            )

        exit_code = _gate_exit_code(exit_code, eval_result)
        artifact_dir = (
            (tools_kwargs.get("claude_code") or {}).get("artifact_dir")
            or os.environ.get("TRITON_CLAUDE_ARTIFACT_DIR", "")
        )
        archived_dir = await archive_text_artifacts(env, metadata, str(artifact_dir), workspace_dir=workspace_dir)

        reward_info = {
            "reward_score": score,
            "claude_code_exit_code": exit_code,
            "archived_dir": archived_dir,
            **eval_result,
        }
        process_completed, global_completed = _record_rollout_progress(
            sample_index=sample_index,
            metadata=metadata,
            reward_info=reward_info,
            exit_code=exit_code,
            archived_dir=archived_dir,
        )
        metrics = reward_info.get("metrics") if isinstance(reward_info.get("metrics"), dict) else {}
        progress_message = (
            "rollout progress: completed=%s process_completed=%s sample=%s uid=%s op=%s "
            "reward=%.4f resolved=%s exit_code=%s reason=%s latency=%s archived=%s"
        ) % (
            global_completed if global_completed is not None else "?",
            process_completed,
            sample_index,
            metadata.get("uid"),
            metadata.get("op_name"),
            score,
            reward_info.get("resolved"),
            exit_code,
            reward_info.get("reason", ""),
            metrics.get("latency") or metrics.get("avg_latency") or metrics.get("median_latency"),
            archived_dir,
        )
        logger.info(progress_message)
        if _truthy_env("TRITON_PROGRESS_STDOUT", "1"):
            print(progress_message, flush=True)
        await session_runtime.complete_session(session.session_id, reward_info=reward_info)
    except Exception as exc:
        logger.exception("Triton Claude Code rollout failed for sample %s", sample_index)
        reward_info = {"reward_score": 0.0, "error": str(exc), "error_type": type(exc).__name__}
        try:
            await session_runtime.complete_session(session.session_id, reward_info=reward_info)
        except Exception:
            pass
        raise
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                logger.debug("Failed to close env", exc_info=True)
        _release_remote_sandbox(remote_sandbox_index)
        if host_workspace is not None and os.environ.get("TRITON_CLAUDE_KEEP_HOST_WORKSPACE", "0") not in (
            "1",
            "true",
            "True",
            "yes",
        ):
            shutil.rmtree(host_workspace, ignore_errors=True)
