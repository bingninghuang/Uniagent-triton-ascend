#!/usr/bin/env python3
"""Shared utilities for Triton operator data synthesis (inference-only).

Provides task loading, workspace setup, prompt construction, sandbox creation,
and trajectory collection. Used by both synth_triton_local.py and synth_triton_api.py.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from uni_agent.interaction import (
    AgentEnv,
    AgentEnvConfig,
    AgentInteraction,
    ToolsManager,
    ToolsManagerConfig,
)
from uni_agent.interaction.model import OpenAICompatibleChatModel
from uni_agent.tools import ToolConfig
from uni_agent.deployment import LocalAttachDeploymentConfig

# Reuse reward evaluation from the Triton agent.
from examples.triton_agent.reward import (
    DEFAULT_WORKSPACE_DIR,
    evaluate_triton_workspace,
    reward_breakdown_from_metrics,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR / "workspace"
BENCHMARK_ROOT = SCRIPT_DIR / "benchmarks" / "NPUKernelBench"

DEFAULT_SANDBOX_IMAGE = os.environ.get("TRITON_CLAUDE_IMAGE", "triton-claude-code-env:latest")

# Attach-mode defaults (for connecting to an existing swerex server)
DEFAULT_ATTACH_HOST = os.environ.get("TRITON_ATTACH_HOST", "http://127.0.0.1")
DEFAULT_ATTACH_PORT = int(os.environ.get("TRITON_ATTACH_PORT", "8000"))
DEFAULT_ATTACH_AUTH_TOKEN = os.environ.get("TRITON_ATTACH_AUTH_TOKEN", "mytoken123")
DEFAULT_WORKSPACE_DIR = "/opt/workspace/agent_workdir"
DEFAULT_TOOL_PARSER = "qwen3_coder"

# ---------------------------------------------------------------------------
# Task loading (adapted from prepare_kernelbench_claude_code_data.py)
# ---------------------------------------------------------------------------


def _level_to_int(level: str) -> int | None:
    text = str(level).strip()
    if text.startswith("level_"):
        text = text[len("level_"):]
    elif text.startswith("level"):
        text = text[len("level"):]
    try:
        return int(text)
    except ValueError:
        return None


def _level_dir_names(levels: tuple[str, ...]) -> set[str]:
    names: set[str] = set()
    for level in levels:
        text = str(level).strip()
        if not text:
            continue
        names.add(text)
        names.add(text.replace("level_", "level"))
        value = _level_to_int(text)
        if value is not None:
            names.add(f"level{value}")
            names.add(f"level_{value}")
    return names


def _parse_npukernelbench_filename(path: Path) -> tuple[int | str, str]:
    match = re.match(r"^(\d+)[_-](.+)$", path.stem)
    if match:
        return int(match.group(1)), match.group(2)
    return path.stem, path.stem


_WARMUP_EXCLUDE_KEYWORDS = (
    "conv_transpose", "conv_transposed", "transpose3d", "transposed_3d",
    "conv3d", "3d_convolution", "attention", "transformer", "conv2d", "conv_standard",
)


def load_tasks(
    dataset_path: str = str(BENCHMARK_ROOT),
    levels: str = "level_1",
    max_rows: int | None = None,
    filter_mode: str = "warmup",
) -> list[dict[str, Any]]:
    """Load NPUKernelBench tasks from a local directory.

    Returns a list of task dicts with keys: op_name, arch, task_code, level,
    problem_id, name, support_files, instruction.
    """
    root = Path(dataset_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    parsed_levels = tuple(part.strip() for part in levels.split(",") if part.strip())
    if len(parsed_levels) == 1 and parsed_levels[0].lower() in {"all", "*"}:
        parsed_levels = tuple(f"level_{idx}" for idx in range(5))

    level_names = _level_dir_names(parsed_levels)
    level_dirs = [
        child
        for child in sorted(root.iterdir())
        if child.is_dir() and (not level_names or child.name in level_names)
    ]

    rows: list[dict[str, Any]] = []
    for level_dir in level_dirs:
        level = _level_to_int(level_dir.name)
        for py_file in sorted(level_dir.glob("*.py")):
            problem_id, name = _parse_npukernelbench_filename(py_file)
            support_files: dict[str, str] = {}
            for sidecar in (
                py_file.with_suffix(".json"),
                py_file.with_name(f"{py_file.stem}_all_case.json"),
            ):
                if sidecar.is_file():
                    support_files[sidecar.name] = sidecar.read_text(encoding="utf-8", errors="replace")

            code = py_file.read_text(encoding="utf-8", errors="replace")
            rows.append({
                "code": code,
                "level": level if level is not None else level_dir.name,
                "problem_id": problem_id,
                "name": name,
                "support_files": support_files,
            })

    # Apply filter
    if filter_mode == "warmup":
        filtered = []
        for row in rows:
            text = f"{row['name']}\n{row['code']}".lower()
            if not any(kw in text for kw in _WARMUP_EXCLUDE_KEYWORDS):
                filtered.append(row)
        rows = filtered

    if max_rows is not None:
        rows = rows[:max_rows]

    # Convert to task format
    arch = os.environ.get("TRITON_KERNELBENCH_ARCH", "ascend910b1")
    tasks: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        level = row.get("level", "")
        problem_id = row.get("problem_id", idx)
        name = row.get("name", f"problem_{problem_id}")
        op_name = re.sub(r"[^0-9a-zA-Z_]+", "_", f"kernelbench_l{level}_{problem_id}_{name}").strip("_")[:96]
        if op_name[0].isdigit():
            op_name = f"task_{op_name}"
        instruction = (
            f"Implement KernelBench problem {problem_id}"
            f"{f' ({name})' if name else ''} as an Ascend NPU Triton operator."
        )
        tasks.append({
            "op_name": op_name,
            "arch": arch,
            "task_code": row.get("code", ""),
            "level": level,
            "problem_id": problem_id,
            "name": name,
            "support_files": row.get("support_files", {}),
            "instruction": instruction,
        })

    print(f"[synth] Loaded {len(tasks)} tasks (levels={levels}, filter={filter_mode})")
    return tasks


# ---------------------------------------------------------------------------
# Workspace setup
# ---------------------------------------------------------------------------

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


def setup_workspace(task: dict[str, Any], work_dir: Path) -> Path:
    """Prepare a host-side workspace directory for one task.

    Copies the workspace template, writes INSTRUCTIONS.md, task code, and
    support files.  Returns the ``agent_workdir`` path inside ``work_dir``.
    """
    agent_dir_src = WORKSPACE_ROOT / "agent_workdir"
    agent_dir_dst = work_dir / "agent_workdir"

    if agent_dir_dst.exists():
        shutil.rmtree(agent_dir_dst)
    shutil.copytree(agent_dir_src, agent_dir_dst)

    op_name = task["op_name"]
    arch = task.get("arch", "ascend910b1")
    instruction = task.get("instruction", f"Implement the {op_name} operator.")
    task_code = task.get("task_code", "")

    # Write INSTRUCTIONS.md
    (agent_dir_dst / "INSTRUCTIONS.md").write_text(
        KERNELBENCH_INSTRUCTION_TEMPLATE.format(
            op_name=op_name, arch=arch, instruction=instruction,
        ),
        encoding="utf-8",
    )

    # Write reference code + support files
    src_dir = agent_dir_dst / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    if task_code:
        (src_dir / f"{op_name}.py").write_text(task_code, encoding="utf-8")

    support_files = task.get("support_files")
    if isinstance(support_files, dict):
        for name, content in support_files.items():
            safe_name = Path(str(name)).name
            (src_dir / safe_name).write_text(str(content), encoding="utf-8")

    return agent_dir_dst


# ---------------------------------------------------------------------------
# Workspace upload to sandbox
# ---------------------------------------------------------------------------


def _tar_directory(source_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(source_dir, arcname="workspace")
    return buffer.getvalue()


async def upload_workspace(env: AgentEnv, host_workdir: Path) -> str:
    """Tar the host workspace, upload to the sandbox container, extract.

    Returns the container-side ``agent_workdir`` path.
    """
    archive_name = f"triton_workspace_{uuid.uuid4().hex[:8]}.tar.gz"
    container_archive = f"/tmp/{archive_name}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        archive_path = Path(tmp_dir) / archive_name
        archive_path.write_bytes(_tar_directory(host_workdir))
        await env.copy_to_container(archive_path, Path(container_archive))

    await env.communicate(
        f"rm -rf /opt/workspace && "
        f"tar -xzf {container_archive} -C /opt && "
        f"rm -f {container_archive}",
        check="raise",
        error_msg="Failed to extract Triton workspace",
    )
    return f"/opt/workspace/agent_workdir"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a coding agent that writes Triton Ascend NPU operators.

Available tools:
- str_replace_editor: view, create, edit files (commands: view, create, str_replace, insert, undo_edit)
- execute_bash: run shell commands (verification, benchmarking, etc.)
- submit: signal task completion when all verification tests pass

Workflow:
1. Read `INSTRUCTIONS.md` and `CLAUDE.md` for task details and validation commands.
2. Read the reference PyTorch implementation in `src/`.
3. Create `src/{op_name}_triton_ascend_impl.py` with your Triton implementation.
4. Run validation commands (AST check, verify, benchmark) using execute_bash.
5. If verification fails, read the compact summary, fix the implementation, and re-verify.
6. Call submit when `passed_cases == total_cases`.

Rules:
- Pass tensors directly to Triton kernels. Do NOT use `.data_ptr()`.
- Prefer module-level `@triton.jit` kernels launched as `kernel[grid](...)`.
- Use `$OPERATOR_PYTHON` via `bash tools/run_npu_command.sh` for NPU commands.
- Do not write summary/status markdown files.
- Only verifier `passed_cases == total_cases` is success. AST check alone is not.
"""

USER_PROMPT_TEMPLATE = """Implement the Triton Ascend operator described in INSTRUCTIONS.md.

Operator: {op_name}
Target architecture: {arch}
Reference file: `src/{op_name}.py`
Implementation target: `src/{op_name}_triton_ascend_impl.py`

Read `INSTRUCTIONS.md` first, then implement the operator. Follow the validation
commands in `INSTRUCTIONS.md` to verify your implementation. Call submit only after
the verifier reports passed_cases == total_cases.

{instruction}
"""


def build_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    """Build the initial message list for AgentInteraction."""
    op_name = task["op_name"]
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(op_name=op_name)},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            op_name=op_name,
            arch=task.get("arch", "ascend910b1"),
            instruction=task.get("instruction", ""),
        )},
    ]


# ---------------------------------------------------------------------------
# Sandbox creation
# ---------------------------------------------------------------------------


def create_sandbox_env(
    run_id: str,
    device_ids: str = "",
    attach_host: str = DEFAULT_ATTACH_HOST,
    attach_port: int = DEFAULT_ATTACH_PORT,
    attach_auth_token: str = DEFAULT_ATTACH_AUTH_TOKEN,
) -> AgentEnv:
    """Create an AgentEnv by attaching to an existing swerex server.

    Connects to a pre-started Docker container running swerex.server via HTTP,
    instead of launching a new container (which requires docker access).
    """
    env_variables: dict[str, str] = {
        "PIP_PROGRESS_BAR": "off",
        "GIT_PAGER": "cat",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "WORKSPACE_BASE": DEFAULT_WORKSPACE_DIR,
        "EVAL_LOCK_DIR": "/shared/device-locks",
        "EVAL_DEVICE_PREFIX": "npu",
        "TRITON_PIPELINE_ERROR_PREVIEW_CHARS": "2000",
    }
    if device_ids:
        env_variables["EVAL_DEVICE_IDS"] = device_ids

    return AgentEnv(
        run_id=run_id,
        env_config=AgentEnvConfig(
            deployment=LocalAttachDeploymentConfig(
                type="local_attach",
                host=attach_host,
                port=attach_port,
                auth_token=attach_auth_token,
                timeout=600,
                startup_timeout=600,
            ),
            env_variables=env_variables,
        ),
    )


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------


async def run_one_task(
    task: dict[str, Any],
    chat_model: OpenAICompatibleChatModel,
    *,
    output_dir: str = "",
    max_turns: int = 50,
    action_timeout: int = 300,
    device_ids: str = "",
) -> dict[str, Any]:
    """Run one Triton operator task through the agent interaction loop.

    Returns a dict with the full trajectory, reward, and evaluation results.
    """
    op_name = task["op_name"]
    run_id = f"synth_{op_name}_{uuid.uuid4().hex[:8]}"

    # 1. Setup host workspace
    workspace_temp_dir = SCRIPT_DIR / "workspace_temp"
    workspace_temp_dir.mkdir(parents=True, exist_ok=True)
    host_workdir = Path(tempfile.mkdtemp(prefix=f"synth-{op_name}-", dir=workspace_temp_dir))
    try:
        setup_workspace(task, host_workdir)

        # 2. Create tools manager
        tools_manager = ToolsManager(
            tools_manager_config=ToolsManagerConfig(
                tools=[
                    ToolConfig(name="str_replace_editor"),
                    ToolConfig(name="execute_bash"),
                    ToolConfig(name="submit"),
                ],
                parser=DEFAULT_TOOL_PARSER,
            )
        )
        chat_model.set_tools_schemas(tools_manager.tools_schemas)

        # 3. Create sandbox env
        env = create_sandbox_env(run_id, device_ids=device_ids)

        # 4. Build messages
        messages = build_messages(task)

        # 5. Create interaction
        interaction = AgentInteraction(
            run_id=run_id,
            env=env,
            model=chat_model,
            tools_manager=tools_manager,
            messages=messages,
            action_timeout=action_timeout,
            max_turns=max_turns,
        )

        # 6. Run
        started_at = time.perf_counter()
        try:
            await env.start()
            await env.install_tools(tools_manager.tools)
            workspace_dir = await upload_workspace(env, host_workdir)

            interaction_result = await interaction.run()

            # 7. Evaluate reward
            metadata = {
                "op_name": op_name,
                "arch": task.get("arch", "ascend910b1"),
                "task_code": task.get("task_code", ""),
            }
            reward_score, eval_result = await evaluate_triton_workspace(
                env, metadata, workspace_dir=workspace_dir,
            )
        except Exception as exc:
            interaction_result = {
                "trajectory": [],
                "rollout_cache": {"metrics": {}},
                "execution_time": 0,
                "messages": messages,
            }
            reward_score = 0.0
            eval_result = {"eval_completed": False, "reward": 0.0, "error": str(exc)}
            import traceback
            print(f"[synth] ERROR op={op_name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            try:
                await env.close()
            except Exception:
                pass

        execution_time = time.perf_counter() - started_at

        # 8. Collect results
        num_turns = len(interaction_result.get("trajectory", []))
        exit_reason = "unknown"
        if num_turns > 0:
            last = interaction_result["trajectory"][-1]
            exit_reason = getattr(last, "exit_reason", "unknown")
            # Debug: dump trajectory + last messages to help diagnose unknown_error
            if exit_reason in ("unknown_error", "format_error", "token_limit"):
                print(f"[synth] DEBUG trajectory for {op_name}:")
                for i, step in enumerate(interaction_result["trajectory"]):
                    print(f"  step {i+1}: exit_reason={getattr(step, 'exit_reason', '?')} "
                          f"response_len={len(getattr(step, 'response', '') or '')} "
                          f"tool_results={len(getattr(step, 'tool_results', []) or [])}")
                msgs = interaction_result.get("messages", [])
                for m in msgs[-3:]:
                    role = m.get("role", "?")
                    content = (m.get("content") or "")[:500]
                    print(f"  [{role}] {content}")

        # Extract reward breakdown
        reward_breakdown = {}
        metrics = eval_result.get("metrics") if isinstance(eval_result, dict) else None
        if isinstance(metrics, dict):
            reward_breakdown = reward_breakdown_from_metrics(metrics)

        # Extract speedup
        speedup = None
        perf_data = metrics.get("perf_data") if isinstance(metrics, dict) else None
        if isinstance(perf_data, dict):
            for key in ("speedup_vs_torch", "speedup", "geomean_speedup"):
                if key in perf_data:
                    speedup = perf_data[key]
                    break

        result = {
            "op_name": op_name,
            "messages": interaction_result.get("messages", []),
            "num_turns": num_turns,
            "exit_reason": exit_reason,
            "reward_score": reward_score,
            "reward_breakdown": reward_breakdown,
            "eval_result": _json_safe(eval_result),
            "pass_rate": metrics.get("pass_rate", 0.0) if isinstance(metrics, dict) else 0.0,
            "speedup_vs_torch": speedup,
            "execution_time": round(execution_time, 3),
        }

        # Archive artifacts if output_dir is set
        if output_dir:
            from examples.triton_agent.reward import archive_text_artifacts
            try:
                await archive_text_artifacts(
                    env, metadata, output_dir, workspace_dir=workspace_dir,
                )
            except Exception:
                pass

        print(
            f"[synth] op={op_name} reward={reward_score:.4f} "
            f"pass_rate={result['pass_rate']:.2f} speedup={speedup} "
            f"turns={num_turns} exit={exit_reason} time={execution_time:.1f}s"
        )
        return result

    finally:
        # Cleanup host workspace
        shutil.rmtree(host_workdir, ignore_errors=True)


def _json_safe(obj: Any) -> Any:
    """Make an object JSON-serializable by converting non-serializable types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(item) for item in obj]
    return str(obj)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def save_results(results: list[dict[str, Any]], output_path: str) -> None:
    """Append results as JSONL lines to output_path."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(_json_safe(result), ensure_ascii=False) + "\n")
