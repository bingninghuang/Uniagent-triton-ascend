#!/usr/bin/env python3
"""Shared utilities for Triton operator data synthesis (inference-only).

Provides task loading, workspace setup, prompt construction, sandbox creation,
and trajectory collection. Used by both synth_triton_local.py and synth_triton_glm.py.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import shlex
import shutil
import tarfile
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from uni_agent.interaction import (
    AgentEnv,
    AgentEnvConfig,
    AgentInteraction,
    ToolsManager,
    ToolsManagerConfig,
)
from uni_agent.interaction.interaction import StepOutput
from uni_agent.interaction.model import MaxTokenExceededError, OpenAICompatibleChatModel
from uni_agent.tools import ToolConfig
from uni_agent.deployment import LocalAttachDeploymentConfig
from uni_agent.compaction import CompactionConfig, ConversationCompactor

# Reuse reward evaluation from the Triton agent.
from examples.triton_agent.reward import (
    evaluate_triton_workspace,
    reward_breakdown_from_metrics,
    _remote_ast_check,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR / "workspace"
BENCHMARK_ROOT = SCRIPT_DIR / "benchmarks" / "NPUKernelBench"

# Skills reference documents (from the Claude Code workspace).
SKILLS_SRC = SCRIPT_DIR / "workspace_knowledge_all"
SKILLS_SANDBOX_DIR = ".skills"

DEFAULT_SANDBOX_IMAGE = os.environ.get("TRITON_SANDBOX_IMAGE", "triton-operator-env:latest")

# Attach-mode defaults (for connecting to an existing swerex server)
DEFAULT_ATTACH_HOST = os.environ.get("TRITON_ATTACH_HOST", "http://127.0.0.1")
DEFAULT_ATTACH_PORT = int(os.environ.get("TRITON_ATTACH_PORT", "17000"))
DEFAULT_ATTACH_AUTH_TOKEN = os.environ.get("TRITON_ATTACH_AUTH_TOKEN", "mytoken123")
DEFAULT_WORKSPACE_DIR = os.environ.get("TRITON_WORKSPACE_DIR", "/opt/workspace/agent_workdir")
DEFAULT_TOOL_PARSER = "qwen3_coder"

# ---------------------------------------------------------------------------
# Task loading
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


def _parse_npukernelbench_filename(path: Path) -> tuple[int | str, str, int | None]:
    """Parse KernelBench filenames.

    Supports:
      - ``10_Relu`` -> (10, Relu, None)
      - ``level0_10_0_Relu`` -> (10, 0_Relu, 0)  # level_y_slice_OpName
    """
    match = re.match(r"^level(\d+)_(\d+)_(\d+)_(.+)$", path.stem)
    if match:
        level = int(match.group(1))
        problem_id = int(match.group(2))
        slice_id = int(match.group(3))
        op_name = match.group(4)
        return problem_id, f"{slice_id}_{op_name}", level
    match = re.match(r"^(\d+)[_-](.+)$", path.stem)
    if match:
        return int(match.group(1)), match.group(2), None
    return path.stem, path.stem, None


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
        dir_level = _level_to_int(level_dir.name)
        for py_file in sorted(level_dir.glob("*.py")):
            problem_id, name, file_level = _parse_npukernelbench_filename(py_file)
            level = file_level if file_level is not None else dir_level
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


def setup_workspace(task: dict[str, Any], work_dir: Path) -> Path:
    """Prepare a host-side workspace directory for one task.

    Copies the workspace template, task code, and support files.
    Returns the ``agent_workdir`` path inside ``work_dir``.
    """
    agent_dir_src = WORKSPACE_ROOT / "agent_workdir"
    agent_dir_dst = work_dir / "agent_workdir"

    if agent_dir_dst.exists():
        shutil.rmtree(agent_dir_dst)
    shutil.copytree(agent_dir_src, agent_dir_dst)

    # Copy skill reference documents so search_skills can find them.
    if SKILLS_SRC.is_dir():
        skills_dst = agent_dir_dst / SKILLS_SANDBOX_DIR
        shutil.copytree(SKILLS_SRC, skills_dst, dirs_exist_ok=True)

    op_name = task["op_name"]
    task_code = task.get("task_code", "")

    # Write reference code + support files
    src_dir = agent_dir_dst / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    if task_code:
        (src_dir / f"{op_name}.py").write_text(task_code, encoding="utf-8")

    support_files = task.get("support_files")
    written_json: dict[str, str] = {}
    if isinstance(support_files, dict):
        for name, content in support_files.items():
            safe_name = Path(str(name)).name
            text = str(content)
            (src_dir / safe_name).write_text(text, encoding="utf-8")
            if safe_name.endswith(".json"):
                written_json[safe_name] = text

    # Reference .py files often hardcode old sidecar names like "10_Relu.json".
    # After levelX_Y_Z_Op renames, also materialize those aliases + {op_name}.json
    # so verify/get_input_groups can find cases without the agent inventing files.
    if written_json:
        primary = next(iter(written_json.values()))
        (src_dir / f"{op_name}.json").write_text(primary, encoding="utf-8")
        if task_code:
            for match in re.finditer(r'["\'](\d+_[^"\']+\.json)["\']', task_code):
                alias = Path(match.group(1)).name
                if not (src_dir / alias).exists():
                    (src_dir / alias).write_text(primary, encoding="utf-8")

    (agent_dir_dst / ".triton_current_op_name").write_text(op_name + "\n", encoding="utf-8")
    return agent_dir_dst


# ---------------------------------------------------------------------------
# Workspace upload to sandbox
# ---------------------------------------------------------------------------


def _tar_directory(source_dir: Path, arcname: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(source_dir, arcname=arcname)
    return buffer.getvalue()


async def upload_workspace(
    env: AgentEnv,
    host_workdir: Path,
    *,
    op_name: str = "",
) -> str:
    """Tar the host workspace, upload to the sandbox container, extract.

    When ``op_name`` is set, writes ``.triton_current_op_name``, removes stale
    impl/output from prior tasks on the same card workspace, and clears verify
    artifacts so ``run_verify`` / eval start fresh for this op.

    Returns the container-side ``agent_workdir`` path.
    """
    archive_name = f"triton_workspace_{uuid.uuid4().hex[:8]}.tar.gz"
    container_archive = f"/tmp/{archive_name}"

    workspace_dir = PurePosixPath(DEFAULT_WORKSPACE_DIR)
    workspace_root = workspace_dir.parent
    workspace_parent = workspace_root.parent
    workspace_root_name = workspace_root.name
    if not workspace_root_name:
        raise ValueError(f"Invalid workspace directory: {DEFAULT_WORKSPACE_DIR}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        archive_path = Path(tmp_dir) / archive_name
        archive_path.write_bytes(_tar_directory(host_workdir, workspace_root_name))
        await env.copy_to_container(archive_path, Path(container_archive))

    workspace_dir_str = str(workspace_dir)
    await env.communicate(
        f"rm -rf {shlex.quote(str(workspace_root))} && "
        f"mkdir -p {shlex.quote(str(workspace_parent))} && "
        f"tar -xzf {shlex.quote(container_archive)} -C {shlex.quote(str(workspace_parent))} && "
        f"rm -f {shlex.quote(container_archive)} && "
        f"test -d {shlex.quote(workspace_dir_str + '/src')}",
        check="raise",
        error_msg="Failed to extract Triton workspace",
    )
    await env.communicate(
        f"cd {shlex.quote(workspace_dir_str)}",
        check="raise",
        error_msg="Failed to enter Triton workspace",
    )
    if op_name:
        await _finalize_workspace_for_op(env, workspace_dir_str, op_name)
    return workspace_dir_str


async def _finalize_workspace_for_op(env: AgentEnv, workspace_dir: str, op_name: str) -> None:
    """Pin the active op and drop stale verify/impl files from prior tasks."""
    await env.communicate(
        f"python3 -c {shlex.quote(_FINALIZE_WORKSPACE_PY)} "
        f"{shlex.quote(workspace_dir)} {shlex.quote(op_name)}",
        check="raise",
        error_msg=f"Failed to reset sandbox workspace for {op_name}",
    )


_FINALIZE_WORKSPACE_PY = r"""
import glob, os, shutil, sys
ws, op = sys.argv[1], sys.argv[2]
open(os.path.join(ws, '.triton_current_op_name'), 'w', encoding='utf-8').write(op + '\n')
src = os.path.join(ws, 'src')
os.makedirs(src, exist_ok=True)
for pat in ('*_triton_ascend_impl.py', '*_triton_ascend_impl_best.py'):
    for p in glob.glob(os.path.join(src, pat)):
        os.remove(p)
out = os.path.join(ws, 'output')
if os.path.isdir(out):
    shutil.rmtree(out)
os.makedirs(out, exist_ok=True)
for name in ('metrics.json', 'metrics_best.json', 'summary.json', 'perf_result.json'):
    p = os.path.join(ws, name)
    if os.path.isfile(p):
        os.remove(p)
print('[workspace] reset op=%s output/ and *_impl.py cleared' % op)
"""


# ---------------------------------------------------------------------------
# Sandbox creation
# ---------------------------------------------------------------------------


def create_sandbox_env(
    run_id: str,
    device_ids: str = "",
    attach_host: str = DEFAULT_ATTACH_HOST,
    attach_port: int = DEFAULT_ATTACH_PORT,
    attach_auth_token: str = DEFAULT_ATTACH_AUTH_TOKEN,
    tool_install_dir: str | None = None,
    op_name: str = "",
) -> AgentEnv:
    """Create an AgentEnv by attaching to an existing swerex server.

    Connects to a pre-started Docker container running swerex.server via HTTP,
    instead of launching a new container (which requires docker access).

    ``tool_install_dir``: per-worker tool install directory inside the sandbox.
    When multiple workers share one sandbox container (e.g. 8 swerex servers in
    one ``cc`` container), each MUST use a distinct tool_install_dir to avoid
    concurrent uploads to /usr/local/bin/<tool> racing with each other. Defaults
    to ``/usr/local/bin`` when unset (safe for the single-worker case).
    """
    install_dir = tool_install_dir or os.environ.get("TRITON_TOOL_INSTALL_DIR", "/usr/local/bin")

    env_variables: dict[str, str] = {
        "PIP_PROGRESS_BAR": "off",
        "GIT_PAGER": "cat",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "WORKSPACE_BASE": DEFAULT_WORKSPACE_DIR,
        "TRITON_WORKSPACE_DIR": DEFAULT_WORKSPACE_DIR,
        "SKILLS_DIR": f"{DEFAULT_WORKSPACE_DIR}/{SKILLS_SANDBOX_DIR}",
        "EVAL_LOCK_DIR": "/shared/device-locks",
        "EVAL_DEVICE_PREFIX": "npu",
        "TRITON_PIPELINE_ERROR_PREVIEW_CHARS": "2000",
    }
    if op_name:
        env_variables["TRITON_CURRENT_OP_NAME"] = op_name
        env_variables["OPERATOR_NAME"] = op_name
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
                timeout=3600,
                startup_timeout=600,
            ),
            env_variables=env_variables,
            tool_install_dir=Path(install_dir),
        ),
    )


# ---------------------------------------------------------------------------
# Hard workflow: skill matching, prompt builders, and orchestration
# ---------------------------------------------------------------------------

# ---- prompt builders --------------------------------------------------------

# System prompt used during coding / fix sessions.
_CODING_SYSTEM_PROMPT_TMPL = """You are a Triton Ascend NPU operator implementation expert.

Your workspace is `{workspace_dir}`. ALL file paths must start with
`{workspace_dir}/`.

Available tools:
- list_skills: list all available skills with name/description/path.
- read_skill: read a skill's SKILL.md (directory + first sentence per heading by default, --section for a chapter, --full for entire file, --file for a reference file).
- str_replace_editor: view, create, and edit files.
- run_verify: run AST check + correctness verify + performance benchmark (speedup_vs_torch). No arguments needed - call it after writing code or after any optimization. If AST fails or any case fails, benchmark is skipped.
- submit: finish the task. Call after verify passes AND you have optimized for speedup (target: {speedup_target}x). When speedup reaches the target, call submit.

## Phase 1: Coding & Verification (steps 1-6)

1. Call `list_skills` to see available skills and their purposes.
2. Read the reference implementation (provided in task) to understand the
   operator type (elementwise / reduce / matmul / attention / ...).
3. Use `read_skill` to gather knowledge ON DEMAND - read only what's
   relevant to this operator, not entire skills. Don't over-explore;
   each read consumes context and leaves less room for code and fixes.
   - `read_skill --skill hardware-specs --file references/hw-{{arch}}.md`:
     read hardware specs for the target architecture.
   - `read_skill --skill op-design-guide`: see operator category design
     guides. Read the section matching your operator type for design
     constraints, algorithm skeletons, and key implementation patterns.
   - Use `list_skills` to discover other skills (api-reference,
     ascend-extension, examples) and `read_skill` to explore them.
4. Write the implementation using `str_replace_editor create` at
   `{workspace_dir}/src/{{op_name}}_triton_ascend_impl.py`.

   ## Implementation Contract (REQUIRED — the verifier's AST check enforces this)

   Your impl file MUST define a class named exactly **`ModelNew`** with a
   **`forward(self, ...)`** method. Mirror the reference `Model` class in
   `{workspace_dir}/src/{{op_name}}.py`:
   - `ModelNew` must accept the SAME constructor arguments as `Model`
     (use `get_init_inputs()` for the signature) and expose a `forward()`
     with the SAME arguments and return shape as `Model.forward()`.
   - Inside `forward`, prepare inputs/outputs and launch your `@triton.jit`
     kernel(s) via `kernel[grid](...)`.
   - Define your `@triton.jit` kernel(s) at module level.
   - Do NOT name the class `Model` or anything other than `ModelNew`, and do
     NOT omit `forward`. Copying the reference `class Model` into the impl
     file will fail AST check.

5. Call `run_verify` to check AST + correctness + speedup. The tool outputs
   a structured summary: ast_valid, passed_cases/total_cases, pass_rate,
   error_groups, and speedup_vs_torch (when all cases pass).
6. If errors exist, fix them with `str_replace_editor str_replace`, then call
   `run_verify` again. Repeat until all cases pass.

## Phase 2: Performance Optimization (step 7, ONLY after verified_success: true)

7. After `run_verify` reports `verified_success: true` (all cases passed),
   your implementation is CORRECT. Now optimize for performance:
   a. Read the `latency-optimizer` skill to learn the optimization workflow
      and available optimization points.
   b. The run_verify SUMMARY also includes `bottleneck_hint` and
      `kernel_metrics` fields -- use them to identify which optimization
      directions are most relevant to your kernel.
   c. Apply ONE optimization at a time, then call `run_verify` to confirm
      correctness is maintained and check the updated speedup_vs_torch.
   d. Keep optimizing until speedup reaches {speedup_target}x, then call submit.

## Critical Rules

- Do NOT read the `latency-optimizer` skill until `run_verify` reports
  `verified_success: true`. It is an optimization-phase skill.
- Pass tensors directly to Triton kernels. Do not use .data_ptr().
- Prefer module-level `@triton.jit` kernels launched as `kernel[grid](...)`.
- NEVER read files under tools/ - they are infrastructure scripts.
- Call `run_verify` after EVERY code change - do NOT assume correctness.
- Do NOT loop on thinking. Write code, verify, fix, verify, optimize, verify, submit.
- Each optimization iteration: read ONE optimization point -> apply -> run_verify.
- Call `submit` to finish ONLY when speedup reaches the target.
"""


def _coding_system_prompt() -> str:
    speedup_target = os.environ.get("TRITON_REWARD_TARGET_SPEEDUP", "2.0")
    return _CODING_SYSTEM_PROMPT_TMPL.format(
        workspace_dir=DEFAULT_WORKSPACE_DIR,
        speedup_target=speedup_target,
    )


def _build_initial_prompt(
    task: dict[str, Any],
    test_case_summary: str = "",
) -> str:
    """Build the Stage 1 prompt: all info the model needs to write the impl."""
    op_name = task["op_name"]
    arch = task.get("arch", "ascend910b1")
    task_code = task.get("task_code", "")
    instruction = task.get("instruction", f"Implement the {op_name} operator.")

    parts = [
        f"# Task: Implement {op_name}",
        f"Target architecture: {arch}",
        "",
        "## Reference Implementation",
        "```python",
        task_code[:8000] if len(task_code) > 8000 else task_code,
        "```",
    ]

    if test_case_summary:
        parts.extend(["", "## Test Cases", test_case_summary])

    ws = DEFAULT_WORKSPACE_DIR
    parts.extend([
        "",
        "## File Paths",
        f"Workspace: `{ws}`",
        f"Implementation file: `{ws}/src/{op_name}_triton_ascend_impl.py`",
        f"Reference file: `{ws}/src/{op_name}.py`",
        "",
        "The reference uses `class Model`. The impl file MUST use `class ModelNew` "
        "(same `__init__` / `forward` signature). Do not copy the reference class name.",
        "",
        instruction,
    ])

    return "\n".join(parts)


# ---- sandbox helpers --------------------------------------------------------

async def _read_impl_file(
    env: AgentEnv,
    op_name: str,
    workspace_dir: str = DEFAULT_WORKSPACE_DIR,
) -> str | None:
    """Read the current implementation file from the sandbox."""
    path = f"{workspace_dir}/src/{op_name}_triton_ascend_impl.py"
    try:
        return await env.read_file(path)
    except Exception:
        return None


# Injected as a user message right after a compaction so the model resumes
# from the checkpoint's state instead of restarting the task flow (re-reading
# skills / rewriting the impl from scratch). The {next_step} slot quotes the
# checkpoint's own "## Next Step" section.
COMPACTION_RESUME_NOTE = (
    "[Context was compacted. The checkpoint above is your own record of all "
    "work so far on this task: attempts, constraints learned, the current "
    "implementation file, and the latest verify result. Do NOT restart the "
    "task, re-read skills, or rewrite the implementation from scratch. "
    "Continue directly from the checkpoint's Current State and execute the "
    "Next Step now.\n\n"
    "Next Step from checkpoint:\n{next_step}"
)


def _extract_next_step_section(checkpoint_text: str) -> str:
    """Pull the '## Next Step' section out of a checkpoint summary."""
    marker = "## Next Step"
    idx = checkpoint_text.find(marker)
    if idx < 0:
        return ""
    rest = checkpoint_text[idx + len(marker):]
    nxt = rest.find("\n## ")
    section = rest if nxt < 0 else rest[:nxt]
    return section.strip()


async def _segment_workspace_snapshot(
    env: AgentEnv,
    workspace_dir: str,
    op_name: str,
) -> dict[str, Any]:
    """Best-effort snapshot of the workspace state at a segment boundary.

    Captures the current implementation file, the last verify result, the
    current-round perf result, and the historical-best perf result so each
    saved trajectory segment is a self-contained rollout unit. Never raises;
    returns whatever could be collected (``{}`` when nothing is available
    yet, e.g. before the agent wrote any code).
    """
    snapshot: dict[str, Any] = {}
    impl = await _read_impl_file(env, op_name, workspace_dir)
    if impl:
        snapshot["impl_file"] = impl
    # ---- AST check on the impl file at THIS segment boundary (not the final
    # impl) so local_reward uses the same ast_ok as the boundary's verify/perf.
    try:
        ast_check = await _remote_ast_check(env, workspace_dir, op_name)
        snapshot["ast_ok"] = bool(ast_check.get("valid")) if isinstance(ast_check, dict) else False
    except Exception:
        snapshot["ast_ok"] = False
    try:
        verify_raw = await env.read_file(
            f"{workspace_dir}/output/verify/verify_result.json"
        )
        if verify_raw:
            snapshot["verify_result"] = _json_safe(json.loads(verify_raw))
    except Exception:
        pass

    # ---- current-round perf (perf_result.json) ----
    try:
        perf_raw = await env.read_file(
            f"{workspace_dir}/output/perf_result.json"
        )
        if perf_raw:
            perf = json.loads(perf_raw)
            snapshot["perf_current"] = _json_safe({
                "speedup_vs_torch": perf.get("speedup_vs_torch"),
                "total_cases": perf.get("total_cases"),
                "passed_cases": perf.get("passed_cases"),
                "implementation": perf.get("implementation"),
                "framework": perf.get("framework"),
            })
    except Exception:
        pass

    # ---- historical-best perf (perf_result_best.json) ----
    try:
        best_raw = await env.read_file(
            f"{workspace_dir}/output/perf_result_best.json"
        )
        if best_raw:
            best = json.loads(best_raw)
            snapshot["perf_best"] = _json_safe({
                "speedup_vs_torch": best.get("speedup_vs_torch"),
                "total_cases": best.get("total_cases"),
                "passed_cases": best.get("passed_cases"),
                "implementation": best.get("implementation"),
                "framework": best.get("framework"),
            })
    except Exception:
        pass

    return snapshot


async def _extract_test_case_summary(
    env: AgentEnv,
    op_name: str,
    workspace_dir: str = DEFAULT_WORKSPACE_DIR,
    max_cases: int = 3,
) -> str:
    """Read the first few test cases and return a human-readable summary."""
    json_path = f"src/{op_name}.json"
    # Find the actual JSON file.
    find_cmd = (
        f"cd {shlex.quote(workspace_dir)} && "
        f"ls src/*.json src/*.jsonl 2>/dev/null | head -1"
    )
    output = await env.communicate(find_cmd, check="ignore")
    found = output.strip().splitlines()
    if not found:
        return ""

    path = found[0]
    # swerex read_file resolves relative paths against its server cwd, not the
    # session cwd -- normalize to an absolute path so the file is found
    # regardless of where the swerex process was started.
    if not path.startswith("/"):
        path = f"{workspace_dir}/{path}"
    try:
        text = await env.read_file(path)
    except Exception:
        return ""

    lines = [l for l in text.splitlines() if l.strip()]
    cases = []
    for i, line in enumerate(lines[:max_cases]):
        try:
            case = json.loads(line)
            inputs = case.get("inputs", [])
            shapes = ", ".join(
                f"{inp.get('name','?')}={inp.get('shape','?')}({inp.get('dtype','?')})"
                for inp in inputs[:6]
            )
            cases.append(f"  Case {i+1}: {shapes}")
        except json.JSONDecodeError:
            cases.append(f"  Case {i+1}: [unparseable]")

    total = len(lines)
    return f"{total} test cases. First {len(cases)}:\n" + "\n".join(cases)


# ---- single-interaction run loop helpers -----------------------------------

async def _run_turns(
    interaction: AgentInteraction,
    num_turns: int,
    start_idx: int,
) -> list[Any]:
    """Run up to ``num_turns`` steps of *interaction*, appending each
    :class:`StepOutput` to ``interaction.trajectory``.

    Returns early when a step sets ``done=True`` for a terminal reason
    (token_limit, generate_timeout, terminal_dead, timeout_budget_exhausted).
    Non-terminal exits (thinking, format_error, completed) just continue the loop.
    """
    terminal_exits = {
        "token_limit",
        "generate_timeout",
        "terminal_dead",
        "timeout_budget_exhausted",
        "unknown_error",
    }

    for i in range(num_turns):
        step_idx = start_idx + i
        try:
            step_output = await interaction.step(step_idx)
        except Exception as exc:
            import traceback
            from uni_agent.interaction.model import _is_context_overflow_error

            tb = traceback.format_exc()
            print(f"  [step {step_idx}] {type(exc).__name__}:")
            traceback.print_exc()
            if _is_context_overflow_error(exc) or isinstance(exc, MaxTokenExceededError):
                exit_reason = "token_limit"
                msg = (
                    "[context_overflow] vLLM refused generate because the prompt already "
                    f"fills max_model_len. {exc}"
                )
            elif isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
                exit_reason = "generate_timeout"
                msg = (
                    "[generate_timeout] Timed out waiting for model or sandbox "
                    f"({type(exc).__name__}): {exc}"
                )
            else:
                exit_reason = "unknown_error"
                msg = f"[{exit_reason}] {type(exc).__name__}: {exc}\n{tb}"
            step_output = StepOutput(
                step_idx=step_idx,
                response=msg,
                thought=msg,
                tool_results=[],
                done=True,
                exit_reason=exit_reason,
            )
        interaction.trajectory.append(step_output)

        # Real-time progress print.
        exit_r = getattr(step_output, "exit_reason", "?")
        tools = getattr(step_output, "tool_results", []) or []
        n_tools = len(tools) if isinstance(tools, list) else 0
        parts = [f"  [{step_idx}] exit={exit_r} tools={n_tools}"]
        for tr in (tools if isinstance(tools, list) else []):
            name = getattr(tr, "name", "?") if hasattr(tr, "name") else tr.get("name", "?")
            status = getattr(tr, "status", "") if hasattr(tr, "status") else tr.get("status", "")
            action = getattr(tr, "action", "") if hasattr(tr, "action") else tr.get("action", "")
            parts.append(f" | {name}({status}): {action[:100]}")
        print("".join(parts), flush=True)

        if step_output.done and step_output.exit_reason in terminal_exits:
            return interaction.trajectory
    return interaction.trajectory


async def _inject_user_message(
    interaction: AgentInteraction,
    chat_model: OpenAICompatibleChatModel,
    content: str,
) -> None:
    """Inject a user-role message into the interaction's conversation.

    Updates both ``interaction.messages`` (for the API) and
    ``interaction.rollout_cache`` (for the training path).
    """
    msg: dict[str, object] = {"role": "user", "content": content}
    interaction.messages.append(msg)
    interaction.rollout_cache = await chat_model.append_messages_to_rollout_cache(
        [{"role": "user", "content": content}],
        interaction.rollout_cache,
    )


# ---- main orchestration -----------------------------------------------------

# When to inject the "verify NOW" fallback hint: this many turns before
# max_turns, if the model has not yet called run_verify, the framework
# injects a nudge so the model verifies at least once.
VERIFY_NUDGE_MARGIN = 8

# Same idea for creating the implementation file in the first place.
IMPL_NUDGE_MARGIN = 12


async def _best_files_mtime(env: AgentEnv, workspace_dir: str) -> tuple[float, float]:
    """Return (perf_result_best_mtime, metrics_best_mtime) from the sandbox.

    ``run_synth_inference`` calls this after every ``run_verify`` to detect when
    the in-loop best snapshots are (re)written, so it can record which assistant
    turn produced each best. That index later seeds ``train_best`` so the
    framework can crop the training trajectory to the best verify turn when
    ``TRITON_TRAIN_BEST_FIRST=1`` -- making reward and crop refer to the same
    verify. Returns -1.0 for a missing file; any error -> (-1.0, -1.0).
    """
    pb_path = os.path.join(str(workspace_dir), "output", "perf_result_best.json")
    mb_path = os.path.join(str(workspace_dir), "metrics_best.json")
    # Two `stat -c %Y` outputs separated by the literal 'SEP' (mtimes are
    # numeric, so SEP cannot collide). A missing file yields an empty field.
    stat_cmd = (
        f"stat -c %Y {shlex.quote(pb_path)} 2>/dev/null; "
        f"printf 'SEP'; "
        f"stat -c %Y {shlex.quote(mb_path)} 2>/dev/null"
    )
    try:
        out = await env.communicate(stat_cmd, timeout=30, check="ignore")
        if not out or "SEP" not in out:
            return -1.0, -1.0
        pb_str, mb_str = out.split("SEP", 1)
        pb = float(pb_str.strip()) if pb_str.strip() else -1.0
        mb = float(mb_str.strip()) if mb_str.strip() else -1.0
        return pb, mb
    except Exception:
        return -1.0, -1.0


async def run_synth_inference(
    *,
    task: dict[str, Any],
    chat_model: OpenAICompatibleChatModel,
    env: AgentEnv,
    workspace_dir: str,
    metadata: dict[str, Any] | None = None,
    max_turns: int = 50,
    action_timeout: int = 300,
    started_at: float | None = None,
    compaction_config: CompactionConfig | None = None,
) -> dict[str, Any]:
    """Shared inference core for Triton operator tasks (model-driven verify flow).

    Builds the synth system+user prompt, installs the 5 tools
    (str_replace_editor / list_skills / read_skill / run_verify / submit),
    drives a step+nudge ReAct loop, then evaluates reward via
    ``evaluate_triton_workspace``.

    Used by BOTH:
      * the standalone synth flow (``run_one_task_hard`` -> synth_triton_local),
      * the RL agent_runner adapter (``synth_agent_runner.triton_synth_runner``),
    so the two shell entry points share one inference implementation.

    The caller owns env lifecycle (start/close) and workspace staging
    (``setup_workspace`` + ``upload_workspace``); this function only runs the
    loop + reward evaluation.

    Returns a dict with: op_name, messages, trajectory, num_turns, exit_reason,
    reward_score, reward_breakdown, eval_result, pass_rate, speedup_vs_torch,
    execution_time, has_verified, has_impl, submitted.
    """
    op_name = task["op_name"]
    run_id = f"synth_{op_name}_{uuid.uuid4().hex[:6]}"
    if started_at is None:
        started_at = time.perf_counter()

    # ---- build initial prompt ----
    test_summary = await _extract_test_case_summary(env, op_name, workspace_dir)
    initial_prompt = _build_initial_prompt(task, test_summary)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _coding_system_prompt()},
        {"role": "user", "content": initial_prompt},
    ]

    # ---- single interaction for the whole task ----
    tools_manager = ToolsManager(
        tools_manager_config=ToolsManagerConfig(
            tools=[
                ToolConfig(name="str_replace_editor"),
                ToolConfig(name="list_skills"),
                ToolConfig(name="read_skill"),
                ToolConfig(name="run_verify"),
                ToolConfig(name="submit"),
            ],
            parser=DEFAULT_TOOL_PARSER,
        )
    )
    chat_model.set_tools_schemas(tools_manager.tools_schemas)
    await env.install_tools(tools_manager.tools)

    interaction = AgentInteraction(
        run_id=run_id,
        env=env,
        model=chat_model,
        tools_manager=tools_manager,
        messages=messages,
        action_timeout=action_timeout,
        max_turns=999,   # we drive step() ourselves
    )
    interaction.trajectory: list[Any] = []
    interaction.rollout_cache = await chat_model.prepare_rollout_cache(messages)

    # ---- automatic context compaction (see docs/对话轨迹压缩方案.md) ----
    # After each round, if token usage exceeds threshold_ratio of the
    # context window, the conversation is compacted to
    # [system, user task, assistant(checkpoint summary)] so the loop can
    # continue. Config comes from TRITON_COMPACTION_* env vars unless
    # overridden via the compaction_config parameter.
    compactor = ConversationCompactor(
        config=compaction_config or CompactionConfig.from_env(
            max_model_len=getattr(chat_model, "max_model_len", None)
        ),
    )
    print(
        f"[synth] compaction: enabled={compactor.config.enabled} "
        f"threshold={compactor.config.threshold_ratio} -> "
        f"{compactor.config.threshold_tokens} tokens "
        f"(max_context={compactor.config.max_context_tokens})"
    )

    # ---- single loop: model drives verify + fix + submit ----
    next_step = 1
    has_verified = False
    has_impl = False
    edited_after_verify = False
    submitted = False
    terminal_exits = {
        "token_limit",
        "generate_timeout",
        "terminal_dead",
        "timeout_budget_exhausted",
        "unknown_error",
    }

    # Track the last assistant trajectory index that called run_verify, so
    # best_index_by_source can resolve the correct turn when metrics_best.json
    # is written by evaluate_triton_workspace after the loop (not during it).
    last_verify_assistant_idx: int | None = None

    # Track which assistant turn produced each in-loop "best" snapshot so the
    # framework can crop the training trajectory to that turn when
    # TRITON_TRAIN_BEST_FIRST=1 (reward and crop then point to the same verify).
    # Keys match reward.py's `selected_metrics_source` values.
    best_index_by_source: dict[str, int | None] = {
        "perf_result_best": None,
        "metrics_best": None,
    }
    last_mtime_by_source: dict[str, float] = {
        "perf_result_best": -1.0,
        "metrics_best": -1.0,
    }

    # Track the best speedup seen so far *within this trajectory* (Python-side,
    # independent of perf_result_best.json which may be from a prior sandbox run).
    best_speedup_so_far: float | None = None

    # ---- segment bookkeeping for compaction boundaries ----
    # Each compaction splits the run into trajectory segments. Every
    # segment is a self-contained rollout unit:
    #   segment 0 messages: [system, user, think, tool, ...,
    #                        summary_instruction, assistant(llm_summary_1)]
    #   segment 1 messages: [system, user, llm_summary_1, think, ...,
    #                        summary_instruction, assistant(llm_summary_2)]
    #   final segment:      [...] ending in submit / max_step_limit
    # Each segment also carries a workspace_snapshot (impl file + last
    # verify result at the boundary). ``segments`` stays empty when no
    # compaction fired (the top-level messages/trajectory are then the
    # single segment).
    segments: list[dict[str, Any]] = []
    segment_traj_start: int = 0

    print(f"[synth] {op_name}: starting model-driven verify flow (max_turns={max_turns})")

    while next_step <= max_turns:
        # -- fallback nudges near the turn budget --
        turns_left = max_turns - next_step + 1
        if not has_impl and turns_left <= IMPL_NUDGE_MARGIN:
            impl_code = await _read_impl_file(env, op_name, workspace_dir)
            has_impl = bool(impl_code)
            if not has_impl:
                print(f"[synth] {op_name}: nudging to create impl file (turns_left={turns_left})")
                await _inject_user_message(
                    interaction, chat_model,
                    f"IMPORTANT: You are running low on turns ({turns_left} left). "
                    f"Create the implementation file NOW using "
                    f"`str_replace_editor create` at "
                    f"`{workspace_dir}/src/{op_name}_triton_ascend_impl.py`, "
                    f"then call `run_verify`.",
                )
        elif has_impl and not has_verified and turns_left <= VERIFY_NUDGE_MARGIN:
            print(f"[synth] {op_name}: nudging to call run_verify (turns_left={turns_left})")
            await _inject_user_message(
                interaction, chat_model,
                f"IMPORTANT: You have written code but haven't verified it. "
                f"Call `run_verify` NOW to check correctness. "
                f"Fix any errors, then `run_verify` again, then `submit`. "
                f"({turns_left} turns left.)",
            )

        # -- run one step --
        await _run_turns(interaction, 1, next_step)
        step = interaction.trajectory[-1] if interaction.trajectory else None
        next_step += 1

        if step is None:
            continue

        # -- track tool usage --
        ran_verify_this_step = False
        for tr in (getattr(step, "tool_results", []) or []):
            tname = getattr(tr, "name", "") if hasattr(tr, "name") else ""
            if tname == "run_verify":
                has_verified = True
                edited_after_verify = False
                ran_verify_this_step = True
            elif tname == "str_replace_editor":
                has_impl = True
                if has_verified:
                    edited_after_verify = True

        # -- record the assistant turn that (re)wrote each best snapshot --
        # The best files are written inside the run_verify tool call that just
        # completed, so stat them now; a newer mtime means this turn produced a
        # new best for that source. The recorded index lets framework.py crop the
        # training trajectory to this turn (best-prefix) to match the best reward.
        if ran_verify_this_step:
            assistant_index_now = len(interaction.trajectory) - 1
            last_verify_assistant_idx = assistant_index_now
            pb_mtime, mb_mtime = await _best_files_mtime(env, workspace_dir)
            if pb_mtime > last_mtime_by_source["perf_result_best"]:
                last_mtime_by_source["perf_result_best"] = pb_mtime
                best_index_by_source["perf_result_best"] = assistant_index_now
            if mb_mtime > last_mtime_by_source["metrics_best"]:
                last_mtime_by_source["metrics_best"] = mb_mtime
                best_index_by_source["metrics_best"] = assistant_index_now

            # ---- track best speedup within this trajectory ----
            try:
                perf_raw = await env.read_file(
                    f"{workspace_dir}/output/perf_result.json"
                )
                if perf_raw:
                    perf = json.loads(perf_raw)
                    cur_speedup = perf.get("speedup_vs_torch")
                    if cur_speedup is not None:
                        if best_speedup_so_far is None or cur_speedup > best_speedup_so_far:
                            best_speedup_so_far = cur_speedup
            except Exception:
                pass

        # -- check terminal conditions --
        exit_r = getattr(step, "exit_reason", "?")
        if exit_r == "finished":
            submitted = True
            print(f"[synth] {op_name}: model called submit at step {next_step - 1}")
            break
        if getattr(step, "done", False) and exit_r in terminal_exits:
            print(f"[synth] {op_name}: terminal exit={exit_r} at step {next_step - 1}")
            break

        # -- compaction checkpoint: compress the trajectory when context
        #    pressure exceeds the threshold AND the current step called
        #    run_verify (so there is a fresh verify result to summarize) --
        compactor.notify_step()
        usage = (
            getattr(step, "prompt_tokens", None),
            getattr(step, "completion_tokens", None),
        )
        if compactor.should_compact(interaction.messages, usage) and ran_verify_this_step:
            new_messages = await compactor.compact(
                interaction.messages,
                chat_model,
                step=next_step - 1,
                last_usage=usage,
            )
            if new_messages is not None:
                interaction.messages[:] = new_messages
                # Steer the model to resume from the checkpoint state (its
                # Next Step) instead of replaying the original task flow.
                interaction.messages.append(
                    {
                        "role": "user",
                        "content": COMPACTION_RESUME_NOTE.format(
                            next_step=_extract_next_step_section(
                                new_messages[-1]["content"]
                            )
                            or "(see the checkpoint's Next Step section)",
                        ),
                    }
                )
                # Record the compaction in the trajectory timeline so the
                # summary step is visible in results (matches the
                # prompt -> ... -> summary_instruction -> llm_summary flow).
                last_event = compactor.state.events[-1]
                marker_response = new_messages[-1]["content"]
                interaction.trajectory.append(
                    StepOutput(
                        step_idx=next_step - 1,
                        response=marker_response,
                        exit_reason="context_compacted",
                    )
                )

                # Compaction closes the current trajectory segment: archive
                # the exact summarizer call (conversation +
                # summary_instruction as prompt, llm_summary as the final
                # assistant response), this segment's steps, and a snapshot
                # of the workspace state at the boundary.
                if compactor.last_summary_call:
                    call = compactor.last_summary_call
                    # raw_response preserves the model's thinking prefix
                    # (if any) for the pre-compaction segment; summary is
                    # the cleaned version used in the rebuilt conversation.
                    snapshot = await _segment_workspace_snapshot(
                        env, workspace_dir, op_name
                    )
                    snapshot["best_speedup_so_far"] = best_speedup_so_far
                    segments.append(
                        {
                            "segment_idx": len(segments),
                            "messages": [
                                *call["prompt_messages"],
                                {"role": "assistant",
                                 "content": call.get("raw_response", call["summary"])},
                            ],
                            "trajectory": interaction.trajectory[segment_traj_start:],
                            "compaction": last_event.as_dict(),
                            "workspace_snapshot": snapshot,
                            "segment_type": "initial" if len(segments) == 0 else "compacted",
                            "prompt_contains_checkpoint": len(segments) > 0,
                            "ends_with_compaction": True,
                        }
                    )
                    segment_traj_start = len(interaction.trajectory)

                print(
                    f"[synth] {op_name}: context compacted at step {next_step - 1} "
                    f"({len(interaction.messages)} messages after compaction)"
                )

    # ---- final verify: ensure artifacts reflect the agent's latest code ----
    # evaluate_triton_workspace reads verify_result.json for correctness
    # and overlays speedup from perf_result.json.
    # produced by run_verify.  If the agent edited code after its last
    # run_verify (or never ran verify at all), those artifacts are stale;
    # run one final verify so the eval always sees fresh results.
    if has_impl and (not has_verified or edited_after_verify):
        print(f"[synth] {op_name}: running final verify on latest code "
              f"(has_verified={has_verified}, edited_after_verify={edited_after_verify})")
        try:
            op_q = shlex.quote(op_name)
            final_verify_cmd = (
                f"cd {shlex.quote(str(workspace_dir))} && "
                f"TRITON_CURRENT_OP_NAME={op_q} OPERATOR_NAME={op_q} run_verify 2>&1"
            )
            verify_output = await env.communicate(
                final_verify_cmd, timeout=2100, check="ignore"
            )
            tail_lines = verify_output.strip().splitlines()[-12:] if verify_output else []
            if tail_lines:
                print(f"[synth] {op_name}: final verify output (tail):\n"
                      + "\n".join(tail_lines))
        except Exception as exc:
            print(f"[synth] {op_name}: final verify failed: {exc}")
    else:
        print(f"[synth] {op_name}: skipping final verify "
              f"(artifacts fresh: has_verified={has_verified}, "
              f"edited_after_verify={edited_after_verify})")

    # ---- re-stat best files after final verify (may have updated perf_result_best.json) ----
    # The final verify runs run_verify which writes a new perf_result_best.json
    # when speedup is higher. Update best_index_by_source so framework.py crops
    # the training trajectory to the correct turn (the last assistant, whose code
    # the final verify ran on).
    final_verify_ran = has_impl and (not has_verified or edited_after_verify)
    if final_verify_ran:
        pb_mtime, _mb_mtime = await _best_files_mtime(env, workspace_dir)
        if pb_mtime > last_mtime_by_source["perf_result_best"]:
            last_mtime_by_source["perf_result_best"] = pb_mtime
            last_assistant_idx = len(interaction.trajectory) - 1
            if last_assistant_idx >= 0:
                best_index_by_source["perf_result_best"] = last_assistant_idx

    # ---- evaluate reward (runs regardless of how the loop ended) ----
    print(f"[synth] {op_name}: evaluating results "
          f"(verified={has_verified}, submitted={submitted}, turns={len(interaction.trajectory)})")
    eval_metadata = metadata or {
        "op_name": op_name,
        "arch": task.get("arch", "ascend910b1"),
        "task_code": task.get("task_code", ""),
    }
    reward_score, eval_result = await evaluate_triton_workspace(
        env, eval_metadata, workspace_dir=workspace_dir,
    )

    # ---- resolve best_index_by_source for metrics_best (written by evaluate) ----
    # metrics_best.json is written by evaluate_triton_workspace._snapshot_metrics_as_best,
    # not during the agent loop, so best_index_by_source["metrics_best"] is never
    # populated by the in-loop mtime tracker. When the reward selected metrics_best,
    # resolve the assistant index: the last run_verify (in-loop or final verify)
    # produced the artifacts that went into metrics_best. If the final verify ran,
    # the code was written by the last assistant turn.
    if best_index_by_source.get("metrics_best") is None:
        if final_verify_ran:
            # Final verify ran on the last assistant's code
            last_idx = len(interaction.trajectory) - 1
            if last_idx >= 0:
                best_index_by_source["metrics_best"] = last_idx
        elif last_verify_assistant_idx is not None:
            best_index_by_source["metrics_best"] = last_verify_assistant_idx

    # Similarly, if perf_result_best was selected but its index wasn't tracked
    # (e.g. final verify ran but didn't change mtime, yet evaluate still chose it),
    # resolve to the best available index.
    if best_index_by_source.get("perf_result_best") is None:
        if final_verify_ran:
            last_idx = len(interaction.trajectory) - 1
            if last_idx >= 0:
                best_index_by_source["perf_result_best"] = last_idx
        elif last_verify_assistant_idx is not None:
            best_index_by_source["perf_result_best"] = last_verify_assistant_idx

    execution_time = time.perf_counter() - started_at

    # ---- final trajectory segment (post-last-compaction) ----
    # Only appended when at least one compaction fired; when the run never
    # compacted, ``segments`` stays empty and the top-level
    # messages/trajectory are the single segment (no duplicated data).
    if segments:
        final_snapshot = await _segment_workspace_snapshot(
            env, workspace_dir, op_name
        )
        final_snapshot["best_speedup_so_far"] = best_speedup_so_far
        segments.append(
            {
                "segment_idx": len(segments),
                "messages": interaction.messages,
                "trajectory": interaction.trajectory[segment_traj_start:],
                "compaction": None,
                "workspace_snapshot": final_snapshot,
                "segment_type": "final",
                "prompt_contains_checkpoint": True,
                "ends_with_compaction": False,
            }
        )

    # --- collect results ---
    last_exit = (
        getattr(interaction.trajectory[-1], "exit_reason", "")
        if interaction.trajectory
        else ""
    )
    if submitted:
        exit_reason = "finished"
    elif last_exit in terminal_exits:
        exit_reason = last_exit
    elif next_step > max_turns:
        exit_reason = "max_step_limit"
    else:
        exit_reason = "completed"
    metrics = eval_result.get("metrics") if isinstance(eval_result, dict) else None
    if not isinstance(metrics, dict):
        metrics = {}
    reward_breakdown = reward_breakdown_from_metrics(metrics) if metrics else {}
    pass_rate = metrics.get("pass_rate", 0.0)
    speedup = None
    if isinstance(metrics.get("perf_data"), dict):
        for key in ("speedup_vs_torch", "speedup", "geomean_speedup"):
            if key in metrics["perf_data"]:
                speedup = metrics["perf_data"][key]
                break

    print(f"  [{run_id}] total: {len(interaction.trajectory)} steps", flush=True)
    print(
        f"[synth] op={op_name} reward={reward_score:.4f} "
        f"pass_rate={pass_rate:.2f} speedup={speedup} "
        f"turns={len(interaction.trajectory)} exit={exit_reason} "
        f"verified={has_verified} submitted={submitted} time={execution_time:.1f}s"
    )

    return {
        "op_name": op_name,
        "messages": interaction.messages,
        "trajectory": interaction.trajectory,
        "num_turns": len(interaction.trajectory),
        "exit_reason": exit_reason,
        "reward_score": reward_score,
        "reward_breakdown": reward_breakdown,
        "eval_result": _json_safe(eval_result),
        "pass_rate": pass_rate,
        "speedup_vs_torch": speedup,
        "execution_time": round(execution_time, 3),
        "has_verified": has_verified,
        "has_impl": has_impl,
        "submitted": submitted,
        "best_index_by_source": best_index_by_source,
        "assistant_messages_seen": len(interaction.trajectory),
        "compaction": compactor.stats(),
        # Compaction-split trajectory segments (each a self-contained
        # rollout unit). Empty when the run never compacted.
        "segments": segments,
        "segments_summary": {
            "num_segments": len(segments),
            "segment_dir": None,  # filled by caller after _persist_segments
        },
    }


def _persist_segments(
    segments: list[dict[str, Any]],
    output_dir: str,
    op_name: str = "",
) -> str | None:
    """Persist compaction segments to disk under ``output_dir/segments/``.

    Each segment gets its own subdirectory:

    - ``messages.json``: full API messages for this segment
    - ``compaction.json``: compaction event metadata (``null`` for final segment)
    - ``impl.py``: workspace snapshot impl file at the segment boundary
    - ``verify_result.json``: workspace snapshot verify result at the boundary
    - ``perf_current.json``: current-round perf result at the boundary
    - ``perf_best.json``: historical-best perf result at the boundary
    - ``best_speedup_so_far.json``: trajectory-best speedup tracked Python-side

    Never raises; returns the segments directory path on success, ``None`` on
    failure.
    """
    if not segments or not output_dir:
        return None
    try:
        seg_root = Path(output_dir) / "segments"
        for seg in segments:
            idx = seg.get("segment_idx", 0)
            seg_dir = seg_root / f"segment_{idx}"
            seg_dir.mkdir(parents=True, exist_ok=True)

            messages = seg.get("messages") or []
            if messages:
                (seg_dir / "messages.json").write_text(
                    json.dumps(_json_safe(messages), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            compaction = seg.get("compaction")
            (seg_dir / "compaction.json").write_text(
                json.dumps(_json_safe(compaction), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            snap = seg.get("workspace_snapshot") or {}
            if isinstance(snap, dict):
                impl = snap.get("impl_file")
                if impl:
                    (seg_dir / "impl.py").write_text(str(impl), encoding="utf-8")
                verify = snap.get("verify_result")
                if verify is not None:
                    (seg_dir / "verify_result.json").write_text(
                        json.dumps(_json_safe(verify), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                perf_current = snap.get("perf_current")
                if perf_current is not None:
                    (seg_dir / "perf_current.json").write_text(
                        json.dumps(_json_safe(perf_current), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                perf_best = snap.get("perf_best")
                if perf_best is not None:
                    (seg_dir / "perf_best.json").write_text(
                        json.dumps(_json_safe(perf_best), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                best_speedup = snap.get("best_speedup_so_far")
                if best_speedup is not None:
                    (seg_dir / "best_speedup_so_far.json").write_text(
                        json.dumps({"best_speedup_so_far": best_speedup}, ensure_ascii=False),
                        encoding="utf-8",
                    )

        print(
            f"[synth] {op_name}: segments persisted to {seg_root} "
            f"({len(segments)} segments)"
        )
        return str(seg_root)
    except Exception as exc:
        print(f"[synth] {op_name}: failed to persist segments: {exc}")
        return None


async def _archive_op_artifacts(
    env: AgentEnv | None,
    task: dict[str, Any],
    output_dir: str,
    workspace_dir: str,
    op_name: str,
) -> None:
    """Copy result artifacts (best/final impl + verify/perf jsons + torch
    reference) from the sandbox into ``output_dir`` (the per-op artifact
    dir prepared by the launcher). Never raises; failures are logged."""
    if not output_dir or not workspace_dir or env is None:
        return
    from examples.triton_agent.reward import archive_text_artifacts
    try:
        meta = {
            "op_name": op_name,
            "arch": task.get("arch", "ascend910b1"),
            "task_code": task.get("task_code", ""),
        }
        dest = await archive_text_artifacts(
            env, meta, output_dir, workspace_dir=workspace_dir, subdir=False,
        )
        print(f"[synth] {op_name}: artifacts archived to {dest or output_dir}")
    except Exception as exc:
        print(f"[synth] {op_name}: artifact archive failed: "
              f"{type(exc).__name__}: {exc}")


async def run_one_task_hard(
    task: dict[str, Any],
    chat_model: OpenAICompatibleChatModel,
    *,
    output_dir: str = "",
    output_path: str = "",
    max_turns: int = 50,
    action_timeout: int = 300,
    device_ids: str = "",
) -> dict[str, Any]:
    """Run one Triton operator task - standalone (inference-only) entry point.

    Thin wrapper around the shared ``run_synth_inference`` core: it owns env
    lifecycle (create_sandbox_env / upload_workspace / env.close) and result
    persistence (save_results / archive). Used by synth_triton_local /
    synth_triton_glm (the ``run_synth_levels_serial.sh`` pure-inference path).

    Behavior is unchanged from the pre-refactor implementation.
    """
    op_name = task["op_name"]
    run_id = f"synth_{op_name}_{uuid.uuid4().hex[:6]}"

    workspace_temp_dir = SCRIPT_DIR / "workspace_temp"
    workspace_temp_dir.mkdir(parents=True, exist_ok=True)
    host_workdir = Path(tempfile.mkdtemp(
        prefix=f"synth-{op_name}-", dir=workspace_temp_dir,
    ))

    result: dict[str, Any] | None = None
    env: AgentEnv | None = None
    workspace_dir = ""
    started_at = time.perf_counter()

    try:
        setup_workspace(task, host_workdir)

        env = create_sandbox_env(run_id, device_ids=device_ids, op_name=op_name)
        await env.start()

        workspace_dir = await upload_workspace(env, host_workdir, op_name=op_name)
        await env.communicate(
            f"export TRITON_CURRENT_OP_NAME={shlex.quote(op_name)} && "
            f"export OPERATOR_NAME={shlex.quote(op_name)}",
            check="ignore",
        )

        result = await run_synth_inference(
            task=task,
            chat_model=chat_model,
            env=env,
            workspace_dir=workspace_dir,
            max_turns=max_turns,
            action_timeout=action_timeout,
            started_at=started_at,
        )

        await _archive_op_artifacts(env, task, output_dir, workspace_dir, op_name)

        # Persist compaction segments to disk and replace the in-memory
        # segments list with a lightweight summary so the JSONL result stays
        # compact (full messages are in the segments/ directory).
        segs = result.get("segments") if result else None
        if segs:
            segment_dir = _persist_segments(segs, output_dir, op_name)
            result["segments_summary"] = {
                "num_segments": len(segs),
                "segment_dir": segment_dir,
            }
            del result["segments"]

        return result

    except Exception as exc:
        execution_time = time.perf_counter() - started_at
        import traceback
        print(f"[synth] ERROR op={op_name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        # Best-effort archive even for crashed runs: the workspace may hold
        # a partial impl / verify result worth keeping.
        await _archive_op_artifacts(env, task, output_dir, workspace_dir, op_name)
        # Also persist any partial compaction segments collected before the crash.
        if result and result.get("segments"):
            _persist_segments(result["segments"], output_dir, op_name)
        return {
            "op_name": op_name,
            "messages": [],
            "trajectory": [],
            "num_turns": 0,
            "exit_reason": "unknown_error",
            "reward_score": 0.0,
            "reward_breakdown": {},
            "eval_result": {"eval_completed": False, "reward": 0.0, "error": str(exc)},
            "pass_rate": 0.0,
            "speedup_vs_torch": None,
            "execution_time": round(execution_time, 3),
        }

    finally:
        if output_path and result is not None:
            try:
                save_results([result], output_path)
            except Exception:
                pass
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        shutil.rmtree(host_workdir, ignore_errors=True)



def _json_safe(obj: Any) -> Any:
    """Make an object JSON-serializable by converting non-serializable types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "model_dump"):
        return _json_safe(obj.model_dump())
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