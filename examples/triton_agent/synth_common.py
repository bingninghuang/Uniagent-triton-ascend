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
import shlex
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

# Skills reference documents (from the Claude Code workspace).
SKILLS_SRC = (
    SCRIPT_DIR / "workspace_claude" / "agent_workdir" / ".claude" / "skills"
)
SKILLS_SANDBOX_DIR = ".skills"

DEFAULT_SANDBOX_IMAGE = os.environ.get("TRITON_SANDBOX_IMAGE", "triton-operator-env:latest")

# Attach-mode defaults (for connecting to an existing swerex server)
DEFAULT_ATTACH_HOST = os.environ.get("TRITON_ATTACH_HOST", "http://127.0.0.1")
DEFAULT_ATTACH_PORT = int(os.environ.get("TRITON_ATTACH_PORT", "17000"))
DEFAULT_ATTACH_AUTH_TOKEN = os.environ.get("TRITON_ATTACH_AUTH_TOKEN", "mytoken123")
DEFAULT_WORKSPACE_DIR = "/opt/workspace/agent_workdir"
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

KERNELBENCH_INSTRUCTION_TEMPLATE = """# KernelBench Triton Ascend Task

Operator: {op_name}
Target: {arch}

Reference implementation:
- `src/{op_name}.py`

You must create:
- `src/{op_name}_triton_ascend_impl.py`

Goal:
Implement a Triton Ascend NPU version of the reference PyTorch operator.
The implementation must match the reference outputs for all provided test cases.

Important rules:
- Work from the current workspace directory.
- Preferred workspace root: `/opt/workspace/agent_workdir`.
- Use this workspace root when an absolute path is needed.
- Relative paths such as `INSTRUCTIONS.md` and `src/...` are valid after checking `pwd`.
- Read `src/{op_name}.py` before coding.
- The companion JSON file `src/*.json` may contain many test cases and be
  truncated when viewed. Read only the first 2-3 lines to understand the
  input format (shapes, dtypes, parameter names). If truncated, use
  `view_range [1, 3]` to see just the first few cases.
- Use local files only. Do not fetch code from the internet.
- Implement `ModelNew` in `src/{op_name}_triton_ascend_impl.py`.
- Pass tensors directly to Triton kernels. Do not use `.data_ptr()`.
- Prefer module-level `@triton.jit` kernels launched as `kernel[grid](...)`.
- Do not write summary/status markdown files.
- Call `submit` only after correctness verification passes all cases.

--- Validation Pipeline ---
The scripts under `tools/` are pre-installed infrastructure. NEVER read or view
their source code — just run the commands below. Each command's output is
self-explanatory.

Step 1 — AST check (static analysis, no NPU needed):
  Command:
    python3 tools/triton-op-verifier/scripts/validate_triton_impl.py \
      src/{op_name}_triton_ascend_impl.py --json
  What it does: checks that your code has @triton.jit kernels and forward()
    actually calls them (not pure PyTorch).
  Output: JSON with "valid": true/false. If false, read the "suggestion" field
    and fix your code before proceeding.

Step 2 — Stage files for the verifier:
  Command:
    mkdir -p output/verify && \
    cp src/{op_name}.py output/verify/{op_name}_torch.py && \
    cp src/{op_name}_triton_ascend_impl.py output/verify/{op_name}_triton_ascend_impl.py && \
    {{ cp src/*.json src/*.jsonl output/verify/ 2>/dev/null || true; }}

Step 3 — Correctness verification (requires NPU):
  Command:
    PY="${{OPERATOR_PYTHON:-/usr/local/python3.11.14/bin/python}}" && \
    bash tools/run_npu_command.sh "$PY" \
      tools/triton-op-verifier/scripts/verify.py \
      --op_name {op_name} \
      --verify_dir output/verify \
      --triton_impl_name triton_ascend_impl \
      --timeout 900 \
      --output output/verify/verify_result.json
  What it does: runs all test cases and compares your Triton output against
    the reference PyTorch output.
  Output files produced:
    - output/verify/verify_result.json        (full results, may be large)
    - output/verify/verify_result_summary.json (compact summary — READ THIS)
    - output/verify/verify_result.raw.log     (raw log if something crashes)
  How to check results:
    cat output/verify/verify_result_summary.json
    Look for: "passed_cases", "total_cases", "verified_success", "error_groups".

If verification fails:
- Read `output/verify/verify_result_summary.json` FIRST. It groups errors by
  type and shows the most common failure reasons with examples.
- Use the error descriptions to make targeted fixes to your implementation.
- Re-run Step 3 (no need to re-stage unless you changed file names).
- If the summary doesn't exist, read `output/verify/verify_result.raw.log`.

Success condition:
- `passed_cases == total_cases` in the verification output.

Task context:
{instruction}
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

    # Copy skill reference documents so search_skills can find them.
    if SKILLS_SRC.is_dir():
        skills_dst = agent_dir_dst / SKILLS_SANDBOX_DIR
        shutil.copytree(SKILLS_SRC, skills_dst, dirs_exist_ok=True)

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

    workspace_dir = "/opt/workspace/agent_workdir"
    await env.communicate(
        f"rm -rf /opt/workspace && "
        f"tar -xzf {container_archive} -C /opt && "
        f"rm -f {container_archive} && "
        f"test -f {workspace_dir}/INSTRUCTIONS.md",
        check="raise",
        error_msg="Failed to extract Triton workspace",
    )
    await env.communicate(
        f"cd {workspace_dir}",
        check="raise",
        error_msg="Failed to enter Triton workspace",
    )
    return workspace_dir


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a coding agent that implements Triton Ascend NPU operators.

Available tools:
- str_replace_editor: view, create, and edit files
- execute_bash: run shell commands
- submit: finish the task after verification passes

Workflow:
1. Confirm the current directory contains INSTRUCTIONS.md. The preferred workspace root is `/opt/workspace/agent_workdir`.
2. Read INSTRUCTIONS.md and the PyTorch reference implementation in src/.
3. Create the required Triton implementation file.
4. Run the AST check.
5. Run correctness verification.
6. If verification fails, inspect the compact error summary, fix the code, and verify again.
7. Call submit only after all correctness cases pass.

Rules:
- Use local files only.
- Do not fetch external repositories or code.
- Pass tensors directly to Triton kernels. Do not use .data_ptr().
- Prefer module-level @triton.jit kernels.
- AST check alone is not success.
- Correctness verification must pass all cases before submit.
- NEVER read or view files under tools/. These are infrastructure scripts
  whose exact commands are already in INSTRUCTIONS.md. Just run the commands
  directly; the scripts' output tells you everything you need.
- Test case JSON files (like `src/*.json`) can be very large. You only need
  to read the first 2-3 cases to understand the input shapes/dtypes. If a
  file view gets truncated (marked `<response clipped>`), immediately use
  `view_range` to read just what you need. Never let a truncated view cause
  you to produce an empty response — always follow up with another action.
"""

USER_PROMPT_TEMPLATE = """Implement the Triton Ascend operator in the current workspace.

Operator: {op_name}
Target architecture: {arch}
Reference file: `src/{op_name}.py`
Implementation target: `src/{op_name}_triton_ascend_impl.py`

Start by checking `pwd` and reading `INSTRUCTIONS.md`. The preferred workspace
root is `/opt/workspace/agent_workdir`; use it when an absolute path is needed.
Then implement and verify according to the commands in `INSTRUCTIONS.md`. Call
submit only after the verifier reports passed_cases == total_cases.

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
# Hard workflow: skill matching, prompt builders, and orchestration
# ---------------------------------------------------------------------------

# ---- skill matching ---------------------------------------------------------

# Operator type keywords → skill reference files (relative to .skills/).
_OP_SKILL_MAP: list[tuple[str, list[str]]] = [
    ("sum|mean|max|min|norm|softmax|var|std|layernorm|groupnorm|batchnorm|"
     "variance|rmsnorm|instancenorm|local_response_norm",
     ["triton-op-coding/references/triton-ascend-reduce.md"]),
    ("matmul|linear|bmm|dot|addmm|einsum|outer",
     ["triton-op-coding/references/triton-ascend-matmul.md"]),
    ("conv|pool|stencil|avgpool|maxpool|adaptive_avg_pool|adaptive_max_pool",
     ["triton-op-coding/references/triton-ascend-conv.md",
      "triton-op-coding/references/triton-ascend-reduce.md"]),
    ("gelu|relu|sigmoid|silu|swish|swiglu|add|mul|abs|sub|div|pow|exp|log|"
     "sqrt|rsqrt|neg|tanh|elu|leaky_relu|hardsigmoid|hardswish|silu|mish|"
     "cast|clamp|threshold|prelu|hardshrink|softplus|softsign|tanhshrink",
     ["triton-op-coding/references/triton-ascend-elementwise.md"]),
    ("sort|topk|argsort|kthvalue|median|msort",
     ["triton-op-coding/references/triton-ascend-sort-select.md"]),
    ("gather|scatter|index|embedding|nonzero|where|take|masked_select|"
     "index_select|index_put|index_add|index_copy|index_fill",
     ["triton-op-coding/references/triton-ascend-elementwise.md"]),
    ("interpolate|upsample|resize|grid_sample|resample|affine_grid",
     ["triton-op-coding/references/triton-ascend-interpolate.md"]),
    ("attention|flash|qkv|multihead|scaled_dot_product",
     ["triton-op-coding/references/triton-ascend-attention.md"]),
    ("cumsum|cumprod|histogram|histc|bincount",
     ["triton-op-coding/references/triton-ascend-reduce.md"]),
    ("cat|concat|split|chunk|stack|unbind|tile|repeat|repeat_interleave|"
     "pad|permute|transpose|flip|roll|rot90|reshape|view|squeeze|unsqueeze|"
     "expand|flatten|unflatten",
     ["triton-op-coding/references/triton-ascend-elementwise.md"]),
]

_DEFAULT_SKILLS: list[str] = [
    "triton-op-coding/references/triton-ascend-fundamentals.md",
    "triton-op-coding/references/triton-ascend-examples.md",
]

# Error type keywords → additional skills for repair.
_ERROR_SKILL_MAP: list[tuple[str, list[str]]] = [
    ("scalar|item\\(\\)|tl\\.|host.side|item\\(|torch\\.tensor",
     ["triton-latency-optimizer/references/avoid_scalar_lowering.md",
      "triton-latency-optimizer/references/scalar_to_vector.md"]),
    ("shape.mismatch|broadcast|stride|transpose|permute|reshape|view|contiguous",
     ["triton-op-coding/references/triton-ascend-elementwise.md"]),
    ("precision|nan|inf|tolerance|relative.error|numerical|overflow|underflow",
     ["triton-op-coding/references/triton-ascend-reduce.md"]),
    ("grid|coreDim|65535|block|program_id|tile|launch",
     ["triton-op-coding/references/triton-ascend-fundamentals.md",
      "triton-latency-optimizer/references/block_size_scaling.md"]),
    ("compile|syntax|typeerror|valueerror|unsupported|constexpr",
     ["triton-op-coding/references/triton-ascend-fundamentals.md",
      "triton-op-coding/references/triton-ascend-examples.md"]),
    ("matmul|dot|linear|gemm|bmm",
     ["triton-op-coding/references/triton-ascend-matmul.md"]),
    ("attention|softmax",
     ["triton-op-coding/references/triton-ascend-attention.md"]),
]


def _select_op_skills(task_code: str, op_name: str) -> list[str]:
    """Return skill file paths relevant to the operator, plus defaults."""
    text = f"{op_name}\n{task_code}".lower()
    paths: list[str] = []
    seen: set[str] = set()

    for pattern, refs in _OP_SKILL_MAP:
        if any(kw in text for kw in pattern.split("|")):
            for ref in refs:
                if ref not in seen:
                    paths.append(ref)
                    seen.add(ref)

    for ref in _DEFAULT_SKILLS:
        if ref not in seen:
            paths.append(ref)
            seen.add(ref)

    return paths


def _select_error_skills(error_groups: list[dict]) -> list[str]:
    """Return additional skill paths for specific error types."""
    text = " ".join(
        g.get("reason", "") + " " + g.get("error_type", "")
        for g in (error_groups or [])
    ).lower()
    paths: list[str] = []
    seen: set[str] = set()

    for pattern, refs in _ERROR_SKILL_MAP:
        if any(kw in text for kw in pattern.split("|")):
            for ref in refs:
                if ref not in seen:
                    paths.append(ref)
                    seen.add(ref)

    return paths


def _format_skills(skill_paths: list[str], sandbox_dir: str = SKILLS_SANDBOX_DIR) -> str:
    """Read skill files and return formatted snippets for prompt inclusion."""
    skills_root = SCRIPT_DIR / "workspace_claude" / "agent_workdir" / ".claude" / "skills"
    parts: list[str] = []

    for rel_path in skill_paths:
        full = skills_root / rel_path
        if not full.is_file():
            continue
        try:
            content = full.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Truncate to keep prompt compact.
        if len(content) > 2000:
            content = content[:2000] + "\n...[truncated]"

        parts.append(f"### {rel_path}\n{content}")

    return "\n\n".join(parts)


# ---- prompt builders --------------------------------------------------------

# System prompt used during coding / fix sessions (no execute_bash, no submit).
CODING_SYSTEM_PROMPT = """You are a Triton Ascend NPU operator implementation expert.

Your workspace is `/opt/workspace/agent_workdir`. ALL file paths must start with
`/opt/workspace/agent_workdir/`.  The directory `/testbed` does NOT exist — do
NOT use it.  Use `pwd` to confirm you are in the right directory.

Available tools:
- str_replace_editor: view, create, and edit files.
- search_skills: search skill reference documents by keywords.

Rules:
- Your FIRST action should write the implementation file using
  `str_replace_editor create` at the correct workspace path.
- Do NOT spend turns exploring the filesystem.
- Implement `ModelNew` in the required implementation file.
- Pass tensors directly to Triton kernels. Do not use .data_ptr().
- Prefer module-level `@triton.jit` kernels launched as `kernel[grid](...)`.
- NEVER read files under tools/ — they are infrastructure scripts.
- Use `search_skills` for Triton-Ascend reference material.
- The verification pipeline runs automatically — you do NOT need to run it.
- After writing the file, improve if needed, then stop. Do NOT loop on thinking.
"""


def _build_initial_prompt(
    task: dict[str, Any],
    skills_text: str,
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

    parts.extend([
        "",
        "## Relevant Skills",
        skills_text,
        "",
        "## Instructions",
        f"Workspace: `/opt/workspace/agent_workdir`",
        f"Implementation file: `/opt/workspace/agent_workdir/src/{op_name}_triton_ascend_impl.py`",
        f"Reference file: `/opt/workspace/agent_workdir/src/{op_name}.py`",
        f"",
        f"IMMEDIATE ACTION: Use `str_replace_editor create` with path",
        f"`/opt/workspace/agent_workdir/src/{op_name}_triton_ascend_impl.py`",
        f"to write the complete ModelNew implementation now.",
        f"",
        f"Rules: ModelNew class with @triton.jit kernel(s). No `for` loops in",
        f"forward(). No torch compute ops — only buffer alloc and view/reshape.",
        f"After writing, use `search_skills` and edit if needed, then stop.",
        "",
        instruction,
    ])

    return "\n".join(parts)


def _build_ast_fix_prompt(
    op_name: str,
    current_code: str,
    ast_error: str,
    skills_text: str,
) -> str:
    """Build a fix prompt when AST check fails."""
    impl_path = f"/opt/workspace/agent_workdir/src/{op_name}_triton_ascend_impl.py"
    parts = [
        f"# AST Check Failed: {op_name}",
        f"Fix the file: `{impl_path}`",
        "",
        "## Error",
        ast_error,
        "",
        "## Your Current Code",
        "```python",
        current_code[:8000] if len(current_code) > 8000 else current_code,
        "```",
        "",
        "## Instructions",
        f"FIX THE ERROR NOW: The AST check found issues in your code.",
        f"Use `str_replace_editor str_replace` on `{impl_path}` to fix them.",
        f"Focus on the error message — it tells you the exact problem.",
        f"Only use `search_skills` after attempting a fix if stuck.",
        f"Do NOT explore the filesystem. Fix first, then stop.",
    ]
    if skills_text:
        parts.extend([
            "",
            "## Reference Skills (supplementary)",
            skills_text,
        ])
    return "\n".join(parts)


def _build_correctness_fix_prompt(
    op_name: str,
    current_code: str,
    passed: int,
    total: int,
    error_groups: list[dict],
    skills_text: str,
) -> str:
    """Build a fix prompt when correctness verification fails."""
    impl_path = f"/opt/workspace/agent_workdir/src/{op_name}_triton_ascend_impl.py"
    parts = [
        f"# Correctness Verification Failed: {op_name}",
        f"Fix the file: `{impl_path}`",
        f"Passed: {passed}/{total} ({100*passed/max(total,1):.1f}%)",
        "",
    ]

    if error_groups:
        parts.append("## Error Summary")
        for g in error_groups:
            parts.append(
                f"- [{g.get('error_type', 'unknown')}] "
                f"{g.get('reason', '')} "
                f"(affects {g.get('count', 0)} cases)"
            )
        parts.append("")

    parts.extend([
        "## Your Current Code",
        "```python",
        current_code[:8000] if len(current_code) > 8000 else current_code,
        "```",
        "",
        "## Instructions",
        f"FIX THE ERRORS NOW: Use `str_replace_editor str_replace` on",
        f"`{impl_path}` to fix the failures described above. The error",
        f"messages tell you exactly what is wrong — act on them directly.",
        f"",
        f"Only use `search_skills` AFTER attempting a fix, and only if you",
        f"are stuck on a specific syntax question. Skills are supplementary.",
        f"Do NOT explore the filesystem. Fix first, then stop.",
    ])

    if skills_text:
        parts.extend([
            "",
            "## Reference Skills (supplementary, search only if stuck)",
            skills_text,
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


async def _run_ast_check(
    env: AgentEnv,
    op_name: str,
    workspace_dir: str = DEFAULT_WORKSPACE_DIR,
) -> dict[str, Any]:
    """Run the AST validation script and return parsed result."""
    impl = f"{workspace_dir}/src/{op_name}_triton_ascend_impl.py"
    checker = (
        f"{workspace_dir}/tools/triton-op-verifier/scripts/"
        "validate_triton_impl.py"
    )
    cmd = (
        f"python3 {shlex.quote(checker)} {shlex.quote(impl)} --json 2>&1"
    )
    try:
        output = await env.communicate(cmd, check="ignore")
    except Exception as exc:
        return {"valid": False, "suggestion": f"AST check error: {exc}"}

    # Extract JSON from output (may be mixed with log lines).
    text = output.strip()
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {"valid": False, "suggestion": text[-1000:]}


async def _run_verify(
    env: AgentEnv,
    op_name: str,
    workspace_dir: str = DEFAULT_WORKSPACE_DIR,
) -> tuple[int, int, list[dict]]:
    """Stage files, run the correctness verifier, parse results.

    Returns (passed_cases, total_cases, error_groups).
    """

    # 1. Stage files
    stage_cmd = (
        f"cd {shlex.quote(workspace_dir)} && "
        f"mkdir -p output/verify && "
        f"cp src/{shlex.quote(op_name)}.py output/verify/{shlex.quote(op_name)}_torch.py && "
        f"cp src/{shlex.quote(op_name)}_triton_ascend_impl.py "
        f"output/verify/{shlex.quote(op_name)}_triton_ascend_impl.py; "
        f"cp src/*.json src/*.jsonl output/verify/ 2>/dev/null; true"
    )
    await env.communicate(stage_cmd, check="ignore")

    # 2. Run verify via the run_npu_command.sh wrapper.
    # Use "&&" so PY is set before "$PY" is expanded (bare "PY=x cmd $PY"
    # expands $PY before the prefix takes effect — bash gotcha).
    python = os.environ.get("OPERATOR_PYTHON", "/usr/local/python3.11.14/bin/python")
    verify_cmd = (
        f"cd {shlex.quote(workspace_dir)} && "
        f"PY={shlex.quote(python)} && "
        f"bash tools/run_npu_command.sh \"$PY\" "
        f"tools/triton-op-verifier/scripts/verify.py "
        f"--op_name {shlex.quote(op_name)} "
        f"--verify_dir output/verify "
        f"--triton_impl_name triton_ascend_impl "
        f"--timeout 900 "
        f"--output output/verify/verify_result.json"
    )
    print(f"  [verify] running verify for {op_name}...")
    try:
        await env.communicate(verify_cmd, check="ignore", timeout=960)
    except Exception as exc:
        print(f"  [verify] command failed: {exc}")
    # Debug: check what files exist after verify
    check = await env.communicate(
        f"ls -la {shlex.quote(workspace_dir)}/output/verify/ 2>&1 || echo 'no-dir'",
        check="ignore",
    )
    print(f"  [verify] output files: {check.strip()[:300]}")

    # 3. Read summary
    summary_path = f"{workspace_dir}/output/verify/verify_result_summary.json"
    try:
        summary_text = await env.read_file(summary_path)
        summary = json.loads(summary_text.lstrip("﻿"))
    except Exception:
        # Fall back to verify_result.json
        try:
            raw_text = await env.read_file(
                f"{workspace_dir}/output/verify/verify_result.json"
            )
            raw = json.loads(raw_text.lstrip("﻿"))
            passed = raw.get("passed_cases", 0)
            total = raw.get("total_cases", 0)
            failed = raw.get("failed_cases", max(total - passed, 0))
            return passed, total, [
                {"error_type": "verify_failed", "reason": "See raw log", "count": failed}
            ]
        except Exception:
            return 0, 1, [{"error_type": "no_output", "reason": "Verifier produced no output", "count": 1}]

    passed = summary.get("passed_cases", 0)
    total = summary.get("total_cases", 0)
    error_groups = summary.get("error_groups", [])
    return passed, total, error_groups


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


# ---- coding session ---------------------------------------------------------

async def _run_coding_session(
    env: AgentEnv,
    chat_model: OpenAICompatibleChatModel,
    run_id: str,
    messages: list[dict[str, str]],
    *,
    max_turns: int = 5,
    action_timeout: int = 300,
) -> dict[str, Any]:
    """Run a limited AgentInteraction session (str_replace_editor + search_skills only).

    No execute_bash, no submit. Returns the trajectory and exit info.
    """
    tools_manager = ToolsManager(
        tools_manager_config=ToolsManagerConfig(
            tools=[
                ToolConfig(name="str_replace_editor"),
                ToolConfig(name="search_skills"),
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
        max_turns=max_turns,
    )

    result = await interaction.run()
    trajectory = result.get("trajectory", [])
    num_turns = len(trajectory)

    # Debug: print every step.
    print(f"  [{run_id}] session done: {num_turns} turns, messages={len(result.get('messages', []))}")
    for step in trajectory:
        exit_r = getattr(step, "exit_reason", "?") if hasattr(step, "exit_reason") else step.get("exit_reason", "?")
        resp = getattr(step, "response", "") if hasattr(step, "response") else step.get("response", "")
        tools = getattr(step, "tool_results", []) or []
        n_tools = len(tools) if isinstance(tools, list) else 0
        print(f"    step {getattr(step, 'step_idx', '?')}: exit={exit_r} resp_len={len(resp)} tools={n_tools}")
        for tr in (tools if isinstance(tools, list) else []):
            name = getattr(tr, "name", "?") if hasattr(tr, "name") else tr.get("name", "?")
            action = getattr(tr, "action", "") if hasattr(tr, "action") else tr.get("action", "")
            status = getattr(tr, "status", "") if hasattr(tr, "status") else tr.get("status", "")
            obs = (getattr(tr, "observation", "") if hasattr(tr, "observation") else tr.get("observation", ""))[:200]
            print(f"      tool={name} status={status} action={action[:120]}")
            if status != "ok":
                print(f"      OBS: {obs}")

    exit_reason = "unknown"
    if num_turns > 0:
        exit_reason = getattr(trajectory[-1], "exit_reason", "unknown")

    return {
        "trajectory": trajectory,
        "messages": result.get("messages", []),
        "num_turns": num_turns,
        "exit_reason": exit_reason,
    }


# ---- main orchestration -----------------------------------------------------

MAX_FIX_ATTEMPTS = 5  # max verify → fix rounds in Stage 2


async def run_one_task_hard(
    task: dict[str, Any],
    chat_model: OpenAICompatibleChatModel,
    *,
    output_dir: str = "",
    max_turns: int = 10,
    action_timeout: int = 300,
    device_ids: str = "",
    max_fix_attempts: int = MAX_FIX_ATTEMPTS,
) -> dict[str, Any]:
    """Run one Triton operator task with the hard workflow.

    Stage 1: model writes the initial implementation (limited tools, no
             verification).
    Stage 2: code-controlled verify → fix loop with structured error feedback.
    """
    op_name = task["op_name"]
    run_id_base = f"synth_{op_name}_{uuid.uuid4().hex[:6]}"

    # --- setup ---
    workspace_temp_dir = SCRIPT_DIR / "workspace_temp"
    workspace_temp_dir.mkdir(parents=True, exist_ok=True)
    host_workdir = Path(tempfile.mkdtemp(
        prefix=f"synth-{op_name}-", dir=workspace_temp_dir,
    ))

    try:
        setup_workspace(task, host_workdir)

        env = create_sandbox_env(f"{run_id_base}_main", device_ids=device_ids)
        await env.start()

        workspace_dir = await upload_workspace(env, host_workdir)

        started_at = time.perf_counter()
        all_trajectory: list[Any] = []

        # --- Stage 1: initial implementation ---
        # Prepare prompt with code-matched skills and test case summary.
        op_skills = _select_op_skills(task.get("task_code", ""), op_name)
        skills_text = _format_skills(op_skills)
        test_summary = await _extract_test_case_summary(env, op_name, workspace_dir)

        initial_prompt = _build_initial_prompt(task, skills_text, test_summary)
        stage1_messages: list[dict[str, str]] = [
            {"role": "system", "content": CODING_SYSTEM_PROMPT},
            {"role": "user", "content": initial_prompt},
        ]

        print(f"[synth] Stage 1: generating initial implementation for {op_name}")
        # Cap Stage 1 at 15 turns (CLI --max-turns is for soft workflow).
        stage1_turns = min(max_turns, 12)
        stage1_result = await _run_coding_session(
            env, chat_model, f"{run_id_base}_s1",
            stage1_messages,
            max_turns=stage1_turns,
            action_timeout=action_timeout,
        )
        all_trajectory.extend(stage1_result.get("trajectory", []))

        # --- Stage 2: verify + fix loop ---
        for fix_attempt in range(1, max_fix_attempts + 1):
            impl_code = await _read_impl_file(env, op_name, workspace_dir)
            if not impl_code:
                print(f"[synth] {op_name}: no implementation file, retrying...")
                # Prompt model to create the file.
                retry_prompt = _build_initial_prompt(task, skills_text, test_summary)
                retry_result = await _run_coding_session(
                    env, chat_model, f"{run_id_base}_retry{fix_attempt}",
                    [{"role": "system", "content": CODING_SYSTEM_PROMPT},
                     {"role": "user", "content": retry_prompt}],
                    max_turns=8, action_timeout=action_timeout,
                )
                all_trajectory.extend(retry_result.get("trajectory", []))
                impl_code = await _read_impl_file(env, op_name, workspace_dir)
                if not impl_code:
                    continue

            # AST check
            print(f"[synth] {op_name}: AST check (attempt {fix_attempt})")
            ast_result = await _run_ast_check(env, op_name, workspace_dir)
            if not ast_result.get("valid"):
                print(f"[synth] {op_name}: AST FAILED — {ast_result.get('suggestion', '')[:200]}")
                error_skills = _select_error_skills([
                    {"reason": ast_result.get("suggestion", ""),
                     "error_type": "ast_check_failed"}
                ])
                ast_skills_text = _format_skills(op_skills + error_skills)
                fix_prompt = _build_ast_fix_prompt(
                    op_name, impl_code or "",
                    ast_result.get("suggestion", "AST check failed"),
                    ast_skills_text,
                )
                fix_result = await _run_coding_session(
                    env, chat_model, f"{run_id_base}_fix_ast{fix_attempt}",
                    [{"role": "system", "content": CODING_SYSTEM_PROMPT},
                     {"role": "user", "content": fix_prompt}],
                    max_turns=8, action_timeout=action_timeout,
                )
                all_trajectory.extend(fix_result.get("trajectory", []))
                continue

            # Correctness verification
            print(f"[synth] {op_name}: correctness verification (attempt {fix_attempt})")
            passed, total, error_groups = await _run_verify(
                env, op_name, workspace_dir,
            )

            if passed == total and total > 0:
                print(f"[synth] {op_name}: VERIFIED {passed}/{total} — SUCCESS")
                # Evaluate reward.
                metadata = {
                    "op_name": op_name,
                    "arch": task.get("arch", "ascend910b1"),
                    "task_code": task.get("task_code", ""),
                }
                reward_score, eval_result = await evaluate_triton_workspace(
                    env, metadata, workspace_dir=workspace_dir,
                )
                break

            print(f"[synth] {op_name}: VERIFY FAILED {passed}/{total}")
            if error_groups:
                for g in error_groups[:3]:
                    print(f"  - [{g.get('error_type','?')}] {g.get('reason','')[:120]}")

            # Build fix prompt.
            error_skills = _select_error_skills(error_groups)
            fix_skills_text = _format_skills(op_skills + error_skills)
            fix_prompt = _build_correctness_fix_prompt(
                op_name, impl_code or "", passed, total,
                error_groups, fix_skills_text,
            )
            fix_result = await _run_coding_session(
                env, chat_model, f"{run_id_base}_fix{fix_attempt}",
                [{"role": "system", "content": CODING_SYSTEM_PROMPT},
                 {"role": "user", "content": fix_prompt}],
                max_turns=8, action_timeout=action_timeout,
            )
            all_trajectory.extend(fix_result.get("trajectory", []))
        else:
            # Loop exhausted without 100% success.  Still evaluate the
            # workspace to capture actual pass_rate / reward as a partial
            # result for training data.
            print(f"[synth] {op_name}: fix attempts exhausted, evaluating partial result")
            metadata = {
                "op_name": op_name,
                "arch": task.get("arch", "ascend910b1"),
                "task_code": task.get("task_code", ""),
            }
            reward_score, eval_result = await evaluate_triton_workspace(
                env, metadata, workspace_dir=workspace_dir,
            )

        execution_time = time.perf_counter() - started_at

        # --- collect results ---
        exit_reason = "finished" if reward_score > 0.5 else "max_fix_attempts"
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

        result = {
            "op_name": op_name,
            "messages": [],
            "trajectory": _json_safe(all_trajectory),
            "num_turns": len(all_trajectory),
            "exit_reason": exit_reason,
            "reward_score": reward_score,
            "reward_breakdown": reward_breakdown,
            "eval_result": _json_safe(eval_result),
            "pass_rate": pass_rate,
            "speedup_vs_torch": speedup,
            "execution_time": round(execution_time, 3),
        }

        if output_dir:
            from examples.triton_agent.reward import archive_text_artifacts
            try:
                metadata = {
                    "op_name": op_name,
                    "arch": task.get("arch", "ascend910b1"),
                    "task_code": task.get("task_code", ""),
                }
                await archive_text_artifacts(
                    env, metadata, output_dir, workspace_dir=workspace_dir,
                )
            except Exception:
                pass

        print(
            f"[synth] op={op_name} reward={reward_score:.4f} "
            f"pass_rate={pass_rate:.2f} speedup={speedup} "
            f"turns={len(all_trajectory)} exit={exit_reason} time={execution_time:.1f}s"
        )
        return result

    except Exception as exc:
        execution_time = time.perf_counter() - started_at
        import traceback
        print(f"[synth] ERROR op={op_name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
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
        try:
            await env.close()
        except Exception:
            pass
        shutil.rmtree(host_workdir, ignore_errors=True)


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
            "trajectory": _json_safe(interaction_result.get("trajectory", [])),
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
