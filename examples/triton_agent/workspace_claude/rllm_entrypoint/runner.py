"""
runner.py 鈥?Main conversation execution logic for the rllm entrypoint.

Responsibilities:
  1. Build OpenHands SDK objects (LLM, Agent, Conversation).
  2. Wire the event callback into the Conversation.
  3. Start background pause controller thread.
  4. Run the conversation loop, honoring pause/resume signals.
  5. Emit lifecycle events (startup / heartbeat / evaluate / finish).
  6. Report final metrics and clean up background threads.

v2 (event-driven):
  The StateUploader background thread has been removed.  Instead, each
  OpenHands SDK event is pushed to the gateway immediately inside the
  event callback.  Heartbeats are emitted from a lightweight timer thread
  so the gateway always has a liveness signal even during long LLM calls.
"""
from __future__ import annotations

import functools
import inspect
import itertools
import json
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from rllm_entrypoint.api_client import ObserverClient
from rllm_entrypoint.config import cfg
from rllm_entrypoint.events import (
    make_event_callback,
    push_evaluate_event,
    push_finish_event,
    push_heartbeat_event,
    push_startup_event,
)
from rllm_entrypoint.pause_ctrl import PauseController
from rllm_entrypoint.state import AgentPhase, RunState

from openhands.sdk.context import Skill
from openhands.sdk.context.skills import load_project_skills, load_skills_from_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_HARDCODED_SYSTEM_PROMPT = """You are an Ascend NPU Triton operator implementation agent.

Act through tools. Do not explain the task, print raw tool JSON, emit
placeholder code, use `<think>` tags, or call the think tool.

For KernelBench tasks, the first file operation must create the requested
implementation file unless executable reference code is missing. Put code only
in file_editor actions, then run the requested validation pipeline. After each
run, read metrics.json and repair concrete code failures with the smallest
targeted edit.

After metrics.json reports success, stop editing implementation files. Save best
files if needed, then stop. Low speedup is still a successful rollout.
"""


_ASCENDOPGEN_FULL_SYSTEM_PROMPT = """You are an Ascend NPU Triton operator implementation agent.

Act through tools. Do not print raw tool JSON, emit placeholder code, include
private reasoning, use `<think>` tags, or call the think tool.

For AscendOpGenAgent full-flow benchmark tasks, use the workspace instructions
and local skill files as the primary source of workflow. Skills and references
must be loaded progressively, not all at once:

1. Inspect `AGENTS.md`. Use only tools listed in Tools Available. If a native
   skill-invocation tool is actually listed, it may be used; otherwise, use the
   skill manifest in `AGENTS.md` and file tools to read the relevant local
   `SKILL.md` file for the current phase only.
2. Designer phase: read only `triton-op-designer/SKILL.md` and the minimal sketch
   references needed for this operator. Create `src/sketch.txt`; it must contain
   strategy only, not executable Python.
3. Generator phase: after the sketch exists, read only `triton-op-coding/SKILL.md`
   plus the target-architecture and op-type references required by that skill.
   Generate the requested implementation file with complete module-scope Triton
   kernels and `ModelNew`.
4. Verifier phase: after the implementation exists, read only
   `triton-op-verifier/SKILL.md`. Run the requested validation pipeline, read
   `metrics.json` and any referenced error file, then repair concrete code
   failures with targeted edits.
5. When `metrics.json` reports success, save best files and stop. Do not keep
   editing the implementation unless the workspace instructions explicitly ask
   for a post-success optimizer stage.

Hard rules:
- Do not run ad-hoc `python` or `python3` tests for operator debugging. The
  default runner Python may not have torch. Use only the requested pipeline.
- Never return dummy constants or code that only satisfies AST checks.
- Large flattened output spaces must not launch a grid larger than 65535 on any
  axis. Use a fixed core grid with an in-kernel stride loop when needed.
- For broadcasting, derive every input offset from the output index before
  writing code; do not assume shapes without reading `get_inputs()` or
  `get_input_groups()`.
- Scalar reductions must be produced by Triton kernels; do not use `.item()`,
  host-side division, or PyTorch tensor construction for target compute.
- Before creating or repairing kernels, classify the operator from
  `Model.forward()`, `get_inputs()` or `get_input_groups()`, and
  `get_init_inputs()`, then open only the
  generator refs for that class: elementwise/broadcast, matmul/linear,
  convolution/stencil, pooling, interpolate/resample, normalization, softmax, reduction, scan,
  layout/index, gather/scatter/embedding, sort/topk/arg, loss/distance,
  attention, or the matching fused combination. Do not rely on memory for
  Triton-Ascend syntax.
- If terminal output says `Process still running (soft timeout)`, the command is
  still alive. Do not edit code or judge failure from that message. Poll the
  same terminal until the command exits, then read `metrics.json`.

Keep assistant messages brief; put code in file edits, not chat text.
Initial phase keyword: sketch.
"""


def _select_system_prompt() -> str:
    mode = os.environ.get("OPENHANDS_AGENT_MODE", "").strip().lower()
    if mode in {"ascend_full", "ascendopgen_full", "benchmark_ascend_full"}:
        return _ASCENDOPGEN_FULL_SYSTEM_PROMPT
    return _HARDCODED_SYSTEM_PROMPT


def _is_ascend_full_mode() -> bool:
    mode = os.environ.get("OPENHANDS_AGENT_MODE", "").strip().lower()
    return mode in {"ascend_full", "ascendopgen_full", "benchmark_ascend_full"}


def _write_system_prompt() -> None:
    os.makedirs(os.path.dirname(cfg.system_prompt_path) or "/tmp", exist_ok=True)
    with open(cfg.system_prompt_path, "w", encoding="utf-8") as f:
        f.write(_select_system_prompt())


def _operator_metrics_guard_enabled() -> bool:
    """Return whether this run must end with metrics.json success."""

    if cfg.npu_operator_task:
        return True
    if cfg.operator_name and cfg.operator_name != "operator":
        return True
    return "tools/operator_pipeline.sh" in cfg.task_instruction


def _read_metrics_json() -> dict[str, Any] | None:
    path = Path(cfg.workspace_base) / "metrics.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _metrics_success(metrics: dict[str, Any] | None) -> bool:
    return bool(metrics and metrics.get("success") is True)


def _metrics_summary(metrics: dict[str, Any] | None, *, max_chars: int = 3000) -> str:
    if metrics is None:
        return "metrics.json is missing or unreadable."
    text = json.dumps(metrics, ensure_ascii=False, indent=2, default=str)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text


def _operator_impl_path() -> Path:
    op_name = cfg.operator_name or os.environ.get("OPERATOR_NAME", "operator")
    return Path(cfg.workspace_base) / "src" / f"{op_name}_triton_ascend_impl.py"


def _operator_task_path() -> Path:
    op_name = cfg.operator_name or os.environ.get("OPERATOR_NAME", "operator")
    return Path(cfg.workspace_base) / "src" / f"{op_name}.py"


def _operator_best_artifact_paths() -> tuple[Path, Path]:
    impl_path = _operator_impl_path()
    best_impl = impl_path.with_name(f"{impl_path.stem}_best{impl_path.suffix}")
    best_metrics = Path(cfg.workspace_base) / "metrics_best.json"
    return best_impl, best_metrics


def _remove_failed_best_artifacts(metrics: dict[str, Any] | None) -> None:
    """Remove misleading best artifacts created before metrics success."""

    if _metrics_success(metrics):
        return
    for path in _operator_best_artifact_paths():
        try:
            if path.exists():
                path.unlink()
                logger.warning(
                    "[runner] removed invalid best artifact because metrics.json "
                    "success is not true: %s",
                    path,
                )
        except OSError:
            logger.warning(
                "[runner] failed to remove invalid best artifact: %s",
                path,
                exc_info=True,
            )


def _continuation_phase_hint() -> str:
    workspace = Path(cfg.workspace_base)
    sketch_path = workspace / "src" / "sketch.txt"
    impl_path = _operator_impl_path()
    if not sketch_path.is_file():
        return (
            "Current phase: designer sketch.\n"
            "Read only `.agents/skills/triton-op-designer/SKILL.md` and the "
            "minimal designer references matching the operator class. First "
            "re-read the executable reference, classify the op as elementwise/"
            "broadcast, matmul/linear/batched-matmul, convolution/stencil, "
            "pooling, normalization, softmax, reduction/statistical, scan, "
            "layout/index/slice/transpose, gather/scatter/embedding, sort/"
            "topk/arg, interpolate/resample, loss/distance, attention, or fused, then open only the "
            "matching case refs before creating "
            "`src/sketch.txt`. Do not read generator or verifier skills yet."
        )
    if not impl_path.is_file():
        return (
            "Current phase: generator implementation.\n"
            "`src/sketch.txt` exists. Read only "
            "`.agents/skills/triton-op-coding/SKILL.md`, then open the target "
            "hardware ref, fundamentals, and only the op-type refs matching "
            "`src/sketch.txt` and the reference code. Examples: matmul uses "
            "`triton-ascend-matmul.md`; reductions use "
            "`triton-ascend-reduce.md`; fused elementwise+reduction uses both "
            "elementwise and reduce. Do not read verifier references yet."
        )
    return (
        "Current phase: verifier and targeted repair.\n"
        "The implementation file exists. Read only "
        "`.agents/skills/triton-op-verifier/SKILL.md`, then run the requested "
        "pipeline and repair the concrete failure from `metrics.json` and its "
        "referenced error file. For compile/AST errors involving `break`, "
        "`continue`, Python `if`, scalar/vector tensor shape mismatches, "
        "`tl.constexpr` or `tl.*` in `forward`, `.item()`, `torch.tensor`, "
        "forbidden PyTorch ops, or grid/coreDim limits, reopen the generator "
        "fundamentals and the op-type refs matching the concrete error before editing."
        " For matmul correctness failures, reopen the matmul ref and re-derive "
        "M/N/K, strides, and any transpose or batch mapping from the executable "
        "reference before editing."
    )


def _operator_context_text() -> str:
    parts = [
        cfg.operator_name or os.environ.get("OPERATOR_NAME", ""),
        os.environ.get("OP_CLASS_HINT", ""),
    ]
    task_path = _operator_task_path()
    if task_path.is_file():
        parts.append(task_path.read_text(encoding="utf-8", errors="replace")[:8000])
    return "\n".join(part for part in parts if part)


def _metrics_error_text(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return ""
    parts = [
        str(metrics.get("error_type") or ""),
        str(metrics.get("error") or ""),
    ]
    error_file = metrics.get("error_file")
    if error_file:
        path = Path(cfg.workspace_base) / str(error_file)
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8", errors="replace")[-6000:])
    return "\n".join(parts)


def _refs_for_repair_text(text: str) -> list[str]:
    text = text.lower()
    refs: list[str] = []

    def add(*paths: str) -> None:
        for path in paths:
            if path not in refs:
                refs.append(path)

    fundamentals = ".agents/skills/triton-op-coding/references/triton-ascend-fundamentals.md"
    examples = ".agents/skills/triton-op-coding/references/triton-ascend-examples.md"
    elementwise = ".agents/skills/triton-op-coding/references/triton-ascend-elementwise.md"
    reduce = ".agents/skills/triton-op-coding/references/triton-ascend-reduce.md"
    matmul = ".agents/skills/triton-op-coding/references/triton-ascend-matmul.md"
    attention = ".agents/skills/triton-op-coding/references/triton-ascend-attention.md"
    interpolate = ".agents/skills/triton-op-coding/references/triton-ascend-interpolate.md"
    sort_select = ".agents/skills/triton-op-coding/references/triton-ascend-sort-select.md"
    npu_arch = ".agents/skills/npu-arch/references/npu-arch-guide-triton.md"
    hardware = ".agents/skills/npu-arch/references/npu-hardware-params.md"

    if any(
        token in text
        for token in (
            "_builder",
            "did you forget to add @triton.jit",
            "outside of jit",
            "outside @triton.jit",
            "tl.* in forward",
            "invalid host-side triton",
            "triton.language apis may only be used",
        )
    ):
        return [fundamentals, examples]
    if any(token in text for token in ("ast_check_failed", "forbidden", "torch.tensor", ".item", "tl.* in forward", "placeholder", "pass")):
        add(fundamentals)
    if any(token in text for token in ("compile", "lowering", "bisheng", "tritontostructured", "unsupported", "constexpr", "syntaxerror", "typeerror", "valueerror")):
        add(fundamentals, examples)
    if any(token in text for token in ("tl.load", "tl.store", "pointer", "ptr", "mask", "block type", "block-shaped", "data_ptr", "int64")):
        add(fundamentals, elementwise)
    if any(token in text for token in ("scalar", "tl.sum", "tl.max", "tl.min", "atomic", "reduction", "mean", "variance", "layernorm", "groupnorm", "softmax", "loss")):
        add(
            reduce,
            ".agents/skills/triton-latency-optimizer/references/scalar_to_vector.md",
            ".agents/skills/triton-latency-optimizer/references/avoid_scalar_lowering.md",
            ".agents/skills/triton-latency-optimizer/references/pass-merge.md",
        )
    if any(token in text for token in ("shape mismatch", "broadcast", "expand", "stride", "contiguous", "permute", "transpose", "view", "reshape")):
        add(fundamentals, elementwise, examples)
    if any(token in text for token in ("matmul", "linear", " bmm", "mm(", "tl.dot", " dot", "m/n/k", "transposed", "transpose")):
        add(matmul)
    if any(token in text for token in ("conv", "convolution", "stencil", "pool", "avgpool", "maxpool")):
        add(fundamentals, matmul, reduce, examples)
    if any(token in text for token in ("attention", "flash", "qkv", "softmax")):
        add(attention)
    if any(token in text for token in ("sort", "topk", "argmax", "argmin", "argsort", "nms")):
        add(sort_select, reduce)
    if any(token in text for token in ("interpolate", "upsample", "resize", "grid_sample", "resample")):
        add(interpolate, elementwise)
    if any(token in text for token in ("gather", "scatter", "index", "embedding", "nonzero", "where", "slice", "cat", "split", "concat", "layout")):
        add(fundamentals, examples)
    if any(token in text for token in ("ub overflow", "coredim", "65535", "grid", "program_id", "aicore", "npu_runtime_failed")):
        add(fundamentals, npu_arch, hardware, ".agents/skills/triton-latency-optimizer/references/block_size_scaling.md")
    if any(token in text for token in ("dtype", "precision", "tolerance", "relative error", "numerical", "mismatch", "nan", "inf")):
        add(fundamentals, reduce, elementwise)

    if refs:
        return refs
    return [fundamentals]


def _repair_ref_hint(metrics: dict[str, Any] | None) -> str:
    text = "\n".join([_metrics_error_text(metrics), _operator_context_text()])
    refs = _refs_for_repair_text(text)
    return "Suggested refs for this concrete failure:\n" + "\n".join(f"- `{path}`" for path in refs)


def _build_continue_message(metrics: dict[str, Any] | None, attempt: int) -> str:
    op_name = cfg.operator_name or os.environ.get("OPERATOR_NAME", "operator")
    repair_refs = _repair_ref_hint(metrics)
    return (
        "The previous conversation ended before the operator pipeline succeeded. "
        "Start from the current files and repair the failure. Do not claim "
        "success unless metrics.json has \"success\": true.\n\n"
        f"Continuation attempt: {attempt}\n"
        f"{_continuation_phase_hint()}\n\n"
        f"{repair_refs + chr(10) + chr(10) if repair_refs else ''}"
        "Required next actions:\n"
        "1. Follow the current phase hint above and load only that phase's skill.\n"
        "2. If the implementation exists, read metrics.json and any referenced error file.\n"
        "3. Patch only the concrete missing artifact or concrete failure.\n"
        "4. When verifying, run exactly: bash tools/operator_pipeline.sh --op_name "
        f"{op_name}\n"
        "5. Read metrics.json again after every pipeline run.\n"
        "6. Save best files only after metrics.json has \"success\": true. "
        "Any previous best artifacts copied while success was false are invalid "
        "and must not be recreated until the pipeline succeeds.\n"
        "7. Do not call a finish tool unless it is listed in Tools Available. "
        "If no finish tool is listed, stop by sending a brief final assistant "
        "message only after success is true.\n\n"
        "Current metrics summary:\n"
        "```json\n"
        f"{_metrics_summary(metrics)}\n"
        "```"
    )


def _remaining_iteration_budget(run_state: RunState) -> int:
    return max(int(cfg.max_iterations) - int(run_state.iteration), 0)


def _set_conversation_run_budget(conversation: Any, budget: int) -> bool:
    """Best-effort cap for the next Conversation.run() call."""

    budget = max(int(budget), 0)
    changed = False
    for attr in ("max_iteration_per_run", "_max_iteration_per_run"):
        if hasattr(conversation, attr):
            try:
                setattr(conversation, attr, budget)
                changed = True
            except Exception:
                logger.debug("[runner] failed to set conversation.%s", attr, exc_info=True)

    config = getattr(conversation, "config", None)
    if config is not None and hasattr(config, "max_iteration_per_run"):
        try:
            setattr(config, "max_iteration_per_run", budget)
            changed = True
        except Exception:
            logger.debug(
                "[runner] failed to set conversation.config.max_iteration_per_run",
                exc_info=True,
            )

    state = getattr(conversation, "state", None)
    if state is not None and hasattr(state, "max_iterations"):
        try:
            setattr(state, "max_iterations", budget)
            changed = True
        except Exception:
            logger.debug(
                "[runner] failed to set conversation.state.max_iterations",
                exc_info=True,
            )

    if changed:
        logger.info("[runner] Conversation.run() iteration budget set to %d", budget)
    else:
        logger.warning(
            "[runner] Could not find a Conversation per-run iteration budget "
            "attribute; continuing with SDK default."
        )
    return changed


def _run_conversation_with_remaining_budget(
    conversation: Any, run_state: RunState
) -> bool:
    remaining = _remaining_iteration_budget(run_state)
    if remaining <= 0:
        logger.warning(
            "[runner] No iteration budget remains before Conversation.run(); "
            "iteration=%d max_iterations=%d",
            run_state.iteration,
            cfg.max_iterations,
        )
        return False
    try:
        run_signature = inspect.signature(conversation.run)
    except (TypeError, ValueError):
        run_signature = None

    if run_signature is not None:
        params = run_signature.parameters
        if "max_iteration_per_run" in params:
            conversation.run(max_iteration_per_run=remaining)
            return True
        if "max_iterations" in params:
            conversation.run(max_iterations=remaining)
            return True

    _set_conversation_run_budget(conversation, remaining)
    conversation.run()
    return True


# ---------------------------------------------------------------------------
# Workspace skill merging
# ---------------------------------------------------------------------------

def merge_workspace_skills(workspace_base: str, task_scope: Skill = None) -> list:
    """Merge AGENTS.md plus optional SDK skills.

    In Ascend full-flow mode we intentionally do not inject `.agents/skills/*`
    into AgentContext. Those skill bodies and references are read through file
    tools only when the current phase needs them. This keeps progressive
    disclosure real and prevents large/garbled reference text from flooding the
    model context.
    """
    ws = Path(workspace_base)
    skills: list = []
    seen: set[str] = set()

    def skill_key(skill: Any) -> str:
        name = getattr(skill, "name", None) or getattr(skill, "id", None)
        if name:
            return str(name)
        path = getattr(skill, "path", None) or getattr(skill, "location", None)
        if path:
            return str(path)
        return repr(skill)

    def extend_unique(items: Any) -> None:
        for skill in items:
            key = skill_key(skill)
            if key in seen:
                continue
            seen.add(key)
            skills.append(skill)

    def is_agent_skill_file(skill: Any) -> bool:
        source = (
            getattr(skill, "source", None)
            or getattr(skill, "path", None)
            or getattr(skill, "location", None)
            or ""
        )
        source_text = str(source).replace("\\", "/")
        return "/.agents/skills/" in source_text or source_text.startswith(".agents/skills/")

    loaded = load_project_skills(work_dir=str(ws))
    if loaded:
        loaded_items = loaded if isinstance(loaded, list) else list(loaded)
        if _is_ascend_full_mode():
            loaded_items = [skill for skill in loaded_items if not is_agent_skill_file(skill)]
        extend_unique(loaded_items)

    inject_agent_skills = os.environ.get("OPENHANDS_ENABLE_AGENT_SKILLS", "0") in (
        "1",
        "true",
        "True",
        "yes",
    )
    if _is_ascend_full_mode():
        inject_agent_skills = os.environ.get(
            "OPENHANDS_ASCEND_FULL_INJECT_SKILLS", "0"
        ) in ("1", "true", "True", "yes")

    if inject_agent_skills:
        agents_skills_root = ws / ".agents" / "skills"
        if agents_skills_root.is_dir():
            repo_skills, knowledge_skills, agent_skills = load_skills_from_dir(
                str(agents_skills_root)
            )
            for collection in (repo_skills, knowledge_skills, agent_skills):
                extend_unique(
                    collection.values() if hasattr(collection, "values") else collection
                )

    if task_scope is not None:
        extend_unique([task_scope])
    return skills


# ---------------------------------------------------------------------------
# Env-var snapshot
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")


def _safe_env_snapshot() -> dict[str, str]:
    result = {}
    for k, v in os.environ.items():
        upper = k.upper()
        if any(pat in upper for pat in _SECRET_PATTERNS):
            result[k] = "***REDACTED***"
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# LLM latency probe
# ---------------------------------------------------------------------------

def install_llm_latency_probe(
    llm,
    run_state: RunState | None = None,
    log_path: str | None = None,
):
    """
    Patch OpenHands SDK LLM instance to log latency for each LLM inference call.

    It tries to wrap common LLM call methods:
      - completion
      - acompletion
      - chat_completion
      - achat_completion
      - __call__

    The exact method name depends on the OpenHands SDK version.
    """

    if log_path is None:
        log_path = f"/home/p00938733/llm_latency_{os.getpid()}.log"

    pid = os.getpid()
    rank = os.getenv("RANK", "NA")
    local_rank = os.getenv("LOCAL_RANK", "NA")
    visible_devices = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "NA")

    call_counter = itertools.count()

    def _write_latency_log(msg: str) -> None:
        """
        Write logs both to logger and to an independent file.

        File logging is important because Ray may deduplicate stdout/stderr logs.
        """

        logger.info(msg)

        try:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            logger.debug("[llm-latency] failed to write latency log", exc_info=True)

    def _summarize_args(args, kwargs) -> str:
        """
        Avoid printing full prompt/messages because they may be huge.

        Only print rough size/shape information.
        """

        parts = []

        if args:
            parts.append(f"args_len={len(args)}")

        if kwargs:
            parts.append(f"kwargs_keys={list(kwargs.keys())}")

            messages = kwargs.get("messages")
            if isinstance(messages, list):
                parts.append(f"messages_len={len(messages)}")
                try:
                    char_len = 0
                    for m in messages:
                        if isinstance(m, dict):
                            char_len += len(str(m.get("content", "")))
                        else:
                            char_len += len(str(m))
                    parts.append(f"messages_chars={char_len}")
                except Exception:
                    pass

            prompt = kwargs.get("prompt")
            if prompt is not None:
                parts.append(f"prompt_chars={len(str(prompt))}")

            tools = kwargs.get("tools")
            if tools is not None:
                try:
                    parts.append(f"tools_len={len(tools)}")
                except Exception:
                    parts.append("tools_present=True")

        return " ".join(parts)

    def _wrap_sync_method(method_name: str, orig_method):
        @functools.wraps(orig_method)
        def wrapper(*args, **kwargs):
            call_id = next(call_counter)
            start_wall = time.time()
            start_perf = time.perf_counter()

            if run_state is not None:
                try:
                    run_state.total_llm_calls += 1
                except Exception:
                    pass

            arg_summary = _summarize_args(args, kwargs)

            _write_latency_log(
                f"[LLM_CALL_START] "
                f"call_id={call_id} method={method_name} "
                f"wall={start_wall:.6f} "
                f"pid={pid} rank={rank} local_rank={local_rank} "
                f"ASCEND_RT_VISIBLE_DEVICES={visible_devices} "
                f"llm_type={type(llm).__name__} llm_id={id(llm)} "
                f"{arg_summary}"
            )

            ok = False
            try:
                ret = orig_method(*args, **kwargs)
                ok = True
                return ret
            except Exception as e:
                elapsed = time.perf_counter() - start_perf
                _write_latency_log(
                    f"[LLM_CALL_ERROR] "
                    f"call_id={call_id} method={method_name} "
                    f"elapsed_s={elapsed:.6f} "
                    f"error={repr(e)}\n"
                    f"{traceback.format_exc()}"
                )
                raise
            finally:
                elapsed = time.perf_counter() - start_perf
                end_wall = time.time()

                _write_latency_log(
                    f"[LLM_CALL_END] "
                    f"call_id={call_id} method={method_name} ok={ok} "
                    f"elapsed_s={elapsed:.6f} "
                    f"start_wall={start_wall:.6f} end_wall={end_wall:.6f} "
                    f"pid={pid} rank={rank} local_rank={local_rank}"
                )

        return wrapper

    def _wrap_async_method(method_name: str, orig_method):
        @functools.wraps(orig_method)
        async def wrapper(*args, **kwargs):
            call_id = next(call_counter)
            start_wall = time.time()
            start_perf = time.perf_counter()

            if run_state is not None:
                try:
                    run_state.total_llm_calls += 1
                except Exception:
                    pass

            arg_summary = _summarize_args(args, kwargs)

            _write_latency_log(
                f"[LLM_CALL_START] "
                f"call_id={call_id} method={method_name} "
                f"wall={start_wall:.6f} "
                f"pid={pid} rank={rank} local_rank={local_rank} "
                f"ASCEND_RT_VISIBLE_DEVICES={visible_devices} "
                f"llm_type={type(llm).__name__} llm_id={id(llm)} "
                f"{arg_summary}"
            )

            ok = False
            try:
                ret = await orig_method(*args, **kwargs)
                ok = True
                return ret
            except Exception as e:
                elapsed = time.perf_counter() - start_perf
                _write_latency_log(
                    f"[LLM_CALL_ERROR] "
                    f"call_id={call_id} method={method_name} "
                    f"elapsed_s={elapsed:.6f} "
                    f"error={repr(e)}\n"
                    f"{traceback.format_exc()}"
                )
                raise
            finally:
                elapsed = time.perf_counter() - start_perf
                end_wall = time.time()

                _write_latency_log(
                    f"[LLM_CALL_END] "
                    f"call_id={call_id} method={method_name} ok={ok} "
                    f"elapsed_s={elapsed:.6f} "
                    f"start_wall={start_wall:.6f} end_wall={end_wall:.6f} "
                    f"pid={pid} rank={rank} local_rank={local_rank}"
                )

        return wrapper

    candidate_methods = [
        "completion",
        "acompletion",
        "chat_completion",
        "achat_completion",
        "__call__",
    ]

    patched = []

    for method_name in candidate_methods:
        if not hasattr(llm, method_name):
            continue

        orig_method = getattr(llm, method_name)

        if not callable(orig_method):
            continue

        if getattr(orig_method, "_llm_latency_probe_patched", False):
            continue

        if inspect.iscoroutinefunction(orig_method):
            wrapped = _wrap_async_method(method_name, orig_method)
        else:
            wrapped = _wrap_sync_method(method_name, orig_method)

        setattr(wrapped, "_llm_latency_probe_patched", True)

        try:
            object.__setattr__(llm, method_name, wrapped)
            patched.append(method_name)
        except Exception:
            logger.exception("[llm-latency] failed to patch method: %s", method_name)

    _write_latency_log(
        f"[LLM_PROBE_INSTALLED] "
        f"pid={pid} rank={rank} local_rank={local_rank} "
        f"llm_type={type(llm)} llm_id={id(llm)} "
        f"patched_methods={patched} log_path={log_path}"
    )

    if not patched:
        candidate_attrs = []
        try:
            for name in dir(llm):
                try:
                    attr = getattr(llm, name)
                except Exception:
                    continue
                if callable(attr) and any(
                    k in name.lower()
                    for k in ["completion", "chat", "call", "response", "generate", "invoke"]
                ):
                    candidate_attrs.append(name)
        except Exception:
            pass

        _write_latency_log(
            f"[LLM_PROBE_WARNING] no method patched. "
            f"llm_type={type(llm)} "
            f"candidate_attrs={candidate_attrs}"
        )

    return patched


# ---------------------------------------------------------------------------
# Heartbeat timer
# ---------------------------------------------------------------------------

class _HeartbeatTimer:
    """
    Periodically emits a HeartbeatEvent via push_heartbeat_event().

    Uses a daemon thread so it never blocks program exit.
    """

    def __init__(
        self,
        client: ObserverClient,
        run_state: RunState,
        interval_s: float = 15.0,
    ) -> None:
        self._client = client
        self._state = run_state
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop,
            name="rllm-heartbeat",
            daemon=True,
        )

    def start(self) -> None:
        if self._client._enabled:
            self._thread.start()
            logger.info("[heartbeat] Timer started (interval=%.1fs)", self._interval)

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 5.0) -> None:
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        while not self._stop.wait(timeout=self._interval):
            try:
                push_heartbeat_event(self._client, self._state)
            except Exception:
                logger.debug("[heartbeat] push failed", exc_info=True)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def build_run_state() -> RunState:
    """Populate a RunState with all static config values."""
    state = RunState(
        session_id=cfg.session_id,
        session_label=cfg.session_label,
        task_instruction=cfg.task_instruction,
        workspace_base=cfg.workspace_base,
        llm_model=cfg.llm_model,
        llm_base_url=cfg.llm_base_url,
        max_iterations=cfg.max_iterations,
        ops_name=cfg.operator_name,
        ops_arch=cfg.operator_arch,
        is_running=False,
        phase=AgentPhase.INITIALIZING,
        env_vars=_safe_env_snapshot(),
        extra_metadata=_parse_extra_metadata(),
    )
    return state


def _parse_extra_metadata() -> dict[str, Any]:
    if not cfg.extra_metadata:
        return {}
    try:
        return json.loads(cfg.extra_metadata)
    except json.JSONDecodeError:
        return {"raw": cfg.extra_metadata}


def run() -> int:
    """Build SDK objects, run the conversation, return exit code."""
    from openhands.sdk import LLM, Agent, AgentContext, Conversation
    from openhands.sdk.tool import Tool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.terminal import TerminalTool

    # Validate config
    if not cfg.llm_base_url:
        logger.error("LLM_BASE_URL is not set. Exiting.")
        return 1

    if not cfg.task_instruction:
        logger.error("No task instruction provided. Exiting.")
        return 1

    logger.info("LLM_BASE_URL : %s...", cfg.llm_base_url[:80])
    logger.info("LLM_MODEL    : %s", cfg.llm_model)
    logger.info("WORKSPACE    : %s", cfg.workspace_base)
    logger.info("MAX_ITER     : %d", cfg.max_iterations)
    logger.info("SESSION_ID   : %s", cfg.session_id)
    logger.info("TASK         : %.120s", cfg.task_instruction)
    logger.info(
        "ASCEND_RT_VISIBLE_DEVICES : %s...",
        os.getenv("ASCEND_RT_VISIBLE_DEVICES", "NA"),
    )

    if cfg.observer_api_url:
        logger.info("OBSERVER_URL : %s", cfg.observer_api_url)
    else:
        logger.info("OBSERVER_URL : (disabled 鈥?running standalone)")

    # Write system prompt
    _write_system_prompt()

    # Initialise RunState
    run_state = build_run_state()

    # Initialise observer client
    client = ObserverClient(
        base_url=cfg.observer_api_url,
        session_id=cfg.session_id,
    )

    # 鈹€鈹€ Push startup lifecycle event 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    push_startup_event(client, run_state)

    # 鈹€鈹€ Start heartbeat timer 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    heartbeat = _HeartbeatTimer(
        client=client,
        run_state=run_state,
        interval_s=cfg.upload_interval_s,
    )
    heartbeat.start()

    # 鈹€鈹€ Start pause controller 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    pause_ctrl = PauseController(
        run_state,
        client,
        poll_interval_s=cfg.pause_poll_interval_s,
    )
    pause_ctrl.start()

    # 鈹€鈹€ Build event callback 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    _event_cb = make_event_callback(run_state, client)

    # 鈹€鈹€ Build OpenHands SDK objects 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    llm = LLM(
        usage_id="rllm-openhands",
        model=cfg.llm_model,
        api_key=SecretStr(cfg.llm_api_key),
        base_url=cfg.llm_base_url or None,
        max_output_tokens=int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "4096")),
    )

    def debug_llm_identity(llm, agent):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        pid = os.getpid()
        rank = os.getenv("RANK", "NA")
        local_rank = os.getenv("LOCAL_RANK", "NA")

        msg = (
            f"[{ts}] "
            f"[pid={pid}] "
            f"[rank={rank}] "
            f"[local_rank={local_rank}] "
            f"[input llm type={type(llm)}] "
            f"[input llm id={id(llm)}] "
            f"[agent.llm type={type(agent.llm)}] "
            f"[agent.llm id={id(agent.llm)}] "
            f"[same_obj={llm is agent.llm}]"
        )

        path = f"/home/p00938733/openhands_{os.getpid()}.log"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"{time.time()} {msg}\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            logger.debug("[debug_llm_identity] failed to write log", exc_info=True)

        logger.info(msg)

    try:
        _merged_skills = merge_workspace_skills(cfg.workspace_base)
    except Exception:
        logger.exception("merge_workspace_skills failed; falling back to empty skills")
        _merged_skills = []

    agent_context = AgentContext(
        skills=_merged_skills,
        load_public_skills=False,
        system_message_suffix=(
            f"Workspace directory: {cfg.workspace_base}. "
            f"Maximum iterations budget: {cfg.max_iterations}. "
            f"Session ID: {cfg.session_id}."
        ),
    )

    terminal_no_change_timeout = float(
        os.environ.get("OPENHANDS_TERMINAL_NO_CHANGE_TIMEOUT_SECONDS", "300")
    )
    logger.info(
        "[runner] TerminalTool no_change_timeout_seconds=%s",
        terminal_no_change_timeout,
    )

    agent = Agent(
        llm=llm,
        tools=[
            Tool(
                name=TerminalTool.name,
                params={
                    "no_change_timeout_seconds": terminal_no_change_timeout,
                },
            ),
            Tool(name=FileEditorTool.name),
        ],
        include_default_tools=[],
        agent_context=agent_context,
        system_prompt_filename=cfg.system_prompt_path,
    )

    debug_llm_identity(llm, agent)

    # 鈹€鈹€ Install LLM latency probe 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    #
    # Important:
    #   Patch agent.llm instead of only llm, because Agent may internally keep,
    #   copy, or wrap the LLM object depending on OpenHands SDK version.
    #
    install_llm_latency_probe(
        agent.llm,
        run_state=run_state,
        log_path=f"/home/p00938733/llm_latency_{cfg.session_id}_{os.getpid()}.log",
    )

    conversation = Conversation(
        agent=agent,
        workspace=cfg.workspace_base,
        max_iteration_per_run=cfg.max_iterations,
        callbacks=[_event_cb],
    )

    # Wire pause controller to the live conversation object
    pause_ctrl.set_conversation(conversation)

    # Reflect conversation ID in state
    run_state.set_conversation_status("idle", str(conversation.id))

    # Run loop
    exit_code = 0
    final_status = "unknown"
    metrics_guard_enabled = _operator_metrics_guard_enabled()
    forced_continue_count = 0
    max_forced_continues = int(
        os.environ.get("OPENHANDS_METRICS_GUARD_CONTINUES", str(cfg.max_iterations))
    )
    run_state.set_running(True)
    run_state.set_phase(AgentPhase.WAITING_FOR_REPLY)

    try:
        conversation.send_message(cfg.task_instruction)

        while True:
            run_state.set_conversation_status(
                conversation.state.execution_status.value,
                str(conversation.id),
            )
            run_state.set_phase(AgentPhase.WAITING_FOR_REPLY)

            if not _run_conversation_with_remaining_budget(conversation, run_state):
                exit_code = 1
                final_status = "iteration_budget_exhausted"
                run_state.set_error("iteration budget exhausted before conversation.run()")
                break

            # Check if resume was requested by pause controller
            if pause_ctrl.resume_requested:
                logger.info("[runner] Resume requested; re-entering conversation.run()")
                pause_ctrl.resume_requested = False
                continue

            # Check conversation terminal status
            exec_status = conversation.state.execution_status.value
            run_state.set_conversation_status(exec_status, str(conversation.id))

            if exec_status in ("finished", "error", "stuck"):
                if exec_status == "finished" and metrics_guard_enabled:
                    metrics = _read_metrics_json()
                    if not _metrics_success(metrics):
                        _remove_failed_best_artifacts(metrics)
                        remaining_iterations = _remaining_iteration_budget(run_state)
                        can_continue = (
                            remaining_iterations > 0
                            and forced_continue_count < max_forced_continues
                        )
                        if can_continue:
                            forced_continue_count += 1
                            logger.warning(
                                "[runner] Conversation finished but metrics.json "
                                "success is not true; starting compact repair conversation "
                                "(attempt=%d remaining_iterations=%d)",
                                forced_continue_count,
                                remaining_iterations,
                            )
                            conversation = Conversation(
                                agent=agent,
                                workspace=cfg.workspace_base,
                                max_iteration_per_run=remaining_iterations,
                                callbacks=[_event_cb],
                            )
                            pause_ctrl.set_conversation(conversation)
                            run_state.set_conversation_status("idle", str(conversation.id))
                            run_state.set_phase(AgentPhase.WAITING_FOR_REPLY)
                            conversation.send_message(
                                _build_continue_message(metrics, forced_continue_count)
                            )
                            continue

                        logger.warning(
                            "[runner] Conversation finished without metrics success "
                            "and cannot continue (remaining_iterations=%d, "
                            "forced_continues=%d/%d)",
                            remaining_iterations,
                            forced_continue_count,
                            max_forced_continues,
                        )

                logger.info("[runner] Conversation reached terminal status: %s", exec_status)
                break

            # Paused but no resume pending: wait for resume signal
            if exec_status == "paused":
                logger.info("[runner] Conversation paused. Waiting for resume signal...")

                while (
                    not pause_ctrl.resume_requested
                    and not pause_ctrl._stop_event.is_set()
                ):
                    time.sleep(0.5)
                    exec_status = conversation.state.execution_status.value

                    if exec_status not in ("paused",):
                        break

                if pause_ctrl.resume_requested:
                    pause_ctrl.resume_requested = False
                    logger.info("[runner] Resuming after pause...")
                    continue

                break

            # All other statuses, for example idle / waiting_for_confirmation
            break

        # Log LLM cost
        cost = 0.0
        llm_calls = run_state.total_llm_calls

        if llm.metrics is not None:
            cost = float(llm.metrics.accumulated_cost or 0.0)
            run_state.update_metrics(cost=cost, llm_calls=llm_calls)
            logger.info("EXAMPLE_COST: %s", cost)

        exec_status = conversation.state.execution_status.value
        final_status = exec_status
        if exec_status in ("error", "stuck"):
            exit_code = 1
            run_state.set_error(f"conversation ended with status={exec_status}")

        if metrics_guard_enabled:
            metrics = _read_metrics_json()
            if not _metrics_success(metrics):
                _remove_failed_best_artifacts(metrics)
                exit_code = 1
                final_status = "metrics_failed"
                run_state.set_conversation_status(final_status, str(conversation.id))
                run_state.set_error(
                    "metrics.json success is not true after conversation ended"
                )
                logger.warning(
                    "[runner] metrics guard failed: %s",
                    _metrics_summary(metrics, max_chars=1000),
                )

        logger.info("Conversation completed. Status=%s", final_status)
        if exit_code == 0:
            run_state.set_phase(AgentPhase.FINISHED)

        # Evaluation event
        push_evaluate_event(
            client,
            run_state,
            status=final_status,
            cost=cost,
            llm_calls=llm_calls,
            iterations=run_state.iteration,
        )

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        run_state.set_phase(AgentPhase.ERROR)
        run_state.set_error("KeyboardInterrupt")
        exit_code = 1

    except Exception as exc:
        logger.exception("Unhandled exception in runner.")
        run_state.set_phase(AgentPhase.ERROR)
        run_state.set_error(str(exc))
        exit_code = 1

    finally:
        run_state.set_running(False)

        # 鈹€鈹€ Finish event 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        try:
            push_finish_event(
                client,
                run_state,
                exit_code=exit_code,
                reason="error" if exit_code != 0 else "completed",
            )
        except Exception:
            logger.debug("[runner] push_finish_event failed", exc_info=True)

        # 鈹€鈹€ Stop background threads 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        heartbeat.stop()
        pause_ctrl.stop()
        heartbeat.join(timeout=5)

    return exit_code
