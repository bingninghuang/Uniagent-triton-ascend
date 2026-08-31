"""Reward helpers for Triton/Ascend KernelBench agent rollouts."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
from pathlib import Path
from typing import Any

# Shared reward formula (single source of truth with snapshot_verify_best.py,
# which imports reward_formula from the same tools/ dir inside the sandbox).
# Load it by path so the trainer and sandbox run identical math.
import sys as _sys
_TOOLS_DIR = str(Path(__file__).parent / "workspace" / "agent_workdir" / "tools")
if _TOOLS_DIR not in _sys.path:
    _sys.path.insert(0, _TOOLS_DIR)
import reward_formula as _rf

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE_DIR = os.environ.get("TRITON_WORKSPACE_DIR", "/opt/workspace/agent_workdir")


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False", "no", "")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _speedup_from_perf_data(perf_data: dict[str, Any]) -> float:
    for key in ("speedup_vs_torch", "speedup", "geomean_speedup"):
        if key in perf_data:
            return max(0.0, _safe_float(perf_data.get(key), 0.0))
    return 0.0


def reward_breakdown_from_metrics(perf: dict[str, Any]) -> dict[str, Any]:
    """Stable partial-credit reward components from ``metrics.json``.

    Delegates to the shared ``reward_formula.reward_breakdown`` so the sandbox
    snapshot path and the trainer compute identical rewards.
    """
    return _rf.reward_breakdown(perf)


def reward_from_metrics(perf: dict[str, Any]) -> float:
    return float(reward_breakdown_from_metrics(perf)["total"])


def _attach_reward_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    """Add flat reward fields without wrapping the verifier metrics payload.

    Delegates to the shared ``reward_formula.attach_reward_fields``.
    """
    return _rf.attach_reward_fields(metrics)


def _metrics_resolved(perf: dict[str, Any] | None) -> bool:
    if not isinstance(perf, dict):
        return False
    return bool(perf.get("success", False) and perf.get("correctness_ok", False))


def _perf_matches_op(perf: dict[str, Any], op_name: str) -> bool:
    """Ignore stale perf/verify JSON left by a different op on the same workspace."""
    file_op = perf.get("op_name")
    if not file_op:
        return True
    return str(file_op) == op_name


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> dict:
    """Reward worker entry for blackbox training.

    The Claude Code runner evaluates inside the same sandbox and passes
    ``reward_score`` through ``complete_session(..., reward_info=...)``. The
    framework injects that into ``extra_info`` before the reward worker calls
    this function.
    """
    del data_source, solution_str, ground_truth
    score = 0.0
    if extra_info and "reward_score" in extra_info:
        score = float(extra_info["reward_score"])
    result: dict[str, Any] = {"score": score}
    if isinstance(extra_info, dict):
        for key in ("selected_metrics_source", "used_best_metrics", "train_best"):
            if key in extra_info:
                result[key] = extra_info[key]
        for metrics_key in ("best_metrics", "metrics"):
            metrics = extra_info.get(metrics_key)
            if not isinstance(metrics, dict):
                continue
            compact_metrics = {
                key: metrics[key]
                for key in (
                    "passed_cases",
                    "total_cases",
                    "pass_rate",
                    "reward",
                    "success",
                    "error_type",
                )
                if key in metrics
            }
            if compact_metrics:
                result[metrics_key] = compact_metrics
    return result


async def _remote_file_exists(env, path: str) -> bool:
    quoted = shlex.quote(path)
    try:
        output = await env.communicate(f"if [ -f {quoted} ]; then echo yes; else echo no; fi", check="ignore")
    except Exception as exc:
        logger.warning("Remote file existence check failed for %s: %s", path, exc)
        return False
    return output.strip().splitlines()[-1:] == ["yes"]


async def _remote_json(env, path: str) -> dict[str, Any] | None:
    if not await _remote_file_exists(env, path):
        return None
    try:
        text = await env.read_file(path)
        return json.loads(text.lstrip("\ufeff"))
    except Exception as exc:
        logger.warning("Failed to read JSON %s: %s", path, exc)
        return None


async def _remote_write_json(env, path: str, payload: dict[str, Any]) -> None:
    # Use runtime write_file (HTTP body), not a shell argv embedding.
    # Large metrics/perf_data used to blow ARG_MAX via:
    #   printf %s '<json>' | python3 -c ...  -> OSError: Argument list too long
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    await env.write_file(path, text)


async def _remote_copy_file(env, src: str, dst: str) -> bool:
    if not await _remote_file_exists(env, src):
        return False
    copy_timeout = _safe_int(os.environ.get("TRITON_REMOTE_COPY_TIMEOUT"), 300)
    try:
        await env.communicate(
            f"/bin/cp -f -- {shlex.quote(src)} {shlex.quote(dst)} </dev/null",
            timeout=copy_timeout,
            check="ignore",
        )
    except Exception as exc:
        logger.warning("Remote copy failed: %s -> %s: %s", src, dst, exc)
        return False
    return await _remote_file_exists(env, dst)



async def _remote_mtime(env, path: str) -> float | None:
    code = "import os, sys; path = sys.argv[1]; print(os.path.getmtime(path) if os.path.exists(path) else '')"
    output = await env.communicate(
        f"python3 -c {shlex.quote(code)} {shlex.quote(path)}",
        check="ignore",
    )
    text = output.strip().splitlines()[-1] if output.strip() else ""
    try:
        return float(text) if text else None
    except ValueError:
        return None


async def _remote_ast_check_file(env, workspace_dir: str, impl_file: str) -> dict[str, Any] | None:
    checker = f"{workspace_dir}/tools/triton-op-verifier/scripts/validate_triton_impl.py"
    if not await _remote_file_exists(env, checker):
        return None
    cmd = (
        f"( cd {shlex.quote(workspace_dir)} && "
        "set +e; "
        f"python3 {shlex.quote(checker)} {shlex.quote(impl_file)} --json 2>&1; "
        "status=$?; printf '\\n__AST_CHECK_EXIT__=%s\\n' \"$status\"; exit 0 )"
    )
    try:
        output = await env.communicate(cmd, check="ignore")
    except Exception as exc:
        logger.warning("AST check command failed unexpectedly: %s", exc)
        return {"valid": False, "error_type": type(exc).__name__, "raw_output": str(exc)[-2000:]}
    text = output.strip()
    if not text:
        return None
    exit_code = None
    marker = "__AST_CHECK_EXIT__="
    if marker in text:
        tail = text.rsplit(marker, 1)[-1].splitlines()[0].strip()
        try:
            exit_code = int(tail)
        except ValueError:
            exit_code = None
        text = text.split(marker, 1)[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {"valid": False, "exit_code": exit_code, "raw_output": text[-2000:]}
    try:
        data = json.loads(text[start : end + 1])
        if isinstance(data, dict):
            data["exit_code"] = exit_code
        return data
    except json.JSONDecodeError:
        return {"valid": False, "exit_code": exit_code, "raw_output": text[-2000:]}


async def _remote_ast_check(env, workspace_dir: str, op_name: str) -> dict[str, Any] | None:
    impl_file = f"{workspace_dir}/src/{op_name}_triton_ascend_impl.py"
    return await _remote_ast_check_file(env, workspace_dir, impl_file)


def _verify_to_metrics(verify: dict[str, Any], op_name: str, ast_ok: bool) -> dict[str, Any]:
    """Correctness metrics come only from ``verify_result.json``.

    ``perf_result.json`` is never used for ``passed_cases`` / ``correctness_ok``
    because benchmark skip/fail (empty tensors, missing profiler samples) is
    not a correctness failure.
    """
    total = _safe_int(verify.get("total_cases"), 0)
    passed = _safe_int(verify.get("passed_cases"), 0)
    failed = _safe_int(verify.get("failed_cases"), max(total - passed, 0))
    correctness_ok = total > 0 and passed == total and failed == 0
    compile_ok = bool(
        correctness_ok
        or passed > 0
        or verify.get("compile_ok")
        or verify.get("output_observed")
    )
    metrics: dict[str, Any] = {
        "op_name": op_name,
        "ast_check_ok": ast_ok,
        "compile_ok": compile_ok,
        "correctness_ok": correctness_ok,
        "success": correctness_ok,
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": failed,
        "pass_rate": round(passed / max(total, 1), 6) if total > 0 else 0.0,
        "perf_data": {},
        "perf_missing": True,
    }
    if not correctness_ok:
        metrics["error_type"] = verify.get("error_type") or "correctness_failed"
        metrics["error"] = verify.get("error") or metrics["error_type"]
    return metrics


def _attach_speedup(metrics: dict[str, Any], perf: dict[str, Any] | None) -> dict[str, Any]:
    """Overlay speedup from perf JSON without touching correctness counts."""
    if not isinstance(perf, dict):
        return _attach_reward_fields(metrics)
    speedup = _speedup_from_perf_data(perf)
    metrics["perf_data"] = perf
    metrics["perf_missing"] = speedup <= 0.0
    return _attach_reward_fields(metrics)


def reward_from_verify_and_perf(
    verify: dict[str, Any] | None,
    perf: dict[str, Any] | None,
    op_name: str,
    ast_ok: bool,
) -> float:
    """Compute a segment-boundary reward from verify + perf + ast_ok.

    Uses the exact same pipeline as ``evaluate_triton_workspace`` so segment
    local rewards are on the same footing as the final reward: correctness
    derived from ``verify_result.json``, speedup overlaid from ``perf``, and
    ``ast_ok`` taken from the AST check at that boundary.
    """
    if not isinstance(verify, dict):
        return 0.0
    metrics = _attach_speedup(_verify_to_metrics(verify, op_name, ast_ok), perf)
    return reward_from_metrics(metrics)


async def _load_matching_json(env, path: str, op_name: str) -> dict[str, Any] | None:
    data = await _remote_json(env, path)
    if isinstance(data, dict) and _perf_matches_op(data, op_name):
        return data
    return None


async def evaluate_triton_workspace(
    env,
    metadata: dict[str, Any],
    *,
    workspace_dir: str = DEFAULT_WORKSPACE_DIR,
) -> tuple[float, dict[str, Any]]:
    """Evaluate a Triton workspace: correctness from verify, speedup from perf.

    ``TRITON_TRAIN_BEST_FIRST=0`` (eval default): latest ``verify_result.json``
    plus latest ``perf_result.json`` speedup. Best snapshots
    (``metrics_best.json`` / ``perf_result_best.json`` / ``*_impl_best.py``)
    are still saved during ``run_verify`` / ``benchmark.py``. ``metrics_best``
    embeds that impl's ``speedup_vs_torch`` / ``perf_data``.

    ``TRITON_TRAIN_BEST_FIRST=1``: score ``metrics_best.json`` as a whole
    (correctness + the speedup stored on that same object).
    """
    op_name = str(metadata.get("op_name", "operator"))
    impl_file = f"{workspace_dir}/src/{op_name}_triton_ascend_impl.py"
    best_impl = f"{workspace_dir}/src/{op_name}_triton_ascend_impl_best.py"
    metrics_file = f"{workspace_dir}/metrics.json"

    def _fail(reason: str, reward: float = 0.0) -> tuple[float, dict[str, Any]]:
        return reward, {"eval_completed": True, "op_name": op_name, "reward": reward, "reason": reason}

    # ---- 1. impl file must exist ----
    if not await _remote_file_exists(env, impl_file):
        return _fail("missing_impl")

    best_first = _truthy_env("TRITON_TRAIN_BEST_FIRST", "0")
    verify_path = f"{workspace_dir}/output/verify/verify_result.json"
    latest_verify = await _load_matching_json(env, verify_path, op_name)
    latest_perf = await _load_matching_json(
        env, f"{workspace_dir}/output/perf_result.json", op_name,
    )
    best_metrics = await _load_matching_json(
        env, f"{workspace_dir}/metrics_best.json", op_name,
    )

    # ---- 2. correctness source is always verify (or metrics_best, which is
    #         itself derived from verify). Never perf_result.passed_cases. ----
    source = "verify_result"
    verify: dict[str, Any] | None = latest_verify
    perf_for_speedup = latest_perf
    ast_impl_is_best = False
    if best_first and isinstance(best_metrics, dict):
        source = "metrics_best"
        verify = best_metrics
        # Speedup must come from the same impl as metrics_best (embedded
        # perf_data / speedup_vs_torch). Do not overlay perf_result_best.json,
        # which can be a faster different kernel.
        perf_for_speedup = None
        ast_impl_is_best = True
    elif verify is None:
        return _fail("no_verify_results")

    # ---- 3. AST check (best impl when scoring the best snapshot) ----
    if ast_impl_is_best and await _remote_file_exists(env, best_impl):
        ast_check = await _remote_ast_check_file(env, workspace_dir, best_impl)
    else:
        ast_check = await _remote_ast_check(env, workspace_dir, op_name)
    ast_ok = bool(ast_check.get("valid")) if isinstance(ast_check, dict) else False
    if not ast_ok:
        return _fail("ast_check_failed")

    # ---- 4. compute reward ----
    if source == "metrics_best" and isinstance(verify, dict) and "passed_cases" in verify:
        metrics = dict(verify)
        metrics["ast_check_ok"] = ast_ok
        metrics = _attach_reward_fields(metrics)
    else:
        metrics = _attach_speedup(_verify_to_metrics(verify, op_name, ast_ok), perf_for_speedup)
    reward = reward_from_metrics(metrics)

    # ---- 5. persist latest metrics (best snapshots are written by run_verify) ----
    await _remote_write_json(env, metrics_file, metrics)

    # ---- 6. log and return ----
    current_speedup = _speedup_from_perf_data(
        metrics.get("perf_data") if isinstance(metrics.get("perf_data"), dict) else {}
    )
    print(
        f"[reward] op={op_name} best_first={int(best_first)} "
        f"source={source} speedup={current_speedup:.4f} "
        f"correctness_ok={metrics.get('correctness_ok')} "
        f"reward={reward:.4f}",
        flush=True,
    )

    eval_result: dict[str, Any] = {
        "eval_completed": True,
        "op_name": op_name,
        "reward": reward,
        "metrics": metrics,
        "used_best_metrics": best_first and source == "metrics_best",
        "selected_metrics_source": source,
        "resolved": _metrics_resolved(metrics),
    }
    if isinstance(best_metrics, dict):
        eval_result["best_metrics"] = best_metrics
    return reward, eval_result


def artifact_candidates(workspace_dir: str, op_name: str) -> list[tuple[str, str]]:
    """Text artifacts that are cheap to copy back from a sandbox."""
    return [
        (f"{workspace_dir}/.claude_code_done", "claude_code_done.txt"),
        (f"{workspace_dir}/.triton_verify_timing.jsonl", "verify_timing.jsonl"),
        (f"{workspace_dir}/.claude_code_run.sh", "claude_code_run.redacted.sh"),
        (f"{workspace_dir}/claude_code_trajectory.jsonl", "claude_code_trajectory.jsonl"),
        (f"{workspace_dir}/claude_code_stdout.log", "claude_code_stdout.log"),
        (f"{workspace_dir}/metrics.json", "metrics.json"),
        (f"{workspace_dir}/metrics_error.log", "metrics_error.log"),
        (f"{workspace_dir}/metrics_best.json", "metrics_best.json"),
        (f"{workspace_dir}/summary.json", "summary.json"),
        (f"{workspace_dir}/output/verify/verify_result.json", "verify_result.json"),
        (f"{workspace_dir}/output/verify/verify_result_summary.json", "verify_result_summary.json"),
        (f"{workspace_dir}/output/verify/verify_result.raw.log", "verify_result.raw.log"),
        (f"{workspace_dir}/output/verify/{op_name}_torch.json", f"verify_{op_name}_torch.json"),
        (f"{workspace_dir}/output/verify/{op_name}_torch.py", f"verify_{op_name}_torch.py"),
        (f"{workspace_dir}/output/perf_result.json", "perf_result.json"),
        (f"{workspace_dir}/output/perf_result.raw.log", "perf_result.raw.log"),
        (f"{workspace_dir}/output/perf_result_best.json", "perf_result_best.json"),
        (f"{workspace_dir}/output/op_summary.csv", "op_summary.csv"),
        (f"{workspace_dir}/profiling_results.json", "profiling_results.json"),
        (f"{workspace_dir}/INSTRUCTIONS.md", "INSTRUCTIONS.md"),
        (f"{workspace_dir}/src/{op_name}.py", f"{op_name}.py"),
        (f"{workspace_dir}/src/{op_name}.json", f"{op_name}.json"),
        (f"{workspace_dir}/src/{op_name}_triton_ascend_impl.py", f"{op_name}_triton_ascend_impl.py"),
        (f"{workspace_dir}/src/{op_name}_triton_ascend_impl_best.py", f"{op_name}_triton_ascend_impl_best.py"),
        (f"{workspace_dir}/{op_name}_generated.py", f"{op_name}_generated.py"),
    ]


def _redact_archived_artifact(name: str, text: str) -> str:
    if name != "claude_code_run.redacted.sh":
        return text
    sensitive_names = (
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_API_TOKEN",
    )
    for env_name in sensitive_names:
        text = re.sub(
            rf"(^|\s)(export\s+)?{re.escape(env_name)}=(['\"]?)[^'\"\s]*\3",
            rf"\1\2{env_name}=\3<redacted>\3",
            text,
            flags=re.MULTILINE,
        )
    return text


def _format_claude_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False, indent=2)

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        block_type = block.get("type")
        if block_type == "text":
            parts.append(str(block.get("text", "")))
        elif block_type == "thinking":
            parts.append(f"[thinking]\n{block.get('thinking', '')}")
        elif block_type == "tool_use":
            parts.append(
                "[tool_use] "
                f"{block.get('name', 'tool')} id={block.get('id', '')}\n"
                f"{json.dumps(block.get('input', {}), ensure_ascii=False, indent=2)}"
            )
        elif block_type == "tool_result":
            parts.append(
                "[tool_result] "
                f"id={block.get('tool_use_id', '')} is_error={block.get('is_error', False)}\n"
                f"{_format_claude_content(block.get('content', ''))}"
            )
        else:
            parts.append(json.dumps(block, ensure_ascii=False, indent=2))
    return "\n\n".join(part for part in parts if part)


def _format_claude_trajectory(jsonl_text: str) -> str:
    sections: list[str] = []
    for line_no, line in enumerate(jsonl_text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            sections.append(f"### RAW line {line_no}\n{line}")
            continue

        event_type = event.get("type", "event")
        subtype = event.get("subtype")
        if event_type in ("assistant", "user") and isinstance(event.get("message"), dict):
            message = event["message"]
            role = str(message.get("role") or event_type).upper()
            sections.append(f"### {role}\n{_format_claude_content(message.get('content', ''))}")
        elif event_type == "system":
            detail = {k: v for k, v in event.items() if k not in {"uuid", "session_id"}}
            sections.append(f"### SYSTEM/{subtype or 'event'}\n{json.dumps(detail, ensure_ascii=False, indent=2)}")
        elif event_type == "result":
            detail = {
                "is_error": event.get("is_error"),
                "api_error_status": event.get("api_error_status"),
                "duration_ms": event.get("duration_ms"),
                "num_turns": event.get("num_turns"),
                "result": event.get("result"),
                "terminal_reason": event.get("terminal_reason"),
            }
            sections.append(f"### RESULT\n{json.dumps(detail, ensure_ascii=False, indent=2)}")
        else:
            sections.append(
                f"### {str(event_type).upper()}{('/' + str(subtype)) if subtype else ''}\n"
                f"{json.dumps(event, ensure_ascii=False, indent=2)}"
            )
    return "\n\n".join(sections).rstrip() + "\n"


async def archive_text_artifacts(
    env,
    metadata: dict[str, Any],
    dest_root: str,
    *,
    workspace_dir: str,
    subdir: bool = True,
) -> str | None:
    """Copy selected text artifacts from the sandbox to ``dest_root``.

    ``subdir=True`` (default, shared by the trainer runners) writes into
    ``dest_root/<op>_<pid>[_<rollout_id>]/`` so multiple ops / rollouts do
    not collide. ``subdir=False`` writes directly into ``dest_root`` (for
    per-op dirs prepared by the eval launcher).
    """
    if not dest_root:
        return None
    op_name = metadata.get("op_name", "operator")
    rollout_id = str(metadata.get("rollout_run_id") or "").strip()
    safe_op_name = str(op_name).replace("/", "_")
    dest_name = f"{safe_op_name}_{os.getpid()}"
    if rollout_id:
        safe_rollout_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in rollout_id)
        dest_name = f"{dest_name}_{safe_rollout_id}"
    dest = Path(dest_root).expanduser()
    if subdir:
        dest = dest / dest_name
    copied = 0
    copied_texts: dict[str, str] = {}
    for remote_path, name in artifact_candidates(workspace_dir, op_name):
        try:
            if not await _remote_file_exists(env, remote_path):
                continue
            dest.mkdir(parents=True, exist_ok=True)
            text = await env.read_file(remote_path)
            text = _redact_archived_artifact(name, text)
            (dest / name).write_text(text, encoding="utf-8", errors="replace")
            copied_texts[name] = text
            copied += 1
        except Exception as exc:
            logger.warning("Failed to archive %s: %s", remote_path, exc)
    if "claude_code_trajectory.jsonl" in copied_texts:
        try:
            (dest / "conversation.log").write_text(
                _format_claude_trajectory(copied_texts["claude_code_trajectory.jsonl"]),
                encoding="utf-8",
                errors="replace",
            )
            copied += 1
        except Exception as exc:
            logger.warning("Failed to render Claude Code conversation log: %s", exc)
    return str(dest) if copied else None
