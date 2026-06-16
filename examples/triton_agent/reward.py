"""Reward helpers for Triton/Ascend KernelBench agent rollouts."""

from __future__ import annotations

import json
import logging
import os
import shlex
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE_DIR = "/opt/workspace/agent_workdir"


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False", "no", "")


def _float_env(name: str, default: float, *fallback_names: str) -> float:
    for key in (name, *fallback_names):
        value = os.environ.get(key)
        if value not in (None, ""):
            return float(value)
    return default


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


def _pass_rate(perf: dict[str, Any]) -> float:
    total = _safe_int(perf.get("total_cases"), 0)
    passed = _safe_int(perf.get("passed_cases"), 0)
    if total <= 0:
        return 1.0 if bool(perf.get("correctness_ok") or perf.get("success")) else 0.0
    return max(0.0, min(passed / max(total, 1), 1.0))


def _speedup_from_perf_data(perf_data: dict[str, Any]) -> float:
    for key in ("speedup_vs_torch", "speedup", "geomean_speedup"):
        if key in perf_data:
            return max(0.0, _safe_float(perf_data.get(key), 0.0))
    return 0.0


def reward_breakdown_from_metrics(perf: dict[str, Any]) -> dict[str, Any]:
    """Stable partial-credit reward components from ``metrics.json``."""
    ast_reward = _float_env("TRITON_REWARD_AST_OK", 0.0)
    compile_reward = _float_env("TRITON_REWARD_COMPILE_OK", 0.10)
    correctness_reward = _float_env(
        "TRITON_REWARD_CORRECTNESS_OK",
        0.55,
    )
    all_correct_bonus = _float_env(
        "TRITON_REWARD_ALL_CORRECT_BONUS",
        0.10,
    )
    speedup_reward = _float_env("TRITON_REWARD_SPEEDUP_MAX", 0.40)

    ast_ok = bool(perf.get("ast_check_ok", False))
    compile_ok = bool(perf.get("compile_ok") or perf.get("output_observed") or perf.get("correctness_ok"))
    pass_rate = _pass_rate(perf)
    success = bool(perf.get("success", False))
    corr_ok = bool(perf.get("correctness_ok", False))

    ast_score = ast_reward if ast_ok else 0.0
    compile_score = compile_reward if compile_ok else 0.0
    correctness_score = correctness_reward * pass_rate
    all_correct_score = all_correct_bonus if corr_ok else 0.0

    speedup = 0.0
    speedup_score = 0.0
    perf_data = perf.get("perf_data") or {}
    if isinstance(perf_data, dict):
        speedup = _speedup_from_perf_data(perf_data)
    if speedup > 0.0 and (success or corr_ok or pass_rate > 0.0):
        target = max(_float_env("TRITON_REWARD_TARGET_SPEEDUP", 2.0), 1e-6)
        speedup_score = speedup_reward * max(0.0, min(speedup / target, 1.0)) * max(pass_rate, 0.0)

    total = round(
        max(0.0, min(ast_score + compile_score + correctness_score + all_correct_score + speedup_score, 1.0)),
        4,
    )
    return {
        "ast": round(ast_score, 4),
        "compile": round(compile_score, 4),
        "correctness": round(correctness_score, 4),
        "all_correct_bonus": round(all_correct_score, 4),
        "speedup": round(speedup_score, 4),
        "pass_rate": round(pass_rate, 6),
        "raw_speedup": round(speedup, 6),
        "total": total,
    }


def reward_from_metrics(perf: dict[str, Any]) -> float:
    return float(reward_breakdown_from_metrics(perf)["total"])


def _attach_reward_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    """Add flat reward fields without wrapping the verifier metrics payload."""
    components = reward_breakdown_from_metrics(metrics)
    for key in ("reward_components", "benchmark_ok", "output_observed"):
        metrics.pop(key, None)
    metrics["reward"] = components["total"]
    metrics["reward_ast"] = components["ast"]
    metrics["reward_compile"] = components["compile"]
    metrics["reward_correctness"] = components["correctness"]
    metrics["reward_all_correct_bonus"] = components["all_correct_bonus"]
    metrics["reward_speedup"] = components["speedup"]
    metrics["raw_speedup"] = components["raw_speedup"]
    return metrics


def _metrics_resolved(perf: dict[str, Any] | None) -> bool:
    if not isinstance(perf, dict):
        return False
    return bool(perf.get("success", False) and perf.get("correctness_ok", False))


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
    return {"score": score}


async def _remote_file_exists(env, path: str) -> bool:
    quoted = shlex.quote(path)
    output = await env.communicate(f"if [ -f {quoted} ]; then echo yes; else echo no; fi", check="ignore")
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
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    code = "import pathlib, sys; pathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')"
    await env.communicate(
        f"printf %s {shlex.quote(text)} | python3 -c {shlex.quote(code)} {shlex.quote(path)}",
        check="ignore",
    )


async def _remote_latest_json(
    env,
    workspace_dir: str,
    patterns: list[str],
) -> tuple[str, dict[str, Any], float] | None:
    code = r"""
import glob
import json
import os
import sys

root = sys.argv[1]
rows = []
for pattern in sys.argv[2:]:
    for path in glob.glob(os.path.join(root, pattern), recursive=True):
        if not os.path.isfile(path):
            continue
        try:
            rows.append({"path": path, "mtime": os.path.getmtime(path)})
        except OSError:
            pass
rows.sort(key=lambda row: (row["mtime"], row["path"]))
print(json.dumps(rows[-1] if rows else {}, ensure_ascii=False))
"""
    cmd = f"python3 -c {shlex.quote(code)} {shlex.quote(workspace_dir)}"
    for pattern in patterns:
        cmd += f" {shlex.quote(pattern)}"
    output = await env.communicate(cmd, check="ignore")
    line = output.strip().splitlines()[-1] if output.strip() else "{}"
    try:
        found = json.loads(line)
    except json.JSONDecodeError:
        return None
    path = found.get("path")
    if not path:
        return None
    data = await _remote_json(env, path)
    if data is None:
        return None
    return str(path), data, float(found.get("mtime") or 0.0)


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


def _int_field(data: dict[str, Any] | None, name: str, default: int = 0) -> int:
    if not isinstance(data, dict):
        return default
    try:
        return int(data.get(name, default) or default)
    except (TypeError, ValueError):
        return default


def _failure_message(failure: Any) -> str:
    if isinstance(failure, dict):
        return str(failure.get("error_msg") or failure.get("error") or failure)
    return str(failure)


def _failure_error_type(failure: Any) -> str:
    if isinstance(failure, dict):
        return str(failure.get("error_type") or "")
    return ""


def _failure_observed_output(failure: Any) -> bool:
    text = _failure_message(failure)
    markers = (
        "输出不一致",
        "输出形状不一致",
        "NaN 位置不匹配",
        "Inf 位置",
        "布尔值不匹配",
        "output shape",
        "MERE=",
        "MARE=",
        "compare(fw_out, impl_out",
    )
    return any(marker in text for marker in markers)


def _compact_reason(text: str) -> str:
    for marker in (
        "coreDim is invalid",
        "Unsupported ptr type",
        "输出形状不一致",
        "输出不一致",
        "NaN 位置不匹配",
        "Inf 位置",
        "CompilationError",
        "RuntimeError",
        "AssertionError",
    ):
        if marker in text:
            idx = text.find(marker)
            line = text[idx:].splitlines()[0]
            return line[:500]
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped[:500]
    return ""


def _compile_status_from_verify(verify: dict[str, Any] | None, perf: dict[str, Any] | None) -> tuple[bool, bool]:
    if isinstance(perf, dict) and perf:
        return True, True
    if not isinstance(verify, dict):
        return False, False
    if bool(verify.get("compile_ok") or verify.get("output_observed")):
        return True, True
    passed = _int_field(verify, "passed_cases", 0)
    if passed > 0:
        return True, True
    failures = verify.get("failures")
    if isinstance(failures, list) and any(_failure_observed_output(item) for item in failures):
        return True, True
    return False, False


def _failure_text(verify: dict[str, Any] | None, summary: dict[str, Any] | None) -> str:
    failures = verify.get("failures") if isinstance(verify, dict) else None
    if isinstance(failures, list) and failures:
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        for item in failures:
            key = (_failure_error_type(item), _compact_reason(_failure_message(item)))
            group = groups.setdefault(
                key,
                {
                    "error_type": key[0],
                    "reason": key[1],
                    "count": 0,
                    "example_case": item.get("case_idx") if isinstance(item, dict) else None,
                    "example_input": item.get("input_desc") if isinstance(item, dict) else None,
                },
            )
            group["count"] += 1
        return json.dumps(list(groups.values())[:5], ensure_ascii=False)
    if failures:
        return json.dumps(failures, ensure_ascii=False)
    for container in (verify, summary):
        error_groups = container.get("error_groups") if isinstance(container, dict) else None
        if isinstance(error_groups, list) and error_groups:
            compact_groups = []
            for group in error_groups[:5]:
                if not isinstance(group, dict):
                    continue
                compact_groups.append(
                    {
                        "error_type": group.get("error_type") or "",
                        "reason": group.get("reason") or "",
                        "count": group.get("count") or 0,
                        "examples": group.get("examples") or [],
                    }
                )
            if compact_groups:
                return json.dumps(compact_groups, ensure_ascii=False)
    if isinstance(summary, dict):
        for key in ("failure_reason", "last_error", "error", "reason"):
            if summary.get(key):
                return str(summary[key])
    if isinstance(verify, dict):
        for key in ("failure_reason", "last_error", "error", "reason"):
            if verify.get(key):
                return str(verify[key])
    return ""


def _metrics_from_upstream_artifacts(
    *,
    op_name: str,
    summary: dict[str, Any] | None,
    verify: dict[str, Any] | None,
    perf: dict[str, Any] | None,
    artifact_paths: dict[str, str],
    ast_check: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not any(isinstance(item, dict) for item in (summary, verify, perf)):
        return None

    summary_perf = summary.get("perf_data") if isinstance(summary, dict) else None
    perf_data = perf if isinstance(perf, dict) else summary_perf if isinstance(summary_perf, dict) else {}

    total_cases = _int_field(verify, "total_cases", _int_field(summary_perf, "total_cases", 0))
    passed_cases = _int_field(verify, "passed_cases", _int_field(summary_perf, "passed_cases", 0))
    failed_cases = _int_field(
        verify,
        "failed_cases",
        max(total_cases - passed_cases, _int_field(summary_perf, "failed_cases", 0)),
    )

    correctness_ok = total_cases > 0 and passed_cases == total_cases
    compile_ok, _output_observed = _compile_status_from_verify(verify, perf)
    ast_ok = bool(ast_check.get("valid")) if isinstance(ast_check, dict) else bool(summary or verify or perf)
    benchmark_ok = bool(perf_data) and not bool(perf_data.get("failed_cases")) if isinstance(perf_data, dict) else False
    require_perf = _truthy_env("TRITON_REQUIRE_PERF_RESULT", "0")
    if isinstance(summary, dict) and "success" in summary:
        success = bool(summary.get("success")) and correctness_ok
        if correctness_ok and not perf_data and not require_perf:
            success = True
    else:
        success = correctness_ok and (benchmark_ok or not require_perf)

    implementation = perf_data.get("implementation") if isinstance(perf_data, dict) else None
    latency = implementation.get("avg_latency_ms") if isinstance(implementation, dict) else None
    if latency is None and isinstance(perf_data, dict):
        latency = perf_data.get("avg_latency_ms")

    metrics: dict[str, Any] = {
        "success": success,
        "ast_check_ok": ast_ok,
        "compile_ok": compile_ok,
        "correctness_ok": correctness_ok,
        "op_name": op_name,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "pass_rate": round(passed_cases / max(total_cases, 1), 6) if total_cases > 0 else 0.0,
        "perf_missing": bool(correctness_ok and not perf_data),
        "perf_data": perf_data if isinstance(perf_data, dict) else {},
        "latency": latency,
    }
    if not correctness_ok:
        if ast_ok and not compile_ok:
            metrics["error_type"] = "compilation_failed"
        else:
            metrics["error_type"] = "correctness_failed" if verify else "missing_verify_result"
        metrics["error"] = _failure_text(verify, summary) or metrics["error_type"]
    elif not perf_data and require_perf:
        metrics["error"] = "Correctness passed, but no upstream perf_result.json or summary perf_data was found."
        metrics["error_type"] = "missing_perf_result"
    elif not perf_data:
        metrics["warning"] = "Correctness passed; no upstream perf_result.json or summary perf_data was found, so no speedup reward is available."
    return _attach_reward_fields(metrics)


async def _metrics_from_upstream_workspace(
    env,
    workspace_dir: str,
    op_name: str,
    ast_check: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, float | None]:
    summary_item = await _remote_latest_json(env, workspace_dir, ["summary.json", "**/summary.json"])
    verify_item = await _remote_latest_json(
        env,
        workspace_dir,
        ["verify_result.json", "**/verify_result.json", "**/verify_result_*.json"],
    )
    perf_item = await _remote_latest_json(
        env,
        workspace_dir,
        ["perf_result.json", "**/perf_result.json", "**/*perf_result*.json"],
    )

    paths: dict[str, str] = {}
    latest_mtime: float | None = None
    for key, item in (("summary", summary_item), ("verify", verify_item), ("perf", perf_item)):
        if item is None:
            continue
        path, _data, mtime = item
        paths[key] = path
        latest_mtime = mtime if latest_mtime is None else max(latest_mtime, mtime)

    metrics = _metrics_from_upstream_artifacts(
        op_name=op_name,
        summary=summary_item[1] if summary_item else None,
        verify=verify_item[1] if verify_item else None,
        perf=perf_item[1] if perf_item else None,
        artifact_paths=paths,
        ast_check=ast_check,
    )
    return metrics, latest_mtime


async def _remote_ast_check(env, workspace_dir: str, op_name: str) -> dict[str, Any] | None:
    impl_file = f"{workspace_dir}/src/{op_name}_triton_ascend_impl.py"
    checker = f"{workspace_dir}/.claude/skills/triton-op-verifier/scripts/validate_triton_impl.py"
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


async def _snapshot_success_as_best(env, workspace_dir: str, op_name: str, perf: dict[str, Any]) -> None:
    if not _metrics_resolved(perf):
        return

    impl_file = f"{workspace_dir}/src/{op_name}_triton_ascend_impl.py"
    metrics_file = f"{workspace_dir}/metrics.json"
    best_impl = f"{workspace_dir}/src/{op_name}_triton_ascend_impl_best.py"
    best_metrics = f"{workspace_dir}/metrics_best.json"

    impl_mtime = await _remote_mtime(env, impl_file)
    metrics_mtime = await _remote_mtime(env, metrics_file)
    if impl_mtime is not None and metrics_mtime is not None and impl_mtime > metrics_mtime + 1e-6:
        return

    current_reward = reward_from_metrics(perf)
    best = await _remote_json(env, best_metrics)
    best_reward = reward_from_metrics(best) if _metrics_resolved(best) else -1.0
    if current_reward >= best_reward:
        await env.communicate(
            f"cp -f {shlex.quote(impl_file)} {shlex.quote(best_impl)} && "
            f"cp -f {shlex.quote(metrics_file)} {shlex.quote(best_metrics)}",
            check="ignore",
        )


async def evaluate_triton_workspace(
    env,
    metadata: dict[str, Any],
    *,
    workspace_dir: str = DEFAULT_WORKSPACE_DIR,
) -> tuple[float, dict[str, Any]]:
    """Evaluate a Triton workspace already modified by an agent."""
    op_name = metadata.get("op_name", "operator")
    impl_file = f"{workspace_dir}/src/{op_name}_triton_ascend_impl.py"
    metrics_file = f"{workspace_dir}/metrics.json"
    best_metrics_file = f"{workspace_dir}/metrics_best.json"

    result: dict[str, Any] = {
        "eval_completed": True,
        "op_name": op_name,
        "workspace_dir": workspace_dir,
        "reward": 0.0,
    }

    async def finish_failure(reason: str, reward: float | None = None, **extra: Any) -> tuple[float, dict[str, Any]]:
        metrics: dict[str, Any] = {
            "success": False,
            "ast_check_ok": False,
            "compile_ok": False,
            "correctness_ok": False,
            "op_name": op_name,
            "error_type": reason,
            "error": reason,
        }
        metrics.update(extra)
        _attach_reward_fields(metrics)
        final_reward = metrics["reward"] if reward is None else reward
        metrics["reward"] = final_reward
        await _remote_write_json(env, metrics_file, metrics)
        result["reason"] = reason
        result["metrics"] = metrics
        result["reward"] = final_reward
        return final_reward, result

    if not await _remote_file_exists(env, impl_file):
        return await finish_failure("missing_impl", 0.0)

    impl_mtime = await _remote_mtime(env, impl_file)
    ast_check = await _remote_ast_check(env, workspace_dir, str(op_name))
    ast_ok = bool(ast_check.get("valid")) if isinstance(ast_check, dict) else False
    if isinstance(ast_check, dict) and not ast_ok:
        return await finish_failure(
            "ast_check_failed",
            ast_check_ok=False,
            error=ast_check.get("suggestion") or ast_check.get("raw_output") or "ast_check_failed",
        )

    perf = await _remote_json(env, metrics_file)
    if not perf:
        perf, upstream_mtime = await _metrics_from_upstream_workspace(env, workspace_dir, str(op_name), ast_check)
        if not perf:
            return await finish_failure("missing_metrics", ast_check_ok=ast_ok)
        if impl_mtime is not None and upstream_mtime is not None and impl_mtime > upstream_mtime + 1e-6:
            if isinstance(perf, dict):
                await _remote_write_json(env, metrics_file, perf)
            result["reason"] = "stale_upstream_artifacts"
            result["metrics"] = perf
            result["reward"] = 0.05
            result["impl_mtime"] = impl_mtime
            result["upstream_mtime"] = upstream_mtime
            return 0.05, result
        await _remote_write_json(env, metrics_file, perf)
    elif isinstance(perf, dict):
        if ast_ok:
            perf["ast_check_ok"] = True
        else:
            perf.setdefault("ast_check_ok", bool(perf.get("ast_check_ok")))
        compile_ok = bool(
            perf.get("compile_ok")
            or perf.get("output_observed")
            or perf.get("correctness_ok")
            or _safe_int(perf.get("passed_cases"), 0) > 0
            or isinstance(perf.get("perf_data"), dict) and bool(perf.get("perf_data"))
        )
        perf.setdefault("compile_ok", compile_ok)
        perf.setdefault("pass_rate", _pass_rate(perf))
        if not perf.get("correctness_ok") and not perf.get("error"):
            perf["error_type"] = perf.get("error_type") or "correctness_failed"
            perf["error"] = perf["error_type"]
        _attach_reward_fields(perf)

    metrics_mtime = await _remote_mtime(env, metrics_file)
    if impl_mtime is not None and metrics_mtime is not None and impl_mtime > metrics_mtime + 1e-6:
        result["reason"] = "stale_metrics"
        result["metrics"] = perf
        result["reward"] = 0.05
        result["impl_mtime"] = impl_mtime
        result["metrics_mtime"] = metrics_mtime
        return 0.05, result

    if isinstance(perf, dict):
        await _remote_write_json(env, metrics_file, perf)

    await _snapshot_success_as_best(env, workspace_dir, op_name, perf)
    reward = reward_from_metrics(perf)

    best = await _remote_json(env, best_metrics_file)
    best_resolved = False
    if best:
        best_resolved = _metrics_resolved(best)
        if best_resolved:
            best_reward = reward_from_metrics(best)
            if best_reward > reward:
                reward = best_reward
                result["used_best_metrics"] = True
                result["best_metrics"] = best
        else:
            result["ignored_best_metrics"] = "not_successful"

    result["metrics"] = perf
    result["reward"] = reward
    result["resolved"] = bool(_metrics_resolved(perf) or best_resolved)
    return reward, result


def artifact_candidates(workspace_dir: str, op_name: str) -> list[tuple[str, str]]:
    """Text artifacts that are cheap to copy back from a sandbox."""
    return [
        (f"{workspace_dir}/claude_code_trajectory.jsonl", "claude_code_trajectory.jsonl"),
        (f"{workspace_dir}/claude_code_stdout.log", "claude_code_stdout.log"),
        (f"{workspace_dir}/metrics.json", "metrics.json"),
        (f"{workspace_dir}/metrics_error.log", "metrics_error.log"),
        (f"{workspace_dir}/metrics_best.json", "metrics_best.json"),
        (f"{workspace_dir}/summary.json", "summary.json"),
        (f"{workspace_dir}/output/verify/verify_result_summary.json", "verify_result_summary.json"),
        (f"{workspace_dir}/output/verify/verify_result.raw.log", "verify_result.raw.log"),
        (f"{workspace_dir}/output/perf_result.json", "perf_result.json"),
        (f"{workspace_dir}/profiling_results.json", "profiling_results.json"),
        (f"{workspace_dir}/INSTRUCTIONS.md", "INSTRUCTIONS.md"),
        (f"{workspace_dir}/src/{op_name}_triton_ascend_impl.py", f"{op_name}_triton_ascend_impl.py"),
        (f"{workspace_dir}/src/{op_name}_triton_ascend_impl_best.py", f"{op_name}_triton_ascend_impl_best.py"),
        (f"{workspace_dir}/{op_name}_generated.py", f"{op_name}_generated.py"),
    ]


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


async def archive_text_artifacts(env, metadata: dict[str, Any], dest_root: str, *, workspace_dir: str) -> str | None:
    """Copy selected text artifacts from the sandbox to ``dest_root``."""
    if not dest_root:
        return None
    op_name = metadata.get("op_name", "operator")
    dest = Path(dest_root).expanduser() / f"{op_name}_{os.getpid()}"
    copied = 0
    copied_texts: dict[str, str] = {}
    for remote_path, name in artifact_candidates(workspace_dir, op_name):
        if not await _remote_file_exists(env, remote_path):
            continue
        try:
            dest.mkdir(parents=True, exist_ok=True)
            text = await env.read_file(remote_path)
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
