#!/usr/bin/env python3
"""Save the best verified implementation snapshot after each verifier/benchmark run.

``metrics_best.json`` tracks the highest-ranked impl (all-pass first, then
pass_rate, then that same impl's ``speedup_vs_torch``). Speedup is stored on
the metrics object itself so it cannot drift from ``perf_result_best.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

# Reward formula / metric helpers are shared with the trainer-side reward.py via
# reward_formula.py (single source of truth). This file lives next to this
# script under tools/ and is deployed into the sandbox alongside it.
from reward_formula import (
    attach_reward_fields,
    has_verify_signal,
    rank,
    safe_int,
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _assistant_messages_seen(workspace: Path) -> int:
    """Best-effort map from a verifier snapshot to a gateway trajectory index.

    Claude Code writes one ``type=assistant`` JSONL record per model exchange.
    The verifier runs inside the tool call emitted by the producing assistant
    exchange, so the best-producing gateway trajectory is normally the last
    assistant record seen when this snapshot helper runs.
    """
    path = workspace / "claude_code_trajectory.jsonl"
    assistant_ids: set[str] = set()
    anonymous_assistant_messages = 0
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not (isinstance(event, dict) and event.get("type") == "assistant"):
                    continue
                message = event.get("message")
                message_id = message.get("id") if isinstance(message, dict) else None
                if message_id:
                    assistant_ids.add(str(message_id))
                else:
                    anonymous_assistant_messages += 1
    except OSError:
        return 0
    return len(assistant_ids) + anonymous_assistant_messages


def _metrics_from_verify(
    *,
    op_name: str,
    verify: dict[str, Any] | None,
    summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(verify, dict) and not isinstance(summary, dict):
        return None

    total = safe_int(
        verify.get("total_cases") if isinstance(verify, dict) else None,
        safe_int(summary.get("total_cases") if isinstance(summary, dict) else None, 0),
    )
    passed = safe_int(
        verify.get("passed_cases") if isinstance(verify, dict) else None,
        safe_int(summary.get("passed_cases") if isinstance(summary, dict) else None, 0),
    )
    failed = safe_int(
        verify.get("failed_cases") if isinstance(verify, dict) else None,
        safe_int(summary.get("failed_cases") if isinstance(summary, dict) else None, max(total - passed, 0)),
    )
    correctness_ok = total > 0 and passed == total and failed == 0
    compile_ok = bool(
        correctness_ok
        or passed > 0
        or (isinstance(summary, dict) and summary.get("compile_ok"))
        or (isinstance(summary, dict) and summary.get("output_observed"))
    )

    metrics: dict[str, Any] = {
        "op_name": op_name,
        "success": correctness_ok,
        "ast_check_ok": True,
        "compile_ok": compile_ok,
        "correctness_ok": correctness_ok,
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": failed,
        "pass_rate": round(passed / max(total, 1), 6) if total > 0 else 0.0,
        "perf_data": {},
        "perf_missing": True,
    }

    if not correctness_ok:
        error_groups = summary.get("error_groups") if isinstance(summary, dict) else None
        if compile_ok:
            metrics["error_type"] = "correctness_failed"
        else:
            metrics["error_type"] = "compilation_failed" if total > 0 else "missing_verify_result"
        metrics["error"] = json.dumps(error_groups, ensure_ascii=False) if error_groups else metrics["error_type"]
    return attach_reward_fields(metrics)


def _attach_matching_perf(
    metrics: dict[str, Any],
    perf: dict[str, Any] | None,
    op_name: str,
) -> bool:
    """Bind bench JSON only when it belongs to this all-pass impl.

    Benchmark is gated on ``passed_cases == total_cases``. Attaching leftover
    ``perf_result.json`` from an earlier impl would mix another kernel's
    ``speedup_vs_torch`` into ``metrics_best``.
    """
    metrics["perf_data"] = {}
    metrics["perf_missing"] = True
    if not metrics.get("correctness_ok"):
        return False
    if not isinstance(perf, dict):
        return False
    file_op = perf.get("op_name")
    if file_op and str(file_op) != op_name:
        return False
    metrics["perf_data"] = perf
    metrics["perf_missing"] = False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--workspace-dir", default=".")
    parser.add_argument("--verify-dir", default="output/verify")
    parser.add_argument("--verify-result", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--triton-impl-name", default="triton_ascend_impl")
    args = parser.parse_args()

    workspace = Path(args.workspace_dir).resolve()
    verify_dir = (workspace / args.verify_dir).resolve() if not Path(args.verify_dir).is_absolute() else Path(args.verify_dir)
    verify_path = (workspace / args.verify_result).resolve() if not Path(args.verify_result).is_absolute() else Path(args.verify_result)
    summary_path = None
    if args.summary:
        summary_path = (workspace / args.summary).resolve() if not Path(args.summary).is_absolute() else Path(args.summary)
    verify = _read_json(verify_path)
    summary = _read_json(summary_path) if summary_path else None
    metrics = _metrics_from_verify(op_name=args.op_name, verify=verify, summary=summary)
    if not metrics or not has_verify_signal(metrics):
        print("[best-snapshot] skipped: no verifier metrics found.")
        return 0

    perf_path = workspace / "output" / "perf_result.json"
    perf = _read_json(perf_path)
    # Bind bench output written at/after this verify, not a leftover file.
    perf_is_current = (
        perf_path.is_file()
        and verify_path.is_file()
        and perf_path.stat().st_mtime >= verify_path.stat().st_mtime
    )
    attached_matching_perf = _attach_matching_perf(
        metrics, perf if perf_is_current else None, args.op_name,
    )
    metrics = attach_reward_fields(metrics)

    assistant_messages = _assistant_messages_seen(workspace)
    metrics["assistant_messages_seen"] = assistant_messages
    if assistant_messages > 0:
        metrics["assistant_index"] = assistant_messages - 1

    suffix = args.triton_impl_name
    staged_impl = verify_dir / f"{args.op_name}_{suffix}.py"
    if not staged_impl.exists():
        print(f"[best-snapshot] skipped: staged implementation missing: {staged_impl}")
        return 0

    best_metrics_path = workspace / "metrics_best.json"
    best_impl_path = workspace / "src" / f"{args.op_name}_triton_ascend_impl_best.py"
    old_best = _read_json(best_metrics_path)
    if old_best and has_verify_signal(old_best) and rank(old_best) > rank(metrics):
        print(
            "[best-snapshot] kept existing best: "
            f"old_rank={rank(old_best)} new_rank={rank(metrics)}"
        )
        return 0

    best_impl_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(staged_impl, best_impl_path)
    best_metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    best_perf_path = workspace / "output" / "perf_result_best.json"
    if attached_matching_perf and perf_path.is_file():
        best_perf_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(perf_path, best_perf_path)
    elif best_perf_path.is_file() and not metrics.get("correctness_ok"):
        best_perf_path.unlink()
    print(
        "[best-snapshot] updated: "
        f"passed_cases={metrics.get('passed_cases')}, total_cases={metrics.get('total_cases')}, "
        f"compile_ok={metrics.get('compile_ok')}, correctness_ok={metrics.get('correctness_ok')}, "
        f"speedup_vs_torch={metrics.get('speedup_vs_torch')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
