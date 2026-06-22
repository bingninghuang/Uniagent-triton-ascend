#!/usr/bin/env python3
"""Show non-view actions from Triton synthesis JSONL results.

This script filters out common read-only exploration calls (str_replace_editor
view, cat/head/tail/sed/grep/ls/pwd/find) and highlights actions that actually
change files, run validation, benchmark, or submit.

Usage:
    python examples/triton_agent/analyze_synth_actions.py /path/to/results.jsonl
    python examples/triton_agent/analyze_synth_actions.py /path/to/results.jsonl --show-read --only-failed
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


READONLY_SHELL_PREFIXES = (
    "pwd",
    "ls",
    "cat",
    "head",
    "tail",
    "sed",
    "grep",
    "find",
    "wc",
    "tree",
    "python - <<'PY'",  # often used for inspection; can be reclassified below by content
    "python3 - <<'PY'",
)

WRITE_KEYWORDS = (
    "create",
    "str_replace",
    "insert",
    "undo_edit",
    "cp ",
    "mkdir ",
    "mv ",
    "rm ",
    "tee ",
    ">",
    "cat >",
)

VALIDATION_KEYWORDS = (
    "validate_triton_impl.py",
    "verify.py",
    "benchmark.py",
    "run_npu_command.sh",
    "passed_cases",
    "total_cases",
)

IMPLEMENTATION_HINTS = (
    "_triton_ascend_impl.py",
    "ModelNew",
    "@triton.jit",
    "triton",
)

BAD_STATUS = {"timeout", "syntax_error", "skipped"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] invalid JSON line {line_no}: {exc}")
    return rows


def shorten(text: Any, limit: int) -> str:
    if text is None:
        return ""
    text = str(text)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n... <truncated {len(text) - limit} chars>"


def one_line(text: Any, limit: int) -> str:
    return shorten(" ".join(str(text or "").split()), limit)


def iter_tools(result: dict[str, Any]):
    trajectory = result.get("trajectory") or []
    if isinstance(trajectory, str):
        # Older result files may contain Pydantic StepOutput objects stringified by
        # _json_safe(), e.g. "step_idx=1 response='' ...". They do not preserve
        # structured tool_results, so skip instead of crashing.
        return
    for step_idx, step in enumerate(trajectory, 1):
        if not isinstance(step, dict):
            continue
        tool_results = step.get("tool_results") or []
        if isinstance(tool_results, str):
            continue
        for tool_idx, tool in enumerate(tool_results, 1):
            if isinstance(tool, dict):
                yield step_idx, tool_idx, step, tool


def is_str_replace_view(tool: dict[str, Any]) -> bool:
    if tool.get("name") != "str_replace_editor":
        return False
    action = tool.get("action") or ""
    # Tool action is usually shell-form, e.g. str_replace_editor --command view --path ...
    return "--command view" in action or re.search(r"\bview\b", action) is not None and "--path" in action


def is_readonly_shell(action: str) -> bool:
    stripped = action.strip()
    if not stripped:
        return True
    first_line = stripped.splitlines()[0].strip()

    # Command chains that begin with read-only checks.
    for prefix in READONLY_SHELL_PREFIXES:
        if first_line == prefix or first_line.startswith(prefix + " "):
            # Reclassify python snippets as non-read if they contain writes or validation.
            if prefix.startswith("python"):
                body = stripped.lower()
                if any(k.lower() in body for k in WRITE_KEYWORDS + VALIDATION_KEYWORDS + IMPLEMENTATION_HINTS):
                    return False
            return True
    return False


def classify(tool: dict[str, Any]) -> str:
    name = tool.get("name") or ""
    action = tool.get("action") or ""
    obs = tool.get("observation") or ""
    blob = f"{action}\n{obs}"

    if name == "submit" or "<<<Finished>>>" in action:
        return "submit"
    if any(k in blob for k in VALIDATION_KEYWORDS):
        return "validation"
    if any(k in blob for k in IMPLEMENTATION_HINTS) and any(k in action for k in WRITE_KEYWORDS):
        return "implementation-write"
    if any(k in action for k in WRITE_KEYWORDS):
        return "file-or-shell-write"
    if name == "execute_bash" and not is_readonly_shell(action):
        return "execute-other"
    if is_str_replace_view(tool) or (name == "execute_bash" and is_readonly_shell(action)):
        return "read-only"
    return "other"


def status_signal(tool: dict[str, Any]) -> str:
    status = tool.get("status", "<missing>")
    obs = str(tool.get("observation") or "")
    if status in BAD_STATUS:
        return status
    lower = obs.lower()
    if "passed_cases" in lower or "total_cases" in lower:
        return "verify-output"
    if "traceback" in lower or "error" in lower or "failed" in lower:
        return "suspicious-output"
    return status


def print_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    kind_counts = Counter()
    status_counts = Counter()
    op_with_impl_write = set()
    op_with_validation = set()
    op_with_submit = set()

    for result in rows:
        op = result.get("op_name", "<unknown>")
        for _, _, _, tool in iter_tools(result):
            kind = classify(tool)
            kind_counts[kind] += 1
            status_counts[(kind, status_signal(tool))] += 1
            if kind == "implementation-write":
                op_with_impl_write.add(op)
            elif kind == "validation":
                op_with_validation.add(op)
            elif kind == "submit":
                op_with_submit.add(op)

    print("=" * 100)
    print("ACTION SUMMARY")
    print("=" * 100)
    print(f"tasks: {len(rows)}")
    print("\naction kinds:")
    for kind, count in kind_counts.most_common():
        print(f"  {kind:24s} {count}")

    print("\nstatus by kind:")
    for (kind, signal), count in sorted(status_counts.items()):
        print(f"  {kind:24s} {signal:20s} {count}")

    print("\ncoverage:")
    print(f"  tasks with implementation write: {len(op_with_impl_write)}/{len(rows)}")
    print(f"  tasks with validation run:       {len(op_with_validation)}/{len(rows)}")
    print(f"  tasks with submit:               {len(op_with_submit)}/{len(rows)}")


def should_print_task(result: dict[str, Any], args: argparse.Namespace) -> bool:
    if not args.only_failed:
        return True
    reward = float(result.get("reward_score") or 0.0)
    exit_reason = result.get("exit_reason")
    return reward <= args.reward_threshold or exit_reason not in {"finished", "completed"}


def print_actions(result: dict[str, Any], args: argparse.Namespace) -> None:
    if not should_print_task(result, args):
        return

    op = result.get("op_name", "<unknown>")
    reward = result.get("reward_score", 0.0)
    pass_rate = result.get("pass_rate", 0.0)
    exit_reason = result.get("exit_reason")
    turns = result.get("num_turns")

    rows = []
    for step_idx, tool_idx, _, tool in iter_tools(result):
        kind = classify(tool)
        if kind == "read-only" and not args.show_read:
            continue
        if args.kind and kind != args.kind:
            continue
        rows.append((step_idx, tool_idx, kind, tool))

    if not rows and not args.show_empty:
        return

    print("\n" + "=" * 100)
    print(f"TASK: {op}")
    print("=" * 100)
    print(f"exit={exit_reason} | reward={reward} | pass_rate={pass_rate} | turns={turns}")

    if not rows:
        print("\n(no matching actions)")
        return

    for step_idx, tool_idx, kind, tool in rows:
        name = tool.get("name", "<missing>")
        status = tool.get("status", "<missing>")
        signal = status_signal(tool)
        execution_time = tool.get("execution_time")
        action = tool.get("action") or ""
        observation = tool.get("observation") or ""
        marker = " !!" if status in BAD_STATUS or signal == "suspicious-output" else ""

        print(f"\nStep {step_idx} Tool {tool_idx} | kind={kind} | name={name} | status={status} | signal={signal} | time={execution_time}{marker}")
        print("Action:")
        print(indent(shorten(action, args.max_action_chars), "  "))

        show_obs = args.show_observation or kind in {"validation", "submit"} or status in BAD_STATUS or signal == "suspicious-output"
        if show_obs:
            print("Observation:")
            print(indent(shorten(observation, args.max_observation_chars), "  "))
        else:
            print("Observation:")
            print("  " + one_line(observation, args.max_observation_oneline))


def indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in str(text).splitlines())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="Path to results.jsonl")
    parser.add_argument("--show-read", action="store_true", help="Also show read-only exploration calls")
    parser.add_argument("--show-observation", action="store_true", help="Show full observations for all printed actions")
    parser.add_argument("--only-failed", action="store_true", help="Only print failed/suspicious tasks")
    parser.add_argument("--show-empty", action="store_true", help="Print tasks even if no matching action remains after filtering")
    parser.add_argument("--kind", choices=["implementation-write", "file-or-shell-write", "validation", "submit", "execute-other", "other", "read-only"],
                        help="Only show one classified action kind")
    parser.add_argument("--reward-threshold", type=float, default=0.5)
    parser.add_argument("--max-action-chars", type=int, default=3000)
    parser.add_argument("--max-observation-chars", type=int, default=4000)
    parser.add_argument("--max-observation-oneline", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.jsonl)
    print_summary(rows, args)
    for result in rows:
        print_actions(result, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
