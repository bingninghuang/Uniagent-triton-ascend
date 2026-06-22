#!/usr/bin/env python3
"""Analyze Triton synthesis JSONL results.

Usage:
    python examples/triton_agent/analyze_synth_results.py /path/to/results.jsonl
    python examples/triton_agent/analyze_synth_results.py /path/to/results.jsonl --show-messages --max-command-chars 2000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STATUS_BAD = {"timeout", "syntax_error", "skipped"}
EXIT_BAD = {
    "unknown_error",
    "terminal_dead",
    "timeout_budget_exhausted",
    "max_step_limit",
    "token_limit",
    "format_error",
}


def shorten(text: Any, limit: int) -> str:
    if text is None:
        return ""
    text = str(text)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n... <truncated {len(text) - limit} chars>"


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
                print(f"[warn] skip invalid JSON at line {line_no}: {exc}")
    return rows


def iter_tool_results(result: dict[str, Any]):
    trajectory = result.get("trajectory") or []
    if isinstance(trajectory, str):
        return
    for step_idx, step in enumerate(trajectory, 1):
        if not isinstance(step, dict):
            continue
        tool_results = step.get("tool_results") or []
        if isinstance(tool_results, str):
            continue
        for tool in tool_results:
            if isinstance(tool, dict):
                yield step_idx, step, tool


def extract_errors(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    eval_result = result.get("eval_result")
    if isinstance(eval_result, dict):
        for key in ("error", "stderr", "message"):
            value = eval_result.get(key)
            if value:
                errors.append(f"eval_result.{key}: {value}")
        metrics = eval_result.get("metrics")
        if isinstance(metrics, dict):
            for key in ("verify_error", "perf_error", "error"):
                value = metrics.get(key)
                if value:
                    errors.append(f"metrics.{key}: {value}")
    for _, _, tool in iter_tool_results(result):
        status = tool.get("status")
        if status in STATUS_BAD:
            action = tool.get("action") or ""
            errors.append(f"tool {status}: {action}")
    return errors


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"total tasks: {len(rows)}")
    if not rows:
        return

    exit_counts = Counter(r.get("exit_reason", "<missing>") for r in rows)
    print("\nexit_reason counts:")
    for reason, count in exit_counts.most_common():
        print(f"  {reason:28s} {count}")

    rewards = [float(r.get("reward_score") or 0.0) for r in rows]
    pass_rates = [float(r.get("pass_rate") or 0.0) for r in rows]
    print(f"\navg reward:    {sum(rewards) / len(rewards):.4f}")
    print(f"avg pass_rate: {sum(pass_rates) / len(pass_rates):.4f}")
    print(f"success >0.5:  {sum(1 for x in rewards if x > 0.5)}/{len(rewards)}")

    tool_status_counts = Counter()
    tool_name_counts = Counter()
    timeout_actions = []
    for result in rows:
        for _, _, tool in iter_tool_results(result):
            tool_status_counts[tool.get("status", "<missing>")] += 1
            tool_name_counts[tool.get("name", "<missing>")] += 1
            if tool.get("status") == "timeout":
                timeout_actions.append((result.get("op_name"), tool.get("name"), tool.get("action")))

    print("\ntool status counts:")
    if tool_status_counts:
        for status, count in tool_status_counts.most_common():
            print(f"  {status:16s} {count}")
    else:
        print("  <no tool results recorded>")

    print("\ntool name counts:")
    if tool_name_counts:
        for name, count in tool_name_counts.most_common():
            print(f"  {name:24s} {count}")
    else:
        print("  <no tool calls recorded>")

    if timeout_actions:
        grouped = defaultdict(int)
        for _, name, action in timeout_actions:
            grouped[(name or "<missing>", action or "")] += 1
        print("\nmost common timeout commands:")
        for (name, action), count in sorted(grouped.items(), key=lambda x: x[1], reverse=True)[:10]:
            one_line = " ".join(str(action).split())
            print(f"  [{count}x] {name}: {shorten(one_line, 220)}")


def print_result(result: dict[str, Any], args: argparse.Namespace) -> None:
    op = result.get("op_name", "<unknown>")
    exit_reason = result.get("exit_reason", "<missing>")
    reward = result.get("reward_score", 0.0)
    pass_rate = result.get("pass_rate", 0.0)
    speedup = result.get("speedup_vs_torch")
    turns = result.get("num_turns")
    exec_time = result.get("execution_time")

    bad = exit_reason in EXIT_BAD or float(reward or 0.0) <= args.reward_threshold
    if args.only_bad and not bad:
        return

    print("\n" + "=" * 100)
    print(f"TASK: {op}")
    print("=" * 100)
    print(
        f"exit={exit_reason} | reward={reward} | pass_rate={pass_rate} | "
        f"speedup={speedup} | turns={turns} | time={exec_time}s"
    )

    errors = extract_errors(result)
    if errors:
        print("\nErrors / suspicious signals:")
        for err in errors[: args.max_errors]:
            print("- " + shorten(err, args.max_error_chars).replace("\n", "\n  "))
        if len(errors) > args.max_errors:
            print(f"- ... {len(errors) - args.max_errors} more")

    trajectory = result.get("trajectory") or []
    if not trajectory:
        print("\nTrajectory: <empty>")
    else:
        print("\nTrajectory:")
        for step_idx, step in enumerate(trajectory, 1):
            response = step.get("response") or ""
            step_exit = step.get("exit_reason") or ""
            done = step.get("done")
            print(f"\n  Step {step_idx}: exit={step_exit or '<none>'} done={done} response_len={len(response)}")
            if args.show_response and response:
                print("  Response:")
                print(indent(shorten(response, args.max_response_chars), "    "))

            tool_results = step.get("tool_results") or []
            if not tool_results:
                print("    tool_results: <none>")
                continue
            for tool_idx, tool in enumerate(tool_results, 1):
                name = tool.get("name", "<missing>")
                status = tool.get("status", "<missing>")
                execution_time = tool.get("execution_time")
                action = tool.get("action") or ""
                observation = tool.get("observation") or ""
                marker = " !!" if status in STATUS_BAD else ""
                print(f"    Tool {tool_idx}: {name} status={status} time={execution_time}{marker}")
                print("      action:")
                print(indent(shorten(action, args.max_command_chars), "        "))
                if args.show_observation or status in STATUS_BAD:
                    print("      observation:")
                    print(indent(shorten(observation, args.max_observation_chars), "        "))

    if args.show_messages:
        messages = result.get("messages") or []
        print("\nMessages:")
        for idx, message in enumerate(messages, 1):
            role = message.get("role", "<missing>")
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls")
            print(f"  Message {idx}: role={role} content_len={len(content)}")
            if tool_calls:
                print("    tool_calls:")
                print(indent(shorten(json.dumps(tool_calls, ensure_ascii=False, indent=2), args.max_message_chars), "      "))
            if content:
                print(indent(shorten(content, args.max_message_chars), "    "))


def indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in str(text).splitlines())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="Path to synth results JSONL")
    parser.add_argument("--only-bad", action="store_true", help="Only print failed/suspicious tasks")
    parser.add_argument("--reward-threshold", type=float, default=0.5,
                        help="Tasks with reward <= this threshold are considered bad")
    parser.add_argument("--show-response", action="store_true", help="Print model responses")
    parser.add_argument("--show-observation", action="store_true", help="Print all tool observations")
    parser.add_argument("--show-messages", action="store_true", help="Print full message history")
    parser.add_argument("--max-command-chars", type=int, default=1200)
    parser.add_argument("--max-observation-chars", type=int, default=2000)
    parser.add_argument("--max-response-chars", type=int, default=2000)
    parser.add_argument("--max-message-chars", type=int, default=2000)
    parser.add_argument("--max-error-chars", type=int, default=1200)
    parser.add_argument("--max-errors", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.jsonl)
    print_summary(rows)
    for result in rows:
        print_result(result, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
