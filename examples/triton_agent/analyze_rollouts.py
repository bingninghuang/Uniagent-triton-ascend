#!/usr/bin/env python3
"""
Comprehensive rollout analysis for Triton Ascend Agentic RL training.

Two modes:
  - Full-analysis mode (TARGET_LINE = None):
      Aggregates all tasks for executive summary, reward decomposition, failure
      categorization, fix-loop effectiveness, per-operator breakdown, trajectory
      quality, and RL-training insights.
  - Single-task deep-dive mode (TARGET_LINE = <line_number>):
      Zooms into one specific task: task overview, reward breakdown, error root
      cause, stage-1 vs stage-2 timeline, step-by-step trajectory walkthrough,
      and fix-attempt history.

Usage:
    python examples/triton_agent/analyze_rollouts.py

Configure the input path, TARGET_LINE, and display toggles in the CONFIG section.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ============================================================================
# CONFIG — edit these variables
# ============================================================================

# Path to the JSONL results file produced by synth_triton_api.py
RESULTS_FILE = "/tmp/synth_api/results.jsonl"

# Single-task deep-dive: set to a 1-indexed line number to zoom into one task.
# Set to None for full-aggregate analysis mode.
TARGET_LINE: int | None = None   # e.g. 3  →  deep-dive the 3rd task in the JSONL

# --- deep-dive detail toggles (only used when TARGET_LINE is set) ---
DEEPDIVE_SHOW_FULL_RESPONSE = False    # show full model response each step (can be very long)
DEEPDIVE_SHOW_FULL_OBSERVATION = False # show full tool observation each step
DEEPDIVE_MAX_RESPONSE_CHARS = 600      # truncation limit for response per step
DEEPDIVE_MAX_ACTION_CHARS = 500        # truncation limit for tool action
DEEPDIVE_MAX_OBS_CHARS = 400           # truncation limit for tool observation

# --- full-analysis display toggles ---
SHOW_REWARD_DECOMPOSITION = True       # per-component reward breakdown
SHOW_PASS_RATE_DISTRIBUTION = True     # pass_rate histogram
SHOW_FAILURE_ANALYSIS = True           # error type categorization & examples
SHOW_FIX_LOOP_ANALYSIS = True          # stage-2 repair effectiveness
SHOW_OPERATOR_BREAKDOWN = True         # per-operator stats table
SHOW_TRAJECTORY_ANALYSIS = True        # turns / tool usage / thinking
SHOW_INSIGHTS = True                   # RL-training recommendations

# --- detail toggles ---
MAX_FAILURE_EXAMPLES_PER_TYPE = 5     # how many error examples to show per category
MAX_OPERATOR_TABLE_ROWS = 40          # cap per-operator table rows
SHOW_SUCCESS_TASKS = False             # print every successful task name in summary
SHOW_ALL_FAILED_TASKS = True           # print every failed task name with error

# ============================================================================
# Data loading
# ============================================================================


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load JSONL file, skipping empty / malformed lines."""
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.is_file():
        print(f"[ERROR] File not found: {p}")
        sys.exit(1)
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] skip invalid JSON at line {line_no}: {exc}")
    return rows


# ============================================================================
# Helpers
# ============================================================================


def _f(val: Any, precision: int = 4) -> str:
    """Format a float with given precision, or '—' for None."""
    if val is None:
        return "—"
    try:
        return f"{float(val):.{precision}f}"
    except (TypeError, ValueError):
        return str(val)


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{100.0 * numerator / denominator:.1f}%"


def _bar(value: float, width: int = 20, max_val: float = 1.0) -> str:
    """ASCII bar for visualising proportions."""
    if max_val <= 0:
        return ""
    filled = int(round(min(value / max_val, 1.0) * width))
    return "█" * filled + "░" * (width - filled)


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _shorten(text: Any, limit: int = 200) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... <truncated {len(text) - limit} chars>"


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in str(text).splitlines())


def _error_summary(eval_result: dict[str, Any] | None) -> tuple[str, str]:
    """Extract (error_type, error_message) from eval_result."""
    if not isinstance(eval_result, dict):
        return ("unknown", "")
    metrics = eval_result.get("metrics") if isinstance(eval_result.get("metrics"), dict) else eval_result
    error_type = str(metrics.get("error_type") or "")
    error_msg = str(metrics.get("error") or metrics.get("reason") or "")
    if not error_type:
        if not error_msg:
            error_msg = str(eval_result.get("error") or eval_result.get("reason") or "")
        if "compile" in error_msg.lower() or "compilation" in error_msg.lower():
            error_type = "compilation_failed"
        elif "ast" in error_msg.lower():
            error_type = "ast_check_failed"
        elif error_msg:
            error_type = "correctness_failed"
        else:
            error_type = "unknown"
    if not error_msg:
        error_msg = error_type
    return error_type, error_msg


def _iter_trajectory_steps(trajectory: list[Any]):
    """Yield (step_idx, step_dict) from trajectory."""
    if isinstance(trajectory, str):
        return
    for idx, step in enumerate(trajectory, 1):
        if isinstance(step, dict):
            yield idx, step


def _detect_stage_boundaries(messages: list[dict[str, Any]]) -> list[int]:
    """Find message indices where fix/verify injections occur (stage transitions).

    Returns list of message indices (0-indexed) that are injected user prompts
    containing verify feedback or AST check results.
    """
    boundaries = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = str(msg.get("content", ""))
        # Detect fix-injection patterns
        if any(kw in content for kw in (
            "AST check FAILED",
            "FIX ATTEMPT",
            "fix attempt",
            "verify.py results",
            "error_groups",
            "Verify Results",
            "PASSED:",
            "FAILED:",
            "passed_cases",
            "total_cases",
            "Please fix",
            "Fix the implementation",
        )):
            boundaries.append(i)
    return boundaries


def _is_stage2_step(step_idx: int, trajectory: list[dict[str, Any]],
                    messages: list[dict[str, Any]]) -> bool:
    """Heuristic: a step is stage-2 if it occurs after the first fix injection."""
    if not messages:
        # Fallback: check response content
        if step_idx > 12:
            return True
        return False

    boundaries = _detect_stage_boundaries(messages)
    if not boundaries:
        return step_idx > 12

    # Map message indices to approximate step indices
    # Each assistant message with tool calls ≈ one step
    first_fix_msg_idx = boundaries[0]

    # Count assistant messages before the first fix injection
    assistant_count = 0
    for i in range(first_fix_msg_idx):
        if messages[i].get("role") == "assistant":
            assistant_count += 1

    return step_idx > assistant_count


# ============================================================================
# Aggregate analysis functions
# ============================================================================


def analyze_reward(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute reward statistics and per-component breakdown."""
    rewards = [_safe_float(r.get("reward_score")) for r in rows]
    pass_rates = [_safe_float(r.get("pass_rate")) for r in rows]

    components = defaultdict(list)
    for r in rows:
        rb = r.get("reward_breakdown") or {}
        for key in ("ast", "compile", "correctness", "all_correct_bonus", "speedup", "pass_rate", "raw_speedup"):
            components[key].append(_safe_float(rb.get(key)))

    success_count = sum(1 for x in rewards if x > 0.5)
    zero_count = sum(1 for x in rewards if x <= 0.01)

    return {
        "rewards": rewards,
        "pass_rates": pass_rates,
        "components": dict(components),
        "success_count": success_count,
        "zero_count": zero_count,
        "total": len(rows),
    }


def analyze_failures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Categorize failures by error type and extract examples."""
    failed = [r for r in rows if _safe_float(r.get("reward_score")) <= 0.5]

    by_error_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pass_rate_buckets = {"0%": [], "1-49%": [], "50-99%": [], "other": []}

    for r in failed:
        pr = _safe_float(r.get("pass_rate"))
        if pr <= 0.0:
            pass_rate_buckets["0%"].append(r)
        elif pr < 0.5:
            pass_rate_buckets["1-49%"].append(r)
        elif pr < 1.0:
            pass_rate_buckets["50-99%"].append(r)
        else:
            pass_rate_buckets["other"].append(r)

        etype, _ = _error_summary(r.get("eval_result"))
        etype_lower = etype.lower()
        if "ast" in etype_lower:
            category = "AST check failed"
        elif "compile" in etype_lower or "compilation" in etype_lower:
            category = "Compilation failed"
        elif "correctness" in etype_lower:
            category = "Correctness failed"
        elif "missing" in etype_lower:
            category = "Missing artifacts"
        elif "timeout" in etype_lower or "token" in etype_lower:
            category = "Timeout / token limit"
        else:
            category = etype if etype else "Unknown"

        by_error_type[category].append(r)

    # extract most common error messages per category
    error_examples: dict[str, list[dict[str, Any]]] = {}
    for cat, cat_rows in by_error_type.items():
        msg_counter: Counter = Counter()
        msg_examples: dict[str, dict[str, Any]] = {}
        for row in cat_rows:
            _, msg = _error_summary(row.get("eval_result"))
            short = msg.splitlines()[0].strip()[:200] if msg else "(empty)"
            msg_counter[short] += 1
            if short not in msg_examples:
                msg_examples[short] = row
        examples = []
        for short_msg, count in msg_counter.most_common(MAX_FAILURE_EXAMPLES_PER_TYPE):
            examples.append({
                "message": short_msg,
                "count": count,
                "op_name": msg_examples[short_msg].get("op_name", "?"),
                "pass_rate": _safe_float(msg_examples[short_msg].get("pass_rate")),
            })
        error_examples[cat] = examples

    return {
        "total_failed": len(failed),
        "by_error_type": dict(by_error_type),
        "error_examples": error_examples,
        "pass_rate_buckets": pass_rate_buckets,
    }


def analyze_fix_loop(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze whether stage-2 repair actually improves results."""
    fix_attempt_counts: Counter = Counter()
    success_by_fix_attempts: dict[int, list[float]] = defaultdict(list)

    for r in rows:
        trajectory = r.get("trajectory") or []
        if isinstance(trajectory, str):
            continue

        fix_count = 0
        for step in trajectory:
            if not isinstance(step, dict):
                continue
            response = step.get("response") or ""
            if isinstance(response, str) and (
                "verify.py" in response.lower()
                or "ast check failed" in response.lower()
                or "fix attempt" in response.lower()
                or "error_groups" in response.lower()
            ):
                fix_count += 1

        fix_attempt_counts[fix_count] += 1
        reward = _safe_float(r.get("reward_score"))
        success_by_fix_attempts[fix_count].append(reward)

    fix_reward_avg = {}
    for count, rewards in sorted(success_by_fix_attempts.items()):
        fix_reward_avg[count] = {
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "count": len(rewards),
            "success_rate": sum(1 for x in rewards if x > 0.5) / len(rewards) if rewards else 0.0,
        }

    return {
        "fix_attempt_distribution": dict(fix_attempt_counts),
        "fix_reward_avg": fix_reward_avg,
    }


def analyze_operators(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute per-operator statistics."""
    by_op: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_op[r.get("op_name", "<unknown>")].append(r)

    stats = []
    for op, op_rows in sorted(by_op.items()):
        rewards = [_safe_float(r.get("reward_score")) for r in op_rows]
        pass_rates = [_safe_float(r.get("pass_rate")) for r in op_rows]
        turns = [_safe_int(r.get("num_turns")) for r in op_rows]
        speedups = [r.get("speedup_vs_torch") for r in op_rows if r.get("speedup_vs_torch") is not None]

        stats.append({
            "op_name": op,
            "count": len(op_rows),
            "avg_reward": sum(rewards) / len(rewards),
            "avg_pass_rate": sum(pass_rates) / len(pass_rates),
            "success_rate": sum(1 for x in rewards if x > 0.5) / len(rewards),
            "avg_turns": sum(turns) / len(turns),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "avg_speedup": sum(speedups) / len(speedups) if speedups else None,
        })

    stats.sort(key=lambda x: x["avg_reward"])
    return stats


def analyze_trajectories(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze trajectory quality: turns, tool usage, thinking tokens."""
    all_turns = []
    tool_name_counts: Counter = Counter()
    thinking_chars_per_task: list[int] = []
    total_tool_calls = 0
    empty_trajectories = 0

    for r in rows:
        trajectory = r.get("trajectory") or []
        if isinstance(trajectory, str) or not trajectory:
            empty_trajectories += 1
            all_turns.append(0)
            continue

        num_turns = len(trajectory)
        all_turns.append(num_turns)

        task_thinking = 0
        task_tool_calls = 0

        for step in trajectory:
            if not isinstance(step, dict):
                continue

            response = step.get("response") or ""
            thought = step.get("thought") or ""
            if isinstance(response, str):
                task_thinking += len(response)
            if isinstance(thought, str):
                task_thinking += len(thought)

            tool_results = step.get("tool_results") or []
            if isinstance(tool_results, list):
                for tool in tool_results:
                    if isinstance(tool, dict):
                        task_tool_calls += 1
                        tool_name_counts[tool.get("name", "<unknown>")] += 1

        thinking_chars_per_task.append(task_thinking)
        total_tool_calls += task_tool_calls

    success_turns = [
        t for t, r in zip(all_turns, rows)
        if _safe_float(r.get("reward_score")) > 0.5
    ]
    failed_turns = [
        t for t, r in zip(all_turns, rows)
        if _safe_float(r.get("reward_score")) <= 0.5
    ]

    return {
        "total_tool_calls": total_tool_calls,
        "empty_trajectories": empty_trajectories,
        "turns_all": all_turns,
        "turns_success": success_turns,
        "turns_failed": failed_turns,
        "tool_name_counts": dict(tool_name_counts),
        "thinking_chars_per_task": thinking_chars_per_task,
    }


def analyze_exit_reasons(rows: list[dict[str, Any]]) -> Counter:
    """Count exit reasons."""
    return Counter(r.get("exit_reason", "<missing>") for r in rows)


# ============================================================================
# Display: print helpers
# ============================================================================


def print_header(title: str, width: int = 90) -> None:
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title: str) -> None:
    print(f"\n  ── {title} ──")


# ============================================================================
# Display: full aggregate analysis
# ============================================================================


def _print_executive_summary(rows: list[dict[str, Any]], reward_stats: dict[str, Any]) -> None:
    print_header("1. EXECUTIVE SUMMARY")

    n = len(rows)
    if n == 0:
        print("  No data.")
        return

    rewards = reward_stats["rewards"]
    pass_rates = reward_stats["pass_rates"]
    success = reward_stats["success_count"]
    zero = reward_stats["zero_count"]

    print(f"  Total tasks:           {n}")
    print(f"  Success (reward>0.5):  {success}  ({_pct(success, n)})")
    print(f"  Zero reward:           {zero}  ({_pct(zero, n)})")
    print(f"  Avg reward:            {_f(sum(rewards) / n)}")
    print(f"  Median reward:         {_f(sorted(rewards)[n // 2])}")
    print(f"  Avg pass_rate:         {_f(sum(pass_rates) / n)}")
    print(f"  Avg execution time:    {_f(sum(_safe_float(r.get('execution_time')) for r in rows) / n, 1)}s")

    exit_counts = analyze_exit_reasons(rows)
    print_subheader("Exit reasons")
    for reason, count in exit_counts.most_common():
        bar = _bar(count, 12, n)
        print(f"  {reason:30s} {count:4d}  {_pct(count, n):>6s}  {bar}")

    if SHOW_SUCCESS_TASKS and success > 0:
        print_subheader("Successful tasks")
        for r in rows:
            if _safe_float(r.get("reward_score")) > 0.5:
                print(f"  ✓ {r['op_name']:<50s}  reward={_f(r.get('reward_score'))}  pass_rate={_f(r.get('pass_rate'))}")

    if SHOW_ALL_FAILED_TASKS:
        failed = [r for r in rows if _safe_float(r.get("reward_score")) <= 0.5]
        if failed:
            print_subheader(f"Failed tasks ({len(failed)})")
            for idx, r in enumerate(failed):
                etype, emsg = _error_summary(r.get("eval_result"))
                # Find the original line number for this task
                line_ref = ""
                for i, orig in enumerate(rows):
                    if orig is r:
                        line_ref = f"  [line {i + 1}]"
                        break
                print(f"  ✗ {r['op_name']:<50s}  reward={_f(r.get('reward_score'))}  pass_rate={_f(r.get('pass_rate'))}  {etype}{line_ref}")
                if emsg and emsg != etype:
                    oneline = " ".join(str(emsg).split())[:120]
                    print(f"    {'':50s}  ↳ {oneline}")


def _print_reward_analysis(reward_stats: dict[str, Any]) -> None:
    if not SHOW_REWARD_DECOMPOSITION:
        return
    print_header("2. REWARD DECOMPOSITION")

    comps = reward_stats["components"]
    n = reward_stats["total"]
    if n == 0:
        print("  No data.")
        return

    max_vals = {"ast": 0.0, "compile": 0.10, "correctness": 0.55, "all_correct_bonus": 0.10, "speedup": 0.40}
    labels = {
        "ast": "AST check (0.00)",
        "compile": "Compile (0.10)",
        "correctness": "Correctness (0.55 × pass_rate)",
        "all_correct_bonus": "All-correct bonus (0.10)",
        "speedup": "Speedup (0.40 × ...)",
    }

    print(f"  {'Component':<30s} {'Avg':>8s}  {'% of max':>8s}  {'Median':>8s}  {'NonZero':>8s}  Distribution")
    print(f"  {'─' * 30} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 30}")

    for key in ("ast", "compile", "correctness", "all_correct_bonus", "speedup"):
        vals = comps.get(key, [])
        if not vals:
            continue
        avg_val = sum(vals) / len(vals)
        med_val = sorted(vals)[len(vals) // 2]
        nonzero = sum(1 for v in vals if v > 0.001)
        max_val = max_vals.get(key, 1.0)
        pct_of_max = (avg_val / max_val * 100) if max_val > 0 else 0.0
        bar = _bar(avg_val, 25, max_val) if max_val > 0 else ""

        print(f"  {labels.get(key, key):30s} {_f(avg_val):>8s}  {pct_of_max:7.1f}%  {_f(med_val):>8s}  {nonzero:4d}/{len(vals):<4d}  {bar}")

    rewards = reward_stats["rewards"]
    print_subheader("Reward distribution histogram")
    buckets = [(0, 0.01), (0.01, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for lo, hi in buckets:
        count = sum(1 for x in rewards if lo <= x < hi)
        label = f"{lo:.2f}–{hi:.2f}"
        bar = _bar(count, 40, len(rewards))
        print(f"  {label:>10s}  {count:4d}  {bar}")

    if SHOW_PASS_RATE_DISTRIBUTION:
        print_subheader("Pass rate distribution")
        pr_buckets = [(0, 0.0), (0.01, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.0)]
        pass_rates = reward_stats["pass_rates"]
        for lo, hi in pr_buckets:
            if lo == hi == 0.0:
                count = sum(1 for x in pass_rates if x <= 0.0)
                label = "exactly 0"
            elif lo == 1.0:
                count = sum(1 for x in pass_rates if x >= 1.0)
                label = "exactly 1.0"
            else:
                count = sum(1 for x in pass_rates if lo <= x < hi)
                label = f"{lo:.2f}–{hi:.2f}"
            bar = _bar(count, 40, len(pass_rates))
            print(f"  {label:>10s}  {count:4d}  {bar}")


def _print_failure_analysis(rows: list[dict[str, Any]], failure_stats: dict[str, Any]) -> None:
    if not SHOW_FAILURE_ANALYSIS:
        return
    print_header("3. FAILURE ANALYSIS")

    n_failed = failure_stats["total_failed"]
    if n_failed == 0:
        print("  No failures! 🎉")
        return

    print(f"  Total failed tasks: {n_failed}")
    print()

    print(f"  {'Error Category':<30s} {'Count':>6s}  {'%':>6s}  {'Avg PassRate':>12s}")
    print(f"  {'─' * 30} {'─' * 6} {'─' * 6} {'─' * 12}")
    by_type = failure_stats["by_error_type"]
    for cat in sorted(by_type, key=lambda c: len(by_type[c]), reverse=True):
        cat_rows = by_type[cat]
        avg_pr = sum(_safe_float(r.get("pass_rate")) for r in cat_rows) / len(cat_rows)
        print(f"  {cat:30s} {len(cat_rows):6d}  {_pct(len(cat_rows), n_failed):>6s}  {_f(avg_pr):>12s}")

    error_examples = failure_stats["error_examples"]
    for cat in sorted(by_type, key=lambda c: len(by_type[c]), reverse=True):
        examples = error_examples.get(cat, [])
        if not examples:
            continue
        print_subheader(f"Top error patterns: {cat}")
        for ex in examples:
            print(f"  [{ex['count']}×] {ex['op_name']} (pass_rate={_f(ex['pass_rate'], 2)})")
            print(f"         {ex['message']}")

    print_subheader("Partial progress among failures")
    buckets = failure_stats["pass_rate_buckets"]
    for label in ("0%", "1-49%", "50-99%"):
        count = len(buckets.get(label, []))
        print(f"  pass_rate {label:>8s}: {count:4d} tasks  ({_pct(count, n_failed)})")


def _print_fix_loop_analysis(fix_stats: dict[str, Any]) -> None:
    if not SHOW_FIX_LOOP_ANALYSIS:
        return
    print_header("4. FIX LOOP (STAGE 2) ANALYSIS")

    dist = fix_stats["fix_attempt_distribution"]
    if not dist:
        print("  No fix-loop data available.")
        return

    print(f"  {'Fix injections':>16s}  {'Tasks':>6s}  {'Avg Reward':>10s}  {'Success Rate':>12s}")
    print(f"  {'─' * 16} {'─' * 6} {'─' * 10} {'─' * 12}")
    for count in sorted(dist):
        info = fix_stats["fix_reward_avg"].get(count, {})
        print(
            f"  {count:16d}  {dist[count]:6d}  {_f(info.get('avg_reward')):>10s}  "
            f"{_pct(int(info.get('success_rate', 0) * dist[count]), dist[count]):>12s}"
        )

    print()
    print("  Interpretation:")
    print("  - Tasks with 0 fix injections = AST+compile passed on first try, or no verify reached.")
    print("  - More injections → model received more repair feedback before giving up.")
    print("  - If success rate drops with more injections, the repair prompts may not help enough.")


def _print_operator_breakdown(op_stats: list[dict[str, Any]]) -> None:
    if not SHOW_OPERATOR_BREAKDOWN:
        return
    print_header("5. OPERATOR BREAKDOWN")

    if not op_stats:
        print("  No data.")
        return

    print(f"  {'Operator':<45s} {'N':>3s}  {'AvgReward':>9s}  {'AvgPR':>6s}  {'Success':>7s}  {'Turns':>5s}  {'Speedup':>7s}")
    print(f"  {'─' * 45} {'─' * 3}  {'─' * 9}  {'─' * 6}  {'─' * 7}  {'─' * 5}  {'─' * 7}")

    for s in op_stats[:MAX_OPERATOR_TABLE_ROWS]:
        succ_str = f"{int(s['success_rate'] * s['count'])}/{s['count']}"
        sp_str = _f(s["avg_speedup"], 2) if s["avg_speedup"] is not None else "—"
        print(
            f"  {s['op_name']:<45s} {s['count']:3d}  {_f(s['avg_reward']):>9s}  "
            f"{_f(s['avg_pass_rate'], 2):>6s}  {succ_str:>7s}  {_f(s['avg_turns'], 1):>5s}  {sp_str:>7s}"
        )

    if len(op_stats) > MAX_OPERATOR_TABLE_ROWS:
        print(f"  ... ({len(op_stats) - MAX_OPERATOR_TABLE_ROWS} more operators not shown)")

    worst = [s for s in op_stats if s["success_rate"] < 0.5]
    if worst:
        print_subheader(f"Operators with <50% success rate ({len(worst)})")
        for s in worst[:15]:
            print(f"  {s['op_name']:<50s}  success={_pct(int(s['success_rate'] * s['count']), s['count'])}  avg_reward={_f(s['avg_reward'])}")


def _print_trajectory_analysis(traj_stats: dict[str, Any]) -> None:
    if not SHOW_TRAJECTORY_ANALYSIS:
        return
    print_header("6. TRAJECTORY QUALITY")

    all_t = traj_stats["turns_all"]
    success_t = traj_stats["turns_success"]
    failed_t = traj_stats["turns_failed"]

    if not all_t:
        print("  No trajectory data.")
        return

    print(f"  Total tool calls:                {traj_stats['total_tool_calls']}")
    print(f"  Empty trajectories:              {traj_stats['empty_trajectories']}")

    print_subheader("Turns per task")
    for label, turns in [("All", all_t), ("Success", success_t), ("Failed", failed_t)]:
        if not turns:
            continue
        avg_t = sum(turns) / len(turns)
        med_t = sorted(turns)[len(turns) // 2]
        print(f"  {label:>10s}: avg={_f(avg_t, 1)}  median={med_t}  min={min(turns)}  max={max(turns)}  n={len(turns)}")

    print_subheader("Tool usage frequency")
    tool_counts = traj_stats["tool_name_counts"]
    total = sum(tool_counts.values())
    if total > 0:
        for name, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
            bar = _bar(count, 30, total)
            print(f"  {name:28s} {count:5d}  {_pct(count, total):>6s}  {bar}")

    thinking = traj_stats["thinking_chars_per_task"]
    if thinking:
        avg_thinking = sum(thinking) / len(thinking)
        print_subheader("Response / thinking volume")
        print(f"  Avg response chars per task: {avg_thinking:,.0f}")
        print(f"  Max response chars per task: {max(thinking):,}")
        print(f"  (Large volumes may hit the 1MB API limit with DeepSeek-V4 reasoning tokens)")


def _print_insights(rows: list[dict[str, Any]], reward_stats: dict[str, Any],
                    failure_stats: dict[str, Any], op_stats: list[dict[str, Any]]) -> None:
    if not SHOW_INSIGHTS:
        return
    print_header("7. RL TRAINING INSIGHTS & RECOMMENDATIONS")

    n = len(rows)
    if n == 0:
        return

    rewards = reward_stats["rewards"]
    comps = reward_stats["components"]

    insights = []

    # 1. Reward signal quality
    nonzero_reward = sum(1 for r in rewards if r > 0.01)
    insights.append(
        f"1. Reward signal: {nonzero_reward}/{n} ({_pct(nonzero_reward, n)}) tasks receive non-zero reward. "
        + ("Good — most tasks get some signal." if nonzero_reward > n * 0.7
           else "⚠ Many tasks get zero reward — consider adding partial credit for compile-only success.")
    )

    # 2. Reward sparsity
    zero_reward = reward_stats["zero_count"]
    if zero_reward > n * 0.3:
        insights.append(
            f"2. ⚠ Reward sparsity: {zero_reward}/{n} ({_pct(zero_reward, n)}) tasks have zero reward. "
            "This makes credit assignment hard. Consider: (a) reward shaping for intermediate steps, "
            "(b) per-tool-call reward signals, or (c) process reward model (PRM) for dense feedback."
        )

    # 3. Speedup component
    speedup_vals = comps.get("speedup", [])
    nonzero_speedup = sum(1 for v in speedup_vals if v > 0.001)
    if nonzero_speedup == 0:
        insights.append(
            "3. ⚠ Speedup reward is always 0 — benchmark.py is disabled. This removes 40% of the reward "
            "signal. Enable benchmark.py or replace with a proxy metric (e.g., compilation success proxy) "
            "to give the model a richer training signal."
        )

    # 4. Correctness component variance
    corr_vals = comps.get("correctness", [])
    if corr_vals:
        corr_nonzero = [v for v in corr_vals if v > 0.001]
        if corr_nonzero:
            corr_variance = sum((v - sum(corr_nonzero) / len(corr_nonzero)) ** 2 for v in corr_nonzero) / len(corr_nonzero)
            if corr_variance < 0.01:
                insights.append(
                    f"4. ℹ Correctness reward has low variance ({corr_variance:.4f}). "
                    "The model may not be learning fine-grained improvements. Consider more test cases "
                    "or harder benchmarks to increase reward discrimination."
                )

    # 5. Failure mode concentration
    by_type = failure_stats["by_error_type"]
    if by_type:
        top_failure = max(by_type, key=lambda k: len(by_type[k]))
        top_pct = len(by_type[top_failure]) / failure_stats["total_failed"] * 100
        if top_pct > 50:
            insights.append(
                f"5. Failure concentration: {top_pct:.0f}% of failures are '{top_failure}'. "
                "Consider adding targeted skill documents or few-shot examples for this failure mode "
                "in the system prompt."
            )

    # 6. Fix loop effectiveness
    success_rate = reward_stats["success_count"] / n if n > 0 else 0
    if success_rate < 0.5:
        insights.append(
            f"6. ⚠ Overall success rate is {_pct(reward_stats['success_count'], n)}. "
            "Consider: (a) increasing max fix attempts from 5, (b) improving error-to-fix prompt quality, "
            "(c) adding retrieval-augmented fixes from a knowledge base."
        )

    # 7. Trajectory length
    all_turns = [_safe_int(r.get("num_turns")) for r in rows]
    if all_turns:
        long_trajectories = sum(1 for t in all_turns if t > 12)
        if long_trajectories > n * 0.3:
            insights.append(
                f"7. ⚠ {long_trajectories}/{n} tasks exceed 12 turns (stage 1 limit). "
                "Long trajectories bloat message history and risk hitting API token limits. "
                "Consider truncating or summarizing older messages."
            )

    # 8. Worst operators
    worst_ops = [s for s in op_stats if s["success_rate"] == 0 and s["count"] >= 1]
    if worst_ops:
        op_names = ", ".join(s["op_name"] for s in worst_ops[:5])
        if len(worst_ops) > 5:
            op_names += f" (+{len(worst_ops) - 5} more)"
        insights.append(
            f"8. {len(worst_ops)} operators have 0% success: {op_names}. "
            "These may need: (a) better skill documents, (b) manual reference implementations, "
            "or (c) exclusion from the training set until the model can handle them."
        )

    for insight in insights:
        print(f"  {insight}")
        print()


# ============================================================================
# Single-task deep-dive
# ============================================================================


def _deep_dive_task_overview(task: dict[str, Any]) -> None:
    """Print a one-card overview of the task."""
    op_name = task.get("op_name", "<unknown>")
    reward = _safe_float(task.get("reward_score"))
    pass_rate = _safe_float(task.get("pass_rate"))
    speedup = task.get("speedup_vs_torch")
    exit_reason = task.get("exit_reason", "?")
    num_turns = _safe_int(task.get("num_turns"))
    exec_time = task.get("execution_time")

    success = reward > 0.5
    icon = "✓ PASSED" if success else "✗ FAILED"

    etype, emsg = _error_summary(task.get("eval_result"))

    print_header(f"DEEP DIVE: {op_name}")
    print(f"  Status:         {icon}")
    print(f"  Reward:         {_f(reward)}")
    print(f"  Pass rate:      {_f(pass_rate)}")
    print(f"  Speedup:        {_f(speedup, 2) if speedup is not None else '—'}")
    print(f"  Exit reason:    {exit_reason}")
    print(f"  Total turns:    {num_turns}")
    print(f"  Execution time: {_f(exec_time, 1)}s" if exec_time else f"  Execution time: —")


def _deep_dive_reward_breakdown(task: dict[str, Any]) -> None:
    """Print reward component breakdown for a single task."""
    rb = task.get("reward_breakdown") or {}
    if not rb:
        return

    print_subheader("Reward breakdown")

    # Weights from synth_common
    weights = {"ast": 0.0, "compile": 0.10, "correctness": 0.55, "all_correct_bonus": 0.10, "speedup": 0.40}
    labels = {
        "ast": "AST check",
        "compile": "Compile",
        "correctness": "Correctness",
        "all_correct_bonus": "All-correct bonus",
        "speedup": "Speedup",
    }

    for key in ("ast", "compile", "correctness", "all_correct_bonus", "speedup"):
        val = _safe_float(rb.get(key))
        w = weights[key]
        pct = (val / w * 100) if w > 0 else 0.0
        bar = _bar(val, 20, w) if w > 0 else ""
        print(f"  {labels[key]:20s}  {_f(val):>6s} / {_f(w):>5s}  ({pct:5.1f}%)  {bar}")

    print(f"  {'─' * 60}")
    print(f"  {'TOTAL':20s}  {_f(rb.get('total', task.get('reward_score'))):>6s}")
    print(f"  pass_rate (reward calc): {_f(rb.get('pass_rate'))}")
    print(f"  raw_speedup:             {_f(rb.get('raw_speedup'), 2)}")


def _deep_dive_error_analysis(task: dict[str, Any]) -> None:
    """Print detailed error info from eval_result."""
    eval_result = task.get("eval_result")
    if not isinstance(eval_result, dict):
        return

    metrics = eval_result.get("metrics") if isinstance(eval_result.get("metrics"), dict) else {}
    if not metrics:
        return

    # Only show if there's a failure or we have interesting metrics
    success = metrics.get("success", True)
    total_cases = _safe_int(metrics.get("total_cases"))
    passed_cases = _safe_int(metrics.get("passed_cases"))
    failed_cases = _safe_int(metrics.get("failed_cases"))

    print_subheader("Evaluation result")

    print(f"  AST check:     {'✓ passed' if metrics.get('ast_check_ok') else '✗ FAILED'}")
    print(f"  Compile:       {'✓ passed' if metrics.get('compile_ok') else '✗ FAILED'}")
    print(f"  Correctness:   {'✓ passed' if metrics.get('correctness_ok') else '✗ FAILED'}")
    print(f"  Test cases:    {passed_cases}/{total_cases} passed ({failed_cases} failed)")
    print(f"  Perf missing:  {metrics.get('perf_missing', '?')}")
    if metrics.get("latency") is not None:
        print(f"  Latency:       {_f(metrics.get('latency'), 3)}ms")

    if not success:
        etype, emsg = _error_summary(eval_result)
        print()
        print(f"  ╔══ ROOT CAUSE ══")
        print(f"  ║  Error type: {etype}")
        for line in emsg.splitlines()[:15]:
            print(f"  ║  {line}")
        print(f"  ╚{'═' * 60}")

        # Show perf data if available
        perf_data = metrics.get("perf_data")
        if isinstance(perf_data, dict) and perf_data:
            print()
            print(f"  Perf data keys: {list(perf_data.keys())}")
            implementation = perf_data.get("implementation")
            if isinstance(implementation, dict):
                print(f"  Implementation latency: {_f(implementation.get('avg_latency_ms'), 3)}ms")


def _deep_dive_stage_timeline(task: dict[str, Any]) -> None:
    """Print a compact timeline showing stage transitions and key events."""
    trajectory = task.get("trajectory") or []
    messages = task.get("messages") or []
    if isinstance(trajectory, str) or not trajectory:
        print_subheader("Stage timeline: <empty trajectory>")
        return

    # Detect stage boundaries from messages
    boundaries = _detect_stage_boundaries(messages) if messages else []
    # Count how many assistant messages before first fix injection
    stage1_turns = 0
    if boundaries:
        first_fix = boundaries[0]
        for i in range(first_fix):
            if i < len(messages) and messages[i].get("role") == "assistant":
                stage1_turns += 1
    else:
        stage1_turns = min(12, len(trajectory))

    print_subheader("Stage timeline")
    print(f"  Stage 1 (initial impl):  turns  1–{stage1_turns}  (code writing, no verify)")
    if stage1_turns < len(trajectory):
        print(f"  Stage 2 (verify+fix):    turns {stage1_turns + 1}–{len(trajectory)}  (fix attempts)")

    # Compact per-turn overview
    print()
    print(f"  {'Turn':>5s}  {'Stage':>6s}  {'Tools':>5s}  {'Tool names':30s}  {'Key event'}")
    print(f"  {'─' * 5}  {'─' * 6}  {'─' * 5}  {'─' * 30}  {'─' * 40}")

    for step_idx, step in enumerate(trajectory, 1):
        if not isinstance(step, dict):
            continue

        stage = "S2" if step_idx > stage1_turns else "S1"

        tool_results = step.get("tool_results") or []
        if isinstance(tool_results, list):
            tool_names = [t.get("name", "?") for t in tool_results if isinstance(t, dict)]
            tool_count = len(tool_names)
            names_str = ", ".join(tool_names[:3])
            if len(tool_names) > 3:
                names_str += f" +{len(tool_names) - 3}"
        else:
            tool_count = 0
            names_str = "(none)"

        # Detect key event
        response = step.get("response") or ""
        key_event = ""
        if "verify.py" in str(response).lower() or "passed_cases" in str(response).lower():
            key_event = "📋 verify results received"
        elif "ast check failed" in str(response).lower():
            key_event = "🔧 AST fix prompted"
        elif step_idx == 1:
            key_event = "🚀 initial code generation"
        elif step_idx == stage1_turns:
            key_event = "⏹ end of stage 1"
        elif "create" in str(tool_results).lower():
            key_event = "📝 file created"
        elif "str_replace" in str(tool_results).lower():
            key_event = "✏️ code edit"

        # Check for error in tool status
        if not key_event:
            for t in (tool_results if isinstance(tool_results, list) else []):
                if isinstance(t, dict) and t.get("status") in ("timeout", "syntax_error", "skipped"):
                    key_event = f"⚠ {t.get('status')}"
                    break

        print(f"  {step_idx:5d}  {stage:>6s}  {tool_count:5d}  {names_str:30s}  {key_event}")


def _deep_dive_step_detail(task: dict[str, Any], stage1_turns: int) -> None:
    """Print detailed step-by-step trajectory."""
    trajectory = task.get("trajectory") or []
    if isinstance(trajectory, str) or not trajectory:
        return

    print_header("STEP-BY-STEP TRAJECTORY")

    for step_idx, step in enumerate(trajectory, 1):
        if not isinstance(step, dict):
            continue

        stage = "S2 (verify+fix)" if step_idx > stage1_turns else "S1 (code writing)"
        response = step.get("response") or ""
        thought = step.get("thought") or ""
        tool_results = step.get("tool_results") or []
        exit_reason = step.get("exit_reason") or ""
        done = step.get("done")

        # Step header
        print(f"\n{'─' * 90}")
        print(f"  TURN {step_idx}  |  {stage}  |  done={done}  |  exit={exit_reason or '<none>'}")
        print(f"{'─' * 90}")

        # Thinking
        if thought:
            print(f"\n  💭 THINKING ({len(thought)} chars):")
            if DEEPDIVE_SHOW_FULL_RESPONSE:
                print(_indent(thought, "    "))
            else:
                print(_indent(_shorten(thought, DEEPDIVE_MAX_RESPONSE_CHARS), "    "))

        # Response
        if response:
            print(f"\n  📝 RESPONSE ({len(response)} chars):")
            if DEEPDIVE_SHOW_FULL_RESPONSE:
                print(_indent(response, "    "))
            else:
                # Show first N chars, but try to show complete first line
                lines = response.splitlines()
                shown = []
                total = 0
                for line in lines:
                    if total + len(line) > DEEPDIVE_MAX_RESPONSE_CHARS and shown:
                        shown.append(f"... <truncated, {len(response) - total} more chars>")
                        break
                    shown.append(line)
                    total += len(line) + 1
                print(_indent("\n".join(shown), "    "))

        # Tool calls
        if isinstance(tool_results, list) and tool_results:
            print(f"\n  🔧 TOOL CALLS ({len(tool_results)}):")
            for ti, tool in enumerate(tool_results, 1):
                if not isinstance(tool, dict):
                    continue
                name = tool.get("name", "<unknown>")
                status = tool.get("status", "<unknown>")
                exec_time = tool.get("execution_time")
                action = tool.get("action") or ""
                observation = tool.get("observation") or ""

                status_marker = ""
                if status in ("timeout", "syntax_error", "skipped"):
                    status_marker = " ⚠"

                print(f"\n    Tool {ti}: {name}  |  status={status}{status_marker}  |  time={exec_time}")

                # Action
                print(f"    ── ACTION ──")
                if DEEPDIVE_SHOW_FULL_OBSERVATION:
                    print(_indent(action if action else "(empty)", "      "))
                else:
                    print(_indent(_shorten(action, DEEPDIVE_MAX_ACTION_CHARS), "      "))

                # Observation
                if observation:
                    print(f"    ── OBSERVATION ──")
                    if DEEPDIVE_SHOW_FULL_OBSERVATION or status in ("timeout", "syntax_error", "skipped"):
                        print(_indent(observation, "      "))
                    else:
                        # Smart truncation: show beginning and end for verify outputs
                        if "passed_cases" in observation.lower() or "total_cases" in observation.lower():
                            # This is verify output — show key lines
                            for line in observation.splitlines():
                                line_stripped = line.strip()
                                if any(kw in line_stripped.lower() for kw in (
                                    "passed", "failed", "total_cases", "error", "success",
                                    "pass_rate", "summary", "correctness"
                                )):
                                    print(f"      {line_stripped[:200]}")
                            print(f"      ... ({len(observation)} total chars)")
                        else:
                            print(_indent(_shorten(observation, DEEPDIVE_MAX_OBS_CHARS), "      "))
        else:
            print(f"\n  🔧 TOOL CALLS: <none>")


def _deep_dive_fix_history(task: dict[str, Any], stage1_turns: int) -> None:
    """Summarize the fix-attempt history across stage 2."""
    trajectory = task.get("trajectory") or []
    if isinstance(trajectory, str) or len(trajectory) <= stage1_turns:
        return

    print_subheader("Fix attempt history")

    messages = task.get("messages") or []
    boundaries = _detect_stage_boundaries(messages) if messages else []

    if boundaries:
        print(f"  Fix injections detected at message indices: {boundaries}")
        print(f"  Total fix prompts injected: {len(boundaries)}")
    else:
        print(f"  No explicit fix injections detected in messages.")

    # Look at stage 2 steps for fix patterns
    fix_num = 0
    for step_idx, step in enumerate(trajectory, 1):
        if step_idx <= stage1_turns:
            continue
        if not isinstance(step, dict):
            continue

        response = step.get("response") or ""
        tool_results = step.get("tool_results") or []

        # Detect if this step is reacting to a fix prompt
        is_fix_reaction = any(kw in str(response).lower() for kw in (
            "fix attempt", "ast check failed", "error_groups", "passed_cases",
            "verify.py", "please fix", "fix the implementation",
        ))

        if is_fix_reaction:
            fix_num += 1
            # Extract what the model tried to fix
            fix_summary = ""
            for line in response.splitlines()[:5]:
                stripped = line.strip()
                if stripped and len(stripped) > 20:
                    fix_summary = stripped[:150]
                    break

            # Check tool actions for what was changed
            edit_actions = []
            for t in (tool_results if isinstance(tool_results, list) else []):
                if not isinstance(t, dict):
                    continue
                action = t.get("action") or ""
                if "str_replace" in str(action):
                    edit_actions.append("str_replace edit")
                elif "create" in str(action):
                    edit_actions.append("file create")

            print(f"  Fix #{fix_num} (turn {step_idx}): {', '.join(edit_actions) if edit_actions else 'no edits'}")
            if fix_summary:
                print(f"    ↳ {fix_summary}")

    if fix_num == 0:
        print("  (No fix reactions detected in stage 2 — model may have ignored fix prompts)")


def _deep_dive_messages_summary(task: dict[str, Any]) -> None:
    """Summarize the message history sizes by role."""
    messages = task.get("messages") or []
    if not messages:
        return

    print_subheader("Messages summary")

    role_counts: Counter = Counter()
    role_chars: dict[str, int] = defaultdict(int)

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "?")
        role_counts[role] += 1
        content = msg.get("content") or ""
        role_chars[role] += len(str(content))

    print(f"  {'Role':>12s}  {'Count':>6s}  {'Total chars':>12s}  {'Avg chars':>10s}")
    print(f"  {'─' * 12}  {'─' * 6}  {'─' * 12}  {'─' * 10}")
    for role in ("system", "user", "assistant", "tool"):
        count = role_counts.get(role, 0)
        chars = role_chars.get(role, 0)
        if count > 0:
            print(f"  {role:>12s}  {count:6d}  {chars:>12,}  {chars // count:>10,}")

    total_chars = sum(role_chars.values())
    print(f"  {'─' * 12}  {'─' * 6}  {'─' * 12}")
    print(f"  {'TOTAL':>12s}  {sum(role_counts.values()):6d}  {total_chars:>12,}")
    if total_chars > 900_000:
        print(f"  ⚠ Warning: total message size ({total_chars:,} chars) is approaching 1MB API limit!")


def deep_dive_single_task(task: dict[str, Any], line_no: int) -> None:
    """Perform in-depth analysis of a single task's rollout."""
    print(f"Analyzing line {line_no} in {RESULTS_FILE}")

    trajectory = task.get("trajectory") or []
    messages = task.get("messages") or []

    if isinstance(trajectory, str):
        print("[ERROR] Trajectory is a string (not parsed) — cannot deep-dive.")
        return

    # Detect stage boundaries
    boundaries = _detect_stage_boundaries(messages) if messages else []
    stage1_turns = 0
    if boundaries:
        first_fix = boundaries[0]
        for i in range(first_fix):
            if i < len(messages) and messages[i].get("role") == "assistant":
                stage1_turns += 1
    if stage1_turns == 0:
        stage1_turns = min(12, len(trajectory) if isinstance(trajectory, list) else 0)

    # --- Display sections ---
    _deep_dive_task_overview(task)
    _deep_dive_reward_breakdown(task)
    _deep_dive_error_analysis(task)
    _deep_dive_stage_timeline(task)
    _deep_dive_fix_history(task, stage1_turns)
    _deep_dive_messages_summary(task)
    _deep_dive_step_detail(task, stage1_turns)

    print()
    print("=" * 90)
    print("  Deep dive complete.")
    print("=" * 90)


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    print("Triton Ascend — Rollout Analysis for Agentic RL")
    print(f"Input: {RESULTS_FILE}")

    rows = load_jsonl(RESULTS_FILE)
    if not rows:
        print("[ERROR] No valid rows found in input file.")
        return 1

    # --- single-task deep-dive mode ---
    if TARGET_LINE is not None:
        if TARGET_LINE < 1 or TARGET_LINE > len(rows):
            print(f"[ERROR] TARGET_LINE={TARGET_LINE} out of range (1–{len(rows)}).")
            return 1
        task = rows[TARGET_LINE - 1]
        deep_dive_single_task(task, TARGET_LINE)
        return 0

    # --- full aggregate analysis mode ---
    print(f"Loaded {len(rows)} task results.\n")

    reward_stats = analyze_reward(rows)
    failure_stats = analyze_failures(rows)
    fix_stats = analyze_fix_loop(rows)
    op_stats = analyze_operators(rows)
    traj_stats = analyze_trajectories(rows)

    _print_executive_summary(rows, reward_stats)
    _print_reward_analysis(reward_stats)
    _print_failure_analysis(rows, failure_stats)
    _print_fix_loop_analysis(fix_stats)
    _print_operator_breakdown(op_stats)
    _print_trajectory_analysis(traj_stats)
    _print_insights(rows, reward_stats, failure_stats, op_stats)

    print()
    print("=" * 90)
    print("  Analysis complete.")
    print(f"  Tip: set TARGET_LINE=<N> to deep-dive a specific task.")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
