#!/usr/bin/env python3
"""Summarize synth eval logs: pass_rate==1.0 and speedup thresholds."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SYNTH_LINE_RE = re.compile(
    r"\[synth\] op=(?P<op>\S+)\s+"
    r"reward=(?P<reward>\S+)\s+"
    r"pass_rate=(?P<pass_rate>\S+)\s+"
    r"speedup=(?P<speedup>\S+)"
)
LEVEL_RE = re.compile(r"(?:^|_)l(?P<level>\d+)(?:_|$)")


def parse_float(value: str) -> float | None:
    if value in ("None", "none", "null", "-"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def collect_ops(logs_dir: Path) -> dict[str, dict]:
    """Parse logs; later lines for the same op overwrite earlier ones."""
    ops: dict[str, dict] = {}
    for log_path in sorted(logs_dir.glob("*.log")):
        with log_path.open(encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                match = SYNTH_LINE_RE.search(line)
                if not match:
                    continue
                op = match.group("op")
                ops[op] = {
                    "op": op,
                    "pass_rate": parse_float(match.group("pass_rate")),
                    "speedup": parse_float(match.group("speedup")),
                    "source": f"{log_path.name}:{lineno}",
                }
    return ops


def collect_done(done_path: Path) -> set[str]:
    """Read unique completed op names from op_done.txt."""
    if not done_path.is_file():
        return set()
    with done_path.open(encoding="utf-8", errors="replace") as fh:
        return {line.strip() for line in fh if line.strip()}


def op_level(op: str) -> str:
    """Return a normalized level name such as level0."""
    match = LEVEL_RE.search(op)
    return f"level{int(match.group('level'))}" if match else "unknown"


def level_sort_key(level: str) -> tuple[int, int]:
    if level.startswith("level") and level[5:].isdigit():
        return (0, int(level[5:]))
    return (1, 0)


def group_records_by_level(ops: dict[str, dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = {}
    for op, rec in ops.items():
        grouped.setdefault(op_level(op), {})[op] = rec
    return grouped


def count_names_by_level(ops: set[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for op in ops:
        level = op_level(op)
        counts[level] = counts.get(level, 0) + 1
    return counts


def summarize(ops: dict[str, dict]) -> dict:
    pass1 = [
        rec
        for rec in ops.values()
        if rec["pass_rate"] is not None and abs(rec["pass_rate"] - 1.0) < 1e-9
    ]
    with_speedup = [rec for rec in pass1 if rec["speedup"] is not None]
    return {
        "total_ops": len(ops),
        "pass_rate_1": pass1,
        "gt_0_6": [rec for rec in with_speedup if rec["speedup"] > 0.6],
        "gt_1": [rec for rec in with_speedup if rec["speedup"] > 1.0],
        "gt_1_2": [rec for rec in with_speedup if rec["speedup"] > 1.2],
        "pass1_no_speedup": [rec for rec in pass1 if rec["speedup"] is None],
    }


def fmt_ops(recs: list[dict]) -> str:
    if not recs:
        return "  (none)"
    lines = []
    for rec in sorted(recs, key=lambda r: r["op"]):
        su = rec["speedup"]
        su_s = "None" if su is None else f"{su:.4f}"
        lines.append(f"  {rec['op']:40s}  pass_rate={rec['pass_rate']:.2f}  speedup={su_s}")
    return "\n".join(lines)


def print_level_stats(level: str, level_ops: dict[str, dict], done_count: int) -> None:
    stats = summarize(level_ops)
    print(f"=== {level} ===")
    print(f"done ops in op_done.txt: {done_count}")
    print(f"finished ops in logs: {stats['total_ops']}")
    print()
    print(f"pass_rate == 1.0: {len(stats['pass_rate_1'])}")
    print(fmt_ops(stats["pass_rate_1"]))
    print()
    print(f"pass_rate == 1.0 and speedup > 0.6: {len(stats['gt_0_6'])}")
    print(fmt_ops(stats["gt_0_6"]))
    print()
    print(f"pass_rate == 1.0 and speedup > 1.0: {len(stats['gt_1'])}")
    print(fmt_ops(stats["gt_1"]))
    print()
    print(f"pass_rate == 1.0 and speedup > 1.2: {len(stats['gt_1_2'])}")
    print(fmt_ops(stats["gt_1_2"]))
    if stats["pass1_no_speedup"]:
        print()
        print(f"pass_rate == 1.0 but speedup is None: {len(stats['pass1_no_speedup'])}")
        print(fmt_ops(stats["pass1_no_speedup"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="/home/h00961522/Triton-Training/eval/output/qwen3.6_NPUKernelbench_level1",
        help="Synth output directory containing logs/",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    logs_dir = run_dir / "logs"
    if not logs_dir.is_dir():
        raise SystemExit(f"logs directory not found: {logs_dir}")

    ops = collect_ops(logs_dir)
    done_path = run_dir / "op_done.txt"
    done_ops = collect_done(done_path)
    ops_by_level = group_records_by_level(ops)
    done_by_level = count_names_by_level(done_ops)
    levels = sorted(set(ops_by_level) | set(done_by_level), key=level_sort_key)

    print(f"run_dir: {run_dir}")
    print(f"done ops in op_done.txt: {len(done_ops)}")
    print(f"finished ops in logs: {len(ops)}")
    if not done_path.is_file():
        print(f"warning: op_done.txt not found: {done_path}")

    for level in levels:
        print()
        print_level_stats(
            level,
            ops_by_level.get(level, {}),
            done_by_level.get(level, 0),
        )


if __name__ == "__main__":
    main()