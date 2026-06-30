#!/usr/bin/env python3
"""Summarise Triton synthesis JSONL results.

Usage:
    python examples/triton_agent/summarize_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# -------- CONFIGURE THIS PATH --------
RESULTS_FILE = "/tmp/synth_api/results.jsonl"
# ------------------------------------


def load(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def main() -> int:
    path = Path(RESULTS_FILE)
    if not path.is_file():
        print(f"File not found: {path}")
        return 1

    rows = load(path)
    if not rows:
        print("No results found.")
        return 1

    passed: list[dict] = []
    failed: list[dict] = []

    for r in rows:
        # A task is "passed" when all correctness cases pass
        # (pass_rate >= 1.0) or reward_score > 0.5.
        if r.get("pass_rate", 0) >= 1.0 or r.get("reward_score", 0) > 0.5:
            passed.append(r)
        else:
            failed.append(r)

    print("=" * 80)
    print(f"TOTAL: {len(rows)}  |  PASSED: {len(passed)}  |  FAILED: {len(failed)}")
    print("=" * 80)

    if passed:
        print("\n--- PASSED ---")
        for r in passed:
            sp = r.get("speedup_vs_torch")
            sp_str = f" speedup={sp:.2f}" if sp is not None else ""
            print(
                f"  {r['op_name']:<55s} "
                f"reward={r.get('reward_score', 0):.4f} "
                f"pass_rate={r.get('pass_rate', 0):.2f} "
                f"turns={r.get('num_turns', 0)}"
                f"{sp_str}"
            )

    if failed:
        print("\n--- FAILED ---")
        for r in failed:
            pr = r.get("pass_rate", 0)
            print(
                f"  {r['op_name']:<55s} "
                f"reward={r.get('reward_score', 0):.4f} "
                f"pass_rate={pr:.2f} "
                f"turns={r.get('num_turns', 0)} "
                f"exit={r.get('exit_reason', '?')}"
            )

    # Summary stats
    if failed:
        partial = [r for r in failed if r.get("pass_rate", 0) > 0]
        zero = [r for r in failed if r.get("pass_rate", 0) == 0]
        print(f"\n  Failed but partial progress (pass_rate > 0): {len(partial)}")
        if partial:
            avg_partial = sum(r["pass_rate"] for r in partial) / len(partial)
            print(f"  Average partial pass_rate: {avg_partial:.2f}")
        print(f"  Failed with zero progress:         {len(zero)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
