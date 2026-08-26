#!/usr/bin/env python3
"""Compact verifier summary for Claude Code rollouts.

The upstream verifier JSON can be very large. This helper keeps the raw artifact
unchanged while emitting a small, repair-oriented summary for the model.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


OUTPUT_MARKERS = (
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

REASON_MARKERS = (
    "coreDim is invalid",
    "Unsupported ptr type",
    "输出形状不一致",
    "输出不一致",
    "NaN 位置不匹配",
    "Inf 位置",
    "CompilationError",
    "RuntimeError",
    "AssertionError",
)


def _message(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("error_msg") or item.get("error") or item)
    return str(item)


def _error_type(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("error_type") or "")
    return ""


def _observed_output(item: Any) -> bool:
    text = _message(item)
    return any(marker in text for marker in OUTPUT_MARKERS)


def _compact_reason(text: str) -> str:
    for marker in REASON_MARKERS:
        if marker in text:
            idx = text.find(marker)
            tail = text[idx:]
            lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
            if not lines:
                return tail[:500]
            first = lines[0][:500]
            if len(lines) == 1:
                return first
            last = lines[-1][:500]
            if first == last:
                return first
            return f"{first} [...] {last}"[:800]
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped[:500]
    return ""


def summarize(data: dict[str, Any], exit_code: int, max_groups: int, max_examples: int) -> dict[str, Any]:
    total = int(data.get("total_cases") or 0)
    passed = int(data.get("passed_cases") or 0)
    failed = int(data.get("failed_cases") if data.get("failed_cases") is not None else max(total - passed, 0))
    failures = data.get("failures") if isinstance(data.get("failures"), list) else []
    output_observed = passed > 0 or any(_observed_output(item) for item in failures)
    compile_ok = output_observed
    ok = total > 0 and passed == total and failed == 0 and exit_code == 0

    groups: "OrderedDict[tuple[str, str], dict[str, Any]]" = OrderedDict()
    for item in failures:
        reason = _compact_reason(_message(item))
        key = (_error_type(item), reason)
        group = groups.setdefault(
            key,
            {
                "error_type": key[0],
                "reason": key[1],
                "count": 0,
                "examples": [],
            },
        )
        group["count"] += 1
        if len(group["examples"]) < max_examples and isinstance(item, dict):
            group["examples"].append(
                {
                    "case_idx": item.get("case_idx"),
                    "input_desc": item.get("input_desc"),
                }
            )

    return {
        "verified_success": ok,
        "verify_exit": exit_code,
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": failed,
        "pass_rate": round(passed / max(total, 1), 6) if total > 0 else 0.0,
        "compile_ok": compile_ok,
        "output_observed": output_observed,
        "error_groups": list(groups.values())[:max_groups],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("verify_result")
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--max-groups", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=2)
    parser.add_argument("--write-json")
    args = parser.parse_args()

    path = Path(args.verify_result)
    prefix = "[verifier-summary]"
    if not path.exists():
        print(f"{prefix} FAILED: {path} is missing after verify.py exit={args.exit_code}.")
        return 0

    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        print(f"{prefix} FAILED: cannot parse {path}: {exc}.")
        return 0

    summary = summarize(data, args.exit_code, args.max_groups, args.max_examples)
    if args.write_json:
        Path(args.write_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    status = "VERIFIED_SUCCESS" if summary["verified_success"] else "FAILED"
    print(
        f"{prefix} {status}: passed_cases={summary['passed_cases']}, "
        f"total_cases={summary['total_cases']}, failed_cases={summary['failed_cases']}, "
        f"pass_rate={summary['pass_rate']}, compile_ok={summary['compile_ok']}, "
        f"verify_exit={summary['verify_exit']}."
    )
    for idx, group in enumerate(summary["error_groups"], start=1):
        print(
            f"{prefix} group{idx}: count={group['count']} "
            f"type={group['error_type']} reason={group['reason']}"
        )
        for example in group["examples"]:
            print(f"{prefix}   example: case={example.get('case_idx')} input={example.get('input_desc')}")
    print(f"{prefix} raw artifact kept at {path}; use this compact summary for repair.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
