#!/usr/bin/env python3
"""Triton operator data synthesis with local model weights + vLLM server.

Launches a vLLM inference server from local model weights, then uses
OpenAICompatibleChatModel + AgentInteraction to generate Triton operator
trajectories in Docker sandboxes. No RL training is involved.

Usage:
    # Start from the repo root so that examples/ and uni_agent/ are importable.
    cd /path/to/UniAgent-Triton-Ascend

    python examples/triton_agent/synth_triton_local.py \
        --model-path /path/to/Qwen3-Coder-30B-A3B-Instruct \
        --tp-size 8 \
        --dataset examples/triton_agent/benchmarks/NPUKernelBench \
        --levels level_1 \
        --max-rows 10 \
        --output /tmp/synth_local/results.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path so examples.triton_agent.* resolves.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import re as _re

from uni_agent.interaction.model import OpenAICompatibleChatModel
from examples.triton_agent.synth_common import (
    load_tasks,
    run_one_task_hard,
    save_results,
    DEFAULT_SANDBOX_IMAGE,
)


def parse_completed_op_names_from_log(log_path: str) -> set[str]:
    """Scan a previous log file and return the set of op_names that completed.

    A task is considered completed when the log contains a summary line like:
        [synth] op=kernelbench_l1_1_1 reward=... pass_rate=...
    Only the op_name portion (e.g. kernelbench_l1_1_1) is returned.
    """
    completed = set()
    pattern = _re.compile(r'\[synth\] op=(\S+)\s+reward=')
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    completed.add(m.group(1))
    except FileNotFoundError:
        print(f"[synth] Resume log not found: {log_path}, starting from scratch")
    return completed


def sort_tasks_by_numeric_id(tasks: list[dict]) -> list[dict]:
    """Sort tasks by numeric problem_id so 1,2,3...10,11 instead of 1,10,11...2,20."""
    def _sort_key(task):
        pid = task.get("problem_id", 0)
        if isinstance(pid, int):
            return (task.get("level", ""), pid)
        try:
            return (task.get("level", ""), int(pid))
        except (TypeError, ValueError):
            return (task.get("level", ""), 0, str(pid))
    return sorted(tasks, key=_sort_key)


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------


def wait_for_vllm(host: str = "127.0.0.1", port: int = 5000, timeout: int = 300) -> bool:
    """Wait until the vLLM server is healthy. Returns True if ready."""
    import urllib.request
    import urllib.error

    url = f"http://{host}:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print(f"[vllm] Server ready at {url}")
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(5)
    print(f"[vllm] Server did not become ready within {timeout}s")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/path/to/Qwen3-Coder-30B-A3B-Instruct",
                        help="Path to local model weights (default: /path/to/Qwen3-Coder-30B-A3B-Instruct)")
    parser.add_argument("--tp-size", type=int, default=8, help="Tensor parallel size (default: 8)")
    parser.add_argument("--max-model-len", type=int, default=40960, help="Max model context length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80, help="vLLM GPU memory utilization")
    parser.add_argument("--port", type=int, default=7778, help="vLLM server port (default: 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="vLLM server host (default: 127.0.0.1)")

    parser.add_argument("--dataset", default=str(SCRIPT_DIR / "benchmarks" / "NPUKernelBench"),
                        help="Path to NPUKernelBench dataset directory")
    parser.add_argument("--levels", default="level_1", help="Comma-separated levels, e.g. level_1,level_2 or all")
    parser.add_argument("--max-rows", type=int, default=None, help="Max number of tasks to run")
    parser.add_argument("--filter-mode", default="warmup", choices=["warmup", "all"],
                        help="Task filter mode: warmup excludes complex ops, all includes everything")
    parser.add_argument("--op-names", default="",
                        help="Comma-separated op_names to run (after level/filter/max-rows applied). "
                             "When set, only these operators are executed. Used by the concurrent "
                             "scheduler to dispatch individual ops to specific cards.")

    parser.add_argument("--max-turns", type=int, default=50, help="Max agent interaction turns per task")
    parser.add_argument("--action-timeout", type=int, default=300, help="Timeout per tool execution (seconds)")

    parser.add_argument("--sandbox-image", default=DEFAULT_SANDBOX_IMAGE, help="Docker sandbox image")
    parser.add_argument("--eval-device-ids", default="", help="Comma-separated NPU device IDs for evaluation")

    parser.add_argument("--output", default="", help="Output JSONL path (default: auto-generated)")
    parser.add_argument("--served-model-name", default="triton-synth", help="Model name advertised by vLLM")

    parser.add_argument("--resume-from-log", default="", metavar="LOG_PATH",
                        help="Path to a previous run's log file. Completed operators are skipped, "
                             "results are appended to the same --output file.")
    parser.add_argument("--numeric-sort", action="store_true",
                        help="Sort tasks by numeric problem_id (1,2,...,10) instead of "
                             "alphabetical (1,10,11,...,2)")
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> int:
    # 1. Wait for server
    if not wait_for_vllm(host=args.host, port=args.port, timeout=300):
        print("[vllm] Failed to start. Check logs above.")
        return 1

    # 3. Create chat model
    chat_model = OpenAICompatibleChatModel(
        base_url=f"http://{args.host}:{args.port}/v1",
        api_key="EMPTY",
        model_name=args.served_model_name,
        sampling_params={
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 4096,
        },
        timeout=600,
    )

    # 4. Load tasks
    tasks = load_tasks(
        dataset_path=args.dataset,
        levels=args.levels,
        max_rows=args.max_rows,
        filter_mode=args.filter_mode,
    )
    if not tasks:
        print("[synth] No tasks loaded. Check --dataset and --levels.")
        return 1

    # 4a. Optionally sort by numeric problem_id
    if args.numeric_sort:
        tasks = sort_tasks_by_numeric_id(tasks)
        print(f"[synth] Tasks sorted by numeric problem_id")

    # 4b. Optionally restrict to an explicit op-name allowlist (concurrent dispatch)
    if args.op_names:
        wanted = {name.strip() for name in args.op_names.split(",") if name.strip()}
        before = len(tasks)
        tasks = [t for t in tasks if t["op_name"] in wanted]
        missing = wanted - {t["op_name"] for t in tasks}
        print(f"[synth] Op filter: {before} -> {len(tasks)} tasks")
        if missing:
            print(f"[synth] WARNING: requested op_names not found in dataset: {sorted(missing)}")
        if not tasks:
            print("[synth] No tasks left after op_names filter.")
            return 1

    # 4c. Optionally skip already-completed tasks from a previous run
    completed_ops: set[str] = set()
    if args.resume_from_log:
        completed_ops = parse_completed_op_names_from_log(args.resume_from_log)
        if completed_ops:
            before = len(tasks)
            tasks = [t for t in tasks if t["op_name"] not in completed_ops]
            print(f"[synth] Resume: {len(completed_ops)} completed ops found in log, "
                  f"skipping {before - len(tasks)}, running {len(tasks)} remaining")
        else:
            print(f"[synth] Resume: no completed ops found in {args.resume_from_log}, running all")

    # 5. Determine output path
    output_path = args.output
    if not output_path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = str(SCRIPT_DIR / "synth_local_runs" / ts / "results.jsonl")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 6. Run tasks
    results = []
    for idx, task in enumerate(tasks):
        print(f"\n[synth] === Task {idx + 1}/{len(tasks)}: {task['op_name']} ===")
        task_output_dir = str(Path(output_path).parent / "artifacts" / task["op_name"])
        Path(task_output_dir).mkdir(parents=True, exist_ok=True)

        result = await run_one_task_hard(
            task,
            chat_model,
            output_dir=task_output_dir,
            max_turns=args.max_turns,
            action_timeout=args.action_timeout,
            device_ids=args.eval_device_ids,
        )
        results.append(result)
        save_results([result], output_path)

    # 7. Summary
    total = len(results)
    success = sum(1 for r in results if r.get("reward_score", 0) > 0.5)
    avg_reward = sum(r.get("reward_score", 0) for r in results) / max(total, 1)
    print(f"\n[synth] Done. total={total} success={success} avg_reward={avg_reward:.4f}")
    print(f"[synth] Results saved to {output_path}")
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())