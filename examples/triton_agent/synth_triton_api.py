#!/usr/bin/env python3
"""Triton operator data synthesis with an external OpenAI-compatible API.

Connects to an external LLM API (e.g., GLM-5.1, GPT-4, etc.) and uses
OpenAICompatibleChatModel + AgentInteraction to generate Triton operator
trajectories in Docker sandboxes. No RL training is involved.

Usage:
    # Start from the repo root so that examples/ and uni_agent/ are importable.
    cd /path/to/uni-agent-claudecode

    python examples/triton_agent/synth_triton_api.py \
        --api-base-url https://open.bigmodel.cn/api/paas/v4 \
        --api-key YOUR_API_KEY \
        --model-name glm-5.1 \
        --dataset examples/triton_agent/benchmarks/NPUKernelBench \
        --levels level_1 \
        --max-rows 10 \
        --output /tmp/synth_api/results.jsonl
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

from uni_agent.interaction.model import OpenAICompatibleChatModel
from examples.triton_agent.synth_common import (
    load_tasks,
    run_one_task,
    save_results,
    DEFAULT_SANDBOX_IMAGE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # API connection
    parser.add_argument("--api-base-url", default="https://open.bigmodel.cn/api/paas/v4",
                        help="OpenAI-compatible API base URL (default: https://open.bigmodel.cn/api/paas/v4)")
    parser.add_argument("--api-key", default="EMPTY",
                        help="API key for authentication (default: EMPTY)")
    parser.add_argument("--model-name", default="glm-5.1",
                        help="Model name to send to the API (default: glm-5.1)")
    parser.add_argument("--timeout", type=float, default=300,
                        help="HTTP request timeout in seconds (default: 300)")

    # Task selection
    parser.add_argument("--dataset", default=str(SCRIPT_DIR / "benchmarks" / "NPUKernelBench"),
                        help="Path to NPUKernelBench dataset directory")
    parser.add_argument("--levels", default="level_1", help="Comma-separated levels, e.g. level_1,level_2 or all")
    parser.add_argument("--max-rows", type=int, default=None, help="Max number of tasks to run")
    parser.add_argument("--filter-mode", default="warmup", choices=["warmup", "all"],
                        help="Task filter mode: warmup excludes complex ops, all includes everything")

    # Agent loop
    parser.add_argument("--max-turns", type=int, default=50, help="Max agent interaction turns per task")
    parser.add_argument("--action-timeout", type=int, default=300, help="Timeout per tool execution (seconds)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Sampling top_p")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per API response")

    # Sandbox
    parser.add_argument("--sandbox-image", default=DEFAULT_SANDBOX_IMAGE, help="Docker sandbox image")
    parser.add_argument("--eval-device-ids", default="", help="Comma-separated NPU device IDs for evaluation")

    # Output
    parser.add_argument("--output", default="", help="Output JSONL path (default: auto-generated)")

    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> int:
    # 1. Create chat model connecting to the external API
    chat_model = OpenAICompatibleChatModel(
        base_url=args.api_base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        sampling_params={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
        timeout=args.timeout,
    )
    print(f"[synth] Connected to {args.api_base_url} model={args.model_name}")

    # 2. Load tasks
    tasks = load_tasks(
        dataset_path=args.dataset,
        levels=args.levels,
        max_rows=args.max_rows,
        filter_mode=args.filter_mode,
    )
    if not tasks:
        print("[synth] No tasks loaded. Check --dataset and --levels.")
        return 1

    # 3. Determine output path
    output_path = args.output
    if not output_path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = str(SCRIPT_DIR / "synth_api_runs" / ts / "results.jsonl")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 4. Run tasks
    results = []
    for idx, task in enumerate(tasks):
        print(f"\n[synth] === Task {idx + 1}/{len(tasks)}: {task['op_name']} ===")
        task_output_dir = str(Path(output_path).parent / "artifacts" / task["op_name"])
        Path(task_output_dir).mkdir(parents=True, exist_ok=True)

        result = await run_one_task(
            task,
            chat_model,
            output_dir=task_output_dir,
            max_turns=args.max_turns,
            action_timeout=args.action_timeout,
            sandbox_image=args.sandbox_image,
            device_ids=args.eval_device_ids,
        )
        results.append(result)
        save_results([result], output_path)

    # 5. Summary
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
