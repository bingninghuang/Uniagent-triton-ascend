#!/usr/bin/env python3
"""Triton operator data synthesis with local model weights + vLLM server.

Launches a vLLM inference server from local model weights, then uses
OpenAICompatibleChatModel + AgentInteraction to generate Triton operator
trajectories in Docker sandboxes. No RL training is involved.

Usage:
    # Start from the repo root so that examples/ and uni_agent/ are importable.
    cd /path/to/uni-agent-claudecode

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
import os
import shutil
import signal
import subprocess
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


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------


def launch_vllm_server(
    model_path: str,
    served_model_name: str = "triton-synth",
    tp_size: int = 8,
    max_model_len: int = 40960,
    gpu_memory_utilization: float = 0.80,
    port: int = 5000,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    """Launch a vLLM OpenAI-compatible server as a subprocess."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", served_model_name,
        "--port", str(port),
        "--tensor-parallel-size", str(tp_size),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--trust-remote-code",
        "--dtype", "bfloat16",
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"[vllm] Launching: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return proc


def wait_for_vllm(port: int = 5000, timeout: int = 300) -> bool:
    """Wait until the vLLM server is healthy. Returns True if ready."""
    import urllib.request
    import urllib.error

    url = f"http://127.0.0.1:{port}/v1/models"
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
    parser.add_argument("--port", type=int, default=5000, help="vLLM server port (default: 5000)")

    parser.add_argument("--dataset", default=str(SCRIPT_DIR / "benchmarks" / "NPUKernelBench"),
                        help="Path to NPUKernelBench dataset directory")
    parser.add_argument("--levels", default="level_1", help="Comma-separated levels, e.g. level_1,level_2 or all")
    parser.add_argument("--max-rows", type=int, default=None, help="Max number of tasks to run")
    parser.add_argument("--filter-mode", default="warmup", choices=["warmup", "all"],
                        help="Task filter mode: warmup excludes complex ops, all includes everything")

    parser.add_argument("--max-turns", type=int, default=50, help="Max agent interaction turns per task")
    parser.add_argument("--action-timeout", type=int, default=300, help="Timeout per tool execution (seconds)")

    parser.add_argument("--sandbox-image", default=DEFAULT_SANDBOX_IMAGE, help="Docker sandbox image")
    parser.add_argument("--eval-device-ids", default="", help="Comma-separated NPU device IDs for evaluation")

    parser.add_argument("--output", default="", help="Output JSONL path (default: auto-generated)")
    parser.add_argument("--served-model-name", default="triton-synth", help="Model name advertised by vLLM")
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> int:
    # 1. Launch vLLM server
    proc = launch_vllm_server(
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        tp_size=args.tp_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        port=args.port,
    )

    try:
        # 2. Wait for server
        if not wait_for_vllm(port=args.port, timeout=300):
            print("[vllm] Failed to start. Check logs above.")
            return 1

        # 3. Create chat model
        chat_model = OpenAICompatibleChatModel(
            base_url=f"http://127.0.0.1:{args.port}/v1",
            api_key="EMPTY",
            model_name=args.served_model_name,
            sampling_params={
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": 4096,
            },
            timeout=300,
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

            result = await run_one_task(
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

    finally:
        # 8. Shutdown vLLM
        print("[vllm] Shutting down server...")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def main() -> int:
    args = parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
