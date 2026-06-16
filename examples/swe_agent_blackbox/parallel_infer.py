"""Parallel inference runner for the blackbox SWE-agent recipe (v2).

Creates an LLM server, GatewayServingRuntime, and SWEAgentFramework,
then runs agent sessions in parallel and reports resolve rate.

Usage (CLI):
    python examples/swe_agent_blackbox/parallel_infer.py \
        --model-path ~/models/Qwen3-Coder-30B-A3B-Instruct \
        --data-path ~/data/swe_agent/swe_bench_verified.parquet \
        --max-samples 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from typing import Any
from uuid import uuid4

import numpy as np
import ray

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.utils import hf_tokenizer
from verl.utils.transferqueue_utils import tq as _tq_mock
from verl.workers.rollout.llm_server import LLMServerManager

from uni_agent.trainer.gateway.runtime import GatewayServingRuntime

from examples.swe_agent_blackbox.framework import SWEAgentFramework
from examples.swe_agent_blackbox.agent_runner import swe_agent_runner
from examples.swe_agent_blackbox.mini_swe_agent_runner import mini_swe_agent_runner

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=os.getenv("VERL_LOGGING_LEVEL", "INFO"),
    force=True,
)
logger = logging.getLogger(__name__)


# =====================================================================
# Dataset loading (inlined from dataset.py — only used here)
# =====================================================================


def _remap_image_to_local(image_name: str) -> str:
    parts = image_name.split("/")
    if len(parts) > 1 and "." in parts[0]:
        basename = parts[-1]
    else:
        basename = image_name
    basename = basename.replace("_1776_", "__")
    if ":" in basename:
        basename = basename.rsplit(":", 1)[0]
    return f"{basename}:latest"


def _remap_sample_images(sample: dict[str, Any]) -> dict[str, Any]:
    extra_info = sample.get("extra_info")
    if not extra_info:
        return sample
    tools_kwargs = extra_info.get("tools_kwargs", {})
    env = tools_kwargs.get("env", {})
    image = env.get("image")
    if not image:
        return sample
    local_image = _remap_image_to_local(image)
    if local_image != image:
        logger.debug("Remapping image: %s -> %s", image, local_image)
        env["image"] = local_image
    return sample


def _inject_reward_fields(sample: dict[str, Any]) -> None:
    """Inject verl-standard data_source and reward_model from extra_info.tools_kwargs.reward."""
    extra_info = sample.get("extra_info", {})
    tools_kwargs = extra_info.get("tools_kwargs", {})
    reward_config = tools_kwargs.get("reward", {})
    sample.setdefault("data_source", reward_config.get("name", "unknown"))
    sample.setdefault("reward_model", {"ground_truth": {}})


def load_swe_dataset(data_path: str | list[str], max_samples: int = -1) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    if isinstance(data_path, list):
        paths = [os.path.expanduser(p) for p in data_path]
    else:
        paths = os.path.expanduser(data_path)

    logger.info("Loading dataset from: %s", data_path)
    if isinstance(paths, list):
        import pyarrow as pa
        tables = [pq.read_table(p) for p in paths]
        table = pa.concat_tables(tables)
    else:
        table = pq.read_table(paths)
    samples = table.to_pylist()

    for i, sample in enumerate(samples):
        samples[i] = _remap_sample_images(sample)
        _inject_reward_fields(samples[i])

    if max_samples > 0:
        samples = samples[:max_samples]
        logger.info("Using first %d samples (max_samples=%d)", len(samples), max_samples)

    logger.info("Loaded %d samples from %s", len(samples), data_path)
    return samples


class _MockReplayBuffer:
    """Minimal replay buffer for inference mode (no actual training)."""

    def add(self, partition_id, items):
        pass


def run_inference(
    *,
    model_path: str,
    data_path: str,
    prompt_length: int = 4096,
    response_length: int = 65536,
    temperature: float = 0.8,
    top_p: float = 0.9,
    n: int = 1,
    max_samples: int = -1,
    engine: str = "vllm",
    nnodes: int = 1,
    n_gpus_per_node: int = 8,
    tensor_parallel_size: int = 4,
    gateway_count: int = 1,
    completion_timeout: float = 600.0,
    tool_parser: str | None = None,
    agent_config_path: str | None = None,
    runner: str = "uniagent",
) -> dict[str, Any]:
    """Run parallel SWE-agent inference using the blackbox framework."""
    if runner == "mini_swe":
        _agent_runner = mini_swe_agent_runner
    else:
        _agent_runner = swe_agent_runner

    if not ray.is_initialized():
        ray.init()

    # 1. Init Hydra config
    config = _init_hydra_config(
        model_path=model_path,
        engine=engine,
        prompt_length=prompt_length,
        response_length=response_length,
        temperature=temperature,
        top_p=top_p,
        n=n,
        nnodes=nnodes,
        n_gpus_per_node=n_gpus_per_node,
        tensor_parallel_size=tensor_parallel_size,
    )

    # 2. Load dataset
    samples = load_swe_dataset(data_path, max_samples=max_samples)
    logger.info("Loaded %d samples, %d rollout(s) each", len(samples), n)

    if not samples:
        raise ValueError("No samples to process")

    # 3. Create LLM server
    logger.info("Initializing LLM server manager...")
    llm_server_manager = LLMServerManager.create(config=config)

    # 4. Create GatewayServingRuntime
    logger.info("Using tool_parser=%r", tool_parser)

    llm_client = llm_server_manager.get_client()
    gateway_actor_kwargs = {
        "tokenizer": hf_tokenizer(os.path.expanduser(model_path)),
        "base_sampling_params": {"temperature": temperature, "top_p": top_p, "max_tokens": response_length},
    }
    if tool_parser:
        gateway_actor_kwargs["tool_parser_name"] = tool_parser

    gateway_runtime = GatewayServingRuntime(
        llm_client=llm_client,
        gateway_count=gateway_count,
        gateway_actor_kwargs=gateway_actor_kwargs,
    )

    # 5. Create RewardLoopWorker for compute_score
    from verl.experimental.reward_loop.reward_loop import RewardLoopWorker
    reward_worker = ray.remote(RewardLoopWorker).remote(config, None)

    # 6. Create framework
    framework = SWEAgentFramework(
        session_runtime=gateway_runtime,
        agent_runner=_agent_runner,
        replay_buffer=_MockReplayBuffer(),
        rollout_config={"n": n, "val_kwargs": {"n": n}},
        completion_timeout=completion_timeout,
        wait_for_completion_after_agent_run=True,
        max_concurrent_sessions=2,
        reward_loop_worker_handles=[reward_worker],
    )

    # 6. Build batch data and run
    _tools_kwargs_list = []
    for sample in samples:
        tk = (sample.get("extra_info") or {}).get("tools_kwargs", {})
        if agent_config_path:
            tk["agent_config_path"] = agent_config_path
        tk["model_path"] = os.path.expanduser(model_path)
        _tools_kwargs_list.append(tk)

    from tensordict import TensorDict
    from verl.utils import tensordict_utils as _tu

    raw_prompts = [sample["prompt"] for sample in samples]
    uids = [str(uuid4()) for _ in samples]
    td = TensorDict({"uid": uids, "global_steps": [0] * len(samples)}, batch_size=[len(samples)])
    _tu.assign_non_tensor_stack(td, "raw_prompt", raw_prompts)
    _tu.assign_non_tensor_stack(td, "tools_kwargs", _tools_kwargs_list)
    _tu.assign_non_tensor_stack(td, "data_source", [sample["data_source"] for sample in samples])
    _tu.assign_non_tensor_stack(td, "reward_model", [sample["reward_model"] for sample in samples])

    batch = DataProto(batch=td, meta_info={}).repeat(n)

    size_divisor = gateway_count
    batch_padded, pad_size = pad_dataproto_to_divisor(batch, size_divisor)
    logger.info("Starting %d agent session(s)...", len(batch_padded))

    _tq_store: dict[str, Any] = {}

    async def _dummy_kv_put(key, partition_id=None, tag=None, **kwargs):
        _tq_store[key] = tag

    async def _dummy_kv_batch_put(keys=None, fields=None, tags=None, partition_id=None, **kwargs):
        for i, key in enumerate(keys):
            _tq_store[key] = {"fields": fields, "tag": tags[i] if tags else None}

    _tq_mock.async_kv_put = _dummy_kv_put
    _tq_mock.async_kv_batch_put = _dummy_kv_batch_put

    async def _generate():
        return await framework.generate_sequences(batch_padded.batch)

    try:
        stats = asyncio.run(_generate())
    except RuntimeError as e:
        logger.warning("generate_sequences failed: %s", e)
        stats = {}

    # 7. Collect scores
    uid_to_sample_idx = {uid: i for i, uid in enumerate(uids)}
    per_sample_scores = [0.0] * len(samples)
    sample_trajectory_counts = [0] * len(samples)
    for key, value in _tq_store.items():
        if not isinstance(value, dict) or "fields" not in value:
            continue
        fields = value["fields"]
        rm_scores = fields.get("rm_scores", None)
        if rm_scores is None:
            continue
        # Key format: {uid}_{session_index}_{index}
        uid = key.rsplit("_", 2)[0]
        sample_idx = uid_to_sample_idx.get(uid)
        if sample_idx is None:
            continue
        score = float(rm_scores.float()[-1, -1].item())
        per_sample_scores[sample_idx] += score
        sample_trajectory_counts[sample_idx] += 1

    for i in range(len(samples)):
        if sample_trajectory_counts[i] > 0:
            per_sample_scores[i] /= sample_trajectory_counts[i]

    resolved_count = sum(1 for s in per_sample_scores if s > 0)
    overall_mean = float(np.mean(per_sample_scores)) if per_sample_scores else 0.0
    logger.info(
        "Resolved %d / %d samples (%.2f%%), mean score: %.4f",
        resolved_count, len(samples), 100.0 * resolved_count / max(len(samples), 1), overall_mean,
    )

    # 8. Cleanup
    asyncio.run(gateway_runtime.shutdown())

    return {
        "stats": stats,
        "mean_score": overall_mean,
        "per_sample_scores": per_sample_scores,
    }


# =====================================================================
# Helpers
# =====================================================================


def _init_hydra_config(
    *,
    model_path: str,
    engine: str,
    prompt_length: int,
    response_length: int,
    temperature: float,
    top_p: float,
    n: int,
    nnodes: int,
    n_gpus_per_node: int,
    tensor_parallel_size: int,
) -> Any:
    """Initialize Hydra config with rollout/model settings."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = os.path.abspath("examples/swe_agent_blackbox/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="parallel_infer")

    config.actor_rollout_ref.model.path = os.path.expanduser(model_path)
    config.actor_rollout_ref.rollout.name = engine
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.prompt_length = prompt_length
    config.actor_rollout_ref.rollout.response_length = response_length
    config.actor_rollout_ref.rollout.max_model_len = prompt_length + response_length + 1024
    config.actor_rollout_ref.rollout.n = n
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = tensor_parallel_size
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.5
    config.actor_rollout_ref.rollout.temperature = temperature
    config.actor_rollout_ref.rollout.top_p = top_p
    config.actor_rollout_ref.rollout.val_kwargs.temperature = temperature
    config.actor_rollout_ref.rollout.val_kwargs.top_p = top_p
    config.actor_rollout_ref.rollout.calculate_log_probs = True
    config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns = 100
    config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls = 1
    config.actor_rollout_ref.rollout.nnodes = nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = n_gpus_per_node
    config.trainer.nnodes = nnodes
    config.trainer.n_gpus_per_node = n_gpus_per_node

    config.reward.custom_reward_function.path = "pkg://examples.swe_agent_blackbox.reward"
    config.reward.custom_reward_function.name = "compute_score"
    config.reward.num_workers = 1

    OmegaConf.set_struct(config.actor_rollout_ref.rollout, False)
    config.actor_rollout_ref.rollout.enable_sleep_mode = False
    OmegaConf.set_struct(config.actor_rollout_ref.rollout, True)
    return config


# =====================================================================
# CLI entry point
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description="SWE-Agent Blackbox Parallel Inference")
    parser.add_argument("--data-path", type=str, default="~/data/swe_agent/swe_bench_verified.parquet")
    parser.add_argument("--model-path", "--model", type=str, default="~/models/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--prompt-length", type=int, default=4096)
    parser.add_argument("--response-length", type=int, default=65536)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--engine", type=str, default="vllm", choices=["vllm", "sglang"])
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--n-gpus-per-node", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=4)
    parser.add_argument("--tool-parser", type=str, default="qwen3_coder")
    parser.add_argument(
        "--runner", type=str, default="uniagent", choices=["uniagent", "mini_swe"],
        help="Agent runner: 'uniagent' or 'mini_swe'.",
    )
    parser.add_argument(
        "--agent-config-path", type=str,
        default="examples/swe_agent_blackbox/config/agent_config.yaml",
        help="Path to agent config YAML.",
    )
    args = parser.parse_args()

    os.environ["SWE_AGENT_MAX_TURNS"] = str(args.max_turns)

    run_inference(
        model_path=args.model_path,
        data_path=args.data_path,
        prompt_length=args.prompt_length,
        response_length=args.response_length,
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n,
        max_samples=args.max_samples,
        engine=args.engine,
        nnodes=args.nnodes,
        n_gpus_per_node=args.n_gpus_per_node,
        tensor_parallel_size=args.tensor_parallel_size,
        tool_parser=args.tool_parser,
        agent_config_path=args.agent_config_path,
        runner=args.runner,
    )


if __name__ == "__main__":
    main()
