"""Reward spec for Triton/Ascend KernelBench workspaces."""

from __future__ import annotations

from typing import Any

from uni_agent.async_logging import get_logger
from uni_agent.interaction import AgentEnv
from uni_agent.reward.base import AbstractRewardSpec
from uni_agent.reward.registry import register_reward_spec
from uni_agent.utils import auto_await

from examples.triton_agent.reward import DEFAULT_WORKSPACE_DIR, evaluate_triton_workspace


@register_reward_spec("triton_kernelbench")
class TritonKernelBenchRewardSpec(AbstractRewardSpec):
    def __init__(
        self,
        *,
        run_id: str,
        metadata: dict[str, Any],
        env: AgentEnv,
        eval_timeout: int | float = 600,
        workspace_dir: str = DEFAULT_WORKSPACE_DIR,
    ) -> None:
        self.run_id = run_id
        self.metadata = metadata
        self.env = env
        self.eval_timeout = float(eval_timeout)
        self.workspace_dir = workspace_dir
        self.logger = get_logger("reward_spec", run_id=run_id)

    @auto_await
    async def compute_reward(self, **kwargs) -> tuple[float, dict]:
        del kwargs
        try:
            return await evaluate_triton_workspace(
                self.env,
                self.metadata,
                workspace_dir=self.workspace_dir,
            )
        except Exception as exc:
            self.logger.error(f"Failed to evaluate Triton KernelBench reward: {exc}")
            return 0.0, {"eval_completed": False, "reward": 0.0, "error": str(exc)}
