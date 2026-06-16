"""Framework subclass for Triton Claude Code blackbox training."""

from __future__ import annotations


try:
    from uni_agent.trainer.framework.framework import OpenAICompatibleAgentFramework
except ImportError:  # pragma: no cover - available only in blackbox-enabled branches
    OpenAICompatibleAgentFramework = object  # type: ignore[assignment]


class TritonClaudeCodeFramework(OpenAICompatibleAgentFramework):  # type: ignore[misc]
    """Inject runner-side reward_info into reward-worker extra_info.

    The Claude Code runner computes the expensive Triton reward in the sandbox
    and calls ``complete_session(..., reward_info=...)``. The base blackbox
    framework then invokes the configured reward worker; by merging
    ``reward_info`` into ``extra_info``, ``examples.triton_agent.reward`` can
    return that score without running evaluation twice.
    """

    async def _score_trajectories(self, session_trajectories, sample_fields):
        if session_trajectories and session_trajectories[-1].reward_info:
            reward_info = dict(session_trajectories[-1].reward_info)
            extra_info = dict(sample_fields.get("extra_info") or {})
            sample_fields = {**sample_fields, "extra_info": {**extra_info, **reward_info}}
        return await super()._score_trajectories(session_trajectories, sample_fields)
