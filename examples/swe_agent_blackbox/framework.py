"""SWE-agent specific framework subclass.

Injects reward_info (from agent_runner's complete_session call)
into sample_fields["extra_info"] so the reward worker's
compute_score can access it via extra_info.
"""

from uni_agent.trainer.framework.framework import OpenAICompatibleAgentFramework


class SWEAgentFramework(OpenAICompatibleAgentFramework):

    async def _score_trajectories(self, session_trajectories, sample_fields):
        if session_trajectories and session_trajectories[-1].reward_info:
            reward_info = session_trajectories[-1].reward_info
            extra_info = dict(sample_fields.get("extra_info") or {})
            sample_fields = {**sample_fields, "extra_info": {**extra_info, **reward_info}}
        return await super()._score_trajectories(session_trajectories, sample_fields)
