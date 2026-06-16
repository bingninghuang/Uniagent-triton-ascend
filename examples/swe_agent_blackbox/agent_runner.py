"""Uniagent runner for the blackbox SWE-agent recipe.

Uses white-box interaction components (AgentInteraction, OpenAICompatibleChatModel,
ToolsManager) with gateway-based LLM routing. Computes reward in-process and
passes it via the gateway's complete_session endpoint.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from typing import Any
from uuid import uuid4

from uni_agent.trainer.framework.types import SessionHandle, SessionRuntime
from uni_agent.interaction.env import AgentEnv, AgentEnvConfig
from uni_agent.interaction.interaction import AgentInteraction
from uni_agent.interaction.model import OpenAICompatibleChatModel
from uni_agent.interaction.tools_manager import ToolsManager, ToolsManagerConfig
from uni_agent.tools import ToolConfig

from examples.swe_agent_blackbox.reward import build_reward_context, evaluate_in_env

logger = logging.getLogger(__name__)
if os.environ.get("DEBUG_MODE"):
    logger.setLevel(logging.DEBUG)

_port_counter = 10000
_port_lock = threading.Lock()


def _allocate_port() -> int:
    global _port_counter
    with _port_lock:
        while True:
            port = _port_counter
            _port_counter += 1
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue


# =====================================================================
# Config helpers (uniagent-specific)
# =====================================================================


def load_agent_config(path: str) -> dict[str, Any]:
    """Load agent config from a YAML file. Returns the first entry."""
    import yaml

    with open(os.path.expanduser(path)) as f:
        configs = yaml.safe_load(f)
    if isinstance(configs, list):
        return configs[0] if configs else {}
    return configs or {}


def _create_agent_env(run_id: str, tools_kwargs: dict, agent_config: dict) -> AgentEnv:
    """Create AgentEnv from agent_config + per-sample tools_kwargs overrides."""
    env_config = dict(agent_config.get("env", {}))
    env_override = dict(tools_kwargs.get("env", {}))
    if env_override:
        deployment = dict(env_config.get("deployment", {}))
        deployment.update({k: env_override.pop(k) for k in ["image", "command"] if k in env_override})
        deployment.setdefault("type", "local")
        # R2E images keep swerex in a dedicated venv; use absolute path
        image = deployment.get("image", "")
        if "r2e" in image.lower():
            deployment["command"] = (
                "/opt/swerex-venv/bin/python3 -m swerex.server"
                " --auth-token {token}"
            )
        else:
            # SWE-bench images may lack swerex; install it before starting the server.
            # pip package is "swe-rex", module is "swerex".
            deployment["command"] = (
                "/usr/bin/python3.10 -m pip install -q swe-rex"
                " && exec /usr/bin/python3.10 -m swerex.server"
                " --auth-token {token}"
            )
        env_config["deployment"] = deployment
        env_config.update(env_override)
    # Pre-allocate a unique port to avoid _pick_free_port race conditions
    # when multiple sessions start concurrently.
    deployment = dict(env_config.get("deployment", {}))
    if "published_port" not in deployment:
        deployment["published_port"] = _allocate_port()
    env_config["deployment"] = deployment
    return AgentEnv(run_id=run_id, env_config=AgentEnvConfig(**env_config))


# =====================================================================
# Agent runner
# =====================================================================


async def swe_agent_runner(
    *,
    raw_prompt,
    session: SessionHandle,
    sample_index: int,
    session_runtime: SessionRuntime,
    tools_kwargs: dict | None = None,
    agent_config_path: str | None = None,
    **kwargs,
) -> None:
    """Run the uniagent SWE-agent through the gateway with in-process reward."""
    tools_kwargs = tools_kwargs or {}
    config_path = agent_config_path or tools_kwargs.get("agent_config_path")
    if not config_path:
        raise ValueError("agent_config_path is required for uni-agent runner (via parameter or tools_kwargs)")
    agent_config = load_agent_config(config_path)
    interaction_cfg = agent_config.get("interaction", {})

    messages = (
        list(raw_prompt) if isinstance(raw_prompt, list)
        else [{"role": "user", "content": str(raw_prompt)}]
    )

    env = _create_agent_env(f"swe_bb_{sample_index}_{uuid4().hex[:8]}", tools_kwargs, agent_config)
    metadata, eval_timeout = build_reward_context(tools_kwargs)

    try:
        logger.info("[sample %d] starting env, image=%s", sample_index, agent_config.get("env", {}).get("deployment", {}).get("image", "N/A"))
        t0 = time.perf_counter()
        await env.start()
        logger.info("[sample %d] env started (%.1fs)", sample_index, time.perf_counter() - t0)

        model = OpenAICompatibleChatModel(
            base_url=session.base_url,
            api_key="not-needed",
            model_name="default",
        )

        tools_config = agent_config.get("tools", [])
        parser_name = agent_config.get("tool_parser") or tools_kwargs.get("tool_parser") or "qwen3_coder"
        tools_manager = ToolsManager(
            tools_manager_config=ToolsManagerConfig(
                tools=[ToolConfig(name=t["name"]) for t in tools_config],
                parser=parser_name,
            ),
        )
        model.set_tools_schemas(tools_manager.tools_schemas)

        env_max_turns = int(os.environ["SWE_AGENT_MAX_TURNS"]) if "SWE_AGENT_MAX_TURNS" in os.environ else None
        interaction = AgentInteraction(
            run_id=f"swe_bb_{sample_index}",
            env=env,
            model=model,
            tools_manager=tools_manager,
            messages=messages,
            action_timeout=interaction_cfg.get("action_timeout", 300),
            timeout_budget=interaction_cfg.get("timeout_budget", 300),
            max_turns=env_max_turns or interaction_cfg.get("max_turns", 100),
        )

        await env.install_tools(tools_manager.tools)
        logger.info("[sample %d] running agent, max_turns=%d", sample_index, env_max_turns or interaction_cfg.get("max_turns", 100))
        t0 = time.perf_counter()
        result = await interaction.run()
        trajectory = result.get("trajectory", [])
        logger.info("[sample %d] agent finished, %d steps (%.1fs)", sample_index, len(trajectory), time.perf_counter() - t0)

        # Evaluate reward in the same Docker env
        logger.info("[sample %d] evaluating reward, data_source=%s", sample_index, metadata["data_source"])
        t0 = time.perf_counter()
        score, eval_result = await evaluate_in_env(env, metadata, eval_timeout)
        logger.info("[sample %d] reward done, score=%s, resolved=%s (%.1fs)", sample_index, score, eval_result.get("resolved"), time.perf_counter() - t0)
        logger.info("[sample %d] reward done, score=%s, resolved=%s", sample_index, score, eval_result.get("resolved"))

        # Signal completion with reward_info
        reward_info = {"reward_score": score, **eval_result}
        await session_runtime.complete_session(session.session_id, reward_info=reward_info)

    except Exception as e:
        logger.warning("Agent runner failed for sample %d: %s", sample_index, e)
        raise
    finally:
        try:
            await env.close()
        except Exception:
            pass
