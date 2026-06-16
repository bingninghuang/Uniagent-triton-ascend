"""Mini-swe-agent runner for the blackbox SWE-agent recipe.

Uses third-party minisweagent components (DefaultAgent, DockerEnvironment,
LitellmModel) with gateway-based LLM routing. Computes reward in-process
and passes it via the gateway's complete_session endpoint.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from pathlib import Path
from typing import Any

from uni_agent.trainer.framework.types import SessionHandle, SessionRuntime

from examples.swe_agent_blackbox.reward import build_reward_context, evaluate_in_env

logger = logging.getLogger(__name__)
if os.environ.get("DEBUG_MODE"):
    logger.setLevel(logging.DEBUG)

try:
    import shlex
    import subprocess
    import uuid as _uuid

    from minisweagent.agents.default import DefaultAgent
    from minisweagent.config import builtin_config_dir, get_config_from_spec
    from minisweagent.environments.docker import DockerEnvironment
    from minisweagent.models.litellm_model import LitellmModel

    _SWEBENCH_CONFIG = get_config_from_spec(str(builtin_config_dir / "benchmarks" / "swebench.yaml"))
except ImportError:
    _SWEBENCH_CONFIG = None


class _FixedCmdDockerEnvironment(DockerEnvironment):
    """DockerEnvironment subclass that adapts CMD to the image's ENTRYPOINT.

    Background:
        DockerEnvironment._start_container hardcodes CMD as ["sleep", timeout].
        This works for images whose ENTRYPOINT exec's CMD directly (e.g.
        nvidia_entrypoint.sh), but fails for images whose ENTRYPOINT is
        "/bin/bash" because "/bin/bash sleep 2h" causes bash to interpret
        the /usr/bin/sleep ELF binary as a shell script and immediately exit
        with "cannot execute binary file" (exit 126).

    Fix:
        Inspect the image's ENTRYPOINT at startup.  If it ends with "bash",
        use ["-lc", "sleep <timeout>"] as CMD (matching the image's intended
        format).  Otherwise, use the original ["sleep", "<timeout>"].
    """

    def _start_container(self):
        container_name = f"minisweagent-{_uuid.uuid4().hex[:8]}"
        entrypoint = self._detect_entrypoint()
        if entrypoint and entrypoint.endswith("bash"):
            cmd_suffix = ["-lc", f"sleep {self.config.container_timeout}"]
        else:
            cmd_suffix = ["sleep", self.config.container_timeout]
        cmd = [
            self.config.executable, "run", "-d",
            "--name", container_name,
            "-w", self.config.cwd,
            *self.config.run_args,
            self.config.image,
            *cmd_suffix,
        ]
        self.logger.debug("Starting container with command: %s", shlex.join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=self.config.pull_timeout, check=True,
        )
        self.logger.info("Started container %s with ID %s", container_name, result.stdout.strip())
        self.container_id = result.stdout.strip()

    def _detect_entrypoint(self) -> str | None:
        """Detect the image's ENTRYPOINT via docker inspect."""
        try:
            result = subprocess.run(
                [self.config.executable, "inspect", self.config.image, "--format", "{{.Config.Entrypoint}}"],
                capture_output=True, text=True, timeout=30,
            )
            ep = result.stdout.strip().strip("[]\"")
            return ep if ep else None
        except Exception:
            return None


# =====================================================================
# DockerEnvForReward: adapts sync DockerEnvironment to async interface
# =====================================================================


class DockerEnvForReward:
    """Adapts minisweagent's sync DockerEnvironment to async interface for reward specs."""

    def __init__(self, docker_env):
        self._env = docker_env

    async def communicate(self, input: str, timeout=60, check="ignore", error_msg="Command failed") -> str:
        result = await asyncio.to_thread(self._env.execute, {"command": input}, timeout=int(timeout))
        output, rc = result.get("output", ""), result.get("returncode", 0)
        if check == "raise" and rc != 0:
            raise RuntimeError(f"{error_msg}: {output[:200]}")
        return output

    async def write_file(self, path: str | Path, content: str) -> None:
        encoded = base64.b64encode(content.encode()).decode()
        await self.communicate(f"echo {encoded} | base64 -d > {path}", check="raise", error_msg=f"write {path}")

    async def read_file(self, path: str | Path, **_) -> str:
        return await self.communicate(f"cat {path}")


# =====================================================================
# Agent runner
# =====================================================================


async def mini_swe_agent_runner(
    *,
    raw_prompt,
    session: SessionHandle,
    sample_index: int,
    session_runtime: SessionRuntime,
    tools_kwargs: dict | None = None,
    **kwargs,
) -> None:
    """Run mini-swe-agent's DefaultAgent through the gateway with in-process reward."""
    if _SWEBENCH_CONFIG is None:
        raise ImportError("minisweagent is required for mini_swe_agent_runner")

    tools_kwargs = tools_kwargs or {}
    logger.info("mini_swe_agent_runner called, sample_index=%d", sample_index)

    # 1. Extract task text
    task = raw_prompt if isinstance(raw_prompt, str) else next(
        (m["content"] for m in raw_prompt if isinstance(m, dict) and m.get("role") == "user"),
        str(raw_prompt),
    )
    logger.info("task extracted, %d chars", len(task))

    # 2. Create DockerEnvironment
    env_config = tools_kwargs.get("env", {})
    image = env_config.get("image", "")
    if not image:
        raise ValueError(f"No Docker image found in tools_kwargs.env for sample {sample_index}")

    env_cfg = dict(_SWEBENCH_CONFIG.get("environment", {}))
    env_cfg.pop("environment_class", None)
    env_cfg["image"] = image
    env_cfg["container_timeout"] = "2h"
    env_cfg.setdefault("env", {})["GIT_PAGER"] = "cat"
    docker_env = _FixedCmdDockerEnvironment(**env_cfg)

    # 2b. Run post_setup_cmd if provided
    post_setup_cmd = env_config.get("post_setup_cmd", "")
    if post_setup_cmd:
        logger.info("Running post_setup_cmd (%d chars)...", len(post_setup_cmd))
        result = docker_env.execute({"command": post_setup_cmd}, timeout=120)
        rc = result.get("returncode", 0)
        if rc != 0:
            logger.warning("post_setup_cmd failed (rc=%d): %s", rc, result.get("output", "")[:200])
        else:
            logger.info("post_setup_cmd done")

    # 3. Prepare metadata
    metadata, eval_timeout = build_reward_context(tools_kwargs)

    try:
        # 4. Create LitellmModel pointing at gateway
        model_cfg = dict(_SWEBENCH_CONFIG.get("model", {}))
        model_cfg.update({
            "model_name": "openai/default",
            "model_kwargs": {
                "api_base": session.base_url,
                "api_key": "not-needed",
                "drop_params": True,
            },
            "cost_tracking": "ignore_errors",
        })
        model = LitellmModel(**model_cfg)

        # 5. Create DefaultAgent
        agent_cfg = dict(_SWEBENCH_CONFIG.get("agent", {}))
        agent_cfg["step_limit"] = int(os.environ.get("SWE_AGENT_MAX_TURNS", str(agent_cfg.get("step_limit", 250))))
        agent_cfg["cost_limit"] = 0
        logger.debug("[sample %d] step_limit=%d", sample_index, agent_cfg["step_limit"])
        agent = DefaultAgent(model, docker_env, **agent_cfg)

        # 6. Run agent in thread (DefaultAgent is synchronous)
        logger.debug("[sample %d] starting agent run", sample_index)
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        info = await loop.run_in_executor(None, lambda: agent.run(task=task))

        exit_status = info.get("exit_status", "unknown")
        submission = info.get("submission", "")
        logger.debug("[sample %d] agent finished: exit_status=%s, steps=%d (%.1fs)", sample_index, exit_status, agent.n_calls, time.perf_counter() - t0)

        # 7. Evaluate reward
        t0 = time.perf_counter()
        reward_env = DockerEnvForReward(docker_env)
        score, eval_result = await evaluate_in_env(reward_env, metadata, eval_timeout)
        logger.debug("[sample %d] reward done: score=%s, resolved=%s (%.1fs)", sample_index, score, eval_result.get("resolved"), time.perf_counter() - t0)

        # 8. Signal completion with reward_info
        reward_info = {"reward_score": score, **eval_result}
        await session_runtime.complete_session(session.session_id, reward_info=reward_info)

    except Exception as e:
        logger.warning("Mini-swe-agent runner failed for sample %d: %s", sample_index, e)
        raise
    finally:
        try:
            docker_env.cleanup()
        except Exception:
            pass
