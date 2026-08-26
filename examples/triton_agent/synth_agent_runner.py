"""Synth-flow Triton/Ascend KernelBench runner for RL training.

Adopts the pure-inference synth flow (``synth_common.run_synth_inference``:
synth system/user prompt + 5 tools ``list_skills`` / ``read_skill`` /
``str_replace_editor`` / ``run_verify`` / ``submit`` + step+nudge loop +
``run_verify``-driven reward) as the RL rollout inference engine, while
satisfying the uni-agent blackbox framework contract: drive the model against
``session.base_url`` so the gateway captures training tokens, and report reward
through ``session_runtime.complete_session(..., reward_info=...)``.

This runner and the standalone synth flow (``run_synth_levels_serial.sh`` ->
``synth_triton_local.py``) share ONE inference implementation
(``synth_common.run_synth_inference``), so improvements to the synth flow
apply to both paths.

Pipeline:
  verl rollout actor  (model under training)
    -> uni-agent gateway (session.base_url, OpenAI-compatible)
      -> run_synth_inference  (synth prompt + 5 tools + step+nudge loop)
        -> AgentEnv  (Docker/swerex sandbox with Triton/CANN + workspace_knowledge_all)
          -> run_verify tool  (AST + correctness + perf -> verify_result.json / perf_result.json)
  reward = evaluate_triton_workspace(env, ...)   # reads verify_result.json + perf_result.json
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from uni_agent.interaction.env import AgentEnv
from uni_agent.interaction.model import OpenAICompatibleChatModel
from uni_agent.trainer.framework.types import SessionHandle, SessionRuntime

# Reuse the Triton remote-sandbox pool, env construction, task-metadata, and
# rollout-progress helpers from the Claude Code runner. These are independent
# of Claude Code itself.
from examples.triton_agent.claude_code_agent_runner import (
    _acquire_remote_sandbox,
    _create_agent_env,
    _format_rollout_progress,
    _format_rollout_start,
    _merge_env_config,
    _record_rollout_progress,
    _release_remote_sandbox,
    _remote_sandbox_pool,
    _task_metadata,
    load_agent_config,
)
from examples.triton_agent.reward import (
    archive_text_artifacts,
)
# Shared inference core + workspace staging from the synth flow.
from examples.triton_agent.synth_common import (
    run_synth_inference,
    setup_workspace,
    upload_workspace,
)
# Reuse the host-side trajectory JSONL dumper from the sandbox runner.
from examples.triton_agent.sandbox_agent_runner import (
    _dump_trajectory_jsonl,
    _safe_run_id,
)

logger = logging.getLogger(__name__)


def _shell_quote(value: str) -> str:
    import shlex

    return shlex.quote(str(value))


def _env_variables_for_synth(metadata: dict[str, Any], workspace_dir: str) -> dict[str, str]:
    """Env vars to export in the sandbox bash session for the synth flow.

    Combines the static workspace/skill vars (so ``list_skills`` /
    ``read_skill`` / ``run_verify`` resolve paths) with the per-operator vars
    the verifier scripts consume (``OPERATOR_*``).
    """
    op_name = str(metadata.get("op_name", "operator"))
    arch = str(metadata.get("arch", "ascend910b1"))
    return {
        "WORKSPACE_BASE": workspace_dir,
        "TRITON_WORKSPACE_DIR": workspace_dir,
        "SKILLS_DIR": f"{workspace_dir}/.skills",
        "EVAL_LOCK_DIR": "/shared/device-locks",
        "EVAL_DEVICE_PREFIX": "npu",
        "TRITON_PIPELINE_ERROR_PREVIEW_CHARS": os.environ.get(
            "TRITON_PIPELINE_ERROR_PREVIEW_CHARS", "2000"
        ),
        "OPERATOR_BACKEND": str(metadata.get("operator_backend", "triton")),
        "OPERATOR_ARCH": arch,
        "OPERATOR_NAME": op_name,
        "OPERATOR_PYTHON": os.environ.get("OPERATOR_PYTHON", "/usr/local/bin/python3"),
        "AST_CHECK_PYTHON": "python3",
        "EVAL_TIMEOUT": os.environ.get("TRITON_EVAL_TIMEOUT", "900"),
    }


async def triton_synth_runner(
    *,
    raw_prompt,
    session: SessionHandle,
    sample_index: int,
    session_runtime: SessionRuntime,
    tools_kwargs: dict | None = None,
    agent_config_path: str | None = None,
    **kwargs,
) -> None:
    """Run one Triton KernelBench rollout through the synth inference flow.

    The model talks to the gateway (``session.base_url``) using standard
    OpenAI tool-call format; ``run_synth_inference`` drives the synth ReAct
    loop (5 tools, step+nudge) and dispatches ``str_replace_editor`` /
    ``list_skills`` / ``read_skill`` / ``run_verify`` / ``submit`` against a
    Docker sandbox that has the Triton + CANN verifier stack and the
    ``workspace_knowledge_all`` skill set. Reward is computed in-sandbox via
    ``evaluate_triton_workspace`` and reported through
    ``session_runtime.complete_session``.
    """
    del kwargs
    tools_kwargs = tools_kwargs or {}
    if getattr(session, "base_url", None) is None:
        raise ValueError("session.base_url is required for triton_synth_runner")

    config_path = agent_config_path or tools_kwargs.get("agent_config_path")
    if not config_path:
        raise ValueError(
            "agent_config_path is required (via parameter or tools_kwargs.agent_config_path)"
        )
    agent_config = load_agent_config(config_path)
    interaction_cfg = agent_config.get("interaction", {})
    print("============================================triton_synth_runner")

    metadata = _task_metadata(raw_prompt, tools_kwargs)
    trace_label = str(
        metadata.get("uid") or metadata.get("op_name") or sample_index
    ).replace("/", "_")[:96]
    run_id = f"triton_synth_{sample_index}_{uuid4().hex[:8]}"
    metadata["rollout_run_id"] = run_id
    metadata["sample_index"] = sample_index

    # Build the synth task dict from RL metadata - mirrors the shape produced by
    # synth_common.load_tasks (op_name / arch / task_code / instruction /
    # support_files).
    task: dict[str, Any] = {
        "op_name": str(metadata.get("op_name", "operator")),
        "arch": str(metadata.get("arch", "ascend910b2")),
        "task_code": str(metadata.get("task_code", "")),
        "instruction": str(
            metadata.get("instruction")
            or f"Implement the {metadata.get('op_name', 'operator')} operator."
        ),
        "support_files": metadata.get("support_files")
        if isinstance(metadata.get("support_files"), dict)
        else {},
    }

    host_workspace: Path | None = None
    env: AgentEnv | None = None
    remote_sandbox_index: int | None = None
    reward_info: dict[str, Any] = {"reward_score": 0.0}
    trajectory: list = []
    messages: list = []
    result: dict[str, Any] = {}

    try:
        # ------------------------------------------------------------------
        # 1. Stage host workspace (workspace_knowledge_all -> .skills + src/)
        # ------------------------------------------------------------------
        workspace_temp_dir = Path(__file__).resolve().parent / "workspace_temp"
        workspace_temp_dir.mkdir(parents=True, exist_ok=True)
        host_workspace = Path(
            tempfile.mkdtemp(prefix=f"synth-rl-{trace_label}-", dir=workspace_temp_dir)
        )
        setup_workspace(task, host_workspace)

        # ------------------------------------------------------------------
        # 2. Acquire a sandbox (remote pool if configured, else local docker)
        # ------------------------------------------------------------------
        remote_pool = _remote_sandbox_pool(tools_kwargs)
        if remote_pool:
            remote_sandbox_index, endpoint = await _acquire_remote_sandbox(remote_pool)
            metadata["sandbox_index"] = remote_sandbox_index
            metadata["sandbox_host"] = endpoint.get("host")
            metadata["sandbox_port"] = endpoint.get("port")
            tools_kwargs = {
                **tools_kwargs,
                "env": _merge_env_config(
                    dict(tools_kwargs.get("env", {})),
                    {"deployment": endpoint},
                ),
            }
        print(_format_rollout_start(metadata), flush=True)

        env = _create_agent_env(run_id, tools_kwargs, agent_config)
        await env.start()

        # ------------------------------------------------------------------
        # 3. Upload workspace + export env vars so run_verify / list_skills work
        # ------------------------------------------------------------------
        workspace_dir = await upload_workspace(env, host_workspace)
        metadata["workspace_dir"] = workspace_dir
        exports = " && ".join(
            f"export {k}={_shell_quote(v)}"
            for k, v in _env_variables_for_synth(metadata, workspace_dir).items()
        )
        if exports:
            await env.communicate(exports, check="ignore")

        # ------------------------------------------------------------------
        # 4. Build the model pointing at the gateway (session.base_url)
        # ------------------------------------------------------------------
        model = OpenAICompatibleChatModel(
            base_url=session.base_url,
            api_key=os.environ.get("TRITON_GATEWAY_API_KEY", "EMPTY"),
            model_name=os.environ.get("TRITON_GATEWAY_MODEL_NAME", "default"),
            timeout=int(os.environ.get("API_TIMEOUT_MS", "1800000")) // 1000 or 1800,
        )

        env_max_turns_env = os.environ.get("TRITON_CLAUDE_MAX_TURNS")
        env_max_turns = int(env_max_turns_env) if env_max_turns_env else None
        max_turns = env_max_turns or int(interaction_cfg.get("max_turns", 32))
        action_timeout = int(interaction_cfg.get("action_timeout", 300))

        _msg = (
            f"[synth-runner] trace={trace_label} starting synth inference: "
            f"workspace={workspace_dir} base_url={session.base_url} "
            f"max_turns={max_turns} action_timeout={action_timeout} "
            f"op_name={metadata.get('op_name')}"
        )
        logger.info(_msg)
        print(_msg, flush=True)

        # ------------------------------------------------------------------
        # 5. Run the shared synth inference core
        #    (synth prompt + 5 tools + step+nudge loop + evaluate_triton_workspace)
        # ------------------------------------------------------------------
        started_at = time.perf_counter()
        result = await run_synth_inference(
            task=task,
            chat_model=model,
            env=env,
            workspace_dir=workspace_dir,
            metadata=metadata,
            max_turns=max_turns,
            action_timeout=action_timeout,
            started_at=started_at,
        )
        trajectory = result.get("trajectory", []) or []
        messages = result.get("messages", []) or []
        reward_score = float(result.get("reward_score", 0.0))
        exit_reason = result.get("exit_reason", "unknown")
        num_turns = int(result.get("num_turns", 0))
        eval_result = result.get("eval_result", {})
        if not isinstance(eval_result, dict):
            eval_result = {}

        _msg = (
            f"[synth-runner] trace={trace_label} synth inference finished: "
            f"turns={num_turns} exit_reason={exit_reason} reward={reward_score:.4f} "
            f"verified={result.get('has_verified')} submitted={result.get('submitted')} "
            f"elapsed={time.perf_counter() - started_at:.1f}s"
        )
        logger.info(_msg)
        print(_msg, flush=True)

        # Diagnostic: list what files the model created in src/
        try:
            ls_output = await env.communicate(
                f"ls -la {workspace_dir}/src/ 2>&1 || true", check="ignore"
            )
            logger.info("[synth-runner] trace=%s src/ listing after inference:\n%s",
                        trace_label, ls_output)
        except Exception as exc:
            logger.debug("Failed to list src/ for diagnostic: %s", exc)

        # ------------------------------------------------------------------
        # 6. Archive artifacts
        # ------------------------------------------------------------------
        archived_dir = None
        try:
            artifact_dir = (
                (tools_kwargs.get("claude_code") or {}).get("artifact_dir")
                or os.environ.get("TRITON_CLAUDE_ARTIFACT_DIR", "")
            )
            if artifact_dir:
                archived_dir = await archive_text_artifacts(
                    env, metadata, str(artifact_dir), workspace_dir=workspace_dir,
                )
        except Exception as exc:
            logger.warning("Failed to archive artifacts for %s: %s", metadata.get("uid"), exc)

        # ------------------------------------------------------------------
        # 7. Pack reward_info and report
        # ------------------------------------------------------------------
        exit_code = 0 if bool(eval_result.get("resolved")) else 1

        # Build `train_best` so framework.py can crop the training trajectory to
        # the best verify turn when TRITON_TRAIN_BEST_FIRST=1. The assistant_index
        # is the 0-based global assistant-message index of the verify whose
        # metrics the reward actually used (selected_metrics_source), tracked in
        # run_synth_inference by stat'ing the in-loop best files. When the source
        # is "metrics" (final state), assistant_index is None -> the framework
        # falls back to the full/final prefix, consistent with the final reward.
        # This makes TRITON_TRAIN_BEST_FIRST govern both sides consistently:
        # =1 -> best reward + best-prefix crop; =0 -> final reward + full prefix
        # (train_best is ignored by the framework when best_first=0).
        best_src = str(eval_result.get("selected_metrics_source") or "metrics")
        best_idx_map = result.get("best_index_by_source")
        train_best_assistant_index = None
        if isinstance(best_idx_map, dict) and best_src in best_idx_map:
            train_best_assistant_index = best_idx_map.get(best_src)

        # Only provide assistant_messages_seen alongside a real best index. When
        # the reward used the final state (source="metrics") we leave both None
        # so framework.py returns None from _train_best_assistant_index and falls
        # back to final_prefixes (full trajectory) -- robustly consistent with the
        # final reward, without depending on our assistant-turn count.
        if train_best_assistant_index is None:
            train_best_messages_seen = None
        else:
            train_best_messages_seen = result.get("assistant_messages_seen")

        reward_info = {
            "reward_score": reward_score,
            "archived_dir": archived_dir,
            "rollout_run_id": run_id,
            "sample_index": sample_index,
            "sandbox_index": metadata.get("sandbox_index"),
            "sandbox_host": metadata.get("sandbox_host"),
            "sandbox_port": metadata.get("sandbox_port"),
            "uid": metadata.get("uid"),
            "op_name": metadata.get("op_name"),
            "exit_reason": exit_reason,
            "num_turns": num_turns,
            **eval_result,
            "train_best": {
                "source": best_src,
                "used_best_metrics": bool(eval_result.get("used_best_metrics")),
                "assistant_messages_seen": train_best_messages_seen,
                "assistant_index": train_best_assistant_index,
            },
        }
        print(
            f"[synth-runner] train_best: source={best_src} "
            f"assistant_index={train_best_assistant_index} "
            f"assistant_messages_seen={train_best_messages_seen} "
            f"used_best_metrics={bool(eval_result.get('used_best_metrics'))} "
            f"trace={trace_label}",
            flush=True,
        )
        # NOTE: _record_rollout_progress returns a 3-tuple
        # (process_completed, global_completed, step_summary); the step_summary
        # is consumed by the Claude Code runner for per-step averaging and is
        # not needed here.
        _process_completed, global_completed, _step_summary = _record_rollout_progress(
            sample_index=sample_index,
            metadata=metadata,
            reward_info=reward_info,
            exit_code=exit_code,
            archived_dir=archived_dir,
        )
        progress_message = _format_rollout_progress(
            global_completed=global_completed,
            sample_index=sample_index,
            metadata=metadata,
            reward_info=reward_info,
            exit_code=exit_code,
            archived_dir=archived_dir,
        )
        logger.info(progress_message)
        if os.environ.get("TRITON_PROGRESS_STDOUT", "1") not in ("0", "false", "False", "no"):
            print(progress_message, flush=True)

        # Dump full trajectory to JSONL (host-side, for debugging).
        try:
            _dump_trajectory_jsonl(
                run_id=run_id,
                trace_label=trace_label,
                trajectory=trajectory,
                messages=messages,
                metadata=metadata,
                raw_prompt=raw_prompt,
                reward_info=reward_info,
            )
        except Exception as exc:
            logger.debug("Failed to dump trajectory JSONL: %s", exc)

        await session_runtime.complete_session(session.session_id, reward_info=reward_info)

    except Exception as exc:
        logger.exception("Triton synth runner failed for sample %s", sample_index)
        reward_info = {
            "reward_score": 0.0,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "rollout_run_id": run_id,
            "sample_index": sample_index,
            "sandbox_index": metadata.get("sandbox_index"),
            "sandbox_host": metadata.get("sandbox_host"),
            "sandbox_port": metadata.get("sandbox_port"),
            "uid": metadata.get("uid"),
            "op_name": metadata.get("op_name"),
        }
        # Dump trajectory even on failure for debugging.
        try:
            _dump_trajectory_jsonl(
                run_id=run_id,
                trace_label=trace_label,
                trajectory=trajectory,
                messages=messages,
                metadata=metadata,
                raw_prompt=raw_prompt,
                reward_info=reward_info,
            )
        except Exception:
            pass
        try:
            await session_runtime.complete_session(session.session_id, reward_info=reward_info)
        except Exception:
            pass
        raise
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                logger.debug("Failed to close env", exc_info=True)
        _release_remote_sandbox(remote_sandbox_index)
        if host_workspace is not None and os.environ.get(
            "TRITON_CLAUDE_KEEP_HOST_WORKSPACE", "0"
        ) not in ("1", "true", "True", "yes"):
            import shutil

            shutil.rmtree(host_workspace, ignore_errors=True)