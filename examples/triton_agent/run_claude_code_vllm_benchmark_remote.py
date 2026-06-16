"""Run one Claude Code vLLM benchmark sample through a uni-agent remote sandbox."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from examples.triton_agent.anthropic_openai_shim import AnthropicOpenAIShim
from examples.triton_agent.claude_code_agent_runner import (
    _create_agent_env,
    _ensure_claude_run_user,
    _gate_exit_code,
    _install_claude_code,
    _prepare_claude_home,
    _run_claude_code_with_repairs,
    _upload_workspace,
    load_agent_config,
)
from examples.triton_agent.reward import archive_text_artifacts


async def _run(args: argparse.Namespace) -> dict:
    env_variables = {
        "PIP_PROGRESS_BAR": "off",
        "PIP_CACHE_DIR": "~/.cache/pip",
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R",
        "TQDM_DISABLE": "1",
        "GIT_PAGER": "cat",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "CONDA_BASE": os.environ.get("CONDA_BASE", "/opt/conda"),
        "OPERATOR_CONDA_ENV": os.environ.get("OPERATOR_CONDA_ENV", "evaluator-py311"),
        "OPERATOR_PYTHON": os.environ.get("OPERATOR_PYTHON", "/opt/conda/envs/evaluator-py311/bin/python"),
        "WORKSPACE_BASE": "/opt/workspace/agent_workdir",
        "TRITON_PIPELINE_ERROR_PREVIEW_CHARS": os.environ.get("TRITON_PIPELINE_ERROR_PREVIEW_CHARS", "2000"),
    }
    metadata = {
        "op_name": args.op_name,
        "uid": args.op_name,
        "arch": args.arch,
        "operator_backend": "triton",
    }
    tools_kwargs = {
        "env": {
            "deployment": {
                "type": "local_attach",
                "host": args.remote_host,
                "port": args.remote_port,
                "auth_token": args.remote_auth_token,
                "timeout": args.sandbox_timeout,
                "startup_timeout": args.sandbox_startup_timeout,
            },
            "env_variables": env_variables,
        },
        "claude_code": {
            "time_budget_sec": args.time_budget_sec,
            "prompt": args.prompt,
            "artifact_dir": args.artifact_dir,
            "model": args.model,
            "run_user": args.run_user,
            "run_home": args.run_home,
            "extra_args": args.extra_args,
        },
    }
    if not args.run_user:
        tools_kwargs["claude_code"].pop("run_user")
    if not args.run_home:
        tools_kwargs["claude_code"].pop("run_home")
    if not args.artifact_dir:
        tools_kwargs["claude_code"].pop("artifact_dir")

    agent_config = load_agent_config(args.agent_config)
    env = _create_agent_env(f"benchmark-{uuid4().hex}", tools_kwargs, agent_config)
    archived_dir = None
    exit_code = -1
    reward_info: dict = {"reward_score": 0.0}

    try:
        await env.start()
        workspace_dir = await _upload_workspace(env, Path(args.workspace))
        await _install_claude_code(env)
        if args.run_home:
            run_home = args.run_home
        elif args.run_user == "root":
            run_home = "/root"
        else:
            run_home = f"/home/{args.run_user}" if args.run_user else None
        await _ensure_claude_run_user(env, args.run_user, run_home)
        await _prepare_claude_home(env, run_user=args.run_user, run_home=run_home)

        with AnthropicOpenAIShim(
            openai_base_url=args.openai_base_url,
            openai_api_key=args.openai_api_key,
            model_name=args.model,
            host=args.shim_bind_host,
            port=args.shim_port,
            request_timeout=args.shim_request_timeout,
        ) as shim:
            shim_url = f"http://{args.shim_public_host}:{shim.port}"
            exit_code, score, eval_result = await _run_claude_code_with_repairs(
                env,
                workspace_dir=workspace_dir,
                shim_url=shim_url,
                session_id=args.session_id,
                tools_kwargs=tools_kwargs,
                metadata=metadata,
            )

        exit_code = _gate_exit_code(exit_code, eval_result)
        archived_dir = await archive_text_artifacts(env, metadata, args.artifact_dir, workspace_dir=workspace_dir)
        metrics = eval_result.get("metrics") if isinstance(eval_result, dict) else {}
        reward_info = {
            "reward_score": score,
            "metrics": metrics,
            "passed": bool(metrics.get("success")) if isinstance(metrics, dict) else False,
            "reason": eval_result.get("reason") if isinstance(eval_result, dict) else "",
        }
    finally:
        try:
            await env.stop()
        except Exception:
            pass

    return {
        "op_name": args.op_name,
        "workspace": args.workspace,
        "claude_exit": exit_code,
        "archived_dir": archived_dir,
        "reward_info": reward_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--arch", default="ascend910b1")
    parser.add_argument("--openai-base-url", required=True)
    parser.add_argument("--openai-api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--time-budget-sec", type=int, default=1800)
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--agent-config", default="examples/triton_agent/agent_config_claude_code.yaml")
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, required=True)
    parser.add_argument("--remote-auth-token", required=True)
    parser.add_argument("--sandbox-timeout", type=float, default=600)
    parser.add_argument("--sandbox-startup-timeout", type=float, default=600)
    parser.add_argument("--run-user", default=os.environ.get("TRITON_CLAUDE_RUN_USER", "claude"))
    parser.add_argument("--run-home", default=os.environ.get("TRITON_CLAUDE_RUN_HOME", ""))
    parser.add_argument("--shim-bind-host", default=os.environ.get("SHIM_BIND_HOST", "0.0.0.0"))
    parser.add_argument("--shim-public-host", default=os.environ.get("SHIM_PUBLIC_HOST", "127.0.0.1"))
    parser.add_argument("--shim-port", type=int, default=int(os.environ.get("SHIM_PORT", "0") or "0"))
    parser.add_argument("--shim-request-timeout", type=float, default=float(os.environ.get("SHIM_REQUEST_TIMEOUT", "600")))
    parser.add_argument("--session-id", default="benchmark-session")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = asyncio.run(_run(args))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
