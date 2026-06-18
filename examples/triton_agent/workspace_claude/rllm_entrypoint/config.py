"""
config.py — Parse all environment variables for the rllm entrypoint.

All configuration lives here so every other module can do:

    from rllm_entrypoint.config import cfg

and get a strongly-typed, immutable config object.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EntrypointConfig:
    # ── Core LLM / agent settings ───────────────────────────────────────────
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    workspace_base: str
    max_iterations: int
    npu_operator_task: bool
    operator_backend: str
    operator_name: str
    operator_arch: str
    task_instruction: str

    # ── Observability / control REST gateway ────────────────────────────────
    # Base URL of the external observer REST API.
    # e.g.  http://host.docker.internal:8765
    # If empty, observability upload is disabled (container runs standalone).
    observer_api_url: str

    # How often (seconds) the background uploader posts state to the gateway.
    upload_interval_s: float

    # How often (seconds) the pause-poller checks whether a pause was requested.
    pause_poll_interval_s: float

    # Unique session / container identifier used as the "topic" for the gateway.
    session_id: str

    # Optional human-readable label for this session.
    session_label: str

    # ── System prompt path ──────────────────────────────────────────────────
    system_prompt_path: str

    # ── Metadata pass-through (arbitrary JSON string) ───────────────────────
    extra_metadata: str  # raw JSON or empty string


def _load() -> EntrypointConfig:
    """Read environment variables and return a frozen config."""
    import json
    import uuid

    llm_base_url = os.environ.get("LLM_BASE_URL", "")
    llm_api_key = os.environ.get("LLM_API_KEY", "EMPTY")
    llm_model = os.environ.get("LLM_MODEL", "openai/openhands-model")
    workspace_base = os.environ.get("WORKSPACE_BASE", "/opt/workspace")
    max_iterations = int(os.environ.get("MAX_ITERATIONS", "30"))
    npu_operator_task = os.environ.get("NPU_OPERATOR_TASK", "0") in ("1", "true", "True", "yes")
    operator_backend = os.environ.get("OPERATOR_BACKEND", "triton")
    operator_name: str = os.environ.get("OPERATOR_NAME", "operator")
    operator_arch: str = os.environ.get("OPERATOR_ARCH", "ascend910b1")

    md_path = os.path.join(workspace_base, "INSTRUCTIONS.md")
    if os.path.exists(md_path):
        with open(md_path) as f:
            lines = [ln for ln in f.read().splitlines()
                        if ln.strip() and not ln.startswith("#")]
        task_instruction = "\n".join(lines).strip()

    observer_api_url = os.environ.get("OBSERVER_API_URL", "").rstrip("/")
    upload_interval_s = float(os.environ.get("OBSERVER_UPLOAD_INTERVAL", "5.0"))
    pause_poll_interval_s = float(os.environ.get("OBSERVER_PAUSE_POLL_INTERVAL", "2.0"))

    session_id = os.environ.get("OBSERVER_SESSION_ID", str(uuid.uuid4()))
    session_label = os.environ.get("OBSERVER_SESSION_LABEL", session_id[:12])

    system_prompt_path = os.environ.get(
        "SYSTEM_PROMPT_PATH", "/tmp/rllm_minimal_system.j2"
    )
    extra_metadata = os.environ.get("OBSERVER_EXTRA_METADATA", "")

    return EntrypointConfig(
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        workspace_base=workspace_base,
        max_iterations=max_iterations,
        npu_operator_task=npu_operator_task,
        operator_backend=operator_backend,
        operator_name=operator_name,
        operator_arch=operator_arch,
        task_instruction=task_instruction,
        observer_api_url=observer_api_url,
        upload_interval_s=upload_interval_s,
        pause_poll_interval_s=pause_poll_interval_s,
        session_id=session_id,
        session_label=session_label,
        system_prompt_path=system_prompt_path,
        extra_metadata=extra_metadata,
    )


# Module-level singleton — import and use everywhere.
cfg: EntrypointConfig = _load()
