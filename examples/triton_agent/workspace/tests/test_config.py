"""
test_config.py — Unit tests for rllm_entrypoint/config.py (EntrypointConfig / _load).

Tests cover:
  - Default values when no env-vars are set
  - Each env-var override
  - TASK_INSTRUCTION fallback to INSTRUCTIONS.md
  - observer_api_url trailing-slash stripping
  - session_id random UUID default
  - extra_metadata passthrough
  - frozen dataclass (immutability)
"""
from __future__ import annotations

import os
import sys
import importlib
import uuid

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_WORKSPACE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "workspace",
)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)


# ---------------------------------------------------------------------------
# Helpers: call _load() with a custom env
# ---------------------------------------------------------------------------

def _load_with_env(env: dict[str, str], tmp_path=None):
    """Import _load freshly with the given environment variables."""
    import rllm_entrypoint.config as _cfg_mod

    # Override os.environ for the duration of this call
    old = {}
    for k, v in env.items():
        old[k] = os.environ.pop(k, None)
        os.environ[k] = v

    # Clear any vars NOT in env that we want to reset
    unset = {
        "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "WORKSPACE_BASE",
        "MAX_ITERATIONS", "NPU_OPERATOR_TASK", "OPERATOR_BACKEND",
        "TASK_INSTRUCTION", "OBSERVER_API_URL", "OBSERVER_UPLOAD_INTERVAL",
        "OBSERVER_PAUSE_POLL_INTERVAL", "OBSERVER_SESSION_ID", "OBSERVER_SESSION_LABEL",
        "SYSTEM_PROMPT_PATH", "OBSERVER_EXTRA_METADATA",
    } - set(env.keys())
    saved_unset = {}
    for k in unset:
        saved_unset[k] = os.environ.pop(k, None)

    try:
        cfg = _cfg_mod._load()
    finally:
        # Restore environment
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for k, v in saved_unset.items():
            if v is not None:
                os.environ[k] = v

    return cfg


# ===========================================================================
# Default values
# ===========================================================================

class TestDefaults:
    def test_llm_api_key_default(self):
        cfg = _load_with_env({"LLM_BASE_URL": "http://proxy"})
        assert cfg.llm_api_key == "EMPTY"

    def test_llm_model_default(self):
        cfg = _load_with_env({"LLM_BASE_URL": "http://proxy"})
        assert cfg.llm_model == "openai/openhands-model"

    def test_workspace_base_default(self):
        cfg = _load_with_env({"LLM_BASE_URL": "http://proxy"})
        assert cfg.workspace_base == "/opt/workspace"

    def test_max_iterations_default(self):
        cfg = _load_with_env({"LLM_BASE_URL": "http://proxy"})
        assert cfg.max_iterations == 30

    def test_npu_operator_task_default_false(self):
        cfg = _load_with_env({"LLM_BASE_URL": "http://proxy"})
        assert cfg.npu_operator_task is False

    def test_operator_backend_default(self):
        cfg = _load_with_env({"LLM_BASE_URL": "http://proxy"})
        assert cfg.operator_backend == "triton"

    def test_observer_api_url_default_empty(self):
        cfg = _load_with_env({"LLM_BASE_URL": "http://proxy"})
        assert cfg.observer_api_url == ""

    def test_upload_interval_default(self):
        cfg = _load_with_env({"LLM_BASE_URL": "http://proxy"})
        assert cfg.upload_interval_s == 5.0

    def test_pause_poll_interval_default(self):
        cfg = _load_with_env({"LLM_BASE_URL": "http://proxy"})
        assert cfg.pause_poll_interval_s == 2.0

    def test_session_id_is_valid_uuid(self):
        cfg = _load_with_env({"LLM_BASE_URL": "http://proxy"})
        parsed = uuid.UUID(cfg.session_id)
        assert str(parsed) == cfg.session_id

    def test_session_label_defaults_to_first_12_of_session_id(self):
        cfg = _load_with_env({"LLM_BASE_URL": "http://proxy"})
        assert cfg.session_label == cfg.session_id[:12]

    def test_system_prompt_path_default(self):
        cfg = _load_with_env({"LLM_BASE_URL": "http://proxy"})
        assert cfg.system_prompt_path == "/tmp/rllm_minimal_system.j2"

    def test_extra_metadata_default_empty(self):
        cfg = _load_with_env({"LLM_BASE_URL": "http://proxy"})
        assert cfg.extra_metadata == ""


# ===========================================================================
# Override via env vars
# ===========================================================================

class TestEnvOverrides:
    def test_llm_base_url(self):
        cfg = _load_with_env({"LLM_BASE_URL": "http://my-proxy:9000/v1"})
        assert cfg.llm_base_url == "http://my-proxy:9000/v1"

    def test_llm_api_key(self):
        cfg = _load_with_env({"LLM_BASE_URL": "http://proxy", "LLM_API_KEY": "sk-secret"})
        assert cfg.llm_api_key == "sk-secret"

    def test_max_iterations(self):
        cfg = _load_with_env({"LLM_BASE_URL": "http://proxy", "MAX_ITERATIONS": "50"})
        assert cfg.max_iterations == 50

    def test_npu_operator_task_truthy_values(self):
        for val in ("1", "true", "True", "yes"):
            cfg = _load_with_env({"LLM_BASE_URL": "http://proxy", "NPU_OPERATOR_TASK": val})
            assert cfg.npu_operator_task is True, f"Expected True for NPU_OPERATOR_TASK={val!r}"

    def test_npu_operator_task_falsy(self):
        for val in ("0", "false", "no", ""):
            cfg = _load_with_env({"LLM_BASE_URL": "http://proxy", "NPU_OPERATOR_TASK": val})
            assert cfg.npu_operator_task is False

    def test_operator_backend_override(self):
        cfg = _load_with_env({"LLM_BASE_URL": "http://proxy", "OPERATOR_BACKEND": "cann"})
        assert cfg.operator_backend == "cann"

    def test_task_instruction_direct(self):
        cfg = _load_with_env({
            "LLM_BASE_URL": "http://proxy",
            "TASK_INSTRUCTION": "My custom task",
        })
        assert cfg.task_instruction == "My custom task"

    def test_observer_api_url_trailing_slash_stripped(self):
        cfg = _load_with_env({
            "LLM_BASE_URL": "http://proxy",
            "OBSERVER_API_URL": "http://gateway:8765/",
        })
        assert cfg.observer_api_url == "http://gateway:8765"

    def test_upload_interval_override(self):
        cfg = _load_with_env({
            "LLM_BASE_URL": "http://proxy",
            "OBSERVER_UPLOAD_INTERVAL": "10.5",
        })
        assert cfg.upload_interval_s == 10.5

    def test_pause_poll_interval_override(self):
        cfg = _load_with_env({
            "LLM_BASE_URL": "http://proxy",
            "OBSERVER_PAUSE_POLL_INTERVAL": "3.0",
        })
        assert cfg.pause_poll_interval_s == 3.0

    def test_session_id_override(self):
        cfg = _load_with_env({
            "LLM_BASE_URL": "http://proxy",
            "OBSERVER_SESSION_ID": "my-fixed-id",
        })
        assert cfg.session_id == "my-fixed-id"

    def test_session_label_override(self):
        cfg = _load_with_env({
            "LLM_BASE_URL": "http://proxy",
            "OBSERVER_SESSION_LABEL": "my-run",
        })
        assert cfg.session_label == "my-run"

    def test_extra_metadata_override(self):
        cfg = _load_with_env({
            "LLM_BASE_URL": "http://proxy",
            "OBSERVER_EXTRA_METADATA": '{"key": "value"}',
        })
        assert cfg.extra_metadata == '{"key": "value"}'


# ===========================================================================
# TASK_INSTRUCTION fallback from INSTRUCTIONS.md
# ===========================================================================

class TestTaskInstructionFallback:
    def test_reads_instructions_md_when_no_env_var(self, tmp_path):
        md = tmp_path / "INSTRUCTIONS.md"
        md.write_text("# Heading\n\nDo the thing\nAnd another thing\n")
        cfg = _load_with_env({
            "LLM_BASE_URL": "http://proxy",
            "WORKSPACE_BASE": str(tmp_path),
        })
        # Non-empty lines that don't start with '#' should be joined
        assert "Do the thing" in cfg.task_instruction
        assert "And another thing" in cfg.task_instruction

    def test_comment_lines_excluded(self, tmp_path):
        md = tmp_path / "INSTRUCTIONS.md"
        md.write_text("# Title\n## Sub\nActual content\n")
        cfg = _load_with_env({
            "LLM_BASE_URL": "http://proxy",
            "WORKSPACE_BASE": str(tmp_path),
        })
        assert "Title" not in cfg.task_instruction
        assert "Actual content" in cfg.task_instruction

    def test_env_var_takes_precedence_over_md(self, tmp_path):
        md = tmp_path / "INSTRUCTIONS.md"
        md.write_text("From file\n")
        cfg = _load_with_env({
            "LLM_BASE_URL": "http://proxy",
            "WORKSPACE_BASE": str(tmp_path),
            "TASK_INSTRUCTION": "From env",
        })
        assert cfg.task_instruction == "From env"

    def test_missing_instructions_md_is_ok(self, tmp_path):
        """No INSTRUCTIONS.md → task_instruction is empty string."""
        cfg = _load_with_env({
            "LLM_BASE_URL": "http://proxy",
            "WORKSPACE_BASE": str(tmp_path),
        })
        assert cfg.task_instruction == ""


# ===========================================================================
# Frozen dataclass immutability
# ===========================================================================

class TestImmutability:
    def test_config_is_frozen(self):
        cfg = _load_with_env({"LLM_BASE_URL": "http://proxy"})
        with pytest.raises((TypeError, AttributeError)):
            cfg.llm_model = "something-else"   # type: ignore[misc]
