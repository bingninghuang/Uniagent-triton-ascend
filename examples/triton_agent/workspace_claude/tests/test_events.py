"""
test_events.py — Unit tests for rllm_entrypoint/events.py

Tests cover:
  - _safe_str: truncation and repr-fail safety
  - _truncate_dict: recursive truncation in dict/list
  - _event_to_dict: serialisation for various event types
  - _build_summary: human-readable summaries per event class
  - _phase_from_event: correct phase mapping
  - make_event_callback: returned callback updates RunState correctly
    and never raises
"""
from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock, PropertyMock

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

from rllm_entrypoint.state import AgentPhase, RunState  # noqa: E402
from rllm_entrypoint.events import (  # noqa: E402
    _build_summary,
    _event_to_dict,
    _phase_from_event,
    _safe_str,
    _truncate_dict,
    make_event_callback,
)


# ===========================================================================
# Helper: create a fake SDK event
# ===========================================================================

def _make_sdk_event(event_type: str, **attrs) -> MagicMock:
    """Create a MagicMock that passes isinstance-like checks by class name."""
    ev = MagicMock()
    type(ev).__name__ = event_type
    ev.id = attrs.get("id", f"uuid-{event_type}")
    ev.source = attrs.get("source", "agent")
    ev.timestamp = attrs.get("timestamp", "2024-01-01T00:00:00")
    # model_dump returns the raw dict
    ev.model_dump.return_value = {"event_type": event_type, **attrs}
    # Set specific attributes used in _build_summary
    for k, v in attrs.items():
        setattr(ev, k, v)
    return ev


# ===========================================================================
# _safe_str
# ===========================================================================

class TestSafeStr:
    def test_short_string_unchanged(self):
        assert _safe_str("hello") == "hello"

    def test_long_string_truncated(self):
        result = _safe_str("x" * 3000, max_len=100)
        assert len(result) == 100

    def test_non_string_coerced(self):
        assert _safe_str(42) == "42"

    def test_repr_failure_returns_placeholder(self):
        class Bad:
            def __str__(self):
                raise RuntimeError("boom")
        result = _safe_str(Bad())
        assert result == "<repr failed>"


# ===========================================================================
# _truncate_dict
# ===========================================================================

class TestTruncateDict:
    def test_short_strings_unchanged(self):
        d = {"key": "value"}
        assert _truncate_dict(d, max_str_len=100) == {"key": "value"}

    def test_long_string_truncated_with_ellipsis(self):
        d = {"key": "A" * 600}
        result = _truncate_dict(d, max_str_len=500)
        assert len(result["key"]) == 501   # 500 chars + "…"
        assert result["key"].endswith("…")

    def test_nested_dict_recursed(self):
        d = {"outer": {"inner": "B" * 600}}
        result = _truncate_dict(d, max_str_len=500)
        assert result["outer"]["inner"].endswith("…")

    def test_list_recursed(self):
        d = {"items": ["x" * 600, "short"]}
        result = _truncate_dict(d, max_str_len=500)
        assert result["items"][0].endswith("…")
        assert result["items"][1] == "short"

    def test_non_string_values_unchanged(self):
        d = {"n": 42, "f": 3.14, "b": True, "none": None}
        assert _truncate_dict(d) == d


# ===========================================================================
# _phase_from_event
# ===========================================================================

class TestPhaseFromEvent:
    def test_action_event_maps_to_executing(self):
        assert _phase_from_event("ActionEvent") == AgentPhase.EXECUTING_COMMAND

    def test_observation_event_maps_to_waiting(self):
        assert _phase_from_event("ObservationEvent") == AgentPhase.WAITING_FOR_REPLY

    def test_message_event_maps_to_waiting(self):
        assert _phase_from_event("MessageEvent") == AgentPhase.WAITING_FOR_REPLY

    def test_pause_event_maps_to_paused(self):
        assert _phase_from_event("PauseEvent") == AgentPhase.PAUSED

    def test_unknown_event_returns_none(self):
        assert _phase_from_event("LLMCompletionLogEvent") is None
        assert _phase_from_event("AgentErrorEvent") is None
        assert _phase_from_event("SomethingElse") is None


# ===========================================================================
# _build_summary
# ===========================================================================

class TestBuildSummary:
    def test_action_event_summary(self):
        thought_part = MagicMock()
        thought_part.text = "run ls -la"
        ev = _make_sdk_event("ActionEvent", tool_name="terminal", thought=[thought_part])
        summary = _build_summary(ev, "ActionEvent")
        assert "[Action:terminal]" in summary
        assert "run ls -la" in summary

    def test_observation_event_summary(self):
        obs = MagicMock()
        content_part = MagicMock()
        content_part.text = "file1.py file2.py"
        obs.to_llm_content = [content_part]
        ev = _make_sdk_event("ObservationEvent", tool_name="terminal", observation=obs)
        summary = _build_summary(ev, "ObservationEvent")
        assert "[Observation:terminal]" in summary
        assert "file1.py" in summary

    def test_message_event_summary(self):
        content_part = MagicMock()
        content_part.text = "Task complete"
        msg = MagicMock()
        msg.content = [content_part]
        msg.role = "assistant"
        ev = _make_sdk_event("MessageEvent", llm_message=msg)
        summary = _build_summary(ev, "MessageEvent")
        assert "[Message:assistant]" in summary
        assert "Task complete" in summary

    def test_agent_error_event_summary(self):
        ev = _make_sdk_event("AgentErrorEvent", error="Timeout")
        summary = _build_summary(ev, "AgentErrorEvent")
        assert "[Error]" in summary
        assert "Timeout" in summary

    def test_pause_event_summary(self):
        ev = _make_sdk_event("PauseEvent")
        summary = _build_summary(ev, "PauseEvent")
        assert "[Paused]" in summary

    def test_state_update_event_summary(self):
        ev = _make_sdk_event("ConversationStateUpdateEvent", key="status", value="running")
        summary = _build_summary(ev, "ConversationStateUpdateEvent")
        assert "[StateUpdate]" in summary
        assert "status" in summary

    def test_llm_completion_log_event_summary(self):
        ev = _make_sdk_event("LLMCompletionLogEvent")
        summary = _build_summary(ev, "LLMCompletionLogEvent")
        assert "[LLMCompletion]" in summary

    def test_unknown_event_summary_fallback(self):
        ev = _make_sdk_event("WeirdEvent")
        summary = _build_summary(ev, "WeirdEvent")
        assert "[WeirdEvent]" in summary

    def test_summary_exception_safe(self):
        """_build_summary should never raise even on broken event objects."""
        ev = MagicMock(spec=[])     # no attributes at all → getattr returns MagicMock
        type(ev).__name__ = "ActionEvent"
        # Remove tool_name to force an exception path
        ev.tool_name = PropertyMock(side_effect=RuntimeError("oops"))
        summary = _build_summary(ev, "ActionEvent")
        assert isinstance(summary, str)


# ===========================================================================
# _event_to_dict
# ===========================================================================

class TestEventToDict:
    def test_returns_required_keys(self):
        ev = _make_sdk_event("ActionEvent")
        d = _event_to_dict(ev)
        assert {"event_type", "event_id", "source", "timestamp", "summary", "raw"} <= set(d.keys())

    def test_event_type_is_class_name(self):
        ev = _make_sdk_event("ObservationEvent")
        d = _event_to_dict(ev)
        assert d["event_type"] == "ObservationEvent"

    def test_event_id_stringified(self):
        ev = _make_sdk_event("ActionEvent")
        ev.id = 12345
        d = _event_to_dict(ev)
        assert d["event_id"] == "12345"

    def test_raw_is_truncated(self):
        ev = _make_sdk_event("MessageEvent")
        ev.model_dump.return_value = {"big_field": "x" * 1000}
        d = _event_to_dict(ev)
        assert len(d["raw"]["big_field"]) <= 501  # truncated to 500 + "…"

    def test_model_dump_failure_still_produces_dict(self):
        ev = _make_sdk_event("ActionEvent")
        ev.model_dump.side_effect = AttributeError("no model_dump")
        d = _event_to_dict(ev)
        assert d["raw"] == {}   # graceful fallback


# ===========================================================================
# make_event_callback
# ===========================================================================

class TestEventCallback:
    @pytest.fixture
    def run_state(self):
        return RunState(session_id="cb-test")

    def test_callback_records_event(self, run_state):
        cb = make_event_callback(run_state)
        ev = _make_sdk_event("ActionEvent")
        cb(ev)
        assert len(run_state.events) == 1
        assert run_state.events[0]["event_type"] == "ActionEvent"

    def test_callback_updates_phase_on_action(self, run_state):
        cb = make_event_callback(run_state)
        cb(_make_sdk_event("ActionEvent"))
        assert run_state.phase == AgentPhase.EXECUTING_COMMAND

    def test_callback_updates_phase_on_observation(self, run_state):
        cb = make_event_callback(run_state)
        cb(_make_sdk_event("ObservationEvent"))
        assert run_state.phase == AgentPhase.WAITING_FOR_REPLY

    def test_callback_updates_phase_on_pause(self, run_state):
        cb = make_event_callback(run_state)
        cb(_make_sdk_event("PauseEvent"))
        assert run_state.phase == AgentPhase.PAUSED

    def test_callback_increments_iteration_on_action(self, run_state):
        cb = make_event_callback(run_state)
        cb(_make_sdk_event("ActionEvent"))
        cb(_make_sdk_event("ActionEvent"))
        assert run_state.iteration == 2

    def test_callback_does_not_increment_iteration_on_observation(self, run_state):
        cb = make_event_callback(run_state)
        cb(_make_sdk_event("ObservationEvent"))
        assert run_state.iteration == 0

    def test_callback_does_not_raise_on_broken_event(self, run_state):
        cb = make_event_callback(run_state)
        broken = MagicMock()
        broken.model_dump.side_effect = ValueError("broken")
        type(broken).__name__ = "WeirdEvent"
        # Should not raise
        cb(broken)

    def test_callback_does_not_change_phase_for_log_events(self, run_state):
        run_state.set_phase(AgentPhase.EXECUTING_COMMAND)
        cb = make_event_callback(run_state)
        cb(_make_sdk_event("LLMCompletionLogEvent"))
        # Phase should remain unchanged
        assert run_state.phase == AgentPhase.EXECUTING_COMMAND

    def test_multiple_callbacks_are_independent(self, run_state):
        """Two callbacks sharing the same RunState is safe."""
        cb1 = make_event_callback(run_state)
        cb2 = make_event_callback(run_state)
        cb1(_make_sdk_event("ActionEvent"))
        cb2(_make_sdk_event("ObservationEvent"))
        assert run_state.iteration == 1       # only ActionEvent increments
        assert len(run_state.events) == 2
