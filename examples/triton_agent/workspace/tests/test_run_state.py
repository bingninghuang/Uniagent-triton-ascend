"""
test_run_state.py — Unit tests for rllm_entrypoint/state.py (RunState).

Tests cover:
  - Default field values
  - set_phase / set_running / set_conversation_status / increment_iteration
  - record_event ring-buffer trim (max_events)
  - record_command ring-buffer trim (max_commands)
  - update_llm_context (keeps last 50)
  - update_metrics
  - set_error (sets phase to ERROR + last_error)
  - to_dict snapshot structure and field constraints
  - to_json round-trips cleanly
  - uptime_seconds logic
  - Thread-safety smoke test for concurrent record_event
"""
from __future__ import annotations

import json
import sys
import os
import threading
import time

import pytest

# ---------------------------------------------------------------------------
# Path setup so we can import rllm_entrypoint from the workspace directory
# ---------------------------------------------------------------------------
_WORKSPACE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "workspace",
)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from rllm_entrypoint.state import AgentPhase, RunState  # noqa: E402


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def state():
    return RunState(
        session_id="test-session",
        session_label="test-label",
    )


# ===========================================================================
# Initial defaults
# ===========================================================================

class TestDefaults:
    def test_phase_is_initializing(self, state):
        assert state.phase == AgentPhase.INITIALIZING

    def test_is_running_false(self, state):
        assert state.is_running is False

    def test_iteration_zero(self, state):
        assert state.iteration == 0

    def test_events_empty(self, state):
        assert state.events == []

    def test_command_history_empty(self, state):
        assert state.command_history == []

    def test_end_time_none(self, state):
        assert state.end_time is None

    def test_accumulated_cost_zero(self, state):
        assert state.accumulated_cost == 0.0

    def test_last_error_empty(self, state):
        assert state.last_error == ""


# ===========================================================================
# Mutation helpers
# ===========================================================================

class TestMutationHelpers:
    def test_set_phase(self, state):
        state.set_phase(AgentPhase.EXECUTING_COMMAND)
        assert state.phase == AgentPhase.EXECUTING_COMMAND

    def test_set_running_true(self, state):
        state.set_running(True)
        assert state.is_running is True
        assert state.end_time is None   # end_time stays None while running

    def test_set_running_false_records_end_time(self, state):
        state.set_running(True)
        state.set_running(False)
        assert state.end_time is not None
        assert state.end_time >= state.start_time

    def test_set_running_false_does_not_overwrite_end_time(self, state):
        state.set_running(True)
        state.set_running(False)
        t1 = state.end_time
        time.sleep(0.01)
        state.set_running(False)
        assert state.end_time == t1   # not overwritten

    def test_set_conversation_status(self, state):
        state.set_conversation_status("running", "conv-123")
        assert state.conversation_execution_status == "running"
        assert state.conversation_id == "conv-123"

    def test_set_conversation_status_no_id(self, state):
        state.set_conversation_status("finished")
        assert state.conversation_execution_status == "finished"
        assert state.conversation_id == ""   # unchanged

    def test_increment_iteration(self, state):
        for i in range(5):
            state.increment_iteration()
        assert state.iteration == 5

    def test_update_metrics(self, state):
        state.update_metrics(cost=1.23, llm_calls=7)
        assert state.accumulated_cost == 1.23
        assert state.total_llm_calls == 7

    def test_set_error(self, state):
        state.set_error("something went wrong")
        assert state.last_error == "something went wrong"
        assert state.phase == AgentPhase.ERROR


# ===========================================================================
# Ring-buffer / trim behaviour
# ===========================================================================

class TestRingBuffer:
    def test_events_trimmed_to_max(self, state):
        state._max_events = 10
        for i in range(25):
            state.record_event({"event_id": f"e{i}", "event_type": "ActionEvent"})
        assert len(state.events) == 10
        # Should keep the most recent 10
        assert state.events[-1]["event_id"] == "e24"
        assert state.events[0]["event_id"] == "e15"

    def test_commands_trimmed_to_max(self, state):
        state._max_commands = 5
        for i in range(12):
            state.record_command({"cmd": f"echo {i}", "exit_code": 0})
        assert len(state.command_history) == 5
        assert state.command_history[-1]["cmd"] == "echo 11"

    def test_llm_context_keeps_last_50(self, state):
        messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
                    for i in range(80)]
        state.update_llm_context(messages)
        assert len(state.llm_context_snapshot) == 50
        assert state.llm_context_snapshot[-1]["content"] == "msg79"


# ===========================================================================
# Snapshot (to_dict / to_json)
# ===========================================================================

class TestSnapshot:
    def test_to_dict_contains_required_keys(self, state):
        d = state.to_dict()
        required = {
            "session_id", "session_label", "is_running", "phase",
            "iteration", "total_llm_calls", "accumulated_cost",
            "events", "command_history", "llm_context_snapshot",
            "conversation_execution_status", "conversation_id",
            "uptime_seconds", "start_time", "end_time",
            "task_instruction", "workspace_base", "llm_model",
            "max_iterations", "env_vars", "extra_metadata", "last_error",
        }
        missing = required - set(d.keys())
        assert missing == set(), f"Missing keys: {missing}"

    def test_phase_serialised_as_string(self, state):
        state.set_phase(AgentPhase.EXECUTING_COMMAND)
        d = state.to_dict()
        assert d["phase"] == "executing_command"

    def test_task_instruction_truncated_to_500(self, state):
        state.task_instruction = "A" * 1000
        d = state.to_dict()
        assert len(d["task_instruction"]) == 500

    def test_llm_base_url_prefix_truncated_to_80(self, state):
        state.llm_base_url = "http://example.com/" + "x" * 200
        d = state.to_dict()
        assert len(d["llm_base_url_prefix"]) == 80

    def test_events_count_matches(self, state):
        for i in range(3):
            state.record_event({"event_id": str(i), "event_type": "ActionEvent"})
        d = state.to_dict()
        assert d["events_count"] == 3
        assert len(d["events"]) == 3

    def test_to_json_round_trips(self, state):
        state.record_event({"event_id": "e1", "event_type": "ActionEvent"})
        state.update_metrics(0.5, 2)
        raw = state.to_json()
        parsed = json.loads(raw)
        assert parsed["accumulated_cost"] == 0.5
        assert parsed["events_count"] == 1

    def test_uptime_seconds_positive(self, state):
        time.sleep(0.05)
        d = state.to_dict()
        assert d["uptime_seconds"] >= 0.0

    def test_uptime_seconds_freezes_when_stopped(self, state):
        state.set_running(True)
        time.sleep(0.02)
        state.set_running(False)
        t1 = state.uptime_seconds()
        time.sleep(0.05)
        t2 = state.uptime_seconds()
        assert t1 == t2   # stopped → uptime is frozen


# ===========================================================================
# Thread-safety smoke test
# ===========================================================================

class TestThreadSafety:
    def test_concurrent_record_event(self, state):
        n_threads = 10
        events_per_thread = 50
        state._max_events = n_threads * events_per_thread + 1  # no trim

        errors: list[Exception] = []

        def worker(tid: int):
            try:
                for i in range(events_per_thread):
                    state.record_event({
                        "event_id": f"t{tid}-e{i}",
                        "event_type": "ActionEvent",
                    })
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == []
        assert len(state.events) == n_threads * events_per_thread

    def test_concurrent_phase_update_and_snapshot(self, state):
        """Snapshot (to_dict) under concurrent mutations should not deadlock."""
        stop = threading.Event()
        errors: list[Exception] = []

        def mutator():
            phases = list(AgentPhase)
            idx = 0
            while not stop.is_set():
                try:
                    state.set_phase(phases[idx % len(phases)])
                    idx += 1
                    time.sleep(0.001)
                except Exception as exc:
                    errors.append(exc)

        def reader():
            while not stop.is_set():
                try:
                    state.to_dict()
                    time.sleep(0.001)
                except Exception as exc:
                    errors.append(exc)

        threads = [
            threading.Thread(target=mutator),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        time.sleep(0.3)
        stop.set()
        for t in threads:
            t.join(timeout=5)

        assert errors == []
