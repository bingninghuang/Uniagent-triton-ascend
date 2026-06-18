"""
events.py — OpenHands event hook that drives event-driven push to the
observer gateway.

Design (v2 — event-driven):
  Each OpenHands SDK event is immediately serialised and pushed to the
  gateway via ObserverClient.push_event().  There is no background upload
  thread; the push happens synchronously inside the callback (with a short
  timeout so it never stalls the agent loop).

  Additional lifecycle helpers are provided for structured events that are
  NOT native OpenHands SDK events:
    - push_startup_event()     — emitted once at container start
    - push_heartbeat_event()   — emitted periodically by a lightweight timer
    - push_evaluate_event()    — emitted when evaluation metrics are available
    - push_finish_event()      — emitted when the conversation ends

Usage:
    from rllm_entrypoint.events import make_event_callback, push_startup_event
    cb = make_event_callback(run_state, client)
    conversation = Conversation(agent=agent, workspace=..., callbacks=[cb])
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable

from rllm_entrypoint.state import AgentPhase, RunState

if TYPE_CHECKING:
    from rllm_entrypoint.api_client import ObserverClient

logger = logging.getLogger(__name__)

_EVENT_SUMMARY_MAX_CHARS = int(os.environ.get("OPENHANDS_OBSERVER_SUMMARY_MAX_CHARS", "2000"))
_EVENT_RAW_MAX_CHARS = int(os.environ.get("OPENHANDS_OBSERVER_RAW_MAX_CHARS", "2000"))
_EVENT_INCLUDE_RAW = os.environ.get("OPENHANDS_OBSERVER_INCLUDE_RAW", "0") in ("1", "true", "True", "yes")
_FINISH_INCLUDE_FINAL_STATE = os.environ.get("OPENHANDS_OBSERVER_INCLUDE_FINAL_STATE", "0") in ("1", "true", "True", "yes")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_str(v: Any, max_len: int = 0) -> str:
    """Return str(v), optionally truncated."""
    try:
        text = str(v)
    except Exception:
        text = "<repr failed>"
    limit = max_len or _EVENT_SUMMARY_MAX_CHARS
    if limit > 0 and len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text


def _truncate_dict(d: Any, max_str_len: int = 0) -> Any:
    """Recursively truncate large strings in event payloads."""
    limit = max_str_len or _EVENT_RAW_MAX_CHARS
    if isinstance(d, dict):
        return {k: _truncate_dict(v, max_str_len) for k, v in d.items()}
    if isinstance(d, list):
        return [_truncate_dict(v, max_str_len) for v in d]
    if isinstance(d, str) and limit > 0 and len(d) > limit:
        return d[:limit] + "...[truncated]"
    return d


def _make_lifecycle_event(event_type: str, **kwargs: Any) -> dict[str, Any]:
    """Build a structured lifecycle event dict (non-SDK events)."""
    return {
        "event_type": event_type,
        "event_id": str(uuid.uuid4()),
        "source": "rllm_entrypoint",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": f"[{event_type}]",
        **kwargs,
    }

# ---------------------------------------------------------------------------
# SDK event serialisation
# ---------------------------------------------------------------------------

def _event_to_dict(event: Any) -> dict[str, Any]:  # noqa: ANN401
    """Convert any OpenHands SDK Event to a flat serialisable dict."""
    raw = {}
    if _EVENT_INCLUDE_RAW:
        # Try pydantic model_dump first (SDK Events are Pydantic models).
        try:
            raw = event.model_dump()
        except AttributeError:
            raw = {}

    event_type = type(event).__name__
    source = getattr(event, "source", "unknown")
    timestamp = getattr(event, "timestamp", None) or time.time()
    event_id = getattr(event, "id", None) or ""

    # Build a human-readable summary per event type.
    summary = _build_summary(event, event_type)

    return {
        "event_type": event_type,
        "event_id": str(event_id),
        "source": str(source),
        "timestamp": str(timestamp),
        "summary": _safe_str(summary, _EVENT_SUMMARY_MAX_CHARS),
        "raw": _truncate_dict(raw, _EVENT_RAW_MAX_CHARS) if _EVENT_INCLUDE_RAW else {},
    }


def _build_summary(event: Any, event_type: str) -> str:
    """Build a concise one-line summary for the event."""
    try:
        # ActionEvent: show tool name + thought preview
        if event_type == "ActionEvent":
            tool = getattr(event, "tool_name", "")
            thought = getattr(event, "thought", [])
            thought_text = " ".join(
                getattr(t, "text", str(t)) for t in thought
            )
            return f"[Action:{tool}] {thought_text}"

        # ObservationEvent: show tool + result preview
        if event_type == "ObservationEvent":
            tool = getattr(event, "tool_name", "")
            obs = getattr(event, "observation", None)
            if obs is not None:
                try:
                    content_list = obs.to_llm_content
                    text = " ".join(
                        getattr(c, "text", str(c)) for c in content_list
                    )
                except Exception:
                    text = _safe_str(obs, _EVENT_SUMMARY_MAX_CHARS)
            else:
                text = ""
            return _safe_str(f"[Observation:{tool}] {text}", _EVENT_SUMMARY_MAX_CHARS)

        # MessageEvent
        if event_type == "MessageEvent":
            msg = getattr(event, "llm_message", None)
            if msg is not None:
                content = getattr(msg, "content", [])
                text = " ".join(
                    getattr(c, "text", str(c)) for c in content
                )
                role = getattr(msg, "role", "?")
                return _safe_str(f"[Message:{role}] {text}", _EVENT_SUMMARY_MAX_CHARS)

        # AgentErrorEvent
        if event_type == "AgentErrorEvent":
            return f"[Error] {getattr(event, 'error', '')}"

        # PauseEvent
        if event_type == "PauseEvent":
            return "[Paused]"

        # ConversationStateUpdateEvent
        if event_type == "ConversationStateUpdateEvent":
            key = getattr(event, "key", "")
            value = getattr(event, "value", "")
            return f"[StateUpdate] {key}={_safe_str(value, 80)}"

        # LLMCompletionLogEvent
        if event_type == "LLMCompletionLogEvent":
            return "[LLMCompletion]"

        return f"[{event_type}]"
    except Exception:
        return f"[{event_type}] <summary failed>"


def _phase_from_event(event_type: str) -> AgentPhase | None:
    """Map event type → AgentPhase transition (None = no change)."""
    _map = {
        "ActionEvent": AgentPhase.EXECUTING_COMMAND,
        "ObservationEvent": AgentPhase.WAITING_FOR_REPLY,
        "MessageEvent": AgentPhase.WAITING_FOR_REPLY,
        "PauseEvent": AgentPhase.PAUSED,
    }
    return _map.get(event_type)

# ---------------------------------------------------------------------------
# Main callback factory
# ---------------------------------------------------------------------------

def make_event_callback(
    run_state: RunState,
    client: "ObserverClient",
) -> Callable[[Any], None]:
    """Return a callback suitable for Conversation(callbacks=[...]).

    Each event is:
      1. Serialised to a dict.
      2. Recorded in RunState (for local state tracking).
      3. Immediately pushed to the gateway via client.push_event().
    """

    def on_event(event: Any) -> None:  # noqa: ANN401
        try:
            event_dict = _event_to_dict(event)
            run_state.record_event(event_dict)

            # Phase transitions
            phase = _phase_from_event(event_dict["event_type"])
            if phase is not None:
                run_state.set_phase(phase)

            # Track iterations: count ActionEvents
            if event_dict["event_type"] == "ActionEvent":
                run_state.increment_iteration()

            # ── Direct push to gateway (event-driven) ──────────────────
            client.push_event(event_dict)

        except Exception:
            # Never crash the agent loop
            pass

    return on_event

# ---------------------------------------------------------------------------
# Lifecycle event helpers  (call from runner.py at key moments)
# ---------------------------------------------------------------------------

def push_startup_event(client: "ObserverClient", run_state: RunState) -> None:
    """Emit a 'startup' lifecycle event when the container starts.

    Carries the full static config so the gateway can populate the session
    record and an initial state_snapshot without a separate /state push.
    """
    with run_state._lock:
        ev = _make_lifecycle_event(
            "StartupEvent",
            summary="[Startup] Container initialised and runner started",
            # Identity
            session_id=run_state.session_id,
            session_label=run_state.session_label,
            # Config
            llm_model=run_state.llm_model,
            llm_base_url=run_state.llm_base_url,
            workspace_base=run_state.workspace_base,
            max_iterations=run_state.max_iterations,
            # Task
            task_preview=run_state.task_instruction,
            task_instruction=run_state.task_instruction,
            # Environment
            env_vars=dict(run_state.env_vars),
            extra_metadata=dict(run_state.extra_metadata),
            # Timing
            start_time=run_state.start_time,
        )
    logger.info("[events] Pushing StartupEvent")
    client.push_event(ev)


def push_heartbeat_event(client: "ObserverClient", run_state: RunState) -> None:
    """Emit a lightweight heartbeat (call from a periodic timer in runner)."""
    with run_state._lock:
        phase = run_state.phase.value if hasattr(run_state.phase, "value") else str(run_state.phase)
        iteration = run_state.iteration
        is_running = run_state.is_running
        uptime = run_state.uptime_seconds() if callable(getattr(run_state, "uptime_seconds", None)) else 0
        conv_id = run_state.conversation_id
        conv_status = run_state.conversation_execution_status
        cost = run_state.accumulated_cost
        llm_calls = run_state.total_llm_calls

    ev = _make_lifecycle_event(
        "HeartbeatEvent",
        summary=f"[Heartbeat] phase={phase} iteration={iteration} running={is_running}",
        phase=phase,
        iteration=iteration,
        is_running=is_running,
        uptime_seconds=uptime,
        conversation_id=conv_id,
        conversation_execution_status=conv_status,
        accumulated_cost=cost,
        total_llm_calls=llm_calls,
    )
    logger.debug("[events] Pushing HeartbeatEvent (phase=%s iter=%d uptime=%.1fs)", phase, iteration, uptime)
    client.push_event(ev)


def push_evaluate_event(
    client: "ObserverClient",
    run_state: RunState,
    *,
    status: str,
    cost: float = 0.0,
    llm_calls: int = 0,
    iterations: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit an evaluation/summary event at conversation end.

    Carries the full command_history and llm_context_snapshot so the
    gateway has a complete picture of what the agent did.
    """
    with run_state._lock:
        conv_id = run_state.conversation_id
        conv_status = run_state.conversation_execution_status
        cmd_history = list(run_state.command_history)
        llm_ctx = list(run_state.llm_context_snapshot)
        last_error = run_state.last_error
        uptime = run_state.uptime_seconds() if callable(getattr(run_state, "uptime_seconds", None)) else 0

    ev = _make_lifecycle_event(
        "EvaluateEvent",
        summary=f"[Evaluate] status={status} cost={cost:.4f} llm_calls={llm_calls} iterations={iterations}",
        status=status,
        accumulated_cost=cost,
        total_llm_calls=llm_calls,
        iterations=iterations,
        conversation_id=conv_id,
        conversation_execution_status=conv_status,
        command_history=cmd_history,
        llm_context_snapshot=llm_ctx,
        last_error=last_error,
        uptime_seconds=uptime,
        **(extra or {}),
    )
    logger.info("[events] Pushing EvaluateEvent (status=%s cost=%.4f llm_calls=%d)", status, cost, llm_calls)
    client.push_event(ev)


def push_finish_event(
    client: "ObserverClient",
    run_state: RunState,
    *,
    exit_code: int,
    reason: str = "",
) -> None:
    """Emit a 'finish' lifecycle event as the last thing the container does.

    Carries the complete final RunState snapshot so the gateway has a
    full picture even if the container exits immediately after.
    """
    # Grab the full state snapshot under the lock
    try:
        state_snapshot = run_state.to_dict()
    except Exception:
        state_snapshot = {}

    phase = state_snapshot.get("phase", "finished")
    iteration = state_snapshot.get("iteration", 0)

    ev = _make_lifecycle_event(
        "FinishEvent",
        summary=f"[Finish] exit_code={exit_code} phase={phase} reason={reason}",
        exit_code=exit_code,
        phase=phase,
        iteration=iteration,
        reason=reason,
    )
    if _FINISH_INCLUDE_FINAL_STATE:
        ev["final_state"] = state_snapshot
    logger.info("[events] Pushing FinishEvent (exit_code=%d, reason=%s)", exit_code, reason)
    client.push_event(ev)
