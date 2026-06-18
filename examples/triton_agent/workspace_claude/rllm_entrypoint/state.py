"""
state.py — Thread-safe RunState container.

RunState is the single source of truth for all observable data about
the currently-running entrypoint. Both the event collector (events.py)
and the background uploader (uploader.py) operate on this object.

Design choices:
  - A plain ``threading.Lock`` guards all mutations.
  - Immutable snapshots are produced via ``to_dict()`` / ``to_json()``
    without holding the lock.
  - Event history is stored as a list of plain dicts (serialisable).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentPhase(str, Enum):
    """High-level phase reported by the entrypoint."""
    INITIALIZING = "initializing"
    WAITING_FOR_REPLY = "waiting_for_reply"   # LLM call in flight
    EXECUTING_COMMAND = "executing_command"   # tool execution running
    PAUSED = "paused"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class RunState:
    """All observable state for one entrypoint execution."""

    # ── Identity ────────────────────────────────────────────────────────────
    session_id: str = ""
    session_label: str = ""

    # ── Task ────────────────────────────────────────────────────────────────
    task_instruction: str = ""
    workspace_base: str = ""
    llm_model: str = ""
    llm_base_url: str = ""        # proxied URL (may contain metadata slug)
    max_iterations: int = 0
    ops_name: str = ""
    ops_arch: str = ""

    # ── Runtime ─────────────────────────────────────────────────────────────
    is_running: bool = False
    phase: AgentPhase = AgentPhase.INITIALIZING
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    # Conversation-level counters
    iteration: int = 0
    total_llm_calls: int = 0
    accumulated_cost: float = 0.0

    # ── Event history ────────────────────────────────────────────────────────
    # Each element is a lightweight serialisable dict:
    #   {type, timestamp, source, summary, raw (optional)}
    events: list[dict[str, Any]] = field(default_factory=list)

    # Command history: list of {cmd, stdout, stderr, exit_code, timestamp}
    command_history: list[dict[str, Any]] = field(default_factory=list)

    # ── LLM context snapshot ─────────────────────────────────────────────────
    # Last N messages sent to the LLM (role + content preview).
    llm_context_snapshot: list[dict[str, str]] = field(default_factory=list)

    # ── OpenHands conversation state ─────────────────────────────────────────
    conversation_execution_status: str = "idle"   # mirrors ConversationExecutionStatus
    conversation_id: str = ""

    # ── Environment metadata ─────────────────────────────────────────────────
    env_vars: dict[str, str] = field(default_factory=dict)
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    # ── Error ────────────────────────────────────────────────────────────────
    last_error: str = ""

    # ── Internal lock (not serialised) ───────────────────────────────────────
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    # Maximum number of events / commands kept in-memory to avoid OOM
    _max_events: int = field(default=500, repr=False, compare=False)
    _max_commands: int = field(default=200, repr=False, compare=False)

    # ── Mutation helpers (all under lock) ────────────────────────────────────

    def set_phase(self, phase: AgentPhase) -> None:
        with self._lock:
            self.phase = phase

    def set_running(self, running: bool) -> None:
        with self._lock:
            self.is_running = running
            if not running and self.end_time is None:
                self.end_time = time.time()

    def set_conversation_status(self, status: str, conv_id: str = "") -> None:
        with self._lock:
            self.conversation_execution_status = status
            if conv_id:
                self.conversation_id = conv_id

    def increment_iteration(self) -> None:
        with self._lock:
            self.iteration += 1

    def record_event(self, event_dict: dict[str, Any]) -> None:
        """Append a serialised event; trim to _max_events."""
        with self._lock:
            self.events.append(event_dict)
            if len(self.events) > self._max_events:
                self.events = self.events[-self._max_events:]

    def record_command(self, cmd_dict: dict[str, Any]) -> None:
        """Append a command execution record; trim to _max_commands."""
        with self._lock:
            self.command_history.append(cmd_dict)
            if len(self.command_history) > self._max_commands:
                self.command_history = self.command_history[-self._max_commands:]

    def update_llm_context(self, messages: list[dict[str, str]]) -> None:
        with self._lock:
            self.llm_context_snapshot = messages[-50:]   # last 50 messages

    def update_metrics(self, cost: float, llm_calls: int) -> None:
        with self._lock:
            self.accumulated_cost = cost
            self.total_llm_calls = llm_calls

    def set_error(self, error: str) -> None:
        with self._lock:
            self.last_error = error
            self.phase = AgentPhase.ERROR

    # ── Snapshot ─────────────────────────────────────────────────────────────

    def uptime_seconds(self) -> float:
        end = self.end_time if self.end_time else time.time()
        return round(end - self.start_time, 2)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot (does NOT hold lock)."""
        # We read fields while briefly holding the lock to get a consistent view.
        with self._lock:
            return {
                "session_id": self.session_id,
                "session_label": self.session_label,
                "task_instruction": self.task_instruction,
                "workspace_base": self.workspace_base,
                "llm_model": self.llm_model,
                "llm_base_url_prefix": self.llm_base_url,
                "max_iterations": self.max_iterations,
                "is_running": self.is_running,
                "phase": self.phase.value if isinstance(self.phase, AgentPhase) else self.phase,
                "uptime_seconds": self.uptime_seconds(),
                "start_time": self.start_time,
                "end_time": self.end_time,
                "iteration": self.iteration,
                "total_llm_calls": self.total_llm_calls,
                "accumulated_cost": self.accumulated_cost,
                "conversation_execution_status": self.conversation_execution_status,
                "conversation_id": self.conversation_id,
                "events_count": len(self.events),
                "events": list(self.events),           # full history
                "command_history": list(self.command_history),
                "llm_context_snapshot": list(self.llm_context_snapshot),
                "env_vars": dict(self.env_vars),
                "extra_metadata": dict(self.extra_metadata),
                "last_error": self.last_error,
            }

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), default=str)
