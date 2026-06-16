"""
observer_gateway.py — Standalone REST API server that receives telemetry
from rllm OpenHands containers and serves the observability dashboard.

Run with:
    python observer_gateway.py [--host HOST] [--port PORT] [--db PATH]

Default: http://0.0.0.0:18858   db: ./observer_data.db

Dashboard: http://localhost:18858/

─────────────────────────────────────────────────────────────────────────────
REST API
─────────────────────────────────────────────────────────────────────────────

Static / Dashboard
  GET    /                                       Redirect to /dashboard/
  GET    /dashboard/*                            Serve dashboard SPA static files

Sessions
  GET    /api/v1/sessions                        List all sessions (enriched summary)
         ?status=running|finished|error|paused|all
  PATCH  /api/v1/sessions/{id}                   Update session label
  DELETE /api/v1/sessions/{id}                   Delete session + all data

Per-session: live state (latest in-memory)
  GET    /api/v1/sessions/{id}/state             Latest state snapshot
  POST   /api/v1/sessions/{id}/state             Upload state (from container)

Per-session: historical snapshots (persisted)
  GET    /api/v1/sessions/{id}/snapshots         Snapshot index (metadata only)
         ?limit=N&since=<unix_ts>
  GET    /api/v1/sessions/{id}/snapshots/{snap_id}
                                                 Full payload of one snapshot

Per-session: events (persisted)
  GET    /api/v1/sessions/{id}/events            Event list
         ?limit=N&offset=N&type=<event_type>&since=<unix_ts>&visible=true|false|all
  POST   /api/v1/sessions/{id}/events            Upload single event (from container)
  GET    /api/v1/sessions/{id}/events/{db_id}/payload
                                                 Full raw payload of one event
  PATCH  /api/v1/sessions/{id}/events/{db_id}    Toggle visibility {"visible": bool}

Per-session: real-time stream
  GET    /api/v1/sessions/{id}/stream            SSE stream (event-driven updates)

Per-session: KPI dashboard
  GET    /api/v1/sessions/{id}/kpi               Aggregated KPI panel data

Per-session: control (in-memory + persisted log)
  GET    /api/v1/sessions/{id}/control           Current pause/resume flags
  POST   /api/v1/sessions/{id}/control           Set {"pause":bool}|{"resume":bool}
  GET    /api/v1/sessions/{id}/control/log       Control action history
         ?limit=N&since=<unix_ts>

Per-session: request log (persisted)
  GET    /api/v1/sessions/{id}/requests          HTTP request log for this session
         ?limit=N&since=<unix_ts>&method=GET|POST|...

Per-session: stats
  GET    /api/v1/sessions/{id}/stats             Counts by category + event breakdown

Global
  GET    /api/v1/stats                           Global counts
  GET    /api/v1/requests                        All request log entries
         ?limit=N&since=<unix_ts>&method=GET|POST|...

─────────────────────────────────────────────────────────────────────────────
Dependencies: stdlib only. No external packages required.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import queue
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Module-level DB handle (set by serve())
_db: Any = None  # GatewayDB instance

# Dashboard static file directory (set by serve())
_DASHBOARD_DIR: Path | None = None

# ---------------------------------------------------------------------------
# In-memory live-state store (fast reads; DB has the full history)
# ---------------------------------------------------------------------------

_SESSION_LOCK = threading.Lock()

# session_id → {state, events (recent), control, updated_at, created_at}
_SESSIONS: dict[str, dict[str, Any]] = {}
_MAX_MEM_EVENTS = 500   # events kept in RAM per session

# ---------------------------------------------------------------------------
# SSE subscriber registry
# ---------------------------------------------------------------------------
# session_id → list of queue.Queue (one per connected SSE client)
_SSE_LOCK = threading.Lock()
_SSE_SUBSCRIBERS: dict[str, list[queue.Queue]] = {}


def _sse_subscribe(session_id: str) -> queue.Queue:
    q: queue.Queue = queue.Queue(maxsize=200)
    with _SSE_LOCK:
        _SSE_SUBSCRIBERS.setdefault(session_id, []).append(q)
    return q


def _sse_unsubscribe(session_id: str, q: queue.Queue) -> None:
    with _SSE_LOCK:
        subs = _SSE_SUBSCRIBERS.get(session_id, [])
        try:
            subs.remove(q)
        except ValueError:
            pass


def _sse_broadcast(session_id: str, event_name: str, data: Any) -> None:
    payload = json.dumps(data, default=str)
    msg = f"event: {event_name}\ndata: {payload}\n\n"
    with _SSE_LOCK:
        subs = list(_SSE_SUBSCRIBERS.get(session_id, []))
    for q in subs:
        try:
            q.put_nowait(msg)
        except queue.Full:
            pass  # slow client — drop the message


def _get_or_create(session_id: str) -> dict[str, Any]:
    with _SESSION_LOCK:
        if session_id not in _SESSIONS:
            _SESSIONS[session_id] = {
                "state": {},
                "events": [],
                "control": {"pause": False, "resume": False},
                "updated_at": time.time(),
                "created_at": time.time(),
            }
        return _SESSIONS[session_id]


def _merge_lifecycle_into_state(
    sess: dict[str, Any],
    body: dict[str, Any],
    event_type: str,
) -> None:
    """Update the in-memory live state dict from a lifecycle event.

    Called while _SESSION_LOCK is held.  Keeps GET /state fresh without
    a dedicated state-push endpoint.
    """
    s = sess.setdefault("state", {})
    if event_type == "StartupEvent":
        s.update({
            "session_id":        body.get("session_id", s.get("session_id", "")),
            "session_label":     body.get("session_label", s.get("session_label", "")),
            "llm_model":         body.get("llm_model", ""),
            "llm_base_url_prefix": body.get("llm_base_url", "")[:80],
            "workspace_base":    body.get("workspace_base", ""),
            "max_iterations":    body.get("max_iterations", 0),
            "task_instruction":  body.get("task_instruction", body.get("task_preview", "")),
            "env_vars":          body.get("env_vars", {}),
            "extra_metadata":    body.get("extra_metadata", {}),
            "start_time":        body.get("start_time", time.time()),
            "phase":             "initializing",
            "is_running":        True,
            "iteration":         0,
        })
    elif event_type == "HeartbeatEvent":
        s.update({
            "phase":                          body.get("phase", s.get("phase", "")),
            "iteration":                      body.get("iteration", s.get("iteration", 0)),
            "is_running":                     body.get("is_running", s.get("is_running", False)),
            "uptime_seconds":                 body.get("uptime_seconds", 0),
            "conversation_id":                body.get("conversation_id", s.get("conversation_id", "")),
            "conversation_execution_status":  body.get("conversation_execution_status", ""),
            "accumulated_cost":               body.get("accumulated_cost", s.get("accumulated_cost", 0.0)),
            "total_llm_calls":                body.get("total_llm_calls", s.get("total_llm_calls", 0)),
        })
    elif event_type == "EvaluateEvent":
        s.update({
            "phase":                          "finished",
            "conversation_execution_status":  body.get("status", ""),
            "accumulated_cost":               body.get("accumulated_cost", 0.0),
            "total_llm_calls":                body.get("total_llm_calls", 0),
            "iteration":                      body.get("iterations", s.get("iteration", 0)),
            "conversation_id":                body.get("conversation_id", s.get("conversation_id", "")),
            "command_history":                body.get("command_history", []),
            "llm_context_snapshot":           body.get("llm_context_snapshot", []),
            "last_error":                     body.get("last_error", ""),
            "uptime_seconds":                 body.get("uptime_seconds", 0),
        })
    elif event_type == "FinishEvent":
        # FinishEvent embeds the full final RunState.to_dict() snapshot
        final = body.get("final_state", {})
        if final:
            # Prefer the embedded full snapshot; only patch exit_code/reason on top
            s.update(final)
        s.update({
            "is_running": False,
            "exit_code":  body.get("exit_code", 0),
            "last_error": body.get("reason", "") if body.get("exit_code", 0) != 0 else s.get("last_error", ""),
        })


# ---------------------------------------------------------------------------
# URL routing
# ---------------------------------------------------------------------------

_R_ROOT       = re.compile(r"^/?$")
_R_DASHBOARD  = re.compile(r"^/dashboard(/.*)?$")
_R_SESSIONS   = re.compile(r"^/api/v1/sessions$")
_R_SESSION    = re.compile(r"^/api/v1/sessions/([^/]+)$")
_R_STATE      = re.compile(r"^/api/v1/sessions/([^/]+)/state$")
_R_SNAPSHOTS  = re.compile(r"^/api/v1/sessions/([^/]+)/snapshots$")
_R_SNAPSHOT1  = re.compile(r"^/api/v1/sessions/([^/]+)/snapshots/(\d+)$")
_R_EVENTS     = re.compile(r"^/api/v1/sessions/([^/]+)/events$")
_R_EVENT1     = re.compile(r"^/api/v1/sessions/([^/]+)/events/(\d+)/payload$")
_R_EVENT_PATCH = re.compile(r"^/api/v1/sessions/([^/]+)/events/(\d+)$")
_R_STREAM     = re.compile(r"^/api/v1/sessions/([^/]+)/stream$")
_R_KPI        = re.compile(r"^/api/v1/sessions/([^/]+)/kpi$")
_R_CONTROL    = re.compile(r"^/api/v1/sessions/([^/]+)/control$")
_R_CTRL_LOG   = re.compile(r"^/api/v1/sessions/([^/]+)/control/log$")
_R_REQ_SESS   = re.compile(r"^/api/v1/sessions/([^/]+)/requests$")
_R_STATS_SESS = re.compile(r"^/api/v1/sessions/([^/]+)/stats$")
_R_STATS_GLOB = re.compile(r"^/api/v1/stats$")
_R_REQS_GLOB  = re.compile(r"^/api/v1/requests$")


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class ObserverHandler(BaseHTTPRequestHandler):
    """HTTP handler for the observer gateway."""

    server_version = "ObserverGateway/2.0"

    def log_message(self, fmt, *args):  # type: ignore[override]
        pass   # We log via the request_log table

    # ── Helpers ───────────────────────────────────────────────────────────

    def _send_json(self, code: int, body: Any, *, _ts_start: float | None = None) -> None:
        data = json.dumps(body, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)
        self._response_code = code
        self._ts_end = time.time()

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    def _parse_json_body(self) -> dict[str, Any] | None:
        try:
            raw = self._read_body()
            self._body_bytes = len(raw)
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return None

    def _qp(self, name: str, default: str = "") -> str:
        """Parse a single query parameter."""
        if "?" in self.path:
            qs = self.path.split("?", 1)[1]
            for part in qs.split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    if k == name:
                        return v
        return default

    def _qp_int(self, name: str, default: int) -> int:
        try:
            return int(self._qp(name, str(default)))
        except ValueError:
            return default

    def _qp_float(self, name: str, default: float | None = None) -> float | None:
        v = self._qp(name)
        if not v:
            return default
        try:
            return float(v)
        except ValueError:
            return default

    def _qp_bool_or_none(self, name: str) -> bool | None:
        """Parse a tri-state bool param: 'true'→True, 'false'→False, else→None."""
        v = self._qp(name, "").lower()
        if v == "true":
            return True
        if v == "false":
            return False
        return None

    def _client_ip(self) -> str:
        return self.client_address[0] if self.client_address else ""

    # ── Lifecycle: log every request to DB ────────────────────────────────

    def handle_one_request(self) -> None:
        self._ts_start = time.time()
        self._body_bytes = 0
        self._response_code = 0
        self._ts_end = self._ts_start
        self._session_id_for_log: str | None = None
        # NOTE: raw_requestline / self.path / self.command are ALL populated
        # inside super().handle_one_request(). Nothing can be logged before that.
        try:
            super().handle_one_request()
        finally:
            elapsed_ms = (self._ts_end - self._ts_start) * 1000
            # getattr guards: parse_request() may not have run if connection was bad
            _cmd  = getattr(self, "command", None) or "?"
            _path = getattr(self, "path",    None) or "?"
            _path_clean = _path.split("?")[0] if "?" in _path else _path
            logger.debug(
                "%s %s  status=%d  body=%db  %.1fms  client=%s",
                _cmd,
                _path_clean,
                self._response_code,
                self._body_bytes,
                elapsed_ms,
                self._client_ip(),
            )
            # Don't log SSE connections and dashboard static requests to request_log
            _skip_log = (
                _path_clean.startswith("/dashboard") or
                "/stream" in _path_clean or
                _path_clean in ("/", "")
            )
            if _db is not None and not _skip_log:
                try:
                    _db.log_request(
                        method=_cmd,
                        path=_path_clean,
                        client_ip=self._client_ip(),
                        received_at=self._ts_start,
                        status_code=self._response_code,
                        body_bytes=self._body_bytes,
                        response_ms=elapsed_ms,
                        session_id=self._session_id_for_log,
                    )
                except Exception:
                    pass

    # ── OPTIONS (CORS) ────────────────────────────────────────────────────

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self._response_code = 204
        self._ts_end = time.time()

    # ── GET ───────────────────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: C901
        path = self.path.split("?")[0]

        # Root redirect → /dashboard/
        if _R_ROOT.match(path):
            self.send_response(302)
            self.send_header("Location", "/dashboard/")
            self.end_headers()
            self._response_code = 302
            self._ts_end = time.time()
            return

        # Dashboard static files
        if _R_DASHBOARD.match(path):
            self._handle_dashboard(path)
            return

        # GET /api/v1/sessions
        if _R_SESSIONS.match(path):
            self._handle_list_sessions()
            return

        # GET /api/v1/stats
        if _R_STATS_GLOB.match(path):
            stats = _db.get_global_stats() if _db else {"error": "no db"}
            self._send_json(200, stats)
            return

        # GET /api/v1/requests
        if _R_REQS_GLOB.match(path):
            self._handle_list_requests(session_id=None)
            return

        m = _R_STATE.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_get_state(m.group(1))
            return

        m = _R_SNAPSHOTS.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_list_snapshots(m.group(1))
            return

        m = _R_SNAPSHOT1.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_get_snapshot(m.group(1), int(m.group(2)))
            return

        m = _R_STREAM.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_sse_stream(m.group(1))
            return

        m = _R_KPI.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_get_kpi(m.group(1))
            return

        m = _R_EVENTS.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_list_events(m.group(1))
            return

        m = _R_EVENT1.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_get_event_payload(m.group(1), int(m.group(2)))
            return

        m = _R_CONTROL.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_get_control(m.group(1))
            return

        m = _R_CTRL_LOG.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_list_control_log(m.group(1))
            return

        m = _R_REQ_SESS.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_list_requests(session_id=m.group(1))
            return

        m = _R_STATS_SESS.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            stats = _db.get_session_stats(m.group(1)) if _db else {}
            self._send_json(200, stats)
            return

        self._send_json(404, {"error": "not found"})

    # ── POST ──────────────────────────────────────────────────────────────

    def do_POST(self) -> None:  # noqa: C901
        path = self.path.split("?")[0]

        m = _R_STATE.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_post_state(m.group(1))
            return

        m = _R_EVENTS.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_post_event(m.group(1))
            return

        m = _R_CONTROL.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_post_control(m.group(1))
            return

        self._send_json(404, {"error": "not found"})

    # ── PATCH ─────────────────────────────────────────────────────────────

    def do_PATCH(self) -> None:
        path = self.path.split("?")[0]

        # PATCH /api/v1/sessions/{id}/events/{db_id}  — toggle visibility
        m = _R_EVENT_PATCH.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_patch_event(m.group(1), int(m.group(2)))
            return

        # PATCH /api/v1/sessions/{id}  — update label
        m = _R_SESSION.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_patch_session(m.group(1))
            return

        self._send_json(404, {"error": "not found"})

    # ── DELETE ────────────────────────────────────────────────────────────

    def do_DELETE(self) -> None:
        path = self.path.split("?")[0]
        m = _R_SESSION.match(path)
        if m:
            sid = m.group(1)
            self._session_id_for_log = sid
            # Remove from memory
            with _SESSION_LOCK:
                _SESSIONS.pop(sid, None)
            # Remove from DB
            ok = _db.delete_session(sid) if _db else False
            if ok:
                logger.info("[gateway] Session %s deleted", sid)
                self._send_json(200, {"ok": True})
            else:
                self._send_json(404, {"error": "session not found"})
            return
        self._send_json(404, {"error": "not found"})

    # ─────────────────────────────────────────────────────────────────────
    # Dashboard static file serving
    # ─────────────────────────────────────────────────────────────────────

    def _handle_dashboard(self, url_path: str) -> None:
        if _DASHBOARD_DIR is None:
            self._send_json(501, {"error": "dashboard not available"})
            return

        # Strip /dashboard prefix, default to index.html
        rel = url_path[len("/dashboard"):]
        if not rel or rel == "/":
            rel = "/index.html"

        # Security: no path traversal
        if ".." in rel:
            self._send_json(403, {"error": "forbidden"})
            return

        file_path = _DASHBOARD_DIR / rel.lstrip("/")
        if not file_path.exists() or not file_path.is_file():
            # SPA fallback: serve index.html for any unknown path
            file_path = _DASHBOARD_DIR / "index.html"

        if not file_path.exists():
            self._send_json(404, {"error": "not found"})
            return

        content_type, _ = mimetypes.guess_type(str(file_path))
        content_type = content_type or "application/octet-stream"

        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)
        self._response_code = 200
        self._ts_end = time.time()

    # ─────────────────────────────────────────────────────────────────────
    # SSE streaming
    # ─────────────────────────────────────────────────────────────────────

    def _handle_sse_stream(self, session_id: str) -> None:
        """Server-Sent Events endpoint. Blocks until client disconnects."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self._response_code = 200
        self._ts_end = time.time()

        # Send initial state snapshot
        with _SESSION_LOCK:
            mem = _SESSIONS.get(session_id)
        if mem is not None:
            initial_state = mem.get("state", {})
        elif _db:
            initial_state = _db.get_latest_state(session_id) or {}
        else:
            initial_state = {}

        def _write_sse(event_name: str, data: Any) -> bool:
            """Write a single SSE frame. Returns False on broken pipe."""
            try:
                payload = json.dumps(data, default=str)
                frame = f"event: {event_name}\ndata: {payload}\n\n"
                self.wfile.write(frame.encode("utf-8"))
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError, OSError):
                return False

        if not _write_sse("state", initial_state):
            return

        # Register this connection as a subscriber
        q = _sse_subscribe(session_id)
        logger.debug("[sse] Client connected to session %s", session_id)

        try:
            while True:
                try:
                    # Block up to 15s waiting for events; send ping on timeout
                    msg = q.get(timeout=15.0)
                    try:
                        self.wfile.write(msg.encode("utf-8"))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        break
                except queue.Empty:
                    # Keepalive ping
                    if not _write_sse("ping", {"ts": time.time()}):
                        break
        finally:
            _sse_unsubscribe(session_id, q)
            logger.debug("[sse] Client disconnected from session %s", session_id)

    # ─────────────────────────────────────────────────────────────────────
    # Handler implementations
    # ─────────────────────────────────────────────────────────────────────

    def _handle_list_sessions(self) -> None:
        status_filter = self._qp("status", "all").lower()

        # Merge in-memory + DB sessions
        if _db:
            db_sessions = {s["session_id"]: s for s in _db.list_sessions()}
        else:
            db_sessions = {}

        with _SESSION_LOCK:
            mem_ids = set(_SESSIONS.keys())

        all_ids = set(db_sessions.keys()) | mem_ids
        result = []
        for sid in all_ids:
            mem = _SESSIONS.get(sid, {})
            db_s = db_sessions.get(sid, {})
            state = mem.get("state", {})
            meta = db_s.get("metadata", {})  # already parsed dict from persistence.py

            # Determine is_running / phase — prefer live state, fall back to metadata inference
            is_running = state.get("is_running", False)
            phase = state.get("phase", "unknown")
            # For DB-only sessions that were finished, infer phase from metadata if state is unknown
            if phase == "unknown" and not mem:
                # If we have a DB entry and it's not in memory, it's finished/historical
                phase = "finished"
                is_running = False

            # Apply status filter
            if status_filter != "all":
                if status_filter == "running" and (not is_running or phase in ("finished", "error")):
                    continue
                elif status_filter == "finished" and phase != "finished":
                    continue
                elif status_filter == "error" and phase != "error":
                    continue
                elif status_filter == "paused" and phase != "paused":
                    continue

            # Enriched session summary — sources: live state > DB metadata > DB session row
            result.append({
                "session_id": sid,
                "label": db_s.get("label", state.get("session_label", "")),
                "is_running": is_running,
                "phase": phase,
                "iteration": state.get("iteration", 0),
                "uptime_seconds": state.get("uptime_seconds", 0),
                # Cost & LLM stats from live state or metadata
                "accumulated_cost": state.get("accumulated_cost", 0.0),
                "total_llm_calls": state.get("total_llm_calls", 0),
                # Model & task info from live state or DB metadata (from StartupEvent)
                "llm_model": state.get("llm_model", meta.get("llm_model", "")),
                "task_preview": (
                    state.get("task_instruction", "")[:120] or
                    meta.get("task_preview", "")[:120]
                ),
                # Event counts — prefer DB count (accurate), fall back to mem ring buffer
                "event_count": db_s.get("event_count", len(mem.get("events", []))),
                "error_count": db_s.get("error_count", 0),
                "created_at": db_s.get("created_at", mem.get("created_at", 0)),
                "updated_at": mem.get("updated_at", db_s.get("last_seen_at", 0)),
                "in_memory": sid in mem_ids,
            })
        result.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        self._send_json(200, result)

    # ── State ─────────────────────────────────────────────────────────────

    def _handle_get_state(self, session_id: str) -> None:
        with _SESSION_LOCK:
            mem = _SESSIONS.get(session_id)
        if mem is not None:
            self._send_json(200, mem["state"])
            return
        # Fall back to DB latest snapshot
        if _db:
            state = _db.get_latest_state(session_id)
            if state is not None:
                self._send_json(200, state)
                return
        self._send_json(404, {"error": "session not found"})

    def _handle_post_state(self, session_id: str) -> None:
        body = self._parse_json_body()
        if body is None:
            self._send_json(400, {"error": "invalid JSON"})
            return

        client_ip = self._client_ip()
        now = time.time()

        phase = body.get("phase", "?")
        iteration = body.get("iteration", "?")
        is_running = body.get("is_running", "?")
        new_events = body.get("events", [])
        logger.debug(
            "[gateway] POST state  session=%s  phase=%s  iter=%s  running=%s  embedded_events=%d  from=%s",
            session_id, phase, iteration, is_running, len(new_events), client_ip,
        )

        sess = _get_or_create(session_id)
        with _SESSION_LOCK:
            sess["state"] = body
            sess["updated_at"] = now
            if new_events:
                existing_ids = {e.get("event_id") for e in sess["events"]}
                for ev in new_events:
                    if ev.get("event_id") not in existing_ids:
                        sess["events"].append(ev)
                        existing_ids.add(ev.get("event_id"))
                if len(sess["events"]) > _MAX_MEM_EVENTS:
                    sess["events"] = sess["events"][-_MAX_MEM_EVENTS:]

        if _db:
            label = body.get("session_label", "")
            meta = body.get("extra_metadata", {})
            _db.upsert_session(
                session_id, label=label, metadata=meta, first_state=True
            )
            _db.save_state(session_id, body, client_ip=client_ip)
            if new_events:
                _db.save_events(session_id, new_events, client_ip=client_ip)

        # Broadcast state update to SSE subscribers
        _sse_broadcast(session_id, "state", body)

        self._send_json(200, {"ok": True})

    def _handle_list_snapshots(self, session_id: str) -> None:
        if not _db:
            self._send_json(501, {"error": "persistence not enabled"})
            return
        limit = self._qp_int("limit", 50)
        since = self._qp_float("since")
        snaps = _db.list_state_snapshots(session_id, limit=limit, since=since)
        self._send_json(200, snaps)

    def _handle_get_snapshot(self, session_id: str, snap_id: int) -> None:
        if not _db:
            self._send_json(501, {"error": "persistence not enabled"})
            return
        snap = _db.get_state_snapshot_by_id(snap_id)
        if snap is None or snap.get("session_id") != session_id:
            self._send_json(404, {"error": "snapshot not found"})
            return
        self._send_json(200, snap)

    # ── KPI ───────────────────────────────────────────────────────────────

    def _handle_get_kpi(self, session_id: str) -> None:
        """Return aggregated KPI panel data, merging DB analytics with live state."""
        # DB-derived KPIs
        if _db:
            kpi = _db.get_session_kpi(session_id)
        else:
            kpi = {"session_id": session_id}

        # Merge live-state fields (cost, iteration, phase, is_running)
        with _SESSION_LOCK:
            mem = _SESSIONS.get(session_id)
        if mem:
            state = mem.get("state", {})
            kpi.update({
                "phase": state.get("phase", kpi.get("phase", "unknown")),
                "is_running": state.get("is_running", False),
                "iteration": state.get("iteration", 0),
                "accumulated_cost": state.get("accumulated_cost", 0.0),
                "total_llm_calls": state.get("total_llm_calls", 0),
                "uptime_seconds": state.get("uptime_seconds", 0),
                "llm_model": state.get("llm_model", ""),
                "task_preview": state.get("task_instruction", "")[:120],
            })

        self._send_json(200, kpi)

    # ── Events ────────────────────────────────────────────────────────────

    def _handle_list_events(self, session_id: str) -> None:
        limit = self._qp_int("limit", 100)
        offset = self._qp_int("offset", 0)
        event_type = self._qp("type") or None
        since = self._qp_float("since")
        visible = self._qp_bool_or_none("visible")  # None=all, True/False=filter

        # Try DB first
        if _db:
            events = _db.list_events(
                session_id,
                limit=limit,
                event_type=event_type,
                since=since,
                offset=offset,
                visible=visible,
            )
            total = _db.count_events(session_id, event_type=event_type)
            self._send_json(200, {"total": total, "items": events})
            return

        # Fall back to in-memory
        with _SESSION_LOCK:
            sess = _SESSIONS.get(session_id)
        if sess is None:
            self._send_json(404, {"error": "session not found"})
            return
        items = sess["events"]
        if event_type:
            items = [e for e in items if e.get("event_type") == event_type]
        items = items[-limit:]
        self._send_json(200, {"total": len(items), "items": items})

    def _handle_post_event(self, session_id: str) -> None:
        body = self._parse_json_body()
        if body is None:
            self._send_json(400, {"error": "invalid JSON"})
            return

        event_type = body.get("event_type", "?")
        summary = body.get("summary", "")
        client_ip = self._client_ip()
        logger.debug(
            "[gateway] POST event  session=%s  type=%s  summary=%s  from=%s",
            session_id, event_type, summary[:120], client_ip,
        )

        sess = _get_or_create(session_id)
        with _SESSION_LOCK:
            sess["events"].append(body)
            sess["updated_at"] = time.time()
            if len(sess["events"]) > _MAX_MEM_EVENTS:
                sess["events"] = sess["events"][-_MAX_MEM_EVENTS:]
            # Mirror live state from lifecycle events for GET /state
            if event_type in ("StartupEvent", "HeartbeatEvent", "EvaluateEvent", "FinishEvent"):
                _merge_lifecycle_into_state(sess, body, event_type)

        if _db:
            # ── Always register the session in the DB ──────────────────────
            label = ""
            metadata: dict = {}
            if event_type == "StartupEvent":
                label = body.get("session_label", "")
                metadata = {
                    "llm_model":      body.get("llm_model", ""),
                    "llm_base_url":   body.get("llm_base_url", ""),
                    "workspace_base": body.get("workspace_base", ""),
                    "max_iterations": body.get("max_iterations", 0),
                    "task_preview":   body.get("task_preview", ""),
                    "extra_metadata": body.get("extra_metadata", {}),
                }
            _db.upsert_session(
                session_id,
                label=label,
                metadata=metadata or None,
                first_state=(event_type == "StartupEvent"),
            )
            # ── Persist event ──────────────────────────────────────────────
            _db.save_events(session_id, [body], client_ip=client_ip)
            # ── Write state snapshot for major lifecycle events ────────────
            if event_type in ("StartupEvent", "EvaluateEvent", "FinishEvent"):
                snapshot = dict(sess.get("state", {}))
                snapshot["_snapshot_trigger"] = event_type
                _db.save_state(session_id, snapshot, client_ip=client_ip)
                logger.debug(
                    "[gateway] state_snapshot written (trigger=%s session=%s)",
                    event_type, session_id,
                )

        # ── Broadcast to SSE subscribers ───────────────────────────────────
        # Include live state summary alongside the event for the dashboard
        with _SESSION_LOCK:
            live_state = _SESSIONS.get(session_id, {}).get("state", {})

        _sse_broadcast(session_id, "event", {
            "event_type": event_type,
            "summary": summary[:200],
            "received_at": time.time(),
            "source": body.get("source", ""),
        })
        _sse_broadcast(session_id, "state", live_state)

        self._send_json(200, {"ok": True})

    def _handle_get_event_payload(self, session_id: str, db_id: int) -> None:
        if not _db:
            self._send_json(501, {"error": "persistence not enabled"})
            return
        payload = _db.get_event_payload(db_id)
        if payload is None:
            self._send_json(404, {"error": "event not found"})
            return
        self._send_json(200, payload)

    def _handle_patch_event(self, session_id: str, db_id: int) -> None:
        """Toggle the visibility of an event."""
        body = self._parse_json_body()
        if body is None or "visible" not in body:
            self._send_json(400, {"error": "body must contain 'visible' field"})
            return
        visible = bool(body["visible"])
        if not _db:
            self._send_json(501, {"error": "persistence not enabled"})
            return
        ok = _db.update_event_visibility(db_id, visible)
        if ok:
            self._send_json(200, {"ok": True, "visible": visible})
        else:
            self._send_json(404, {"error": "event not found"})

    # ── Session PATCH ──────────────────────────────────────────────────────

    def _handle_patch_session(self, session_id: str) -> None:
        """Update session metadata (currently: label)."""
        body = self._parse_json_body()
        if body is None:
            self._send_json(400, {"error": "invalid JSON"})
            return

        if "label" in body:
            new_label = str(body["label"])[:200]
            # Update in-memory state
            with _SESSION_LOCK:
                sess = _SESSIONS.get(session_id)
                if sess:
                    sess["state"]["session_label"] = new_label
            # Update in DB
            if _db:
                _db.update_session_label(session_id, new_label)
            self._send_json(200, {"ok": True, "label": new_label})
        else:
            self._send_json(400, {"error": "no updatable fields provided"})

    # ── Control ───────────────────────────────────────────────────────────

    def _handle_get_control(self, session_id: str) -> None:
        sess = _get_or_create(session_id)
        with _SESSION_LOCK:
            ctrl = dict(sess["control"])
        logger.debug(
            "[gateway] GET control  session=%s  pause=%s  resume=%s  from=%s",
            session_id, ctrl.get("pause"), ctrl.get("resume"), self._client_ip(),
        )
        if _db:
            _db.log_control(
                session_id, direction="outbound", action="poll",
                client_ip=self._client_ip(),
            )
        self._send_json(200, ctrl)

    def _handle_post_control(self, session_id: str) -> None:
        body = self._parse_json_body()
        if body is None:
            self._send_json(400, {"error": "invalid JSON"})
            return

        sess = _get_or_create(session_id)
        client_ip = self._client_ip()

        with _SESSION_LOCK:
            ctrl = sess["control"]
            if "pause" in body:
                ctrl["pause"] = bool(body["pause"])
                if ctrl["pause"]:
                    ctrl["resume"] = False
                logger.info("[gateway] Session %s → pause=%s (from %s)",
                            session_id, ctrl["pause"], client_ip)
                if _db:
                    _db.log_control(session_id, "inbound", "pause",
                                    value=ctrl["pause"], client_ip=client_ip)
            if "resume" in body:
                ctrl["resume"] = bool(body["resume"])
                if ctrl["resume"]:
                    ctrl["pause"] = False
                logger.info("[gateway] Session %s → resume=%s (from %s)",
                            session_id, ctrl["resume"], client_ip)
                if _db:
                    _db.log_control(session_id, "inbound", "resume",
                                    value=ctrl["resume"], client_ip=client_ip)

        self._send_json(200, {"ok": True, "control": dict(sess["control"])})

    def _handle_list_control_log(self, session_id: str) -> None:
        if not _db:
            self._send_json(501, {"error": "persistence not enabled"})
            return
        limit = self._qp_int("limit", 100)
        since = self._qp_float("since")
        rows = _db.list_control_log(session_id, limit=limit, since=since)
        self._send_json(200, rows)

    # ── Requests ──────────────────────────────────────────────────────────

    def _handle_list_requests(self, session_id: str | None) -> None:
        if not _db:
            self._send_json(501, {"error": "persistence not enabled"})
            return
        limit = self._qp_int("limit", 100)
        since = self._qp_float("since")
        method = self._qp("method") or None
        rows = _db.list_requests(
            session_id=session_id, limit=limit, since=since, method=method
        )
        self._send_json(200, rows)


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------

def serve(
    host: str = "0.0.0.0",
    port: int = 18858,
    db_path: str = "observer_data.db",
    log_level: str = "INFO",
) -> None:
    global _db, _DASHBOARD_DIR

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.debug(
        "[gateway] Log level set to %s — DEBUG output is active", log_level.upper()
    )

    # Initialise persistence
    from persistence import GatewayDB
    _db = GatewayDB(db_path)

    # Locate dashboard directory (sibling of this script)
    _script_dir = Path(__file__).parent.resolve()
    _dash = _script_dir / "dashboard"
    if _dash.is_dir():
        _DASHBOARD_DIR = _dash
        logger.info("Dashboard: %s", _dash)
    else:
        logger.warning("Dashboard directory not found at %s — static serving disabled", _dash)

    # ThreadingHTTPServer: each connection runs in its own thread.
    # This is REQUIRED for SSE — the blocking q.get() in _handle_sse_stream
    # would otherwise stall the entire server.
    server = ThreadingHTTPServer((host, port), ObserverHandler)
    server.daemon_threads = True   # threads terminate when main thread exits
    logger.info("Observer gateway listening on http://%s:%d", host, port)
    logger.info("DB: %s", db_path)
    logger.info("─" * 60)
    logger.info("  Dashboard         http://%s:%d/", host, port)
    logger.info("  Sessions          GET  /api/v1/sessions")
    logger.info("  Live state        GET  /api/v1/sessions/{id}/state")
    logger.info("  SSE stream        GET  /api/v1/sessions/{id}/stream")
    logger.info("  KPI panel         GET  /api/v1/sessions/{id}/kpi")
    logger.info("  Events            GET  /api/v1/sessions/{id}/events")
    logger.info("  Event payload     GET  /api/v1/sessions/{id}/events/{db_id}/payload")
    logger.info("  Toggle visibility PATCH /api/v1/sessions/{id}/events/{db_id}")
    logger.info("  Control           GET  /api/v1/sessions/{id}/control")
    logger.info("  Set control       POST /api/v1/sessions/{id}/control")
    logger.info("  Session stats     GET  /api/v1/sessions/{id}/stats")
    logger.info("  Session KPI       GET  /api/v1/sessions/{id}/kpi")
    logger.info("─" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Gateway stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rllm OpenHands Observer Gateway")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=18858, help="Bind port (default: 18858)")
    parser.add_argument(
        "--db", default="observer_data.db",
        help="SQLite database path (default: observer_data.db in CWD)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)"
    )
    args = parser.parse_args()
    serve(host=args.host, port=args.port, db_path=args.db, log_level=args.log_level)
