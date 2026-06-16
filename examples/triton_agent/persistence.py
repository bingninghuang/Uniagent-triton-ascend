"""
persistence.py — SQLite persistence layer for the observer gateway.

Schema (all tables keyed by session_id for per-instance isolation):

    sessions
        session_id      TEXT PRIMARY KEY
        label           TEXT
        created_at      REAL   (unix timestamp)
        last_seen_at    REAL
        first_state_at  REAL
        metadata        TEXT   (JSON)

    state_snapshots
        id              INTEGER PRIMARY KEY AUTOINCREMENT
        session_id      TEXT
        received_at     REAL   (unix timestamp, server-side)
        client_ip       TEXT
        payload         TEXT   (JSON — full RunState.to_dict())

    events
        id              INTEGER PRIMARY KEY AUTOINCREMENT
        session_id      TEXT
        event_id        TEXT   (OpenHands event UUID — unique per session)
        event_type      TEXT
        source          TEXT
        timestamp_str   TEXT   (event-side ISO timestamp)
        received_at     REAL   (server-side unix timestamp)
        summary         TEXT
        payload         TEXT   (JSON)
        visible         INTEGER  (1=visible, 0=hidden; default 1)

    request_log
        id              INTEGER PRIMARY KEY AUTOINCREMENT
        session_id      TEXT   (NULL for session-unrelated requests)
        method          TEXT
        path            TEXT
        client_ip       TEXT
        received_at     REAL
        status_code     INTEGER
        body_bytes      INTEGER
        response_ms     REAL

    control_log
        id              INTEGER PRIMARY KEY AUTOINCREMENT
        session_id      TEXT
        received_at     REAL
        direction       TEXT   ('inbound' | 'outbound')
        action          TEXT   ('pause' | 'resume' | 'poll')
        value           INTEGER  (1=true, 0=false, NULL=poll)
        client_ip       TEXT

Indexes are built on (session_id, received_at) for all time-series tables.

Thread safety: SQLite in WAL mode with per-connection check_same_thread=False.
We use a threading.Lock to serialise writes; reads use separate per-thread
connections via threading.local().
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    label           TEXT NOT NULL DEFAULT '',
    created_at      REAL NOT NULL,
    last_seen_at    REAL NOT NULL,
    first_state_at  REAL,
    metadata        TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS state_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    received_at     REAL NOT NULL,
    client_ip       TEXT NOT NULL DEFAULT '',
    payload         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    event_id        TEXT NOT NULL DEFAULT '',
    event_type      TEXT NOT NULL DEFAULT '',
    source          TEXT NOT NULL DEFAULT '',
    timestamp_str   TEXT NOT NULL DEFAULT '',
    received_at     REAL NOT NULL,
    summary         TEXT NOT NULL DEFAULT '',
    payload         TEXT NOT NULL DEFAULT '{}',
    visible         INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS request_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT,
    method          TEXT NOT NULL,
    path            TEXT NOT NULL,
    client_ip       TEXT NOT NULL DEFAULT '',
    received_at     REAL NOT NULL,
    status_code     INTEGER NOT NULL DEFAULT 0,
    body_bytes      INTEGER NOT NULL DEFAULT 0,
    response_ms     REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS control_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    received_at     REAL NOT NULL,
    direction       TEXT NOT NULL,
    action          TEXT NOT NULL,
    value           INTEGER,
    client_ip       TEXT NOT NULL DEFAULT ''
);

-- Indexes for fast per-session time-series queries
CREATE INDEX IF NOT EXISTS idx_state_session_time
    ON state_snapshots (session_id, received_at);

CREATE INDEX IF NOT EXISTS idx_events_session_time
    ON events (session_id, received_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_events_dedup
    ON events (session_id, event_id)
    WHERE event_id != '';

CREATE INDEX IF NOT EXISTS idx_reqlog_session_time
    ON request_log (session_id, received_at);

CREATE INDEX IF NOT EXISTS idx_ctrl_session_time
    ON control_log (session_id, received_at);
"""


class GatewayDB:
    """SQLite persistence layer for the observer gateway."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = str(db_path)
        self._write_lock = threading.Lock()
        self._local = threading.local()   # per-thread read connections
        # Initialise schema using a dedicated connection
        conn = self._open()
        conn.executescript(_SCHEMA)
        conn.commit()
        # Live migration: add `visible` column if missing on older DBs
        self._migrate(conn)
        conn.close()
        logger.info("[db] Opened SQLite database at %s", self._path)

    # ── Schema migration ──────────────────────────────────────────────────

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Apply schema migrations for backward compatibility."""
        # Check whether 'visible' column exists in events
        cols = {row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()}
        if "visible" not in cols:
            conn.execute("ALTER TABLE events ADD COLUMN visible INTEGER NOT NULL DEFAULT 1")
            conn.commit()
            logger.info("[db] Migration: added 'visible' column to events table")

    # ── Connection management ─────────────────────────────────────────────

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._path,
            check_same_thread=False,
            isolation_level=None,   # autocommit; we manage transactions manually
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _read_conn(self) -> sqlite3.Connection:
        """Return a per-thread read connection, creating if needed."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            self._local.conn = self._open()
        return self._local.conn

    def close(self):
        pass

    # ── Session ───────────────────────────────────────────────────────────

    def upsert_session(
        self,
        session_id: str,
        label: str = "",
        metadata: dict | None = None,
        first_state: bool = False,
    ) -> None:
        now = time.time()
        meta_json = json.dumps(metadata or {}, default=str)
        with self._write_lock:
            conn = self._open()
            try:
                conn.execute("BEGIN")
                conn.execute(
                    """
                    INSERT INTO sessions (session_id, label, created_at, last_seen_at, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        last_seen_at = excluded.last_seen_at,
                        label = CASE WHEN excluded.label != '' THEN excluded.label ELSE label END,
                        metadata = CASE WHEN excluded.metadata != '{}' THEN excluded.metadata ELSE metadata END
                    """,
                    (session_id, label, now, now, meta_json),
                )
                if first_state:
                    conn.execute(
                        """
                        UPDATE sessions SET first_state_at = ?
                        WHERE session_id = ? AND first_state_at IS NULL
                        """,
                        (now, session_id),
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    def update_session_label(self, session_id: str, label: str) -> bool:
        """Update the human-readable label of a session. Returns True if found."""
        with self._write_lock:
            conn = self._open()
            try:
                cur = conn.execute(
                    "UPDATE sessions SET label = ? WHERE session_id = ?",
                    (label, session_id),
                )
                conn.commit()
                return cur.rowcount > 0
            except Exception:
                return False
            finally:
                conn.close()

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions enriched with event counts and error counts."""
        conn = self._read_conn()
        rows = conn.execute(
            """
            SELECT
                s.session_id,
                s.label,
                s.created_at,
                s.last_seen_at,
                s.first_state_at,
                s.metadata,
                COUNT(e.id) AS event_count,
                SUM(CASE WHEN e.event_type = 'AgentErrorEvent' THEN 1 ELSE 0 END) AS error_count
            FROM sessions s
            LEFT JOIN events e ON s.session_id = e.session_id
            GROUP BY s.session_id
            ORDER BY s.last_seen_at DESC
            """
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["metadata"] = json.loads(d.get("metadata") or "{}")
            except Exception:
                d["metadata"] = {}
            result.append(d)
        return result

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        conn = self._read_conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        try:
            d["metadata"] = json.loads(d.get("metadata") or "{}")
        except Exception:
            d["metadata"] = {}
        return d

    def delete_session(self, session_id: str) -> bool:
        """Remove a session and all associated data."""
        with self._write_lock:
            conn = self._open()
            try:
                conn.execute("BEGIN")
                for table in ("state_snapshots", "events", "request_log", "control_log"):
                    conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
                cur = conn.execute(
                    "DELETE FROM sessions WHERE session_id = ?", (session_id,)
                )
                deleted = cur.rowcount > 0
                conn.execute("COMMIT")
                return deleted
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    # ── State snapshots ───────────────────────────────────────────────────

    def save_state(
        self,
        session_id: str,
        payload: dict[str, Any],
        client_ip: str = "",
    ) -> None:
        now = time.time()
        payload_json = json.dumps(payload, default=str)
        with self._write_lock:
            conn = self._open()
            try:
                conn.execute("BEGIN")
                conn.execute(
                    "INSERT INTO state_snapshots (session_id, received_at, client_ip, payload) VALUES (?,?,?,?)",
                    (session_id, now, client_ip, payload_json),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    def get_latest_state(self, session_id: str) -> dict[str, Any] | None:
        conn = self._read_conn()
        row = conn.execute(
            """SELECT payload FROM state_snapshots
               WHERE session_id = ?
               ORDER BY received_at DESC LIMIT 1""",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["payload"])
        except json.JSONDecodeError:
            return {}

    def list_state_snapshots(
        self,
        session_id: str,
        limit: int = 50,
        since: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return metadata (not full payload) of recent snapshots."""
        conn = self._read_conn()
        if since:
            rows = conn.execute(
                """SELECT id, received_at, client_ip, length(payload) AS payload_bytes
                   FROM state_snapshots
                   WHERE session_id = ? AND received_at >= ?
                   ORDER BY received_at DESC LIMIT ?""",
                (session_id, since, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, received_at, client_ip, length(payload) AS payload_bytes
                   FROM state_snapshots
                   WHERE session_id = ?
                   ORDER BY received_at DESC LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_state_snapshot_by_id(self, snapshot_id: int) -> dict[str, Any] | None:
        conn = self._read_conn()
        row = conn.execute(
            "SELECT * FROM state_snapshots WHERE id = ?", (snapshot_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        try:
            d["payload"] = json.loads(d["payload"])
        except Exception:
            pass
        return d

    # ── Events ────────────────────────────────────────────────────────────

    def save_events(
        self,
        session_id: str,
        events: list[dict[str, Any]],
        client_ip: str = "",
    ) -> int:
        """Bulk-insert events (skip duplicates by event_id).  Returns insert count."""
        if not events:
            return 0
        now = time.time()
        inserted = 0
        with self._write_lock:
            conn = self._open()
            try:
                conn.execute("BEGIN")
                for ev in events:
                    event_id = str(ev.get("event_id", ""))
                    event_type = str(ev.get("event_type", ""))
                    source = str(ev.get("source", ""))
                    timestamp_str = str(ev.get("timestamp", ""))
                    summary = str(ev.get("summary", ""))[:2000]
                    payload_json = json.dumps(ev, default=str)
                    cur = conn.execute(
                        """INSERT OR IGNORE INTO events
                           (session_id, event_id, event_type, source,
                            timestamp_str, received_at, summary, payload, visible)
                           VALUES (?,?,?,?,?,?,?,?,1)""",
                        (session_id, event_id, event_type, source,
                         timestamp_str, now, summary, payload_json),
                    )
                    inserted += cur.rowcount  # 1 on insert, 0 on IGNORE
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()
        return inserted

    def update_event_visibility(self, event_db_id: int, visible: bool) -> bool:
        """Toggle the visibility flag of an event. Returns True if found."""
        val = 1 if visible else 0
        with self._write_lock:
            conn = self._open()
            try:
                cur = conn.execute(
                    "UPDATE events SET visible = ? WHERE id = ?",
                    (val, event_db_id),
                )
                conn.commit()
                return cur.rowcount > 0
            except Exception:
                return False
            finally:
                conn.close()

    def list_events(
        self,
        session_id: str,
        limit: int = 100,
        event_type: str | None = None,
        since: float | None = None,
        offset: int = 0,
        visible: bool | None = None,  # None=all, True=visible only, False=hidden only
    ) -> list[dict[str, Any]]:
        conn = self._read_conn()
        conditions = ["session_id = ?"]
        params: list[Any] = [session_id]
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if since:
            conditions.append("received_at >= ?")
            params.append(since)
        if visible is True:
            conditions.append("visible = 1")
        elif visible is False:
            conditions.append("visible = 0")
        where = " AND ".join(conditions)
        params += [limit, offset]
        rows = conn.execute(
            f"""SELECT id, event_id, event_type, source, timestamp_str,
                       received_at, summary, visible
                FROM events WHERE {where}
                ORDER BY received_at ASC, id ASC
                LIMIT ? OFFSET ?""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_event_payload(self, event_db_id: int) -> dict[str, Any] | None:
        conn = self._read_conn()
        row = conn.execute(
            "SELECT payload FROM events WHERE id = ?", (event_db_id,)
        ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["payload"])
        except Exception:
            return {}

    def count_events(self, session_id: str, event_type: str | None = None) -> int:
        conn = self._read_conn()
        if event_type:
            row = conn.execute(
                "SELECT COUNT(*) FROM events WHERE session_id=? AND event_type=?",
                (session_id, event_type),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM events WHERE session_id=?",
                (session_id,),
            ).fetchone()
        return row[0] if row else 0

    # ── Request log ───────────────────────────────────────────────────────

    def log_request(
        self,
        method: str,
        path: str,
        client_ip: str,
        received_at: float,
        status_code: int,
        body_bytes: int,
        response_ms: float,
        session_id: str | None = None,
    ) -> None:
        with self._write_lock:
            conn = self._open()
            try:
                conn.execute(
                    """INSERT INTO request_log
                       (session_id, method, path, client_ip, received_at,
                        status_code, body_bytes, response_ms)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (session_id, method, path, client_ip, received_at,
                     status_code, body_bytes, response_ms),
                )
                conn.commit()
            except Exception:
                pass
            finally:
                conn.close()

    def list_requests(
        self,
        session_id: str | None = None,
        limit: int = 100,
        since: float | None = None,
        method: str | None = None,
    ) -> list[dict[str, Any]]:
        conn = self._read_conn()
        conditions: list[str] = []
        params: list[Any] = []
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if since:
            conditions.append("received_at >= ?")
            params.append(since)
        if method:
            conditions.append("method = ?")
            params.append(method)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        rows = conn.execute(
            f"""SELECT id, session_id, method, path, client_ip,
                       received_at, status_code, body_bytes, response_ms
                FROM request_log {where}
                ORDER BY received_at DESC LIMIT ?""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Control log ───────────────────────────────────────────────────────

    def log_control(
        self,
        session_id: str,
        direction: str,   # 'inbound' | 'outbound'
        action: str,      # 'pause' | 'resume' | 'poll'
        value: bool | None = None,
        client_ip: str = "",
    ) -> None:
        now = time.time()
        val_int = None if value is None else (1 if value else 0)
        with self._write_lock:
            conn = self._open()
            try:
                conn.execute(
                    """INSERT INTO control_log
                       (session_id, received_at, direction, action, value, client_ip)
                       VALUES (?,?,?,?,?,?)""",
                    (session_id, now, direction, action, val_int, client_ip),
                )
                conn.commit()
            except Exception:
                pass
            finally:
                conn.close()

    def list_control_log(
        self,
        session_id: str,
        limit: int = 100,
        since: float | None = None,
    ) -> list[dict[str, Any]]:
        conn = self._read_conn()
        if since:
            rows = conn.execute(
                """SELECT * FROM control_log
                   WHERE session_id = ? AND received_at >= ?
                   ORDER BY received_at DESC LIMIT ?""",
                (session_id, since, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM control_log
                   WHERE session_id = ?
                   ORDER BY received_at DESC LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── KPI / Stats ────────────────────────────────────────────────────────

    def get_global_stats(self) -> dict[str, Any]:
        conn = self._read_conn()
        return {
            "total_sessions": conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
            "total_state_snapshots": conn.execute("SELECT COUNT(*) FROM state_snapshots").fetchone()[0],
            "total_events": conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            "total_requests": conn.execute("SELECT COUNT(*) FROM request_log").fetchone()[0],
            "total_control_actions": conn.execute("SELECT COUNT(*) FROM control_log").fetchone()[0],
            "db_path": self._path,
        }

    def get_session_stats(self, session_id: str) -> dict[str, Any]:
        conn = self._read_conn()
        return {
            "session_id": session_id,
            "state_snapshot_count": conn.execute(
                "SELECT COUNT(*) FROM state_snapshots WHERE session_id=?", (session_id,)
            ).fetchone()[0],
            "event_count": conn.execute(
                "SELECT COUNT(*) FROM events WHERE session_id=?", (session_id,)
            ).fetchone()[0],
            "request_count": conn.execute(
                "SELECT COUNT(*) FROM request_log WHERE session_id=?", (session_id,)
            ).fetchone()[0],
            "control_action_count": conn.execute(
                "SELECT COUNT(*) FROM control_log WHERE session_id=?", (session_id,)
            ).fetchone()[0],
            "event_type_breakdown": {
                row["event_type"]: row["cnt"]
                for row in conn.execute(
                    "SELECT event_type, COUNT(*) AS cnt FROM events WHERE session_id=? GROUP BY event_type",
                    (session_id,),
                ).fetchall()
            },
        }

    def get_session_kpi(self, session_id: str) -> dict[str, Any]:
        """Aggregated KPI fields for the dashboard panel.

        Returns timing metrics, event rates, and phase transitions derived
        from the events table.  Live fields (cost, iteration, phase) are
        intentionally omitted — the caller (gateway) merges them from the
        in-memory state.
        """
        conn = self._read_conn()

        # --- basic counts ---
        counts_row = conn.execute(
            """SELECT
                COUNT(*) AS total_events,
                SUM(CASE WHEN event_type='ActionEvent' THEN 1 ELSE 0 END) AS action_count,
                SUM(CASE WHEN event_type='ObservationEvent' THEN 1 ELSE 0 END) AS observation_count,
                SUM(CASE WHEN event_type='LLMCompletionLogEvent' THEN 1 ELSE 0 END) AS llm_completion_count,
                SUM(CASE WHEN event_type='AgentErrorEvent' THEN 1 ELSE 0 END) AS error_count,
                SUM(CASE WHEN event_type='HeartbeatEvent' THEN 1 ELSE 0 END) AS heartbeat_count,
                MIN(received_at) AS first_event_at,
                MAX(received_at) AS last_event_at
            FROM events WHERE session_id=?""",
            (session_id,),
        ).fetchone()

        total = counts_row["total_events"] or 0
        first_at = counts_row["first_event_at"]
        last_at = counts_row["last_event_at"]
        duration = (last_at - first_at) if (first_at and last_at and last_at > first_at) else 0
        events_per_minute = round((total / duration * 60), 2) if duration > 0 else 0.0

        # --- event_type breakdown ---
        breakdown = {
            row["event_type"]: row["cnt"]
            for row in conn.execute(
                "SELECT event_type, COUNT(*) AS cnt FROM events WHERE session_id=? GROUP BY event_type",
                (session_id,),
            ).fetchall()
        }

        # --- phase transitions ---
        # Extract from HeartbeatEvent and lifecycle event payloads.
        # We select event_type + received_at + payload for phase-carrying events,
        # then parse phase field from JSON payload, deduplicating consecutive identical phases.
        phase_rows = conn.execute(
            """SELECT event_type, received_at, payload
               FROM events
               WHERE session_id=?
                 AND event_type IN (
                   'StartupEvent','HeartbeatEvent','ActionEvent','ObservationEvent',
                   'MessageEvent','PauseEvent','EvaluateEvent','FinishEvent'
                 )
               ORDER BY received_at ASC""",
            (session_id,),
        ).fetchall()

        # Map event_type → inferred phase (fallback if payload lacks explicit phase)
        _phase_map = {
            "StartupEvent": "initializing",
            "ActionEvent": "executing_command",
            "ObservationEvent": "waiting_for_reply",
            "MessageEvent": "waiting_for_reply",
            "PauseEvent": "paused",
            "EvaluateEvent": "finished",
            "FinishEvent": "finished",
        }

        phase_transitions: list[dict[str, Any]] = []
        last_phase = None
        for pr in phase_rows:
            try:
                pl = json.loads(pr["payload"])
                phase = pl.get("phase") or _phase_map.get(pr["event_type"], "")
            except Exception:
                phase = _phase_map.get(pr["event_type"], "")
            if phase and phase != last_phase:
                phase_transitions.append({"phase": phase, "at": pr["received_at"]})
                last_phase = phase

        return {
            "session_id": session_id,
            "total_events": total,
            "action_count": counts_row["action_count"] or 0,
            "observation_count": counts_row["observation_count"] or 0,
            "llm_completion_count": counts_row["llm_completion_count"] or 0,
            "error_count": counts_row["error_count"] or 0,
            "heartbeat_count": counts_row["heartbeat_count"] or 0,
            "event_type_breakdown": breakdown,
            "first_event_at": first_at,
            "last_event_at": last_at,
            "duration_seconds": round(duration, 2),
            "events_per_minute": events_per_minute,
            "phase_transitions": phase_transitions,
        }
