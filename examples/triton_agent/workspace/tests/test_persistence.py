"""
test_persistence.py — Unit tests for examples/openhands_sdk/persistence.py

Tests cover:
  - Schema initialisation
  - Session upsert / list / get / delete (cascades)
  - State snapshot save / list / get
  - Events save (bulk, dedup) / list (filtered) / count / payload
  - Request log
  - Control log
  - Global / session stats
  - Thread-safety smoke test
"""
from __future__ import annotations

import sys
import os
import threading
import time

import pytest

# ---------------------------------------------------------------------------
# Make the openhands_sdk package importable regardless of CWD.
# ---------------------------------------------------------------------------
_SDK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SDK_DIR not in sys.path:
    sys.path.insert(0, _SDK_DIR)

from persistence import GatewayDB  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """Fresh in-process SQLite database for each test."""
    return GatewayDB(tmp_path / "test.db")


def _sid(n: int = 0) -> str:
    return f"session-{n:04d}"


# ===========================================================================
# Session tests
# ===========================================================================

class TestSession:
    def test_upsert_creates_new_session(self, db):
        db.upsert_session(_sid(1), label="alpha")
        sessions = db.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == _sid(1)
        assert sessions[0]["label"] == "alpha"

    def test_upsert_updates_last_seen_at(self, db):
        db.upsert_session(_sid(1), label="alpha")
        t0 = db.get_session(_sid(1))["last_seen_at"]
        time.sleep(0.02)
        db.upsert_session(_sid(1))
        t1 = db.get_session(_sid(1))["last_seen_at"]
        assert t1 >= t0

    def test_upsert_preserves_label_if_new_is_empty(self, db):
        db.upsert_session(_sid(1), label="keeper")
        db.upsert_session(_sid(1), label="")          # empty should not overwrite
        assert db.get_session(_sid(1))["label"] == "keeper"

    def test_upsert_overrides_label_when_provided(self, db):
        db.upsert_session(_sid(1), label="old")
        db.upsert_session(_sid(1), label="new")
        assert db.get_session(_sid(1))["label"] == "new"

    def test_upsert_first_state_flag(self, db):
        db.upsert_session(_sid(1))
        assert db.get_session(_sid(1))["first_state_at"] is None
        db.upsert_session(_sid(1), first_state=True)
        assert db.get_session(_sid(1))["first_state_at"] is not None
        # Second mark should NOT overwrite the first timestamp
        t1 = db.get_session(_sid(1))["first_state_at"]
        time.sleep(0.02)
        db.upsert_session(_sid(1), first_state=True)
        assert db.get_session(_sid(1))["first_state_at"] == t1

    def test_get_session_returns_none_for_unknown(self, db):
        assert db.get_session("nonexistent") is None

    def test_list_sessions_ordered_by_last_seen(self, db):
        db.upsert_session(_sid(1), label="a")
        time.sleep(0.02)
        db.upsert_session(_sid(2), label="b")
        sessions = db.list_sessions()
        assert sessions[0]["session_id"] == _sid(2)

    def test_delete_session_returns_true_when_found(self, db):
        db.upsert_session(_sid(1))
        assert db.delete_session(_sid(1)) is True

    def test_delete_session_returns_false_for_unknown(self, db):
        assert db.delete_session("ghost") is False

    def test_delete_session_cascades(self, db):
        sid = _sid(99)
        db.upsert_session(sid)
        db.save_state(sid, {"x": 1})
        db.save_events(sid, [{"event_id": "e1", "event_type": "ActionEvent"}])
        db.log_request("GET", "/foo", "127.0.0.1", time.time(), 200, 0, 5.0, session_id=sid)
        db.log_control(sid, "inbound", "pause", True)

        db.delete_session(sid)

        assert db.get_session(sid) is None
        assert db.get_latest_state(sid) is None
        assert db.list_events(sid) == []
        assert db.list_requests(session_id=sid) == []
        assert db.list_control_log(sid) == []


# ===========================================================================
# State snapshot tests
# ===========================================================================

class TestStateSnapshot:
    def test_save_and_get_latest(self, db):
        db.upsert_session(_sid(1))
        payload = {"is_running": True, "phase": "executing_command"}
        db.save_state(_sid(1), payload)
        result = db.get_latest_state(_sid(1))
        assert result["is_running"] is True
        assert result["phase"] == "executing_command"

    def test_get_latest_returns_most_recent(self, db):
        db.upsert_session(_sid(1))
        db.save_state(_sid(1), {"v": 1})
        time.sleep(0.01)
        db.save_state(_sid(1), {"v": 2})
        assert db.get_latest_state(_sid(1))["v"] == 2

    def test_get_latest_returns_none_for_unknown_session(self, db):
        assert db.get_latest_state("no-such") is None

    def test_list_state_snapshots_returns_metadata_not_payload(self, db):
        db.upsert_session(_sid(1))
        db.save_state(_sid(1), {"big": "data"})
        snaps = db.list_state_snapshots(_sid(1))
        assert len(snaps) == 1
        assert "id" in snaps[0]
        assert "payload_bytes" in snaps[0]
        assert "payload" not in snaps[0]          # metadata only

    def test_list_state_snapshots_limit(self, db):
        db.upsert_session(_sid(1))
        for i in range(5):
            db.save_state(_sid(1), {"i": i})
        assert len(db.list_state_snapshots(_sid(1), limit=3)) == 3

    def test_list_state_snapshots_since_filter(self, db):
        db.upsert_session(_sid(1))
        db.save_state(_sid(1), {"old": True})
        ts = time.time()
        time.sleep(0.02)
        db.save_state(_sid(1), {"new": True})
        recent = db.list_state_snapshots(_sid(1), since=ts)
        assert len(recent) == 1

    def test_get_state_snapshot_by_id(self, db):
        db.upsert_session(_sid(1))
        db.save_state(_sid(1), {"key": "val"})
        snaps = db.list_state_snapshots(_sid(1))
        snap = db.get_state_snapshot_by_id(snaps[0]["id"])
        assert snap is not None
        assert snap["payload"]["key"] == "val"

    def test_get_state_snapshot_by_id_returns_none_for_unknown(self, db):
        assert db.get_state_snapshot_by_id(999999) is None


# ===========================================================================
# Events tests
# ===========================================================================

def _make_event(event_id: str, event_type: str = "ActionEvent", source: str = "agent") -> dict:
    return {
        "event_id": event_id,
        "event_type": event_type,
        "source": source,
        "timestamp": "2024-01-01T00:00:00",
        "summary": f"[{event_type}] test event {event_id}",
    }


class TestEvents:
    def test_save_events_returns_insert_count(self, db):
        db.upsert_session(_sid(1))
        n = db.save_events(_sid(1), [_make_event("e1"), _make_event("e2")])
        assert n == 2

    def test_save_events_dedup_by_event_id(self, db):
        db.upsert_session(_sid(1))
        db.save_events(_sid(1), [_make_event("e1")])
        n = db.save_events(_sid(1), [_make_event("e1"), _make_event("e2")])
        assert n == 1   # only e2 is new

    def test_save_events_empty_event_id_not_deduped(self, db):
        """Events with empty event_id should all be stored (no dedup)."""
        db.upsert_session(_sid(1))
        ev = {"event_id": "", "event_type": "MessageEvent", "summary": "anon"}
        n1 = db.save_events(_sid(1), [ev])
        n2 = db.save_events(_sid(1), [ev])
        assert n1 == 1
        assert n2 == 1   # no dedup on empty id

    def test_save_events_empty_list_returns_zero(self, db):
        assert db.save_events(_sid(1), []) == 0

    def test_list_events_basic(self, db):
        db.upsert_session(_sid(1))
        db.save_events(_sid(1), [_make_event("e1"), _make_event("e2", "ObservationEvent")])
        items = db.list_events(_sid(1))
        assert len(items) == 2

    def test_list_events_filter_by_type(self, db):
        db.upsert_session(_sid(1))
        db.save_events(_sid(1), [
            _make_event("e1", "ActionEvent"),
            _make_event("e2", "ObservationEvent"),
            _make_event("e3", "ActionEvent"),
        ])
        items = db.list_events(_sid(1), event_type="ActionEvent")
        assert len(items) == 2
        assert all(i["event_type"] == "ActionEvent" for i in items)

    def test_list_events_limit_and_offset(self, db):
        db.upsert_session(_sid(1))
        for i in range(5):
            db.save_events(_sid(1), [_make_event(f"e{i}")])
        page1 = db.list_events(_sid(1), limit=3, offset=0)
        page2 = db.list_events(_sid(1), limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 2
        ids1 = [r["event_id"] for r in page1]
        ids2 = [r["event_id"] for r in page2]
        assert not set(ids1) & set(ids2)

    def test_list_events_since_filter(self, db):
        db.upsert_session(_sid(1))
        db.save_events(_sid(1), [_make_event("old")])
        ts = time.time()
        time.sleep(0.02)
        db.save_events(_sid(1), [_make_event("new")])
        items = db.list_events(_sid(1), since=ts)
        assert len(items) == 1
        assert items[0]["event_id"] == "new"

    def test_count_events(self, db):
        db.upsert_session(_sid(1))
        db.save_events(_sid(1), [_make_event("e1"), _make_event("e2", "ObservationEvent")])
        assert db.count_events(_sid(1)) == 2
        assert db.count_events(_sid(1), event_type="ObservationEvent") == 1
        assert db.count_events(_sid(1), event_type="MessageEvent") == 0

    def test_get_event_payload(self, db):
        db.upsert_session(_sid(1))
        db.save_events(_sid(1), [_make_event("e42", "MessageEvent")])
        items = db.list_events(_sid(1))
        payload = db.get_event_payload(items[0]["id"])
        assert payload is not None
        assert payload["event_id"] == "e42"

    def test_get_event_payload_returns_none_for_unknown(self, db):
        assert db.get_event_payload(99999) is None

    def test_summary_truncated_to_2000(self, db):
        db.upsert_session(_sid(1))
        long_summary = "x" * 5000
        ev = {**_make_event("e-long"), "summary": long_summary}
        db.save_events(_sid(1), [ev])
        # The stored row should have summary ≤ 2000 chars
        items = db.list_events(_sid(1))
        assert len(items[0]["summary"]) <= 2000


# ===========================================================================
# Request log tests
# ===========================================================================

class TestRequestLog:
    def _log(self, db, method="GET", path="/foo", sid=None):
        db.log_request(
            method=method,
            path=path,
            client_ip="127.0.0.1",
            received_at=time.time(),
            status_code=200,
            body_bytes=42,
            response_ms=5.0,
            session_id=sid,
        )

    def test_log_and_list(self, db):
        self._log(db)
        rows = db.list_requests()
        assert len(rows) == 1
        assert rows[0]["method"] == "GET"

    def test_list_filter_by_session(self, db):
        self._log(db, sid="s1")
        self._log(db, sid="s2")
        rows = db.list_requests(session_id="s1")
        assert len(rows) == 1
        assert rows[0]["session_id"] == "s1"

    def test_list_filter_by_method(self, db):
        self._log(db, method="GET")
        self._log(db, method="POST")
        rows = db.list_requests(method="POST")
        assert len(rows) == 1

    def test_list_filter_since(self, db):
        self._log(db)
        ts = time.time()
        time.sleep(0.02)
        self._log(db)
        rows = db.list_requests(since=ts)
        assert len(rows) == 1

    def test_list_limit(self, db):
        for _ in range(5):
            self._log(db)
        assert len(db.list_requests(limit=3)) == 3


# ===========================================================================
# Control log tests
# ===========================================================================

class TestControlLog:
    def test_log_and_list(self, db):
        db.upsert_session(_sid(1))
        db.log_control(_sid(1), "inbound", "pause", True, "1.2.3.4")
        rows = db.list_control_log(_sid(1))
        assert len(rows) == 1
        assert rows[0]["direction"] == "inbound"
        assert rows[0]["action"] == "pause"
        assert rows[0]["value"] == 1

    def test_log_resume(self, db):
        db.upsert_session(_sid(1))
        db.log_control(_sid(1), "inbound", "resume", True)
        rows = db.list_control_log(_sid(1))
        assert rows[0]["action"] == "resume"

    def test_log_poll_with_none_value(self, db):
        db.upsert_session(_sid(1))
        db.log_control(_sid(1), "outbound", "poll", None)
        rows = db.list_control_log(_sid(1))
        assert rows[0]["value"] is None

    def test_list_control_log_limit(self, db):
        db.upsert_session(_sid(1))
        for _ in range(5):
            db.log_control(_sid(1), "inbound", "pause", True)
        assert len(db.list_control_log(_sid(1), limit=3)) == 3

    def test_list_control_log_since(self, db):
        db.upsert_session(_sid(1))
        db.log_control(_sid(1), "inbound", "pause", True)
        ts = time.time()
        time.sleep(0.02)
        db.log_control(_sid(1), "inbound", "resume", True)
        rows = db.list_control_log(_sid(1), since=ts)
        assert len(rows) == 1
        assert rows[0]["action"] == "resume"


# ===========================================================================
# Global stats
# ===========================================================================

class TestGlobalStats:
    def test_global_stats_initial(self, db):
        stats = db.get_global_stats()
        assert stats["total_sessions"] == 0
        assert stats["total_events"] == 0
        assert "db_path" in stats

    def test_global_stats_after_activity(self, db):
        db.upsert_session(_sid(1))
        db.save_state(_sid(1), {"x": 1})
        db.save_events(_sid(1), [_make_event("e1")])
        stats = db.get_global_stats()
        assert stats["total_sessions"] == 1
        assert stats["total_state_snapshots"] == 1
        assert stats["total_events"] == 1

    def test_session_stats(self, db):
        sid = _sid(1)
        db.upsert_session(sid)
        db.save_state(sid, {"v": 1})
        db.save_events(sid, [_make_event("e1", "ActionEvent"), _make_event("e2", "ObservationEvent")])
        db.log_control(sid, "inbound", "pause", True)
        stats = db.get_session_stats(sid)
        assert stats["state_snapshot_count"] == 1
        assert stats["event_count"] == 2
        assert stats["control_action_count"] == 1
        assert stats["event_type_breakdown"]["ActionEvent"] == 1
        assert stats["event_type_breakdown"]["ObservationEvent"] == 1


# ===========================================================================
# Thread safety smoke test
# ===========================================================================

class TestThreadSafety:
    def test_concurrent_writes(self, db):
        """Multiple threads writing events should not cause corruption."""
        n_threads = 8
        events_per_thread = 20
        errors: list[Exception] = []

        def worker(tid: int):
            try:
                db.upsert_session(_sid(tid))
                for i in range(events_per_thread):
                    db.save_events(_sid(tid), [_make_event(f"t{tid}-e{i}")])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Concurrent write errors: {errors}"
        total = db.get_global_stats()["total_events"]
        assert total == n_threads * events_per_thread
