"""
test_gateway_e2e.py — End-to-end tests for observer_gateway.py + persistence.py.

These tests:
  1. Spin up the real ObserverGateway HTTP server (in a background thread)
     with a temporary SQLite database.
  2. Use urllib.request (same as the container client) to hit each endpoint.
  3. Assert on HTTP status codes AND response JSON structure.
  4. Verify persistence: data posted by the container survives and is
     queryable through the read endpoints.

No mock patching is used — this is a full stack test.

Test groups:
  - Session lifecycle (list, state upload, delete)
  - State snapshots (post → get latest, list snapshots, get by id)
  - Events (post → list, filter, count, payload)
  - Control (get, post pause, post resume, log)
  - Request log (auto-populated by each HTTP call)
  - Stats (session & global)
  - Error handling (404, 400, DELETE non-existent)
"""
from __future__ import annotations

import json
import socket
import sys
import os
import time
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Path setup: openhands_sdk dir is the root
# ---------------------------------------------------------------------------
_SDK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SDK_DIR not in sys.path:
    sys.path.insert(0, _SDK_DIR)

import observer_gateway  # noqa: E402
from persistence import GatewayDB  # noqa: E402


# ===========================================================================
# Fixture: running gateway server
# ===========================================================================

def _free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def gateway(tmp_path_factory):
    """Start the gateway server once for the whole module."""
    db_path = tmp_path_factory.mktemp("gw") / "test_gw.db"
    port = _free_port()

    # Wire up the module-level _db used by ObserverHandler
    db = GatewayDB(db_path)
    observer_gateway._db = db
    # Clear in-memory session store
    with observer_gateway._SESSION_LOCK:
        observer_gateway._SESSIONS.clear()

    server = HTTPServer(("127.0.0.1", port), observer_gateway.ObserverHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    yield base_url, db

    server.shutdown()
    thread.join(timeout=5)


# ===========================================================================
# HTTP helpers
# ===========================================================================

def _get(url: str, expected: int = 200) -> Any:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == expected, f"GET {url}: expected {expected}, got {resp.status}"
        return json.loads(resp.read())


def _post(url: str, body: dict, expected: int = 200) -> Any:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", "Content-Length": str(len(data))},
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == expected
        return json.loads(resp.read())


def _delete(url: str) -> tuple[int, Any]:
    req = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _make_state_payload(sid: str, phase: str = "executing_command", iteration: int = 1) -> dict:
    return {
        "session_id": sid,
        "session_label": f"label-{sid[:8]}",
        "is_running": True,
        "phase": phase,
        "iteration": iteration,
        "total_llm_calls": iteration,
        "accumulated_cost": iteration * 0.01,
        "events": [],
        "command_history": [],
        "llm_context_snapshot": [],
        "extra_metadata": {"env": "test"},
    }


def _make_event_payload(event_id: str, event_type: str = "ActionEvent") -> dict:
    return {
        "event_id": event_id,
        "event_type": event_type,
        "source": "agent",
        "timestamp": "2024-01-01T12:00:00",
        "summary": f"[{event_type}] {event_id}",
    }


# ===========================================================================
# Session lifecycle
# ===========================================================================

class TestSessionLifecycle:
    def test_list_sessions_empty_initially(self, gateway):
        base, db = gateway
        # We cannot guarantee isolation across tests in module scope, so just check type
        resp = _get(f"{base}/api/v1/sessions")
        assert isinstance(resp, list)

    def test_post_state_creates_session(self, gateway):
        base, db = gateway
        sid = "e2e-lifecycle-01"
        _post(f"{base}/api/v1/sessions/{sid}/state", _make_state_payload(sid))
        sessions_resp = _get(f"{base}/api/v1/sessions")
        sids = [s["session_id"] for s in sessions_resp]
        assert sid in sids

    def test_delete_session_returns_ok(self, gateway):
        base, db = gateway
        sid = "e2e-del-01"
        _post(f"{base}/api/v1/sessions/{sid}/state", _make_state_payload(sid))
        status, body = _delete(f"{base}/api/v1/sessions/{sid}")
        assert status == 200
        assert body["ok"] is True

    def test_delete_unknown_session_returns_404(self, gateway):
        base, db = gateway
        status, body = _delete(f"{base}/api/v1/sessions/ghost-9999")
        assert status == 404

    def test_delete_removes_from_session_list(self, gateway):
        base, db = gateway
        sid = "e2e-del-02"
        _post(f"{base}/api/v1/sessions/{sid}/state", _make_state_payload(sid))
        _delete(f"{base}/api/v1/sessions/{sid}")
        resp = _get(f"{base}/api/v1/sessions")
        sids = [s["session_id"] for s in resp]
        assert sid not in sids


# ===========================================================================
# State
# ===========================================================================

class TestState:
    def test_get_state_returns_posted_state(self, gateway):
        base, db = gateway
        sid = "e2e-state-01"
        payload = _make_state_payload(sid, phase="paused", iteration=3)
        _post(f"{base}/api/v1/sessions/{sid}/state", payload)
        resp = _get(f"{base}/api/v1/sessions/{sid}/state")
        assert resp["phase"] == "paused"
        assert resp["iteration"] == 3

    def test_get_state_404_for_unknown(self, gateway):
        base, db = gateway
        req = urllib.request.Request(
            f"{base}/api/v1/sessions/ghost-state/state", method="GET"
        )
        try:
            with urllib.request.urlopen(req):
                pass
            pytest.fail("Expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404

    def test_post_state_updates_latest(self, gateway):
        base, db = gateway
        sid = "e2e-state-02"
        _post(f"{base}/api/v1/sessions/{sid}/state", _make_state_payload(sid, iteration=1))
        _post(f"{base}/api/v1/sessions/{sid}/state", _make_state_payload(sid, iteration=5))
        resp = _get(f"{base}/api/v1/sessions/{sid}/state")
        assert resp["iteration"] == 5

    def test_post_state_invalid_json_returns_400(self, gateway):
        base, db = gateway
        sid = "e2e-state-03"
        raw = b"not json at all"
        req = urllib.request.Request(
            f"{base}/api/v1/sessions/{sid}/state",
            data=raw,
            method="POST",
            headers={"Content-Type": "application/json", "Content-Length": str(len(raw))},
        )
        try:
            with urllib.request.urlopen(req):
                pass
            pytest.fail("Expected 400")
        except urllib.error.HTTPError as e:
            assert e.code == 400


# ===========================================================================
# Snapshots
# ===========================================================================

class TestSnapshots:
    def test_list_snapshots(self, gateway):
        base, db = gateway
        sid = "e2e-snaps-01"
        for i in range(3):
            _post(f"{base}/api/v1/sessions/{sid}/state", _make_state_payload(sid, iteration=i))
        snaps = _get(f"{base}/api/v1/sessions/{sid}/snapshots")
        assert len(snaps) == 3
        assert "id" in snaps[0]
        assert "payload_bytes" in snaps[0]

    def test_get_snapshot_by_id(self, gateway):
        base, db = gateway
        sid = "e2e-snaps-02"
        _post(f"{base}/api/v1/sessions/{sid}/state", _make_state_payload(sid, iteration=7))
        snaps = _get(f"{base}/api/v1/sessions/{sid}/snapshots")
        snap_id = snaps[0]["id"]
        snap = _get(f"{base}/api/v1/sessions/{sid}/snapshots/{snap_id}")
        assert snap["payload"]["iteration"] == 7

    def test_get_snapshot_404_for_wrong_session(self, gateway):
        """Snapshot belonging to another session should 404."""
        base, db = gateway
        sid_a = "e2e-snaps-a"
        sid_b = "e2e-snaps-b"
        _post(f"{base}/api/v1/sessions/{sid_a}/state", _make_state_payload(sid_a))
        snaps = _get(f"{base}/api/v1/sessions/{sid_a}/snapshots")
        snap_id = snaps[0]["id"]
        req = urllib.request.Request(
            f"{base}/api/v1/sessions/{sid_b}/snapshots/{snap_id}", method="GET"
        )
        try:
            with urllib.request.urlopen(req):
                pass
            pytest.fail("Expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404


# ===========================================================================
# Events
# ===========================================================================

class TestEvents:
    def test_post_and_list_event(self, gateway):
        base, db = gateway
        sid = "e2e-events-01"
        ev = _make_event_payload("ev-001", "ActionEvent")
        _post(f"{base}/api/v1/sessions/{sid}/events", ev)
        resp = _get(f"{base}/api/v1/sessions/{sid}/events")
        assert resp["total"] >= 1
        event_ids = [e["event_id"] for e in resp["items"]]
        assert "ev-001" in event_ids

    def test_filter_events_by_type(self, gateway):
        base, db = gateway
        sid = "e2e-events-02"
        _post(f"{base}/api/v1/sessions/{sid}/events", _make_event_payload("ea", "ActionEvent"))
        _post(f"{base}/api/v1/sessions/{sid}/events", _make_event_payload("eb", "ObservationEvent"))
        resp = _get(f"{base}/api/v1/sessions/{sid}/events?type=ActionEvent")
        assert all(e["event_type"] == "ActionEvent" for e in resp["items"])

    def test_event_dedup_via_post_state(self, gateway):
        """Events embedded in state payload should be deduped."""
        base, db = gateway
        sid = "e2e-events-03"
        ev = _make_event_payload("dedup-001", "MessageEvent")
        payload = {**_make_state_payload(sid), "events": [ev]}
        _post(f"{base}/api/v1/sessions/{sid}/state", payload)
        _post(f"{base}/api/v1/sessions/{sid}/state", payload)  # same event again
        resp = _get(f"{base}/api/v1/sessions/{sid}/events")
        count = sum(1 for e in resp["items"] if e["event_id"] == "dedup-001")
        assert count == 1

    def test_get_event_payload(self, gateway):
        base, db = gateway
        sid = "e2e-events-04"
        _post(f"{base}/api/v1/sessions/{sid}/events", _make_event_payload("pl-001"))
        resp = _get(f"{base}/api/v1/sessions/{sid}/events")
        db_id = resp["items"][0]["id"]
        payload = _get(f"{base}/api/v1/sessions/{sid}/events/{db_id}/payload")
        assert payload["event_id"] == "pl-001"

    def test_invalid_event_json_returns_400(self, gateway):
        base, db = gateway
        sid = "e2e-events-05"
        raw = b"not-json"
        req = urllib.request.Request(
            f"{base}/api/v1/sessions/{sid}/events",
            data=raw,
            method="POST",
            headers={"Content-Type": "application/json", "Content-Length": str(len(raw))},
        )
        try:
            with urllib.request.urlopen(req):
                pass
            pytest.fail("Expected 400")
        except urllib.error.HTTPError as e:
            assert e.code == 400


# ===========================================================================
# Control
# ===========================================================================

class TestControl:
    def test_get_control_default_state(self, gateway):
        base, db = gateway
        sid = "e2e-ctrl-01"
        resp = _get(f"{base}/api/v1/sessions/{sid}/control")
        assert resp == {"pause": False, "resume": False}

    def test_post_pause_true(self, gateway):
        base, db = gateway
        sid = "e2e-ctrl-02"
        resp = _post(f"{base}/api/v1/sessions/{sid}/control", {"pause": True})
        assert resp["ok"] is True
        assert resp["control"]["pause"] is True
        assert resp["control"]["resume"] is False   # pause clears resume

    def test_post_resume_true(self, gateway):
        base, db = gateway
        sid = "e2e-ctrl-03"
        _post(f"{base}/api/v1/sessions/{sid}/control", {"pause": True})
        resp = _post(f"{base}/api/v1/sessions/{sid}/control", {"resume": True})
        assert resp["control"]["resume"] is True
        assert resp["control"]["pause"] is False    # resume clears pause

    def test_control_reflected_in_get(self, gateway):
        base, db = gateway
        sid = "e2e-ctrl-04"
        _post(f"{base}/api/v1/sessions/{sid}/control", {"pause": True})
        resp = _get(f"{base}/api/v1/sessions/{sid}/control")
        assert resp["pause"] is True

    def test_control_log_records_inbound(self, gateway):
        base, db = gateway
        sid = "e2e-ctrl-05"
        _post(f"{base}/api/v1/sessions/{sid}/control", {"pause": True})
        log = _get(f"{base}/api/v1/sessions/{sid}/control/log")
        assert isinstance(log, list)
        actions = [r["action"] for r in log]
        assert "pause" in actions

    def test_control_log_records_outbound_poll(self, gateway):
        base, db = gateway
        sid = "e2e-ctrl-06"
        _get(f"{base}/api/v1/sessions/{sid}/control")
        log = _get(f"{base}/api/v1/sessions/{sid}/control/log")
        directions = [r["direction"] for r in log]
        assert "outbound" in directions


# ===========================================================================
# Stats
# ===========================================================================

class TestStats:
    def test_global_stats_structure(self, gateway):
        base, db = gateway
        stats = _get(f"{base}/api/v1/stats")
        required = {
            "total_sessions", "total_state_snapshots",
            "total_events", "total_requests", "total_control_actions", "db_path",
        }
        assert required <= set(stats.keys())

    def test_session_stats_structure(self, gateway):
        base, db = gateway
        sid = "e2e-stats-01"
        _post(f"{base}/api/v1/sessions/{sid}/state", _make_state_payload(sid))
        _post(f"{base}/api/v1/sessions/{sid}/events", _make_event_payload("stat-e1", "ActionEvent"))
        stats = _get(f"{base}/api/v1/sessions/{sid}/stats")
        assert stats["session_id"] == sid
        assert stats["event_count"] >= 1
        assert "event_type_breakdown" in stats
        assert stats["state_snapshot_count"] >= 1

    def test_global_stats_counts_increase(self, gateway):
        base, db = gateway
        stats_before = _get(f"{base}/api/v1/stats")
        sid = "e2e-stats-02"
        _post(f"{base}/api/v1/sessions/{sid}/state", _make_state_payload(sid))
        _post(f"{base}/api/v1/sessions/{sid}/events", _make_event_payload("cnt-e1"))
        stats_after = _get(f"{base}/api/v1/stats")
        assert stats_after["total_state_snapshots"] > stats_before["total_state_snapshots"]


# ===========================================================================
# Request log
# ===========================================================================

class TestRequestLog:
    def test_global_request_log_populated(self, gateway):
        base, db = gateway
        rows = _get(f"{base}/api/v1/requests")
        assert isinstance(rows, list)
        # The test itself has made multiple requests → log is non-empty
        assert len(rows) > 0

    def test_session_request_log(self, gateway):
        base, db = gateway
        sid = "e2e-reqlog-01"
        _post(f"{base}/api/v1/sessions/{sid}/state", _make_state_payload(sid))
        rows = _get(f"{base}/api/v1/sessions/{sid}/requests")
        assert isinstance(rows, list)
        assert len(rows) >= 1
        assert all(r["session_id"] == sid for r in rows)


# ===========================================================================
# 404 for unknown routes
# ===========================================================================

class TestUnknownRoutes:
    def test_get_unknown_route_404(self, gateway):
        base, db = gateway
        req = urllib.request.Request(f"{base}/api/v1/does-not-exist", method="GET")
        try:
            with urllib.request.urlopen(req):
                pass
            pytest.fail("Expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404

    def test_post_unknown_route_404(self, gateway):
        base, db = gateway
        req = urllib.request.Request(
            f"{base}/api/v1/totally-unknown",
            data=b"{}",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req):
                pass
            pytest.fail("Expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404


# ===========================================================================
# Full round-trip: simulate container ↔ gateway interaction
# ===========================================================================

class TestContainerGatewayRoundTrip:
    """Simulate a realistic container session: upload state + events → query."""

    def test_full_session_roundtrip(self, gateway):
        base, db = gateway
        sid = "e2e-roundtrip-01"

        # 1. Container uploads initial state
        initial = _make_state_payload(sid, phase="initializing", iteration=0)
        _post(f"{base}/api/v1/sessions/{sid}/state", initial)

        # 2. Container uploads events
        for i in range(5):
            ev_type = "ActionEvent" if i % 2 == 0 else "ObservationEvent"
            _post(f"{base}/api/v1/sessions/{sid}/events", _make_event_payload(f"rt-{i}", ev_type))

        # 3. Gateway issues pause
        _post(f"{base}/api/v1/sessions/{sid}/control", {"pause": True})

        # 4. Container polls control
        ctrl = _get(f"{base}/api/v1/sessions/{sid}/control")
        assert ctrl["pause"] is True

        # 5. Container uploads paused state
        paused = _make_state_payload(sid, phase="paused", iteration=3)
        _post(f"{base}/api/v1/sessions/{sid}/state", paused)

        # 6. Gateway resumes
        _post(f"{base}/api/v1/sessions/{sid}/control", {"resume": True})

        # 7. Container finishes and uploads final state
        done = _make_state_payload(sid, phase="finished", iteration=5)
        done["is_running"] = False
        _post(f"{base}/api/v1/sessions/{sid}/state", done)

        # --- Assertions ---
        final_state = _get(f"{base}/api/v1/sessions/{sid}/state")
        assert final_state["phase"] == "finished"
        assert final_state["iteration"] == 5

        events_resp = _get(f"{base}/api/v1/sessions/{sid}/events")
        assert events_resp["total"] == 5

        action_events = _get(f"{base}/api/v1/sessions/{sid}/events?type=ActionEvent")
        assert action_events["total"] == 3   # indices 0, 2, 4

        stats = _get(f"{base}/api/v1/sessions/{sid}/stats")
        assert stats["event_type_breakdown"].get("ActionEvent") == 3
        assert stats["event_type_breakdown"].get("ObservationEvent") == 2

        snaps = _get(f"{base}/api/v1/sessions/{sid}/snapshots")
        assert len(snaps) == 3   # initial + paused + final

        ctrl_log = _get(f"{base}/api/v1/sessions/{sid}/control/log")
        actions = {r["action"] for r in ctrl_log}
        assert "pause" in actions
        assert "resume" in actions
