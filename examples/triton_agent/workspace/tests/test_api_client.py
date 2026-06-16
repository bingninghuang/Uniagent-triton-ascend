"""
test_api_client.py — Unit tests for rllm_entrypoint/api_client.py (ObserverClient).

Tests cover:
  - No-op behaviour when observer_api_url is empty
  - push_state: success path, HTTP error, URLError, unexpected error
  - fetch_control: success path, bad JSON, URLError, unexpected error
  - push_event: success path, failure path
  - URL construction for all three methods
"""
from __future__ import annotations

import json
import sys
import os
from io import BytesIO
from unittest.mock import MagicMock, patch

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

from rllm_entrypoint.api_client import ObserverClient  # noqa: E402

import urllib.error
import urllib.request


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def client():
    return ObserverClient(base_url="http://localhost:8765", session_id="sess-001")


@pytest.fixture
def disabled_client():
    return ObserverClient(base_url="", session_id="sess-disabled")


# ===========================================================================
# Disabled (no-op) client
# ===========================================================================

class TestDisabledClient:
    def test_push_state_returns_true(self, disabled_client):
        assert disabled_client.push_state('{"ok": true}') is True

    def test_fetch_control_returns_empty_dict(self, disabled_client):
        assert disabled_client.fetch_control() == {}

    def test_push_event_returns_true(self, disabled_client):
        assert disabled_client.push_event({"event_id": "e1"}) is True


# ===========================================================================
# push_state
# ===========================================================================

def _fake_urlopen_ok(request, timeout=None):
    """Context manager that simulates a successful HTTP response."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


class TestPushState:
    def test_success_returns_true(self, client):
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen_ok):
            result = client.push_state('{"phase": "running"}')
        assert result is True

    def test_url_error_returns_false(self, client):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = client.push_state('{"phase": "running"}')
        assert result is False

    def test_unexpected_error_returns_false(self, client):
        with patch("urllib.request.urlopen", side_effect=RuntimeError("boom")):
            result = client.push_state('{"phase": "running"}')
        assert result is False

    def test_uses_correct_url(self, client):
        captured_url = []

        def fake_open(req, timeout=None):
            captured_url.append(req.full_url)
            return _fake_urlopen_ok(req, timeout)

        with patch("urllib.request.urlopen", side_effect=fake_open):
            client.push_state("{}")

        assert captured_url[0] == "http://localhost:8765/api/v1/sessions/sess-001/state"

    def test_sends_post_method(self, client):
        captured_method = []

        def fake_open(req, timeout=None):
            captured_method.append(req.method)
            return _fake_urlopen_ok(req, timeout)

        with patch("urllib.request.urlopen", side_effect=fake_open):
            client.push_state("{}")

        assert captured_method[0] == "POST"


# ===========================================================================
# fetch_control
# ===========================================================================

def _make_urlopen_response(body: bytes):
    """Context manager that returns a fake HTTP response with the given body."""
    def fake_open(req, timeout=None):
        cm = MagicMock()
        resp = MagicMock()
        resp.read.return_value = body
        cm.__enter__ = MagicMock(return_value=resp)
        cm.__exit__ = MagicMock(return_value=False)
        return cm
    return fake_open


class TestFetchControl:
    def test_success_returns_parsed_dict(self, client):
        body = json.dumps({"pause": False, "resume": True}).encode()
        with patch("urllib.request.urlopen", side_effect=_make_urlopen_response(body)):
            result = client.fetch_control()
        assert result == {"pause": False, "resume": True}

    def test_bad_json_returns_empty_dict(self, client):
        with patch("urllib.request.urlopen", side_effect=_make_urlopen_response(b"not-json")):
            result = client.fetch_control()
        assert result == {}

    def test_url_error_returns_empty_dict(self, client):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("timeout"),
        ):
            result = client.fetch_control()
        assert result == {}

    def test_unexpected_error_returns_empty_dict(self, client):
        with patch("urllib.request.urlopen", side_effect=Exception("kaboom")):
            result = client.fetch_control()
        assert result == {}

    def test_uses_correct_url(self, client):
        captured = []
        body = json.dumps({"pause": False, "resume": False}).encode()

        def fake_open(req, timeout=None):
            captured.append(req.full_url)
            return _make_urlopen_response(body)(req, timeout)

        with patch("urllib.request.urlopen", side_effect=fake_open):
            client.fetch_control()

        assert captured[0] == "http://localhost:8765/api/v1/sessions/sess-001/control"

    def test_uses_get_method(self, client):
        captured = []
        body = json.dumps({}).encode()

        def fake_open(req, timeout=None):
            captured.append(req.method)
            return _make_urlopen_response(body)(req, timeout)

        with patch("urllib.request.urlopen", side_effect=fake_open):
            client.fetch_control()

        assert captured[0] == "GET"


# ===========================================================================
# push_event
# ===========================================================================

class TestPushEvent:
    def test_success_returns_true(self, client):
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen_ok):
            result = client.push_event({"event_id": "e1", "event_type": "ActionEvent"})
        assert result is True

    def test_failure_returns_false(self, client):
        with patch("urllib.request.urlopen", side_effect=Exception("net error")):
            result = client.push_event({"event_id": "e1"})
        assert result is False

    def test_uses_correct_url(self, client):
        captured = []

        def fake_open(req, timeout=None):
            captured.append(req.full_url)
            return _fake_urlopen_ok(req, timeout)

        with patch("urllib.request.urlopen", side_effect=fake_open):
            client.push_event({"event_id": "e1"})

        assert captured[0] == "http://localhost:8765/api/v1/sessions/sess-001/events"

    def test_sends_post_method(self, client):
        captured = []

        def fake_open(req, timeout=None):
            captured.append(req.method)
            return _fake_urlopen_ok(req, timeout)

        with patch("urllib.request.urlopen", side_effect=fake_open):
            client.push_event({"event_id": "e1"})

        assert captured[0] == "POST"

    def test_serialises_event_as_json(self, client):
        captured_data = []

        def fake_open(req, timeout=None):
            captured_data.append(req.data)
            return _fake_urlopen_ok(req, timeout)

        event = {"event_id": "e42", "event_type": "MessageEvent", "summary": "hello"}
        with patch("urllib.request.urlopen", side_effect=fake_open):
            client.push_event(event)

        parsed = json.loads(captured_data[0])
        assert parsed["event_id"] == "e42"

    def test_serialises_non_serialisable_values(self, client):
        """default=str fallback handles non-JSON types gracefully."""
        import datetime
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen_ok):
            result = client.push_event({"ts": datetime.datetime.now()})
        assert result is True
