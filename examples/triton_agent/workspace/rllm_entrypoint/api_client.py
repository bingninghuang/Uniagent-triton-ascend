"""
api_client.py — Lightweight REST API client for the observer gateway.

All communication initiated BY the container (push model):
  POST  {OBSERVER_API_URL}/api/v1/sessions/{session_id}/state
        Body: full RunState snapshot (JSON)

  GET   {OBSERVER_API_URL}/api/v1/sessions/{session_id}/control
        Returns: {"pause": bool, "resume": bool, "stop": bool}

This module uses only the standard-library ``urllib.request`` so it works
inside the OpenHands Docker image with no extra dependencies.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 10  # seconds


class ObserverClient:
    """HTTP client for the observer gateway."""

    def __init__(self, base_url: str, session_id: str, timeout: float = _DEFAULT_TIMEOUT):
        self._base = base_url.rstrip("/")
        self._session_id = session_id
        self._timeout = timeout
        self._enabled = bool(base_url)

    # ── State upload ─────────────────────────────────────────────────────────

    def push_state(self, state_json: str) -> bool:
        """POST the current state snapshot to the observer gateway.

        Returns True on success, False on any network / HTTP error.
        """
        if not self._enabled:
            return True  # no-op when observer not configured

        url = f"{self._base}/api/v1/sessions/{self._session_id}/state"
        try:
            data = state_json.encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(data)),
                },
            )
            with urllib.request.urlopen(req, timeout=self._timeout):
                pass
            return True
        except urllib.error.URLError as exc:
            logger.debug("[observer] push_state failed: %s", exc.reason)
        except Exception as exc:
            logger.debug("[observer] push_state unexpected error: %s", exc)
        return False

    # ── Control poll ─────────────────────────────────────────────────────────

    def fetch_control(self) -> dict[str, Any]:
        """GET the control flags from the observer gateway.

        Returns a dict with at least {"pause": bool, "resume": bool}.
        Returns empty dict on any network / HTTP error.
        """
        if not self._enabled:
            return {}

        url = f"{self._base}/api/v1/sessions/{self._session_id}/control"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.URLError as exc:
            logger.debug("[observer] fetch_control failed: %s", exc.reason)
        except json.JSONDecodeError as exc:
            logger.debug("[observer] fetch_control bad JSON: %s", exc)
        except Exception as exc:
            logger.debug("[observer] fetch_control unexpected error: %s", exc)
        return {}

    # ── Lifecycle events ─────────────────────────────────────────────────────

    def push_event(self, event_dict: dict[str, Any]) -> bool:
        """POST a single serialised OpenHands event to the gateway.

        Returns True on success.
        """
        if not self._enabled:
            return True

        url = f"{self._base}/api/v1/sessions/{self._session_id}/events"
        try:
            data = json.dumps(event_dict, default=str).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(data)),
                },
            )
            with urllib.request.urlopen(req, timeout=self._timeout):
                pass
            return True
        except Exception as exc:
            logger.debug("[observer] push_event failed: %s", exc)
        return False
