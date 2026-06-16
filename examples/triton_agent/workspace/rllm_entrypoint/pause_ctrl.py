"""
pause_ctrl.py — Background thread that polls the observer gateway for a
pause/resume signal and acts on the OpenHands Conversation accordingly.

Design:
  - Runs in a daemon thread so it exits automatically with the process.
  - Every ``poll_interval_s`` seconds it calls ``client.fetch_control()``.
  - If the gateway signals ``pause=True``, it calls ``conversation.pause()``.
  - If the gateway signals ``resume=True``, it calls ``conversation.run()``
    in the same thread (note: conversation.run() is NOT re-entrant; if the
    main thread is still in run() we only set a flag and let the caller
    restart the loop after the current run() returns).

Threading contract:
  - ``conversation.pause()`` is thread-safe (documented by the SDK).
  - ``conversation.run()`` is NOT called from this thread to avoid
    re-entrancy issues. Instead, a ``resume_requested`` flag is exposed
    so the main runner can check it.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from rllm_entrypoint.api_client import ObserverClient
from rllm_entrypoint.state import AgentPhase, RunState

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class PauseController(threading.Thread):
    """Polls the gateway control endpoint and pauses/resumes the conversation."""

    def __init__(
        self,
        run_state: RunState,
        client: ObserverClient,
        poll_interval_s: float = 2.0,
    ) -> None:
        super().__init__(name="rllm-pause-ctrl", daemon=True)
        self._state = run_state
        self._client = client
        self._interval = poll_interval_s
        self._stop_event = threading.Event()
        self._conversation: Any = None   # set via set_conversation()

        # Exposed flag: if True the main runner should call conversation.run() again
        self.resume_requested: bool = False

    def set_conversation(self, conversation: Any) -> None:  # noqa: ANN401
        """Attach the OpenHands Conversation object (thread-safe, set once)."""
        self._conversation = conversation

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        logger.info("[pause_ctrl] Started (poll=%.1fs)", self._interval)
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self._interval)
            if self._stop_event.is_set():
                break

            try:
                control = self._client.fetch_control()
                if not control:
                    continue

                if control.get("pause") and self._conversation is not None:
                    logger.info("[pause_ctrl] Pause requested by gateway — pausing conversation")
                    try:
                        self._conversation.pause()
                        self._state.set_phase(AgentPhase.PAUSED)
                    except Exception:
                        logger.exception("[pause_ctrl] conversation.pause() failed")

                if control.get("resume"):
                    logger.info("[pause_ctrl] Resume requested by gateway — flagging runner")
                    self.resume_requested = True
                    self._state.set_phase(AgentPhase.WAITING_FOR_REPLY)

            except Exception:
                logger.exception("[pause_ctrl] Unexpected error in control poll loop")

        logger.info("[pause_ctrl] Thread stopping.")
