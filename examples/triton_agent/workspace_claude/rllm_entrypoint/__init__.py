"""
rllm_entrypoint — container-side sub-project for OpenHands rllm training.

Package structure:
    config.py       — Environment variable parsing
    state.py        — RunState dataclass (shared, thread-safe)
    events.py       — OpenHands event hook + lifecycle event push
    api_client.py   — REST API client (push events, query pause)
    pause_ctrl.py   — Background pause-poller thread
    runner.py       — Main conversation runner logic

"""
