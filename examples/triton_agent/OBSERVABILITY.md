# OpenHands SDK Example — Observability & Control System (v2)

This documentation covers the **observability and control** infrastructure for the
`openhands_sdk` example. It explains how to monitor running containers, browse events,
and send pause/resume commands.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│ rllm Training Host                                                   │
│                                                                      │
│  openhands_agent.py ──docker run──► Container                        │
│                                     ├─ entrypoint.py (thin shell)    │
│                                     └─ rllm_entrypoint/              │
│                                          ├─ config.py                │
│                                          ├─ state.py (RunState)      │
│                                          ├─ events.py ─── push ──►   │
│                                          ├─ api_client.py            │
│                                          ├─ pause_ctrl.py (thread) ◄─┤
│                                          └─ runner.py                │
│                                                 │ HeartbeatTimer     │
│                                                 └──── push ─────►    │
│                                                                      │
│  observer_gateway.py ◄─── POST /events (per-event push) ────────────┘
│  (HTTP :18858)        ──── GET  /control ─────────────────────────► │
│      │                                                              │
│      └── persistence.py (SQLite: observer_data.db)                  │
│                                                                      │
│  observer_cli.py      ──── interacts with gateway ──────────────►   │
└──────────────────────────────────────────────────────────────────────┘
```

### Communication Model (v2: Event-Driven)

All communication is **push** — the container initiates every HTTP request.

| Direction | Endpoint | When |
|-----------|----------|------|
| Container → Gateway | `POST /events` | **Immediately** on each OpenHands SDK event |
| Container → Gateway | `POST /events` | On lifecycle events (startup / heartbeat / evaluate / finish) |
| Container → Gateway | `GET /control` | Every `OBSERVER_PAUSE_POLL_INTERVAL` seconds |
| CLI/script → Gateway | `POST /control` | To send pause/resume |

> **v2 change**: The `StateUploader` background thread has been **removed**.  
> Events are pushed synchronously inside the event callback (zero latency).  
> A lightweight `HeartbeatTimer` daemon in `runner.py` sends a `HeartbeatEvent` every  
> `OBSERVER_UPLOAD_INTERVAL` seconds to keep the gateway alive during long LLM calls.

---

## Key Components

| File | Role |
|------|------|
| `workspace/entrypoint.py` | Container ENTRYPOINT — thin shell, delegates to package |
| `workspace/rllm_entrypoint/config.py` | All env-var config, frozen dataclass |
| `workspace/rllm_entrypoint/state.py` | Thread-safe `RunState` — single source of truth |
| `workspace/rllm_entrypoint/events.py` | OpenHands event callback → direct push; lifecycle helpers |
| `workspace/rllm_entrypoint/api_client.py` | Stdlib-only REST client (`push_event`, `fetch_control`) |
| `workspace/rllm_entrypoint/pause_ctrl.py` | Daemon thread: pause/resume polling |
| `workspace/rllm_entrypoint/runner.py` | Full conversation lifecycle + HeartbeatTimer |
| `workspace/rllm_entrypoint/uploader.py` | ~~Deprecated~~ (no longer used) |
| `observer_gateway.py` | Standalone REST server (stdlib only, no external deps) |
| `observer_cli.py` | CLI for humans / scripts |
| `persistence.py` | SQLite persistence layer (`GatewayDB`) |

---

## Quick Start

### 1. Start the Observer Gateway

```bash
# Default port 18858, DB in CWD
python examples/openhands_sdk/observer_gateway.py

# With debug logging to see every request
python examples/openhands_sdk/observer_gateway.py \
  --port 18858 --db /data/runs.db --log-level DEBUG
```

The gateway persists **all** uploaded data to SQLite:
- Session records (populated from the first `StartupEvent`)
- State snapshots (written on `StartupEvent`, `EvaluateEvent`, `FinishEvent`)
- Every OpenHands event (deduplicated by `event_id`)
- Every HTTP request (method, path, timing, status)
- Control actions (pause/resume) with direction and IP

### 2. Launch a Container with Observability Enabled

```bash
docker run --rm \
  -e LLM_BASE_URL="http://host.docker.internal:4000/.../v1" \
  -e LLM_MODEL="openai/openhands-model" \
  -e LLM_API_KEY="EMPTY" \
  -e TASK_INSTRUCTION="Fix the bug in main.py" \
  -e WORKSPACE_BASE=/opt/workspace \
  -e MAX_ITERATIONS=30 \
  -e OBSERVER_API_URL="http://host.docker.internal:18858" \
  -e OBSERVER_SESSION_ID="run-001" \
  -e OBSERVER_SESSION_LABEL="my-experiment" \
  -e OBSERVER_UPLOAD_INTERVAL="15" \
  -e OBSERVER_PAUSE_POLL_INTERVAL="2" \
  --add-host host.docker.internal:host-gateway \
  -v /path/to/workspace:/opt/workspace \
  rllm-openhands
```

> `OBSERVER_UPLOAD_INTERVAL` controls the **heartbeat** period (default 15 s).  
> Individual SDK events are always pushed immediately, regardless of this setting.

### 3. Monitor from the CLI

```bash
python observer_cli.py sessions                       # list all sessions
python observer_cli.py watch   run-001                # live refresh every 3s
python observer_cli.py state   run-001 --verbose      # detailed state
python observer_cli.py events  run-001 --limit 50     # recent events
python observer_cli.py events  run-001 --type ActionEvent   # filter by type
python observer_cli.py payload run-001 <db_event_id>  # full event payload
python observer_cli.py commands run-001               # command history
python observer_cli.py stats   run-001                # event-type breakdown
python observer_cli.py pause   run-001                # send pause signal
python observer_cli.py resume  run-001                # send resume signal
```

---

## Lifecycle Events

These are **rllm-specific** events (not native OpenHands SDK events), pushed at key
moments by `runner.py`. They are stored in the same `events` table and also trigger
`state_snapshots` writes.

### `StartupEvent`

Pushed once, immediately after `RunState` is initialized (before `conversation.run()`).

```json
{
  "event_type": "StartupEvent",
  "source": "rllm_entrypoint",
  "summary": "[Startup] Container initialised and runner started",
  "session_id": "run-001",
  "session_label": "my-experiment",
  "llm_model": "openai/openhands-model",
  "llm_base_url": "http://host.docker.internal:4000/...",
  "workspace_base": "/opt/workspace",
  "max_iterations": 30,
  "task_instruction": "Fix the bug in main.py",
  "task_preview": "Fix the bug in main.py",
  "env_vars": {"WORKSPACE_BASE": "/opt/workspace", "LLM_API_KEY": "***REDACTED***", ...},
  "extra_metadata": {},
  "start_time": 1712345678.0
}
```

→ Gateway action: **upserts session record** + writes initial `state_snapshot`.

### `HeartbeatEvent`

Pushed every `OBSERVER_UPLOAD_INTERVAL` seconds (default: 15 s) by `_HeartbeatTimer`.
Provides a liveness signal during long LLM calls when no SDK events are fired.

```json
{
  "event_type": "HeartbeatEvent",
  "source": "rllm_entrypoint",
  "summary": "[Heartbeat] phase=waiting_for_reply iteration=3 running=True",
  "phase": "waiting_for_reply",
  "iteration": 3,
  "is_running": true,
  "uptime_seconds": 62.1,
  "conversation_id": "3f9a1b2c-...",
  "conversation_execution_status": "running",
  "accumulated_cost": 0.0012,
  "total_llm_calls": 5
}
```

→ Gateway action: updates in-memory live state for `GET /state`.

### `EvaluateEvent`

Pushed once when `conversation.run()` reaches a terminal state (before `FinishEvent`).
Contains the full run summary including command history and LLM context.

```json
{
  "event_type": "EvaluateEvent",
  "summary": "[Evaluate] status=finished cost=0.0048 llm_calls=12 iterations=8",
  "status": "finished",
  "accumulated_cost": 0.0048,
  "total_llm_calls": 12,
  "iterations": 8,
  "conversation_id": "3f9a1b2c-...",
  "conversation_execution_status": "finished",
  "command_history": [{"cmd": "pytest", "stdout": "...", "exit_code": 0}, ...],
  "llm_context_snapshot": [{"role": "user", "content": "..."}, ...],
  "last_error": "",
  "uptime_seconds": 187.4
}
```

→ Gateway action: writes `state_snapshot`.

### `FinishEvent`

Pushed as the very last thing the container does (in `finally` block). Contains the
complete `RunState.to_dict()` snapshot embedded in `final_state`.

```json
{
  "event_type": "FinishEvent",
  "summary": "[Finish] exit_code=0 phase=finished reason=completed",
  "exit_code": 0,
  "phase": "finished",
  "iteration": 8,
  "reason": "completed",
  "final_state": { /* full RunState.to_dict() */ }
}
```

→ Gateway action: writes `state_snapshot` using `final_state`.

---

## REST API Reference

Base URL: `http://<host>:18858`

### Sessions

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/sessions` | List all known sessions (summary) |
| `DELETE` | `/api/v1/sessions/{id}` | Remove session and all data |

### State (derived from lifecycle events)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/sessions/{id}/state` | Latest live state (in-memory or DB fallback) |
| `POST` | `/api/v1/sessions/{id}/state` | Upload a state snapshot (still supported) |
| `GET` | `/api/v1/sessions/{id}/snapshots` | Snapshot index `?limit=N&since=unix_ts` |
| `GET` | `/api/v1/sessions/{id}/snapshots/{snap_id}` | Full payload of one snapshot |

### Events

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/sessions/{id}/events` | Push a single event (container → gateway) |
| `GET` | `/api/v1/sessions/{id}/events` | Event list `?limit=N&offset=N&type=T&since=T` |
| `GET` | `/api/v1/sessions/{id}/events/{db_id}/payload` | Full raw payload |

### Control

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/sessions/{id}/control` | Current `{pause, resume}` flags |
| `POST` | `/api/v1/sessions/{id}/control` | Set `{"pause": true}` or `{"resume": true}` |
| `GET` | `/api/v1/sessions/{id}/control/log` | Control action history |

### Diagnostics

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/sessions/{id}/stats` | Event-type breakdown + counts |
| `GET` | `/api/v1/sessions/{id}/requests` | HTTP request log for this session |
| `GET` | `/api/v1/stats` | Global counts |
| `GET` | `/api/v1/requests` | All HTTP requests |

---

## `GET /state` — Available Fields

The live state is updated from lifecycle events (no separate state push needed).

| Field | Source event |
|-------|-------------|
| `session_id`, `session_label` | StartupEvent |
| `task_instruction` | StartupEvent |
| `llm_model`, `llm_base_url_prefix` | StartupEvent |
| `workspace_base`, `max_iterations` | StartupEvent |
| `env_vars` | StartupEvent |
| `extra_metadata` | StartupEvent |
| `start_time` | StartupEvent |
| `phase` | HeartbeatEvent / SDK events (ActionEvent etc.) |
| `iteration` | HeartbeatEvent / ActionEvent count |
| `is_running` | HeartbeatEvent / FinishEvent |
| `uptime_seconds` | HeartbeatEvent |
| `conversation_id` | HeartbeatEvent |
| `conversation_execution_status` | HeartbeatEvent / EvaluateEvent |
| `accumulated_cost`, `total_llm_calls` | HeartbeatEvent / EvaluateEvent |
| `command_history` | EvaluateEvent |
| `llm_context_snapshot` | EvaluateEvent |
| `last_error` | EvaluateEvent / FinishEvent |
| Full snapshot | FinishEvent (`final_state`) |

---

## Agent Phase Values

| Phase | Meaning |
|-------|---------|
| `initializing` | SDK objects being built |
| `waiting_for_reply` | LLM call in flight |
| `executing_command` | Tool (terminal/file) executing |
| `paused` | Paused by gateway request |
| `finished` | Completed normally |
| `error` | Unhandled exception |

---

## Native OpenHands SDK Event Types

These events are fired by the OpenHands SDK and immediately pushed to the gateway.

| `event_type` | Trigger |
|-------------|---------|
| `ActionEvent` | Agent decides to call a tool (after LLM reply) |
| `ObservationEvent` | Tool execution result returned |
| `MessageEvent` | User message or agent message |
| `PauseEvent` | `conversation.pause()` was called |
| `AgentErrorEvent` | Agent error |
| `ConversationStateUpdateEvent` | Conversation state field changed |
| `LLMCompletionLogEvent` | LLM call completed |

Each SDK event has this shape:

```json
{
  "event_type": "ActionEvent",
  "event_id": "3f8a...",
  "source": "agent",
  "timestamp": "2024-04-01T12:00:00",
  "summary": "[Action:terminal] Running pytest...",
  "raw": { /* truncated model_dump() */ }
}
```

---

## Gateway DEBUG Logging

Run the gateway with `--log-level DEBUG` to see all requests:

```
2026-04-06 01:00:00 [DEBUG] POST /api/v1/sessions/run-001/events  status=200  body=1234b  1.2ms  client=172.17.0.2
2026-04-06 01:00:00 [DEBUG] [gateway] POST event  session=run-001  type=StartupEvent  summary=[Startup] Container initialised...
2026-04-06 01:00:00 [DEBUG] [gateway] state_snapshot written (trigger=StartupEvent  session=run-001)
2026-04-06 01:00:15 [DEBUG] POST /api/v1/sessions/run-001/events  status=200  body=312b  0.8ms  client=172.17.0.2
2026-04-06 01:00:15 [DEBUG] [gateway] POST event  session=run-001  type=HeartbeatEvent  summary=[Heartbeat] phase=waiting_for_reply ...
2026-04-06 01:00:17 [DEBUG] GET  /api/v1/sessions/run-001/control  status=200  body=35b   0.3ms  client=172.17.0.2
2026-04-06 01:00:17 [DEBUG] [gateway] GET control  session=run-001  pause=False  resume=False  from=172.17.0.2
```

---

## Environment Variables Reference

### Core (required)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | — | Proxied rllm LiteLLM URL |
| `LLM_API_KEY` | `EMPTY` | API key |
| `LLM_MODEL` | `openai/openhands-model` | Model name |
| `TASK_INSTRUCTION` | (from INSTRUCTIONS.md) | Task instruction text |
| `WORKSPACE_BASE` | `/opt/workspace` | Workspace directory |
| `MAX_ITERATIONS` | `30` | Max agent iterations |
| `NPU_OPERATOR_TASK` | `0` | `1` = kernel/operator task |
| `OPERATOR_BACKEND` | `triton` | `triton` \| `ascendc` |

### Observability (all optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `OBSERVER_API_URL` | `""` | Base URL of gateway; empty = disabled |
| `OBSERVER_SESSION_ID` | random UUID | Session identifier |
| `OBSERVER_SESSION_LABEL` | first 12 chars of ID | Human label |
| `OBSERVER_UPLOAD_INTERVAL` | `15.0` | **Heartbeat** interval (seconds) |
| `OBSERVER_PAUSE_POLL_INTERVAL` | `2.0` | Pause/resume poll interval (seconds) |
| `OBSERVER_EXTRA_METADATA` | `""` | JSON string with arbitrary metadata |
| `SYSTEM_PROMPT_PATH` | `/tmp/rllm_minimal_system.j2` | System prompt path |

> `OBSERVER_UPLOAD_INTERVAL` previously controlled the state-upload period.  
> In v2 it controls the heartbeat frequency. Individual events are always pushed immediately.

---

## Pause / Resume Flow

1. **Pause**: `POST /api/v1/sessions/{id}/control` with `{"pause": true}`
2. Container's `pause_ctrl` thread polls gateway every `OBSERVER_PAUSE_POLL_INTERVAL` seconds
3. On detecting `pause: true` → calls `conversation.pause()` (SDK thread-safe)
4. Conversation loop breaks at next agent step boundary; phase becomes `paused`
5. **Resume**: `POST /api/v1/sessions/{id}/control` with `{"resume": true}`
6. `pause_ctrl` sets `resume_requested = True`
7. Main runner loop detects flag → calls `conversation.run()` again

> **Note**: Pause takes effect at the **next step boundary** (after the current LLM call completes).

---

## Project File Structure

```
examples/openhands_sdk/
├── observer_gateway.py          ← host-side REST gateway server
├── observer_cli.py              ← CLI for humans/scripts
├── persistence.py               ← SQLite GatewayDB layer
├── openhands_agent.py           ← rllm rollout function (unchanged API)
├── train_openhands.py
├── train_openhands.sh
├── OBSERVABILITY.md             ← this file
└── workspace/
    ├── Dockerfile               ← builds rllm-openhands image
    ├── entrypoint.py            ← thin container ENTRYPOINT shell
    ├── INSTRUCTIONS.md
    └── rllm_entrypoint/         ← container-side observability package
        ├── __init__.py
        ├── config.py            ← env-var config (EntrypointConfig)
        ├── state.py             ← RunState dataclass (thread-safe)
        ├── events.py            ← SDK event callback + lifecycle helpers
        ├── api_client.py        ← stdlib REST client (push_event, fetch_control)
        ├── pause_ctrl.py        ← background pause/resume thread
        ├── runner.py            ← main conversation runner + HeartbeatTimer
        └── uploader.py          ← [DEPRECATED, not used in v2]
```
