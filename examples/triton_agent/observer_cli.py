#!/usr/bin/env python3
"""
observer_cli.py — Command-line tool for inspecting rllm OpenHands sessions.

Usage:
    python observer_cli.py sessions                      # list all sessions
    python observer_cli.py state   <id> [-v]             # live state
    python observer_cli.py events  <id> [--limit N] [--type T] [--since UNIX_TS]
    python observer_cli.py payload <id> <db_event_id>    # full event payload
    python observer_cli.py snapshots <id> [--limit N]    # snapshot index
    python observer_cli.py snapshot  <id> <snap_id>      # full snapshot payload
    python observer_cli.py commands <id>                 # command history (from state)
    python observer_cli.py requests [<id>] [--limit N] [--method GET|POST]
    python observer_cli.py ctrllog  <id> [--limit N]     # control action log
    python observer_cli.py stats   [<id>]                # stats (global or per-session)
    python observer_cli.py pause  <id>
    python observer_cli.py resume <id>
    python observer_cli.py watch  <id> [--interval N]

Environment:
    OBSERVER_API_URL    Base URL of the observer gateway (default: http://localhost:18858)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any


# Mutable config dict — avoids 'global' keyword scoping issues
_CONFIG = {"base_url": os.environ.get("OBSERVER_API_URL", "http://localhost:18858").rstrip("/")}


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(path: str) -> Any:
    base = _CONFIG["base_url"]
    url = f"{base}{path}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as exc:
        print(f"[ERROR] Cannot reach gateway at {base}: {exc.reason}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Bad JSON from gateway: {exc}", file=sys.stderr)
        sys.exit(1)


def _post(path: str, body: dict) -> Any:
    base = _CONFIG["base_url"]
    url = f"{base}{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as exc:
        print(f"[ERROR] {exc.reason}", file=sys.stderr)
        sys.exit(1)


def _qs(**kwargs: Any) -> str:
    """Build a query string from kwargs, skipping None values."""
    parts = [f"{k}={v}" for k, v in kwargs.items() if v is not None]
    return ("?" + "&".join(parts)) if parts else ""


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_time(ts: float | None) -> str:
    if ts is None:
        return "N/A"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return str(ts)


def _fmt_uptime(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _pp(obj: Any) -> None:
    """Pretty-print a JSON-serialisable object."""
    print(json.dumps(obj, indent=2, default=str))


# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_sessions(args: argparse.Namespace) -> None:
    sessions = _get("/api/v1/sessions")
    if not sessions:
        print("No sessions found.")
        return
    print(f"{'SESSION ID':<40} {'LABEL':<18} {'PHASE':<22} {'ITER':>5} "
          f"{'UPTIME':>10} {'EVENTS':>8} {'UPDATED'}")
    print("─" * 130)
    for s in sessions:
        now_str = _fmt_time(s.get("updated_at"))
        ev = s.get("db_events_count", s.get("mem_events_count", 0))
        print(
            f"{s['session_id']:<40} "
            f"{str(s.get('label', '')):<18} "
            f"{s.get('phase', '?'):<22} "
            f"{s.get('iteration', 0):>5} "
            f"{_fmt_uptime(s.get('uptime_seconds', 0)):>10} "
            f"{ev:>8} "
            f"{now_str}"
        )


def cmd_state(args: argparse.Namespace) -> None:
    state = _get(f"/api/v1/sessions/{args.session_id}/state")
    if not state:
        print("No state found.")
        return

    print(f"{'='*65}")
    print(f" Session:   {state.get('session_id', '?')}")
    print(f" Label:     {state.get('session_label', '?')}")
    print(f" Phase:     {state.get('phase', '?')}")
    print(f" Status:    {state.get('conversation_execution_status', '?')}")
    print(f" Running:   {state.get('is_running', '?')}")
    print(f" Uptime:    {_fmt_uptime(state.get('uptime_seconds', 0))}")
    print(f" Started:   {_fmt_time(state.get('start_time'))}")
    print(f"{'='*65}")
    print(f" Model:     {state.get('llm_model', '?')}")
    print(f" LLM URL:   {state.get('llm_base_url_prefix', '?')}")
    print(f" Conv ID:   {state.get('conversation_id', '?')}")
    print(f" Iter:      {state.get('iteration', 0)} / {state.get('max_iterations', 0)}")
    print(f" LLM Cost:  {state.get('accumulated_cost', 0.0):.4f}")
    print(f" Events:    {state.get('events_count', 0)} (in payload)")
    print(f"{'='*65}")
    task = state.get("task_instruction", "")
    print(f" Task:\n   {task[:400]}")
    last_err = state.get("last_error", "")
    if last_err:
        print(f"\n[ERROR] {last_err}")
    print()

    if getattr(args, "verbose", False):
        ctx = state.get("llm_context_snapshot", [])
        if ctx:
            print(f" LLM Context ({len(ctx)} messages, showing last 5):")
            for msg in ctx[-5:]:
                role = msg.get("role", "?")
                content = str(msg.get("content", ""))[:200]
                print(f"   [{role}] {content}")
            print()


def cmd_events(args: argparse.Namespace) -> None:
    qs = _qs(
        limit=args.limit,
        offset=getattr(args, "offset", None),
        type=getattr(args, "type", None) or None,
        since=getattr(args, "since", None),
    )
    result = _get(f"/api/v1/sessions/{args.session_id}/events{qs}")
    # Backwards compat: might be a plain list (no-db mode)
    if isinstance(result, list):
        items = result
        total = len(items)
    else:
        items = result.get("items", [])
        total = result.get("total", len(items))

    if not items:
        print("No events found.")
        return
    print(f"Showing {len(items)} / {total} events")
    print(f"{'DB_ID':>6} {'TYPE':<30} {'SOURCE':<12} {'RECEIVED':<22} {'SUMMARY'}")
    print("─" * 120)
    for ev in items:
        db_id = ev.get("id", "?")
        ts = _fmt_time(ev.get("received_at"))
        print(
            f"{str(db_id):>6} "
            f"{ev.get('event_type', '?'):<30} "
            f"{ev.get('source', '?'):<12} "
            f"{ts:<22} "
            f"{ev.get('summary', '')[:60]}"
        )


def cmd_payload(args: argparse.Namespace) -> None:
    """Show full raw payload of a single persisted event."""
    result = _get(f"/api/v1/sessions/{args.session_id}/events/{args.db_id}/payload")
    _pp(result)


def cmd_snapshots(args: argparse.Namespace) -> None:
    qs = _qs(limit=args.limit, since=getattr(args, "since", None))
    snaps = _get(f"/api/v1/sessions/{args.session_id}/snapshots{qs}")
    if not snaps:
        print("No snapshots found.")
        return
    print(f"{'ID':>8} {'RECEIVED':<22} {'SIZE_BYTES':>12} {'CLIENT_IP'}")
    print("─" * 65)
    for s in snaps:
        print(
            f"{s.get('id', '?'):>8} "
            f"{_fmt_time(s.get('received_at')):<22} "
            f"{s.get('payload_bytes', 0):>12} "
            f"{s.get('client_ip', '')}"
        )


def cmd_snapshot(args: argparse.Namespace) -> None:
    """Show full payload of one snapshot."""
    result = _get(f"/api/v1/sessions/{args.session_id}/snapshots/{args.snap_id}")
    if isinstance(result.get("payload"), dict):
        _pp(result["payload"])
    else:
        _pp(result)


def cmd_commands(args: argparse.Namespace) -> None:
    state = _get(f"/api/v1/sessions/{args.session_id}/state")
    cmds = state.get("command_history", [])
    if not cmds:
        print("No command history in latest state snapshot.")
        return
    for i, c in enumerate(cmds, 1):
        print(f"\n[{i}] {_fmt_time(c.get('timestamp'))}")
        print(f"    CMD:  {c.get('cmd', '')[:200]}")
        stdout = c.get("stdout", "")
        if stdout:
            print(f"    OUT:  {stdout[:300]}")
        stderr = c.get("stderr", "")
        if stderr:
            print(f"    ERR:  {stderr[:300]}")
        print(f"    EXIT: {c.get('exit_code', '?')}")


def cmd_requests(args: argparse.Namespace) -> None:
    sid = getattr(args, "session_id", None)
    qs = _qs(
        limit=args.limit,
        since=getattr(args, "since", None),
        method=getattr(args, "method", None) or None,
    )
    if sid:
        rows = _get(f"/api/v1/sessions/{sid}/requests{qs}")
    else:
        rows = _get(f"/api/v1/requests{qs}")
    if not rows:
        print("No request log entries.")
        return
    print(f"{'ID':>7} {'METHOD':<7} {'STATUS':>6} {'TIME_MS':>8} {'BYTES':>7} "
          f"{'RECEIVED':<22} {'SESSION_ID':<36} {'PATH'}")
    print("─" * 135)
    for r in rows:
        print(
            f"{r.get('id', '?'):>7} "
            f"{r.get('method', '?'):<7} "
            f"{r.get('status_code', 0):>6} "
            f"{r.get('response_ms', 0):>8.1f} "
            f"{r.get('body_bytes', 0):>7} "
            f"{_fmt_time(r.get('received_at')):<22} "
            f"{str(r.get('session_id') or ''):36} "
            f"{r.get('path', '')}"
        )


def cmd_ctrllog(args: argparse.Namespace) -> None:
    qs = _qs(limit=args.limit, since=getattr(args, "since", None))
    rows = _get(f"/api/v1/sessions/{args.session_id}/control/log{qs}")
    if not rows:
        print("No control log entries.")
        return
    print(f"{'ID':>7} {'DIRECTION':<10} {'ACTION':<8} {'VALUE':>6} "
          f"{'RECEIVED':<22} {'CLIENT_IP'}")
    print("─" * 75)
    for r in rows:
        val = r.get("value")
        val_str = "true" if val == 1 else ("false" if val == 0 else "—")
        print(
            f"{r.get('id', '?'):>7} "
            f"{r.get('direction', '?'):<10} "
            f"{r.get('action', '?'):<8} "
            f"{val_str:>6} "
            f"{_fmt_time(r.get('received_at')):<22} "
            f"{r.get('client_ip', '')}"
        )


def cmd_stats(args: argparse.Namespace) -> None:
    sid = getattr(args, "session_id", None)
    if sid:
        result = _get(f"/api/v1/sessions/{sid}/stats")
    else:
        result = _get("/api/v1/stats")
    _pp(result)


def cmd_pause(args: argparse.Namespace) -> None:
    result = _post(f"/api/v1/sessions/{args.session_id}/control", {"pause": True})
    print(f"Pause signal sent → {result}")


def cmd_resume(args: argparse.Namespace) -> None:
    result = _post(f"/api/v1/sessions/{args.session_id}/control", {"resume": True})
    print(f"Resume signal sent → {result}")


def cmd_watch(args: argparse.Namespace) -> None:
    interval = args.interval
    print(f"Watching {args.session_id} (refresh every {interval}s). Ctrl-C to stop.")
    try:
        while True:
            print("\033[H\033[J", end="")  # clear screen
            args.verbose = False
            cmd_state(args)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nWatching stopped.")


# ── CLI parser ────────────────────────────────────────────────────────────────

def main() -> None:
    _default_url = _CONFIG["base_url"]
    parser = argparse.ArgumentParser(
        description="rllm OpenHands observer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url", default=_default_url,
        help=f"Observer gateway base URL (default: {_default_url})"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # sessions
    sub.add_parser("sessions", help="List all known sessions")

    # state
    p = sub.add_parser("state", help="Show live state for a session")
    p.add_argument("session_id")
    p.add_argument("-v", "--verbose", action="store_true")

    # events
    p = sub.add_parser("events", help="Show persisted event history")
    p.add_argument("session_id")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--type", dest="type", default="", help="Filter by event_type")
    p.add_argument("--since", type=float, default=None, help="Unix timestamp lower bound")

    # payload
    p = sub.add_parser("payload", help="Show full raw payload of one persisted event")
    p.add_argument("session_id")
    p.add_argument("db_id", type=int, help="DB id from 'events' command")

    # snapshots
    p = sub.add_parser("snapshots", help="List state snapshot index for a session")
    p.add_argument("session_id")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--since", type=float, default=None)

    # snapshot (single)
    p = sub.add_parser("snapshot", help="Show full payload of one state snapshot")
    p.add_argument("session_id")
    p.add_argument("snap_id", type=int, help="Snapshot ID from 'snapshots' command")

    # commands
    p = sub.add_parser("commands", help="Show command history from latest state snapshot")
    p.add_argument("session_id")

    # requests
    p = sub.add_parser("requests", help="Show HTTP request log")
    p.add_argument("session_id", nargs="?", default=None, help="Filter by session (optional)")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--since", type=float, default=None)
    p.add_argument("--method", default="", help="Filter by HTTP method (GET, POST, …)")

    # ctrllog
    p = sub.add_parser("ctrllog", help="Show control action log for a session")
    p.add_argument("session_id")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--since", type=float, default=None)

    # stats
    p = sub.add_parser("stats", help="Show stats (global or per-session)")
    p.add_argument("session_id", nargs="?", default=None)

    # pause
    p = sub.add_parser("pause", help="Request pause on a session")
    p.add_argument("session_id")

    # resume
    p = sub.add_parser("resume", help="Request resume on a paused session")
    p.add_argument("session_id")

    # watch
    p = sub.add_parser("watch", help="Live-watch a session state")
    p.add_argument("session_id")
    p.add_argument("--interval", type=float, default=3.0)

    args = parser.parse_args()
    _CONFIG["base_url"] = args.url.rstrip("/")

    dispatch = {
        "sessions": cmd_sessions,
        "state": cmd_state,
        "events": cmd_events,
        "payload": cmd_payload,
        "snapshots": cmd_snapshots,
        "snapshot": cmd_snapshot,
        "commands": cmd_commands,
        "requests": cmd_requests,
        "ctrllog": cmd_ctrllog,
        "stats": cmd_stats,
        "pause": cmd_pause,
        "resume": cmd_resume,
        "watch": cmd_watch,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
