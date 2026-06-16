#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OPENHANDS_OBSERVER_HOST="${OPENHANDS_OBSERVER_HOST:-0.0.0.0}"
export OPENHANDS_OBSERVER_PORT="${OPENHANDS_OBSERVER_PORT:-18858}"
export OPENHANDS_OBSERVER_DB="${OPENHANDS_OBSERVER_DB:-${SCRIPT_DIR}/observer_data.db}"
export OPENHANDS_OBSERVER_LOG_LEVEL="${OPENHANDS_OBSERVER_LOG_LEVEL:-INFO}"

echo "Starting OpenHands observer gateway"
echo "  bind      : ${OPENHANDS_OBSERVER_HOST}:${OPENHANDS_OBSERVER_PORT}"
echo "  dashboard : http://127.0.0.1:${OPENHANDS_OBSERVER_PORT}/"
echo "  db        : ${OPENHANDS_OBSERVER_DB}"

exec python "${SCRIPT_DIR}/observer_gateway.py" \
  --host "${OPENHANDS_OBSERVER_HOST}" \
  --port "${OPENHANDS_OBSERVER_PORT}" \
  --db "${OPENHANDS_OBSERVER_DB}" \
  --log-level "${OPENHANDS_OBSERVER_LOG_LEVEL}"
