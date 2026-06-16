# =============================================================================
# Dockerfile for the rllm sandbox worker container (examples/openhands-sdk)
#
# This container is managed by rllm's SandboxOrchestrator / DockerSandbox.
# It runs worker_server.py and the openhands_agent.py rollout function.
#
# It does NOT contain OpenHands itself — OpenHands runs in its own separate
# container launched by openhands_agent.py via the Docker daemon socket.
#
# Layer topology:
#
#   Host
#   └─ rllm sandbox container  (this image)
#        ├─ worker_server.py   ← managed by SandboxOrchestrator
#        └─ rollout()          ← calls docker run → OpenHands container
#             └─ OpenHands container (ghcr.io/all-hands-ai/openhands)
#
# Build:
#   docker build -t rllm-openhands-sandbox \
#       -f examples/openhands-sdk/Dockerfile .
# =============================================================================

FROM python:3.11-slim

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Git — for cloning repos in _setup_workspace()
    git \
    # curl — for health-check polling inside DockerSandbox._wait_for_ready()
    curl \
    # docker-cli — fallback if Docker SDK is not sufficient (optional)
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    # HTTP server for worker_server.py
    aiohttp \
    # Docker SDK — used by openhands_agent.py to launch OpenHands containers
    docker

# ---------------------------------------------------------------------------
# App layout
#   /app/runner/   — worker_server.py (uploaded by SandboxOrchestrator)
#   /app/agent/    — openhands_agent.py (uploaded by SandboxOrchestrator)
# ---------------------------------------------------------------------------
RUN mkdir -p /app/runner /app/agent

WORKDIR /app

# Default entrypoint (overridden by SandboxOrchestrator via start_agent_process)
CMD ["python", "/app/runner/worker_server.py", \
     "--agent-module", "openhands_agent", \
     "--agent-func", "rollout", \
     "--agent-dir", "/app/agent", \
     "--port", "8100"]
