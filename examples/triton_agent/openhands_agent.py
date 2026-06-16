"""
OpenHands agent — rollout function for rllm training.

Architecture:
    rllm training process
    └─ rollout(task, config)
         ├─ _build_proxied_base_url(proxy_url, metadata)  ← metadata slug
         └─ docker run rllm-openhands  (workspace/Dockerfile)
              └─ workspace/entrypoint.py
                   └─ OpenHands SDK (LLM, Agent, Conversation, Tool)
                        └─ LLM calls → LLM_BASE_URL (proxied) → rllm proxy

No openhands Python library is imported in this file. OpenHands runs entirely
inside its own container (built from workspace/Dockerfile). The proxied base
URL with embedded metadata slug is passed as LLM_BASE_URL into the container
so the rllm proxy can attribute all LLM calls to the correct session.

Environment Variables:
    OPENHANDS_IMAGE             : Custom rllm-openhands image
                                  (built from workspace/Dockerfile)
                                  Default: rllm-openhands
    OPENHANDS_MODEL_NAME        : Model name on the LiteLLM proxy
                                  Default: openai/openhands-model
    OPENHANDS_MAX_ITERATIONS    : Max agent iterations, default 30
    OPENHANDS_CONTAINER_TIMEOUT : Seconds to wait for container, default 600
"""

from __future__ import annotations

import time
import base64
import io
import json
import logging
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import uuid
import urllib.error
import urllib.request
from urllib.parse import urlparse, urlunparse
from typing import Any
from pathlib import Path

from rllm.types import Trajectory
from rllm.sdk.proxy.metadata_slug import assemble_routing_metadata, build_proxied_base_url

import sys
import socket
import faulthandler
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)
faulthandler.enable(all_threads=True)


def dbg(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    host = socket.gethostname()
    pid = os.getpid()
    rank = os.getenv("RANK", "NA")
    local_rank = os.getenv("LOCAL_RANK", "NA")
    msg=f"[{ts}] [host={host}] [pid={pid}] [rank={rank}] [local_rank={local_rank}] {msg}"
    path = f"/home/p00938733/openhands_{socket.gethostname()}_{os.getpid()}.log"
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{time.time()} {msg}\n")
        f.flush()

# ---------------------------------------------------------------------------
# NPU operator mock (align with openhands-npu bring-up)
# ---------------------------------------------------------------------------

_OPENHANDS_IMAGE = os.environ.get("OPENHANDS_IMAGE", "rllm-openhands")
_MODEL_NAME = os.environ.get("OPENHANDS_MODEL_NAME", "openhands-model")
_MAX_ITERATIONS = int(os.environ.get("OPENHANDS_MAX_ITERATIONS", "30"))
_CONTAINER_TIMEOUT = int(os.environ.get("OPENHANDS_CONTAINER_TIMEOUT", "600"))
_ARTIFACT_DIR = os.environ.get("OPENHANDS_ARTIFACT_DIR", "")
_OPENHANDS_MOCK_PIPELINE = os.environ.get("OPENHANDS_MOCK_PIPELINE", "0") == "1"
_OPENHANDS_REMOTE_EVAL_URL = os.environ.get("OPENHANDS_REMOTE_EVAL_URL", "").strip().rstrip("/")
_OPENHANDS_CONTAINER_HOST_ALIAS = os.environ.get("OPENHANDS_CONTAINER_HOST_ALIAS", "host.docker.internal").strip()
_OPENHANDS_EVAL_DEVICE_IDS = os.environ.get("OPENHANDS_EVAL_DEVICE_IDS", "").strip()
_OPENHANDS_EVAL_DEVICE_COUNT = os.environ.get("OPENHANDS_EVAL_DEVICE_COUNT", "").strip()
if not _OPENHANDS_EVAL_DEVICE_COUNT:
    _OPENHANDS_EVAL_DEVICE_COUNT = str(
        len([x for x in _OPENHANDS_EVAL_DEVICE_IDS.split(",") if x.strip()])
        if _OPENHANDS_EVAL_DEVICE_IDS
        else 1
    )
# _OPENHANDS_ASCEND_VISIBLE_DEVICES = os.environ.get("OPENHANDS_ASCEND_VISIBLE_DEVICES", "").strip()

# ---------------------------------------------------------------------------
# NPU operator workspace setup
# ---------------------------------------------------------------------------


# _OPERATOR_SEED_NAMES = (
#     "AGENTS.md",
#     "INSTRUCTIONS.md",
#     ".agents",
#     "tools",
#     "src",
# )


# def _chmod_executable_scripts(directory: str) -> None:
#     """Recursively set +x on .sh and .py files under *directory*."""
#     for root, _, files in os.walk(directory):
#         for name in files:
#             if name.endswith((".sh", ".py")):
#                 path = os.path.join(root, name)
#                 try:
#                     os.chmod(
#                         path,
#                         os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
#                     )
#                 except OSError:
#                     pass


# ---------------------------------------------------------------------------
# Mock pipeline (bring-up testing without real NPU)
#
# When OPENHANDS_MOCK_PIPELINE=1, this script **replaces**
# tools/operator_pipeline.sh in the temp workspace.  The interface
# (--op_name flag, metrics.json output schema) is identical to the real
# pipeline, so INSTRUCTIONS.md and reward logic work unchanged.
# ---------------------------------------------------------------------------

_MOCK_PIPELINE_SCRIPT = r"""#!/bin/bash
# Mock operator pipeline — drop-in replacement for operator_pipeline.sh.
# Performs Python syntax + ModelNew class check; outputs metrics.json
# with randomised speedup. No NPU or torch_npu required.
set -euo pipefail

OP_NAME="operator"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --op_name) OP_NAME="$2"; shift 2;;
        *) shift;;
    esac
done

IMPL="src/${OP_NAME}_triton_ascend_impl.py"
M="metrics.json"

if [ ! -f "$IMPL" ]; then
    printf '{"success":false,"error":"File not found: %s","ast_check_ok":false,"correctness_ok":false}\n' "$IMPL" > "$M"
    echo "[mock] FAIL: $IMPL not found"; exit 0
fi

# Python syntax check
if ! python3 -c "import ast; ast.parse(open('${IMPL}').read())" 2>/dev/null; then
    printf '{"success":false,"error":"Syntax error in %s","ast_check_ok":false,"correctness_ok":false}\n' "$IMPL" > "$M"
    echo "[mock] FAIL: syntax error"; exit 0
fi

# ModelNew class check
if ! python3 -c "
import ast
tree = ast.parse(open('${IMPL}').read())
assert any(n.name == 'ModelNew' for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
" 2>/dev/null; then
    printf '{"success":false,"error":"ModelNew class not found in %s","ast_check_ok":true,"correctness_ok":false}\n' "$IMPL" > "$M"
    echo "[mock] FAIL: ModelNew not found"; exit 0
fi

# Mock success with random speedup
SP=$(python3 -c "import random; print(f'{random.uniform(0.8, 2.0):.2f}')")
TL=$(python3 -c "print(f'{1.0/float(${SP}):.4f}')")
cat > "$M" <<EOFM
{
  "success": true,
  "ast_check_ok": true,
  "correctness_ok": true,
  "perf_data": {"speedup_vs_torch": ${SP}, "torch_latency_ms": 1.0, "triton_latency_ms": ${TL}},
  "error": null
}
EOFM
echo "[mock] OK: speedup=${SP}x"
"""


# ---------------------------------------------------------------------------
# Instruction template
# ---------------------------------------------------------------------------

_NPU_INSTRUCTION_TEMPLATE = """# Current Task

- Operator name: **{op_name}**
- Target architecture: **{arch}**

{instruction}

## Task Format

Task file: `src/{op_name}.py`; it contains `Model`, `get_init_inputs()`, and
either `get_inputs()` or `get_input_groups()`.

## Requirements

1. Read `AGENTS.md` for global constraints and workflow.
2. Implement `ModelNew` in `src/{op_name}_triton_ascend_impl.py`.
3. Run the validation pipeline:
   ```bash
   bash tools/operator_pipeline.sh --op_name {op_name}
   ```
4. Read `metrics.json`, repair according to its `error` field, and rerun until
   `"success": true`.
"""


_KERNELBENCH_INLINE_INSTRUCTION_TEMPLATE = """# KernelBench Operator Task

Operator: {op_name}
Target: {arch}
Create: `src/{op_name}_triton_ascend_impl.py`

First action: create the implementation file with complete executable Python.

Reference behavior is defined by this executable code:
```python
{task_code}
```

Implement `ModelNew` compatible with the reference `Model`. Use module-scope
`@triton.jit` kernels. Preserve dtype, shape, broadcasting, reductions, and
layout from `forward`, `get_inputs()` or `get_input_groups()`, and
`get_init_inputs()`.

Hints:
- Elementwise/broadcast: vectorized output tile.
- `diag(A) @ B`: `C[i, j] = A[i] * B[i, j]`.
- Matmul/linear: tiled `tl.dot`.
- Reduction: tile reduction, or partial plus final kernel.
- Layout/slice/transpose: exact indexed write.
- Launch kernels with tensor arguments, never `.data_ptr()`.

Run:
```bash
bash tools/operator_pipeline.sh --op_name {op_name}
```

After each run, read `metrics.json`. If failed, patch only the smallest failing
block and rerun. If `error_file` is present, read it before editing. If success,
save best files and finish:
```bash
cp -f src/{op_name}_triton_ascend_impl.py src/{op_name}_triton_ascend_impl_best.py
cp -f metrics.json metrics_best.json
```

Task context: {instruction}
"""


def _setup_npu_operator_workspace(task: dict[str, Any], trace_label: str) -> str:
    pwd = Path(__file__).parent
    _WORKSPACE_PKG = pwd / "workspace"
    workspace = tempfile.mkdtemp(prefix=f"trajectory-{trace_label}-", dir=pwd/"workspace_temp")
    op_name = task.get("op_name", "operator")
    arch = task.get("arch", "ascend910b1")
    instruction = task.get("instruction", "Implement a simple vector_add-style operator.")
    task_code = task.get("task_code", "")


    shutil.copytree(_WORKSPACE_PKG, workspace, dirs_exist_ok=True)
    os.utime(workspace, None)

    lock_src = pwd / "distributed_npu_lock.py"
    lock_dst = Path(workspace) / "agent_workdir" / "tools" / "scripts" / "distributed_npu_lock.py"
    if lock_src.exists():
        lock_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lock_src, lock_dst)

    # for name in _OPERATOR_SEED_NAMES:
    #     src = os.path.join(_WORKSPACE_PKG, name)
    #     dst = os.path.join(workspace, name)
    #     if not os.path.exists(src):
    #         continue
    #     if os.path.isdir(src):
    #         shutil.copytree(src, dst)
    #     else:
    #         shutil.copy2(src, dst)
    # _chmod_executable_scripts(os.path.join(workspace, "tools"))

    if _OPENHANDS_MOCK_PIPELINE:
        mock_path = os.path.join(workspace, "agent_workdir", "tools", "operator_pipeline.sh")
        with open(mock_path, "w") as f:
            f.write(_MOCK_PIPELINE_SCRIPT)
        os.chmod(mock_path, 0o755)

    with open(os.path.join(workspace, "agent_workdir", "INSTRUCTIONS.md"), "w") as f:
        f.write(
            _KERNELBENCH_INLINE_INSTRUCTION_TEMPLATE.format(
                op_name=op_name,
                arch=arch,
                instruction=instruction,
                task_code=task_code,
            )
        )

    if task_code:
        src_dir = os.path.join(workspace, "agent_workdir", "src")
        os.makedirs(src_dir, exist_ok=True)
        with open(os.path.join(src_dir, f"{op_name}.py"), "w") as f:
            f.write(task_code)
    # breakpoint()
    return workspace


# ---------------------------------------------------------------------------
# Metrics & reward
# ---------------------------------------------------------------------------

def _has_metrics(path: str) -> dict[str, Any] | None:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            dbg("[openhands-npu] bad metrics.json: %s, %s", exc, path) # TODO
    return None


def _reward_from_metrics(perf: dict[str, Any]) -> float:
    """Reward from metrics.json with partial credit for stable RL signal."""
    reward = 0.0
    ast_reward = float(os.environ.get("OPENHANDS_REWARD_AST_OK", "0.10"))
    correctness_reward = float(os.environ.get("OPENHANDS_REWARD_CORRECTNESS_OK", "0.55"))
    speedup_reward = float(os.environ.get("OPENHANDS_REWARD_SPEEDUP_MAX", "0.40"))

    ast_ok = bool(perf.get("ast_check_ok", False))
    corr_ok = bool(perf.get("correctness_ok", False))
    success = bool(perf.get("success", False))

    if ast_ok:
        reward += ast_reward
    if corr_ok:
        reward += correctness_reward

    if success and corr_ok:
        perf_data = perf.get("perf_data") or {}
        try:
            speedup = float(perf_data.get("speedup_vs_torch", 0.0) or 0.0)
        except (TypeError, ValueError):
            speedup = 0.0

        target = float(os.environ.get("OPENHANDS_REWARD_TARGET_SPEEDUP", "2.0"))
        target = max(target, 1e-6)
        reward += speedup_reward * max(0.0, min(speedup / target, 1.0))

    return round(max(0.0, min(reward, 1.0)), 4)


def _impl_newer_than_metrics(impl_file: str, metrics_file: str) -> bool:
    try:
        return os.path.getmtime(impl_file) > os.path.getmtime(metrics_file) + 1e-6
    except OSError:
        return False


def _snapshot_success_as_best(workspace_dir: str, op_name: str, perf: dict[str, Any]) -> None:
    """Persist the validated implementation when the agent forgets to copy best files."""
    if not (bool(perf.get("success", False)) and bool(perf.get("correctness_ok", False))):
        return

    impl_file = os.path.join(workspace_dir, "src", f"{op_name}_triton_ascend_impl.py")
    metrics_file = os.path.join(workspace_dir, "metrics.json")
    if _impl_newer_than_metrics(impl_file, metrics_file):
        return

    best_impl = os.path.join(workspace_dir, "src", f"{op_name}_triton_ascend_impl_best.py")
    best_metrics = os.path.join(workspace_dir, "metrics_best.json")

    current_reward = _reward_from_metrics(perf)
    best_reward = -1.0
    if os.path.exists(best_metrics):
        try:
            with open(best_metrics) as f:
                best_reward = _reward_from_metrics(json.load(f))
        except (json.JSONDecodeError, OSError):
            best_reward = -1.0

    if current_reward >= best_reward:
        try:
            shutil.copy2(impl_file, best_impl)
            shutil.copy2(metrics_file, best_metrics)
            logger.info("[openhands-npu] snapshotted successful impl as best")
        except OSError:
            logger.debug("[openhands-npu] failed to snapshot best files", exc_info=True)


def _npu_operator_reward(task: dict[str, Any], workspace_dir: str, output: str) -> float:
    del output
    op_name = task.get("op_name", "operator")

    impl_file = os.path.join(workspace_dir, "src", f"{op_name}_triton_ascend_impl.py")
    if not os.path.exists(impl_file):
        logger.info("[openhands-npu] no impl file %s -> reward=0.0", impl_file)
        return 0.0
    # TODO: consider rejecting empty/stub-only impl files (os.path.getsize check)

    basic_path = os.path.join(workspace_dir, "metrics.json")
    perf = _has_metrics(basic_path)
    if not perf:
        logger.info("[openhands-npu] impl present but no metrics -> reward=0.05")
        return 0.05

    if _impl_newer_than_metrics(impl_file, basic_path):
        logger.info(
            "[openhands-npu] impl was edited after metrics.json; treating metrics as stale -> reward=0.05"
        )
        return 0.05

    _snapshot_success_as_best(workspace_dir, op_name, perf)

    reward = _reward_from_metrics(perf)

    best_path = os.path.join(workspace_dir, "metrics_best.json")
    if os.path.exists(best_path):
        try:
            with open(best_path) as f:
                best = json.load(f)
            best_reward = _reward_from_metrics(best)
            if best_reward > reward:
                logger.info(
                    "[openhands-npu] best version has higher reward: "
                    "current=%.3f best=%.3f -> using best",
                    reward, best_reward,
                )
                reward = best_reward
        except (json.JSONDecodeError, OSError):
            pass

    logger.info("[openhands-npu] final reward=%.3f", reward)
    return reward


def _archive_npu_artifacts(
    workspace_dir: str,
    task: dict[str, Any],
    trace_label: str,
    reward: float,
) -> None:
    """Copy key rollout artifacts to a persistent directory before cleanup.

    Controlled by env OPENHANDS_ARTIFACT_DIR. No-op if unset/empty.
    Never raises — archival failure must not affect training.
    """
    metric_dir = workspace_dir + "/agent_workdir"
    if not _ARTIFACT_DIR:
        return
    if not os.path.isdir(metric_dir):
        return

    op_name = task.get("op_name", "operator")
    ts = time.strftime("%Y%m%d_%H%M%S")
    # short_id = uuid.uuid4().hex[:6]
    dest = os.path.join(_ARTIFACT_DIR, f"{workspace_dir.split('/')[-1]}_{trace_label}_{op_name}_{ts}")

    try:
        os.makedirs(dest, exist_ok=True)

        candidates = [
            (os.path.join(metric_dir, "conversation.log"), "conversation.log"),
            (os.path.join(metric_dir, "conversation_tail.log"), "conversation_tail.log"),
            (os.path.join(metric_dir, "conversation_result.json"), "conversation_result.json"),
            (os.path.join(metric_dir, "src", f"{op_name}_triton_ascend_impl.py"),
             f"{op_name}_triton_ascend_impl.py"),
            (os.path.join(metric_dir, "src", f"{op_name}_triton_ascend_impl_best.py"),
             f"{op_name}_triton_ascend_impl_best.py"),
            (os.path.join(metric_dir, "metrics.json"), "metrics.json"),
            (os.path.join(metric_dir, "metrics_best.json"), "metrics_best.json"),
            (os.path.join(metric_dir, "INSTRUCTIONS.md"), "INSTRUCTIONS.md"),
        ]

        copied = 0
        for src_path, dst_name in candidates:
            if os.path.isfile(src_path):
                shutil.copy2(src_path, os.path.join(dest, dst_name))
                copied += 1

        manifest = {
            "trace_label": trace_label,
            "op_name": op_name,
            "arch": task.get("arch", "ascend910b1"),
            "reward": reward,
            "timestamp": ts,
        }
        with open(os.path.join(dest, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        if copied == 0:
            shutil.rmtree(dest, ignore_errors=True)
            logger.debug("[openhands] No artifacts to archive for trace=%s", trace_label)
        else:
            logger.info("[openhands] Archived %d artifacts → %s (reward=%.3f)", copied, dest, reward)
    except Exception:
        logger.warning("[openhands] Failed to archive artifacts for trace=%s", trace_label, exc_info=True)



def _trace_label_from_routing_metadata(metadata: dict[str, Any]) -> str:
    # uids = metadata.get("session_uids") or []
    # if uids:
    #     return str(uids[-1])[:18]
    name = metadata.get("session_name")
    trace_label = str(name or "none").split('-')[-1].replace(':', '-')
    return trace_label


def _to_container_url(url: str) -> str:
    """Replace localhost/127.0.0.1 with host.docker.internal so the URL
    is reachable from inside an OpenHands Docker container."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if host in ("localhost", "127.0.0.1"):
        netloc = parsed.netloc.replace(host, _OPENHANDS_CONTAINER_HOST_ALIAS, 1)
        url = urlunparse(parsed._replace(netloc=netloc))
    return url


def _metadata_for_proxy_url(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep the proxy URL slug small.

    AgentSdkEngine stores the whole task in session metadata for local context.
    For proxy routing we only need the session identity; including task_code in
    the URL can make docker commands huge and leak the reference implementation.
    """
    if os.environ.get("OPENHANDS_FULL_PROXY_METADATA", "0") in ("1", "true", "True", "yes"):
        return metadata

    compact = {key: metadata[key] for key in ("session_name", "session_uids") if key in metadata}
    return compact or metadata


def _tar_directory_b64(path: str, arcname: str = ".") -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(path, arcname=arcname)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _manifest_directory(path: str, limit: int = 80) -> list[str]:
    root = Path(path)
    if not root.exists():
        return []
    items: list[str] = []
    for item in root.rglob("*"):
        rel = item.relative_to(root).as_posix()
        items.append(rel + ("/" if item.is_dir() else ""))
        if len(items) >= limit:
            items.append("...")
            break
    return items


def _extract_tar_b64_into(encoded: str, destination: str) -> None:
    raw = base64.b64decode(encoded.encode("ascii"))
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        dest = Path(destination).resolve()
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"unsafe tar member path: {member.name}")
        tar.extractall(destination)


def _run_remote_eval_worker(
    workspace: str,
    proxied_url: str,
    instruction: str,
    *,
    task: dict[str, Any] | None = None,
) -> int:
    if not _OPENHANDS_REMOTE_EVAL_URL:
        return _run_openhands_container(workspace, proxied_url, instruction, task=task)

    payload = {
        "workspace_tar_gz_b64": _tar_directory_b64(workspace, arcname="workspace"),
        "workspace_manifest": _manifest_directory(workspace),
        "proxied_url": proxied_url,
        "instruction": instruction,
        "task": task or {},
        "image": _OPENHANDS_IMAGE,
        "model_name": _MODEL_NAME,
        "max_iterations": _MAX_ITERATIONS,
        "container_timeout": _CONTAINER_TIMEOUT,
        "eval_device_ids": _OPENHANDS_EVAL_DEVICE_IDS,
        "eval_device_count": _OPENHANDS_EVAL_DEVICE_COUNT,
        "agent_mode": os.environ.get("OPENHANDS_AGENT_MODE", ""),
        "enable_agent_skills": os.environ.get("OPENHANDS_ENABLE_AGENT_SKILLS", "0"),
        "observer_api_url": os.environ.get(
            "OPENHANDS_REMOTE_OBSERVER_API_URL",
            os.environ.get("OBSERVER_API_URL", ""),
        ),
        "return_full_log": os.environ.get("OPENHANDS_REMOTE_RETURN_FULL_LOG", "0"),
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{_OPENHANDS_REMOTE_EVAL_URL}/run",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout = int(os.environ.get("OPENHANDS_REMOTE_EVAL_TIMEOUT", str(_CONTAINER_TIMEOUT + 300)))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            result = json.loads(body)
        except json.JSONDecodeError:
            raise RuntimeError(f"remote eval worker HTTP {exc.code}: {body[:1000]}") from exc

    if "agent_workdir_tar_gz_b64" in result:
        agent_dir = Path(workspace) / "agent_workdir"
        if agent_dir.exists():
            shutil.rmtree(agent_dir)
        agent_dir.mkdir(parents=True, exist_ok=True)
        _extract_tar_b64_into(result["agent_workdir_tar_gz_b64"], str(agent_dir.parent))

    result_for_log = {k: v for k, v in result.items() if k != "agent_workdir_tar_gz_b64"}
    try:
        (Path(workspace) / "agent_workdir").mkdir(parents=True, exist_ok=True)
        (Path(workspace) / "agent_workdir" / "remote_eval_result.json").write_text(
            json.dumps(result_for_log, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        logger.debug("[openhands-remote] failed to write remote_eval_result.json", exc_info=True)

    if result.get("worker_error"):
        logger.warning("[openhands-remote] worker_error=%s", result.get("worker_error"))

    exit_code = int(result.get("exit_code", -1))
    if exit_code != 0:
        logger.warning("[openhands-remote] exit_code=%s result=%s", exit_code, result_for_log)

    return exit_code


# ---------------------------------------------------------------------------
# OpenHands container launch
# ---------------------------------------------------------------------------

def _run_openhands_container(
    workspace: str,
    proxied_url: str,
    instruction: str,
    *,
    task: dict[str, Any] | None = None,
) -> str:
    """Start an OpenHands headless container, wait for completion, return logs.

    Args:
        workspace:    Host-side workspace directory (mounted into container).
        proxied_url:  LiteLLM proxy URL with embedded rllm metadata slug.
                      Passed as LLM_BASE_URL so rllm can track all LLM calls.
        instruction:  Task instruction (also written to INSTRUCTIONS.md).
    """
    # TODO(遗留): 禁止 agent 外网（如误 apt）目前仅靠 workspace/AGENTS.md 软约束。若需硬隔离，在此
    # 组装 docker cmd 时加入 --network（例如宿主机预先 docker network create --internal …），并验证
    # 仍能访问 host.docker.internal 上的 LiteLLM；勿用 network=none 除非 LLM 不依赖宿主机 HTTP。
    task = task or {}
    container_name = f"rllm-openhands-{uuid.uuid4().hex[:12]}"
    op_name = task.get("op_name", "operator")
    arch = task.get("arch", "ascend910b1")
    operator_backend = str(task.get("operator_backend", "triton"))

    path = Path("/tmp/shared_npu_lock")
    path.mkdir(mode=0o755, parents=True, exist_ok=True)

    # TODO(遗留): Ascend NPU 入容器所需的 docker --device / 额外 -v 挂载（如 /dev/davinci*、驱动相关路径）
    # 待按 CANN 与所用基础镜像文档确定，并与 OPENHANDS_ASCEND_VISIBLE_DEVICES 一起在真机验证。
    # 当前仅挂载 workspace、entrypoint 及注入 ASCEND_RT_VISIBLE_DEVICES（若设置）。
    cmd = [
        "docker", "run",
        # "--rm",  # TODO
        "-d",
        "--name", container_name,
        "--network", "host",
        "--ipc", "host",
        "--shm-size", "500g",
        "--privileged",
        "-e", f"LLM_BASE_URL={proxied_url}",
        "-e", "LLM_API_KEY=EMPTY",
        "-e", f"LLM_MODEL=openai/{_MODEL_NAME}",
        "-e", f"OPERATOR_BACKEND={operator_backend}",
        "-e", f"OPERATOR_ARCH={arch}",
        "-e", f"OPERATOR_NAME={op_name}",
        "-e", f"OPENHANDS_AGENT_MODE={os.environ.get('OPENHANDS_AGENT_MODE', '')}",
        "-e", f"OPENHANDS_ENABLE_AGENT_SKILLS={os.environ.get('OPENHANDS_ENABLE_AGENT_SKILLS', '0')}",
        "-e", f"OPENHANDS_TERMINAL_NO_CHANGE_TIMEOUT_SECONDS={os.environ.get('OPENHANDS_TERMINAL_NO_CHANGE_TIMEOUT_SECONDS', '300')}",
        "-e", "PYTHONUTF8=1",
        "-e", "PYTHONIOENCODING=utf-8",
        "-e", "LANG=C.UTF-8",
        "-e", "LC_ALL=C.UTF-8",
        
        "-e", "http_proxy=", 
        "-e", "https_proxy=",
        "-e", "no_proxy=host.docker.internal,127.0.0.1,localhost,172.17.0.1",
        "-e", "WORKSPACE_BASE=/opt/workspace/agent_workdir",
        "-e", f"OBSERVER_API_URL={os.environ.get('OPENHANDS_OBSERVER_API_URL', os.environ.get('OBSERVER_API_URL', ''))}",
        "-e", f"MAX_ITERATIONS={_MAX_ITERATIONS}",
        
        # 评估专用
        "-e", "EVAL_LOCK_DIR=/shared/device-locks",
        "-e", "EVAL_DEVICE_PREFIX=npu",
        "-e", f"EVAL_DEVICE_COUNT={_OPENHANDS_EVAL_DEVICE_COUNT}",
        "-e", f"EVAL_DEVICE_IDS={_OPENHANDS_EVAL_DEVICE_IDS}",
        "-e", "EVAL_ENV_NAME=ASCEND_RT_VISIBLE_DEVICES",
        "-e", "EVAL_RETRY_INTERVAL=1.0",
        "-e", "EVAL_TIMEOUT=None",
        "-e", "EVAL_VERBOSE=true",
        
        # "--device", "/dev/davinci0",
        # "--device", "/dev/davinci1",
        # "--device", "/dev/davinci2",
        # "--device", "/dev/davinci3",
        # "--device", "/dev/davinci4",
        # "--device", "/dev/davinci5",
        # "--device", "/dev/davinci6",
        # "--device", "/dev/davinci7",
        # "--device", "/dev/davinci_manager",
        # "--device", "/dev/hisi_hdc",
        # "--device", "/dev/devmm_svm",
        
        "-v", "/dev:/dev",
        "-v", "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
        "-v", "/usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro",
        "-v", "/usr/local/dcmi:/usr/local/dcmi:ro",
        "-v", "/usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro",
        "-v", "/etc/ascend_install.info:/etc/ascend_install.info:ro",
        "-v", "/usr/local/sbin:/usr/local/sbin:ro",
        "-v", "/etc/localtime:/etc/localtime:ro",
        "-v", "/etc/timezone:/etc/timezone:ro",
        "-v", "/mnt/pipeline-data:/mnt/pipeline-data",
        #"-v", "/opt/DPC:/opt/DPC",

        # os.environ["OBSERVER_API_URL"] = "http://127.0.0.1:18858"
        "-v", f"{workspace}:/opt/workspace",
        "-v", f"/tmp/shared_npu_lock:/shared/device-locks",   # 共享文件锁所在路径
        # "-v", f"/home/g00841271/rllm-071/examples/openhands_sdk/workspace_debug:/opt/workspace",
        "--entrypoint", f"/opt/workspace/entrypoint.py",
        "--add-host", "host.docker.internal:host-gateway",
    ]
    cmd.extend([_OPENHANDS_IMAGE,])
    
    import shlex
    print("[DEBUG] OPENHANDS CMD:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    dbg(" ".join(shlex.quote(c) for c in cmd))
    # breakpoint()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            #timeout=_CONTAINER_TIMEOUT, # TODO
        )
        output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        if result.returncode != 0:
            logger.warning(
                "[openhands] Container %s exited with code %d",
                container_name, result.returncode,
            )
            return output
    
        wait_result = subprocess.run(
            ["docker", "wait", container_name],
            capture_output=True,
            #timeout=_CONTAINER_TIMEOUT, # TODO
        )
        
        dbg(f"wait_result={wait_result}, {wait_result.stdout.decode().strip()}")
        try:
            exit_code = int(wait_result.stdout.decode().strip())
        except Exception as e: # TODO
            logger.error(f"exit_code error, {e}")
        
        # 获取并保存日志
        logs_result = subprocess.run(
            ["docker", "logs", container_name],
            capture_output=True,
        )

        logs = logs_result.stdout + logs_result.stderr
        with open(workspace+"/agent_workdir/conversation.log", "wb") as f:
            f.write(logs)

        with open(workspace+"/agent_workdir/conversation_result.json", "w") as f:
            json.dump({
                "container": container_name,
                "exit_code": exit_code,
                "timestamp": time.time(),
            }, f, indent=2)

        return exit_code
    
    except subprocess.TimeoutExpired:
        logger.error("[openhands] Container %s timed out", container_name)
        subprocess.run(["docker", "kill", container_name])
        return -1
    except Exception:
        logger.exception("[openhands] Failed to run container %s", container_name)
        return -1
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL)
        print(f"[CLEANUP] removed {container_name}")

# ---------------------------------------------------------------------------
# Rollout entry point
# ---------------------------------------------------------------------------

# Keys passed through AgentSdkEngine / session wrapper as kwargs alongside
# extra_info fields; they must not be treated as task payload.
_ROLLOUT_CONFIG_KEYS = frozenset({
    "config",
    "base_url",
    "session_uid",
    "is_validation",
})


def _rollout_task_and_config(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize call shapes.

    - AgentSdkEngine: ``partial(wrapped, metadata, **extra_info)`` → session
      calls ``rollout(instruction=..., scenario=..., ...)`` (kwargs only).
    - Legacy / manual: ``rollout(task_dict, config_dict)``.
    """
    config: dict[str, Any] = {}
    raw_cfg = kwargs.get("config")
    if isinstance(raw_cfg, dict):
        config = dict(raw_cfg)
    if "base_url" in kwargs:
        config["base_url"] = kwargs["base_url"]
    if "session_uid" in kwargs:
        config["session_uid"] = kwargs["session_uid"]

    if len(args) >= 1 and isinstance(args[0], dict):
        task = dict(args[0])
        if len(args) >= 2 and isinstance(args[1], dict):
            config = {**config, **args[1]}
        return task, config

    task = {k: v for k, v in kwargs.items() if k not in _ROLLOUT_CONFIG_KEYS}
    return task, config


def rollout(*args: Any, **kwargs: Any) -> list[dict]:

    """rllm rollout entry point for OpenHands.

    Generates a session-specific metadata slug, builds a proxied LiteLLM URL,
    and launches OpenHands in an isolated Docker container. All LLM calls made
    by OpenHands will carry the metadata slug so the rllm proxy can attribute
    them to this training session.

    Args:
        task:   Task dict (instruction, repo_url, test_file, …).
        config: Config dict from rllm. Expected keys:
                  base_url  — raw LiteLLM proxy URL, e.g.
                              "http://127.0.0.1:4000/v1"

    Returns:
        List with one trajectory dict: {name, steps, reward}.
    """
    task, config = _rollout_task_and_config(args, kwargs)
    # Raw proxy URL — rllm passes this before any slug is applied
    base_url_port = os.getenv("OPENHANDS_BASE_URL_PORT", 4000)
    proxy_url = config.get("base_url", f"http://127.0.0.1:{base_url_port}/v1")

    # Generate a unique session identifier for this rollout.
    # Encoding it in the URL lets the rllm proxy associate every OpenHands
    # LLM call with this specific training episode.
    # session_uid = str(uuid.uuid4())
    # metadata: dict[str, Any] = {
    #     "session_uids": [session_uid],
    #     "session_name": f"openhands-{session_uid[:8]}",
    # }
    
    
    metadata = assemble_routing_metadata()
    url_metadata = _metadata_for_proxy_url(metadata)
    trace_label = _trace_label_from_routing_metadata(url_metadata)

    logger.info(
        "[openhands] proxy slug: session_uids=%r session_name=%r",
        url_metadata.get("session_uids"),
        url_metadata.get("session_name"),
    )

    # Build the proxied URL with embedded metadata slug.
    # _build_proxied_base_url inlines the same logic as rllm.sdk.proxy.
    # proxied_url = _build_proxied_base_url(proxy_url, metadata)
    proxied_url = build_proxied_base_url(proxy_url, url_metadata)

    # Rewrite localhost/127.0.0.1 → host.docker.internal so the URL is
    # reachable from inside the OpenHands Docker container.
    proxied_url = _to_container_url(proxied_url)

    workspace = _setup_npu_operator_workspace(task, trace_label)
    instruction = task.get("instruction", "")
    reward = 0.0

    try:
        output = _run_remote_eval_worker(
            workspace, proxied_url, instruction, task=task
        )
        metrics_dir = workspace + "/agent_workdir"
        reward = _npu_operator_reward(task, metrics_dir, output)
        logger.info(
            "[openhands] trace_label=%s reward=%.2f instruction=%s",
            trace_label, reward, instruction[:80],
        )
    except Exception:
        logger.exception("[openhands] Rollout failed (trace_label=%s)", trace_label)
    finally:
        _archive_npu_artifacts(workspace, task, trace_label, reward)
        if os.environ.get("OPENHANDS_KEEP_WORKSPACE", "0") not in ("1", "true", "True", "yes"):
            shutil.rmtree(workspace, ignore_errors=True)
    return reward

