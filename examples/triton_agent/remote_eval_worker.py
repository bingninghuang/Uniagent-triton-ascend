#!/usr/bin/env python3
"""Remote OpenHands evaluator worker.

Run this on a separate Ascend/NPU evaluation machine. The training machine
posts a packed workspace to /run; this worker unpacks it, starts the OpenHands
container locally, then returns the resulting agent_workdir archive.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _extract_tar_b64(encoded: str, destination: str) -> None:
    raw = base64.b64decode(encoded.encode("ascii"))
    dest = Path(destination).resolve()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"unsafe tar member path: {member.name}")
        tar.extractall(dest)


def _tar_directory_b64(path: str, arcname: str) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(path, arcname=arcname)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _add_text_file(tar: tarfile.TarFile, arcname: str, text: str) -> None:
    data = text.encode("utf-8", errors="replace")
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mtime = time.time()
    tar.addfile(info, io.BytesIO(data))


def _truthy(value: Any) -> bool:
    return str(value).strip() in ("1", "true", "True", "yes", "on")


def _tar_agent_artifacts_b64(agent_dir: str, *, include_full_log: bool | None = None) -> str:
    """Return a compact agent_workdir archive for the trainer.

    The full OpenHands workspace can contain tools, references, caches, and a
    very large conversation.log. The trainer only needs metrics and generated
    implementation files to compute reward and archive useful artifacts.
    """
    root = Path(agent_dir)
    if include_full_log is None:
        include_full_log = _truthy(os.environ.get("OPENHANDS_REMOTE_RETURN_FULL_LOG", "0"))
    log_tail_chars = int(os.environ.get("OPENHANDS_REMOTE_LOG_TAIL_CHARS", "12000"))

    selected: list[Path] = []
    for name in (
        "INSTRUCTIONS.md",
        "conversation_result.json",
        "metrics.json",
        "metrics_best.json",
        "metrics_error.log",
        "profiling_results.json",
        "remote_docker_start.log",
    ):
        path = root / name
        if path.is_file():
            selected.append(path)

    src_dir = root / "src"
    if src_dir.is_dir():
        for path in src_dir.iterdir():
            if path.is_file() and path.suffix in {".py", ".json", ".log"}:
                selected.append(path)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in selected:
            tar.add(path, arcname=(Path("agent_workdir") / path.relative_to(root)).as_posix())

        log_path = root / "conversation.log"
        if log_path.is_file():
            if include_full_log:
                tar.add(log_path, arcname="agent_workdir/conversation.log")
            else:
                _add_text_file(
                    tar,
                    "agent_workdir/conversation_tail.log",
                    _read_tail(log_path, max_chars=log_tail_chars),
                )

    return base64.b64encode(buf.getvalue()).decode("ascii")


def _manifest_directory(path: str, limit: int = 120) -> list[str]:
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


def _read_tail(path: Path, max_chars: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def _run_container(workspace: str, request: dict[str, Any]) -> int:
    task = request.get("task") or {}
    container_name = f"rllm-openhands-eval-{uuid.uuid4().hex[:12]}"
    image = request.get("image") or os.environ.get("OPENHANDS_IMAGE", "openhands-triton-env:v1")
    model_name = request.get("model_name") or os.environ.get("OPENHANDS_MODEL_NAME", "openhands-model")
    max_iterations = int(request.get("max_iterations") or os.environ.get("OPENHANDS_MAX_ITERATIONS", "30"))
    op_name = task.get("op_name", "operator")
    arch = task.get("arch", "ascend910b1")
    operator_backend = str(task.get("operator_backend", "triton"))
    proxied_url = request["proxied_url"]
    agent_mode = str(request.get("agent_mode") or os.environ.get("OPENHANDS_AGENT_MODE", "")).strip()
    enable_agent_skills = str(
        request.get("enable_agent_skills") or os.environ.get("OPENHANDS_ENABLE_AGENT_SKILLS", "0")
    ).strip()
    observer_api_url = (
        request.get("observer_api_url")
        or os.environ.get("OPENHANDS_OBSERVER_API_URL")
        or ""
    )

    eval_device_ids = str(request.get("eval_device_ids") or os.environ.get("OPENHANDS_EVAL_DEVICE_IDS", "")).strip()
    eval_device_count = str(request.get("eval_device_count") or os.environ.get("OPENHANDS_EVAL_DEVICE_COUNT", "")).strip()
    if not eval_device_count:
        eval_device_count = str(len([x for x in eval_device_ids.split(",") if x.strip()]) if eval_device_ids else 1)

    lock_dir = os.environ.get("OPENHANDS_EVAL_LOCK_DIR", "/tmp/shared_npu_lock")
    Path(lock_dir).mkdir(mode=0o755, parents=True, exist_ok=True)

    cmd = [
        "docker", "run",
        "-d",
        "--name", container_name,
        "--network", "host",
        "--ipc", "host",
        "--shm-size", os.environ.get("OPENHANDS_DOCKER_SHM_SIZE", "500g"),
        "--privileged",
        "-e", f"LLM_BASE_URL={proxied_url}",
        "-e", "LLM_API_KEY=EMPTY",
        "-e", f"LLM_MODEL=openai/{model_name}",
        "-e", f"OPERATOR_BACKEND={operator_backend}",
        "-e", f"OPERATOR_ARCH={arch}",
        "-e", f"OPERATOR_NAME={op_name}",
        "-e", f"OPENHANDS_AGENT_MODE={agent_mode}",
        "-e", f"OPENHANDS_ENABLE_AGENT_SKILLS={enable_agent_skills}",
        "-e", f"OPENHANDS_TERMINAL_NO_CHANGE_TIMEOUT_SECONDS={os.environ.get('OPENHANDS_TERMINAL_NO_CHANGE_TIMEOUT_SECONDS', '300')}",
        "-e", "PYTHONUTF8=1",
        "-e", "PYTHONIOENCODING=utf-8",
        "-e", "LANG=C.UTF-8",
        "-e", "LC_ALL=C.UTF-8",
        "-e", "http_proxy=",
        "-e", "https_proxy=",
        "-e", "no_proxy=127.0.0.1,localhost,172.17.0.1",
        "-e", "WORKSPACE_BASE=/opt/workspace/agent_workdir",
        "-e", f"OBSERVER_API_URL={observer_api_url}",
        "-e", f"MAX_ITERATIONS={max_iterations}",
        "-e", "EVAL_LOCK_DIR=/shared/device-locks",
        "-e", "EVAL_DEVICE_PREFIX=npu",
        "-e", f"EVAL_DEVICE_COUNT={eval_device_count}",
        "-e", f"EVAL_DEVICE_IDS={eval_device_ids}",
        "-e", "EVAL_ENV_NAME=ASCEND_RT_VISIBLE_DEVICES",
        "-e", "EVAL_RETRY_INTERVAL=1.0",
        "-e", "EVAL_TIMEOUT=None",
        "-e", "EVAL_VERBOSE=true",
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
        "-v", f"{workspace}:/opt/workspace",
        "-v", f"{lock_dir}:/shared/device-locks",
        "--entrypoint", "/opt/workspace/entrypoint.py",
        image,
    ]

    import shlex 
    print("[DEBUG] OPENHANDS CMD:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    
    try:
        start = subprocess.run(cmd, capture_output=True)
        if start.returncode != 0:
            agent_dir = Path(workspace) / "agent_workdir"
            agent_dir.mkdir(parents=True, exist_ok=True)
            (agent_dir / "remote_docker_start.log").write_bytes(start.stdout + start.stderr)
            (agent_dir / "conversation_result.json").write_text(
                json.dumps(
                    {
                        "container": container_name,
                        "exit_code": int(start.returncode),
                        "phase": "docker_run",
                        "timestamp": time.time(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return int(start.returncode)

        wait = subprocess.run(
            ["docker", "wait", container_name],
            capture_output=True,
            timeout=int(request.get("container_timeout") or os.environ.get("OPENHANDS_CONTAINER_TIMEOUT", "1800")),
        )
        try:
            exit_code = int(wait.stdout.decode().strip())
        except Exception:
            exit_code = -1

        logs = subprocess.run(["docker", "logs", container_name], capture_output=True)
        agent_dir = Path(workspace) / "agent_workdir"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "conversation.log").write_bytes(logs.stdout + logs.stderr)
        (agent_dir / "conversation_result.json").write_text(
            json.dumps({"container": container_name, "exit_code": exit_code, "timestamp": time.time()}, indent=2),
            encoding="utf-8",
        )
        return exit_code
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Handler(BaseHTTPRequestHandler):
    server_version = "RemoteOpenHandsEval/0.1"

    def _json_response(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json_response(200, {"ok": True})
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/run":
            self._json_response(404, {"error": "not found"})
            return

        tmp_root = tempfile.mkdtemp(prefix="openhands-remote-eval-", dir=self.server.work_dir)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            _extract_tar_b64(request["workspace_tar_gz_b64"], tmp_root)
            workspace_path = Path(tmp_root) / "workspace"
            workspace = str(workspace_path)
            unpack_manifest = _manifest_directory(workspace)
            missing = [
                rel for rel in ("entrypoint.py", "agent_workdir/INSTRUCTIONS.md")
                if not (workspace_path / rel).exists()
            ]
            if missing:
                agent_dir = workspace_path / "agent_workdir"
                agent_dir.mkdir(parents=True, exist_ok=True)
                (agent_dir / "remote_unpack_error.json").write_text(
                    json.dumps(
                        {
                            "tmp_root": tmp_root,
                            "missing": missing,
                            "unpack_manifest": unpack_manifest,
                            "request_workspace_manifest": request.get("workspace_manifest", []),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                payload = {
                    "exit_code": -1,
                    "worker_error": f"remote workspace unpack missing required files: {missing}",
                    "tmp_root": tmp_root,
                    "unpack_manifest": unpack_manifest,
                    "request_workspace_manifest": request.get("workspace_manifest", []),
                    "agent_workdir_tar_gz_b64": _tar_agent_artifacts_b64(
                        str(agent_dir),
                        include_full_log=_truthy(request.get("return_full_log")),
                    ),
                }
                self._json_response(200, payload)
                return

            exit_code = _run_container(workspace, request)
            agent_dir = Path(workspace) / "agent_workdir"
            payload: dict[str, Any] = {
                "exit_code": exit_code,
                "tmp_root": tmp_root,
                "unpack_manifest": unpack_manifest,
            }
            if agent_dir.exists():
                payload["agent_workdir_tar_gz_b64"] = _tar_agent_artifacts_b64(
                    str(agent_dir),
                    include_full_log=_truthy(request.get("return_full_log")),
                )
                payload["conversation_log_tail"] = _read_tail(agent_dir / "conversation.log")
                payload["remote_docker_start_log_tail"] = _read_tail(agent_dir / "remote_docker_start.log")
            self._json_response(200, payload)
        except Exception as exc:
            payload = {"exit_code": -1, "worker_error": repr(exc)}
            agent_dir = Path(tmp_root) / "workspace" / "agent_workdir"
            if agent_dir.exists():
                try:
                    payload["agent_workdir_tar_gz_b64"] = _tar_agent_artifacts_b64(str(agent_dir))
                except Exception:
                    pass
            self._json_response(500, payload)
        finally:
            if os.environ.get("OPENHANDS_REMOTE_KEEP_WORKDIR", "0") not in ("1", "true", "True", "yes"):
                shutil.rmtree(tmp_root, ignore_errors=True)


def main() -> None:
    default_work_dir = str(Path(__file__).resolve().parent / "openhands-remote-eval")
    parser = argparse.ArgumentParser(description="Run remote OpenHands evaluation worker.")
    parser.add_argument("--host", default=os.environ.get("OPENHANDS_REMOTE_EVAL_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("OPENHANDS_REMOTE_EVAL_PORT", "18880")))
    parser.add_argument("--work-dir", default=os.environ.get("OPENHANDS_REMOTE_EVAL_WORKDIR", default_work_dir))
    args = parser.parse_args()

    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.work_dir = args.work_dir  # type: ignore[attr-defined]
    print(f"[remote-eval] listening on http://{args.host}:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
