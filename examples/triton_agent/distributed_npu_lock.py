#!/usr/bin/env python3
# -----------------------------
# 分布式NPU锁，用于多个设备抢占NPU的情况
# 通过文件系统实现设备锁
# 注意：代码是一个临时版本，只能适用于单台机器内多个程序抢占情况
# 对于多个容器使用，需要所有容器共享一个存放文件锁的目录
# -----------------------------

import os
import sys
import time
import json
import fcntl
import socket
import signal
import argparse
import subprocess
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class DeviceLease:
    fd: Optional[int] = None
    device_id: int = -1
    path: str = ""

    def valid(self) -> bool:
        return self.fd is not None and self.fd >= 0

    def release(self) -> None:
        if self.fd is not None and self.fd >= 0:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None
                self.device_id = -1
                self.path = ""


class FairDevicePool:
    """
    基于共享文件系统的公平设备锁。

    公平策略：
    1. 先通过 ticket 文件分配递增 ticket
    2. 使用 ticket % device_count 作为扫描起点
    3. 轮转扫描所有设备，避免永远偏向低编号设备
    """

    def __init__(
        self,
        lock_dir: str,
        prefix: str,
        device_count: int,
        device_ids: Optional[List[int]] = None,
    ):
        if device_ids is not None:
            if not device_ids:
                raise ValueError("device_ids must not be empty")
            self.device_ids = list(device_ids)
            device_count = len(self.device_ids)
        else:
            if device_count <= 0:
                raise ValueError("device_count must be > 0")
            self.device_ids = list(range(device_count))

        self.lock_dir = lock_dir
        self.prefix = prefix
        self.device_count = device_count

        os.makedirs(self.lock_dir, exist_ok=True)

    def _device_lock_path(self, device_id: int) -> str:
        return os.path.join(self.lock_dir, f"{self.prefix}{device_id}.lock")

    def _ticket_path(self) -> str:
        return os.path.join(self.lock_dir, f"{self.prefix}.ticket")

    def _now(self) -> float:
        return time.time()

    def _next_ticket(self) -> int:
        path = self._ticket_path()
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)

            os.lseek(fd, 0, os.SEEK_SET)
            raw = os.read(fd, 128).decode("utf-8", errors="ignore").strip()
            ticket = int(raw) if raw else 0
            next_ticket = ticket + 1

            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, str(next_ticket).encode("utf-8"))
            os.fsync(fd)

            return ticket
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _fair_order(self, ticket: int) -> List[int]:
        start = ticket % self.device_count
        return [self.device_ids[(start + i) % self.device_count] for i in range(self.device_count)]

    def _try_acquire_device(self, device_id: int, meta: dict) -> Optional[DeviceLease]:
        path = self._device_lock_path(device_id)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        except Exception:
            os.close(fd)
            raise

        payload = (json.dumps(meta, ensure_ascii=False) + "\n").encode("utf-8")
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, payload)
        os.fsync(fd)

        return DeviceLease(fd=fd, device_id=device_id, path=path)

    def acquire_any_fair(
        self,
        wait: bool = True,
        retry_interval: float = 1.0,
        timeout: Optional[float] = None,
        extra_meta: Optional[dict] = None,
        jitter: float = 0.2,
    ) -> Optional[DeviceLease]:
        start_time = self._now()
        hostname = socket.gethostname()
        pid = os.getpid()

        while True:
            ticket = self._next_ticket()
            order = self._fair_order(ticket)

            base_meta = {
                "pid": pid,
                "hostname": hostname,
                "prefix": self.prefix,
                "ticket": ticket,
                "order": order,
                "acquire_time": self._now(),
            }
            if extra_meta:
                base_meta.update(extra_meta)

            for device_id in order:
                meta = dict(base_meta)
                meta["device_id"] = device_id

                lease = self._try_acquire_device(device_id, meta)
                if lease is not None:
                    return lease

            if not wait:
                return None

            if timeout is not None and (self._now() - start_time) >= timeout:
                return None

            sleep_time = retry_interval + ((pid % 997) / 997.0) * jitter
            time.sleep(sleep_time)


def forward_signal(child_proc: subprocess.Popen, sig):
    try:
        if child_proc.poll() is None:
            child_proc.send_signal(sig)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Fair GPU/NPU wrapper based on shared filesystem locks."
    )
    parser.add_argument(
        "--device-ids",
        default=os.environ.get("EVAL_DEVICE_IDS", ""),
        help="Comma-separated physical device ids to allocate from, e.g. 8,9,10,11",
    )
    parser.add_argument("--lock-dir", default=os.environ.get("EVAL_LOCK_DIR"), help="共享锁目录，例如 /shared/device-locks")
    parser.add_argument("--device-prefix", default="gpu", help="设备前缀，如 gpu / npu")
    parser.add_argument("--device-count", type=int, required=True, help="设备数量")
    parser.add_argument("--env-name", default="CUDA_VISIBLE_DEVICES", help="设置给子进程的环境变量名")
    parser.add_argument("--retry-interval", type=float, default=1.0, help="重试间隔秒")
    parser.add_argument("--timeout", type=float, default=None, help="获取设备超时时间，秒")
    parser.add_argument("--verbose", action="store_true", help="打印调试日志")

    # 用 REMAINDER 接收要执行的目标命令
    parser.add_argument("command", nargs=argparse.REMAINDER, help="要执行的命令，例如 -- python train.py")

    args = parser.parse_args()

    if not args.command:
        print("ERROR: missing command. Example:", file=sys.stderr)
        print(
            "  python gpu_wrapper.py --lock-dir /shared/locks --device-count 8 -- python train.py --config xx",
            file=sys.stderr,
        )
        sys.exit(2)

    command = args.command
    if command and command[0] == "--":
        command = command[1:]

    if not command:
        print("ERROR: empty command after --", file=sys.stderr)
        sys.exit(2)

    device_ids = None
    if args.device_ids:
        device_ids = [int(x.strip()) for x in args.device_ids.split(",") if x.strip()]

    pool = FairDevicePool(
        lock_dir=args.lock_dir,
        prefix=args.device_prefix,
        device_count=args.device_count,
        device_ids=device_ids,
    )

    extra_meta = {
        "wrapper_pid": os.getpid(),
        "command": command,
    }

    if args.verbose:
        print(
            f"[wrapper] acquiring {args.device_prefix} from {args.lock_dir}, "
            f"count={args.device_count}, env={args.env_name}",
            flush=True,
        )

    lease = pool.acquire_any_fair(
        wait=True,
        retry_interval=args.retry_interval,
        timeout=args.timeout,
        extra_meta=extra_meta,
    )

    if lease is None:
        print("[wrapper] failed to acquire device within timeout", file=sys.stderr, flush=True)
        sys.exit(1)

    if args.verbose:
        print(
            f"[wrapper] acquired {args.device_prefix}{lease.device_id}, "
            f"lock={lease.path}",
            flush=True,
        )

    env = os.environ.copy()
    env[args.env_name] = str(lease.device_id)

    # 你也可以额外暴露一个物理设备编号环境变量
    env["ALLOCATED_DEVICE_ID"] = str(lease.device_id)
    env["ALLOCATED_DEVICE_PREFIX"] = args.device_prefix

    child_proc = subprocess.Popen(command, env=env)

    def _handle_signal(signum, frame):
        if args.verbose:
            print(f"[wrapper] received signal {signum}, forwarding to child...", flush=True)
        forward_signal(child_proc, signum)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        returncode = child_proc.wait()
    finally:
        if args.verbose:
            print(
                f"[wrapper] child exited, releasing {args.device_prefix}{lease.device_id}",
                flush=True,
            )
        lease.release()

    sys.exit(returncode)


if __name__ == "__main__":
    main()
