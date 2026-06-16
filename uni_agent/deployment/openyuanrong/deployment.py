import asyncio
import os
import time
import uuid
from typing import Any, Self

from swerex import PACKAGE_NAME, REMOTE_EXECUTABLE_NAME
from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.hooks.abstract import CombinedDeploymentHook, DeploymentHook
from swerex.exceptions import DeploymentNotStartedError
from swerex.runtime.abstract import CreateBashSessionRequest, IsAliveResponse
from swerex.utils.wait import _wait_until_alive

from uni_agent.async_logging import get_logger
from uni_agent.deployment.config import YRDeploymentConfig
from uni_agent.deployment.remote_runtime import RemoteRuntime, RemoteRuntimeConfig

__all__ = ["YRDeployment"]


def _configure_env() -> None:
    """Map OPENYUANRONG_* env vars to names required by akernel-sdk before use."""
    server = os.getenv("OPENYUANRONG_SERVER_ADDRESS")
    token = os.getenv("OPENYUANRONG_TOKEN")
    if not server:
        raise ValueError("OPENYUANRONG_SERVER_ADDRESS environment variable must be set")
    if not token:
        raise ValueError("OPENYUANRONG_TOKEN environment variable must be set")
    os.environ["AKERNEL_SERVER_ADDRESS"] = server
    os.environ["AKERNEL_TOKEN"] = token


def _create_sandbox(config: YRDeploymentConfig) -> Any:
    """Create sandbox with port_forwardings=[port] (Port Forwarding, sandbox-api)."""
    _configure_env()
    from akernel_sdk import Sandbox

    kwargs: dict[str, Any] = {
        "cpu": config.cpu,
        "memory": config.memory,
        "cpu_limit": config.cpu_limit,
        "mem_limit": config.mem_limit,
        "idle_timeout": config.idle_timeout,
    }
    if config.image is not None:
        kwargs["image"] = config.image
    if config.env is not None:
        kwargs["env"] = config.env
    if config.name is not None:
        kwargs["name"] = config.name
    if config.cwd is not None:
        kwargs["cwd"] = config.cwd
    kwargs.update(config.sandbox_kwargs)
    kwargs["port_forwardings"] = [config.port]

    return Sandbox(**kwargs)


def _start_swerex_via_port_forwarding(sandbox: Any, *, command: str, port: int, internal: bool) -> tuple[Any, str]:
    """Port Forwarding flow from sandbox-api.md:

    1. Sandbox created with port_forwardings=[port]
    2. commands.run(server_cmd, background=True)  — start swerex on the forwarded port
    3. get_port_url(port) — gateway URL for RemoteRuntime
    """
    handle = sandbox.commands.run(command, background=True)
    url = sandbox.get_port_url(port, internal=internal).replace("http://", "https://")
    return handle, url


def _kill_sandbox(sandbox: Any) -> None:
    sandbox.kill()


class YRDeployment(AbstractDeployment):
    """YR (AKernel) deployment: sandbox + Port Forwarding + swe-rex + RemoteRuntime."""

    def __init__(self, run_id: str, **kwargs: Any):
        self.run_id = run_id
        self._config = YRDeploymentConfig(**kwargs)
        self._runtime: RemoteRuntime | None = None
        self._sandbox: Any | None = None
        self._command_handle: Any | None = None
        self._runtime_url: str | None = None
        self._port = self._config.port
        self.logger = get_logger("deployment", run_id)
        self._hooks = CombinedDeploymentHook()
        self._stopped = False

    def add_hook(self, hook: DeploymentHook):
        self._hooks.add_hook(hook)

    @classmethod
    def from_config(cls, config: YRDeploymentConfig, run_id: str | None = None) -> Self:
        if run_id is None:
            run_id = str(uuid.uuid4())
        return cls(run_id=run_id, **config.model_dump())

    def _get_token(self) -> str:
        return str(uuid.uuid4())

    def _swerex_start_command(self, token: str) -> str:
        """Command that binds swerex.server to the port-forwarded port inside the sandbox."""
        if self._config.command:
            return self._config.command.format(token=token, port=self._port)
        rex_args = f"--host 0.0.0.0 --port {self._port} --auth-token {token}"
        prepare_cmd = os.getenv("OPENYUANRONG_ENV_PREPARE_CMD")
        if prepare_cmd:
            return (
                f"{prepare_cmd} && "
                f"({REMOTE_EXECUTABLE_NAME} {rex_args} || pipx run {PACKAGE_NAME} {rex_args})"
            )
        return f"{REMOTE_EXECUTABLE_NAME} {rex_args} || pipx run {PACKAGE_NAME} {rex_args}"

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        if self._runtime is None or self._sandbox is None:
            raise DeploymentNotStartedError()
        loop = asyncio.get_running_loop()
        running = await loop.run_in_executor(None, self._sandbox.is_running)
        if not running:
            msg = f"YR sandbox {self._sandbox.sandbox_id} is not running"
            return IsAliveResponse(is_alive=False, message=msg)
        return await self._runtime.is_alive(timeout=timeout)

    async def _wait_until_alive(self, timeout: float = 10.0):
        assert self._runtime is not None
        return await _wait_until_alive(
            self.is_alive,
            timeout=timeout,
            function_timeout=self._runtime._config.timeout,
        )

    async def _start(self) -> None:
        if self._runtime is not None and self._sandbox is not None:
            self.logger.warning("Deployment is already started. Ignoring duplicate start() call.")
            return

        loop = asyncio.get_running_loop()
        token = self._get_token()
        swerex_cmd = self._swerex_start_command(token)

        self.logger.info(
            f"Starting YR sandbox (port={self._port}, port_forwardings=[{self._port}], "
            f"image={self._config.image!r}, cpu={self._config.cpu}, memory={self._config.memory}), "
            f"swerex_cmd: {swerex_cmd}"
        )
        self._hooks.on_custom_step("Creating YR sandbox (port forwarding)")
        t0 = time.time()

        self._sandbox = await loop.run_in_executor(None, _create_sandbox, self._config)
        elapsed_sandbox_creation = time.time() - t0
        self.logger.info(f"Sandbox {self._sandbox.sandbox_id} created in {elapsed_sandbox_creation:.2f}s")

        self._hooks.on_custom_step(f"Starting swerex on port {self._port} (background)")
        self._command_handle, self._runtime_url = await loop.run_in_executor(
            None,
            lambda: _start_swerex_via_port_forwarding(
                self._sandbox,
                command=swerex_cmd,
                port=self._port,
                internal=self._config.internal,
            ),
        )

        self.logger.info(f"Port forwarding URL: {self._runtime_url}")
        self._hooks.on_custom_step("Connecting RemoteRuntime via port forwarding")

        self._runtime = RemoteRuntime.from_config(
            RemoteRuntimeConfig(
                host=self._runtime_url,
                timeout=self._config.timeout,
                auth_token=token,
                proxy=self._config.proxy,
                ssl_verify=self._config.ssl_verify,
            ),
            run_id=self.run_id,
        )

        remaining_startup_timeout = max(0, self._config.startup_timeout - elapsed_sandbox_creation)
        t1 = time.time()
        await self._wait_until_alive(timeout=remaining_startup_timeout)
        await self.runtime.create_session(CreateBashSessionRequest(startup_timeout=60))
        self.logger.info(f"Runtime started in {time.time() - t1:.2f}s")

    async def start(self, max_retries: int = 5) -> None:
        last_error: Exception | None = None
        for retry in range(max_retries):
            try:
                await self._start()
                return
            except Exception as exc:
                last_error = exc
                self.logger.critical(f"Failed to create YR sandbox: {exc}")
                await self.stop()
                if retry < max_retries - 1:
                    sleep_time = min(30, 2**retry)
                    self.logger.info(f"Retrying YR deployment startup in {sleep_time} seconds...")
                    await asyncio.sleep(sleep_time)

        raise RuntimeError(f"Failed to create YR sandbox after {max_retries} retries") from last_error

    async def stop(self) -> None:
        if self._stopped:
            return

        if self._runtime is not None:
            try:
                await self._runtime.close()
            except Exception as e:
                self.logger.error(f"Failed to close YR runtime: {e}")
            self._runtime = None

        if self._command_handle is not None:
            try:
                self._command_handle.kill()
            except Exception as e:
                self.logger.debug(f"Failed to kill swerex background process: {e}")
            self._command_handle = None

        if self._sandbox is not None:
            loop = asyncio.get_running_loop()
            sandbox_id = self._sandbox.sandbox_id
            try:
                if self._sandbox.is_running():
                    await loop.run_in_executor(None, _kill_sandbox, self._sandbox)
                    self.logger.info(f"Sandbox {sandbox_id} killed")
            except Exception as e:
                self.logger.error(f"Failed to kill sandbox {sandbox_id}: {e}")
            self._sandbox = None

        self._runtime_url = None
        self._stopped = True

    @property
    def runtime(self) -> RemoteRuntime:
        if self._runtime is None:
            raise DeploymentNotStartedError()
        return self._runtime

    @property
    def sandbox(self) -> Any:
        if self._sandbox is None:
            raise DeploymentNotStartedError()
        return self._sandbox

    @property
    def port_forward_url(self) -> str:
        """Gateway URL for the forwarded swerex port (from get_port_url)."""
        if self._runtime_url is None:
            raise DeploymentNotStartedError()
        return self._runtime_url

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
