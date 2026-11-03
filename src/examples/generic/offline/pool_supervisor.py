# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own single-worker offline backends behind an optional stream dispatcher.

The existing server entrypoint is unchanged. This separate startup mode binds
each backend to its own inherited loopback socket, so connection acceptance can
never move a reserved stream to another process.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import secrets
import socket
import sys
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass

import httpx


class PoolStartupError(RuntimeError):
    """A fixed startup/ownership failure without a backend response body."""


def _seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


@dataclass
class _OwnedWorker:
    worker_id: str
    base_url: str
    process: asyncio.subprocess.Process


class WorkerSupervisor:
    """Start and stop only this dispatcher's directly owned backend processes."""

    def __init__(
        self,
        count=10,
        *,
        worker_factory="server:app_factory",
        startup_timeout_s=60.0,
        stop_timeout_s=5.0,
        environment=None,
        inherit_worker_process_group=False,
    ):
        """Bound process count and watchdogs independently of response deadlines."""
        if type(count) is not int or not 1 <= count <= 64:
            raise ValueError("Offline pool size must be an integer from 1 through 64")
        if not isinstance(worker_factory, str) or not worker_factory or ":" not in worker_factory:
            raise ValueError("Worker factory must be a trusted module:attribute")
        if type(inherit_worker_process_group) is not bool:
            raise ValueError("Inherited worker process group must be an explicit boolean")
        self.count = count
        self.worker_factory = worker_factory
        self.startup_timeout_s = _seconds(startup_timeout_s, "startup_timeout_s")
        self.stop_timeout_s = _seconds(stop_timeout_s, "stop_timeout_s")
        self.token = secrets.token_urlsafe(32)
        self._environment = dict(os.environ if environment is None else environment)
        self.inherit_worker_process_group = inherit_worker_process_group
        self._workers: list[_OwnedWorker] = []
        self._lock = asyncio.Lock()
        self._started = False
        self._closed = False
        self._shutdown_task: asyncio.Task | None = None

    @property
    def owned_pids(self):
        """Return process identities from owned subprocess handles only."""
        return tuple(worker.process.pid for worker in self._workers)

    async def start(self):
        """Require every owned backend's authenticated readiness before serving."""
        from .pool_proxy import BackendDescriptor

        async with self._lock:
            if self._closed:
                raise PoolStartupError("offline_pool_already_closed")
            if self._started:
                return [BackendDescriptor(w.worker_id, w.base_url, w.process.pid) for w in self._workers]
            if os.name != "posix":
                raise PoolStartupError("offline_pool_requires_posix_inherited_sockets")
            if self.inherit_worker_process_group and os.getpgrp() != os.getpid():
                raise PoolStartupError("offline_pool_inherited_group_requires_owned_session_leader")
            try:
                for index in range(self.count):
                    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    try:
                        listener.bind(("127.0.0.1", 0))
                        listener.listen(socket.SOMAXCONN)
                        listener.set_inheritable(True)
                        port = listener.getsockname()[1]
                        environment = dict(self._environment)
                        environment.update(NVA_OFFLINE_POOL_TOKEN=self.token, UVICORN_WORKERS="1")
                        process = await asyncio.create_subprocess_exec(
                            sys.executable,
                            "-m",
                            "examples.generic.offline.worker",
                            "--fd",
                            str(listener.fileno()),
                            "--parent-pid",
                            str(os.getpid()),
                            "--factory",
                            self.worker_factory,
                            env=environment,
                            pass_fds=(listener.fileno(),),
                            start_new_session=not self.inherit_worker_process_group,
                            stdin=asyncio.subprocess.DEVNULL,
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL,
                        )
                        self._workers.append(
                            _OwnedWorker(f"worker-{index + 1:02d}", f"http://127.0.0.1:{port}", process)
                        )
                    finally:
                        listener.close()
                deadline = time.monotonic() + self.startup_timeout_s
                async with (
                    httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=1.0) as client,
                    asyncio.TaskGroup() as group,
                ):
                    for worker in self._workers:
                        group.create_task(self._ready(client, worker, deadline))
                if len(set(self.owned_pids)) != self.count:
                    raise PoolStartupError("offline_pool_worker_identity_collision")
                self._started = True
                return [BackendDescriptor(w.worker_id, w.base_url, w.process.pid) for w in self._workers]
            except BaseException as startup_error:
                self._closed = True
                try:
                    await self._stop_workers()
                except asyncio.CancelledError as cancellation:
                    cancellation.add_note(f"offline_pool_startup_failed:{type(startup_error).__name__}")
                    if cancellation.__cause__ is None:
                        raise cancellation from startup_error
                    raise
                except BaseException as cleanup_error:
                    startup_error.add_note("offline_pool_owned_backend_shutdown_failed")
                    raise startup_error from cleanup_error
                raise

    async def _ready(self, client, worker, deadline):
        from .worker import TOKEN_HEADER

        while time.monotonic() < deadline:
            if worker.process.returncode is not None:
                raise PoolStartupError("offline_pool_backend_exited_during_startup")
            try:
                response = await client.get(
                    worker.base_url + "/__offline_pool/status", headers={TOKEN_HEADER: self.token}
                )
                if response.status_code == 200 and len(response.content) <= 16384:
                    value = response.json()
                    if (
                        isinstance(value, dict)
                        and value.get("pid") == worker.process.pid
                        and value.get("busy") is False
                        and value.get("active_lease") is None
                    ):
                        return
            except (httpx.HTTPError, ValueError):
                pass
            await asyncio.sleep(min(0.05, max(0, deadline - time.monotonic())))
        raise PoolStartupError("offline_pool_backend_startup_deadline")

    async def _stop_workers(self):
        """Join owned bounded cleanup before propagating caller cancellation."""
        if self._shutdown_task is None or self._shutdown_task.done():
            self._shutdown_task = asyncio.create_task(self._reap_workers())
        task = self._shutdown_task
        cancellation = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
            except Exception:
                break
        try:
            task.result()
        except BaseException as cleanup_error:
            if cancellation is not None:
                cancellation.add_note("offline_pool_owned_backend_shutdown_failed")
                raise cancellation from cleanup_error
            raise
        if cancellation is not None:
            raise cancellation

    async def _reap_workers(self):
        """Reap direct handles; never discover or signal a listener by port/PID text."""

        async def stop(worker):
            process = worker.process
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.terminate()
            try:
                await asyncio.wait_for(asyncio.shield(process.wait()), self.stop_timeout_s)
            except TimeoutError:
                if process.returncode is None:
                    with suppress(ProcessLookupError):
                        process.kill()
                await asyncio.wait_for(process.wait(), self.stop_timeout_s)

        outcomes = await asyncio.gather(*(stop(worker) for worker in self._workers), return_exceptions=True)
        if any(isinstance(outcome, BaseException) for outcome in outcomes):
            raise PoolStartupError("offline_pool_owned_backend_shutdown_failed")
        self._closed = True

    async def aclose(self):
        """Close all owned processes even after a partial startup failure."""
        async with self._lock:
            if self._closed and all(worker.process.returncode is not None for worker in self._workers):
                return
            await self._stop_workers()
            self._closed = True


def create_managed_pool_app(
    *,
    count=10,
    worker_factory="server:app_factory",
    startup_timeout_s=60.0,
    drain_timeout_s=60.0,
    admission_timeout_s=1.0,
    inherit_worker_process_group=False,
):
    """Build an opt-in offline front door without changing server:app_factory."""
    from .pool_proxy import ExclusiveWorkerPool, create_pool_proxy

    supervisor = WorkerSupervisor(
        count,
        worker_factory=worker_factory,
        startup_timeout_s=startup_timeout_s,
        inherit_worker_process_group=inherit_worker_process_group,
    )

    @asynccontextmanager
    async def lifespan(app):
        pool = None
        try:
            backends = await supervisor.start()
            pool = ExclusiveWorkerPool(
                backends,
                token=supervisor.token,
                drain_timeout_s=drain_timeout_s,
                admission_timeout_s=admission_timeout_s,
            )
            readiness = await pool.initialize()
            if readiness.get("idle") != count or readiness.get("quarantined") != 0:
                raise PoolStartupError("offline_pool_not_all_backends_ready")
            app.state.worker_pool = pool
            app.state.worker_supervisor = supervisor
            yield
        finally:
            try:
                if pool is not None:
                    await pool.aclose()
            finally:
                await supervisor.aclose()

    return create_pool_proxy(lifespan=lifespan)


def app_factory():
    """Use trusted operator configuration for the separate dispatcher entrypoint."""
    value = os.getenv("NVA_OFFLINE_POOL_SIZE", "10")
    try:
        count = int(value)
    except ValueError as error:
        raise ValueError("NVA_OFFLINE_POOL_SIZE must be an integer") from error
    return create_managed_pool_app(count=count)


def main():
    """Serve one front door and a fixed private backend pool on POSIX hosts."""
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17927)
    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument("--startup-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--drain-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--admission-wait-seconds", type=float, default=1.0)
    parser.add_argument(
        "--inherit-worker-process-group",
        action="store_true",
        help="Keep children in this session leader's group for explicit CI-owned cleanup",
    )
    args = parser.parse_args()
    if not 0 < args.port <= 65535:
        parser.error("--port must be from 1 through 65535")
    app = create_managed_pool_app(
        count=args.pool_size,
        startup_timeout_s=args.startup_timeout_seconds,
        drain_timeout_s=args.drain_timeout_seconds,
        admission_timeout_s=args.admission_wait_seconds,
        inherit_worker_process_group=args.inherit_worker_process_group,
    )
    uvicorn.run(
        app, host=args.host, port=args.port, workers=1, access_log=False, ws_max_size=8 * 1024 * 1024, ws_max_queue=1
    )


if __name__ == "__main__":
    main()
