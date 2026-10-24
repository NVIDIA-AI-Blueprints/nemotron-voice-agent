# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker processes: spawn, ready, stop, kill, warm spares (plan section 4.2).

:class:`ProcessWorkerHandle` runs one worker per Realtime session in its own process
group over a Unix socket (JSON lines, one writer); the backend's ``worker_setup`` prepares
it (Hermes: its own ``HERMES_HOME``). :class:`InProcessWorkerHandle` runs the same
:class:`WorkerCore` on a thread in this process (``workers.mode: in_process``): the LLM
backend, and the fake one for tests and ``--stub-backend``.

Both deliver worker→gateway messages to ``on_message`` and report exactly one
``on_exit(reason)`` when the worker goes away for any reason.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import os
import shutil
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from examples.frontend_delegation_hermes.core.backend import protocol as proto
from examples.frontend_delegation_hermes.core.backend.writer import QueueWriter, WriterClosedError
from examples.frontend_delegation_hermes.core.backends import load
from examples.frontend_delegation_hermes.core.backends.base import WorkerPrep
from examples.frontend_delegation_hermes.core.sidecar.gateway_config import SRC_DIR, GatewayConfig

logger = logging.getLogger(__name__)

OnMessage = Callable[[dict[str, Any]], None]
OnExit = Callable[[str], None]

STREAM_LIMIT = 64 * 1024 * 1024
WORKER_MODULE = "examples.frontend_delegation_hermes.core.worker.worker_main"


class WorkerStartError(RuntimeError):
    """The worker did not become ready."""


class WorkerHandle(Protocol):
    """One worker, whatever runs it."""

    pid: int | None
    home: Path | None

    @property
    def alive(self) -> bool:
        """Whether the worker can take commands."""
        ...

    def set_listener(self, on_message: OnMessage, on_exit: OnExit) -> None:
        """Where worker messages and the single exit notice go."""
        ...

    async def start(self) -> dict[str, Any]:
        """Start (idempotent) and return the ``ready`` message; raises :class:`WorkerStartError`."""
        ...

    def send(self, message: dict[str, Any]) -> None:
        """Send one gateway→worker message (``{"type": ..., **fields}``)."""
        ...

    async def stop(self) -> str:
        """Graceful ``close`` → wait → SIGTERM → SIGKILL; returns how it ended."""
        ...

    async def kill(self, reason: str) -> None:
        """SIGTERM → SIGKILL now (a hung worker)."""
        ...

    def rss_mb(self) -> float | None:
        """Resident memory of the worker process."""
        ...


def resolve_python(value: str) -> str:
    """``self`` (or empty) → this interpreter; a bare command → ``PATH`` lookup; else the path."""
    if not value or value == "self":
        return sys.executable
    if "/" not in value:
        return shutil.which(value) or value
    return value


def examples_only_path(base: str | Path) -> Path:
    """A directory whose only entry is an ``examples`` symlink to this repo's package."""
    target = Path(base) / "pythonpath"
    link = target / "examples"
    source = SRC_DIR / "examples"
    target.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() == source.resolve():
        return target
    with contextlib.suppress(FileNotFoundError):
        link.unlink()
    with contextlib.suppress(FileExistsError):  # another worker created it concurrently
        link.symlink_to(source, target_is_directory=True)
    return target


def _frame(message: dict[str, Any]) -> str:
    fields = dict(message)
    kind = fields.pop("type")
    fields.pop("v", None)
    return proto.encode(proto.GATEWAY_TO_WORKER, kind, **fields)


class ProcessWorkerHandle:
    """A worker subprocess speaking JSON lines over a Unix socket."""

    _counter = itertools.count(1)

    def __init__(self, config: GatewayConfig, session_id: str, *, soul: str) -> None:
        """Nothing starts until :meth:`start`."""
        self._config = config
        self.session_id = session_id
        self._soul = soul
        self._n = next(self._counter)
        self.pid: int | None = None
        self.home: Path | None = None
        self._prep = WorkerPrep()
        self._proc: asyncio.subprocess.Process | None = None
        self._server: asyncio.base_events.Server | None = None
        self._socket: Path | None = None
        self._writer: QueueWriter | None = None
        self._stream_writer: asyncio.StreamWriter | None = None
        self._ready: asyncio.Future[dict[str, Any]] | None = None
        self._on_message: OnMessage = lambda _msg: None
        self._on_exit: OnExit = lambda _reason: None
        self._exit_reported = False
        self._stopping = False
        self._tasks: list[asyncio.Task[Any]] = []
        self._log_file: Any = None
        self.start_ms: int | None = None

    @property
    def alive(self) -> bool:
        """Connected, ready and not exited."""
        return (
            self._proc is not None
            and self._proc.returncode is None
            and self._writer is not None
            and not self._writer.closed
            and not self._exit_reported
        )

    def set_listener(self, on_message: OnMessage, on_exit: OnExit) -> None:
        """Bind the controller's callbacks."""
        self._on_message, self._on_exit = on_message, on_exit

    async def start(self) -> dict[str, Any]:
        """Spawn, wait for ``ready`` within ``workers.start_timeout_s``."""
        if self._ready is not None:
            return await self._ready
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        w = self._config.workers
        started = time.monotonic()
        name = f"{self.session_id}-{self._n}"
        setup = self._setup()
        if setup is not None:
            self._prep = setup.prepare(self._config, name, soul=self._soul)
            self.home = self._prep.home
        socket_dir = Path(w.socket_dir)
        socket_dir.mkdir(parents=True, exist_ok=True)
        self._socket = socket_dir / f"{name}.sock"
        with contextlib.suppress(FileNotFoundError):
            self._socket.unlink()
        self._server = await asyncio.start_unix_server(self._on_connect, path=str(self._socket), limit=STREAM_LIMIT)
        log_dir = Path(w.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = open(log_dir / f"{name}.log", "ab")  # noqa: SIM115 - closed in _cleanup
        python = resolve_python(w.python)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                python,
                "-m",
                WORKER_MODULE,
                "--socket",
                str(self._socket),
                "--agent-kind",
                self._config.backend.kind,
                "--session-id",
                self.session_id,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=self._log_file,
                stderr=self._log_file,
                env=self._env(),
                start_new_session=True,
            )
        except OSError as exc:
            await self._cleanup()
            raise WorkerStartError(f"cannot start worker with {python}: {exc}") from exc
        self.pid = self._proc.pid
        self._tasks.append(asyncio.create_task(self._watch_process(), name=f"worker-wait-{name}"))
        try:
            ready = await asyncio.wait_for(asyncio.shield(self._ready), timeout=w.start_timeout_s)
        except TimeoutError as exc:
            self._stopping = True
            await self._terminate()
            await self._cleanup()
            raise WorkerStartError(f"worker did not become ready within {w.start_timeout_s}s") from exc
        except Exception as exc:
            self._stopping = True
            await self._terminate()
            await self._cleanup()
            raise WorkerStartError(f"worker exited before it was ready: {exc}") from exc
        self.start_ms = int((time.monotonic() - started) * 1000)
        return ready

    def _env(self) -> dict[str, str]:
        w = self._config.workers
        env = dict(os.environ)
        # Only the ``prototypes`` package: ``src/`` also holds top-level modules (``utils.py``,
        # ``server.py``) that would shadow Hermes' own top-level modules of the same names.
        paths = [str(examples_only_path(w.socket_dir)), *self._prep.pythonpath]
        # The gateway's own PYTHONPATH (usually ``src``) is deliberately not inherited.
        env["PYTHONPATH"] = os.pathsep.join(paths)
        env.update(self._prep.env)
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _setup(self) -> Any:
        target = self._config.spec.worker_setup
        return load(target)() if target else None

    async def _on_connect(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._writer is not None:  # one worker, one connection
            writer.close()
            return
        self._stream_writer = writer

        async def send(frame: str) -> None:
            writer.write(frame.encode("utf-8") + b"\n")
            await writer.drain()

        self._writer = QueueWriter(send, name=f"gateway->worker-{self.pid}")
        self._writer.start()
        self._tasks.append(asyncio.create_task(self._read(reader), name=f"worker-read-{self.pid}"))

    async def _read(self, reader: asyncio.StreamReader) -> None:
        reason = "socket_closed"
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    msg = proto.decode(proto.WORKER_TO_GATEWAY, line)
                except proto.ProtocolError as exc:
                    logger.error("worker %s: bad frame: %s", self.pid, exc)
                    continue
                if msg["type"] == "ready" and self._ready is not None and not self._ready.done():
                    self._ready.set_result(msg)
                    continue
                self._on_message(msg)
        except (ConnectionError, asyncio.IncompleteReadError) as exc:
            reason = f"socket_error: {exc}"
        self._report_exit(reason)

    async def _watch_process(self) -> None:
        assert self._proc is not None  # noqa: S101
        code = await self._proc.wait()
        if self._ready is not None and not self._ready.done():
            self._ready.set_exception(RuntimeError(f"exit code {code}"))
        self._report_exit(f"exited: {code}")

    def _report_exit(self, reason: str) -> None:
        if self._exit_reported:
            return
        self._exit_reported = True
        if self._writer is not None:
            asyncio.get_running_loop().create_task(self._writer.close(drain=False))
        ready = self._ready
        if ready is not None and ready.done() and not ready.cancelled() and ready.exception() is None:
            self._on_exit(reason if not self._stopping else f"stopped ({reason})")

    def send(self, message: dict[str, Any]) -> None:
        """Enqueue one frame; raises ``WriterClosedError`` if the worker is gone."""
        if self._writer is None:
            raise WriterClosedError("worker not connected")
        self._writer.put(_frame(message))

    async def stop(self) -> str:
        """``close`` → wait ``stop_timeout_s`` → SIGTERM → ``kill_grace_s`` → SIGKILL; then clean up."""
        self._stopping = True
        how = "exited"
        if self._proc is not None and self._proc.returncode is None:
            with contextlib.suppress(Exception):
                self.send({"type": "close"})
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=self._config.workers.stop_timeout_s)
                how = "closed"
            except TimeoutError:
                how = await self._terminate()
        await self._cleanup()
        return how

    async def kill(self, reason: str) -> None:
        """SIGTERM → SIGKILL the process group now."""
        logger.warning("killing worker %s (%s)", self.pid, reason)
        await self._terminate()

    async def _terminate(self) -> str:
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return "exited"
        self._signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=self._config.workers.kill_grace_s)
            return "sigterm"
        except TimeoutError:
            self._signal(signal.SIGKILL)
            with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            return "sigkill"

    def _signal(self, sig: int) -> None:
        if self.pid is None:
            return
        try:
            os.killpg(os.getpgid(self.pid), sig)
        except (ProcessLookupError, PermissionError):
            with contextlib.suppress(ProcessLookupError):
                os.kill(self.pid, sig)

    async def _cleanup(self) -> None:
        if self._writer is not None:
            await self._writer.close(drain=False, timeout=1.0)
        if self._stream_writer is not None:
            with contextlib.suppress(Exception):
                self._stream_writer.close()
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
        if self._socket is not None:
            with contextlib.suppress(FileNotFoundError):
                self._socket.unlink()
        for task in self._tasks:
            if not task.done():
                task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if self._log_file is not None:
            with contextlib.suppress(Exception):
                self._log_file.close()
        setup = self._setup()
        if setup is not None and not self._config.workers.keep_homes:
            setup.remove(self._prep)

    def rss_mb(self) -> float | None:
        """VmRSS from /proc (Linux)."""
        if self.pid is None:
            return None
        try:
            for line in Path(f"/proc/{self.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
        except OSError:
            return None
        return None


class InProcessWorkerHandle:
    """A :class:`WorkerCore` on a thread in this process (``Placement.IN_PROCESS`` backends: llm, fake)."""

    def __init__(self, config: GatewayConfig, session_id: str, *, soul: str = "") -> None:
        """Nothing starts until :meth:`start`."""
        self._config = config
        self.session_id = session_id
        self.pid: int | None = os.getpid()
        self.home: Path | None = None
        self._core: Any = None
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._on_message: OnMessage = lambda _msg: None
        self._on_exit: OnExit = lambda _reason: None
        self._dead = False
        self._ready: dict[str, Any] | None = None
        self.start_ms: int | None = 0

    @property
    def alive(self) -> bool:
        """Until killed or closed."""
        return self._ready is not None and not self._dead

    def set_listener(self, on_message: OnMessage, on_exit: OnExit) -> None:
        """Bind the controller's callbacks."""
        self._on_message, self._on_exit = on_message, on_exit

    async def start(self) -> dict[str, Any]:
        """Build the core; ``ready`` immediately."""
        if self._ready is not None:
            return self._ready
        from examples.frontend_delegation_hermes.core.worker.worker_core import (  # noqa: PLC0415
            WorkerCore,
            ready_message,
        )
        from examples.frontend_delegation_hermes.core.worker.worker_main import make_host_factory  # noqa: PLC0415

        self._loop = asyncio.get_running_loop()
        self._core = WorkerCore(
            session_id=self.session_id,
            host_factory=make_host_factory(self._config.backend.kind, self.session_id, in_process=True),
            emit=self._emit,
            on_exit=lambda: None,
        )
        self._task = asyncio.create_task(self._serve(), name=f"inproc-worker-{self.session_id}")
        self._ready = {"v": proto.PROTOCOL_VERSION, **ready_message(self._core.host)}
        return self._ready

    def _emit(self, data: dict[str, Any]) -> None:
        fields = dict(data)
        kind = fields.pop("type")
        msg = proto.message(proto.WORKER_TO_GATEWAY, kind, **fields)
        loop = self._loop
        if self._dead or loop is None or loop.is_closed():
            raise WriterClosedError("in-process worker is gone")
        loop.call_soon_threadsafe(self._deliver, msg)

    def _deliver(self, msg: dict[str, Any]) -> None:
        if not self._dead:
            self._on_message(msg)

    async def _serve(self) -> None:
        while True:
            msg = await self._queue.get()
            if msg is None:
                return
            await self._core.handle(msg)
            if self._core.exited:
                self._mark_dead("closed")
                return

    def send(self, message: dict[str, Any]) -> None:
        """Validate and enqueue."""
        if self._dead:
            raise WriterClosedError("in-process worker is gone")
        self._queue.put_nowait(proto.decode(proto.GATEWAY_TO_WORKER, _frame(message)))

    async def stop(self) -> str:
        """Close the core (bounded), then stop serving."""
        if self._core is not None and not self._dead:
            self.send({"type": "close"})
            if self._task is not None:
                with contextlib.suppress(TimeoutError, asyncio.TimeoutError, Exception):
                    await asyncio.wait_for(asyncio.shield(self._task), timeout=self._config.workers.stop_timeout_s)
        self._mark_dead("stopped", notify=False)
        await self._finish()
        return "closed"

    async def kill(self, reason: str) -> None:
        """Abandon the core: its thread cannot be killed in-process; in-process agents honour interrupts."""
        if self._core is not None:
            self._core.futures.close_all()
            with contextlib.suppress(Exception):
                if self._core.host.built:
                    self._core.host.agent().hard_interrupt(reason)
        self._mark_dead(f"killed: {reason}")
        await self._finish()

    def _mark_dead(self, reason: str, *, notify: bool = True) -> None:
        if self._dead:
            return
        self._dead = True
        if notify:
            self._on_exit(reason)

    async def _finish(self) -> None:
        if self._task is not None and not self._task.done():
            self._queue.put_nowait(None)
            with contextlib.suppress(TimeoutError, asyncio.TimeoutError, asyncio.CancelledError, Exception):
                await asyncio.wait_for(self._task, timeout=1.0)
            if not self._task.done():
                self._task.cancel()

    def rss_mb(self) -> float | None:
        """Not meaningful in-process."""
        return None


class WorkerPool:
    """Creates worker handles for sessions; keeps ``workers.warm`` spares when processes are used."""

    def __init__(self, config: GatewayConfig, *, soul: str) -> None:
        """No worker starts until :meth:`acquire` or :meth:`prewarm`."""
        self._config = config
        self._soul = soul
        self._spares: list[WorkerHandle] = []
        self._warm_task: asyncio.Task[None] | None = None
        self._warm_counter = itertools.count(1)

    def create(self, session_id: str) -> WorkerHandle:
        """A new, not yet started handle."""
        if self._config.workers.mode == "in_process":
            return InProcessWorkerHandle(self._config, session_id, soul=self._soul)
        return ProcessWorkerHandle(self._config, session_id, soul=self._soul)

    def acquire(self, session_id: str) -> WorkerHandle:
        """A warm spare when available, else a new handle; refills spares in the background."""
        handle: WorkerHandle | None = None
        while self._spares:
            spare = self._spares.pop(0)
            if spare.alive:
                handle = spare
                break
        self.prewarm()
        return handle if handle is not None else self.create(session_id)

    def prewarm(self) -> None:
        """Top the spares up to ``workers.warm`` (no-op for in-process workers)."""
        if self._config.workers.warm <= 0 or self._config.workers.mode != "process_per_session":
            return
        if self._warm_task is not None and not self._warm_task.done():
            return
        self._warm_task = asyncio.get_running_loop().create_task(self._fill())

    async def _fill(self) -> None:
        while len(self._spares) < self._config.workers.warm:
            handle = self.create(f"warm{next(self._warm_counter)}")
            try:
                await handle.start()
            except WorkerStartError as exc:
                logger.error("warm worker failed to start: %s", exc)
                return
            handle.set_listener(lambda _m: None, lambda _r: None)
            self._spares.append(handle)

    async def close(self) -> None:
        """Stop spares."""
        if self._warm_task is not None and not self._warm_task.done():
            self._warm_task.cancel()
        spares, self._spares = self._spares, []
        for spare in spares:
            with contextlib.suppress(Exception):
                await spare.stop()
