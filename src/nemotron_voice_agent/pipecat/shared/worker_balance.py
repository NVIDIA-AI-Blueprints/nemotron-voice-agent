# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spread WebSocket sessions evenly over the Uvicorn workers of one process group.

With ``UVICORN_WORKERS > 1`` every worker waits on the same listening socket, so the worker that wakes
first takes the next connection. Sessions then pile up on a few workers while the others stay idle, and a
busy worker's event loop falls behind. This module makes a worker take new connections only while it holds
the fewest sessions of all live workers. The first connections therefore go one to each worker, and after
every worker has one, the load grows evenly.

How it works:

* Each worker claims a slot in a small shared file and publishes its session count there. A worker's count
  is the WebSocket sessions it runs plus connections it accepted that have not reached the app yet.
* Every few milliseconds a worker compares its count with the lowest count of the live workers. When it is
  above the lowest, it stops watching the listening socket, so the kernel offers new connections to the other
  workers. When it is at the lowest again, it starts watching the socket again.

Off by default (``NVA_WORKER_BALANCE=1`` turns it on). It uses asyncio's reader registrations on the
listening socket, so it needs the standard asyncio event loop (the server selects it when balancing is on;
uvloop does not expose the selector) and a selector-based loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import mmap
import os
import struct
import tempfile
import time
from pathlib import Path

from loguru import logger

BALANCE_ENV = "NVA_WORKER_BALANCE"
INTERVAL_ENV = "NVA_WORKER_BALANCE_INTERVAL_MS"
DEFAULT_INTERVAL_MS = 10
# An accepted connection that never reaches the app (a port scan, a failed TLS handshake) stops counting after this.
PENDING_EXPIRY_S = 3.0

_SLOT = struct.Struct("ii")  # pid, session count
_EMPTY_PID = 0


class SlotTable:
    """Per-worker session counts in a shared file. Each worker writes only its own slot."""

    def __init__(self, directory: Path, slots: int) -> None:
        """Open or create the table.

        Args:
            directory: Directory shared by the workers of one process group.
            slots: Most workers that may claim a slot.
        """
        self.directory = directory
        self.slots = slots
        directory.mkdir(parents=True, exist_ok=True)
        self._path = directory / "counts"
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if os.fstat(fd).st_size < slots * _SLOT.size:
                os.ftruncate(fd, slots * _SLOT.size)
            self._map = mmap.mmap(fd, slots * _SLOT.size)
        finally:
            os.close(fd)
        self._cells = memoryview(self._map).cast("i")
        self.index: int | None = None
        self._live_slots: list[int] = []
        self._live_checked = 0.0

    def claim(self) -> int:
        """Take a free slot (or one left by a dead worker) and return its index."""
        for index in range(self.slots):
            lock = self.directory / f"slot-{index}"
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if not self._lock_is_stale(lock):
                    continue
                lock.unlink(missing_ok=True)
                try:
                    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                except FileExistsError:
                    continue
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            self.index = index
            self.publish(0)
            return index
        raise RuntimeError(f"No free worker slot among {self.slots}")

    @staticmethod
    def _lock_is_stale(lock: Path) -> bool:
        try:
            return not _pid_alive(int(lock.read_text() or 0))
        except (OSError, ValueError):
            return True

    def publish(self, count: int) -> None:
        """Write this worker's session count."""
        assert self.index is not None
        _SLOT.pack_into(self._map, self.index * _SLOT.size, os.getpid(), max(count, 0))

    def counts(self) -> list[int]:
        """Return the session counts of every live worker, this one included."""
        now = time.monotonic()
        if now - self._live_checked > 1.0:
            self._live_checked = now
            pids = self._cells[0::2]
            self._live_slots = [i for i, pid in enumerate(pids) if pid != _EMPTY_PID and _pid_alive(pid)]
        counts = self._cells[1::2]
        return [counts[i] for i in self._live_slots]

    def release(self) -> None:
        """Give the slot back when the worker stops."""
        if self.index is None:
            return
        _SLOT.pack_into(self._map, self.index * _SLOT.size, _EMPTY_PID, 0)
        (self.directory / f"slot-{self.index}").unlink(missing_ok=True)
        self.index = None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def should_accept(own: int, counts: list[int]) -> bool:
    """Return whether a worker holding ``own`` sessions may take the next connection.

    Args:
        own: This worker's session count.
        counts: Session counts of all live workers, this one included.
    """
    return not counts or own <= min(counts)


class WorkerBalancer:
    """Pause and resume accepting on a worker's listening sockets so the least-loaded workers take sessions."""

    def __init__(self, table: SlotTable, loop: asyncio.AbstractEventLoop, interval_s: float) -> None:
        """Create the balancer.

        Args:
            table: Shared per-worker counts, already claimed by this worker.
            loop: The worker's event loop.
            interval_s: How often to re-check whether to accept.
        """
        self.table = table
        self.loop = loop
        self.interval_s = interval_s
        self.sessions = 0
        self._pending: list[float] = []
        self._listeners: dict[int, tuple] = {}
        self._paused = False
        self._task: asyncio.Task | None = None

    # ---- counts --------------------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Sessions plus fresh connections that have not reached the app."""
        cutoff = time.monotonic() - PENDING_EXPIRY_S
        self._pending = [t for t in self._pending if t > cutoff]
        return self.sessions + len(self._pending)

    def connection_accepted(self) -> None:
        """Record a connection that was accepted but has not started a request yet."""
        self._pending.append(time.monotonic())

    def scope_started(self, websocket: bool) -> None:
        """Record that a connection reached the app; a WebSocket scope also counts as a session."""
        if self._pending:
            self._pending.pop(0)
        if websocket:
            self.sessions += 1

    def scope_ended(self, websocket: bool) -> None:
        """Record that a WebSocket session ended."""
        if websocket:
            self.sessions = max(self.sessions - 1, 0)

    # ---- listening sockets ---------------------------------------------------------------------

    def _discover(self) -> None:
        """Find the accept registrations of listening sockets, wrapping the connection factory to count accepts."""
        selector = getattr(self.loop, "_selector", None)
        if selector is None:
            return
        for fd, key in list(selector.get_map().items()):
            if fd in self._listeners:
                continue
            reader = key.data[0] if key.data else None
            if reader is None or getattr(reader, "_cancelled", False):
                continue
            callback, args = reader._callback, reader._args
            # Asyncio's accept loop is the reader registered on a listening socket.
            if getattr(callback, "__name__", "") != "_accept_connection":
                continue
            if args:
                factory = args[0]

                def counted(*a, _factory=factory, **k):
                    self.connection_accepted()
                    return _factory(*a, **k)

                args = (counted, *args[1:])
                self.loop.remove_reader(fd)
                self.loop.add_reader(fd, callback, *args)
            self._listeners[fd] = (callback, args)

    def _set_accepting(self, accepting: bool) -> None:
        if accepting == (not self._paused):
            return
        logger.debug(f"balance: accepting={accepting} own={self.count} listeners={len(self._listeners)}")
        for fd, (callback, args) in self._listeners.items():
            if accepting:
                self.loop.add_reader(fd, callback, *args)
            else:
                self.loop.remove_reader(fd)
        self._paused = not accepting

    async def _run(self) -> None:
        while True:
            if not self._listeners:
                self._discover()
            own = self.count
            self.table.publish(own)
            self._set_accepting(should_accept(own, self.table.counts()))
            await asyncio.sleep(self.interval_s)

    def start(self) -> None:
        """Start the periodic check on the worker's event loop."""
        self._task = self.loop.create_task(self._run())

    async def stop(self) -> None:
        """Stop the check, resume accepting and give the slot back."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._set_accepting(True)
        self.table.release()


class BalanceMiddleware:
    """ASGI middleware that reports scope starts and ends to the balancer."""

    def __init__(self, app, balancer_ref) -> None:
        """Wrap ``app``.

        Args:
            app: The ASGI application.
            balancer_ref: Callable returning the worker's ``WorkerBalancer`` or ``None`` before it starts.
        """
        self.app = app
        self._balancer = balancer_ref

    async def __call__(self, scope, receive, send):
        """Count the scope while the wrapped app handles it."""
        balancer = self._balancer()
        if balancer is None or scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        websocket = scope["type"] == "websocket"
        balancer.scope_started(websocket)
        try:
            return await self.app(scope, receive, send)
        finally:
            balancer.scope_ended(websocket)


def balancing_enabled(workers: int) -> bool:
    """Return whether this worker should balance: opted in and more than one worker."""
    return os.getenv(BALANCE_ENV, "").strip().lower() in {"1", "true", "yes", "on"} and workers > 1


def balance_directory() -> Path:
    """Return the directory shared by the workers of this process group (keyed by the supervisor's pid)."""
    base = Path(os.getenv("NVA_WORKER_BALANCE_DIR") or tempfile.gettempdir())
    return base / f"nva-worker-balance-{os.getppid()}"


async def start_balancer(workers: int) -> WorkerBalancer:
    """Claim a slot and start balancing on the running loop.

    Args:
        workers: Number of Uvicorn workers in the process group.
    """
    table = SlotTable(balance_directory(), slots=workers + 8)
    table.claim()
    interval = max(int(os.getenv(INTERVAL_ENV, DEFAULT_INTERVAL_MS)), 1) / 1000
    balancer = WorkerBalancer(table, asyncio.get_running_loop(), interval)
    balancer.start()
    logger.info(f"Worker balancing on: slot {table.index} of {table.slots}, check every {interval * 1000:.0f} ms")
    return balancer
