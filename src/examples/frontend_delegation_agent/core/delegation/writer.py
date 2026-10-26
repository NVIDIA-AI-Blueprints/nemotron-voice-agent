# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The single outbound writer of one connection end (plan section 9).

Nothing else writes to the socket: producers (tasks, or threads through
:meth:`QueueWriter.put_threadsafe`) enqueue whole frames and one task drains the
queue in order, so frames are never interleaved and the order per producer is kept.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

Send = Callable[[str], Awaitable[None]]

_CLOSE = object()


class WriterClosedError(RuntimeError):
    """The writer no longer accepts frames."""


class QueueWriter:
    """Drains an ``asyncio.Queue`` of frames into ``send`` on one task."""

    def __init__(self, send: Send, *, name: str = "writer", on_error: Callable[[BaseException], None] | None = None):
        """Bind the transport's ``send``; call :meth:`start` on the owning loop."""
        self._send = send
        self._name = name
        self._on_error = on_error
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` was called or sending failed."""
        return self._closed

    def start(self) -> None:
        """Start the drain task on the running loop (idempotent)."""
        if self._task is None:
            self._loop = asyncio.get_running_loop()
            self._task = asyncio.create_task(self._drain(), name=self._name)

    def put(self, frame: str) -> None:
        """Enqueue one frame from the owning loop; raises :class:`WriterClosedError` once closed."""
        if self._closed:
            raise WriterClosedError(f"{self._name} is closed")
        self._queue.put_nowait(frame)

    def put_threadsafe(self, frame: str) -> None:
        """Enqueue from another thread; raises :class:`WriterClosedError` once closed."""
        loop = self._loop
        if self._closed or loop is None or loop.is_closed():
            raise WriterClosedError(f"{self._name} is closed")
        loop.call_soon_threadsafe(self._put_from_loop, frame)

    def _put_from_loop(self, frame: str) -> None:
        if not self._closed:
            self._queue.put_nowait(frame)

    async def close(self, *, drain: bool = True, timeout: float = 5.0) -> None:
        """Stop accepting frames; send what is queued (``drain``) or drop it, then stop the task."""
        if self._task is None:
            self._closed = True
            return
        if not self._closed:
            self._closed = True
            if not drain:
                while not self._queue.empty():
                    self._queue.get_nowait()
            self._queue.put_nowait(_CLOSE)
        with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError, Exception):
            await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)
        if not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task

    async def _drain(self) -> None:
        while True:
            frame = await self._queue.get()
            if frame is _CLOSE:
                return
            try:
                await self._send(frame)  # type: ignore[arg-type]
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a dead transport closes the writer
                self._closed = True
                logger.warning("%s: send failed: %s", self._name, exc)
                if self._on_error is not None:
                    self._on_error(exc)
                return
