# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-writer send queue for one WebSocket.

Every server event goes through :meth:`WireWriter.emit`, which only enqueues;
one sender task serializes and sends in FIFO order. Events from different
tasks therefore can never interleave mid-sequence, and a slow socket never
blocks the audio path.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from loguru import logger

_AUDIO_EVENT_TYPES = frozenset({"response.output_audio.delta"})


class WireTransport(Protocol):
    """Where serialized events go (a WebSocket in production, a list in tests)."""

    async def send_text(self, text: str) -> None:
        """Send one text frame."""


class WireWriter:
    """FIFO event sender with optional wire logging (never logs audio payloads)."""

    def __init__(
        self,
        transport: WireTransport,
        *,
        log_wire: bool = False,
        session_id: str = "",
        observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Bind to ``transport``; call :meth:`start` inside the event loop."""
        self._transport = transport
        self._log_wire = log_wire
        self._session_id = session_id
        self._observer = observer
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self.sent = 0

    @property
    def closed(self) -> bool:
        """Whether the transport failed or the writer was closed."""
        return self._closed

    def start(self) -> None:
        """Start the sender task."""
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name=f"wire-writer-{self._session_id}")

    def emit(self, event: dict[str, Any]) -> None:
        """Enqueue one event (non-blocking)."""
        if self._closed:
            return
        if self._observer is not None:
            self._observer(event)
        self._queue.put_nowait(event)

    async def drain(self) -> None:
        """Wait until every enqueued event has been sent."""
        await self._queue.join()

    async def close(self) -> None:
        """Flush and stop the sender task."""
        if self._task is None:
            self._closed = True
            return
        if not self._closed:
            self._queue.put_nowait(None)
        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except (TimeoutError, asyncio.CancelledError):
            self._task.cancel()
        self._closed = True

    async def _run(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                if event is None:
                    return
                if self._closed:
                    continue
                if self._log_wire and event.get("type") not in _AUDIO_EVENT_TYPES:
                    logger.debug(f"[wire:{self._session_id}] -> {event.get('type')}")
                try:
                    await self._transport.send_text(json.dumps(event, ensure_ascii=False))
                    self.sent += 1
                except Exception as exc:  # noqa: BLE001 - a dead socket ends the session, not the server
                    logger.info(f"[wire:{self._session_id}] send failed, closing writer: {exc}")
                    self._closed = True
            finally:
                self._queue.task_done()


class CallbackTransport:
    """Adapter from an ``async (text) -> None`` callable to :class:`WireTransport`."""

    def __init__(self, send: Callable[[str], Awaitable[None]]) -> None:
        """Wrap ``send``."""
        self._send = send

    async def send_text(self, text: str) -> None:
        """Send via the wrapped callable."""
        await self._send(text)
