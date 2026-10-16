# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The voice side's link to the backend gateway (plan section 9.1): one WebSocket per session.

Frames are protocol JSON (``backend.protocol``). One :class:`QueueWriter` sends; one
reader task decodes and hands every gateway message to ``on_message`` on the loop.
A lost link is reported as a synthetic fatal ``error{code: "link_lost"}`` message, so
the turn manager handles it like any other gateway error.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from typing import Any, Protocol

from examples.frontend_delegation_hermes.core.backend.protocol import (
    GATEWAY_TO_VOICE,
    VOICE_TO_GATEWAY,
    ProtocolError,
    decode,
    dumps,
    message,
    validate,
)
from examples.frontend_delegation_hermes.core.backend.writer import QueueWriter, WriterClosedError

logger = logging.getLogger(__name__)

OnMessage = Callable[[dict[str, Any]], None]


class BackendLink(Protocol):
    """Voice ⇄ gateway transport for one Realtime session."""

    async def open(self, on_message: OnMessage) -> None:
        """Connect; raises when the gateway cannot be reached."""
        ...

    def send(self, message: dict[str, Any]) -> None:
        """Enqueue one voice->gateway message (never blocks)."""
        ...

    async def close(self) -> None:
        """Close the link (idempotent)."""
        ...


def link_lost(reason: str) -> dict[str, Any]:
    """The synthetic message delivered when the link dies."""
    return message(GATEWAY_TO_VOICE, "error", code="link_lost", message=reason, fatal=True)


class WebSocketBackendLink:
    """``websockets`` client to ``ws://gateway/v1/backend``; no keepalive pings (plan 9.1)."""

    def __init__(self, url: str, *, open_timeout_s: float = 10.0) -> None:
        """Bind the gateway URL."""
        self._url = url
        self._open_timeout = open_timeout_s
        self._ws: Any = None
        self._writer: QueueWriter | None = None
        self._reader: asyncio.Task[None] | None = None
        self._on_message: OnMessage | None = None
        self._closing = False
        self._lost_reported = False

    async def open(self, on_message: OnMessage) -> None:
        """Connect and start the reader and the writer."""
        import websockets

        self._on_message = on_message
        self._ws = await websockets.connect(
            self._url, ping_interval=None, max_size=None, open_timeout=self._open_timeout, close_timeout=2
        )
        self._writer = QueueWriter(self._ws.send, name="backend-link-writer", on_error=self._on_write_error)
        self._writer.start()
        self._reader = asyncio.create_task(self._read(), name="backend-link-reader")

    def send(self, message: dict[str, Any]) -> None:
        """Validate and enqueue; a closed link reports ``link_lost`` once."""
        validate(VOICE_TO_GATEWAY, message)
        if self._writer is None:
            raise RuntimeError("backend link is not open")
        try:
            self._writer.put(dumps(message))
        except WriterClosedError:
            self._report_lost("backend link closed")

    async def close(self) -> None:
        """Flush pending frames, then close the socket and stop the reader."""
        self._closing = True
        if self._writer is not None:
            await self._writer.close(drain=True, timeout=2.0)
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        if self._reader is not None and not self._reader.done():
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader

    async def _read(self) -> None:
        reason = "gateway closed the connection"
        try:
            async for raw in self._ws:
                try:
                    data = decode(GATEWAY_TO_VOICE, raw)
                except ProtocolError as exc:
                    logger.error("backend link: dropping invalid gateway message: %s", exc)
                    continue
                if self._on_message is not None:
                    self._on_message(data)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any transport failure ends the link
            reason = f"backend link failed: {exc}"
        if not self._closing:
            self._report_lost(reason)

    def _on_write_error(self, exc: BaseException) -> None:
        if not self._closing:
            self._report_lost(f"backend link send failed: {exc}")

    def _report_lost(self, reason: str) -> None:
        if self._lost_reported or self._on_message is None:
            return
        self._lost_reported = True
        self._on_message(link_lost(reason))
