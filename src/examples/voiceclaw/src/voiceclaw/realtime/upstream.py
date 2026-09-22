# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Concrete WebSocket adapter for a configurable Realtime upstream."""

from __future__ import annotations

from types import TracebackType
from typing import Any, Self
from urllib.parse import quote


class RealtimeUpstreamError(RuntimeError):
    """The configured upstream could not provide a valid text WebSocket."""


class WebSocketRealtimeUpstream:
    """Direct, proxy-disabled connection to an OpenAI Realtime-compatible server."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        bearer: str | None,
        connect_timeout_seconds: float,
        max_event_bytes: int,
    ) -> None:
        """Keep upstream credentials server-side and out of facade URLs."""
        self._url = f"{endpoint}?model={quote(model, safe='')}"
        self._bearer = bearer
        self._connect_timeout_seconds = connect_timeout_seconds
        self._max_event_bytes = max_event_bytes
        self._socket: Any | None = None

    async def __aenter__(self) -> Self:
        """Open a direct TLS-validating upstream WebSocket."""
        try:
            from websockets.asyncio.client import connect
        except ImportError as error:
            raise RealtimeUpstreamError("install nemotron-voiceclaw[server]") from error
        headers = {"Authorization": f"Bearer {self._bearer}"} if self._bearer else None
        try:
            self._socket = await connect(
                self._url,
                additional_headers=headers,
                subprotocols=["realtime"],
                open_timeout=self._connect_timeout_seconds,
                close_timeout=5,
                max_size=self._max_event_bytes,
                proxy=None,
            )
        except Exception as error:
            raise RealtimeUpstreamError("realtime upstream connection failed") from error
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the private upstream transport."""
        await self.close()

    async def send_text(self, value: str) -> None:
        """Send one already-validated JSON text event."""
        socket = self._require_socket()
        await socket.send(value)

    async def send(self, message: str) -> None:
        """Implement the framework-neutral facade transport protocol."""
        await self.send_text(message)

    async def receive_text(self) -> str:
        """Receive one text event and reject binary payloads."""
        socket = self._require_socket()
        value = await socket.recv()
        if not isinstance(value, str):
            raise RealtimeUpstreamError("realtime upstream sent a binary event")
        return value

    async def receive(self) -> str:
        """Implement the framework-neutral facade transport protocol."""
        return await self.receive_text()

    async def close(self) -> None:
        """Close idempotently."""
        socket, self._socket = self._socket, None
        if socket is not None:
            await socket.close()

    def _require_socket(self) -> Any:
        if self._socket is None:
            raise RealtimeUpstreamError("realtime upstream is not connected")
        return self._socket
