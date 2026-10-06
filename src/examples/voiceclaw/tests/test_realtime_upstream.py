# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Realtime upstream WebSocket closure semantics."""

import asyncio

import pytest
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

from voiceclaw.realtime.upstream import WebSocketRealtimeUpstream


class _ClosedSocket:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def recv(self) -> str:
        raise self.error


def _upstream(error: BaseException) -> WebSocketRealtimeUpstream:
    upstream = WebSocketRealtimeUpstream(
        endpoint="ws://127.0.0.1:7861/v1/realtime",
        model="test-model",
        bearer=None,
        connect_timeout_seconds=1,
        max_event_bytes=1024,
    )
    upstream._socket = _ClosedSocket(error)
    return upstream


def test_clean_upstream_close_becomes_transport_eof() -> None:
    """Treat a normal peer shutdown as end-of-stream."""
    close = Close(1000, "session complete")
    upstream = _upstream(ConnectionClosedOK(close, close, True))

    with pytest.raises(EOFError):
        asyncio.run(upstream.receive_text())


def test_abnormal_upstream_close_remains_an_error() -> None:
    """Do not hide a failed WebSocket close behind EOF semantics."""
    error = ConnectionClosedError(Close(1011, "upstream failed"), None)
    upstream = _upstream(error)

    with pytest.raises(ConnectionClosedError) as raised:
        asyncio.run(upstream.receive_text())

    assert raised.value is error
