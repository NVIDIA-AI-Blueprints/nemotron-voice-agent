# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Low-level RTVI WebSocket and protobuf transport."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import ssl
from dataclasses import dataclass
from typing import Any

import websockets
from pipecat.frames.protobufs import frames_pb2
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RTVIAudioFrame:
    """Decoded RTVI PCM audio payload."""

    audio: bytes
    sample_rate: int
    num_channels: int


class RTVITransportClosed(Exception):
    """The RTVI WebSocket closed unexpectedly."""


class RTVITransport:
    """Connect, frame audio, and decode RTVI WebSocket messages."""

    def __init__(
        self,
        *,
        stream_id: str,
        host: str,
        port: int,
        connect_timeout: float,
        verify_tls: bool = True,
    ) -> None:
        """Create an RTVI transport for one benchmark connection."""
        self.stream_id = stream_id
        self.uri = f"wss://{host}:{port}/api/ws"
        self.connect_timeout = connect_timeout
        self.verify_tls = verify_tls
        self.websocket: Any = None

    async def __aenter__(self) -> RTVITransport:
        """Connect and send the RTVI readiness handshake."""
        ssl_context = ssl.create_default_context()
        if not self.verify_tls:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        try:
            self.websocket = await websockets.connect(
                self.uri,
                open_timeout=self.connect_timeout,
                ssl=ssl_context,
            )
            ready_payload = {
                "label": "rtvi-ai",
                "type": "client-ready",
                "id": f"{self.stream_id}-client-ready",
                "data": {"version": "0.1.0", "about": {"name": "scaling-perf-benchmark"}},
            }
            message = frames_pb2.MessageFrame(data=json.dumps(ready_payload))
            await self.websocket.send(frames_pb2.Frame(message=message).SerializeToString())
            return self
        except Exception as exc:
            await self.close()
            if isinstance(exc, ConnectionClosed):
                raise RTVITransportClosed from exc
            raise

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Close the WebSocket when the session exits."""
        del exc_type, exc, tb
        await self.close()

    async def close(self) -> None:
        """Close and clear the active WebSocket."""
        websocket = self.websocket
        self.websocket = None
        if websocket is not None:
            with contextlib.suppress(Exception):
                await websocket.close()

    async def send_audio(self, audio_data: bytes, sample_rate: int, num_channels: int) -> None:
        """Serialize and send one PCM audio frame."""
        audio = frames_pb2.AudioRawFrame(
            audio=audio_data,
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        try:
            await self.websocket.send(frames_pb2.Frame(audio=audio).SerializeToString())
        except ConnectionClosed as exc:
            raise RTVITransportClosed from exc

    async def recv(self, timeout: float | None = None) -> RTVIAudioFrame | dict[str, Any]:
        """Receive one valid audio or JSON message payload."""
        deadline = None if timeout is None else asyncio.get_running_loop().time() + timeout
        while True:
            wait = None if deadline is None else max(0.0, deadline - asyncio.get_running_loop().time())
            if wait == 0:
                raise TimeoutError
            try:
                data = (
                    await self.websocket.recv()
                    if wait is None
                    else await asyncio.wait_for(self.websocket.recv(), timeout=wait)
                )
            except ConnectionClosed as exc:
                raise RTVITransportClosed from exc
            try:
                frame = frames_pb2.Frame.FromString(data)
            except Exception as exc:
                logger.warning("Failed to parse protobuf frame: %s", exc)
                continue
            kind = frame.WhichOneof("frame")
            if kind == "audio":
                return RTVIAudioFrame(frame.audio.audio, frame.audio.sample_rate, frame.audio.num_channels)
            if kind == "message":
                try:
                    message = json.loads(frame.message.data)
                except (TypeError, json.JSONDecodeError) as exc:
                    logger.warning("Failed to parse message frame payload: %s", exc)
                    continue
                if isinstance(message, dict):
                    return message
