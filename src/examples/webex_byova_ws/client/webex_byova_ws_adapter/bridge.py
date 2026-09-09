"""One-conversation bridge to the Nemotron RTVI protobuf WebSocket."""

from __future__ import annotations

import asyncio
import contextlib
import json
import ssl
from dataclasses import dataclass, field
from typing import Any

import websockets
from pipecat.frames.protobufs import frames_pb2

from webex_byova_ws_adapter.config import AdapterConfig


def _backend_ssl(url: str, insecure: bool) -> ssl.SSLContext | None:
    if not url.startswith("wss://"):
        return None
    context = ssl.create_default_context()
    if insecure:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    return context


@dataclass(slots=True)
class NemotronBridge:
    """Own exactly one internal RTVI connection."""

    config: AdapterConfig
    conversation_id: str
    events: asyncio.Queue[dict[str, Any]] = field(init=False)
    websocket: Any | None = None
    reader_task: asyncio.Task[None] | None = None
    closed: bool = False

    def __post_init__(self) -> None:
        """Create the bounded upstream event queue."""
        self.events = asyncio.Queue(maxsize=self.config.queue_size)

    async def start(self) -> None:
        """Connect and announce this sidecar as an RTVI client."""
        url = self.config.backend_ws_url.rstrip("/")
        if not url.endswith("/api/ws"):
            url += "/api/ws"
        self.websocket = await asyncio.wait_for(
            websockets.connect(
                url,
                ssl=_backend_ssl(url, self.config.backend_insecure_tls),
                max_queue=self.config.queue_size,
                max_size=self.config.max_message_bytes,
            ),
            timeout=self.config.send_timeout_seconds,
        )
        self.reader_task = asyncio.create_task(self._reader(), name=f"rtvi-reader-{self.conversation_id}")
        await self._send_message(
            {
                "label": "rtvi-ai",
                "type": "client-ready",
                "id": f"client-ready-{self.conversation_id}",
                "data": {"version": "1.2.0", "about": {"library": "cisco-webex-byova-ws-adapter"}},
            }
        )

    async def send_audio(self, pcm16: bytes) -> None:
        """Forward PCM16 16 kHz mono audio without artificial pacing."""
        if not self.websocket:
            raise RuntimeError("Nemotron bridge is not connected")
        frame = frames_pb2.Frame(audio=frames_pb2.AudioRawFrame(audio=pcm16, sample_rate=16_000, num_channels=1))
        await asyncio.wait_for(self.websocket.send(frame.SerializeToString()), self.config.send_timeout_seconds)

    async def send_input(self, kind: str, data: dict[str, Any]) -> None:
        """Forward DTMF and Cisco input events as an RTVI client message."""
        await self._send_message(
            {
                "label": "rtvi-ai",
                "type": "client-message",
                "id": f"{kind}-{self.conversation_id}",
                "data": {"type": f"webex-{kind}", **data},
            }
        )

    async def _send_message(self, payload: dict[str, Any]) -> None:
        if not self.websocket:
            raise RuntimeError("Nemotron bridge is not connected")
        frame = frames_pb2.Frame(message=frames_pb2.MessageFrame(data=json.dumps(payload)))
        await asyncio.wait_for(self.websocket.send(frame.SerializeToString()), self.config.send_timeout_seconds)

    async def _reader(self) -> None:
        try:
            assert self.websocket is not None
            async for raw in self.websocket:
                if not isinstance(raw, bytes):
                    await self._put({"kind": "error", "detail": "upstream sent a non-binary RTVI frame"})
                    return
                frame = frames_pb2.Frame.FromString(raw)
                frame_type = frame.WhichOneof("frame")
                if frame_type == "audio" and frame.audio.audio:
                    await self._put(
                        {
                            "kind": "audio",
                            "audio": bytes(frame.audio.audio),
                            "sample_rate": frame.audio.sample_rate,
                            "num_channels": frame.audio.num_channels,
                        }
                    )
                elif frame_type == "message":
                    await self._handle_message(frame.message.data)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._put({"kind": "error", "detail": "Nemotron connection failed"})

    async def _handle_message(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        if payload.get("type") == "server-message" and isinstance(payload.get("data"), dict):
            payload = payload["data"]
        event_type = payload.get("type")
        if event_type == "bot-stopped-speaking":
            await self._put({"kind": "output-complete"})
        elif event_type in {
            "user-started-speaking",
            "user-turn-finalized",
            "bot-started-speaking",
        }:
            await self._put({"kind": "event", "type": event_type, "data": payload.get("data", {})})
        elif event_type == "bot-output":
            data = payload.get("data")
            if isinstance(data, dict) and data.get("spoken") and isinstance(data.get("text"), str):
                await self._put({"kind": "transcript", "text": data["text"]})

    async def _put(self, event: dict[str, Any]) -> None:
        try:
            await asyncio.wait_for(self.events.put(event), self.config.send_timeout_seconds)
        except TimeoutError:
            self.closed = True
            if self.websocket:
                await self.websocket.close(code=1013, reason="adapter backpressure")

    async def close(self) -> None:
        """Cancel tasks and close the upstream connection idempotently."""
        if self.closed and (self.reader_task is None or self.reader_task.done()):
            return
        self.closed = True
        if self.websocket is not None:
            with contextlib.suppress(Exception):
                await self.websocket.close()
        if self.reader_task is not None and self.reader_task is not asyncio.current_task():
            self.reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.reader_task
