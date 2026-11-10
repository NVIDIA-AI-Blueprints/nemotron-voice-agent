# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebSockets: the primary connection that starts a session, and the sideband that observes and steers one."""

from __future__ import annotations

import asyncio
import contextlib
import json
from functools import partial
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from gateways.live.events import error_event
from gateways.live.protocol import ProtocolError, SessionConfig, validation_error
from gateways.live.session import audio_input

if TYPE_CHECKING:
    from starlette.websockets import WebSocketState

    from gateways.live.gateway.gateway import LiveGateway
    from gateways.live.protocol import AudioFormat
    from gateways.live.session import LiveProtocolSession

MAX_EVENT_BYTES = 2**20
START_TIMEOUT_SECONDS = 15
# A normal closure, or the client going away (RFC 6455 §7.4.1).
CLEAN_CLOSE_CODES = frozenset({1000, 1001})
POLICY_VIOLATION = 1008
TRY_AGAIN_LATER = 1013


def close_reason(code: int | None) -> str:
    """Return the session's close reason for a WebSocket close code: a clean hang-up, or a lost connection."""
    return "remote_hangup" if code in CLEAN_CLOSE_CODES else "connection_lost"


async def parse_event(session: LiveProtocolSession, connection: str, raw: str | bytes | None) -> Any:
    """Return an inbound frame's JSON event, or ``None`` after an ``error`` for one with none; one too large raises."""
    if not isinstance(raw, str):
        refused = ProtocolError("Binary frames are not supported: send each event as a JSON text frame")
        await session.fanout.send_to(connection, error_event(refused))
        return None
    if len(raw) > MAX_EVENT_BYTES:
        raise ProtocolError(f"WebSocket event exceeds {MAX_EVENT_BYTES // 2**20} MiB")
    try:
        return json.loads(raw)
    except ValueError:
        await session.fanout.send_to(connection, error_event(ProtocolError("Invalid JSON event")))
        return None


async def serve_commands(ws: WebSocket, session: LiveProtocolSession, connection: str) -> int | None:
    """Hand the session each command its connection sends; return the client's close code, or ``None``."""
    while session.status not in {"closing", "closed"}:
        receiver = asyncio.create_task(ws.receive())
        closer = asyncio.create_task(session.closed.wait())
        done, pending = await asyncio.wait({receiver, closer}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if pending:
            # Not gather: see gateways.live.events.fanout.Subscription.close.
            await asyncio.wait(pending)
        if receiver not in done:
            return None
        frame = receiver.result()
        if frame["type"] == "websocket.disconnect":
            return frame.get("code", 1000)
        message = await parse_event(session, connection, frame.get("text"))
        if message is not None:
            await session.handle(message, connection)
    return None


class PrimarySocketView:
    """A primary WebSocket as its session's frontend adapter reads it (``gateways.live.ports.PrimarySocket``)."""

    def __init__(self, ws: WebSocket, audio_format: AudioFormat):
        """Wrap the accepted ``ws``, whose session's wire audio format is ``audio_format``; :meth:`attach` follows."""
        self.ws, self.format = ws, audio_format
        self.session: LiveProtocolSession | None = None
        self.connection: str | None = None
        self.left = asyncio.Event()
        self.code: int | None = None
        self.refused: ProtocolError | None = None

    def attach(self, session: LiveProtocolSession) -> None:
        """Make the socket ``session``'s primary connection: the session's events go to the client from now on."""
        self.session, self.connection = session, session.subscribe(self.ws.send_json, "primary")

    @property
    def client_state(self) -> WebSocketState:
        """Whether the client is still connected, as the transport checks before it writes."""
        return self.ws.client_state

    async def receive(self) -> dict:
        """Return the client's next frame, keeping a close's code; a receive that fails is a lost connection."""
        try:
            frame = await self.ws.receive()
        except Exception:
            self.left.set()
            raise
        if frame["type"] == "websocket.disconnect":
            self.code = frame.get("code", 1000)
            self.left.set()
        return frame

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        """Leave the socket open: the gateway closes it once ``session.closed`` has gone out."""

    async def handle(self, data: str | bytes) -> bytes | None:
        """Take a client frame; return a ``session.input_audio.append``'s wire audio, else ``None``; never raise."""
        if self.left.is_set():
            return None
        try:
            message = await parse_event(self.session, self.connection, data)
        except ProtocolError as refused:
            self.refused = refused
            self.left.set()
            return None
        if isinstance(message, dict) and message.get("type") == "session.input_audio.append":
            try:
                return audio_input.wire_audio(message, self.format)
            except ProtocolError as refused:
                await self.session.fanout.send_to(self.connection, error_event(refused, message.get("event_id")))
                return None
        if message is not None:
            await self.session.handle(message, self.connection)
        return None

    async def wait_closed(self) -> int | None:
        """Wait until the client leaves or the session closes; return its close code or ``None``; re-raise a refusal."""
        leaver = asyncio.create_task(self.left.wait())
        closer = asyncio.create_task(self.session.closed.wait())
        try:
            await asyncio.wait({leaver, closer}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            leaver.cancel()
            closer.cancel()
        if self.refused is not None:
            raise self.refused
        return self.code if self.left.is_set() else None


async def run_primary_websocket(gateway: LiveGateway, ws: WebSocket) -> None:
    """Serve one primary WebSocket: ``session.start`` first, then the session, whose adapter reads the socket."""
    session, first, reason = None, None, "connection_lost"
    try:
        gateway.authorize(ws)
        await ws.accept()
        async with asyncio.timeout(START_TIMEOUT_SECONDS):
            frame = await ws.receive()
        if frame["type"] == "websocket.disconnect":
            return
        if len(frame.get("text") or "") > MAX_EVENT_BYTES:
            raise ProtocolError(f"WebSocket event exceeds {MAX_EVENT_BYTES // 2**20} MiB")
        first = json.loads(frame["text"]) if frame.get("text") is not None else None
        if not isinstance(first, dict) or first.get("type") != "session.start":
            raise ProtocolError("session.start must be the first message")
        try:
            config = SessionConfig.model_validate(first.get("session"))
        except ValidationError as refused:
            raise validation_error(refused, "session") from refused
        config.validate_transport("websocket")
        config = config.with_wire_format()
        socket = PrimarySocketView(ws, config.audio.format)
        session = await gateway.register(config, "websocket", None, socket=socket)
        socket.attach(session)
        try:
            await session.start(first.get("event_id"))
        except Exception:
            # The start reported its failure and closed the session.
            return
        code = await socket.wait_closed()
        reason = close_reason(code)
    except ProtocolError as exc:
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            start_id = first.get("event_id") if session is None and isinstance(first, dict) else None
            await ws.send_json(error_event(exc, start_id))
            if session is None:
                await ws.close(code=POLICY_VIOLATION)
    except json.JSONDecodeError:
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await ws.send_json(error_event(ProtocolError("Invalid JSON event")))
            await ws.close(code=POLICY_VIOLATION)
    except HTTPException:
        await ws.close(code=POLICY_VIOLATION)
    except WebSocketDisconnect as gone:
        reason = close_reason(gone.code)
    except TimeoutError:
        pass
    finally:
        if session and session.status != "closed":
            await session.close(reason, asked_by="the primary WebSocket")
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await ws.close()


async def close_dropped(ws: WebSocket, reason: str) -> None:
    """Close a sideband the session dropped, with the reason, so its client knows to attach again."""
    with contextlib.suppress(RuntimeError, WebSocketDisconnect):
        await ws.close(code=TRY_AGAIN_LATER, reason=reason)


async def run_sideband(gateway: LiveGateway, ws: WebSocket, session_id: str) -> None:
    """Serve one sideband observer: events from now on, and trusted commands."""
    key, session = None, gateway.sessions.get(session_id)
    try:
        gateway.authorize(ws)
        if session is None or session.status in {"closing", "closed"}:
            await ws.close(code=POLICY_VIOLATION)
            return
        await ws.accept()
        # No session.started replay: attaching observes only subsequent events.
        key = session.subscribe(ws.send_json, "sideband", partial(close_dropped, ws))
        await serve_commands(ws, session, key)
    except ProtocolError as exc:
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await ws.send_json(error_event(exc))
    except HTTPException:
        await ws.close(code=POLICY_VIOLATION)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if session and key:
            await session.unsubscribe(key)
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await ws.close()
