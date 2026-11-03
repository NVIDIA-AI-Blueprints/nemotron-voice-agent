# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebSockets: the primary connection that starts a session, and the sideband that observes and steers one.

The primary connection carries the session's audio, so this edge builds its wire codec (:func:`wire_codec`) and hands
it to the session, which decodes caller audio and encodes reply audio with it. Both connections read their commands
through one loop (:func:`serve_commands`).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from functools import partial
from typing import TYPE_CHECKING

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from live.events import error_event
from live.media.audio import WireCodec
from live.protocol import ProtocolError, SessionConfig, validation_error

if TYPE_CHECKING:
    from live.gateway.gateway import LiveGateway
    from live.session import LiveProtocolSession

MAX_EVENT_BYTES = 2**20
START_TIMEOUT_SECONDS = 15
# A WebSocket its client closed on purpose: a normal closure, or the client going away (RFC 6455 §7.4.1).
CLEAN_CLOSE_CODES = frozenset({1000, 1001})
POLICY_VIOLATION = 1008
# A sideband the session dropped because it fell behind or failed; it may attach again (IANA: Try Again Later).
TRY_AGAIN_LATER = 1013


def wire_codec(config: SessionConfig) -> WireCodec:
    """Build the codec between the session's wire audio format and the PCM16 the gateway exchanges with its adapter."""
    return WireCodec(config.audio.format)


def close_reason(code: int | None) -> str:
    """The official reason a WebSocket's close gives its session: a clean hang-up, or a lost connection."""
    return "remote_hangup" if code in CLEAN_CLOSE_CODES else "connection_lost"


async def parse_event(session: LiveProtocolSession, connection: str, raw: str) -> dict | None:
    """Decode one inbound text frame: not JSON is an ``error`` for its connection; an oversized frame raises."""
    if len(raw) > MAX_EVENT_BYTES:
        raise ProtocolError(f"WebSocket event exceeds {MAX_EVENT_BYTES // 2**20} MiB")
    try:
        return json.loads(raw)
    except ValueError:
        await session.fanout.send_to(connection, error_event(ProtocolError("Invalid JSON event")))
        return None


async def serve_commands(ws: WebSocket, session: LiveProtocolSession, connection: str) -> int | None:
    """Hand the session each command its connection sends, until the session closes or the client leaves.

    Returns the client's close code when it left, ``None`` when the session closed first. A binary frame gets an error
    event, since the protocol's events are JSON text, and the connection stays open. The receive that loses the
    race to the session's close is cancelled; one that finished is always read, so no task error goes unread.
    """
    while session.status not in {"closing", "closed"}:
        receiver = asyncio.create_task(ws.receive())
        closer = asyncio.create_task(session.closed.wait())
        done, pending = await asyncio.wait({receiver, closer}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.wait(pending)  # not gather: see live.events.fanout.Subscription.close
        if receiver not in done:
            return None
        frame = receiver.result()
        if frame["type"] == "websocket.disconnect":
            return frame.get("code", 1000)
        if frame.get("text") is None:
            refused = ProtocolError("Binary frames are not supported: send each event as a JSON text frame")
            await session.fanout.send_to(connection, error_event(refused))
            continue
        message = await parse_event(session, connection, frame["text"])
        if message is not None:
            await session.handle(message, connection)
    return None


async def run_primary_websocket(gateway: LiveGateway, ws: WebSocket) -> None:
    """Serve one primary WebSocket: ``session.start`` first, then commands until the session ends.

    A refused start (an invalid session, a full server, a factory that refuses) gets its ``error`` event, then a 1008
    close. A client that closes its WebSocket ends the session as a clean hang-up (1000, 1001) or a lost connection.
    """
    session, first, reason = None, None, "connection_lost"
    try:
        gateway.authorize(ws)
        await ws.accept()
        async with asyncio.timeout(START_TIMEOUT_SECONDS):
            frame = await ws.receive()
        if frame["type"] == "websocket.disconnect":
            return
        first = json.loads(frame["text"]) if frame.get("text") is not None else None  # a binary frame is no start
        if not isinstance(first, dict) or first.get("type") != "session.start":
            raise ProtocolError("session.start must be the first message")
        try:
            config = SessionConfig.model_validate(first.get("session"))
        except ValidationError as refused:
            raise validation_error(refused, "session") from refused
        config.validate_transport("websocket")
        config = config.with_wire_format()  # the resolved setting, which session.started shows
        session = await gateway.register(config, "websocket", None, codec=wire_codec(config))
        connection = session.subscribe(ws.send_json, "primary")
        await session.start(first.get("event_id"))
        code = await serve_commands(ws, session, connection)
        reason = close_reason(code)
    except ProtocolError as exc:
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await ws.send_json(error_event(exc, first.get("event_id") if isinstance(first, dict) else None))
            if session is None:
                await ws.close(code=POLICY_VIOLATION)
    except json.JSONDecodeError:  # the first frame is not JSON
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
