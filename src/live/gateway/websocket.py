# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebSockets: the primary connection that starts a session, and the sideband that observes and steers one."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from live.events import error_event
from live.protocol import ProtocolError, SessionConfig

if TYPE_CHECKING:
    from live.gateway.gateway import LiveGateway

MAX_EVENT_BYTES = 2**20
START_TIMEOUT_SECONDS = 15


async def parse_event(session, raw: str) -> dict | None:
    """Decode one inbound text frame. A frame that is not JSON becomes an ``error`` event; an oversized one raises."""
    if len(raw) > MAX_EVENT_BYTES:
        raise ProtocolError("WebSocket event exceeds 1 MiB")
    try:
        return json.loads(raw)
    except ValueError:
        await session.publish(error_event(ProtocolError("Invalid JSON event")))
        return None


async def run_primary_websocket(gateway: LiveGateway, ws: WebSocket) -> None:
    """Serve one primary WebSocket: ``session.start`` first, then commands until the session ends."""
    session, first = None, None
    try:
        gateway.authorize(ws)
        await ws.accept()
        async with asyncio.timeout(START_TIMEOUT_SECONDS):
            first = await ws.receive_json()
        if not isinstance(first, dict) or first.get("type") != "session.start":
            await ws.send_json(error_event(ProtocolError("session.start must be the first message")))
            await ws.close(code=1008)
            return
        config = SessionConfig.model_validate(first.get("session"))
        config.validate_transport("websocket")
        session = await gateway.register(config, "websocket", None)
        session.subscribe(ws.send_json, "primary")
        await session.start(first.get("event_id"))
        while session.status == "active":
            # Wake on server expiry as well as inbound commands.
            receiver = asyncio.create_task(ws.receive_text())
            closer = asyncio.create_task(session.closed.wait())
            done, pending = await asyncio.wait({receiver, closer}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if closer in done:
                break
            message = await parse_event(session, receiver.result())
            if message is not None:
                await session.handle(message)
    except (ProtocolError, ValidationError, ValueError) as exc:
        await ws.send_json(
            error_event(
                exc if isinstance(exc, ProtocolError) else ProtocolError(str(exc)[:1000]),
                first.get("event_id") if isinstance(first, dict) else None,
            )
        )
    except HTTPException:
        await ws.close(code=1008)
    except (WebSocketDisconnect, TimeoutError):
        pass
    finally:
        if session and session.status != "closed":
            await session.close("connection_lost")
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await ws.close()


async def run_sideband(gateway: LiveGateway, ws: WebSocket, session_id: str) -> None:
    """Serve one sideband observer: events from now on, and trusted commands."""
    key, session = None, gateway.sessions.get(session_id)
    try:
        gateway.authorize(ws)
        if session is None or session.status in {"closing", "closed"}:
            await ws.close(code=1008)
            return
        await ws.accept()
        # No session.started replay: attaching observes only subsequent events.
        key = session.subscribe(ws.send_json, "sideband")
        while session.status not in {"closing", "closed"}:
            receiver = asyncio.create_task(ws.receive_text())
            closer = asyncio.create_task(session.closed.wait())
            done, pending = await asyncio.wait({receiver, closer}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if closer in done:
                break
            message = await parse_event(session, receiver.result())
            if message is not None:
                await session.handle(message, "sideband")
    except ProtocolError as exc:
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await ws.send_json(error_event(exc))
    except HTTPException:
        await ws.close(code=1008)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if session and key:
            await session.unsubscribe(key)
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await ws.close()
