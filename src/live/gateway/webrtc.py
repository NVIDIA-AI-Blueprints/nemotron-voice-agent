# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebRTC sessions: answering the SDP offer and carrying events on the data channel."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pipecat.transports.smallwebrtc.connection import IceServer

from live.media.connection import LiveWebRTCConnection
from live.protocol import SessionConfig

if TYPE_CHECKING:
    from live.gateway.gateway import LiveGateway
    from live.session import LiveProtocolSession

SDP_TIMEOUT_SECONDS = 20
UNCONNECTED_TIMEOUT_SECONDS = 30
MAX_SDP_CHARS = 262144


async def create_webrtc_session(gateway: LiveGateway, config: SessionConfig, sdp: object) -> JSONResponse:
    """Negotiate a WebRTC session: answer the SDP offer and start the session in the background."""
    config.validate_transport("webrtc")
    if not isinstance(sdp, str) or not sdp.strip() or len(sdp) > MAX_SDP_CHARS:
        raise HTTPException(400, "An SDP offer is required")
    if not sdp.lstrip().startswith("v=0") or "m=audio" not in sdp:
        raise HTTPException(400, "The SDP offer must include an audio track")
    connection = LiveWebRTCConnection(
        ice_servers=[IceServer(urls=list(gateway.settings.ice_urls))] if gateway.settings.ice_urls else None
    )
    try:
        async with asyncio.timeout(SDP_TIMEOUT_SECONDS):
            await connection.initialize(sdp, "offer")
    except Exception as exc:
        await connection.close_after_drain()
        raise HTTPException(400, "WebRTC SDP negotiation failed") from exc
    try:
        session = await gateway.register(config, "webrtc", connection)
    except BaseException:
        await connection.close_after_drain()
        raise
    session.subscribe(lambda message: _send_app_message(connection, message), "primary")

    @connection.event_handler("app-message")
    async def on_message(_connection, message):
        await session.handle(message)

    @connection.event_handler("closed")
    async def on_closed(_connection):
        await session.close("connection_lost")

    @connection.event_handler("failed")
    async def on_failed(_connection):
        await session.close("connection_lost")

    async def start() -> None:
        try:
            await session.start()
        except Exception:
            logger.exception(f"session={session.id} WebRTC startup failed")

    session.spawn(start(), "start")

    session.spawn(expire_unconnected(session, connection), "connection-deadline")
    answer = connection.get_answer()
    return JSONResponse(
        {
            "session": session.snapshot(),
            "transport": {"type": "webrtc", "sdp": answer["sdp"]},
            "sent_session_config": config.model_dump(exclude_none=True),
        },
        status_code=201,
    )


async def expire_unconnected(
    session: LiveProtocolSession, connection: LiveWebRTCConnection, timeout: float = UNCONNECTED_TIMEOUT_SECONDS
) -> None:
    """Close a session whose peer never connected.

    The engine starts without waiting for the peer, so such a session is already ``active``; without this a client
    that posts an offer and never completes ICE would hold a session (and its models) until it expires.
    """
    await asyncio.sleep(timeout)
    if session.status in {"created", "starting"} or not connection.is_connected():
        await session.close("connection_lost")


async def _send_app_message(connection: LiveWebRTCConnection, message: dict) -> None:
    """Send one event on the data channel."""
    connection.send_app_message(message)
