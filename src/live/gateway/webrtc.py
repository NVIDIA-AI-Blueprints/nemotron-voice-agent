# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebRTC sessions: answering the SDP offer and carrying events on the data channel."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request
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


def peer_ice_servers(gateway: LiveGateway, request: Request) -> list[IceServer] | None:
    """The ICE servers of a new session's peer: the root's, TURN credentials included, then ``LIVE_ICE_URLS``."""
    servers = [
        IceServer(urls=server["urls"], username=server.get("username"), credential=server.get("credential"))
        for server in (gateway.ice_servers(request) if gateway.ice_servers is not None else [])
    ]
    if gateway.settings.ice_urls:
        servers.append(IceServer(urls=list(gateway.settings.ice_urls)))
    return servers or None


async def create_webrtc_session(
    gateway: LiveGateway, config: SessionConfig, sdp: object, request: Request
) -> JSONResponse:
    """Negotiate a WebRTC session: answer the SDP offer and start the session in the background.

    The session is registered before the offer is answered, so a refused session (a full server, a factory that
    refuses) never starts ICE, and no connection task is left with an unread error. A peer or a data channel that
    closes is a clean hang-up (``remote_hangup``); a peer that fails is ``connection_lost``. The answer is the
    official one: the session's id and the SDP answer; ``session.started`` follows on the data channel.
    """
    config.validate_transport("webrtc")
    if not isinstance(sdp, str) or not sdp.strip() or len(sdp) > MAX_SDP_CHARS:
        raise HTTPException(400, "An SDP offer is required")
    if not sdp.lstrip().startswith("v=0") or "m=audio" not in sdp:
        raise HTTPException(400, "The SDP offer must include an audio track")
    connection = LiveWebRTCConnection(ice_servers=peer_ice_servers(gateway, request))
    try:
        session = await gateway.register(config, "webrtc", connection)
    except BaseException:
        await connection.close_after_drain()
        raise
    try:
        async with asyncio.timeout(SDP_TIMEOUT_SECONDS):
            await connection.initialize(sdp, "offer")
    except Exception as exc:
        await session.close("connection_lost", asked_by="the gateway: the SDP offer could not be answered")
        raise HTTPException(400, "WebRTC SDP negotiation failed") from exc
    primary = session.subscribe(lambda message: _send_app_message(connection, message), "primary")

    @connection.event_handler("app-message")
    async def on_message(_connection, message):
        await session.handle(message, primary)

    @connection.event_handler("closed")
    async def on_closed(_connection):
        await session.close("remote_hangup", asked_by="the WebRTC peer")

    @connection.event_handler("data-channel-closed")
    async def on_channel_closed(_connection):
        await session.close("remote_hangup", asked_by="the WebRTC data channel")

    @connection.event_handler("failed")
    async def on_failed(_connection):
        await session.close("connection_lost", asked_by="the WebRTC peer")

    async def start() -> None:
        try:
            await session.start()
        except Exception:
            logger.exception(f"session={session.id} WebRTC startup failed")

    session.spawn(start(), "start")

    session.spawn(expire_unconnected(session, connection), "connection-deadline")
    answer = connection.get_answer()
    return JSONResponse(
        {"session": {"id": session.id}, "transport": {"type": "webrtc", "sdp": answer["sdp"]}}, status_code=201
    )


async def expire_unconnected(
    session: LiveProtocolSession, connection: LiveWebRTCConnection, timeout: float = UNCONNECTED_TIMEOUT_SECONDS
) -> None:
    """Close a session whose peer never connected.

    The frontend adapter starts without waiting for the peer, so such a session is already ``active``; without this a
    client that posts an offer and never completes ICE would hold a session (and its models) until it expires.
    """
    await asyncio.sleep(timeout)
    if session.status in {"created", "starting"} or not connection.is_connected():
        await session.close("connection_lost", asked_by="the gateway: the WebRTC peer never connected")


async def _send_app_message(connection: LiveWebRTCConnection, message: dict) -> None:
    """Send one event on the data channel."""
    connection.send_app_message(message)
