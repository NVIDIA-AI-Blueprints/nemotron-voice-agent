# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The session registry and the entry points of the three routes."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse
from loguru import logger

from live.engine_contract import LiveEngine
from live.gateway import security, webrtc, websocket
from live.gateway.settings import LiveSettings
from live.media.connection import LiveWebRTCConnection
from live.protocol import ProtocolError, SessionConfig
from live.session import LiveProtocolSession

EngineFactory = Callable[[SessionConfig, str, "LiveWebRTCConnection | None"], Awaitable[LiveEngine]]


class LiveGateway:
    """Session registry and the shared logic of the three routes."""

    def __init__(self, engine_factory: EngineFactory, settings: LiveSettings | None = None):
        """Create the gateway for ``engine_factory`` with ``settings`` (default: from the environment)."""
        self.engine_factory = engine_factory
        self.settings = settings or LiveSettings.from_env()
        self.sessions: dict[str, LiveProtocolSession] = {}
        self.connections: dict[str, LiveWebRTCConnection] = {}
        self.lock = asyncio.Lock()

    # ------------------------------------------------------------------------------ security
    def authorize(self, connection) -> None:
        """Check the bearer token (when configured) and reject cross-site browser origins."""
        security.authorize(self.settings, connection)

    async def body(self, request: Request) -> dict:
        """Read a bounded JSON object request body."""
        return await security.read_json_body(self.settings, request)

    # ------------------------------------------------------------------------------ sessions
    async def _on_closed(self, session: LiveProtocolSession) -> None:
        self.sessions.pop(session.id, None)
        connection = self.connections.pop(session.id, None)
        if connection is not None:
            await connection.close_after_drain()

    async def register(
        self, config: SessionConfig, transport: str, connection: LiveWebRTCConnection | None
    ) -> LiveProtocolSession:
        """Build the engine and the session for a new connection, within the concurrent-session limit."""
        async with self.lock:
            if len(self.sessions) >= self.settings.max_sessions:
                raise HTTPException(429, "Maximum concurrent sessions reached")
            try:
                engine = await self.engine_factory(config, transport, connection)
            except (HTTPException, ProtocolError):
                raise
            except Exception as exc:
                logger.exception(f"Engine creation failed: {exc}")
                raise ProtocolError(
                    "The voice pipeline could not be created. Check server model access and configuration.",
                    "server_error",
                ) from exc
            session = LiveProtocolSession(
                config,
                transport,
                engine,
                max_session_seconds=self.settings.max_session_seconds,
                on_closed=self._on_closed,
            )
            self.sessions[session.id] = session
            if connection is not None:
                self.connections[session.id] = connection
            return session

    async def close_all(self) -> None:
        """Close every session (server shutdown)."""
        await asyncio.gather(*(s.close("remote_hangup") for s in list(self.sessions.values())), return_exceptions=True)

    # ---------------------------------------------------------------------------- the routes
    async def create_webrtc(self, config: SessionConfig, sdp: object) -> JSONResponse:
        """Negotiate a WebRTC session: answer the SDP offer and start the session in the background."""
        return await webrtc.create_webrtc_session(self, config, sdp)

    async def run_primary_websocket(self, ws: WebSocket) -> None:
        """Serve one primary WebSocket: ``session.start`` first, then commands until the session ends."""
        await websocket.run_primary_websocket(self, ws)

    async def run_sideband(self, ws: WebSocket, session_id: str) -> None:
        """Serve one sideband observer: events from now on, and trusted commands."""
        await websocket.run_sideband(self, ws, session_id)
