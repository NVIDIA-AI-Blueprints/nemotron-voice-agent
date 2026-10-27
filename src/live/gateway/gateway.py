# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The session registry: sessions, settings, security, the one delegation core client, ``register`` and ``close_all``.

The routes (``live.gateway.routes``) serve each connection with the edge's own functions (``live.gateway.webrtc``,
``live.gateway.websocket``); the gateway only builds, holds and closes the sessions.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING

from fastapi import Request
from loguru import logger

from live.gateway import security
from live.gateway.settings import LiveSettings
from live.ports import AdapterFactory, CoreHook
from live.protocol import ProtocolError, SessionConfig
from live.session import LiveProtocolSession
from live.session.session import no_core

if TYPE_CHECKING:
    from live.media.audio import WireCodec
    from live.media.connection import LiveWebRTCConnection
    from live.server_delegation.responses_client import ResponsesClient

# The ICE servers of a WebRTC session's peer, for the request that creates it: ``{urls, username?, credential?}``
# objects, the shape of an ``RTCIceServer`` (TURN credentials included).
IceServers = Callable[[Request], list[dict]]


class CoreClient:
    """The gateway's one delegation core client: the root's ``core`` hook runs once, on first use.

    It is a ``ResponsesClient`` a session's server delegation gets at once; the core is built when the first
    ``responses`` session or console round uses it. Concurrent first uses wait for the one build; a failed build is
    reported as ``server_error`` and tried again on the next use. :meth:`aclose` closes the built client once, at
    server shutdown, and never builds a core only to close it.
    """

    def __init__(self, hook: CoreHook):
        """Build the client with the root's ``hook``, which returns the core's client (or an awaitable of it)."""
        self.hook = hook
        self._client: ResponsesClient | None = None
        self._building: asyncio.Future | None = None

    async def client(self) -> ResponsesClient:
        """The core's client, built by the hook on the first call.

        Raises:
            ProtocolError: The core could not be built (``server_error``, with no detail of the server).
        """
        if self._client is None:
            if self._building is None:
                self._building = asyncio.ensure_future(self._build())
            building = self._building
            try:
                self._client = await asyncio.shield(building)
            except ProtocolError:
                if self._building is building:
                    self._building = None  # tried again on the next use
                raise
        return self._client

    async def _build(self) -> ResponsesClient:
        try:
            built = self.hook()
            return await built if inspect.isawaitable(built) else built
        except ProtocolError:
            raise
        except Exception as exc:
            logger.exception(f"The delegation core could not be built: {exc}")
            raise ProtocolError(
                "The delegation core could not be created. Check server model access and configuration.",
                "server_error",
            ) from exc

    @property
    def close_seconds(self) -> float | None:
        """How long the core's close of a conversation may take, as its client says; ``None`` before it is built."""
        return getattr(self._client, "close_seconds", None)

    async def create_conversation(self, metadata: dict[str, str]) -> dict:
        """``POST /v1/conversations`` on the core."""
        return await (await self.client()).create_conversation(metadata)

    async def delete_conversation(self, conversation_id: str) -> dict:
        """``DELETE /v1/conversations/{id}`` on the core."""
        return await (await self.client()).delete_conversation(conversation_id)

    async def retrieve_model(self, model: str) -> dict:
        """``GET /v1/models/{model}`` on the core."""
        return await (await self.client()).retrieve_model(model)

    async def create_response(self, params: dict):
        """``POST /v1/responses`` on the core."""
        return await (await self.client()).create_response(params)

    async def list_conversation_items(self, conversation_id: str) -> dict:
        """``GET /v1/conversations/{id}/items`` on the core."""
        return await (await self.client()).list_conversation_items(conversation_id)

    async def input_tokens(self, params: dict) -> dict:
        """``POST /v1/responses/input_tokens`` on the core."""
        return await (await self.client()).input_tokens(params)

    async def aclose(self) -> None:
        """Close the core's client once, if the hook built it: a core that was never built is not built to close it."""
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()


class LiveGateway:
    """The registry of one server's live sessions and what they share: settings, security and the core's client."""

    def __init__(
        self,
        factory: AdapterFactory,
        settings: LiveSettings | None = None,
        *,
        core: CoreHook | None = None,
        ice_servers: IceServers | None = None,
    ):
        """Create the gateway.

        Args:
            factory: Builds each session's frontend adapter.
            settings: The operator settings; read from the environment when omitted.
            core: The root's hook that builds the one client of the delegation core every ``responses`` session and
                every console round uses; a coroutine function or a plain one, run once, on first use
                (:class:`CoreClient`). Without it, ``responses`` sessions and console rounds are refused.
            ice_servers: The ICE servers (TURN included) of each WebRTC session's peer, for the creating request.
        """
        self.factory = factory
        self.settings = settings or LiveSettings.from_env()
        self.ice_servers = ice_servers
        self._core = CoreClient(core) if core is not None else None
        self.sessions: dict[str, LiveProtocolSession] = {}
        self.connections: dict[str, LiveWebRTCConnection] = {}
        self.starting = 0  # sessions whose parts are being built: they count against the limit

    # ------------------------------------------------------------------------------ security
    def authorize(self, connection) -> None:
        """Check the bearer token (when configured) and reject cross-site browser origins."""
        security.authorize(self.settings, connection)

    async def body(self, request: Request, limit: int = security.MAX_BODY_BYTES) -> dict:
        """Authorize the request and read its JSON object body, at most ``limit`` bytes."""
        return await security.read_json_body(self.settings, request, limit)

    # ------------------------------------------------------------------------------ the core
    def core(self) -> ResponsesClient:
        """The one client of the delegation core (:class:`CoreClient`): the root's hook builds the core on first use.

        Raises:
            ProtocolError: The server runs no delegation core.
        """
        if self._core is None:
            no_core()
        return self._core

    # ------------------------------------------------------------------------------ sessions
    async def _on_closed(self, session: LiveProtocolSession) -> None:
        self.sessions.pop(session.id, None)
        connection = self.connections.pop(session.id, None)
        if connection is not None:
            await connection.close_after_drain()

    async def register(
        self,
        config: SessionConfig,
        transport: str,
        connection: LiveWebRTCConnection | None,
        *,
        codec: WireCodec | None = None,
    ) -> LiveProtocolSession:
        """Build the parts and the session for a new connection, within the concurrent-session limit.

        A place is taken before the parts are built, and the parts are built with no lock held, so a slow factory never
        holds up another connection. ``connection`` is a WebRTC session's peer, and ``codec`` a primary WebSocket's
        wire audio codec.

        Raises:
            ProtocolError: A full server (``rate_limit_exceeded``), a factory's own refusal, a factory that failed
                (``server_error``), or a ``responses`` session on a server with no core: each edge answers it in its
                own way, a WebSocket with an ``error`` event.
        """
        if len(self.sessions) + self.starting >= self.settings.max_sessions:
            raise ProtocolError(
                f"Maximum concurrent sessions reached: at most {self.settings.max_sessions}", "rate_limit_exceeded"
            )
        self.starting += 1
        try:
            try:
                parts = await self.factory(config, transport, connection)
            except ProtocolError:
                raise
            except Exception as exc:
                logger.exception(f"The frontend adapter could not be built: {exc}")
                raise ProtocolError(
                    "The voice pipeline could not be created. Check server model access and configuration.",
                    "server_error",
                ) from exc
            try:
                session = LiveProtocolSession(
                    config,
                    transport,
                    parts,
                    core=self.core,
                    codec=codec,
                    max_session_seconds=self.settings.max_session_seconds,
                    on_closed=self._on_closed,
                )
            except BaseException:  # no core for a responses session: the adapter built for it is released
                await _release(parts.frontend)
                raise
        finally:
            self.starting -= 1
        self.sessions[session.id] = session
        if connection is not None:
            self.connections[session.id] = connection
        return session

    async def close_all(self) -> None:
        """Close every session, then the one core (server shutdown); ``connection_lost`` is the nearest official reason.

        The core closes after every session, through its client, once and only if it was built. A core whose close
        fails is logged, and the shutdown goes on.
        """
        await asyncio.gather(
            *(s.close("connection_lost", asked_by="server shutdown") for s in list(self.sessions.values())),
            return_exceptions=True,
        )
        if self._core is not None:
            try:
                await self._core.aclose()
            except Exception:
                logger.exception("The delegation core could not be closed at server shutdown")


async def _release(frontend) -> None:
    """Close a frontend adapter that no session took; its failure is logged."""
    try:
        await frontend.close()
    except Exception:
        logger.exception("A frontend adapter that no session took could not be closed")
