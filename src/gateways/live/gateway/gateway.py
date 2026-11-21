# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The session registry: sessions, settings, security and the one delegation core client."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING

from fastapi import Request
from loguru import logger

from gateways.live.gateway import security
from gateways.live.gateway.settings import LiveSettings
from gateways.live.ports import AdapterFactory, CoreHook, MediaConnection, PrimarySocket, SessionParts
from gateways.live.protocol import ProtocolError, SessionConfig
from gateways.live.server_delegation.responses_client import AttachmentClient, CallerContextClient
from gateways.live.session import LiveProtocolSession
from gateways.live.session.session import no_core

if TYPE_CHECKING:
    from examples.shared.delegation_contract import CallerContext
    from gateways.live.server_delegation.responses_client import ResponsesClient, ResponseStream

# The ICE servers of a request's WebRTC peer, shaped as ``RTCIceServer`` objects, TURN credentials included.
IceServers = Callable[[Request], list[dict]]


class CoreClient:
    """The gateway's one delegation core client, which the root's ``core`` hook builds on first use."""

    def __init__(self, hook: CoreHook):
        """Keep the root's ``hook``, which returns the core's client (or an awaitable of it); nothing is built yet."""
        self.hook = hook
        self._client: ResponsesClient | None = None
        self._building: asyncio.Future | None = None

    async def client(self) -> ResponsesClient:
        """Return the core's client, built by the hook on the first call; raise ``ProtocolError`` if it cannot be."""
        if self._client is None:
            if self._building is None:
                self._building = asyncio.ensure_future(self._build())
            building = self._building
            try:
                self._client = await asyncio.shield(building)
            except ProtocolError:
                if self._building is building:
                    self._building = None
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
        """Call ``POST /v1/conversations`` on the core."""
        return await (await self.client()).create_conversation(metadata)

    async def update_conversation(self, conversation_id: str, metadata: dict[str, str]) -> dict:
        """Call ``POST /v1/conversations/{id}`` on the core."""
        return await (await self.client()).update_conversation(conversation_id, metadata)

    async def delete_conversation(self, conversation_id: str) -> dict:
        """Call ``DELETE /v1/conversations/{id}`` on the core."""
        return await (await self.client()).delete_conversation(conversation_id)

    async def attachment_supported(self) -> bool:
        """Check the actual core boundary rather than assuming every Responses implementation supports retention."""
        return isinstance(await self.client(), AttachmentClient)

    async def set_attachment(self, conversation_id: str, attached: bool) -> dict:
        """Fence caller-dependent effects while retaining the logical conversation lease."""
        client = await self.client()
        change = getattr(client, "set_attachment", None)
        if change is None:
            raise ProtocolError("This core does not support attachment retention", "resume_unavailable")
        return await change(conversation_id, attached)

    def begin_caller_turn(self, conversation_id: str) -> Callable[[], None] | None:
        """Forward the local routing hold to the client that opened the conversation."""
        begin = getattr(self._client, "begin_caller_turn", None)
        return begin(conversation_id) if callable(begin) else None

    async def retrieve_model(self, model: str) -> dict:
        """Call ``GET /v1/models/{model}`` on the core."""
        return await (await self.client()).retrieve_model(model)

    async def create_response(self, params: dict):
        """Call ``POST /v1/responses`` on the core."""
        return await (await self.client()).create_response(params)

    async def create_response_with_caller_context(self, params: dict, context: CallerContext) -> ResponseStream:
        """Forward private caller evidence to a core that accepts it."""
        client = await self.client()
        if isinstance(client, CallerContextClient):
            return await client.create_response_with_caller_context(params, context)
        return await client.create_response(params)

    async def list_conversation_items(self, conversation_id: str) -> dict:
        """Call ``GET /v1/conversations/{id}/items`` on the core."""
        return await (await self.client()).list_conversation_items(conversation_id)

    async def input_tokens(self, params: dict) -> dict:
        """Call ``POST /v1/responses/input_tokens`` on the core."""
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
        """Create the gateway; without a ``core`` hook, ``responses`` sessions and delegate rounds are refused."""
        self.factory = factory
        self.settings = settings or LiveSettings.from_env()
        self.ice_servers = ice_servers
        self._core = CoreClient(core) if core is not None else None
        self.sessions: dict[str, LiveProtocolSession] = {}
        self.connections: dict[str, MediaConnection] = {}
        self.starting = 0
        self.resume_leases = {}

    def authorize(self, connection) -> None:
        """Check the bearer token (when configured), reject cross-site browser origins, then check the host name."""
        security.authorize(self.settings, connection)

    async def body(self, request: Request, limit: int = security.MAX_BODY_BYTES) -> dict:
        """Authorize the request and read its JSON object body, at most ``limit`` bytes."""
        return await security.read_json_body(self.settings, request, limit)

    def core(self) -> ResponsesClient:
        """Return the one delegation core client; raise ``ProtocolError`` when the server runs no core."""
        if self._core is None:
            no_core()
        return self._core

    async def _on_closed(self, session: LiveProtocolSession) -> None:
        self.sessions.pop(session.id, None)
        connection = self.connections.pop(session.id, None)
        if connection is not None:
            await connection.close_after_drain()

    async def register(
        self,
        config: SessionConfig,
        transport: str,
        ice_servers: list[dict] | None,
        *,
        socket: PrimarySocket | None = None,
    ) -> LiveProtocolSession:
        """Build the parts and the session for a new connection, within the concurrent-session limit.

        A place is taken before the parts are built, with no lock held, so a slow factory holds up no other connection.
        """
        if len(self.sessions) + self.starting >= self.settings.max_sessions:
            raise ProtocolError(
                f"Maximum concurrent sessions reached: at most {self.settings.max_sessions}", "rate_limit_exceeded"
            )
        self.starting += 1
        try:
            try:
                parts = await self.factory(config, transport, ice_servers)
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
                    socket=socket,
                    max_session_seconds=self.settings.max_session_seconds,
                    max_appends_per_minute=self.settings.max_appends_per_minute,
                    on_closed=self._on_closed,
                )
            except BaseException:
                # No core for a responses session: release the parts built for it.
                await _release(parts)
                raise
        finally:
            self.starting -= 1
        self.sessions[session.id] = session
        if parts.connection is not None:
            self.connections[session.id] = parts.connection
        return session

    async def close_all(self) -> None:
        """Close every session, then the one core, at shutdown; ``connection_lost`` is the nearest official reason."""
        await asyncio.gather(
            *(s.close("connection_lost", asked_by="server shutdown") for s in list(self.sessions.values())),
            return_exceptions=True,
        )
        if self._core is not None:
            try:
                await self._core.aclose()
            except Exception:
                logger.exception("The delegation core could not be closed at server shutdown")


async def _release(parts: SessionParts) -> None:
    try:
        await parts.frontend.close()
    except Exception:
        logger.exception("A frontend adapter that no session took could not be closed")
    if parts.connection is not None:
        await parts.connection.close_after_drain()
