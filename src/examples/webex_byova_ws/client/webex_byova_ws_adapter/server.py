"""aiohttp listener for Cisco BYOVA WebSocket endpoints."""

from __future__ import annotations

import asyncio
import json
import ssl
from typing import Any

from aiohttp import WSMsgType, web

from webex_byova_ws_adapter.auth import CiscoJwsValidator
from webex_byova_ws_adapter.config import AdapterConfig
from webex_byova_ws_adapter.protocol import ProtocolError, parse_request, parse_text
from webex_byova_ws_adapter.session import Conversation

CONFIG = web.AppKey("config", AdapterConfig)
VALIDATOR = web.AppKey("validator", CiscoJwsValidator)
CONVERSATIONS = web.AppKey("conversations", dict[str, Conversation])
CONVERSATION_LOCK = web.AppKey("conversation_lock", asyncio.Lock)


async def _authenticate(request: web.Request) -> None:
    config = request.app[CONFIG]
    if not config.auth_enabled:
        return
    try:
        await request.app[VALIDATOR].validate(request.headers.get("Authorization", ""))
    except Exception as exc:
        raise web.HTTPUnauthorized(text="invalid Cisco bearer token") from exc


async def voice_agent(request: web.Request) -> web.WebSocketResponse:
    """Route Cisco physical sockets to a stable logical conversation."""
    await _authenticate(request)
    config = request.app[CONFIG]
    websocket = web.WebSocketResponse(
        autoping=True,
        heartbeat=None,
        max_msg_size=config.max_message_bytes,
        receive_timeout=None,
    )
    await websocket.prepare(request)
    first_message = await websocket.receive(timeout=config.send_timeout_seconds)
    if first_message.type != WSMsgType.TEXT:
        await websocket.close(code=1003, message=b"JSON text required")
        return websocket
    try:
        envelope = parse_text(first_message.data)
        parsed = parse_request(envelope)
    except ProtocolError:
        await websocket.close(code=1007, message=b"invalid first request")
        return websocket

    event_type = parsed.input_value.get("event_type") if parsed.input_kind == "event_input" else None
    async with request.app[CONVERSATION_LOCK]:
        conversation = request.app[CONVERSATIONS].get(parsed.conversation_id)
        if conversation is not None and conversation.closed:
            request.app[CONVERSATIONS].pop(parsed.conversation_id, None)
            conversation = None
        if conversation is None:
            if event_type != "SESSION_START":
                await websocket.close(code=1008, message=b"SESSION_START required")
                return websocket
            conversation = Conversation(config, websocket)
            request.app[CONVERSATIONS][parsed.conversation_id] = conversation
            continuation = False
        elif event_type == "SESSION_START":
            await websocket.close(code=1008, message=b"conversation already active")
            return websocket
        else:
            continuation = True
    try:
        if continuation:
            await conversation.attach(websocket, envelope)
            await conversation.run(websocket=websocket)
        else:
            await conversation.run(envelope, websocket)
    except TimeoutError:
        await websocket.close(code=1013, message=b"active reply did not complete")
    except (ProtocolError, ValueError):
        pass
    finally:
        await conversation.closed_event.wait()
        async with request.app[CONVERSATION_LOCK]:
            if request.app[CONVERSATIONS].get(parsed.conversation_id) is conversation:
                request.app[CONVERSATIONS].pop(parsed.conversation_id, None)
    return websocket


async def list_virtual_agents(request: web.Request) -> web.WebSocketResponse:
    """Serve one unwrapped ListVARequest/ListVAResponse exchange."""
    await _authenticate(request)
    config = request.app[CONFIG]
    websocket = web.WebSocketResponse(max_msg_size=config.max_message_bytes)
    await websocket.prepare(request)
    message = await websocket.receive(timeout=config.send_timeout_seconds)
    if message.type != WSMsgType.TEXT:
        await websocket.close(code=1003, message=b"JSON text required")
        return websocket
    try:
        payload: Any = json.loads(message.data)
        if not isinstance(payload, dict) or set(payload) - {
            "customer_org_id",
            "is_default_virtual_agent_enabled",
        }:
            raise ValueError
        if not isinstance(payload.get("customer_org_id"), str) or not payload["customer_org_id"]:
            raise ValueError
        if "is_default_virtual_agent_enabled" in payload and not isinstance(
            payload["is_default_virtual_agent_enabled"], bool
        ):
            raise ValueError
        if config.auth_enabled and payload["customer_org_id"] != config.org_uuid:
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        await websocket.close(code=1007, message=b"invalid ListVARequest")
        return websocket
    await websocket.send_json(
        {
            "virtual_agents": [
                {
                    "id": config.virtual_agent_id,
                    "name": config.virtual_agent_name,
                    "description": config.virtual_agent_description,
                }
            ]
        }
    )
    await websocket.close(code=1000)
    return websocket


async def health(_: web.Request) -> web.Response:
    """Return local/operator process health (not a Cisco schema endpoint)."""
    return web.json_response({"status": "ok", "scope": "local-operator"})


def create_app(config: AdapterConfig | None = None) -> web.Application:
    """Construct the adapter application."""
    resolved = config or AdapterConfig()
    resolved.validate()
    app = web.Application(client_max_size=resolved.max_message_bytes)
    app[CONFIG] = resolved
    app[VALIDATOR] = CiscoJwsValidator(resolved)
    app[CONVERSATIONS] = {}
    app[CONVERSATION_LOCK] = asyncio.Lock()
    app.router.add_get("/v1/va", voice_agent)
    app.router.add_get("/v1/listVirtualAgents", list_virtual_agents)
    app.router.add_get("/health", health)
    return app


def listener_ssl(config: AdapterConfig) -> ssl.SSLContext | None:
    """Build the server TLS context, or return None for explicit local mode."""
    if not config.tls_cert:
        return None
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(config.tls_cert, config.tls_key)
    return context
