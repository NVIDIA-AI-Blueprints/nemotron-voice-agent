# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The routes of the live session protocol.

  POST      /v1/live/sessions                 - WebRTC session creation (JSON SDP offer and answer, 201)
  WebSocket /v1/live/sessions                 - Primary WebSocket (``session.start`` first, base64 audio events)
  WebSocket /v1/live/sessions/{id}/attach     - Sideband observer and trusted control channel
  GET       /v1/live/sessions/{id}/history    - What the session's models hold now

A composition root supplies the factory that builds each session's frontend adapter and the hook that builds the one
delegation core client (``live.ports``). Nothing here knows about models or prompts.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from live.events import error_event
from live.gateway import webrtc, websocket
from live.gateway.gateway import IceServers, LiveGateway
from live.gateway.settings import LiveSettings
from live.ports import AdapterFactory, CoreHook
from live.protocol import ProtocolError, SessionConfig, validation_error

STATUS_BY_CODE = {"session_not_found": 404, "rate_limit_exceeded": 429, "permission_denied": 403, "server_error": 503}


def create_live_router(
    factory: AdapterFactory,
    settings: LiveSettings | None = None,
    *,
    core: CoreHook | None = None,
    ice_servers: IceServers | None = None,
) -> tuple[APIRouter, LiveGateway]:
    """Build the live-session router and its gateway (use the gateway's ``close_all`` on shutdown)."""
    gateway = LiveGateway(factory, settings, core=core, ice_servers=ice_servers)
    router = APIRouter()

    @router.post("/v1/live/sessions")
    async def create_session(request: Request):
        data = await gateway.body(request)
        transport = data.get("transport", {})
        if not isinstance(transport, dict) or transport.get("type") != "webrtc":
            raise HTTPException(400, "HTTP session creation requires WebRTC")
        try:
            config = SessionConfig.model_validate(data.get("session"))
            return await webrtc.create_webrtc_session(gateway, config, transport.get("sdp"), request)
        except ProtocolError as exc:
            return JSONResponse(error_event(exc), status_code=STATUS_BY_CODE.get(exc.code, 400))
        except ValidationError as exc:
            return JSONResponse(error_event(validation_error(exc, "session")), status_code=400)

    @router.websocket("/v1/live/sessions")
    async def primary(ws: WebSocket):
        await websocket.run_primary_websocket(gateway, ws)

    @router.websocket("/v1/live/sessions/{session_id}/attach")
    async def attach(ws: WebSocket, session_id: str):
        await websocket.run_sideband(gateway, ws, session_id)

    @router.get("/v1/live/sessions/{session_id}/history")
    async def history(request: Request, session_id: str) -> dict:
        gateway.authorize(request)
        session = gateway.sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "Unknown session")
        return await session.history()

    return router, gateway
