# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The routes of the live session protocol.

  POST      /v1/live/sessions                 - WebRTC session creation (JSON SDP offer and answer, 201)
  WebSocket /v1/live/sessions                 - Primary WebSocket (``session.start`` first, base64 audio events)
  WebSocket /v1/live/sessions/{id}/attach     - Sideband observer and trusted control channel

An example supplies an engine factory that builds the voice pipeline for each session. Nothing here knows about
models or prompts.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from live.events import error_event
from live.gateway.gateway import EngineFactory, LiveGateway
from live.gateway.settings import LiveSettings
from live.protocol import ProtocolError, SessionConfig

STATUS_BY_CODE = {"session_not_found": 404, "rate_limit_exceeded": 429, "permission_denied": 403, "server_error": 503}


def create_live_router(
    engine_factory: EngineFactory, settings: LiveSettings | None = None
) -> tuple[APIRouter, LiveGateway]:
    """Build the live-session router and its gateway (use the gateway's ``close_all`` on shutdown)."""
    gateway = LiveGateway(engine_factory, settings)
    router = APIRouter()

    @router.post("/v1/live/sessions")
    async def create_session(request: Request):
        data = await gateway.body(request)
        transport = data.get("transport", {})
        if not isinstance(transport, dict) or transport.get("type") != "webrtc":
            raise HTTPException(400, "HTTP session creation requires WebRTC")
        try:
            config = SessionConfig.model_validate(data.get("session"))
            return await gateway.create_webrtc(config, transport.get("sdp"))
        except ProtocolError as exc:
            return JSONResponse(error_event(exc), status_code=STATUS_BY_CODE.get(exc.code, 400))
        except ValidationError as exc:
            return JSONResponse(error_event(ProtocolError(str(exc)[:1000])), status_code=400)

    @router.websocket("/v1/live/sessions")
    async def primary(ws: WebSocket):
        await gateway.run_primary_websocket(ws)

    @router.websocket("/v1/live/sessions/{session_id}/attach")
    async def attach(ws: WebSocket, session_id: str):
        await gateway.run_sideband(ws, session_id)

    return router, gateway
