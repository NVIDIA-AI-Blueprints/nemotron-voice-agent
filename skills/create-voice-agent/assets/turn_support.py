"""Inject coturn TURN credentials into Pipecat runner WebRTC ICE (SSH + Mac browser).

Copy into project root. Pipecat 1.3+ compatible.
Do NOT add `from __future__ import annotations` — breaks FastAPI Request typing.
"""

import json
import os

from dotenv import load_dotenv
from loguru import logger
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

load_dotenv(override=True)

_STUN = {"urls": "stun:stun.l.google.com:19302"}


def _client_ice_servers() -> list[dict]:
    turn_url = os.getenv("TURN_URL", "").strip()
    turn_user = os.getenv("TURN_USERNAME", "").strip()
    turn_pass = os.getenv("TURN_PASSWORD", "").strip()
    servers = [_STUN]
    if turn_url and turn_user and turn_pass:
        servers.append({"urls": turn_url, "username": turn_user, "credential": turn_pass})
        logger.info("TURN ICE configured: {}", turn_url)
    else:
        logger.warning(
            "TURN_URL / TURN_USERNAME / TURN_PASSWORD incomplete — "
            "Mac browser over SSH may fail to connect audio without coturn"
        )
    return servers


def _install_start_middleware(app, ice_servers: list[dict]) -> None:
    @app.middleware("http")
    async def inject_turn_ice(request: Request, call_next):
        response: Response = await call_next(request)
        if request.method != "POST" or request.url.path.rstrip("/") != "/start":
            return response
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        if not body:
            return response
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )
        payload.setdefault("iceConfig", {})["iceServers"] = ice_servers
        return JSONResponse(content=payload, status_code=response.status_code)


def apply_turn_patches() -> None:
    """Patch Pipecat runner and WebRTC handler to inject coturn ICE servers."""
    from pipecat.runner import run as runner_run
    from pipecat.transports.smallwebrtc.connection import IceServer
    from pipecat.transports.smallwebrtc.request_handler import SmallWebRTCRequestHandler

    client_servers = _client_ice_servers()
    ice_servers = [
        IceServer(
            urls=s["urls"],
            username=s.get("username"),
            credential=s.get("credential"),
        )
        for s in client_servers
    ]

    original_init = SmallWebRTCRequestHandler.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["ice_servers"] = ice_servers
        original_init(self, *args, **kwargs)

    SmallWebRTCRequestHandler.__init__ = patched_init

    if getattr(runner_run, "_turn_support_patched", False):
        return

    original_setup = runner_run._setup_webrtc_routes

    def patched_setup(app, args, active_sessions):
        original_setup(app, args, active_sessions)
        _install_start_middleware(app, client_servers)

    runner_run._setup_webrtc_routes = patched_setup
    runner_run._turn_support_patched = True
    logger.info("Applied TURN patches ({} ICE servers)", len(client_servers))
