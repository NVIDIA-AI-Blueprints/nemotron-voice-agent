# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mount the live routes on the server's FastAPI app.

``src/server.py`` decides whether Live is served and for which example (the registry's ``live:`` block), then calls
:func:`mount_live` with that example's hooks, each of which may be a coroutine function. ``mount_live(app)`` with no
factory serves no live route.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from loguru import logger

from gateways.live.extras import Describe, add_extras
from gateways.live.gateway import LiveGateway, LiveSettings, create_live_router
from gateways.live.gateway.gateway import IceServers
from gateways.live.ports import AdapterFactory, CoreHook


def add_not_found(app: FastAPI) -> None:
    """Add a last route under ``/v1/live``: a path no live route serves answers 404, never the web app's page."""

    async def not_found(path: str) -> None:
        raise HTTPException(404, "Not Found")

    app.add_api_route(
        "/v1/live/{path:path}",
        not_found,
        methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )


def mount_live(
    app: FastAPI,
    *,
    factory: AdapterFactory | None = None,
    core: CoreHook | None = None,
    describe: Describe | None = None,
    settings: LiveSettings | None = None,
    ice_servers: IceServers | None = None,
) -> LiveGateway | None:
    """Add the live routes to ``app``; when the app shuts down, close their sessions, then the one core's client.

    This is additive: it registers routes under ``/v1/live`` and wraps the app's lifespan; it changes nothing else.
    A server with several uvicorn workers serves no live route, since a session lives in one process; either way, a
    last route answers 404 for any other path under ``/v1/live``.

    Args:
        app: The server's app.
        factory: Builds each session's parts (its frontend adapter, and a WebRTC session's peer connection);
            without it, no live route is served.
        core: Builds the one delegation core client of the gateway, on first use (``responses`` sessions and delegate
            rounds); without it they are refused.
        describe: What ``GET /v1/live/info`` adds about the example.
        settings: The operator settings; when omitted, the defaults, with the token and the worker count from the
            environment.
        ice_servers: The ICE servers, TURN credentials included, of each WebRTC session's peer, for its request.

    Returns:
        The gateway, or ``None`` when no live route is served.

    Raises:
        ValueError: ``UVICORN_WORKERS`` is not a whole number of at least 1.
    """
    if factory is None:
        add_not_found(app)
        return None
    settings = settings or LiveSettings.from_env()
    if settings.workers > 1:
        logger.error(
            f"Live session routes are not served with UVICORN_WORKERS={settings.workers}: a live session lives in one "
            "server process, so a sideband attach or a history request could reach a worker without it. Run one worker."
        )
        add_not_found(app)
        return None
    router, gateway = create_live_router(factory, settings, core=core, ice_servers=ice_servers)
    app.include_router(router)
    if not settings.api_token:
        logger.warning(
            "LIVE_API_TOKEN is not set: anyone who can reach this server can start live sessions and run backend "
            "rounds through POST /v1/live/delegate (which use its model and speech services), and read "
            "GET /v1/live/info. Set it on any deployment that others can reach."
        )
    add_extras(app, gateway, describe)
    add_not_found(app)

    original = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        async with original(application) as state:
            yield state
            await gateway.close_all()

    app.router.lifespan_context = lifespan
    return gateway
