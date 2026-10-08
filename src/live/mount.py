# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mount the live routes on the server's FastAPI app."""

from __future__ import annotations

import importlib
import inspect
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from loguru import logger

from live.gateway import EngineFactory, LiveGateway, create_live_router
from live.protocol import LIVE_VOICES, SessionConfig

DEFAULT_ENGINE_FACTORY = "examples.frontend_backend_live.engine.factory:create_engine"
LIVE_EXAMPLE_KEY = "frontend-backend-live"


def live_enabled() -> bool:
    """Return whether the live routes should be served.

    They are open to anyone who can reach the server unless ``LIVE_API_TOKEN`` is set, so they are not added to a
    deployment of another example. ``LIVE_ENABLED=true`` or ``false`` decides explicitly; otherwise they are served
    when a token is set or when the deployment is pinned to the example that uses them.
    """
    explicit = os.getenv("LIVE_ENABLED", "").strip().lower()
    if explicit:
        return explicit in {"1", "true", "yes", "on"}
    if os.getenv("LIVE_API_TOKEN", "").strip():
        return True
    import examples_registry

    return examples_registry.visible_example_keys() == (LIVE_EXAMPLE_KEY,)


def _split_spec(spec: str) -> tuple[str, str]:
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"LIVE_ENGINE_FACTORY must look like 'package.module:function' (got {spec!r})")
    return module_name, attribute


def resolve_engine_factory(spec: str | None = None) -> EngineFactory:
    """Return the engine factory named by ``LIVE_ENGINE_FACTORY`` (``module:function``).

    The module is imported on the first session, so the server starts without loading the example.
    """
    module_name, attribute = _split_spec(spec or os.getenv("LIVE_ENGINE_FACTORY", DEFAULT_ENGINE_FACTORY))

    async def factory(config: SessionConfig, transport: str, connection):
        return await getattr(importlib.import_module(module_name), attribute)(config, transport, connection)

    return factory


def engine_hook(name: str):
    """Return the optional function ``name`` from the module named by ``LIVE_ENGINE_FACTORY``, or ``None``."""
    module_name, _ = _split_spec(os.getenv("LIVE_ENGINE_FACTORY", DEFAULT_ENGINE_FACTORY))
    return getattr(importlib.import_module(module_name), name, None)


async def call_hook(hook, *args):
    """Call a hook that may be a plain function or a coroutine function."""
    result = hook(*args)
    return await result if inspect.isawaitable(result) else result


async def describe_engine() -> dict:
    """Return what the engine's module says about itself: its optional ``describe()`` function, else ``{}``.

    The module is the one named by ``LIVE_ENGINE_FACTORY``. ``describe`` returns plain JSON (models, prompts, guards)
    for clients such as the live console; it must not include credentials.
    """
    describe = engine_hook("describe")
    return {} if describe is None else await call_hook(describe)


def mount_live(app: FastAPI, engine_factory: EngineFactory | None = None) -> LiveGateway | None:
    """Add the live routes to ``app`` and close their sessions when the app shuts down.

    This is additive: it registers routes under ``/v1/live`` and wraps the app's lifespan; it changes nothing else.
    With no ``engine_factory`` given, nothing is added unless :func:`live_enabled`; the return value is then ``None``.
    """
    if engine_factory is None and not live_enabled():
        logger.info("Live session routes are not served (set LIVE_ENABLED=true, or LIVE_API_TOKEN, to serve them).")
        return None
    router, gateway = create_live_router(engine_factory or resolve_engine_factory())
    app.include_router(router)
    if not gateway.settings.api_token:
        logger.warning(
            "LIVE_API_TOKEN is not set: anyone who can reach this server can start live sessions "
            "(which use its model and speech services) and read GET /v1/live/info. "
            "Set it unless access is already limited."
        )

    @app.get("/v1/live/info")
    async def info(request: Request) -> dict:
        """Describe the server for clients: the protocol's voices, plus the engine's models, prompts and guards."""
        gateway.authorize(request)
        engine = await describe_engine() if engine_factory is None else {}
        return {"voices": sorted(LIVE_VOICES), "max_sessions": gateway.settings.max_sessions, **engine}

    @app.post("/v1/live/delegate")
    async def delegate(request: Request) -> dict:
        """Run one backend round for a client that owns its delegation, when the engine supports it.

        The client sends the whole conversation as ``input`` (and optionally ``tools``, ``model``, ``instructions``),
        runs the function calls in the reply, and sends the next round with their outputs.
        """
        data = await gateway.body(request)
        hook = engine_hook("delegate") if engine_factory is None else None
        if hook is None:
            raise HTTPException(404, "This engine does not run delegated work for clients")
        try:
            return await call_hook(hook, data)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            logger.exception(f"Client delegation round failed: {exc}")
            raise HTTPException(502, "The backend could not complete the round") from exc

    original = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        async with original(application) as state:
            yield state
            await gateway.close_all()

    app.router.lifespan_context = lifespan
    return gateway
