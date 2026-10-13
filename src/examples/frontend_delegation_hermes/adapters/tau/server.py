# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The τ³ head's FastAPI app: ``WS /v1/realtime`` (OpenAI Realtime GA), ``GET /health`` and ``GET /``.

Every session gets the turn manager from :class:`SessionHooks` (the delegation
runtime); :mod:`examples.frontend_delegation_hermes.adapters.tau.app` builds the hooks
and runs the server. ``ServerOptions.stub_speech`` swaps Riva ASR/TTS for
deterministic stand-ins (energy VAD, stub ASR, tone TTS) for the no-GPU gates.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger

from examples.frontend_delegation_hermes.adapters.tau.session import RealtimeSession
from examples.frontend_delegation_hermes.adapters.tau.speech.factory import build_speech_services
from examples.frontend_delegation_hermes.adapters.tau.speech.ports import SpeechServices
from examples.frontend_delegation_hermes.core.engine.filler import FillerLog
from examples.frontend_delegation_hermes.core.engine.sinks import EventLog, SessionRoutingSink
from examples.frontend_delegation_hermes.core.head_api import TurnManagerFactory
from examples.frontend_delegation_hermes.core.server_common import (
    WEB_PAGE,
    SessionSlots,
    bearer_ok,
    health_body,
    realtime_subprotocol,
    show_silent_filler,
)
from examples.frontend_delegation_hermes.core.voice_config import VoiceConfig
from examples.frontend_delegation_hermes.core.wire import server_events as ev
from examples.frontend_delegation_hermes.core.wire.writer import CallbackTransport


@dataclass(slots=True)
class ServerOptions:
    """Runtime switches that are not part of the YAML (test and gate stand-ins)."""

    stub_speech: bool = False
    tls: bool = False


@dataclass(slots=True)
class SessionHooks:
    """The turn manager every session uses, plus process start/stop and ``/health`` extras."""

    turn_manager_factory: TurnManagerFactory
    startup: Callable[[], Awaitable[None]] | None = None
    shutdown: Callable[[], Awaitable[None]] | None = None
    health_extra: Callable[[], dict[str, Any]] | None = None
    title: str = "Voice agent (τ³ head)"


@dataclass(slots=True)
class _AppState:
    config: VoiceConfig
    options: ServerOptions
    slots: SessionSlots
    services: SpeechServices | None = None
    routing_sink: SessionRoutingSink = field(default_factory=SessionRoutingSink)
    filler_log: FillerLog = field(default_factory=FillerLog)
    ready: bool = False
    startup_error: str = ""


def build_app(
    config: VoiceConfig,
    *,
    session_hooks: SessionHooks,
    options: ServerOptions | None = None,
    services: SpeechServices | None = None,
) -> FastAPI:
    """Create the app; ``services`` may be injected (tests)."""
    hooks = session_hooks
    state = _AppState(
        config=config,
        options=options or ServerOptions(),
        slots=SessionSlots(max_sessions=config.server.max_sessions),
        services=services,
    )
    event_log = EventLog(config.logging.event_log, redact_content=config.logging.redact_content)
    state.routing_sink = SessionRoutingSink(event_log)
    state.filler_log = FillerLog(config.filler.log_path, event_log=event_log)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        for warning in config.warnings:
            logger.warning(warning)
        if state.services is None:
            state.services = build_speech_services(config, stub=state.options.stub_speech)
        if hooks.startup is not None:
            try:
                await hooks.startup()
            except Exception as exc:
                state.startup_error = str(exc)
                logger.error(f"startup hook failed: {exc}")
                raise
        if config.server.warmup and not state.options.stub_speech:
            logger.info("warming up ASR and TTS ...")
            try:
                await state.services.warmup()
            except Exception as exc:
                state.startup_error = str(exc)
                logger.error(f"speech warm-up failed: {exc}")
                raise
        state.ready = True
        ws_scheme, http_scheme = ("wss", "https") if state.options.tls else ("ws", "http")
        logger.info(f"browser mic page: {http_scheme}://{config.server.host}:{config.server.port}/")
        logger.info(f"{hooks.title} ready: {ws_scheme}://{config.server.host}:{config.server.port}{config.server.path}")
        try:
            yield
        finally:
            state.ready = False
            if hooks.shutdown is not None:
                with contextlib.suppress(Exception):
                    await hooks.shutdown()
            if state.services is not None:
                await state.services.aclose()

    app = FastAPI(title=hooks.title, lifespan=lifespan)
    app.state.voice = state

    @app.get("/health")
    async def health() -> JSONResponse:
        body, status = health_body(
            ready=state.ready, slots=state.slots, startup_error=state.startup_error, extra=hooks.health_extra
        )
        return JSONResponse(body, status_code=status)

    @app.get("/", include_in_schema=False)
    async def web_page() -> FileResponse:
        return FileResponse(WEB_PAGE, media_type="text/html")

    @app.websocket(config.server.path)
    async def realtime(websocket: WebSocket) -> None:
        if not bearer_ok(websocket.headers, config):
            await websocket.close(code=1008, reason="invalid bearer token")
            return
        await websocket.accept(subprotocol=realtime_subprotocol(websocket.headers))
        reason = state.slots.refusal(state.ready)
        if reason:
            await websocket.send_text(json.dumps(ev.error(reason, code="server_busy")))
            await websocket.close(code=1013, reason=reason)
            return
        state.slots.acquire()
        assert state.services is not None  # noqa: S101 - built in lifespan

        async def receive() -> str | None:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return None
            return message.get("text") if message.get("text") is not None else message.get("bytes")

        session = RealtimeSession(
            config=config,
            transport=CallbackTransport(websocket.send_text),
            services=state.services,
            turn_manager_factory=hooks.turn_manager_factory,
            routing_sink=state.routing_sink,
            filler_log=state.filler_log,
            model=websocket.query_params.get("model", ""),
            show_silent_filler=show_silent_filler(websocket.query_params),
        )
        try:
            await session.run(receive)
        except Exception:  # noqa: BLE001 - log and drop this connection only
            logger.exception(f"[{session.session_id}] session crashed")
        finally:
            state.slots.release()
            with contextlib.suppress(Exception):
                await websocket.close()

    return app
