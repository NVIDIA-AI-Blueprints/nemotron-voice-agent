# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The Pipecat adapter's FastAPI app: ``WS /v1/realtime`` (OpenAI Realtime GA), ``GET /health``, ``GET /``.

WebSocket + OpenAI Realtime only (no WebRTC, no RTVI). Turn taking, VAD, interruptions
and audio pacing are Pipecat's; the shared delegation runtime (frontend decider,
Hermes backend through the gateway, tool relay, normalization) is the same as the τ³
adapter's. Start it through ``server.py``::

    PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.server \
        --config src/examples/frontend_delegation_hermes/config/profiles/pipecat_demo.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger

from examples.frontend_delegation_hermes.adapters.pipecat.settings import (
    SETTINGS_ENV,
    PipecatHeadSettings,
    load_pipecat_settings,
)
from examples.frontend_delegation_hermes.core.config import DelegationConfig, load_delegation_config
from examples.frontend_delegation_hermes.core.engine.filler import FillerLog
from examples.frontend_delegation_hermes.core.engine.sinks import EventLog, SessionRoutingSink
from examples.frontend_delegation_hermes.core.frontend.llm import ChatModel
from examples.frontend_delegation_hermes.core.runtime import DelegationRuntime
from examples.frontend_delegation_hermes.core.server_common import (
    WEB_PAGE,
    SessionSlots,
    bearer_ok,
    health_body,
    realtime_subprotocol,
    show_silent_filler,
    tls_files,
)
from examples.frontend_delegation_hermes.core.wire import server_events as ev

DEFAULT_PROFILE = Path(__file__).resolve().parents[2] / "config" / "profiles" / "pipecat_demo.yaml"
TLS_ENV = "FDH_PIPECAT_TLS"


@dataclass(slots=True)
class _AppState:
    slots: SessionSlots
    routing_sink: SessionRoutingSink
    filler_log: FillerLog
    ready: bool = False
    startup_error: str = ""


def build_pipecat_app(
    config: DelegationConfig,
    settings: PipecatHeadSettings,
    *,
    chat_model: ChatModel | None = None,
    speech_factory: Callable[[Any], Any] | None = None,
) -> FastAPI:
    """The Pipecat adapter's app (``speech_factory`` builds stub speech per session in tests)."""
    from examples.frontend_delegation_hermes.adapters.pipecat.session import PipecatSession  # noqa: PLC0415

    voice = config.voice
    runtime = DelegationRuntime(config, chat_model=chat_model)
    event_log = EventLog(voice.logging.event_log, redact_content=voice.logging.redact_content)
    state = _AppState(
        slots=SessionSlots(max_sessions=voice.server.max_sessions),
        routing_sink=SessionRoutingSink(event_log),
        filler_log=FillerLog(voice.filler.log_path, event_log=event_log),
    )

    def health_extra() -> dict[str, Any]:
        return {**runtime.health(), "adapter": "pipecat", "pipecat": settings.summary()}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        for warning in voice.warnings:
            logger.warning(warning)
        try:
            await runtime.startup()
        except Exception as exc:
            state.startup_error = str(exc)
            logger.error(f"startup failed: {exc}")
            raise
        state.ready = True
        logger.info(f"Voice Frontend Delegation to Hermes (Pipecat adapter) ready on {voice.server.path}")
        try:
            yield
        finally:
            state.ready = False
            with contextlib.suppress(Exception):
                await runtime.shutdown()

    app = FastAPI(title="Voice Frontend Delegation to Hermes (Pipecat adapter)", lifespan=lifespan)
    app.state.delegation = runtime
    app.state.pipecat = state

    @app.get("/health")
    async def health() -> JSONResponse:
        body, status = health_body(
            ready=state.ready, slots=state.slots, startup_error=state.startup_error, extra=health_extra
        )
        return JSONResponse(body, status_code=status)

    @app.get("/", include_in_schema=False)
    async def web_page() -> FileResponse:
        return FileResponse(WEB_PAGE, media_type="text/html")

    @app.websocket(voice.server.path)
    async def realtime(websocket: WebSocket) -> None:
        if not bearer_ok(websocket.headers, voice):
            await websocket.close(code=1008, reason="invalid bearer token")
            return
        await websocket.accept(subprotocol=realtime_subprotocol(websocket.headers))
        reason = state.slots.refusal(state.ready)
        if reason:
            await websocket.send_text(json.dumps(ev.error(reason, code="server_busy")))
            await websocket.close(code=1013, reason=reason)
            return
        state.slots.acquire()
        session = None
        try:
            session = PipecatSession(
                config=voice,
                settings=settings,
                websocket=websocket,
                routing_sink=state.routing_sink,
                filler_log=state.filler_log,
                turn_manager_factory=runtime.turn_manager,
                model=websocket.query_params.get("model", ""),
                show_silent_filler=show_silent_filler(websocket.query_params),
                speech=speech_factory(None) if speech_factory is not None else None,
            )
            await session.run()
        except Exception:  # noqa: BLE001 - log and drop this connection only
            logger.exception(f"[{session.session_id if session else '-'}] session crashed")
        finally:
            state.slots.release()
            with contextlib.suppress(Exception):
                await websocket.close()

    return app


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_PROFILE), help="a delegation profile (pipecat_*.yaml)")
    parser.add_argument("--pipecat-settings", default=None, help=f"adapters/pipecat*.yaml (or ${SETTINGS_ENV})")
    parser.add_argument("--host", default=None, help="override server.host")
    parser.add_argument("--port", type=int, default=None, help="override server.port")
    parser.add_argument("--max-sessions", type=int, default=None, help="override server.max_sessions")
    parser.add_argument("--backend-url", default=None, help="override backend.url (the gateway WebSocket)")
    parser.add_argument("--tls", action="store_true", help=f"serve https/wss (or {TLS_ENV}=true)")
    parser.add_argument("--tls-cert", default=os.environ.get("FDH_PIPECAT_TLS_CERT", ""), help="PEM certificate")
    parser.add_argument("--tls-key", default=os.environ.get("FDH_PIPECAT_TLS_KEY", ""), help="PEM private key")
    parser.add_argument(
        "--stub-backend", action="store_true", help="in-process gateway with a fake backend and the rule-based frontend"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load configuration and serve."""
    import uvicorn  # noqa: PLC0415

    with contextlib.suppress(ImportError):
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv(Path.cwd() / ".env", override=False)
    args = _parse_args(argv)
    config = load_delegation_config(args.config)
    server = config.voice.server
    server = replace(
        server,
        host=args.host or server.host,
        port=args.port if args.port is not None else server.port,
        max_sessions=args.max_sessions if args.max_sessions is not None else server.max_sessions,
    )
    config = replace(config, voice=replace(config.voice, server=server))
    if args.backend_url:
        config = replace(config, backend=replace(config.backend, url=args.backend_url))
    if args.stub_backend:
        config = replace(
            config,
            backend=replace(config.backend, link="in_process_fake", verbalizer_mode="template"),
            frontend=replace(config.frontend, decider="scripted"),
        )
    settings = load_pipecat_settings(args.pipecat_settings)
    logger.remove()
    logger.add(sys.stderr, level=config.voice.logging.level)
    tls = args.tls or os.environ.get(TLS_ENV, "").lower() in ("1", "true", "yes")
    ssl_kwargs = tls_files(args.tls_cert, args.tls_key) if tls else {}
    app = build_pipecat_app(config, settings)
    logger.info(f"loaded {', '.join(str(path) for path in config.source_files)} (config hash {config.config_hash})")
    logger.info(f"Pipecat adapter settings {settings.summary()}")
    logger.info(
        f"frontend={config.frontend.decider}:{config.frontend.llm.model} backend={config.backend.link}:"
        f"{config.backend.url} tools={config.tools.executor}"
    )
    uvicorn.run(
        app,
        host=server.host,
        port=server.port,
        log_level=config.voice.logging.level.lower(),
        ws_max_size=16 * 1024 * 1024,
        ws_ping_interval=None,  # τ² sends no pings and pauses for seconds
        ws_ping_timeout=None,
        **ssl_kwargs,
    )
