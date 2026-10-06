# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""FastAPI app: ``WS /v1/realtime`` (OpenAI Realtime GA) and ``GET /health``.

Run from the repository root (or ``/app`` in the container)::

    PYTHONPATH=src uv run python -m examples.frontend_backend_verdict.voice.server \
        --config src/prototypes/voice_frontend_backend_agent/config/voice_agent.yaml

``--stub-speech`` replaces Riva ASR/TTS with deterministic stand-ins and
``--stub-agent scripted`` replaces the LLM agent with a script; together they
serve the tau3 gates and smoke tests with no GPU, speech endpoint or LLM.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger

from examples.frontend_backend_verdict.text.llm import OpenAIChatClient
from examples.frontend_backend_verdict.voice.agent.filler import FillerLog
from examples.frontend_backend_verdict.voice.agent.port import AgentPort
from examples.frontend_backend_verdict.voice.agent.runner import (
    AgentClients,
    TextAgentRunner,
    frontend_verdict_settings,
)
from examples.frontend_backend_verdict.voice.agent.scripted import ScriptedAgentPort
from examples.frontend_backend_verdict.voice.agent.sinks import EventLog, SessionRoutingSink
from examples.frontend_backend_verdict.voice.config import (
    DEFAULT_CONFIG_PATH,
    VoiceConfig,
    load_voice_config,
    prompt_context,
)
from examples.frontend_backend_verdict.voice.engine.session import RealtimeSession
from examples.frontend_backend_verdict.voice.engine.turn_api import TurnManagerFactory
from examples.frontend_backend_verdict.voice.speech.factory import build_speech_services
from examples.frontend_backend_verdict.voice.speech.ports import SpeechServices
from examples.frontend_backend_verdict.voice.wire import server_events as ev
from examples.frontend_backend_verdict.voice.wire.writer import CallbackTransport

# ``?x_nvidia_filler=1`` on the WebSocket URL: send unspoken filler as ``x_nvidia.filler``
# events. Off unless asked for, so a tau2 session sees the standard event stream only.
SHOW_FILLER_PARAM = "x_nvidia_filler"

# Browser microphone client served at "/" (it connects back to server.path on the same origin).
WEB_PAGE = Path(__file__).resolve().parent / "web" / "index.html"


@dataclass(slots=True)
class ServerOptions:
    """Runtime switches that are not part of the YAML (test and gate stand-ins)."""

    stub_speech: bool = False
    stub_agent: str = ""
    stub_agent_script: str = ""
    tls: bool = False


@dataclass(slots=True)
class SessionHooks:
    """Replace the text agent with another turn manager (additive seam for sibling prototypes).

    With hooks the lifespan builds no text-agent LLM clients, sessions get
    ``turn_manager_factory`` instead of an agent, and ``session.update`` is transactional.
    """

    turn_manager_factory: TurnManagerFactory
    startup: Callable[[], Awaitable[None]] | None = None
    shutdown: Callable[[], Awaitable[None]] | None = None
    health_extra: Callable[[], dict[str, Any]] | None = None
    title: str = "Voice agent (prototype)"


@dataclass(slots=True)
class _AppState:
    config: VoiceConfig
    options: ServerOptions
    services: SpeechServices | None = None
    clients: AgentClients | None = None
    routing_sink: SessionRoutingSink = field(default_factory=SessionRoutingSink)
    filler_log: FillerLog = field(default_factory=FillerLog)
    ready: bool = False
    sessions: int = 0
    total_sessions: int = 0
    startup_error: str = ""


def build_clients(config: VoiceConfig) -> AgentClients:
    """One OpenAI-compatible client per LLM role, shared by all sessions."""
    agent = config.agent
    frontend = OpenAIChatClient(agent.frontend.llm, accounting=agent.accounting) if agent.frontend_enabled else None
    return AgentClients(backend=OpenAIChatClient(agent.backend.llm, accounting=agent.accounting), frontend=frontend)


def _agent_factory(state: _AppState) -> Any:
    config, options = state.config, state.options
    barge_in = frontend_verdict_settings(config)

    def factory(session_id: str) -> AgentPort:
        if options.stub_agent == "scripted":
            agent: AgentPort = (
                ScriptedAgentPort.from_file(options.stub_agent_script)
                if options.stub_agent_script
                else ScriptedAgentPort(backend_only=not config.agent.frontend_enabled)
            )
            return agent
        assert state.clients is not None  # noqa: S101 - built in lifespan
        return TextAgentRunner(
            base_config=config.agent,
            tools_config=config.tools,
            instructions_config=config.instructions,
            clients=state.clients,
            sink=state.routing_sink,
            session_id=session_id,
            seed_greeting=config.protocol.seed_history_with_client_greeting,
            normalization=config.normalization,
            barge_in=barge_in,
            prompt_context=prompt_context(config),
        )

    return factory


def _bearer_ok(websocket: WebSocket, config: VoiceConfig) -> bool:
    if not config.server.require_bearer:
        return True
    header = websocket.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    return bool(config.server.bearer_token) and token == config.server.bearer_token


def build_app(
    config: VoiceConfig,
    *,
    options: ServerOptions | None = None,
    services: SpeechServices | None = None,
    clients: AgentClients | None = None,
    session_hooks: SessionHooks | None = None,
) -> FastAPI:
    """Create the app; ``services``/``clients`` may be injected (tests), ``session_hooks`` swap the agent."""
    state = _AppState(config=config, options=options or ServerOptions(), services=services, clients=clients)
    hooks = session_hooks
    event_log = EventLog(config.logging.event_log, redact_content=config.logging.redact_content)
    state.routing_sink = SessionRoutingSink(event_log)
    state.filler_log = FillerLog(config.filler.log_path, event_log=event_log)
    factory = _agent_factory(state) if hooks is None else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        for warning in config.warnings:
            logger.warning(warning)
        if state.services is None:
            state.services = build_speech_services(config, stub=state.options.stub_speech)
        if hooks is None and state.clients is None and state.options.stub_agent != "scripted":
            state.clients = build_clients(config)
        if hooks is not None and hooks.startup is not None:
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
        if hooks is not None:
            ws_scheme, http_scheme = ("wss", "https") if state.options.tls else ("ws", "http")
            logger.info(f"browser mic page: {http_scheme}://{config.server.host}:{config.server.port}/")
            logger.info(
                f"{hooks.title} ready: {ws_scheme}://{config.server.host}:{config.server.port}{config.server.path}"
            )
            try:
                yield
            finally:
                state.ready = False
                if hooks.shutdown is not None:
                    with contextlib.suppress(Exception):
                        await hooks.shutdown()
                if state.services is not None:
                    await state.services.aclose()
            return
        mode = "backend_only" if not config.agent.frontend_enabled else "frontend_backend"
        ws_scheme, http_scheme = ("wss", "https") if state.options.tls else ("ws", "http")
        logger.info(f"browser mic page: {http_scheme}://{config.server.host}:{config.server.port}/")
        logger.info(
            f"voice agent ready: {ws_scheme}://{config.server.host}:{config.server.port}{config.server.path} "
            f"(mode={mode}, filler={config.filler.mode}, tools={config.tools.source}, "
            f"barge_in={config.barge_in.while_thinking}, "
            f"stub_speech={state.options.stub_speech}, stub_agent={state.options.stub_agent or '-'})"
        )
        try:
            yield
        finally:
            state.ready = False
            if state.services is not None:
                await state.services.aclose()

    app = FastAPI(title=hooks.title if hooks else "Voice Frontend/Backend Agent (prototype)", lifespan=lifespan)
    app.state.voice = state

    @app.get("/health")
    async def health() -> JSONResponse:
        body = {
            "status": "ok" if state.ready else "starting",
            "sessions": state.sessions,
            "max_sessions": config.server.max_sessions,
            "total_sessions": state.total_sessions,
        }
        if state.startup_error:
            body["error"] = state.startup_error
        if hooks is not None and hooks.health_extra is not None:
            with contextlib.suppress(Exception):
                body.update(hooks.health_extra())
        return JSONResponse(body, status_code=200 if state.ready else 503)

    @app.get("/", include_in_schema=False)
    async def web_page() -> FileResponse:
        return FileResponse(WEB_PAGE, media_type="text/html")

    @app.websocket(config.server.path)
    async def realtime(websocket: WebSocket) -> None:
        if not _bearer_ok(websocket, config):
            await websocket.close(code=1008, reason="invalid bearer token")
            return
        offered = [value.strip() for value in websocket.headers.get("sec-websocket-protocol", "").split(",")]
        await websocket.accept(subprotocol="realtime" if "realtime" in offered else None)
        if not state.ready or state.sessions >= config.server.max_sessions:
            reason = "server is starting" if not state.ready else "server at capacity (server.max_sessions)"
            await websocket.send_text(json.dumps(ev.error(reason, code="server_busy")))
            await websocket.close(code=1013, reason=reason)
            return
        state.sessions += 1
        state.total_sessions += 1
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
            agent_factory=factory,
            turn_manager_factory=hooks.turn_manager_factory if hooks is not None else None,
            routing_sink=state.routing_sink,
            filler_log=state.filler_log,
            model=websocket.query_params.get("model", ""),
            show_silent_filler=websocket.query_params.get(SHOW_FILLER_PARAM, "").lower() in ("1", "true"),
        )
        try:
            await session.run(receive)
        except Exception:  # noqa: BLE001 - log and drop this connection only
            logger.exception(f"[{session.session_id}] session crashed")
        finally:
            state.sessions -= 1
            with contextlib.suppress(Exception):
                await websocket.close()

    return app


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="voice_agent.yaml or a profile")
    parser.add_argument("--host", default=None, help="override server.host")
    parser.add_argument("--port", type=int, default=None, help="override server.port")
    parser.add_argument(
        "--tls",
        action="store_true",
        help="serve https/wss (browsers allow the microphone only on https or localhost); "
        "self-signed certificate in .certs/ unless --tls-cert/--tls-key are given",
    )
    parser.add_argument("--tls-cert", default="", help="PEM certificate for --tls")
    parser.add_argument("--tls-key", default="", help="PEM private key for --tls")
    parser.add_argument("--stub-speech", action="store_true", help="energy VAD + stub ASR + tone TTS (no GPU)")
    parser.add_argument("--stub-agent", choices=["scripted"], default="", help="replace the LLM agent")
    parser.add_argument("--stub-agent-script", default="", help="JSON script for --stub-agent scripted")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load configuration and serve."""
    import uvicorn

    try:
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env", override=False)
    except ImportError:  # pragma: no cover - python-dotenv is a core dependency
        pass
    args = _parse_args(argv)
    config = load_voice_config(args.config)
    logger.remove()
    logger.add(sys.stderr, level=config.logging.level)
    host = args.host or config.server.host
    port = args.port if args.port is not None else config.server.port
    config = replace(config, server=replace(config.server, host=host, port=port))
    options = ServerOptions(
        stub_speech=args.stub_speech,
        stub_agent=args.stub_agent,
        stub_agent_script=args.stub_agent_script,
        tls=args.tls,
    )
    ssl_kwargs = _tls_files(args) if args.tls else {}
    app = build_app(config, options=options)
    logger.info(f"loaded {', '.join(str(path) for path in config.source_files)}")
    logger.info(f"ASR {config.asr.endpoint.describe()}")
    logger.info(f"TTS {config.tts.endpoint.describe()}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=config.logging.level.lower(),
        ws_max_size=16 * 1024 * 1024,
        ws_ping_interval=config.server.ws_ping_interval_s or None,
        ws_ping_timeout=config.server.ws_ping_timeout_s,
        **ssl_kwargs,
    )


def _tls_files(args: argparse.Namespace) -> dict[str, str]:
    """Certificate and key for ``--tls``: the given pair, or the stock app's self-signed one."""
    if bool(args.tls_cert) != bool(args.tls_key):
        raise SystemExit("--tls-cert and --tls-key must be given together")
    if args.tls_cert:
        return {"ssl_certfile": args.tls_cert, "ssl_keyfile": args.tls_key}
    from utils import ensure_self_signed_cert  # src/utils.py, as used by src/server.py

    cert, key = ensure_self_signed_cert(Path.cwd() / ".certs")
    return {"ssl_certfile": cert, "ssl_keyfile": key}


if __name__ == "__main__":
    main()
