# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The τ³ head: the custom OpenAI Realtime (GA) server used for τ³ benchmarking.

It plugs the shared :class:`DelegationTurnManager` into the τ³ head's own server
(``adapters/tau/server.py``: audio clock driven by client audio, Silero VAD with the
profile's silence threshold, sent vs heard tracking, transcript before audio). The
backend runs in the gateway (``core/sidecar/gateway_server.py``), which starts one
Hermes worker process per session. Start it through ``tau_server.py``::

    PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.tau_server \
        --config src/examples/frontend_delegation_hermes/config/profiles/realtime_eval.yaml

``--stub-speech`` replaces Riva ASR/TTS; ``--stub-backend`` runs an in-process gateway
with a fake backend agent and the rule-based frontend (no GPU, no LLM, no Hermes).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI
from loguru import logger

from examples.frontend_delegation_hermes.adapters.tau.server import ServerOptions, SessionHooks, build_app
from examples.frontend_delegation_hermes.adapters.tau.speech.ports import SpeechServices
from examples.frontend_delegation_hermes.core.config import (
    DEFAULT_CONFIG_PATH,
    DelegationConfig,
    load_delegation_config,
)
from examples.frontend_delegation_hermes.core.frontend.llm import ChatModel
from examples.frontend_delegation_hermes.core.runtime import DelegationRuntime
from examples.frontend_delegation_hermes.core.server_common import tls_files


def build_delegation_app(
    config: DelegationConfig,
    *,
    options: ServerOptions | None = None,
    services: SpeechServices | None = None,
    chat_model: ChatModel | None = None,
) -> FastAPI:
    """The τ³ head's app with the delegation turn manager plugged in."""
    runtime = DelegationRuntime(config, chat_model=chat_model)
    hooks = SessionHooks(
        turn_manager_factory=runtime.turn_manager,
        startup=runtime.startup,
        shutdown=runtime.shutdown,
        health_extra=runtime.health,
        title="Voice Frontend Delegation to Hermes (τ³ head)",
    )
    app = build_app(config.voice, options=options, services=services, session_hooks=hooks)
    app.state.delegation = runtime
    return app


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="delegation_agent.yaml or a profile")
    parser.add_argument("--host", default=None, help="override server.host")
    parser.add_argument("--port", type=int, default=None, help="override server.port")
    parser.add_argument("--max-sessions", type=int, default=None, help="override server.max_sessions")
    parser.add_argument("--backend-url", default=None, help="override backend.url (the gateway WebSocket)")
    parser.add_argument("--tls", action="store_true", help="serve https/wss (the browser microphone needs it)")
    parser.add_argument("--tls-cert", default="", help="PEM certificate for --tls")
    parser.add_argument("--tls-key", default="", help="PEM private key for --tls")
    parser.add_argument("--stub-speech", action="store_true", help="energy VAD + stub ASR + tone TTS (no GPU)")
    parser.add_argument(
        "--stub-backend", action="store_true", help="in-process gateway with a fake backend and the rule-based frontend"
    )
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
    logger.remove()
    logger.add(sys.stderr, level=config.voice.logging.level)
    options = ServerOptions(stub_speech=args.stub_speech, tls=args.tls)
    ssl_kwargs = tls_files(args.tls_cert, args.tls_key) if args.tls else {}
    app = build_delegation_app(config, options=options)
    logger.info(f"loaded {', '.join(str(path) for path in config.source_files)} (config hash {config.config_hash})")
    logger.info(
        f"frontend={config.frontend.decider}:{config.frontend.llm.model} backend={config.backend.link}:"
        f"{config.backend.url} tools={config.tools.executor} delay={config.backend.delay_seconds:g}s "
        f"VAD silence={config.voice.turn_detection.silence_duration_ms}ms"
    )
    logger.info(f"features {config.features}")
    uvicorn.run(
        app,
        host=server.host,
        port=server.port,
        log_level=config.voice.logging.level.lower(),
        ws_max_size=16 * 1024 * 1024,
        ws_ping_interval=server.ws_ping_interval_s or None,
        ws_ping_timeout=server.ws_ping_timeout_s,
        **ssl_kwargs,
    )


if __name__ == "__main__":
    main()
