# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend gateway: ``WS /v1/backend`` (one per Realtime session) and ``GET /health``.

A connection beyond ``gateway.max_sessions`` gets ``error{code: "capacity"}``. A
session occupies its slot from accept until its worker has exited, so starting and
stopping workers count. Hermes is never imported here; workers run it.

``python -m examples.frontend_delegation_hermes.core.sidecar.gateway_server --config
src/examples/frontend_delegation_hermes/config/gateway.yaml``
"""

from __future__ import annotations

import argparse
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from examples.frontend_delegation_hermes.core.backend import protocol as proto
from examples.frontend_delegation_hermes.core.backend.writer import QueueWriter, WriterClosedError
from examples.frontend_delegation_hermes.core.sidecar import homes
from examples.frontend_delegation_hermes.core.sidecar.events import GatewayEventLog
from examples.frontend_delegation_hermes.core.sidecar.gateway_config import (
    DEFAULT_GATEWAY_CONFIG,
    GatewayConfig,
    load_gateway_config,
)
from examples.frontend_delegation_hermes.core.sidecar.session_runtime import SessionRuntime
from examples.frontend_delegation_hermes.core.sidecar.templates import BackendTemplates
from examples.frontend_delegation_hermes.core.sidecar.worker_pool import WorkerPool, resolve_python

logger = logging.getLogger("fdh.gateway")

PATH = "/v1/backend"


@dataclass(slots=True)
class GatewayState:
    """Counters shared by the endpoints."""

    config: GatewayConfig
    templates: BackendTemplates
    pool: WorkerPool
    event_log: GatewayEventLog
    sessions: int = 0
    total_sessions: int = 0
    refused: int = 0
    runtimes: set[SessionRuntime] = field(default_factory=set)


def build_gateway_app(config: GatewayConfig) -> FastAPI:
    """The gateway application."""
    templates = BackendTemplates(config.hermes.prompts_path, prompt_features=config.prompt_features)
    state = GatewayState(
        config=config,
        templates=templates,
        pool=WorkerPool(config, soul=templates.render("backend_soul")),
        event_log=GatewayEventLog(config.gateway.log),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if config.workers.mode == "process_per_session":
            removed = homes.clear_stale(config.workers.home_root)
            socket_dir = Path(config.workers.socket_dir)
            if socket_dir.is_dir():
                for stale in socket_dir.glob("*.sock"):
                    with contextlib.suppress(OSError):
                        stale.unlink()
            if removed:
                logger.info("removed %d stale worker homes", removed)
            state.pool.prewarm()
        state.event_log.write(
            "gateway_start",
            None,
            {
                "config": config.summary(),
                "files": list(config.files),
                "backend_features": dict(templates.features),
                "backend_catalog_sha256": templates.catalog_sha256,
            },
        )
        logger.info(
            "gateway ready: max_sessions=%d workers=%s/%s",
            config.gateway.max_sessions,
            config.workers.mode,
            config.workers.agent_kind,
        )
        yield
        for runtime in list(state.runtimes):
            with contextlib.suppress(Exception):
                await runtime.close()
        await state.pool.close()
        state.event_log.write("gateway_stop", None, {"total_sessions": state.total_sessions})

    app = FastAPI(title="fdh backend gateway", lifespan=lifespan)
    app.state.gateway = state

    @app.get("/health")
    async def health() -> dict[str, Any]:
        workers = [
            runtime.controller._worker.pid  # noqa: SLF001 - diagnostics only
            for runtime in state.runtimes
            if runtime.controller is not None and runtime.controller._worker is not None  # noqa: SLF001
        ]
        return {
            "ok": True,
            "sessions": state.sessions,
            "max_sessions": config.gateway.max_sessions,
            "total_sessions": state.total_sessions,
            "refused": state.refused,
            "workers": len(workers),
            "worker_pids": workers,
            "worker_mode": config.workers.mode,
            "agent_kind": config.workers.agent_kind,
            "backend_features": dict(templates.features),
            "backend_catalog_sha256": templates.catalog_sha256,
            "domains": [name for name, _ in config.domains],  # domain-note detection order (gateway.yaml)
            "hermes": {
                "model": config.hermes.model,
                "base_url": config.hermes.base_url,
                "backend_trace": config.hermes.backend_trace,
                "reasoning": bool(
                    ((config.hermes.request_overrides.get("extra_body") or {}).get("chat_template_kwargs") or {}).get(
                        "enable_thinking", False
                    )
                ),
            },
        }

    @app.websocket(PATH)
    async def backend(websocket: WebSocket) -> None:
        await websocket.accept()
        if state.sessions >= config.gateway.max_sessions:
            state.refused += 1
            state.event_log.write("capacity_refused", None, {"sessions": state.sessions})
            await websocket.send_text(
                proto.encode(
                    proto.GATEWAY_TO_VOICE,
                    "error",
                    code="capacity",
                    message=f"gateway at capacity (gateway.max_sessions={config.gateway.max_sessions})",
                    fatal=True,
                )
            )
            await websocket.close(code=1013)
            return
        state.sessions += 1
        state.total_sessions += 1
        writer = QueueWriter(websocket.send_text, name="gateway->voice")
        writer.start()

        def send(message: dict[str, Any]) -> None:
            with contextlib.suppress(WriterClosedError):
                writer.put(proto.dumps(message))

        runtime = SessionRuntime(config, send, pool=state.pool, templates=templates, event_log=state.event_log)
        state.runtimes.add(runtime)
        try:
            while True:
                try:
                    raw = await websocket.receive_text()
                except WebSocketDisconnect:
                    break
                try:
                    msg = proto.decode(proto.VOICE_TO_GATEWAY, raw)
                except proto.ProtocolError as exc:
                    send(
                        proto.message(
                            proto.GATEWAY_TO_VOICE, "error", code="bad_message", message=str(exc), fatal=False
                        )
                    )
                    continue
                await runtime.handle(msg)
                if msg["type"] == "session.close":
                    break
        except Exception:  # noqa: BLE001 - logged; the session is cleaned up below
            logger.exception("gateway session failed")
        finally:
            with contextlib.suppress(Exception):
                await runtime.close()
            state.runtimes.discard(runtime)
            state.sessions -= 1
            await writer.close(drain=True, timeout=2.0)
            with contextlib.suppress(Exception):
                await websocket.close()

    return app


def _load_env() -> None:
    try:
        from dotenv import load_dotenv  # noqa: PLC0415
    except ImportError:
        return
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / ".env"
        if candidate.is_file() and (parent / "pyproject.toml").is_file():
            load_dotenv(candidate, override=False)
            return


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="fdh backend gateway (Hermes worker per session)")
    parser.add_argument("--config", default=str(DEFAULT_GATEWAY_CONFIG))
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _load_env()
    config = load_gateway_config(args.config)
    gateway = config.gateway
    if args.host or args.port or args.max_sessions:
        gateway = replace(
            gateway,
            host=args.host or gateway.host,
            port=args.port or gateway.port,
            max_sessions=args.max_sessions or gateway.max_sessions,
        )
        config = replace(config, gateway=gateway)
    if config.workers.agent_kind == "hermes" and not Path(resolve_python(config.workers.python)).exists():
        raise SystemExit(
            f"workers.python {config.workers.python!r} does not exist; build the Hermes venv first "
            "(see .agents/skills/nemotron-voice-agent-fdh-deploy/references/prerequisites.md)"
        )
    import uvicorn  # noqa: PLC0415

    uvicorn.run(
        build_gateway_app(config),
        host=gateway.host,
        port=gateway.port,
        ws_ping_interval=None,
        ws_ping_timeout=None,
        ws_max_size=64 * 1024 * 1024,
        log_level="info",
    )


if __name__ == "__main__":
    main()
