# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Realtime transport factory (FastAPI WS + RealtimeFrameSerializer)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger
from pipecat.observers.base_observer import BaseObserver
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from realtime.events import BETA_EVENT_ALIASES
from realtime.observer import RealtimeLifecycleObserver
from realtime.serializer import RealtimeFrameSerializer
from utils import parse_env_int


def _prefers_beta_event_names(websocket: WebSocket) -> bool:
    """Return whether the browser client explicitly requested the beta dialect."""
    protocols = websocket.headers.get("sec-websocket-protocol") or ""
    return any(token.strip() == "openai-beta.realtime-v1" for token in protocols.split(","))


def create_realtime_transport(
    websocket: WebSocket,
    *,
    session_view: dict[str, Any] | None = None,
    runtime_config: dict[str, Any] | None = None,
) -> FastAPIWebsocketTransport:
    """Build a FastAPI WebSocket transport that speaks Realtime JSON."""
    serializer = RealtimeFrameSerializer(
        session_view=session_view,
        runtime_config=runtime_config,
    )
    emit_lock = asyncio.Lock()
    beta_event_names = _prefers_beta_event_names(websocket)

    async def _emit(event: dict[str, Any]) -> None:
        # ``emit_with_aliases`` calls us with canonical then beta payloads.
        # Select exactly one dialect so GA clients never receive legacy events
        # and beta clients never receive duplicate audio.
        event_type = event.get("type")
        if beta_event_names and event_type in BETA_EVENT_ALIASES:
            return
        if not beta_event_names and event_type in BETA_EVENT_ALIASES.values():
            return
        try:
            async with emit_lock:
                await websocket.send_text(json.dumps(event, default=str))
        except (RuntimeError, WebSocketDisconnect) as exc:
            logger.debug(f"Realtime emit skipped; websocket closed: {exc}")

    serializer.set_emit(_emit)

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_enabled=True,
            audio_out_sample_rate=16000,
            audio_out_10ms_chunks=parse_env_int("AUDIO_OUT_10MS_CHUNKS", 10),
            add_wav_header=False,
            serializer=serializer,
        ),
    )
    transport._realtime_serializer = serializer  # type: ignore[attr-defined]

    @transport.event_handler("on_client_disconnected")
    async def _shutdown_realtime_on_disconnect(_transport, _client) -> None:  # noqa: ARG001
        shutdown_realtime_transport(transport)

    return transport


def realtime_lifecycle_observer(transport: Any) -> BaseObserver | None:
    """Return a Realtime lifecycle observer when ``transport`` is Realtime-backed."""
    existing = getattr(transport, "_realtime_observer", None)
    if isinstance(existing, RealtimeLifecycleObserver):
        return existing
    serializer = getattr(transport, "_realtime_serializer", None)
    if not isinstance(serializer, RealtimeFrameSerializer) or serializer.emit is None:
        return None
    observer = RealtimeLifecycleObserver(
        emit=serializer.emit,
        conversation=serializer.conversation,
    )
    serializer.set_on_response_cancel(observer.on_response_cancelled)
    transport._realtime_observer = observer  # type: ignore[attr-defined]
    return observer


def shutdown_realtime_transport(transport: Any) -> None:
    """Cancel deferred Realtime observer work on session teardown."""
    observer = getattr(transport, "_realtime_observer", None)
    if isinstance(observer, RealtimeLifecycleObserver):
        observer.shutdown()
