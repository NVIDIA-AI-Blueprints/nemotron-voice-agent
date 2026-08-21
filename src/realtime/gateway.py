# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""WebSocket gateway for ``WS /v1/realtime`` (session lifecycle → pipeline handoff)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from realtime.client_tools import validate_tool_choice_names, validate_tool_ownership
from realtime.events import (
    CLIENT_SESSION_UPDATE,
    SERVER_SESSION_CREATED,
    SERVER_SESSION_UPDATED,
    error_event,
    server_event,
)
from realtime.session import DEFAULT_PIPELINE_MODE, RealtimeSession, map_session_update_to_flat_config
from realtime.voice import resolve_realtime_tts_voice, tts_routing_changed
from utils import parse_env_int

SanitizeFn = Callable[..., dict[str, Any]]
EnsureReadyFn = Callable[[dict[str, Any]], Awaitable[None]]
StartBotFn = Callable[[WebSocket, dict[str, Any], dict[str, Any]], Awaitable[None]]
ServerToolsFn = Callable[[dict[str, Any]], list[str]]

# Pre-handoff: disconnect idle / abusive clients before session.update succeeds.
_DEFAULT_SESSION_UPDATE_TIMEOUT_SECS = 60
_DEFAULT_MAX_REJECTED_EVENTS = 32


def _select_realtime_subprotocol(websocket: WebSocket) -> str | None:
    """Echo a negotiated ``Sec-WebSocket-Protocol`` so browser clients can connect.

    Prefer ``realtime`` when offered; otherwise echo the first non-meta token.
    """
    protocols = websocket.headers.get("sec-websocket-protocol") or ""
    parts = [p.strip() for p in protocols.split(",") if p.strip()]
    if not parts:
        return None
    if "realtime" in parts:
        return "realtime"
    for part in parts:
        if not part.startswith("openai-insecure-api-key.") and not part.startswith("openai-beta."):
            return part
    # Never negotiate an API-key / beta token as the subprotocol (would echo secrets).
    return None


async def _send_json(websocket: WebSocket, payload: dict[str, Any]) -> None:
    await websocket.send_text(json.dumps(payload))


async def handle_realtime_websocket(
    websocket: WebSocket,
    *,
    sanitize_session_config: SanitizeFn,
    ensure_services_ready: EnsureReadyFn | None = None,
    start_bot: StartBotFn | None = None,
    resolve_server_tools: ServerToolsFn | None = None,
    fallback_example_key: str = "",
    default_pipeline_mode: str = DEFAULT_PIPELINE_MODE,
) -> None:
    """Accept a Realtime WebSocket, configure session, then hand off to a pipeline.

    Pre-handoff: only ``session.update`` is accepted (plus errors for anything else).
    After a successful update + readiness check, ``start_bot`` owns the socket
    (RealtimeFrameSerializer handles audio events).

    Readiness failures emit ``error`` and **keep the WebSocket open** so the client
    can fix config and retry ``session.update``.
    """
    # Browser clients negotiate Sec-WebSocket-Protocol; echo or the handshake fails.
    subprotocol = _select_realtime_subprotocol(websocket)
    if subprotocol:
        await websocket.accept(subprotocol=subprotocol)
    else:
        await websocket.accept()

    session = RealtimeSession(default_pipeline_mode=default_pipeline_mode)
    with logger.contextualize(stream_id=session.id):
        logger.info(f"Realtime WS connected session_id={session.id}")

        await _send_json(
            websocket,
            server_event(SERVER_SESSION_CREATED, session=session.public_session()),
        )

        deadline_secs = parse_env_int(
            "REALTIME_SESSION_UPDATE_TIMEOUT_SECS",
            _DEFAULT_SESSION_UPDATE_TIMEOUT_SECS,
            min_value=5,
        )
        max_rejected = parse_env_int(
            "REALTIME_MAX_REJECTED_EVENTS",
            _DEFAULT_MAX_REJECTED_EVENTS,
            min_value=1,
        )
        rejected = 0
        failed_updates = 0

        async def _reject(payload: dict[str, Any]) -> bool:
            """Emit an error and return True if the socket should close for abuse."""
            nonlocal rejected
            await _send_json(websocket, payload)
            rejected += 1
            if rejected >= max_rejected:
                logger.info(f"Realtime WS closing after {rejected} rejected pre-handoff events session_id={session.id}")
                await websocket.close(code=1008, reason="too many rejected events")
                return True
            return False

        try:
            while True:
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=deadline_secs)
                except TimeoutError:
                    logger.info(f"Realtime WS idle before session.update session_id={session.id}")
                    await websocket.close(code=1008, reason="session.update timeout")
                    return
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    if await _reject(error_event("Invalid JSON", code="invalid_json")):
                        return
                    continue

                if not isinstance(message, dict):
                    if await _reject(
                        error_event("Event must be a JSON object", code="invalid_event"),
                    ):
                        return
                    continue

                event_type = message.get("type")
                client_event_id = message.get("event_id")
                echo_id = client_event_id if isinstance(client_event_id, str) else None

                if not isinstance(event_type, str) or not event_type:
                    if await _reject(
                        error_event("Missing event type", code="missing_type", event_id=echo_id),
                    ):
                        return
                    continue

                if event_type != CLIENT_SESSION_UPDATE:
                    if await _reject(
                        error_event(
                            f"Event type '{event_type}' is not supported before session.update handoff",
                            code="unsupported_event",
                            event_id=echo_id,
                            param="type",
                        ),
                    ):
                        return
                    continue

                started = await _handle_session_update(
                    websocket,
                    session,
                    message,
                    sanitize_session_config=sanitize_session_config,
                    ensure_services_ready=ensure_services_ready,
                    start_bot=start_bot,
                    resolve_server_tools=resolve_server_tools,
                    fallback_example_key=fallback_example_key,
                    default_pipeline_mode=default_pipeline_mode,
                )
                if started:
                    # Bot owns the WebSocket receive loop now.
                    return
                # Valid session.update that failed sanitize/readiness: keep open for
                # retry, but bound failed handoff attempts.
                failed_updates += 1
                if failed_updates >= max_rejected:
                    logger.info(
                        f"Realtime WS closing after {failed_updates} failed session.update "
                        f"attempts session_id={session.id}"
                    )
                    await websocket.close(code=1008, reason="too many failed session.update attempts")
                    return
        except WebSocketDisconnect:
            logger.info(f"Realtime WS disconnected session_id={session.id}")
        except Exception:
            logger.exception(f"Realtime WS error session_id={session.id}")
            raise


async def _handle_session_update(
    websocket: WebSocket,
    session: RealtimeSession,
    message: dict[str, Any],
    *,
    sanitize_session_config: SanitizeFn,
    ensure_services_ready: EnsureReadyFn | None,
    start_bot: StartBotFn | None,
    fallback_example_key: str,
    default_pipeline_mode: str,
    resolve_server_tools: ServerToolsFn | None = None,
) -> bool:
    """Validate then apply session.update. Return True if the pipeline was started."""
    client_event_id = message.get("event_id")
    echo_id = client_event_id if isinstance(client_event_id, str) else None
    session_patch = message.get("session")
    if not isinstance(session_patch, dict):
        await _send_json(
            websocket,
            error_event(
                "session.update requires a session object",
                code="invalid_session",
                event_id=echo_id,
                param="session",
            ),
        )
        return False

    try:
        flat = map_session_update_to_flat_config(
            session_patch,
            default_pipeline_mode=default_pipeline_mode,
        )
        voice_was_set = "tts_voice_id" in flat
        merged = {**session.flat_config, **flat}
        sanitized = sanitize_session_config(merged, fallback_example_key=fallback_example_key)
        if "tool_choice" in flat:
            sanitized["tool_choice"] = flat["tool_choice"]
        if "temperature" in flat:
            sanitized["temperature"] = flat["temperature"]
        if "max_tokens" in flat:
            sanitized["max_tokens"] = flat["max_tokens"]
        if "client_tools" in flat:
            sanitized["client_tools"] = flat["client_tools"]
        if resolve_server_tools is not None:
            sanitized["server_tools"] = resolve_server_tools(sanitized)
        validate_tool_ownership(
            sanitized.get("client_tools", []),
            sanitized.get("server_tools", []),
            pipeline_mode=str(sanitized.get("pipeline_mode") or default_pipeline_mode),
        )
        validate_tool_choice_names(
            sanitized.get("tool_choice"),
            {
                *[str(tool.get("name") or "") for tool in sanitized.get("client_tools", [])],
                *[str(name) for name in sanitized.get("server_tools", [])],
            },
        )

        # Soft-resolve voice against the TTS catalog (same list path as RTVI UI).
        # Unknown voices warn and fall back to the catalog default — no reject.
        # Cold catalog fetch runs in a worker thread so the event loop stays free.
        routing_changed = tts_routing_changed(session.flat_config, sanitized)
        resolved_voice = await asyncio.to_thread(
            resolve_realtime_tts_voice,
            sanitized,
            voice_was_set=voice_was_set,
            tts_routing_changed=routing_changed,
        )
        if resolved_voice and "tts_voice_id" not in flat and routing_changed:
            # Ensure apply_update / public session see the re-resolved voice.
            flat["tts_voice_id"] = resolved_voice
    except ValueError as exc:
        await _send_json(
            websocket,
            error_event(str(exc), code="invalid_session", event_id=echo_id, param="session"),
        )
        return False
    except Exception as exc:
        logger.exception("session.update sanitize failed")
        await _send_json(
            websocket,
            error_event(
                f"Failed to apply session.update: {exc}",
                code="session_update_failed",
                event_id=echo_id,
            ),
        )
        return False

    if ensure_services_ready is not None:
        try:
            await ensure_services_ready(sanitized)
        except RuntimeError as exc:
            logger.warning(f"Realtime readiness check failed (WS kept open): {exc}")
            await _send_json(
                websocket,
                error_event(
                    str(exc),
                    code="services_not_ready",
                    event_id=echo_id,
                ),
            )
            return False
        except Exception as exc:
            logger.exception("Realtime readiness check raised unexpectedly")
            await _send_json(
                websocket,
                error_event(
                    f"Service readiness check failed: {exc}",
                    code="services_not_ready",
                    event_id=echo_id,
                ),
            )
            return False

    public = session.apply_update(session_patch, sanitized_flat=sanitized)
    await _send_json(websocket, server_event(SERVER_SESSION_UPDATED, session=public))

    if start_bot is None:
        logger.warning("Realtime start_bot not configured; session configured only")
        return False

    logger.info(f"Realtime handoff to pipeline session_id={session.id} pipeline_mode={sanitized.get('pipeline_mode')}")
    await start_bot(websocket, sanitized, public)
    return True
