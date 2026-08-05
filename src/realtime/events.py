# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""OpenAI Realtime–shaped event helpers for the v1 WebSocket shim."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

EmitFn = Callable[[dict[str, Any]], Awaitable[None]]

# Client → server
CLIENT_SESSION_UPDATE = "session.update"
CLIENT_AUDIO_APPEND = "input_audio_buffer.append"
CLIENT_AUDIO_COMMIT = "input_audio_buffer.commit"
CLIENT_AUDIO_CLEAR = "input_audio_buffer.clear"
CLIENT_RESPONSE_CREATE = "response.create"
CLIENT_RESPONSE_CANCEL = "response.cancel"
CLIENT_ITEM_CREATE = "conversation.item.create"
CLIENT_ITEM_TRUNCATE = "conversation.item.truncate"

# Server → client
SERVER_SESSION_CREATED = "session.created"
SERVER_SESSION_UPDATED = "session.updated"
SERVER_ERROR = "error"
SERVER_SPEECH_STARTED = "input_audio_buffer.speech_started"
SERVER_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
SERVER_AUDIO_COMMITTED = "input_audio_buffer.committed"
SERVER_AUDIO_CLEARED = "input_audio_buffer.cleared"
SERVER_ITEM_CREATED = "conversation.item.created"
SERVER_INPUT_TRANSCRIPT_DELTA = "conversation.item.input_audio_transcription.delta"
SERVER_INPUT_TRANSCRIPT_COMPLETED = "conversation.item.input_audio_transcription.completed"
SERVER_RESPONSE_CREATED = "response.created"
SERVER_RESPONSE_DONE = "response.done"
SERVER_OUTPUT_ITEM_ADDED = "response.output_item.added"
SERVER_OUTPUT_ITEM_DONE = "response.output_item.done"
SERVER_CONTENT_PART_ADDED = "response.content_part.added"
SERVER_CONTENT_PART_DONE = "response.content_part.done"
SERVER_OUTPUT_AUDIO_DELTA = "response.output_audio.delta"
SERVER_OUTPUT_AUDIO_DONE = "response.output_audio.done"
SERVER_OUTPUT_AUDIO_TRANSCRIPT_DELTA = "response.output_audio_transcript.delta"
SERVER_OUTPUT_AUDIO_TRANSCRIPT_DONE = "response.output_audio_transcript.done"
SERVER_OUTPUT_TEXT_DELTA = "response.output_text.delta"
SERVER_OUTPUT_TEXT_DONE = "response.output_text.done"
SERVER_FUNCTION_CALL_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
SERVER_FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"

# GA names are canonical; dual-emit pre-GA aliases for older SDKs.
BETA_EVENT_ALIASES: dict[str, str] = {
    SERVER_OUTPUT_AUDIO_DELTA: "response.audio.delta",
    SERVER_OUTPUT_AUDIO_DONE: "response.audio.done",
    SERVER_OUTPUT_AUDIO_TRANSCRIPT_DELTA: "response.audio_transcript.delta",
    SERVER_OUTPUT_AUDIO_TRANSCRIPT_DONE: "response.audio_transcript.done",
    SERVER_OUTPUT_TEXT_DELTA: "response.text.delta",
    SERVER_OUTPUT_TEXT_DONE: "response.text.done",
}


def new_event_id() -> str:
    """Return a unique server/client event id."""
    return f"evt_{uuid.uuid4().hex}"


def server_event(event_type: str, **payload: Any) -> dict[str, Any]:
    """Build a server → client event with a fresh ``event_id`` (GA type names)."""
    body: dict[str, Any] = {"event_id": new_event_id(), "type": event_type}
    body.update(payload)
    return body


def response_created_body(response_id: str) -> dict[str, Any]:
    """Shape for ``response.created``; ``output`` must be a list for client SDKs."""
    return {
        "id": response_id,
        "object": "realtime.response",
        "status": "in_progress",
        "output": [],
    }


def with_beta_aliases(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Return ``[event]`` plus a pre-GA alias when the GA/legacy pair differs."""
    events = [event]
    alias_type = BETA_EVENT_ALIASES.get(str(event.get("type") or ""))
    if alias_type:
        alias = dict(event)
        alias["type"] = alias_type
        alias["event_id"] = new_event_id()
        events.append(alias)
    return events


async def emit_with_aliases(emit: EmitFn, event: dict[str, Any]) -> None:
    """Send ``event`` and any pre-GA alias via ``emit``."""
    for payload in with_beta_aliases(event):
        await emit(payload)


def error_event(
    message: str,
    *,
    code: str | None = None,
    event_id: str | None = None,
    param: str | None = None,
) -> dict[str, Any]:
    """Build a Realtime-shaped ``error`` event."""
    err: dict[str, Any] = {
        "type": "invalid_request_error",
        "message": message,
    }
    if code:
        err["code"] = code
    if param:
        err["param"] = param
    if event_id:
        err["event_id"] = event_id
    logger.info(f"Realtime error event code={code or '-'} param={param or '-'} message={message}")
    return server_event(SERVER_ERROR, error=err)
