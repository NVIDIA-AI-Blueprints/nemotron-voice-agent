# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse client events into typed commands.

Parsing is pure: a JSON object in, one frozen command out, or a
:class:`WireProtocolError` carrying the Realtime error code and parameter. The
session never inspects raw client dictionaries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from examples.frontend_backend_verdict.voice.audio.pcm import b64decode_audio
from examples.frontend_backend_verdict.voice.errors import WireProtocolError

#: Largest accepted ``input_audio_buffer.append`` payload (the OpenAI limit).
MAX_APPEND_BYTES = 15 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class SessionUpdate:
    """``session.update``."""

    session: dict[str, Any]
    event_id: str | None = None


@dataclass(frozen=True, slots=True)
class AudioAppend:
    """``input_audio_buffer.append`` (payload already base64-decoded)."""

    audio: bytes
    event_id: str | None = None


@dataclass(frozen=True, slots=True)
class AudioCommit:
    """``input_audio_buffer.commit``."""

    event_id: str | None = None


@dataclass(frozen=True, slots=True)
class AudioClear:
    """``input_audio_buffer.clear``."""

    event_id: str | None = None


@dataclass(frozen=True, slots=True)
class UserText:
    """A user message item with ``input_text`` content."""

    text: str
    item_id: str | None = None


@dataclass(frozen=True, slots=True)
class FunctionCallOutput:
    """A ``function_call_output`` item."""

    call_id: str
    output: str
    item_id: str | None = None


@dataclass(frozen=True, slots=True)
class ItemCreate:
    """``conversation.item.create``."""

    item: UserText | FunctionCallOutput
    previous_item_id: str | None = None
    event_id: str | None = None


@dataclass(frozen=True, slots=True)
class ItemTruncate:
    """``conversation.item.truncate``."""

    item_id: str
    content_index: int
    audio_end_ms: int
    event_id: str | None = None


@dataclass(frozen=True, slots=True)
class ItemDelete:
    """``conversation.item.delete``."""

    item_id: str
    event_id: str | None = None


@dataclass(frozen=True, slots=True)
class ResponseCreate:
    """``response.create`` (``overrides`` is the optional ``response`` object)."""

    overrides: dict[str, Any] = field(default_factory=dict)
    event_id: str | None = None


@dataclass(frozen=True, slots=True)
class ResponseCancel:
    """``response.cancel``."""

    response_id: str | None = None
    event_id: str | None = None


@dataclass(frozen=True, slots=True)
class OutputAudioBufferClear:
    """``output_audio_buffer.clear``."""

    event_id: str | None = None


ClientCommand = (
    SessionUpdate
    | AudioAppend
    | AudioCommit
    | AudioClear
    | ItemCreate
    | ItemTruncate
    | ItemDelete
    | ResponseCreate
    | ResponseCancel
    | OutputAudioBufferClear
)


def decode_message(text: str | bytes) -> dict[str, Any]:
    """Decode one WebSocket text frame into a JSON object."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise WireProtocolError(f"client message is not valid JSON: {exc}", code="invalid_json") from exc
    if not isinstance(data, dict):
        raise WireProtocolError("client message must be a JSON object", code="invalid_json")
    return data


def _str(data: dict[str, Any], key: str, *, param: str, required: bool = True) -> str | None:
    value = data.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or (required and not value):
        raise WireProtocolError(f"{param} must be a non-empty string", code="invalid_value", param=param)
    return value


def _int(data: dict[str, Any], key: str, *, param: str, default: int | None = None) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise WireProtocolError(f"{param} must be an integer", code="invalid_value", param=param)
    return int(value)


def _parse_item(item: Any) -> UserText | FunctionCallOutput:
    if not isinstance(item, dict):
        raise WireProtocolError("item must be an object", code="invalid_value", param="item")
    kind = item.get("type")
    if kind == "function_call_output":
        call_id = _str(item, "call_id", param="item.call_id")
        output = item.get("output", "")
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False)
        return FunctionCallOutput(call_id=str(call_id), output=output, item_id=item.get("id"))
    if kind == "message" and item.get("role") == "user":
        texts: list[str] = []
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "input_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
            else:
                raise WireProtocolError(
                    "only input_text content is supported for user message items",
                    code="unsupported_content",
                    param="item.content",
                )
        text = " ".join(t.strip() for t in texts if t.strip())
        if not text:
            raise WireProtocolError("user message item has no text", code="invalid_value", param="item.content")
        return UserText(text=text, item_id=item.get("id"))
    raise WireProtocolError(
        f"conversation.item.create with item type {kind!r} (role {item.get('role')!r}) is not supported; "
        "use a user input_text message or a function_call_output",
        code="unsupported_item",
        param="item.type",
    )


def parse_client_event(data: dict[str, Any]) -> ClientCommand:
    """Parse one client event."""
    kind = data.get("type")
    event_id = data.get("event_id") if isinstance(data.get("event_id"), str) else None
    if kind == "session.update":
        session = data.get("session")
        if not isinstance(session, dict):
            raise WireProtocolError("session must be an object", code="invalid_value", param="session")
        return SessionUpdate(session=session, event_id=event_id)
    if kind == "input_audio_buffer.append":
        audio = data.get("audio")
        if not isinstance(audio, str):
            raise WireProtocolError("audio must be a base64 string", code="invalid_value", param="audio")
        try:
            payload = b64decode_audio(audio)
        except ValueError as exc:
            raise WireProtocolError(str(exc), code="invalid_value", param="audio") from exc
        if len(payload) > MAX_APPEND_BYTES:
            raise WireProtocolError("audio payload exceeds 15 MB", code="invalid_value", param="audio")
        return AudioAppend(audio=payload, event_id=event_id)
    if kind == "input_audio_buffer.commit":
        return AudioCommit(event_id=event_id)
    if kind == "input_audio_buffer.clear":
        return AudioClear(event_id=event_id)
    if kind == "conversation.item.create":
        previous = data.get("previous_item_id")
        return ItemCreate(
            item=_parse_item(data.get("item")),
            previous_item_id=previous if isinstance(previous, str) else None,
            event_id=event_id,
        )
    if kind == "conversation.item.truncate":
        return ItemTruncate(
            item_id=str(_str(data, "item_id", param="item_id")),
            content_index=_int(data, "content_index", param="content_index", default=0),
            audio_end_ms=_int(data, "audio_end_ms", param="audio_end_ms", default=0),
            event_id=event_id,
        )
    if kind == "conversation.item.delete":
        return ItemDelete(item_id=str(_str(data, "item_id", param="item_id")), event_id=event_id)
    if kind == "response.create":
        overrides = data.get("response") or {}
        if not isinstance(overrides, dict):
            raise WireProtocolError("response must be an object", code="invalid_value", param="response")
        return ResponseCreate(overrides=overrides, event_id=event_id)
    if kind == "response.cancel":
        response_id = data.get("response_id")
        return ResponseCancel(response_id=response_id if isinstance(response_id, str) else None, event_id=event_id)
    if kind == "output_audio_buffer.clear":
        return OutputAudioBufferClear(event_id=event_id)
    raise WireProtocolError(f"unknown or unsupported client event type {kind!r}", code="unknown_event", param="type")
