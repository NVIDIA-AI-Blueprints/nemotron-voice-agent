# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builders for every server event this server emits (GA names only).

Event names mirror ``realtime.events`` but are defined here, because importing
the ``realtime`` package pulls in its Pipecat gateway and ``wire/`` must stay
Pipecat-free. No builder ever produces a beta event name (plan section 7.4).
"""

from __future__ import annotations

from typing import Any

from examples.frontend_backend_verdict.voice.wire.ids import new_event_id

SESSION_CREATED = "session.created"
SESSION_UPDATED = "session.updated"
ERROR = "error"
SPEECH_STARTED = "input_audio_buffer.speech_started"
SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
AUDIO_COMMITTED = "input_audio_buffer.committed"
AUDIO_CLEARED = "input_audio_buffer.cleared"
ITEM_ADDED = "conversation.item.added"
ITEM_DONE = "conversation.item.done"
ITEM_TRUNCATED = "conversation.item.truncated"
ITEM_DELETED = "conversation.item.deleted"
INPUT_TRANSCRIPTION_DELTA = "conversation.item.input_audio_transcription.delta"
INPUT_TRANSCRIPTION_COMPLETED = "conversation.item.input_audio_transcription.completed"
INPUT_TRANSCRIPTION_FAILED = "conversation.item.input_audio_transcription.failed"
RESPONSE_CREATED = "response.created"
RESPONSE_DONE = "response.done"
OUTPUT_ITEM_ADDED = "response.output_item.added"
OUTPUT_ITEM_DONE = "response.output_item.done"
CONTENT_PART_ADDED = "response.content_part.added"
CONTENT_PART_DONE = "response.content_part.done"
OUTPUT_AUDIO_DELTA = "response.output_audio.delta"
OUTPUT_AUDIO_DONE = "response.output_audio.done"
OUTPUT_AUDIO_TRANSCRIPT_DELTA = "response.output_audio_transcript.delta"
OUTPUT_AUDIO_TRANSCRIPT_DONE = "response.output_audio_transcript.done"
OUTPUT_TEXT_DELTA = "response.output_text.delta"
OUTPUT_TEXT_DONE = "response.output_text.done"
FUNCTION_CALL_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"

#: Every event type this server can emit.
GA_EVENT_TYPES: frozenset[str] = frozenset(
    {
        SESSION_CREATED,
        SESSION_UPDATED,
        ERROR,
        SPEECH_STARTED,
        SPEECH_STOPPED,
        AUDIO_COMMITTED,
        AUDIO_CLEARED,
        ITEM_ADDED,
        ITEM_DONE,
        ITEM_TRUNCATED,
        ITEM_DELETED,
        INPUT_TRANSCRIPTION_DELTA,
        INPUT_TRANSCRIPTION_COMPLETED,
        INPUT_TRANSCRIPTION_FAILED,
        RESPONSE_CREATED,
        RESPONSE_DONE,
        OUTPUT_ITEM_ADDED,
        OUTPUT_ITEM_DONE,
        CONTENT_PART_ADDED,
        CONTENT_PART_DONE,
        OUTPUT_AUDIO_DELTA,
        OUTPUT_AUDIO_DONE,
        OUTPUT_AUDIO_TRANSCRIPT_DELTA,
        OUTPUT_AUDIO_TRANSCRIPT_DONE,
        OUTPUT_TEXT_DELTA,
        OUTPUT_TEXT_DONE,
        FUNCTION_CALL_ARGUMENTS_DELTA,
        FUNCTION_CALL_ARGUMENTS_DONE,
    }
)

#: Beta-only names; asserted never to be emitted.
BETA_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "conversation.item.created",
        "response.audio.delta",
        "response.audio.done",
        "response.audio_transcript.delta",
        "response.audio_transcript.done",
        "response.text.delta",
        "response.text.done",
    }
)


def event(event_type: str, **payload: Any) -> dict[str, Any]:
    """Build one server event with a fresh ``event_id``."""
    return {"event_id": new_event_id(), "type": event_type, **payload}


def x_nvidia_filler(*, turn_id: int, text: str, mode: str) -> dict[str, Any]:
    """Non-standard: filler text the agent produced but did not speak (opt-in per connection)."""
    return event("x_nvidia.filler", turn_id=turn_id, text=text, mode=mode, spoken=False)


# -- session / errors --------------------------------------------------------


def session_created(session: dict[str, Any]) -> dict[str, Any]:
    """``session.created``."""
    return event(SESSION_CREATED, session=session)


def session_updated(session: dict[str, Any]) -> dict[str, Any]:
    """``session.updated``."""
    return event(SESSION_UPDATED, session=session)


def error(
    message: str,
    *,
    code: str | None = None,
    param: str | None = None,
    client_event_id: str | None = None,
    error_type: str = "invalid_request_error",
) -> dict[str, Any]:
    """``error`` (never fatal in this server)."""
    body: dict[str, Any] = {"type": error_type, "code": code, "message": message, "param": param}
    body["event_id"] = client_event_id
    return event(ERROR, error=body)


# -- input audio / conversation ---------------------------------------------


def speech_started(audio_start_ms: int, item_id: str) -> dict[str, Any]:
    """``input_audio_buffer.speech_started`` on the cumulative input clock."""
    return event(SPEECH_STARTED, audio_start_ms=int(audio_start_ms), item_id=item_id)


def speech_stopped(audio_end_ms: int, item_id: str) -> dict[str, Any]:
    """``input_audio_buffer.speech_stopped``."""
    return event(SPEECH_STOPPED, audio_end_ms=int(audio_end_ms), item_id=item_id)


def audio_committed(item_id: str, previous_item_id: str | None) -> dict[str, Any]:
    """``input_audio_buffer.committed``."""
    return event(AUDIO_COMMITTED, item_id=item_id, previous_item_id=previous_item_id)


def audio_cleared() -> dict[str, Any]:
    """``input_audio_buffer.cleared``."""
    return event(AUDIO_CLEARED)


def item_added(item: dict[str, Any], previous_item_id: str | None) -> dict[str, Any]:
    """``conversation.item.added``."""
    return event(ITEM_ADDED, previous_item_id=previous_item_id, item=item)


def item_done(item: dict[str, Any], previous_item_id: str | None) -> dict[str, Any]:
    """``conversation.item.done``."""
    return event(ITEM_DONE, previous_item_id=previous_item_id, item=item)


def item_truncated(item_id: str, content_index: int, audio_end_ms: int) -> dict[str, Any]:
    """``conversation.item.truncated``."""
    return event(ITEM_TRUNCATED, item_id=item_id, content_index=content_index, audio_end_ms=int(audio_end_ms))


def item_deleted(item_id: str) -> dict[str, Any]:
    """``conversation.item.deleted``."""
    return event(ITEM_DELETED, item_id=item_id)


def input_transcription_delta(item_id: str, delta: str) -> dict[str, Any]:
    """``conversation.item.input_audio_transcription.delta``."""
    return event(INPUT_TRANSCRIPTION_DELTA, item_id=item_id, content_index=0, delta=delta)


def input_transcription_completed(item_id: str, transcript: str) -> dict[str, Any]:
    """``conversation.item.input_audio_transcription.completed``."""
    return event(
        INPUT_TRANSCRIPTION_COMPLETED,
        item_id=item_id,
        content_index=0,
        transcript=transcript,
        usage={"type": "duration", "seconds": 0},
    )


def input_transcription_failed(item_id: str, message: str) -> dict[str, Any]:
    """``conversation.item.input_audio_transcription.failed``."""
    return event(
        INPUT_TRANSCRIPTION_FAILED,
        item_id=item_id,
        content_index=0,
        error={"type": "transcription_error", "code": "asr_failed", "message": message, "param": None},
    )


# -- items -------------------------------------------------------------------


def user_audio_item(item_id: str, transcript: str | None = None) -> dict[str, Any]:
    """A user message item carrying input audio."""
    return {
        "id": item_id,
        "object": "realtime.item",
        "type": "message",
        "status": "completed",
        "role": "user",
        "content": [{"type": "input_audio", "transcript": transcript}],
    }


def user_text_item(item_id: str, text: str) -> dict[str, Any]:
    """A user message item carrying text."""
    return {
        "id": item_id,
        "object": "realtime.item",
        "type": "message",
        "status": "completed",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }


def assistant_message_item(
    item_id: str, *, status: str, modality: str = "audio", transcript: str | None = None
) -> dict[str, Any]:
    """An assistant message item (empty content while in progress)."""
    content: list[dict[str, Any]] = []
    if transcript is not None:
        part_type = "output_audio" if modality == "audio" else "output_text"
        key = "transcript" if modality == "audio" else "text"
        content = [{"type": part_type, key: transcript}]
    return {
        "id": item_id,
        "object": "realtime.item",
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": content,
    }


def function_call_item(item_id: str, *, call_id: str, name: str, arguments: str, status: str) -> dict[str, Any]:
    """A function-call item."""
    return {
        "id": item_id,
        "object": "realtime.item",
        "type": "function_call",
        "status": status,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def function_call_output_item(item_id: str, *, call_id: str, output: str) -> dict[str, Any]:
    """A function-call output item (echo of the client's)."""
    return {
        "id": item_id,
        "object": "realtime.item",
        "type": "function_call_output",
        "status": "completed",
        "call_id": call_id,
        "output": output,
    }


# -- responses ---------------------------------------------------------------


def usage_object(*, input_tokens: int = 0, output_tokens: int = 0, cached_tokens: int = 0) -> dict[str, Any]:
    """A ``response.usage`` object (always present; some clients dereference it unconditionally)."""
    return {
        "total_tokens": input_tokens + output_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_token_details": {"text_tokens": input_tokens, "audio_tokens": 0, "cached_tokens": cached_tokens},
        "output_token_details": {"text_tokens": output_tokens, "audio_tokens": 0},
    }


def response_object(
    response_id: str,
    *,
    status: str,
    output: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    status_details: dict[str, Any] | None = None,
    modality: str = "audio",
) -> dict[str, Any]:
    """A ``response`` object."""
    return {
        "id": response_id,
        "object": "realtime.response",
        "status": status,
        "status_details": status_details,
        "output": list(output or []),
        "output_modalities": [modality],
        "usage": usage,
    }


def response_created(response: dict[str, Any]) -> dict[str, Any]:
    """``response.created``."""
    return event(RESPONSE_CREATED, response=response)


def response_done(response: dict[str, Any]) -> dict[str, Any]:
    """``response.done``."""
    return event(RESPONSE_DONE, response=response)


def output_item_added(response_id: str, output_index: int, item: dict[str, Any]) -> dict[str, Any]:
    """``response.output_item.added``."""
    return event(OUTPUT_ITEM_ADDED, response_id=response_id, output_index=output_index, item=item)


def output_item_done(response_id: str, output_index: int, item: dict[str, Any]) -> dict[str, Any]:
    """``response.output_item.done``."""
    return event(OUTPUT_ITEM_DONE, response_id=response_id, output_index=output_index, item=item)


def content_part_added(
    response_id: str, item_id: str, output_index: int, part: dict[str, Any], content_index: int = 0
) -> dict[str, Any]:
    """``response.content_part.added``."""
    return event(
        CONTENT_PART_ADDED,
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        part=part,
    )


def content_part_done(
    response_id: str, item_id: str, output_index: int, part: dict[str, Any], content_index: int = 0
) -> dict[str, Any]:
    """``response.content_part.done``."""
    return event(
        CONTENT_PART_DONE,
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        part=part,
    )


def _content_event(event_type: str, response_id: str, item_id: str, output_index: int, **payload: Any) -> dict:
    return event(
        event_type, response_id=response_id, item_id=item_id, output_index=output_index, content_index=0, **payload
    )


def output_audio_delta(response_id: str, item_id: str, output_index: int, delta_b64: str) -> dict[str, Any]:
    """``response.output_audio.delta``."""
    return _content_event(OUTPUT_AUDIO_DELTA, response_id, item_id, output_index, delta=delta_b64)


def output_audio_done(response_id: str, item_id: str, output_index: int) -> dict[str, Any]:
    """``response.output_audio.done``."""
    return _content_event(OUTPUT_AUDIO_DONE, response_id, item_id, output_index)


def output_audio_transcript_delta(response_id: str, item_id: str, output_index: int, delta: str) -> dict[str, Any]:
    """``response.output_audio_transcript.delta``."""
    return _content_event(OUTPUT_AUDIO_TRANSCRIPT_DELTA, response_id, item_id, output_index, delta=delta)


def output_audio_transcript_done(response_id: str, item_id: str, output_index: int, transcript: str) -> dict[str, Any]:
    """``response.output_audio_transcript.done``."""
    return _content_event(OUTPUT_AUDIO_TRANSCRIPT_DONE, response_id, item_id, output_index, transcript=transcript)


def output_text_delta(response_id: str, item_id: str, output_index: int, delta: str) -> dict[str, Any]:
    """``response.output_text.delta``."""
    return _content_event(OUTPUT_TEXT_DELTA, response_id, item_id, output_index, delta=delta)


def output_text_done(response_id: str, item_id: str, output_index: int, text: str) -> dict[str, Any]:
    """``response.output_text.done``."""
    return _content_event(OUTPUT_TEXT_DONE, response_id, item_id, output_index, text=text)


def function_call_arguments_delta(
    response_id: str, item_id: str, output_index: int, call_id: str, delta: str
) -> dict[str, Any]:
    """``response.function_call_arguments.delta``."""
    return event(
        FUNCTION_CALL_ARGUMENTS_DELTA,
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        call_id=call_id,
        delta=delta,
    )


def function_call_arguments_done(
    response_id: str, item_id: str, output_index: int, *, call_id: str, name: str, arguments: str
) -> dict[str, Any]:
    """``response.function_call_arguments.done`` with top-level ``call_id``/``name`` (where tau2 reads them)."""
    return event(
        FUNCTION_CALL_ARGUMENTS_DONE,
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        call_id=call_id,
        name=name,
        arguments=arguments,
    )
