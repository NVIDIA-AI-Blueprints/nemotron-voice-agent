# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared announce / finish sequences for Realtime responses."""

from __future__ import annotations

from typing import Any

from realtime.conversation import ConversationState, ResponseSnapshot, new_item_id
from realtime.events import (
    SERVER_CONTENT_PART_ADDED,
    SERVER_CONTENT_PART_DONE,
    SERVER_FUNCTION_CALL_ARGUMENTS_DONE,
    SERVER_ITEM_CREATED,
    SERVER_OUTPUT_AUDIO_DONE,
    SERVER_OUTPUT_AUDIO_TRANSCRIPT_DONE,
    SERVER_OUTPUT_ITEM_ADDED,
    SERVER_OUTPUT_ITEM_DONE,
    SERVER_RESPONSE_CREATED,
    SERVER_RESPONSE_DONE,
    EmitFn,
    emit_with_aliases,
    response_created_body,
    server_event,
)
from realtime.metrics import RealtimeResponseTelemetry


async def announce_response(
    conversation: ConversationState,
    emit: EmitFn,
    *,
    message_item: bool = True,
) -> tuple[str, bool]:
    """Ensure ``response.created`` + assistant item / content_part are on the wire.

    With ``message_item=False`` only ``response.created`` is ensured, for a
    response whose first output is a function call.

    Returns ``(response_id, newly_created)``.
    """
    response_id, created = conversation.begin_response()
    if created:
        await emit_with_aliases(
            emit,
            server_event(
                SERVER_RESPONSE_CREATED,
                response=response_created_body(response_id),
            ),
        )
    if not message_item:
        return response_id, created
    if not conversation.output_item_announced:
        conversation.output_item_announced = True
        item_id = conversation.assistant_item_id
        assistant_item = {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        }
        await emit_with_aliases(
            emit,
            server_event(SERVER_ITEM_CREATED, previous_item_id=None, item=assistant_item),
        )
        await emit_with_aliases(
            emit,
            server_event(
                SERVER_OUTPUT_ITEM_ADDED,
                response_id=response_id,
                output_index=0,
                item=assistant_item,
            ),
        )
    if not conversation.content_part_announced:
        conversation.content_part_announced = True
        await emit_with_aliases(
            emit,
            server_event(
                SERVER_CONTENT_PART_ADDED,
                response_id=response_id,
                item_id=conversation.assistant_item_id,
                output_index=0,
                content_index=0,
                part={"type": "audio", "transcript": ""},
            ),
        )
    return response_id, created


async def announce_function_call(
    conversation: ConversationState,
    emit: EmitFn,
    *,
    call_id: str,
    name: str,
    arguments: str,
) -> str:
    """Emit one client function call as a completed ``function_call`` output item.

    Returns the item id. The item is also recorded on the active response so
    ``response.done`` lists it.
    """
    response_id, _created = await announce_response(conversation, emit, message_item=False)
    item_id = new_item_id()
    output_index = len(conversation.function_call_items) + (1 if conversation.output_item_announced else 0)
    item = {
        "id": item_id,
        "object": "realtime.item",
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }
    conversation.function_call_items.append(item)
    await emit_with_aliases(
        emit,
        server_event(
            SERVER_OUTPUT_ITEM_ADDED,
            response_id=response_id,
            output_index=output_index,
            item={**item, "status": "in_progress", "arguments": ""},
        ),
    )
    await emit_with_aliases(emit, server_event(SERVER_ITEM_CREATED, previous_item_id=None, item=item))
    await emit_with_aliases(
        emit,
        server_event(
            SERVER_FUNCTION_CALL_ARGUMENTS_DONE,
            response_id=response_id,
            item_id=item_id,
            output_index=output_index,
            call_id=call_id,
            name=name,
            arguments=arguments,
        ),
    )
    await emit_with_aliases(
        emit,
        server_event(SERVER_OUTPUT_ITEM_DONE, response_id=response_id, output_index=output_index, item=item),
    )
    return item_id


async def finish_response(
    conversation: ConversationState,
    emit: EmitFn,
    *,
    status: str,
    telemetry: RealtimeResponseTelemetry | None = None,
) -> bool:
    """Complete the in-flight response, emit the done sequence, then reset the slot.

    Uses a :class:`ResponseSnapshot` so concurrent ``begin_response`` during the
    await chain cannot corrupt the events being emitted, and ``reset`` is a no-op
    if a newer generation already owns the slot.

    Returns ``True`` when a finish sequence was emitted.
    """
    snap = conversation.complete_response(status)
    if snap is None:
        return False
    await emit_finish_from_snapshot(snap, emit, telemetry=telemetry)
    conversation.reset_response_slot(generation=snap.generation)
    return True


async def emit_finish_from_snapshot(
    snap: ResponseSnapshot,
    emit: EmitFn,
    *,
    telemetry: RealtimeResponseTelemetry | None = None,
) -> None:
    """Emit audio/item/response done events for a completed snapshot."""
    response_id = snap.response_id
    item_id = snap.item_id
    transcript = snap.transcript
    # Item statuses are in_progress|completed|incomplete; cancelled is response-only.
    item_status = "completed" if snap.status == "completed" else "incomplete"
    function_calls = [dict(item) for item in snap.function_call_items]

    if function_calls and not snap.message_item_announced:
        # Function-call-only response: the call items are already done.
        await _emit_response_done(response_id, snap.status, function_calls, emit, telemetry=telemetry)
        return

    if not snap.audio_done_emitted:
        await emit_with_aliases(
            emit,
            server_event(
                SERVER_OUTPUT_AUDIO_DONE,
                response_id=response_id,
                item_id=item_id,
                output_index=0,
                content_index=0,
            ),
        )
    if not snap.transcript_done_emitted:
        await emit_with_aliases(
            emit,
            server_event(
                SERVER_OUTPUT_AUDIO_TRANSCRIPT_DONE,
                response_id=response_id,
                item_id=item_id,
                output_index=0,
                content_index=0,
                transcript=transcript,
            ),
        )
    await emit_with_aliases(
        emit,
        server_event(
            SERVER_CONTENT_PART_DONE,
            response_id=response_id,
            item_id=item_id,
            output_index=0,
            content_index=0,
            part={"type": "audio", "transcript": transcript},
        ),
    )
    await emit_with_aliases(
        emit,
        server_event(
            SERVER_OUTPUT_ITEM_DONE,
            response_id=response_id,
            output_index=0,
            item={
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "status": item_status,
                "content": [{"type": "output_audio", "transcript": transcript}],
            },
        ),
    )
    message = {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": item_status,
        "content": [{"type": "output_audio", "transcript": transcript}],
    }
    await _emit_response_done(response_id, snap.status, [message, *function_calls], emit, telemetry=telemetry)


async def _emit_response_done(
    response_id: str,
    status: str,
    output: list[dict[str, Any]],
    emit: EmitFn,
    *,
    telemetry: RealtimeResponseTelemetry | None,
) -> None:
    response: dict[str, Any] = {"id": response_id, "status": status, "output": output}
    if telemetry is not None:
        if telemetry.usage is not None:
            response["usage"] = telemetry.usage
        if telemetry.metadata is not None:
            response["metadata"] = telemetry.metadata
    await emit_with_aliases(
        emit,
        server_event(
            SERVER_RESPONSE_DONE,
            response=response,
        ),
    )
