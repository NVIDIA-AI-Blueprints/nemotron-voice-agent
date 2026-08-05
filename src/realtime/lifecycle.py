# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Shared announce / finish sequences for Realtime responses."""

from __future__ import annotations

from realtime.conversation import ConversationState, ResponseSnapshot
from realtime.events import (
    SERVER_CONTENT_PART_ADDED,
    SERVER_CONTENT_PART_DONE,
    SERVER_ITEM_CREATED,
    SERVER_OUTPUT_AUDIO_DONE,
    SERVER_OUTPUT_AUDIO_TRANSCRIPT_DONE,
    SERVER_OUTPUT_ITEM_ADDED,
    SERVER_OUTPUT_ITEM_DONE,
    SERVER_OUTPUT_TEXT_DONE,
    SERVER_RESPONSE_CREATED,
    SERVER_RESPONSE_DONE,
    EmitFn,
    emit_with_aliases,
    response_created_body,
    server_event,
)


async def announce_response(conversation: ConversationState, emit: EmitFn) -> tuple[str, bool]:
    """Ensure ``response.created`` + assistant item / content_part are on the wire.

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


async def finish_response(
    conversation: ConversationState,
    emit: EmitFn,
    *,
    status: str,
    output_text: str = "",
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
    await emit_finish_from_snapshot(snap, emit, output_text=output_text)
    conversation.reset_response_slot(generation=snap.generation)
    return True


async def emit_finish_from_snapshot(
    snap: ResponseSnapshot,
    emit: EmitFn,
    *,
    output_text: str = "",
) -> None:
    """Emit audio/text/item/response done events for a completed snapshot."""
    response_id = snap.response_id
    item_id = snap.item_id
    transcript = snap.transcript
    # Item statuses are in_progress|completed|incomplete; cancelled is response-only.
    item_status = "completed" if snap.status == "completed" else "incomplete"
    text_done = output_text or transcript

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
    if snap.output_text_emitted:
        await emit_with_aliases(
            emit,
            server_event(
                SERVER_OUTPUT_TEXT_DONE,
                response_id=response_id,
                item_id=item_id,
                output_index=0,
                content_index=0,
                text=text_done,
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
                "content": [{"type": "audio", "transcript": transcript}],
            },
        ),
    )
    await emit_with_aliases(
        emit,
        server_event(
            SERVER_RESPONSE_DONE,
            response={
                "id": response_id,
                "status": snap.status,
                "output": [
                    {
                        "id": item_id,
                        "type": "message",
                        "role": "assistant",
                        "status": item_status,
                        "content": [{"type": "audio", "transcript": transcript}],
                    }
                ],
            },
        ),
    )
