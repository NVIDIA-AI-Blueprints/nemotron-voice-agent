# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""An assistant reply that never reached model context can still be truncated or deleted."""

# ruff: noqa: D101, D102

from __future__ import annotations

import json
import unittest
from typing import Any
from unittest.mock import patch

from pipecat.processors.aggregators.llm_context import LLMContext

from realtime.controller import RealtimeSessionController
from realtime.serializer import RealtimeFrameSerializer


class _Recorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def emit(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    async def emit_batch(self, events: list[dict[str, Any]]) -> None:
        self.events.extend(events)


def _runtime(messages: list[dict[str, Any]] | None = None):
    controller = RealtimeSessionController(model="test-model", voice="test-voice", runtime_config={})
    context = LLMContext(list(messages or []))
    recorder = _Recorder()
    serializer = RealtimeFrameSerializer(controller=controller)
    serializer.set_emit(recorder.emit, recorder.emit_batch)
    serializer.bind_context(context)
    return controller, serializer, context, recorder


def _spoken_item(controller: RealtimeSessionController, serializer: RealtimeFrameSerializer, *, status: str) -> str:
    """Speak one assistant reply, arm its context barrier, and end it with ``status``."""
    controller.start_response()
    for fragment in ("Checking", "that"):
        controller.output_audio_delta_event("encoded", sample_count=2_400, sample_rate=24_000)
        controller.append_assistant_audio_transcript(fragment, includes_inter_frame_spaces=False, context_text=fragment)
    controller.output_audio_delta_event("encoded", sample_count=2_400, sample_rate=24_000)
    item_id = controller.assistant_item_id
    assert item_id is not None
    controller.finish_response(status=status, **({"reason": "turn_detected"} if status == "cancelled" else {}))
    serializer.prepare_response_done_publication("resp_test")
    assert item_id in serializer._context_applied_events
    return item_id


def _truncate(item_id: str) -> str:
    return json.dumps(
        {"type": "conversation.item.truncate", "item_id": item_id, "content_index": 0, "audio_end_ms": 100}
    )


class UnboundAssistantEditTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_cancelled_unbound_reply_is_truncated_without_failing_the_session(self) -> None:
        tool_message = {"role": "tool", "content": "IN_PROGRESS", "tool_call_id": "call_1"}
        controller, serializer, context, recorder = _runtime([tool_message])
        item_id = _spoken_item(controller, serializer, status="cancelled")
        self.assertEqual(controller.conversation.item(item_id)["status"], "incomplete")

        with patch("realtime.serializer._CANCELLED_CONTEXT_APPLY_GRACE_SECS", 0.01):
            await serializer.deserialize(_truncate(item_id))

        self.assertFalse(serializer.connection_closed)
        self.assertEqual(recorder.events[-1]["type"], "conversation.item.truncated")
        self.assertEqual(context.get_messages(), [tool_message])
        self.assertIn(item_id, serializer._context_detached_item_ids)

    async def test_a_late_bind_does_not_attach_the_full_text_afterwards(self) -> None:
        controller, serializer, context, _recorder = _runtime()
        item_id = _spoken_item(controller, serializer, status="cancelled")
        with patch("realtime.serializer._CANCELLED_CONTEXT_APPLY_GRACE_SECS", 0.01):
            await serializer.deserialize(_truncate(item_id))

        context.add_message({"role": "assistant", "content": "Checking that"})

        self.assertFalse(serializer.bind_latest_assistant_context_message())
        self.assertNotIn(item_id, serializer._context_messages_by_item_id)

    async def test_a_finished_unbound_reply_can_be_deleted(self) -> None:
        controller, serializer, context, recorder = _runtime()
        item_id = _spoken_item(controller, serializer, status="cancelled")

        with patch("realtime.serializer._CANCELLED_CONTEXT_APPLY_GRACE_SECS", 0.01):
            await serializer.deserialize(json.dumps({"type": "conversation.item.delete", "item_id": item_id}))

        self.assertFalse(serializer.connection_closed)
        self.assertEqual(recorder.events[-1]["type"], "conversation.item.deleted")
        self.assertNotIn(item_id, controller.conversation.ordered_item_ids())

    async def test_a_bound_reply_still_edits_model_context(self) -> None:
        controller, serializer, context, recorder = _runtime()
        item_id = _spoken_item(controller, serializer, status="completed")
        context.add_message({"role": "assistant", "content": "Checking that"})
        self.assertTrue(serializer.bind_latest_assistant_context_message())

        await serializer.deserialize(_truncate(item_id))

        self.assertEqual(recorder.events[-1]["type"], "conversation.item.truncated")
        self.assertEqual(context.get_messages()[-1]["content"], "Checking")
        self.assertNotIn(item_id, serializer._context_detached_item_ids)

    async def test_a_completed_reply_keeps_the_full_wait_before_editing_alone(self) -> None:
        controller, serializer, _context, recorder = _runtime()
        item_id = _spoken_item(controller, serializer, status="completed")

        with (
            patch("realtime.serializer._CANCELLED_CONTEXT_APPLY_GRACE_SECS", 60.0),
            patch("realtime.serializer._CONTEXT_APPLY_TIMEOUT_SECS", 0.01),
        ):
            await serializer.deserialize(_truncate(item_id))

        self.assertFalse(serializer.connection_closed)
        self.assertEqual(recorder.events[-1]["type"], "conversation.item.truncated")


if __name__ == "__main__":
    unittest.main()
