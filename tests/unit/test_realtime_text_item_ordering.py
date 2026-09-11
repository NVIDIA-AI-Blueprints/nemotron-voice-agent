# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Ordering tests for client text items followed by explicit responses."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from realtime.controller import RealtimeSessionController
from realtime.frames import (
    RealtimeConversationAppendFrame,
    RealtimeDeferredResponseCreateFrame,
    RealtimeResponseContextFrame,
)
from realtime.serializer import RealtimeFrameSerializer
from realtime.transport import RealtimeManualResponseGate


class _Recorder:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def emit(self, event: dict) -> None:
        self.events.append(event)

    async def emit_batch(self, events: list[dict]) -> None:
        self.events.extend(events)

    @property
    def types(self) -> list[str]:
        return [event["type"] for event in self.events]


def _runtime() -> tuple[
    RealtimeFrameSerializer,
    RealtimeSessionController,
    RealtimeManualResponseGate,
    LLMContext,
    _Recorder,
]:
    controller = RealtimeSessionController(
        model="test-model",
        voice="test-voice",
        runtime_config={"asr_server": "localhost:50051"},
    )
    controller.apply_session_update({"type": "realtime", "output_modalities": ["text"]})
    recorder = _Recorder()
    serializer = RealtimeFrameSerializer(controller=controller)
    serializer.set_emit(recorder.emit, recorder.emit_batch)
    context = LLMContext([])
    gate = RealtimeManualResponseGate(controller=controller)
    gate.push_frame = AsyncMock()
    serializer.bind_context(context, instructions_renderer=lambda _instructions: [])
    serializer.set_response_gate(gate)
    return serializer, controller, gate, context, recorder


async def _create_text_item(
    serializer: RealtimeFrameSerializer,
    *,
    item_id: str,
    text: str,
) -> RealtimeConversationAppendFrame:
    frame = await serializer.deserialize(
        json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "id": item_id,
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
    )
    assert isinstance(frame, RealtimeConversationAppendFrame)
    return frame


class RealtimeTextItemResponseOrderingTests(unittest.IsolatedAsyncioTestCase):
    """Verify sequential client events remain ordered at the pipeline and wire."""

    async def test_immediate_response_waits_for_item_context_and_terminal_publication(self) -> None:
        """Defer response activation until one preceding text item is durable."""
        serializer, controller, gate, context, recorder = _runtime()
        append = await _create_text_item(
            serializer,
            item_id="item_text_turn",
            text="Reply with one greeting.",
        )

        response = await serializer.deserialize(json.dumps({"type": "response.create", "event_id": "event_response"}))

        self.assertIsInstance(response, RealtimeDeferredResponseCreateFrame)
        self.assertEqual(response.conversation_item_ids, ("item_text_turn",))
        self.assertIsNone(controller.active_response_id)
        self.assertEqual(recorder.events, [])

        await gate.process_frame(append, FrameDirection.DOWNSTREAM)
        await gate.process_frame(response, FrameDirection.DOWNSTREAM)

        self.assertEqual(
            recorder.types,
            ["conversation.item.added", "conversation.item.done", "response.created"],
        )
        self.assertEqual(
            [recorder.events[index]["item"]["status"] for index in (0, 1)],
            ["completed", "completed"],
        )
        self.assertIs(context.get_messages()[0], append.context_message)
        response_context = gate.push_frame.await_args_list[-1].args[0]
        self.assertIsInstance(response_context, RealtimeResponseContextFrame)
        self.assertEqual(
            response_context.context.get_messages(),
            [{"role": "user", "content": "Reply with one greeting."}],
        )

    async def test_response_snapshots_all_preceding_unapplied_text_items_in_order(self) -> None:
        """Keep every accepted pending text append in its original order."""
        serializer, controller, gate, context, recorder = _runtime()
        first = await _create_text_item(serializer, item_id="item_first", text="First")
        second = await _create_text_item(serializer, item_id="item_second", text="Second")

        response = await serializer.deserialize(json.dumps({"type": "response.create"}))

        self.assertIsInstance(response, RealtimeDeferredResponseCreateFrame)
        self.assertEqual(response.conversation_item_ids, ("item_first", "item_second"))
        self.assertIsNone(controller.active_response_id)

        await gate.process_frame(first, FrameDirection.DOWNSTREAM)
        await gate.process_frame(second, FrameDirection.DOWNSTREAM)
        await gate.process_frame(response, FrameDirection.DOWNSTREAM)

        self.assertEqual(
            recorder.types,
            [
                "conversation.item.added",
                "conversation.item.done",
                "conversation.item.added",
                "conversation.item.done",
                "response.created",
            ],
        )
        self.assertEqual(
            context.get_messages(),
            [
                {"role": "user", "content": "First"},
                {"role": "user", "content": "Second"},
            ],
        )

    async def test_missing_context_times_out_without_starting_response(self) -> None:
        """Fail a stalled pipeline barrier without creating a phantom response."""
        serializer, controller, gate, _context, recorder = _runtime()
        await _create_text_item(serializer, item_id="item_stalled", text="Never applied")
        response = await serializer.deserialize(json.dumps({"type": "response.create", "event_id": "event_stalled"}))
        self.assertIsInstance(response, RealtimeDeferredResponseCreateFrame)

        with patch("realtime.serializer._CONTEXT_APPLY_TIMEOUT_SECS", 0.01):
            await asyncio.wait_for(
                gate.process_frame(response, FrameDirection.DOWNSTREAM),
                timeout=1,
            )

        self.assertIsNone(controller.active_response_id)
        self.assertEqual(recorder.types, ["error"])
        self.assertEqual(recorder.events[0]["error"]["code"], "conversation_context_apply_timeout")
        self.assertEqual(recorder.events[0]["error"]["event_id"], "event_stalled")

    async def test_second_response_while_first_is_deferred_has_native_correlated_error(self) -> None:
        """Reject duplicate response ownership without leaking an internal error."""
        serializer, controller, gate, _context, recorder = _runtime()
        await _create_text_item(serializer, item_id="item_first_response", text="First response")
        first = await serializer.deserialize(json.dumps({"type": "response.create", "event_id": "response_one"}))
        self.assertIsInstance(first, RealtimeDeferredResponseCreateFrame)

        second = await serializer.deserialize(json.dumps({"type": "response.create", "event_id": "response_two"}))

        self.assertIsNone(second)
        self.assertIsNone(controller.active_response_id)
        self.assertEqual(recorder.types, ["error"])
        self.assertEqual(recorder.events[0]["error"]["type"], "invalid_request_error")
        self.assertEqual(recorder.events[0]["error"]["code"], "response_in_progress")
        self.assertEqual(recorder.events[0]["error"]["event_id"], "response_two")
        gate.reset()

    async def test_cancel_during_deferred_activation_does_not_launch_provider(self) -> None:
        """Recheck a deferred owner after response.created publication yields."""
        serializer, controller, gate, _context, _recorder = _runtime()
        append = await _create_text_item(serializer, item_id="item_cancelled_response", text="Cancel this")
        response = await serializer.deserialize(json.dumps({"type": "response.create"}))
        self.assertIsInstance(response, RealtimeDeferredResponseCreateFrame)
        await gate.process_frame(append, FrameDirection.DOWNSTREAM)

        original_handler = gate._deferred_response_handler
        self.assertIsNotNone(original_handler)
        activated = asyncio.Event()
        release_handler = asyncio.Event()

        async def delayed_handler(frame, generation):
            response_id = await original_handler(frame, generation)
            activated.set()
            await release_handler.wait()
            return response_id

        gate.set_deferred_response_handler(delayed_handler)
        response_task = asyncio.create_task(gate.process_frame(response, FrameDirection.DOWNSTREAM))
        await asyncio.wait_for(activated.wait(), timeout=1)
        response_id = controller.active_response_id
        self.assertIsNotNone(response_id)

        cancel = await serializer.deserialize(json.dumps({"type": "response.cancel", "response_id": response_id}))
        self.assertIsNotNone(cancel)
        release_handler.set()
        await asyncio.wait_for(response_task, timeout=1)

        gate.push_frame.assert_not_awaited()
        self.assertIsNone(gate._running_response_id)
        self.assertIsNone(gate._pending_deferred_response_id)
        gate.reset()

    async def test_reset_drops_a_queued_deferred_response(self) -> None:
        """A socket close cannot turn an already queued frame into an invariant error."""
        serializer, _controller, gate, _context, _recorder = _runtime()
        await _create_text_item(serializer, item_id="item_closed_response", text="Never run")
        response = await serializer.deserialize(json.dumps({"type": "response.create"}))
        self.assertIsInstance(response, RealtimeDeferredResponseCreateFrame)

        serializer.notify_connection_closed()
        gate.reset()
        await asyncio.wait_for(
            gate.process_frame(response, FrameDirection.DOWNSTREAM),
            timeout=1,
        )

        gate.push_frame.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
