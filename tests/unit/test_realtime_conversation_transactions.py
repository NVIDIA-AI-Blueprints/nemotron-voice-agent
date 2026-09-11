# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Failure-injection tests for Realtime conversation/context transactions."""

from __future__ import annotations

import asyncio
import json
import unittest
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from realtime.controller import RealtimeSessionController
from realtime.frames import RealtimeConversationAppendFrame
from realtime.protocol import RealtimeProtocolError
from realtime.serializer import RealtimeFrameSerializer
from realtime.transport import RealtimeManualResponseGate


class _Recorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def emit(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    async def emit_batch(self, events: list[dict[str, Any]]) -> None:
        self.events.extend(events)


def _controller() -> RealtimeSessionController:
    return RealtimeSessionController(model="test-model", voice="test-voice", runtime_config={})


def _serializer(
    controller: RealtimeSessionController,
    context: LLMContext,
) -> tuple[RealtimeFrameSerializer, _Recorder]:
    recorder = _Recorder()
    serializer = RealtimeFrameSerializer(controller=controller)
    serializer.set_emit(recorder.emit, recorder.emit_batch)
    serializer.bind_context(context)
    return serializer, recorder


def _add_text_item(
    controller: RealtimeSessionController,
    serializer: RealtimeFrameSerializer,
    *,
    item_id: str,
    context_message: dict[str, Any],
) -> None:
    controller.create_conversation_item(
        {
            "id": item_id,
            "type": "message",
            "role": context_message["role"],
            "content": [{"type": "input_text", "text": context_message["content"]}],
        }
    )
    serializer.bind_conversation_context_message(item_id, context_message)


def _finish_audio_item(controller: RealtimeSessionController) -> str:
    controller.start_response()
    for fragment in ("Hello", "there"):
        controller.output_audio_delta_event("encoded", sample_count=2_400, sample_rate=24_000)
        controller.append_assistant_audio_transcript(
            fragment,
            includes_inter_frame_spaces=False,
            context_text=fragment,
        )
    controller.output_audio_delta_event("encoded", sample_count=2_400, sample_rate=24_000)
    controller.finish_response(status="completed")
    item_id = controller.last_assistant_item_id
    assert item_id is not None
    return item_id


def _fail_once_after(original: Callable[..., Any]) -> Callable[..., Any]:
    failed = False

    def fail(*args: Any, **kwargs: Any) -> Any:
        nonlocal failed
        result = original(*args, **kwargs)
        if not failed:
            failed = True
            raise RuntimeError("injected conversation transaction failure")
        return result

    return fail


class ConversationMutationTransactionTests(unittest.IsolatedAsyncioTestCase):
    """Verify delete and truncate keep their controller/context owners atomic."""

    async def test_delete_rolls_back_each_mutation_boundary(self) -> None:
        """Restore every exact owner when any delete commit boundary fails."""
        for boundary in ("context", "controller", "ownership"):
            with self.subTest(boundary=boundary):
                controller = _controller()
                first = {"role": "user", "content": "first"}
                second = {"role": "user", "content": "second"}
                context = LLMContext([first, second])
                serializer, recorder = _serializer(controller, context)
                _add_text_item(controller, serializer, item_id="item_first", context_message=first)
                _add_text_item(controller, serializer, item_id="item_second", context_message=second)
                controller_state = controller.snapshot_conversation_mutation_state()
                owners = dict(serializer._context_messages_by_item_id)
                applied = dict(serializer._context_applied_events)
                cleared = set(serializer._context_cleared_item_ids)

                target, attribute = {
                    "context": (context, "set_messages"),
                    "controller": (controller, "delete_item_event"),
                    "ownership": (serializer, "_apply_conversation_ownership"),
                }[boundary]
                original = getattr(target, attribute)
                with patch.object(target, attribute, side_effect=_fail_once_after(original)):
                    frame = await serializer.deserialize(
                        json.dumps({"type": "conversation.item.delete", "item_id": "item_first"})
                    )

                self.assertIsNone(frame)
                self.assertEqual(recorder.events[-1]["error"]["code"], "event_processing_failed")
                self.assertFalse(serializer.connection_closed)
                self.assertEqual(controller.snapshot_conversation_mutation_state(), controller_state)
                self.assertEqual(context.get_messages(), [first, second])
                self.assertIs(context.get_messages()[0], first)
                self.assertIs(context.get_messages()[1], second)
                self.assertEqual(serializer._context_messages_by_item_id, owners)
                self.assertEqual(serializer._context_applied_events, applied)
                self.assertEqual(serializer._context_cleared_item_ids, cleared)

                recorder.events.clear()
                await serializer.deserialize(json.dumps({"type": "conversation.item.delete", "item_id": "item_first"}))
                self.assertEqual(recorder.events[-1]["type"], "conversation.item.deleted")
                self.assertIs(context.get_messages()[0], second)
                with self.assertRaises(RealtimeProtocolError):
                    controller.conversation.item("item_first")

    async def test_truncate_rolls_back_each_mutation_boundary(self) -> None:
        """Restore journal, alignment, and context after any truncate failure."""
        for boundary in ("context", "controller", "ownership"):
            with self.subTest(boundary=boundary):
                controller = _controller()
                item_id = _finish_audio_item(controller)
                prefix = {"role": "user", "content": "question"}
                assistant = {"role": "assistant", "content": "Hello there"}
                context = LLMContext([prefix, assistant])
                serializer, recorder = _serializer(controller, context)
                self.assertTrue(serializer.bind_latest_assistant_context_message())
                controller_state = controller.snapshot_conversation_mutation_state()
                owners = dict(serializer._context_messages_by_item_id)

                target, attribute = {
                    "context": (context, "set_messages"),
                    "controller": (controller, "truncate_item_event"),
                    "ownership": (serializer, "_apply_conversation_ownership"),
                }[boundary]
                original = getattr(target, attribute)
                with patch.object(target, attribute, side_effect=_fail_once_after(original)):
                    frame = await serializer.deserialize(
                        json.dumps(
                            {
                                "type": "conversation.item.truncate",
                                "item_id": item_id,
                                "content_index": 0,
                                "audio_end_ms": 100,
                            }
                        )
                    )

                self.assertIsNone(frame)
                self.assertEqual(recorder.events[-1]["error"]["code"], "event_processing_failed")
                self.assertFalse(serializer.connection_closed)
                self.assertEqual(controller.snapshot_conversation_mutation_state(), controller_state)
                self.assertEqual(context.get_messages(), [prefix, assistant])
                self.assertIs(context.get_messages()[0], prefix)
                self.assertIs(context.get_messages()[1], assistant)
                self.assertEqual(serializer._context_messages_by_item_id, owners)

                recorder.events.clear()
                await serializer.deserialize(
                    json.dumps(
                        {
                            "type": "conversation.item.truncate",
                            "item_id": item_id,
                            "content_index": 0,
                            "audio_end_ms": 100,
                        }
                    )
                )
                self.assertEqual(recorder.events[-1]["type"], "conversation.item.truncated")
                self.assertEqual(controller.conversation.item(item_id)["content"][0]["transcript"], "Hello")
                self.assertIs(context.get_messages()[0], prefix)
                self.assertEqual(context.get_messages()[1], {"role": "assistant", "content": "Hello"})

    async def test_rollback_failure_retires_connection(self) -> None:
        """Reject later events when an external context setter cannot roll back."""
        controller = _controller()
        message = {"role": "user", "content": "keep me"}
        context = LLMContext([message])
        serializer, recorder = _serializer(controller, context)
        _add_text_item(controller, serializer, item_id="item_keep", context_message=message)
        original = context.set_messages

        def fail_after_mutation(value: list[dict[str, Any]]) -> None:
            original(value)
            raise RuntimeError("persistent context failure")

        with patch.object(context, "set_messages", side_effect=fail_after_mutation):
            await serializer.deserialize(json.dumps({"type": "conversation.item.delete", "item_id": "item_keep"}))

        self.assertTrue(serializer.connection_closed)
        self.assertEqual(recorder.events[-1]["error"]["code"], "event_processing_failed")
        event_count = len(recorder.events)
        await serializer.deserialize(json.dumps({"type": "conversation.item.retrieve", "item_id": "item_keep"}))
        self.assertEqual(len(recorder.events), event_count)

    async def test_wire_failure_keeps_committed_delete_and_retires_connection(self) -> None:
        """Do not undo a delete whose acknowledgement may have crossed the wire."""
        controller = _controller()
        message = {"role": "user", "content": "delete me"}
        context = LLMContext([message])
        serializer, recorder = _serializer(controller, context)
        _add_text_item(controller, serializer, item_id="item_delete", context_message=message)
        original_emit = serializer._emit_event
        failed = False

        async def fail_once(event: dict[str, Any]) -> None:
            nonlocal failed
            if not failed:
                failed = True
                raise RuntimeError("wire failure")
            await original_emit(event)

        with patch.object(serializer, "_emit_event", side_effect=fail_once):
            await serializer.deserialize(json.dumps({"type": "conversation.item.delete", "item_id": "item_delete"}))

        self.assertTrue(serializer.connection_closed)
        self.assertEqual(context.get_messages(), [])
        with self.assertRaises(RealtimeProtocolError):
            controller.conversation.item("item_delete")
        self.assertEqual(recorder.events[-1]["error"]["code"], "event_processing_failed")


class ConversationCreateTransactionTests(unittest.IsolatedAsyncioTestCase):
    """Verify provisional text-item creation fails closed at its context owner."""

    async def _runtime(
        self,
    ) -> tuple[RealtimeSessionController, RealtimeFrameSerializer, RealtimeManualResponseGate, LLMContext, _Recorder]:
        controller = _controller()
        context = LLMContext([])
        serializer, recorder = _serializer(controller, context)
        gate = RealtimeManualResponseGate(controller=controller)
        serializer.set_response_gate(gate)
        return controller, serializer, gate, context, recorder

    @staticmethod
    def _event() -> str:
        return json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "id": "item_create",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            }
        )

    async def test_controller_create_failure_rolls_back_before_pipeline_handoff(self) -> None:
        """Restore the journal if creating the provisional item fails midway."""
        controller, serializer, _gate, context, recorder = await self._runtime()
        original = controller.create_conversation_item
        with patch.object(
            controller,
            "create_conversation_item",
            side_effect=_fail_once_after(original),
        ):
            frame = await serializer.deserialize(self._event())

        self.assertIsNone(frame)
        self.assertFalse(serializer.connection_closed)
        self.assertEqual(context.get_messages(), [])
        self.assertEqual(controller.conversation.ordered_item_ids(), ())
        self.assertEqual(serializer._pending_conversation_appends, {})
        self.assertEqual(recorder.events[-1]["error"]["code"], "event_processing_failed")

    async def test_context_append_failure_rolls_back_and_retires_dependent_pipeline(self) -> None:
        """Undo a provisional item when its exact context append fails."""
        controller, serializer, gate, context, _recorder = await self._runtime()
        sentinel = {"role": "system", "content": "prompt"}
        context.add_message(sentinel)
        frame = await serializer.deserialize(self._event())
        self.assertIsInstance(frame, RealtimeConversationAppendFrame)
        original = context.add_message

        with (
            patch.object(context, "add_message", side_effect=_fail_once_after(original)),
            self.assertRaisesRegex(RuntimeError, "injected conversation transaction failure"),
        ):
            await gate.process_frame(frame, FrameDirection.DOWNSTREAM)

        self.assertTrue(serializer.connection_closed)
        self.assertEqual(context.get_messages(), [sentinel])
        self.assertIs(context.get_messages()[0], sentinel)
        self.assertEqual(controller.conversation.ordered_item_ids(), ())
        self.assertEqual(serializer._pending_conversation_appends, {})

    async def test_context_owner_failure_rolls_back_and_retires_dependent_pipeline(self) -> None:
        """Undo both owners when exact identity binding fails after context append."""
        controller, serializer, gate, context, _recorder = await self._runtime()
        frame = await serializer.deserialize(self._event())
        self.assertIsInstance(frame, RealtimeConversationAppendFrame)
        original = serializer.bind_conversation_context_message

        with (
            patch.object(
                serializer,
                "bind_conversation_context_message",
                side_effect=_fail_once_after(original),
            ),
            self.assertRaisesRegex(RuntimeError, "injected conversation transaction failure"),
        ):
            await gate.process_frame(frame, FrameDirection.DOWNSTREAM)

        self.assertTrue(serializer.connection_closed)
        self.assertEqual(context.get_messages(), [])
        self.assertEqual(controller.conversation.ordered_item_ids(), ())
        self.assertEqual(serializer._context_messages_by_item_id, {})

    async def test_unrelated_delete_waits_for_preceding_create_publication(self) -> None:
        """Do not acknowledge a later journal mutation ahead of a queued create."""
        controller, serializer, gate, context, recorder = await self._runtime()
        existing = {"role": "user", "content": "existing"}
        context.add_message(existing)
        _add_text_item(
            controller,
            serializer,
            item_id="item_existing",
            context_message=existing,
        )
        append = await serializer.deserialize(self._event())
        self.assertIsInstance(append, RealtimeConversationAppendFrame)

        deletion = asyncio.create_task(
            serializer.deserialize(
                json.dumps(
                    {
                        "type": "conversation.item.delete",
                        "item_id": "item_existing",
                    }
                )
            )
        )
        await asyncio.sleep(0)

        self.assertFalse(deletion.done())
        self.assertEqual(recorder.events, [])

        await gate.process_frame(append, FrameDirection.DOWNSTREAM)
        await asyncio.wait_for(deletion, timeout=1)

        self.assertEqual(
            [event["type"] for event in recorder.events],
            ["conversation.item.added", "conversation.item.done", "conversation.item.deleted"],
        )
        self.assertEqual(context.get_messages(), [append.context_message])
        self.assertEqual(controller.conversation.ordered_item_ids(), ("item_create",))

    async def test_concurrent_provider_mutation_is_not_erased_by_create_rollback(self) -> None:
        """Retire instead of restoring a stale snapshot across provider output."""
        controller, serializer, gate, context, _recorder = await self._runtime()
        append = await serializer.deserialize(self._event())
        self.assertIsInstance(append, RealtimeConversationAppendFrame)

        controller.start_response()
        provider_events = controller.append_assistant_text("provider output")
        assistant_item_id = next(
            event["item"]["id"] for event in provider_events if event["type"] == "conversation.item.added"
        )

        with (
            patch.object(context, "add_message", side_effect=RuntimeError("context append failed")),
            self.assertRaisesRegex(RuntimeError, "connection retired"),
        ):
            await gate.process_frame(append, FrameDirection.DOWNSTREAM)

        self.assertTrue(serializer.connection_closed)
        self.assertEqual(
            controller.conversation.ordered_item_ids(),
            ("item_create", assistant_item_id),
        )
        assistant_item = controller.conversation.item(assistant_item_id)
        self.assertEqual(assistant_item["role"], "assistant")
        self.assertEqual(assistant_item["content"][0]["transcript"], "provider output")
        self.assertEqual(context.get_messages(), [])

    async def test_publication_barrier_timeout_retires_connection(self) -> None:
        """Bound a stalled provisional create before accepting later mutations."""
        controller, serializer, _gate, context, recorder = await self._runtime()
        existing = {"role": "user", "content": "existing"}
        context.add_message(existing)
        _add_text_item(
            controller,
            serializer,
            item_id="item_existing",
            context_message=existing,
        )
        append = await serializer.deserialize(self._event())
        self.assertIsInstance(append, RealtimeConversationAppendFrame)

        with patch("realtime.serializer._CONTEXT_APPLY_TIMEOUT_SECS", 0.01):
            await serializer.deserialize(
                json.dumps(
                    {
                        "type": "conversation.item.delete",
                        "item_id": "item_existing",
                    }
                )
            )

        self.assertTrue(serializer.connection_closed)
        self.assertEqual(serializer._pending_conversation_appends, {})
        self.assertEqual(recorder.events[-1]["error"]["code"], "conversation_context_apply_timeout")
        self.assertEqual(controller.conversation.ordered_item_ids(), ("item_existing", "item_create"))
        self.assertEqual(context.get_messages(), [existing])

    async def test_publication_failure_keeps_committed_create_and_retires_connection(self) -> None:
        """Keep a create that might be public and close after publication failure."""
        controller, serializer, gate, context, _recorder = await self._runtime()
        frame = await serializer.deserialize(self._event())
        self.assertIsInstance(frame, RealtimeConversationAppendFrame)

        with (
            patch.object(serializer, "_emit_events", side_effect=RuntimeError("wire failure")),
            self.assertRaisesRegex(RuntimeError, "wire failure"),
        ):
            await gate.process_frame(frame, FrameDirection.DOWNSTREAM)

        self.assertTrue(serializer.connection_closed)
        self.assertEqual(controller.conversation.ordered_item_ids(), ("item_create",))
        self.assertEqual(context.get_messages(), [frame.context_message])
        self.assertIs(context.get_messages()[0], frame.context_message)
