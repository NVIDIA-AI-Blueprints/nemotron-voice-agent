# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pipecat.frames.frames import FunctionCallResultFrame, FunctionCallsStartedFrame, LLMRunFrame
from pipecat.processors.frame_processor import FrameDirection

from realtime.client_tools import MAX_TOOL_OUTPUT_BYTES, ClientToolBroker, RealtimeToolCoordinator
from realtime.conversation import ConversationState
from realtime.observer import RealtimeLifecycleObserver
from realtime.serializer import RealtimeFrameSerializer

CLIENT_TOOL = {
    "type": "function",
    "name": "lookup_order",
    "description": "Look up an order",
    "parameters": {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
}
SECOND_CLIENT_TOOL = {
    **CLIENT_TOOL,
    "name": "lookup_customer",
}


def _call(call_id: str = "call_client", name: str = "lookup_order"):
    return SimpleNamespace(
        tool_call_id=call_id,
        function_name=name,
        arguments={"order_id": "A-1"},
    )


class ClientToolBrokerTests(unittest.IsolatedAsyncioTestCase):
    async def test_deferred_handler_accepts_one_output_without_auto_run(self) -> None:
        broker = ClientToolBroker(client_tools=[CLIENT_TOOL])
        broker.register_batch([_call()])
        callbacks: list[tuple[str, object]] = []

        async def result_callback(result, *, properties=None):
            callbacks.append((result, properties))

        params = SimpleNamespace(tool_call_id="call_client", result_callback=result_callback)
        handler = asyncio.create_task(broker.client_handler(params))
        await asyncio.sleep(0)
        await broker.submit_output("call_client", '{"status":"shipped"}')
        await handler

        self.assertEqual(callbacks[0][0], {"status": "shipped"})
        self.assertFalse(callbacks[0][1].run_llm)
        with self.assertRaisesRegex(ValueError, "duplicate_call_output"):
            await broker.submit_output("call_client", "{}")

    async def test_expired_and_cancelled_calls_reject_late_output(self) -> None:
        emitted: list[dict] = []

        async def emit(event: dict) -> None:
            emitted.append(event)

        broker = ClientToolBroker(client_tools=[CLIENT_TOOL])
        broker.timeout_secs = 0.01
        broker.set_emit(emit)
        broker.register_batch([_call()])

        async def result_callback(result, *, properties=None):  # noqa: ARG001
            return None

        await broker.client_handler(SimpleNamespace(tool_call_id="call_client", result_callback=result_callback))
        self.assertEqual(emitted[-1]["error"]["code"], "client_tool_timeout")
        with self.assertRaisesRegex(ValueError, "stale_call_id"):
            await broker.submit_output("call_client", "{}")

        other = ClientToolBroker(client_tools=[CLIENT_TOOL])
        other.register_batch([_call("call_cancel")])
        await other.cancel_call("call_cancel")
        with self.assertRaisesRegex(ValueError, "stale_call_id"):
            await other.submit_output("call_cancel", "{}")

    async def test_mixed_batch_suppresses_every_automatic_continuation(self) -> None:
        broker = ClientToolBroker(client_tools=[CLIENT_TOOL], server_tools=["get_weather"])
        broker.register_batch([_call(), _call("call_server", "get_weather")])
        coordinator = RealtimeToolCoordinator(broker)
        coordinator.push_frame = AsyncMock()  # type: ignore[method-assign]

        result = FunctionCallResultFrame(
            function_name="get_weather",
            tool_call_id="call_server",
            arguments={},
            result={"sunny": True},
        )
        await coordinator.process_frame(result, FrameDirection.DOWNSTREAM)
        self.assertFalse(result.run_llm)
        self.assertFalse(result.properties.run_llm)

    async def test_queued_continuation_releases_once_after_result_is_applied(self) -> None:
        queued = AsyncMock()
        broker = ClientToolBroker(client_tools=[CLIENT_TOOL])
        broker.set_queue_run(queued)
        broker.register_batch([_call()])
        await broker.submit_output("call_client", "{}")
        self.assertEqual(await broker.prepare_continuation(), "queued")
        await broker.mark_result_applied("call_client")
        await broker.mark_result_applied("call_client")
        queued.assert_awaited_once()

    async def test_parallel_client_calls_require_every_output(self) -> None:
        broker = ClientToolBroker(client_tools=[CLIENT_TOOL, SECOND_CLIENT_TOOL])
        broker.register_batch([_call(), _call("call_customer", "lookup_customer")])
        await broker.submit_output("call_client", "{}")
        self.assertEqual(await broker.prepare_continuation(), "missing_output")
        await broker.submit_output("call_customer", "{}")
        await broker.mark_result_applied("call_client")
        await broker.mark_result_applied("call_customer")
        self.assertEqual(await broker.prepare_continuation(), "ready")

    async def test_oversized_output_is_rejected(self) -> None:
        broker = ClientToolBroker(client_tools=[CLIENT_TOOL])
        broker.register_batch([_call()])
        with self.assertRaisesRegex(ValueError, "tool_output_too_large"):
            await broker.submit_output("call_client", "x" * (MAX_TOOL_OUTPUT_BYTES + 1))

    async def test_disconnect_cancels_deferred_handler(self) -> None:
        broker = ClientToolBroker(client_tools=[CLIENT_TOOL])
        broker.register_batch([_call()])

        async def result_callback(result, *, properties=None):  # noqa: ARG001
            return None

        handler = asyncio.create_task(
            broker.client_handler(SimpleNamespace(tool_call_id="call_client", result_callback=result_callback))
        )
        await asyncio.sleep(0)
        await broker.close()
        with self.assertRaises(asyncio.CancelledError):
            await handler


class ClientToolWireTests(unittest.IsolatedAsyncioTestCase):
    async def test_function_call_is_a_completed_non_audio_response(self) -> None:
        emitted: list[dict] = []

        async def emit(event: dict) -> None:
            emitted.append(event)

        broker = ClientToolBroker(client_tools=[CLIENT_TOOL])
        state = ConversationState(assistant_has_responded=True)
        observer = RealtimeLifecycleObserver(emit=emit, conversation=state, client_tool_broker=broker)
        await observer._handle_frame(FunctionCallsStartedFrame(function_calls=[_call()]))

        types = [event["type"] for event in emitted]
        self.assertEqual(
            types,
            [
                "response.created",
                "conversation.item.created",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.done",
            ],
        )
        self.assertNotIn("response.output_audio.done", types)
        done = emitted[-1]["response"]
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["output"][0]["type"], "function_call")
        self.assertEqual(done["output"][0]["call_id"], "call_client")
        self.assertIsNone(state.response_id)

    async def test_parallel_function_calls_preserve_output_indices(self) -> None:
        emitted: list[dict] = []

        async def emit(event: dict) -> None:
            emitted.append(event)

        broker = ClientToolBroker(client_tools=[CLIENT_TOOL, SECOND_CLIENT_TOOL])
        observer = RealtimeLifecycleObserver(
            emit=emit,
            conversation=ConversationState(assistant_has_responded=True),
            client_tool_broker=broker,
        )
        await observer._handle_frame(
            FunctionCallsStartedFrame(
                function_calls=[
                    _call(),
                    _call("call_customer", "lookup_customer"),
                ]
            )
        )
        added = [event for event in emitted if event["type"] == "response.output_item.added"]
        self.assertEqual([event["output_index"] for event in added], [0, 1])
        done = emitted[-1]["response"]
        self.assertEqual([item["call_id"] for item in done["output"]], ["call_client", "call_customer"])

    async def test_output_is_accepted_then_explicit_response_create_runs(self) -> None:
        emitted: list[dict] = []

        async def emit(event: dict) -> None:
            emitted.append(event)

        broker = ClientToolBroker(client_tools=[CLIENT_TOOL])
        broker.register_batch([_call()])
        serializer = RealtimeFrameSerializer(
            session_view={"tools": [CLIENT_TOOL]},
            runtime_config={"pipeline_mode": "generic-assistant", "client_tools": [CLIENT_TOOL]},
            conversation=ConversationState(assistant_has_responded=True),
            client_tool_broker=broker,
        )
        serializer.set_emit(emit)
        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": "call_client",
                        "output": '{"status":"shipped"}',
                    },
                }
            )
        )
        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["type"], "conversation.item.created")
        await broker.mark_result_applied("call_client")

        frame = await serializer.deserialize(json.dumps({"type": "response.create"}))
        self.assertIsInstance(frame, LLMRunFrame)
        self.assertEqual(emitted[-1]["type"], "response.created")

    async def test_live_tools_update_calls_pipeline_hook(self) -> None:
        emitted: list[dict] = []
        updates: list[tuple[list[dict], object]] = []

        async def emit(event: dict) -> None:
            emitted.append(event)

        async def update(tools: list[dict], choice) -> None:
            updates.append((tools, choice))

        broker = ClientToolBroker()
        broker.set_tools_update(update)
        serializer = RealtimeFrameSerializer(
            session_view={"tools": [], "tool_choice": "auto"},
            runtime_config={"pipeline_mode": "generic-assistant", "server_tools": []},
            client_tool_broker=broker,
        )
        serializer.set_emit(emit)
        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "tools": [CLIENT_TOOL],
                        "tool_choice": {"type": "function", "name": "lookup_order"},
                    },
                }
            )
        )
        self.assertIsNone(frame)
        self.assertEqual(updates[0][0][0]["name"], "lookup_order")
        self.assertEqual(updates[0][1]["function"]["name"], "lookup_order")
        self.assertEqual(emitted[-1]["type"], "session.updated")
