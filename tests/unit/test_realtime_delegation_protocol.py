# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D101, D102

"""Protocol tests for the delegation route options (parked rounds, late outputs, response ordering).

Each test replays the client's side of the Realtime protocol without a
network: the function call the agent publishes, the client's
``function_call_output`` and ``response.create``, and a server-VAD barge-in.
The same steps on a route without the options keep today's errors.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from realtime_helpers import FakeWebSocket

from realtime.client_tools import ClientToolBroker
from realtime.controller import RealtimeSessionController
from realtime.frames import RealtimeClientToolOutputFrame, RealtimeDeferredResponseCreateFrame
from realtime.protocol import RealtimeProtocolError
from realtime.serializer import RealtimeFrameSerializer
from realtime.session import RealtimeSessionCapabilities
from realtime.transport import (
    RealtimeManualResponseGate,
    bind_realtime_context,
    configure_realtime_delegation_protocol,
    create_realtime_transport,
    realtime_client_tool_executor,
    realtime_response_gate_processors,
    realtime_tool_result_processors,
    shutdown_realtime_transport,
)

VOICE = "Magpie-Multilingual.EN-US.Aria"
TOOL = "lookup_member"


def _schema() -> dict:
    return {
        "type": "function",
        "name": TOOL,
        "description": "Look up a member by id.",
        "parameters": {"type": "object", "properties": {"member_id": {"type": "string"}}},
    }


def _controller() -> RealtimeSessionController:
    controller = RealtimeSessionController(
        model="nvidia/nemotron-realtime-generic-frontend-backend",
        voice=VOICE,
        runtime_config={"pipeline_mode": "generic-frontend-backend-agent"},
        capabilities=RealtimeSessionCapabilities(voices=frozenset({VOICE}), function_tools=True),
        delegate_tools=["call_backend", "cancel_backend"],
    )
    controller.apply_session_update({"output_modalities": ["text"], "tools": [_schema()]})
    controller.bind_session_tool_projection(client_tool_bindings={TOOL: TOOL}, mcp_pipeline_names=frozenset())
    return controller


class _Session:
    """One fake-socket transport with a pipeline-owned client-tool executor."""

    def __init__(self, *, delegation_route: bool) -> None:
        self.websocket = FakeWebSocket([])
        self.controller = _controller()
        self.transport = create_realtime_transport(self.websocket, controller=self.controller)
        [self.gate] = realtime_response_gate_processors(self.transport)
        bind_realtime_context(self.transport, LLMContext([]), render_instructions=lambda _instructions: [])
        realtime_tool_result_processors(self.transport)
        if delegation_route:
            self.assert_configured = configure_realtime_delegation_protocol(self.transport, enabled=True)
        self.executor = realtime_client_tool_executor(self.transport)
        self.serializer = self.transport.input()._params.serializer

    async def start_round(self, timeout_secs: float = 5.0) -> tuple[asyncio.Task, str]:
        task = asyncio.create_task(self.executor(((TOOL, {"member_id": "jordan_lee_82"}),), timeout_secs))
        for _attempt in range(400):
            done = [
                event for event in self.websocket.sent if event.get("type") == "response.function_call_arguments.done"
            ]
            if done:
                return task, done[-1]["call_id"]
            await asyncio.sleep(0.005)
        raise AssertionError("the client tool call never reached the wire")

    async def send_output(self, call_id: str, output: str = '{"ok": true}') -> RealtimeClientToolOutputFrame | None:
        """Send one output; return its frame, or None when the agent answered with an ``error`` event."""
        frame = await self.serializer.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "function_call_output", "call_id": call_id, "output": output},
                }
            )
        )
        if frame is None:
            return None
        assert isinstance(frame, RealtimeClientToolOutputFrame)
        await self.gate.process_frame(frame, FrameDirection.DOWNSTREAM)
        return frame

    def errors(self) -> list[dict]:
        return [event for event in self.websocket.sent if event.get("type") == "error"]

    def error_codes(self) -> list[str]:
        return [event["error"]["code"] for event in self.errors()]

    def close(self) -> None:
        shutdown_realtime_transport(self.transport)


class ParkedRoundTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_barge_in_keeps_a_parked_round_on_the_delegation_route(self) -> None:
        session = _Session(delegation_route=True)
        try:
            self.assertTrue(session.assert_configured)
            task, call_id = await session.start_round()
            session.controller.observe_interruption(1)
            session.controller.observe_interruption(2)

            await session.send_output(call_id)

            self.assertEqual(await asyncio.wait_for(task, timeout=2.0), ['{"ok": true}'])
            self.assertEqual(session.errors(), [])
        finally:
            session.close()

    async def test_a_barge_in_still_cancels_the_round_on_other_routes(self) -> None:
        session = _Session(delegation_route=False)
        try:
            task, call_id = await session.start_round()
            session.controller.observe_interruption(1)

            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2.0)
            self.assertIsNone(await session.send_output(call_id))
            self.assertEqual(session.error_codes(), ["client_tool_call_cancelled"])
        finally:
            session.close()


class LateOutputTests(unittest.IsolatedAsyncioTestCase):
    async def test_an_output_after_the_deadline_is_acknowledged_without_an_error(self) -> None:
        session = _Session(delegation_route=True)
        try:
            task, call_id = await session.start_round(timeout_secs=0.2)
            [timed_out] = await asyncio.wait_for(task, timeout=2.0)
            self.assertEqual(timed_out["error"]["code"], "client_tool_timeout")

            frame = await session.send_output(call_id)

            self.assertTrue(frame.late)
            added = [
                event
                for event in session.websocket.sent
                if event.get("type") in {"conversation.item.added", "conversation.item.done"}
                and event.get("item", {}).get("call_id") == call_id
                and event.get("item", {}).get("type") == "function_call_output"
            ]
            self.assertEqual([event["type"] for event in added], ["conversation.item.added", "conversation.item.done"])
            self.assertEqual(session.errors(), [])
        finally:
            session.close()

    async def test_a_second_output_for_a_settled_call_is_still_a_duplicate(self) -> None:
        session = _Session(delegation_route=True)
        try:
            task, call_id = await session.start_round(timeout_secs=0.2)
            await asyncio.wait_for(task, timeout=2.0)
            await session.send_output(call_id)

            self.assertIsNone(await session.send_output(call_id))
            self.assertEqual(session.error_codes(), ["duplicate_tool_output"])
        finally:
            session.close()

    async def test_an_output_after_the_deadline_is_an_error_on_other_routes(self) -> None:
        session = _Session(delegation_route=False)
        try:
            task, call_id = await session.start_round(timeout_secs=0.2)
            await asyncio.wait_for(task, timeout=2.0)

            self.assertIsNone(await session.send_output(call_id))
            self.assertEqual(session.error_codes(), ["client_tool_timeout"])
        finally:
            session.close()


class BrokerLateOutputTests(unittest.IsolatedAsyncioTestCase):
    async def _cancelled_call(self, *, accept: bool) -> ClientToolBroker:
        broker = ClientToolBroker(output_timeout_secs=5.0)
        broker.accept_late_outputs = accept
        await broker.register_direct_calls((("call_1", TOOL),), timeout_secs=5.0)
        await broker.cancel_direct_calls(("call_1",))
        return broker

    async def test_a_cancelled_call_settles_late_ok_once(self) -> None:
        broker = await self._cancelled_call(accept=True)

        self.assertTrue(await broker.stage_output(call_id="call_1", name=TOOL, output="{}"))
        with self.assertRaises(RealtimeProtocolError):
            await broker.stage_output(call_id="call_1", name=TOOL, output="{}")
        self.assertEqual(broker.pending_context_call_ids(), ())

    async def test_a_name_mismatch_is_rejected_even_when_late_outputs_are_accepted(self) -> None:
        broker = await self._cancelled_call(accept=True)

        with self.assertRaises(RealtimeProtocolError) as caught:
            await broker.stage_output(call_id="call_1", name="other_tool", output="{}")
        self.assertEqual(caught.exception.code, "tool_name_mismatch")

    async def test_a_cancelled_call_keeps_its_error_without_the_option(self) -> None:
        broker = await self._cancelled_call(accept=False)

        with self.assertRaises(RealtimeProtocolError) as caught:
            await broker.stage_output(call_id="call_1", name=TOOL, output="{}")
        self.assertEqual(caught.exception.code, "client_tool_call_cancelled")

    async def test_a_closed_broker_rejects_late_outputs(self) -> None:
        broker = await self._cancelled_call(accept=True)
        broker.shutdown()

        with self.assertRaises(RealtimeProtocolError) as caught:
            await broker.stage_output(call_id="call_1", name=TOOL, output="{}")
        self.assertEqual(caught.exception.code, "client_tool_broker_closed")


class ResponseOrderingTests(unittest.IsolatedAsyncioTestCase):
    def _serializer(self, *, delegation_route: bool) -> tuple[RealtimeFrameSerializer, RealtimeManualResponseGate]:
        controller = _controller()
        serializer = RealtimeFrameSerializer(controller=controller)
        self.sent: list[dict] = []

        async def emit(event: dict) -> None:
            self.sent.append(event)

        async def emit_batch(events: list[dict]) -> None:
            self.sent.extend(events)

        serializer.set_emit(emit, emit_batch)
        serializer.bind_context(LLMContext([]), instructions_renderer=lambda _instructions: [])
        gate = RealtimeManualResponseGate(controller=controller)
        gate.push_frame = AsyncMock()
        serializer.set_response_gate(gate)
        gate.defer_client_responses_behind_pipeline = delegation_route
        return serializer, gate

    async def test_no_pipeline_response_serves_the_client_request_at_once(self) -> None:
        serializer, _gate = self._serializer(delegation_route=True)

        frame = await serializer.deserialize(json.dumps({"type": "response.create"}))

        self.assertNotIsInstance(frame, RealtimeDeferredResponseCreateFrame)
        self.assertIsNotNone(serializer._controller.active_response_id)

    async def test_a_client_request_during_a_pipeline_response_waits_then_runs(self) -> None:
        serializer, gate = self._serializer(delegation_route=True)
        controller = serializer._controller
        controller.start_response()
        pipeline_response_id = controller.active_response_id

        frame = await serializer.deserialize(json.dumps({"type": "response.create"}))

        self.assertIsInstance(frame, RealtimeDeferredResponseCreateFrame)
        self.assertEqual([event for event in self.sent if event["type"] == "error"], [])
        waiting = asyncio.create_task(gate.process_frame(frame, FrameDirection.DOWNSTREAM))
        await asyncio.sleep(0)
        self.assertFalse(waiting.done())

        self.sent.extend(controller.finish_response(status="completed"))
        serializer.notify_response_done_published(pipeline_response_id)
        await asyncio.wait_for(waiting, timeout=1.0)

        self.assertIsNotNone(controller.active_response_id)
        self.assertNotEqual(controller.active_response_id, pipeline_response_id)
        types = [event["type"] for event in self.sent]
        self.assertNotIn("error", types)
        self.assertLess(types.index("response.done"), types.index("response.created"))

    async def test_a_second_overlapping_client_request_keeps_its_error(self) -> None:
        serializer, _gate = self._serializer(delegation_route=True)
        serializer._controller.start_response()
        await serializer.deserialize(json.dumps({"type": "response.create"}))

        self.assertIsNone(await serializer.deserialize(json.dumps({"type": "response.create"})))
        self.assertEqual(self._error_codes(), ["response_in_progress"])

    async def test_other_routes_keep_response_in_progress(self) -> None:
        serializer, _gate = self._serializer(delegation_route=False)
        serializer._controller.start_response()

        self.assertIsNone(await serializer.deserialize(json.dumps({"type": "response.create"})))
        self.assertEqual(self._error_codes(), ["response_in_progress"])

    def _error_codes(self) -> list[str]:
        return [event["error"]["code"] for event in self.sent if event["type"] == "error"]
