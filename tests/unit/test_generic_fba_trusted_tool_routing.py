# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Trusted delegate tools stay live when the client advertises only its own tools.

The Generic Frontend/Backend adapter hides ``call_backend``/``cancel_backend``
from the Realtime client, so every projection that derives trusted state from
the client-facing session must not erase them.
"""

# Test names describe the contract; separate public API docstrings add no value here.
# ruff: noqa: D101, D102

from __future__ import annotations

import asyncio
import json
import unittest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from realtime_helpers import FakeWebSocket

from realtime.controller import RealtimeSessionController
from realtime.frames import RealtimeClientToolOutputFrame
from realtime.gateway import _session_patch_to_runtime
from realtime.session import RealtimeSessionCapabilities
from realtime.transport import (
    bind_realtime_context,
    create_realtime_transport,
    realtime_client_tool_executor,
    realtime_response_gate_processors,
    realtime_tool_result_processors,
    shutdown_realtime_transport,
)

VOICE = "Magpie-Multilingual.EN-US.Aria"
DELEGATE_TOOLS = ["call_backend", "cancel_backend"]


def _client_schema(name: str = "get_reservation_details") -> dict:
    return {
        "type": "function",
        "name": name,
        "description": "Get the details of a reservation.",
        "parameters": {
            "type": "object",
            "properties": {"reservation_id": {"type": "string"}},
            "required": ["reservation_id"],
        },
    }


def _controller() -> RealtimeSessionController:
    return RealtimeSessionController(
        model="nvidia/nemotron-realtime-generic-frontend-backend",
        voice=VOICE,
        runtime_config={"pipeline_mode": "generic-frontend-backend-agent"},
        capabilities=RealtimeSessionCapabilities(voices=frozenset({VOICE}), function_tools=True),
        delegate_tools=DELEGATE_TOOLS,
    )


class DelegateToolProjectionTests(unittest.TestCase):
    def test_delegate_tools_survive_a_session_that_declares_only_client_tools(self):
        runtime = _session_patch_to_runtime(
            {"tools": [_client_schema()]},
            {"pipeline_mode": "generic-frontend-backend-agent", "delegate_tools": DELEGATE_TOOLS},
        )
        self.assertEqual(runtime["delegate_tools"], DELEGATE_TOOLS)
        self.assertEqual([tool["name"] for tool in runtime["client_tools"]], ["get_reservation_details"])

    def test_server_tools_still_require_the_client_to_declare_them(self):
        runtime = _session_patch_to_runtime(
            {"tools": [_client_schema()]},
            {"pipeline_mode": "generic-frontend-backend-agent", "server_tools": ["get_weather"]},
        )
        self.assertEqual(runtime["server_tools"], [])


class TrustedToolActivationTests(unittest.TestCase):
    def test_a_delegate_tool_is_dispatchable_without_being_advertised(self):
        controller = _controller()
        controller.apply_session_update({"output_modalities": ["text"], "tools": [_client_schema()]})
        controller.bind_session_tool_projection(
            client_tool_bindings={"get_reservation_details": "get_reservation_details"},
            mcp_pipeline_names=frozenset(),
        )
        self.assertNotIn(
            "call_backend",
            {tool["name"] for tool in controller.session.public_view().get("tools", [])},
        )
        controller.start_function_call(
            call_id="call-1",
            name="call_backend",
            arguments={"query": "Get reservation ABC123.", "filler_text": "One moment."},
        )
        self.assertEqual(controller.pipeline_tool_owner(call_id="call-1", pipeline_name="call_backend"), "delegate")

    def test_an_undeclared_client_tool_is_still_rejected(self):
        controller = _controller()
        controller.apply_session_update({"output_modalities": ["text"], "tools": [_client_schema()]})
        controller.bind_session_tool_projection(
            client_tool_bindings={"get_reservation_details": "get_reservation_details"},
            mcp_pipeline_names=frozenset(),
        )
        with self.assertRaises(Exception) as caught:
            controller.start_function_call(call_id="call-2", name="cancel_reservation", arguments={})
        self.assertIn("unknown tool", str(caught.exception))


class PendingDelegateCallTests(unittest.TestCase):
    """An in-flight delegation must not block the next client turn.

    The client cannot answer a delegate call, so gating ``response.create`` on
    one would strand the session for the rest of the conversation.
    """

    def _controller_with_pending_delegate(self) -> RealtimeSessionController:
        controller = _controller()
        controller.apply_session_update({"output_modalities": ["text"], "tools": [_client_schema()]})
        controller.bind_session_tool_projection(
            client_tool_bindings={"get_reservation_details": "get_reservation_details"},
            mcp_pipeline_names=frozenset(),
        )
        controller.start_function_call(
            call_id="delegate-1",
            name="call_backend",
            arguments={"query": "Cancel reservation ABC123.", "filler_text": "One moment."},
        )
        return controller

    def test_a_pending_delegate_call_is_not_owed_by_the_client(self):
        controller = self._controller_with_pending_delegate()
        self.assertIn("delegate-1", controller.pending_tool_call_ids())
        self.assertEqual(controller.pending_client_tool_call_ids(), ())

    def test_a_pending_client_call_is_still_owed_by_the_client(self):
        controller = self._controller_with_pending_delegate()
        controller.start_function_call(
            call_id="client-1",
            name="get_reservation_details",
            arguments={"reservation_id": "ABC123"},
        )
        self.assertEqual(controller.pending_client_tool_call_ids(), ("client-1",))


class ClientToolRoundUnderBoundPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_round_runs_when_a_canonical_prompt_prefix_is_bound(self):
        """The round freezes an empty context, which must not fail the prompt-prefix check."""
        prompt = [{"role": "system", "content": "static prompt"}, {"role": "user", "content": "few-shot"}]
        websocket = FakeWebSocket([])
        controller = _controller()
        controller.apply_session_update(
            {"output_modalities": ["text"], "instructions": "be helpful", "tools": [_client_schema()]}
        )
        controller.bind_session_tool_projection(
            client_tool_bindings={"get_reservation_details": "get_reservation_details"},
            mcp_pipeline_names=frozenset(),
        )
        transport = create_realtime_transport(websocket, controller=controller)
        [response_gate] = realtime_response_gate_processors(transport)
        bind_realtime_context(
            transport,
            LLMContext(list(prompt)),
            render_instructions=lambda _instructions: list(prompt),
        )
        realtime_tool_result_processors(transport)
        executor = realtime_client_tool_executor(transport)
        self.assertIsNotNone(executor)

        try:
            round_task = asyncio.create_task(
                executor((("get_reservation_details", {"reservation_id": "ABC123"}),), 5.0)
            )
            done_events: list[dict] = []
            for _attempt in range(200):
                done_events = [
                    event for event in websocket.sent if event.get("type") == "response.function_call_arguments.done"
                ]
                if done_events:
                    break
                await asyncio.sleep(0.005)
            self.assertEqual(len(done_events), 1, "the client tool call never reached the wire")

            output_frame = await transport.input()._params.serializer.deserialize(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": done_events[0]["call_id"],
                            "output": '{"status":"confirmed"}',
                        },
                    }
                )
            )
            self.assertIsInstance(output_frame, RealtimeClientToolOutputFrame)
            await response_gate.process_frame(output_frame, FrameDirection.DOWNSTREAM)
            self.assertEqual(
                await asyncio.wait_for(round_task, timeout=2.0),
                ['{"status":"confirmed"}'],
            )
        finally:
            shutdown_realtime_transport(transport)


if __name__ == "__main__":
    unittest.main()
