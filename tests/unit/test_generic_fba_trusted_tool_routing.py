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
from realtime.protocol import RealtimeProtocolError
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


class DelegateToolNameContractTests(unittest.TestCase):
    """Pin the delegate tool names that out-of-repo clients must filter.

    The gateway publishes delegate calls as ordinary wire ``function_call``
    items and deliberately does not tell the client which names are
    server-owned, so every client hardcodes the set. The evaluation harness
    does this in its endpoint profile. Renaming or adding a delegate tool
    without updating those clients makes them dispatch it as a benchmark
    tool, which fails as an unknown tool rather than as a clear contract
    break. Treat a change here as a breaking wire-contract change.
    """

    def test_generic_delegate_tool_names_are_exactly_call_and_cancel_backend(self) -> None:
        from pipecat.adapters.schemas.tools_schema import AdapterType

        from examples.frontend_backend_agent.generic.tools import TOOLS_SCHEMA

        names = {tool["function"]["name"] for tool in TOOLS_SCHEMA.custom_tools[AdapterType.OPENAI]}

        self.assertEqual(names, {"call_backend", "cancel_backend"})


class DelegateCallWireContractTests(unittest.TestCase):
    """Pin what a Realtime client actually observes for a delegate call.

    A delegate call is the pipeline's own internal mechanics, so a client
    following the OpenAI Realtime contract needs zero special handling to
    avoid it: no function_call ever announces it, no field hints it exists,
    and Response A -- the delegate's own response lifecycle -- closes with an
    empty output array. Only the response envelope (created, then done) is
    observable, and it is indistinguishable from a response that produced no
    output for any other reason.

    These tests exist so that contract is discoverable in this repository and
    cannot change silently. If the gateway ever starts publishing trusted
    calls, or marking them, update these tests and
    docs/how-to/use-realtime-gateway.md together.
    """

    def _delegate_call_events(self) -> list[dict]:
        controller = _controller()
        return controller.start_function_call(
            call_id="call-1",
            name="call_backend",
            arguments={"query": "Cancel reservation ABC123.", "filler_text": "One moment."},
        )

    def test_a_delegate_call_never_announces_a_function_call(self) -> None:
        events = self._delegate_call_events()
        types = [event.get("type") for event in events]

        self.assertNotIn("response.output_item.added", types)
        self.assertNotIn("response.function_call_arguments.delta", types)
        self.assertNotIn("response.function_call_arguments.done", types)
        self.assertNotIn("response.output_item.done", types)
        # The response envelope is all that is left, and it carries nothing
        # that names the call, its arguments, or its owner.
        self.assertEqual(types, ["response.created"])

    def test_no_published_field_identifies_the_call_or_its_owner(self) -> None:
        payload = json.dumps(self._delegate_call_events())

        for marker in ("delegate", "server_owned", "owner", "trusted", "call_backend", "call-1"):
            with self.subTest(marker=marker):
                self.assertNotIn(marker, payload)

    def test_response_a_closes_with_an_empty_output_array(self) -> None:
        controller = _controller()
        controller.start_function_call(
            call_id="call-1",
            name="call_backend",
            arguments={"query": "Cancel reservation ABC123.", "filler_text": "One moment."},
        )

        events = controller.finish_response(status="completed")
        done = next(event for event in events if event["type"] == "response.done")

        self.assertEqual(done["response"]["output"], [])

    def test_the_delegate_result_is_never_published_either(self) -> None:
        controller = _controller()
        controller.start_function_call(
            call_id="call-1",
            name="call_backend",
            arguments={"query": "Cancel reservation ABC123.", "filler_text": "One moment."},
        )
        controller.finish_response(status="completed")

        events = controller.add_function_output(call_id="call-1", output='{"status":"confirmed"}', owner="delegate")

        self.assertEqual(events, [])

    def test_answering_a_delegate_call_is_still_rejected(self) -> None:
        controller = _controller()
        controller.start_function_call(
            call_id="call-1",
            name="call_backend",
            arguments={"query": "Cancel reservation ABC123.", "filler_text": "One moment."},
        )

        with self.assertRaises(RealtimeProtocolError) as raised:
            controller.add_function_output(call_id="call-1", output="{}", owner="client")

        self.assertEqual(raised.exception.code, "tool_owner_mismatch")


if __name__ == "__main__":
    unittest.main()
