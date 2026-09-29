# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Focused tests for Generic FBA Realtime client-tool suspend/resume."""

# Test names describe the contract; separate public API docstrings add no value here.
# ruff: noqa: D101, D102

from __future__ import annotations

import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from realtime_helpers import FakeWebSocket

from examples.frontend_backend_agent.generic.backend import GenericThinkerBackend
from examples.frontend_backend_agent.generic.client_tools import (
    build_client_tool_specs,
    client_call_fingerprint,
    format_client_result,
)
from examples.frontend_backend_agent.generic.dispatcher import dispatch_plan
from examples.frontend_backend_agent.generic.tools import TOOLS_SCHEMA
from realtime.client_tools import ClientToolBroker, ClientToolTimeoutResult
from realtime.controller import RealtimeSessionController
from realtime.frames import RealtimeClientToolOutputFrame, RealtimeResponseContextFrame, RealtimeResponseCreateFrame
from realtime.session import RealtimeSessionCapabilities
from realtime.transport import (
    bind_realtime_context,
    bind_realtime_session_prompt_updates,
    configure_realtime_client_tools,
    create_realtime_transport,
    prepare_realtime_tools,
    realtime_client_tool_executor,
    realtime_response_gate_processors,
    realtime_tool_result_processors,
    shutdown_realtime_transport,
)


def _client_schema(name: str = "lookup") -> dict:
    return {
        "type": "function",
        "name": name,
        "description": "Look up a record by id.",
        "parameters": {
            "type": "object",
            "properties": {"record_id": {"type": "string"}},
            "required": ["record_id"],
            "additionalProperties": False,
        },
    }


class _PromptUpdateOwner:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.session_instruction_context = LLMContext([{"role": "system", "content": ""}])
        self.current_talker_prompt_messages: list[dict] = [
            {"role": "system", "content": "Static routing plus a model capability summary."}
        ]
        self.prepared: list[tuple[str, list[dict], object]] = []

    @staticmethod
    def render_session_instructions(instructions: str) -> list[dict]:
        return [{"role": "system", "content": instructions}]

    @staticmethod
    def render_response_instructions(_instructions: str) -> list[dict]:
        return []

    async def prepare_session_update(self, *, instructions, tools, tool_choice):
        self.prepared.append((instructions, tools, tool_choice))
        if self.fail:
            raise RuntimeError("capability generation failed")
        return SimpleNamespace(
            talker_prompt_messages=tuple(self.current_talker_prompt_messages), instructions=instructions
        )

    def snapshot_session_update(self):
        return list(self.session_instruction_context.get_messages())

    def commit_session_update(self, prepared) -> None:
        self.session_instruction_context.set_messages(self.render_session_instructions(prepared.instructions))

    def restore_session_update(self, snapshot) -> None:
        self.session_instruction_context.set_messages(snapshot)


class _ToolLLM:
    def __init__(self) -> None:
        self._functions: dict[object, object] = {}

    def register_function(self, name, handler, **_kwargs) -> None:
        self._functions[name] = handler


class DirectClientToolBrokerTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_call_waits_for_release_and_marks_context_applied(self) -> None:
        broker = ClientToolBroker(output_timeout_secs=5)
        await broker.register_direct_calls((("call-1", "lookup"),), timeout_secs=1)
        waiter = asyncio.create_task(broker.wait_direct_output(call_id="call-1", name="lookup"))

        await broker.stage_output(call_id="call-1", name="lookup", output='{"answer":"found"}')
        self.assertFalse(waiter.done())
        await broker.release_output(call_id="call-1", name="lookup")

        self.assertEqual(await waiter, '{"answer":"found"}')
        await broker.wait_context_applied("call-1", timeout=0.2)
        self.assertEqual(broker.pending_context_call_ids(), ())

    async def test_direct_call_has_a_bounded_timeout_and_rejects_late_output(self) -> None:
        broker = ClientToolBroker(output_timeout_secs=5)
        await broker.register_direct_calls((("call-timeout", "lookup"),), timeout_secs=0.01)

        result = await broker.wait_direct_output(call_id="call-timeout", name="lookup")

        self.assertIsInstance(result, ClientToolTimeoutResult)
        with self.assertRaisesRegex(Exception, "deadline"):
            await broker.stage_output(call_id="call-timeout", name="lookup", output="late")


class RealtimeClientToolRoundContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_backend_completes_four_round_client_tool_workflow(self) -> None:
        tool_names = (
            "get_user_details",
            "get_reservation_details",
            "search_direct_flight",
            "update_reservation_flights",
        )
        client_tools = build_client_tool_specs(
            tuple(
                {
                    "type": "function",
                    "name": name,
                    "description": f"Run {name}.",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                }
                for name in tool_names
            )
        )
        plans = [{"tool": name, "params": {"value": str(index)}} for index, name in enumerate(tool_names, start=1)] + [
            {"complete": True}
        ]

        class SequencedPlanner:
            def __init__(self) -> None:
                self.states: list[dict] = []

            async def plan(self, *, query: str, state: dict) -> dict:
                del query
                self.states.append(copy.deepcopy(state))
                return plans.pop(0)

        planner = SequencedPlanner()
        wire_rounds: list[tuple[tuple[str, dict], ...]] = []

        async def execute(calls, _timeout):
            wire_rounds.append(calls)
            return [{"ok": True, "round": len(wire_rounds)} for _call in calls]

        backend = GenericThinkerBackend(
            planner=planner,
            tools={},
            enabled_tools=tool_names,
            client_tools=client_tools,
            client_tool_executor=execute,
            overall_timeout_seconds=3,
            planner_timeout_seconds=1,
            max_planning_rounds=8,
        )

        payload = await backend.call("Change the reservation after looking up every prerequisite.")

        self.assertEqual([round_calls[0][0] for round_calls in wire_rounds], list(tool_names))
        self.assertEqual(len(planner.states), 5)
        self.assertEqual([state["planning_round"] for state in planner.states], [1, 2, 3, 4, 5])
        self.assertEqual(len(planner.states[-1]["prior_tool_results"]), 4)
        self.assertEqual(
            [result["tool"] for result in payload["data"]["results"]],
            list(tool_names),
        )

    async def test_pipeline_created_round_surfaces_and_resumes_through_wire_output(self) -> None:
        websocket = FakeWebSocket([])
        voice = "Magpie-Multilingual.EN-US.Aria"
        controller = RealtimeSessionController(
            model="nvidia/nemotron-realtime-generic-frontend-backend",
            voice=voice,
            runtime_config={"pipeline_mode": "generic-frontend-backend-agent"},
            capabilities=RealtimeSessionCapabilities(voices=frozenset({voice}), function_tools=True),
        )
        controller.apply_session_update(
            {
                "output_modalities": ["text"],
                "tools": [_client_schema()],
            }
        )
        controller.bind_session_tool_projection(
            client_tool_bindings={"lookup": "lookup"},
            mcp_pipeline_names=frozenset(),
        )
        transport = create_realtime_transport(websocket, controller=controller)
        [response_gate] = realtime_response_gate_processors(transport)
        bind_realtime_context(transport, LLMContext([]))
        realtime_tool_result_processors(transport)
        executor = realtime_client_tool_executor(transport)
        self.assertIsNotNone(executor)

        try:
            round_task = asyncio.create_task(executor((("lookup", {"record_id": "one"}),), 1.0))
            for _attempt in range(100):
                done_events = [
                    event for event in websocket.sent if event.get("type") == "response.function_call_arguments.done"
                ]
                if done_events:
                    break
                await asyncio.sleep(0.005)
            self.assertEqual(len(done_events), 1)
            call_event = done_events[0]
            output_frame = await transport.input()._params.serializer.deserialize(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call_event["call_id"],
                            "output": '{"ok":true,"answer":"found"}',
                        },
                    }
                )
            )
            self.assertIsInstance(output_frame, RealtimeClientToolOutputFrame)
            await response_gate.process_frame(output_frame, FrameDirection.DOWNSTREAM)

            self.assertEqual(
                await asyncio.wait_for(round_task, timeout=1.0),
                ['{"ok":true,"answer":"found"}'],
            )
            event_types = [event.get("type") for event in websocket.sent]
            self.assertIn("response.created", event_types)
            self.assertIn("response.output_item.done", event_types)
            self.assertIn("response.done", event_types)
            self.assertIn("conversation.item.added", event_types)
            self.assertIn("conversation.item.done", event_types)
        finally:
            shutdown_realtime_transport(transport)

    async def test_live_generic_planner_policy_update_refreshes_prompt_owner(self) -> None:
        websocket = FakeWebSocket([])
        voice = "Magpie-Multilingual.EN-US.Aria"
        controller = RealtimeSessionController(
            model="nvidia/nemotron-realtime-generic-frontend-backend",
            voice=voice,
            runtime_config={"pipeline_mode": "generic-frontend-backend-agent"},
            capabilities=RealtimeSessionCapabilities(voices=frozenset({voice}), function_tools=True),
        )
        transport = create_realtime_transport(websocket, controller=controller)
        realtime_response_gate_processors(transport)
        owner = _PromptUpdateOwner()
        bind_realtime_context(transport, LLMContext(owner.current_talker_prompt_messages))
        bind_realtime_session_prompt_updates(transport, owner)
        try:
            frame = await transport.input()._params.serializer.deserialize(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {"instructions": "Replace the domain policy."},
                    }
                )
            )
            self.assertIsNone(frame)
            self.assertEqual(websocket.sent[-1]["type"], "session.updated")
            self.assertEqual(controller.public_session()["instructions"], "Replace the domain policy.")
            self.assertEqual(
                owner.session_instruction_context.get_messages(),
                [{"role": "system", "content": "Replace the domain policy."}],
            )
            self.assertEqual(owner.prepared, [("Replace the domain policy.", [], "auto")])
        finally:
            shutdown_realtime_transport(transport)

    async def test_client_tools_remain_thinker_owned_across_session_and_response_updates(self) -> None:
        websocket = FakeWebSocket([])
        voice = "Magpie-Multilingual.EN-US.Aria"
        controller = RealtimeSessionController(
            model="nvidia/nemotron-realtime-generic-frontend-backend",
            voice=voice,
            runtime_config={"pipeline_mode": "generic-frontend-backend-agent"},
            capabilities=RealtimeSessionCapabilities(voices=frozenset({voice}), function_tools=True),
        )
        transport = create_realtime_transport(websocket, controller=controller)
        [response_gate] = realtime_response_gate_processors(transport)
        response_gate.push_frame = AsyncMock()
        owner = _PromptUpdateOwner()
        talker_context = LLMContext(
            owner.current_talker_prompt_messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        thinker_llm = _ToolLLM()
        configure_realtime_client_tools(
            transport,
            thinker_llm,
            [],
            trusted_tools=TOOLS_SCHEMA,
            trusted_tool_names=("call_backend", "cancel_backend"),
        )
        await prepare_realtime_tools(transport, thinker_llm)
        bind_realtime_context(transport, talker_context)
        bind_realtime_session_prompt_updates(transport, owner)
        trusted_talker_tools = talker_context.tools

        try:
            update = await transport.input()._params.serializer.deserialize(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "output_modalities": ["text"],
                            "tools": [_client_schema("get_reservation_details")],
                            "tool_choice": "required",
                        },
                    }
                )
            )

            self.assertIsNone(update)
            self.assertEqual(websocket.sent[-1]["type"], "session.updated")
            self.assertEqual(talker_context.tools, trusted_talker_tools)
            self.assertEqual(talker_context.tool_choice, "auto")
            self.assertIn(None, thinker_llm._functions)
            self.assertEqual(owner.prepared[-1][1][0]["name"], "get_reservation_details")

            frame = await transport.input()._params.serializer.deserialize(json.dumps({"type": "response.create"}))

            self.assertIsInstance(frame, RealtimeResponseCreateFrame)
            self.assertIsNone(frame.tools)
            self.assertIsNone(frame.tool_choice)
            self.assertEqual(frame.client_tool_bindings, {"get_reservation_details": "get_reservation_details"})
            await response_gate.process_frame(frame, FrameDirection.DOWNSTREAM)
            response_context_frame = response_gate.push_frame.await_args_list[-1].args[0]
            self.assertIsInstance(response_context_frame, RealtimeResponseContextFrame)
            self.assertEqual(response_context_frame.context.tools, trusted_talker_tools)
            self.assertEqual(response_context_frame.context.tool_choice, "auto")
        finally:
            shutdown_realtime_transport(transport)

    async def test_failed_capability_refresh_leaves_session_and_both_prompts_unchanged(self) -> None:
        websocket = FakeWebSocket([])
        voice = "Magpie-Multilingual.EN-US.Aria"
        controller = RealtimeSessionController(
            model="nvidia/nemotron-realtime-generic-frontend-backend",
            voice=voice,
            runtime_config={"pipeline_mode": "generic-frontend-backend-agent"},
            capabilities=RealtimeSessionCapabilities(voices=frozenset({voice}), function_tools=True),
        )
        transport = create_realtime_transport(websocket, controller=controller)
        realtime_response_gate_processors(transport)
        owner = _PromptUpdateOwner(fail=True)
        talker_context = LLMContext(owner.current_talker_prompt_messages)
        bind_realtime_context(transport, talker_context)
        bind_realtime_session_prompt_updates(transport, owner)
        before_talker = list(talker_context.get_messages())
        try:
            frame = await transport.input()._params.serializer.deserialize(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {"instructions": "Do not commit this."},
                    }
                )
            )
            self.assertIsNone(frame)
            self.assertEqual(websocket.sent[-1]["type"], "error")
            self.assertEqual(controller.public_session()["instructions"], "")
            self.assertEqual(owner.session_instruction_context.messages, [{"role": "system", "content": ""}])
            self.assertEqual(talker_context.messages, before_talker)
        finally:
            shutdown_realtime_transport(transport)


class GenericClientToolDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_schema_is_validated_then_one_batch_is_executed(self) -> None:
        specs = build_client_tool_specs((_client_schema(), _client_schema("second")))
        batches: list[tuple[tuple[str, dict], ...]] = []

        async def execute(calls, _timeout):
            batches.append(calls)
            return ['{"answer":"first"}', "second result"]

        accumulated: list[dict] = []
        payload = await dispatch_plan(
            {
                "tool_calls": [
                    {"tool": "lookup", "params": {"record_id": "one"}},
                    {"tool": "second", "params": {"record_id": "two"}},
                ]
            },
            {},
            ("lookup", "second"),
            client_tools=specs,
            client_tool_executor=execute,
            accumulated_results=accumulated,
            seen_client_calls=set(),
        )

        self.assertEqual(len(batches), 1)
        self.assertEqual([name for name, _arguments in batches[0]], ["lookup", "second"])
        self.assertEqual(payload["tool"], "multi_tool")
        self.assertEqual([item["status"] for item in accumulated], ["success", "success"])

    async def test_invalid_arguments_fail_before_client_execution(self) -> None:
        specs = build_client_tool_specs((_client_schema(),))
        called = False

        async def execute(_calls, _timeout):
            nonlocal called
            called = True
            return []

        payload = await dispatch_plan(
            {"tool": "lookup", "params": {}},
            {},
            ("lookup",),
            client_tools=specs,
            client_tool_executor=execute,
        )

        self.assertFalse(called)
        self.assertEqual(payload["reason"], "params_invalid")

    async def test_duplicate_in_one_client_batch_is_rejected_before_wire_execution(self) -> None:
        specs = build_client_tool_specs((_client_schema(),))
        called = False

        async def execute(_calls, _timeout):
            nonlocal called
            called = True
            return []

        seen: set[str] = set()
        payload = await dispatch_plan(
            {
                "tool_calls": [
                    {"tool": "lookup", "params": {"record_id": "one"}},
                    {"tool": "lookup", "params": {"record_id": "one"}},
                ]
            },
            {},
            ("lookup",),
            client_tools=specs,
            client_tool_executor=execute,
            seen_client_calls=seen,
        )

        self.assertFalse(called)
        self.assertEqual(seen, {client_call_fingerprint("lookup", {"record_id": "one"})})
        self.assertEqual(payload["status"], "unavailable")
        self.assertIn("repeated tool request", payload["response_text"])

    async def test_failed_duplicate_is_suppressed_without_another_wire_round(self) -> None:
        specs = build_client_tool_specs((_client_schema(),))
        arguments = {"record_id": "one"}
        seen = {client_call_fingerprint("lookup", arguments)}
        called = False

        async def execute(_calls, _timeout):
            nonlocal called
            called = True
            return []

        payload = await dispatch_plan(
            {"tool": "lookup", "params": arguments},
            {},
            ("lookup",),
            client_tools=specs,
            client_tool_executor=execute,
            seen_client_calls=seen,
        )

        self.assertFalse(called)
        self.assertEqual(payload["status"], "unavailable")
        self.assertIn("repeated tool request", payload["response_text"])


class JsonEncodedClientResultTests(unittest.TestCase):
    """A JSON-encoded envelope must be classified by its contents, not its type."""

    _TAU_ERROR = (
        '{"call_id": "call_1", "error": {"category": "benchmark_error", '
        '"message": "Error: User era_of_Garcia_1177 not found"}, "output": null}'
    )

    def test_json_encoded_error_envelope_is_not_reported_as_success(self) -> None:
        payload = format_client_result("get_user_details", {"user_id": "x"}, self._TAU_ERROR)

        self.assertEqual(payload["status"], "unavailable")
        self.assertIn("not found", payload["response_text"])

    def test_json_encoded_ok_false_is_an_error(self) -> None:
        payload = format_client_result("lookup", {}, '{"ok": false, "detail": "nope"}')

        self.assertEqual(payload["status"], "error")

    def test_json_encoded_success_stays_successful(self) -> None:
        payload = format_client_result("lookup", {}, '{"user_id": "aarav_garcia_1177"}')

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["data"]["result"], {"user_id": "aarav_garcia_1177"})

    def test_mapping_and_json_string_envelopes_agree(self) -> None:
        as_mapping = format_client_result("lookup", {}, {"error": {"message": "boom"}})
        as_json = format_client_result("lookup", {}, '{"error": {"message": "boom"}}')

        self.assertEqual(as_mapping["status"], as_json["status"])

    def test_a_structured_result_is_never_spoken_verbatim(self) -> None:
        """A client tool returns data, not a sentence; speaking it leaks records.

        Observed live: a reservation lookup was read out field by field,
        including a passenger name and date of birth belonging to someone
        other than the caller. The record must reach the Talker to compose
        from, and never the speaker unchanged.
        """
        record = {
            "reservation_id": "EHGLP3",
            "user_id": "emma_kim_9957",
            "passengers": [{"first_name": "Evelyn", "last_name": "Taylor", "dob": "1965-01-16"}],
        }

        payload = format_client_result("get_reservation_details", {"reservation_id": "EHGLP3"}, record)

        self.assertEqual(payload["status"], "success")
        for secret in ("Evelyn", "Taylor", "1965-01-16", "emma_kim_9957"):
            with self.subTest(value=secret):
                self.assertNotIn(secret, payload["response_text"])
        self.assertEqual(payload["data"]["result"], record)

    def test_a_client_result_is_never_delivered_directly(self) -> None:
        """Direct delivery speaks response_text as-is, so client data must not use it."""
        from examples.frontend_backend_agent.src.tool_handlers import _should_deliver_directly

        payload = format_client_result("get_reservation_details", {}, {"reservation_id": "EHGLP3"})

        for mode in ("direct", "hybrid", "talker"):
            with self.subTest(mode=mode):
                self.assertFalse(_should_deliver_directly(payload, default_mode=mode))

    def test_non_json_output_keeps_its_legacy_classification(self) -> None:
        self.assertEqual(format_client_result("lookup", {}, "Reservation confirmed")["status"], "success")
        self.assertEqual(format_client_result("lookup", {}, "Error: not found")["status"], "error")
        self.assertEqual(format_client_result("lookup", {}, "{not json")["status"], "success")
        self.assertEqual(format_client_result("lookup", {}, "   ")["status"], "error")


class JsonEncodedFailureSuppressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_json_encoded_failure_keeps_the_call_suppressed(self) -> None:
        """A JSON-encoded failure must retain its fingerprint, not clear it."""
        specs = build_client_tool_specs((_client_schema(),))
        arguments = {"record_id": "one"}
        seen: set[str] = set()

        async def execute(_calls, _timeout):
            return ['{"call_id": "c1", "error": {"message": "Error: not found"}, "output": null}']

        payload = await dispatch_plan(
            {"tool": "lookup", "params": arguments},
            {},
            ("lookup",),
            client_tools=specs,
            client_tool_executor=execute,
            seen_client_calls=seen,
        )

        self.assertEqual(payload["status"], "unavailable")
        self.assertEqual(seen, {client_call_fingerprint("lookup", arguments)})


class SessionScopedClientSuppressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_suppression_state_is_shared_across_backend_calls(self) -> None:
        """The frontend re-delegates per turn; suppression must outlive one call."""
        backend = GenericThinkerBackend(
            planner=SimpleNamespace(),
            enabled_tools=(),
            tools={},
            client_tools=build_client_tool_specs((_client_schema(),)),
        )
        captured: list[set[str]] = []

        async def fake_plan(*_args, **_kwargs):
            return {"tool": "none", "params": {}}

        async def fake_dispatch(*_args, **kwargs):
            captured.append(kwargs["seen_client_calls"])
            return {"tool": "none", "status": "success"}

        with (
            patch.object(GenericThinkerBackend, "_plan_with_retry", fake_plan),
            patch("examples.frontend_backend_agent.generic.backend.dispatch_plan", fake_dispatch),
        ):
            await backend._run_call("call-1", "first query")
            await backend._run_call("call-2", "second query")

        self.assertEqual(len(captured), 2)
        self.assertIs(captured[0], captured[1])
        self.assertIs(captured[0], backend._seen_client_calls)


if __name__ == "__main__":
    unittest.main()
