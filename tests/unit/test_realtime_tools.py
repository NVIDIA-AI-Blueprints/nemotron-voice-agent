# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103, D107

from __future__ import annotations

import json
import unittest
from typing import Any
from unittest.mock import AsyncMock

from pipecat.frames.frames import FunctionCallResultFrame, FunctionCallResultProperties, FunctionCallsStartedFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM
from realtime_helpers import FakeWebSocket

from realtime.controller import RealtimeSessionController
from realtime.gateway import handle_realtime_websocket
from realtime.protocol import RealtimeProtocolError
from realtime.serializer import RealtimeFrameSerializer
from realtime.session import CanonicalRealtimeSession, RealtimeSessionCapabilities
from realtime.tool_schema import MAX_SESSION_TOOL_COUNT, MAX_SESSION_TOOLS_JSON_BYTES
from realtime.transport import RealtimeToolResultPolicy

MODEL = "nvidia/nemotron-realtime"
VOICE = "Magpie-Multilingual.EN-US.Aria"
CLIENT_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
SERVER_TOOL = {
    "type": "function",
    "name": "set_memory",
    "description": "Store a memory",
    "parameters": {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    },
}


def _minimal_tools(count: int) -> list[dict[str, Any]]:
    return [{"type": "function", "name": f"tool_{index}"} for index in range(count)]


def _aggregate_oversized_tools() -> list[dict[str, Any]]:
    description = "x" * 30_000
    tools = [
        {
            "type": "function",
            "name": f"aggregate_{index}",
            "parameters": {"type": "object", "description": description},
        }
        for index in range(9)
    ]
    assert all(
        len(json.dumps(tool["parameters"], ensure_ascii=False, separators=(",", ":")).encode("utf-8")) < 32 * 1024
        for tool in tools
    )
    assert len(json.dumps(tools, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > (
        MAX_SESSION_TOOLS_JSON_BYTES
    )
    return tools


def _capabilities(
    *,
    function_tools: bool = True,
    mcp_tools: bool = False,
    trusted_tools: frozenset[str] = frozenset(),
    sequential_tool_calls: bool = False,
) -> RealtimeSessionCapabilities:
    return RealtimeSessionCapabilities(
        voices=frozenset({VOICE}),
        function_tools=function_tools,
        mcp_tools=mcp_tools,
        trusted_function_tools=trusted_tools,
        sequential_tool_calls=sequential_tool_calls,
    )


def _session(
    *,
    capabilities: RealtimeSessionCapabilities | None = None,
    trusted_tools: list[dict[str, Any]] | None = None,
) -> CanonicalRealtimeSession:
    return CanonicalRealtimeSession(
        model=MODEL,
        voice=VOICE,
        capabilities=capabilities or _capabilities(),
        trusted_tools=trusted_tools,
    )


def _sanitize_runtime(data: dict[str, Any], **_: Any) -> dict[str, Any]:
    runtime = dict(data)
    runtime.setdefault("pipeline_mode", "generic-assistant")
    runtime.setdefault("model_id", MODEL)
    runtime.setdefault("tts_voice_id", VOICE)
    return runtime


def _tool_round_controller(*, mixed: bool) -> RealtimeSessionController:
    controller = RealtimeSessionController(
        model=MODEL,
        voice=VOICE,
        runtime_config={},
        server_tools=["set_memory"],
        trusted_tool_schemas=[SERVER_TOOL],
        capabilities=_capabilities(trusted_tools=frozenset({"set_memory"})),
    )
    tools = [SERVER_TOOL, CLIENT_TOOL] if mixed else [SERVER_TOOL]
    controller.apply_session_update({"tools": tools})
    controller.bind_session_tool_projection(
        client_tool_bindings={"get_weather": "get_weather"} if mixed else {},
        mcp_pipeline_names=frozenset(),
    )
    controller.start_response()
    return controller


class CanonicalSessionToolTests(unittest.TestCase):
    def test_public_controller_view_redacts_mcp_credentials(self) -> None:
        controller = RealtimeSessionController(
            model=MODEL,
            voice=VOICE,
            runtime_config={"pipeline_mode": "generic-assistant"},
            capabilities=_capabilities(mcp_tools=True),
        )
        mcp_tool = {
            "type": "mcp",
            "server_label": "private",
            "server_url": "https://mcp.example.test/rpc",
            "authorization": "private-bearer",
            "headers": {"X-Service-Key": "private-header"},
        }

        controller.apply_session_update({"tools": [mcp_tool]})

        private_tool = controller.session.public_view()["tools"][0]
        public_tool = controller.public_session()["tools"][0]
        self.assertEqual(private_tool["authorization"], "private-bearer")
        self.assertEqual(private_tool["headers"], {"X-Service-Key": "private-header"})
        self.assertNotIn("authorization", public_tool)
        self.assertNotIn("headers", public_tool)

    def test_mcp_configuration_and_forced_choice_follow_the_native_schema(self) -> None:
        session = _session(capabilities=_capabilities(mcp_tools=True))
        mcp_tool = {
            "type": "mcp",
            "server_label": "documents",
            "server_url": "https://mcp.example.test/rpc",
            "authorization": "private-bearer",
            "headers": {"X-Tenant": "tenant-1"},
            "allowed_tools": {"read_only": True, "tool_names": ["read_document"]},
            "require_approval": {
                "always": {"tool_names": ["write_document"]},
                "never": {"tool_names": ["read_document"]},
            },
            "allowed_callers": ["direct"],
            "defer_loading": False,
        }

        updated = session.apply_update(
            {
                "tools": [mcp_tool],
                "tool_choice": {
                    "type": "mcp",
                    "server_label": "documents",
                    "name": "read_document",
                },
            }
        )

        self.assertEqual(updated["tools"], [mcp_tool])
        self.assertEqual(
            updated["tool_choice"],
            {"type": "mcp", "server_label": "documents", "name": "read_document"},
        )

    def test_mcp_definition_validation_rejects_ambiguous_or_unsafe_shapes(self) -> None:
        base = {
            "type": "mcp",
            "server_label": "documents",
            "server_url": "https://mcp.example.test/rpc",
        }
        cases = (
            (
                [{**base, "connector_id": "connector_googlecalendar"}],
                "session.tools[0]",
            ),
            (
                [base, {**base, "server_url": "https://other.example.test/rpc"}],
                "session.tools[1].server_label",
            ),
            (
                [{**base, "authorization": "secret", "headers": {"Authorization": "Bearer other"}}],
                "session.tools[0].headers",
            ),
            (
                [{**base, "headers": {"X-Tenant": "one", "x-tenant": "two"}}],
                "session.tools[0].headers.x-tenant",
            ),
            (
                [{**base, "require_approval": {}}],
                "session.tools[0].require_approval",
            ),
            (
                [{**base, "allowed_callers": ["direct", "direct"]}],
                "session.tools[0].allowed_callers[1]",
            ),
        )

        for tools, param in cases:
            with self.subTest(param=param), self.assertRaises(RealtimeProtocolError) as raised:
                _session(capabilities=_capabilities(mcp_tools=True)).apply_update({"tools": tools})
            self.assertEqual(raised.exception.param, param)

    def test_missing_parameters_are_canonicalized_to_empty_object(self) -> None:
        session = _session()
        tool = {"type": "function", "name": "ping"}

        updated = session.apply_update({"tools": [tool]})

        self.assertEqual(updated["tools"], [{"type": "function", "name": "ping", "parameters": {}}])

    def test_invalid_and_duplicate_tools_are_rejected(self) -> None:
        cases = (
            ([CLIENT_TOOL, CLIENT_TOOL], "invalid_value"),
            ([{"type": "mcp", "name": "remote"}], "unsupported_capability"),
            ([{"type": "function", "name": "bad", "parameters": []}], "invalid_type"),
            ([{"type": "function", "name": "bad", "strict": "yes"}], "unknown_parameter"),
        )

        for tools, code in cases:
            with self.subTest(code=code), self.assertRaises(RealtimeProtocolError) as raised:
                _session().apply_update({"tools": tools})
            self.assertEqual(raised.exception.code, code)

    def test_invalid_json_schema_is_rejected_without_partial_session_mutation(self) -> None:
        session = _session()
        original = session.public_view()
        with self.assertRaises(RealtimeProtocolError) as raised:
            session.apply_update(
                {
                    "tools": [
                        {
                            "type": "function",
                            "name": "invalid_schema",
                            "parameters": {"type": 7},
                        }
                    ]
                }
            )

        self.assertEqual(raised.exception.code, "invalid_tool_schema")
        self.assertEqual(raised.exception.param, "session.tools[0].parameters")
        self.assertEqual(session.public_view(), original)

    def test_tool_count_limit_rejects_a_large_minimal_array_atomically(self) -> None:
        session = _session()
        accepted = _minimal_tools(MAX_SESSION_TOOL_COUNT)
        self.assertEqual(len(session.apply_update({"tools": accepted})["tools"]), MAX_SESSION_TOOL_COUNT)
        original = session.public_view()

        with self.assertRaises(RealtimeProtocolError) as raised:
            session.apply_update({"tools": _minimal_tools(10_000)})

        self.assertEqual(raised.exception.code, "invalid_tool_schema")
        self.assertEqual(raised.exception.param, "session.tools")
        self.assertEqual(
            raised.exception.message,
            f"session.tools supports at most {MAX_SESSION_TOOL_COUNT} tools",
        )
        self.assertEqual(session.public_view(), original)

    def test_aggregate_tool_json_limit_rejects_individually_bounded_schemas_atomically(self) -> None:
        session = _session()
        original = session.public_view()

        with self.assertRaises(RealtimeProtocolError) as raised:
            session.apply_update({"tools": _aggregate_oversized_tools()})

        self.assertEqual(raised.exception.code, "invalid_tool_schema")
        self.assertEqual(raised.exception.param, "session.tools")
        self.assertEqual(
            raised.exception.message,
            f"session.tools exceeds the {MAX_SESSION_TOOLS_JSON_BYTES}-byte aggregate JSON limit",
        )
        self.assertEqual(session.public_view(), original)

    def test_backend_can_disable_client_defined_tools(self) -> None:
        session = _session(capabilities=_capabilities(function_tools=False))

        self.assertEqual(session.apply_update({"tools": []})["tools"], [])

        with self.assertRaises(RealtimeProtocolError) as raised:
            session.apply_update({"tools": [CLIENT_TOOL]})

        self.assertEqual(raised.exception.code, "unsupported_capability")
        self.assertEqual(raised.exception.param, "session.tools")

    def test_trusted_tool_schema_requires_a_runtime_owner(self) -> None:
        with self.assertRaisesRegex(ValueError, "has no trusted runtime owner"):
            _session(trusted_tools=[SERVER_TOOL])

    def test_trusted_and_client_tool_ownership_can_be_mixed_without_schema_override(self) -> None:
        session = _session(
            capabilities=_capabilities(trusted_tools=frozenset({"set_memory"})),
            trusted_tools=[SERVER_TOOL],
        )

        self.assertEqual(session.apply_update({"tools": [SERVER_TOOL]})["tools"], [SERVER_TOOL])
        with self.assertRaises(RealtimeProtocolError) as conflict:
            session.apply_update({"tools": [{**SERVER_TOOL, "description": "Conflicting definition"}]})
        self.assertEqual(conflict.exception.code, "tool_name_conflict")

        mixed = session.apply_update({"tools": [SERVER_TOOL, CLIENT_TOOL]})
        self.assertEqual(mixed["tools"], [SERVER_TOOL, CLIENT_TOOL])

    def test_tool_choice_must_reference_an_active_tool(self) -> None:
        session = _session()

        with self.assertRaises(RealtimeProtocolError) as required:
            session.apply_update({"tool_choice": "required"})
        self.assertEqual(required.exception.code, "unsupported_capability")

        session.apply_update({"tools": [CLIENT_TOOL]})
        with self.assertRaises(RealtimeProtocolError) as missing:
            session.apply_update({"tool_choice": {"type": "function", "name": "missing"}})
        self.assertEqual(missing.exception.code, "invalid_value")

        self.assertEqual(session.apply_update({"tool_choice": "required"})["tool_choice"], "required")

    def test_parallel_and_sequential_modes_follow_capabilities(self) -> None:
        with self.assertRaises(RealtimeProtocolError) as unsupported:
            _session().apply_update({"parallel_tool_calls": False})
        self.assertEqual(unsupported.exception.code, "unsupported_capability")

        sequential = _session(capabilities=_capabilities(sequential_tool_calls=True))
        self.assertFalse(sequential.apply_update({"parallel_tool_calls": False})["parallel_tool_calls"])


class RealtimeToolResultPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_mixed_round_suppresses_pipeline_continuation_and_preserves_properties(self) -> None:
        context = LLMContext([])
        calls_started = FunctionCallsStartedFrame(
            function_calls=(
                FunctionCallFromLLM("set_memory", "call_server", {"value": "x"}, context),
                FunctionCallFromLLM("get_weather", "call_client", {"city": "Pune"}, context),
            )
        )
        for observer_finished_first in (False, True):
            with self.subTest(observer_finished_first=observer_finished_first):
                controller = _tool_round_controller(mixed=True)
                if observer_finished_first:
                    controller.start_function_call(call_id="call_server", name="set_memory", arguments={"value": "x"})
                    controller.start_function_call(
                        call_id="call_client", name="get_weather", arguments={"city": "Pune"}
                    )
                    controller.finish_response(status="completed")
                callback = AsyncMock()
                frame = FunctionCallResultFrame(
                    function_name="set_memory",
                    tool_call_id="call_server",
                    arguments={"value": "x"},
                    result={"ok": True},
                    properties=FunctionCallResultProperties(run_llm=True, on_context_updated=callback),
                )
                policy = RealtimeToolResultPolicy(controller=controller)
                policy.push_frame = AsyncMock()

                await policy.process_frame(calls_started, FrameDirection.DOWNSTREAM)
                await policy.process_frame(frame, FrameDirection.DOWNSTREAM)

                forwarded = policy.push_frame.await_args.args[0]
                self.assertIs(forwarded, frame)
                self.assertFalse(forwarded.properties.run_llm)
                self.assertIs(forwarded.properties.on_context_updated, callback)
                self.assertTrue(forwarded.properties.is_final)

    async def test_backend_only_round_keeps_pipecat_continuation_policy(self) -> None:
        frame = FunctionCallResultFrame(
            function_name="set_memory",
            tool_call_id="call_server",
            arguments={"value": "x"},
            result={"ok": True},
        )
        policy = RealtimeToolResultPolicy(controller=_tool_round_controller(mixed=False))
        policy.push_frame = AsyncMock()
        context = LLMContext([])

        await policy.process_frame(
            FunctionCallsStartedFrame(
                function_calls=(FunctionCallFromLLM("set_memory", "call_server", {"value": "x"}, context),)
            ),
            FrameDirection.DOWNSTREAM,
        )

        await policy.process_frame(frame, FrameDirection.DOWNSTREAM)

        self.assertIs(policy.push_frame.await_args.args[0], frame)


class GatewayToolProjectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_cascaded_initial_sequential_policy_reaches_prepared_runtime(self) -> None:
        captured: dict[str, Any] = {}

        async def start_bot(ws: Any, config: dict[str, Any], controller: RealtimeSessionController) -> None:  # noqa: ARG001
            captured["config"] = config
            captured["session"] = controller.public_session()

        ws = FakeWebSocket(
            [
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "tools": [CLIENT_TOOL],
                            "parallel_tool_calls": False,
                        },
                    }
                )
            ]
        )

        await handle_realtime_websocket(
            ws,  # type: ignore[arg-type]
            sanitize_session_config=_sanitize_runtime,
            start_bot=start_bot,
        )

        self.assertIs(captured["config"]["parallel_tool_calls"], False)
        self.assertIs(captured["session"]["parallel_tool_calls"], False)

    async def test_oversized_tools_error_is_correlated_and_valid_follow_up_starts_pipeline(self) -> None:
        captured: dict[str, Any] = {}
        start_count = 0

        async def start_bot(ws: Any, config: dict[str, Any], controller: RealtimeSessionController) -> None:  # noqa: ARG001
            nonlocal start_count
            start_count += 1
            captured["controller"] = controller

        ws = FakeWebSocket(
            [
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": "too_many_tools",
                        "session": {"tools": _minimal_tools(10_000)},
                    }
                ),
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": "valid_follow_up",
                        "session": {"tools": [CLIENT_TOOL]},
                    }
                ),
            ]
        )

        await handle_realtime_websocket(
            ws,  # type: ignore[arg-type]
            sanitize_session_config=_sanitize_runtime,
            start_bot=start_bot,
        )

        error = next(event for event in ws.sent if event["type"] == "error")
        self.assertEqual(error["error"]["code"], "invalid_tool_schema")
        self.assertEqual(error["error"]["param"], "session.tools")
        self.assertEqual(error["error"]["event_id"], "too_many_tools")
        self.assertEqual(start_count, 1)
        self.assertEqual(captured["controller"].client_tool_names(), frozenset({"get_weather"}))
        self.assertEqual(ws.sent[-1]["type"], "session.updated")

    async def test_gateway_projects_mixed_server_and_client_tool_ownership(self) -> None:
        captured: dict[str, Any] = {}

        async def start_bot(ws: Any, config: dict[str, Any], controller: RealtimeSessionController) -> None:  # noqa: ARG001
            captured["config"] = config
            captured["controller"] = controller

        ws = FakeWebSocket(
            [
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": "mixed_tools_1",
                        "session": {"tools": [SERVER_TOOL, CLIENT_TOOL], "tool_choice": "required"},
                    }
                )
            ]
        )

        await handle_realtime_websocket(
            ws,  # type: ignore[arg-type]
            sanitize_session_config=_sanitize_runtime,
            start_bot=start_bot,
            resolve_server_tools=lambda _config: ["set_memory"],
            resolve_server_tool_schemas=lambda _config: [SERVER_TOOL],
        )

        self.assertEqual(captured["config"]["server_tools"], ["set_memory"])
        self.assertEqual(captured["config"]["client_tools"], [CLIENT_TOOL])
        self.assertEqual(captured["config"]["tool_choice"], "required")
        self.assertEqual(captured["controller"].client_tool_names(), frozenset({"get_weather"}))
        self.assertEqual(ws.sent[-1]["type"], "session.updated")
        self.assertEqual(ws.sent[-1]["session"]["tools"], [SERVER_TOOL, CLIENT_TOOL])
        self.assertEqual(ws.sent[-1]["session"]["nvidia"]["server_tools"], ["set_memory"])

    async def test_pipeline_without_function_tool_support_rejects_client_tools(self) -> None:
        ws = FakeWebSocket(
            [
                json.dumps(
                    {"type": "session.update", "event_id": "tools_disabled_1", "session": {"tools": [CLIENT_TOOL]}}
                )
            ]
        )

        await handle_realtime_websocket(
            ws,  # type: ignore[arg-type]
            sanitize_session_config=_sanitize_runtime,
            default_pipeline_mode="omni-assistant-subagents",
        )

        self.assertEqual(ws.sent[-1]["type"], "error")
        self.assertEqual(ws.sent[-1]["error"]["code"], "unsupported_capability")
        self.assertEqual(ws.sent[-1]["error"]["param"], "session.tools")


class SerializerToolAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_oversized_tools_error_is_correlated_and_serializer_remains_usable(self) -> None:
        controller = RealtimeSessionController(model=MODEL, voice=VOICE, runtime_config={})
        serializer = RealtimeFrameSerializer(controller=controller)
        events: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            events.append(event)

        async def emit_batch(batch: list[dict[str, Any]]) -> None:
            events.extend(batch)

        serializer.set_emit(emit, emit_batch)
        await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "event_id": "aggregate_live_tools",
                    "session": {"tools": _aggregate_oversized_tools()},
                }
            )
        )
        await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "event_id": "valid_live_follow_up",
                    "session": {"type": "realtime"},
                }
            )
        )

        self.assertEqual(events[0]["type"], "error")
        self.assertEqual(events[0]["error"]["code"], "invalid_tool_schema")
        self.assertEqual(events[0]["error"]["param"], "session.tools")
        self.assertEqual(events[0]["error"]["event_id"], "aggregate_live_tools")
        self.assertEqual(events[1]["type"], "session.updated")


if __name__ == "__main__":
    unittest.main()
