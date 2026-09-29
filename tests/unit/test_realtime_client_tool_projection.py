# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.services.llm_service import FunctionCallParams
from pipecat.utils.async_tool_cancellation import CANCEL_ASYNC_TOOL_NAME

from realtime.client_tools import RealtimeClientToolProjection
from realtime.controller import RealtimeSessionController
from realtime.mcp import RealtimeMCPRuntime
from realtime.protocol import RealtimeProtocolError
from realtime.session import RealtimeSessionCapabilities

MODEL = "nvidia/nemotron-realtime"
VOICE = "Magpie-Multilingual.EN-US.Aria"


class _FakeLLM:
    def __init__(self) -> None:
        self.handlers: dict[str | None, Any] = {}
        self.options: dict[str | None, dict[str, Any]] = {}

    def register_function(self, name: str | None, handler: Any, **options: Any) -> None:
        if name == CANCEL_ASYNC_TOOL_NAME:
            raise ValueError("Pipecat reserved tool name")
        self.handlers[name] = handler
        self.options[name] = options


def _tool(name: str, *, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": f"Run {name}",
        "parameters": parameters or {"type": "object", "properties": {}},
    }


def _params(name: str, arguments: dict[str, Any]) -> FunctionCallParams:
    async def result_callback(_result: Any, *, properties: Any = None) -> None:
        return None

    return FunctionCallParams(
        function_name=name,
        tool_call_id="call_projection",
        arguments=arguments,
        llm=MagicMock(),
        pipeline_worker=MagicMock(),
        context=MagicMock(),
        result_callback=result_callback,
    )


def _controller() -> RealtimeSessionController:
    return RealtimeSessionController(
        model=MODEL,
        voice=VOICE,
        runtime_config={},
        capabilities=RealtimeSessionCapabilities(
            voices=frozenset({VOICE}),
            function_tools=True,
        ),
    )


class RealtimeClientToolProjectionTests(unittest.IsolatedAsyncioTestCase):
    def test_trusted_collision_rejects_before_handler_registration(self) -> None:
        llm = _FakeLLM()
        trusted = ToolsSchema(
            standard_tools=[
                FunctionSchema(
                    name="calculate_bmi",
                    description="Calculate BMI",
                    properties={},
                    required=[],
                )
            ]
        )

        async def client_broker(_params: FunctionCallParams) -> None:
            return None

        projection = RealtimeClientToolProjection(scope_id="session-collision")
        with self.assertRaisesRegex(ValueError, "conflicts with a trusted pipeline tool"):
            projection.configure(
                llm,  # type: ignore[arg-type]
                [_tool("calculate_bmi")],
                client_tool_handler=client_broker,
                trusted_tools=trusted,
            )

        self.assertEqual(llm.handlers, {})

    def test_empty_client_projection_preserves_trusted_runtime(self) -> None:
        llm = _FakeLLM()
        trusted = ToolsSchema(standard_tools=[])

        async def trusted_handler(_params: FunctionCallParams) -> None:
            return None

        async def client_broker(_params: FunctionCallParams) -> None:
            return None

        llm.register_function("trusted", trusted_handler, cancel_on_interruption=True)
        combined = RealtimeClientToolProjection(scope_id="session-trusted").configure(
            llm,  # type: ignore[arg-type]
            [],
            client_tool_handler=client_broker,
            trusted_tools=trusted,
            trusted_tool_names={"trusted"},
        )

        self.assertIs(combined, trusted)
        self.assertEqual(set(llm.handlers), {"trusted", None})
        self.assertIs(llm.handlers["trusted"], trusted_handler)
        self.assertEqual(
            llm.options, {"trusted": {"cancel_on_interruption": True}, None: {"cancel_on_interruption": False}}
        )

    def test_existing_catch_all_handler_is_not_replaced(self) -> None:
        llm = _FakeLLM()
        existing = object()
        llm._functions = {None: existing}

        async def client_broker(_params: FunctionCallParams) -> None:
            return None

        projection = RealtimeClientToolProjection(scope_id="session-catch-all")
        with self.assertRaisesRegex(RuntimeError, "cannot replace an existing catch-all"):
            projection.configure(llm, [], client_tool_handler=client_broker)  # type: ignore[arg-type]

        self.assertIs(llm._functions[None], existing)
        self.assertEqual(llm.handlers, {})

    def test_mixed_projection_preserves_trusted_schema_and_registration(self) -> None:
        llm = _FakeLLM()

        async def trusted_handler(_params: FunctionCallParams) -> None:
            return None

        async def client_broker(_params: FunctionCallParams) -> None:
            return None

        standard_tool = FunctionSchema(
            name="trusted_standard",
            description="A trusted standard tool",
            properties={},
            required=[],
        )
        trusted_openai_tool = {
            "type": "function",
            "function": {
                "name": "trusted_custom",
                "parameters": {"type": "object"},
            },
        }
        trusted_gemini_tool = {"google_search": {}}
        trusted = ToolsSchema(
            standard_tools=[standard_tool],
            custom_tools={
                AdapterType.OPENAI: [trusted_openai_tool],
                AdapterType.GEMINI: [trusted_gemini_tool],
            },
        )
        llm.register_function("trusted_standard", trusted_handler, cancel_on_interruption=False)
        projection = RealtimeClientToolProjection(scope_id="session-mixed")
        combined = projection.configure(
            llm,  # type: ignore[arg-type]
            [_tool("client_lookup")],
            client_tool_handler=client_broker,
            trusted_tools=trusted,
            trusted_tool_names={"trusted_standard", "trusted_custom"},
        )

        self.assertIsNot(combined, trusted)
        self.assertEqual(combined.standard_tools, [standard_tool])
        self.assertEqual(combined.custom_tools[AdapterType.GEMINI], [trusted_gemini_tool])
        self.assertEqual(trusted.custom_tools[AdapterType.OPENAI], [trusted_openai_tool])
        projected_name = combined.custom_tools[AdapterType.OPENAI][-1]["function"]["name"]
        self.assertEqual(projected_name, "client_lookup")
        self.assertIs(llm.handlers["trusted_standard"], trusted_handler)
        self.assertEqual(llm.options["trusted_standard"], {"cancel_on_interruption": False})
        self.assertIn(None, llm.handlers)

    def test_reserved_alias_cannot_be_reclaimed_as_a_public_name(self) -> None:
        llm = _FakeLLM()

        async def client_broker(_params: FunctionCallParams) -> None:
            return None

        projection = RealtimeClientToolProjection(scope_id="session-alias-collision")
        projection.configure(llm, [], client_tool_handler=client_broker)  # type: ignore[arg-type]
        first = projection.project_tools([_tool(CANCEL_ASYNC_TOOL_NAME)], "auto")
        first_pipeline_name = first.pipeline_tools[0]["name"]
        second = projection.project_tools(
            [_tool(CANCEL_ASYNC_TOOL_NAME), _tool(first_pipeline_name)],
            {"type": "function", "name": first_pipeline_name},
        )
        second_provider_names = {tool["name"] for tool in second.pipeline_tools}

        self.assertEqual(first.bindings, {first_pipeline_name: CANCEL_ASYNC_TOOL_NAME})
        self.assertTrue(second_provider_names.isdisjoint({CANCEL_ASYNC_TOOL_NAME, first_pipeline_name}))
        self.assertEqual(set(second.bindings.values()), {CANCEL_ASYNC_TOOL_NAME, first_pipeline_name})
        selected_provider_name = second.pipeline_tool_choice["name"]
        self.assertEqual(second.bindings[selected_provider_name], first_pipeline_name)

    def test_active_response_keeps_its_exact_binding_snapshot(self) -> None:
        llm = _FakeLLM()

        async def client_broker(_params: FunctionCallParams) -> None:
            return None

        projection = RealtimeClientToolProjection(scope_id="session-response-snapshot")
        projection.configure(llm, [], client_tool_handler=client_broker)  # type: ignore[arg-type]
        first = projection.project_tools([_tool("lookup")], "auto")
        first_pipeline_name = next(iter(first.bindings))
        self.assertEqual(first_pipeline_name, "lookup")
        controller = _controller()
        controller.bind_session_tool_projection(client_tool_bindings={}, mcp_pipeline_names=frozenset())
        controller.start_response(
            tools=[_tool("lookup")],
            client_tool_bindings=first.bindings,
        )

        projection.project_tools(
            [_tool("lookup")],
            "auto",
            occupied_names={"lookup"},
        )
        events = controller.start_function_call(
            call_id="call_snapshot",
            name=first_pipeline_name,
            arguments={"query": "weather"},
        )

        record = controller.tool_call("call_snapshot")
        self.assertEqual(record.name, "lookup")
        self.assertEqual(record.pipeline_name, first_pipeline_name)
        self.assertEqual(
            next(event for event in events if event["type"] == "response.function_call_arguments.done")["name"],
            "lookup",
        )

    def test_controller_rejects_cross_public_shadowing(self) -> None:
        controller = _controller()
        tools = [_tool("first"), _tool("second")]

        controller.apply_session_update({"tools": tools})
        with self.assertRaisesRegex(ValueError, "cannot shadow another public tool"):
            controller.bind_session_tool_projection(
                client_tool_bindings={"second": "first"},
                mcp_pipeline_names=frozenset(),
            )

    def test_provider_filtered_client_tool_call_is_rejected(self) -> None:
        controller = _controller()
        controller.apply_session_update({"tools": [_tool("lookup")]})
        controller.bind_session_tool_projection(
            client_tool_bindings={},
            mcp_pipeline_names=frozenset(),
        )
        controller.start_response()

        with self.assertRaises(RealtimeProtocolError) as raised:
            controller.start_function_call(
                call_id="call_unprojected",
                name="lookup",
                arguments={},
            )

        self.assertEqual(raised.exception.code, "unknown_tool")
        self.assertIsNone(controller.tool_call("call_unprojected"))

    async def test_session_projection_keeps_public_schema_and_wire_identity(self) -> None:
        public_tool = _tool(CANCEL_ASYNC_TOOL_NAME)
        public_choice = {"type": "function", "name": CANCEL_ASYNC_TOOL_NAME}
        controller = _controller()
        controller.apply_session_update(
            {
                "tools": [public_tool],
                "tool_choice": public_choice,
            }
        )
        llm = _FakeLLM()
        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=AsyncMock())
        broker_calls: list[FunctionCallParams] = []

        async def client_broker(params: FunctionCallParams) -> None:
            broker_calls.append(params)

        configured = runtime.client_tool_projection.configure(
            llm,  # type: ignore[arg-type]
            [public_tool],
            client_tool_handler=client_broker,
        )
        prepared = await runtime.prepare_session(llm)  # type: ignore[arg-type]

        pipeline_name = next(iter(prepared.client_tool_bindings))
        self.assertEqual(prepared.client_tool_bindings, {pipeline_name: CANCEL_ASYNC_TOOL_NAME})
        self.assertEqual(
            prepared.pipeline_tool_choice,
            {"type": "function", "function": {"name": pipeline_name}},
        )
        self.assertEqual(configured.custom_tools[AdapterType.OPENAI][0]["function"]["name"], pipeline_name)
        self.assertEqual(controller.public_session()["tools"], [public_tool])
        self.assertEqual(controller.public_session()["tool_choice"], public_choice)
        self.assertEqual(set(llm.handlers), {None})
        self.assertFalse(llm.options[None]["cancel_on_interruption"])

        arguments = {"tool_call_id": "call_other"}
        await llm.handlers[None](_params(pipeline_name, arguments))
        self.assertEqual(len(broker_calls), 1)
        self.assertEqual(broker_calls[0].function_name, CANCEL_ASYNC_TOOL_NAME)
        self.assertIs(broker_calls[0].arguments, arguments)

        controller.start_response()
        events = controller.start_function_call(
            call_id="call_session_reserved",
            name=pipeline_name,
            arguments={},
        )
        record = controller.tool_call("call_session_reserved")
        self.assertEqual(record.name, CANCEL_ASYNC_TOOL_NAME)
        self.assertEqual(record.pipeline_name, pipeline_name)
        self.assertEqual(
            next(event for event in events if event["type"] == "response.function_call_arguments.done")["name"],
            CANCEL_ASYNC_TOOL_NAME,
        )

    async def test_response_local_projection_does_not_mutate_session_defaults(self) -> None:
        public_tool = _tool(CANCEL_ASYNC_TOOL_NAME)
        public_choice = {"type": "function", "name": CANCEL_ASYNC_TOOL_NAME}
        controller = _controller()
        llm = _FakeLLM()
        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=AsyncMock())

        async def client_broker(_params: FunctionCallParams) -> None:
            return None

        runtime.client_tool_projection.configure(
            llm,  # type: ignore[arg-type]
            [],
            client_tool_handler=client_broker,
        )
        await runtime.prepare_session(llm)  # type: ignore[arg-type]
        prepared = await runtime.prepare_tools([public_tool], public_choice)
        runtime.commit_response_tools(prepared)

        pipeline_name = next(iter(prepared.client_tool_bindings))
        self.assertEqual(prepared.client_tool_bindings, {pipeline_name: CANCEL_ASYNC_TOOL_NAME})
        self.assertEqual(
            prepared.pipeline_tool_choice,
            {"type": "function", "function": {"name": pipeline_name}},
        )
        self.assertEqual(controller.public_session()["tools"], [])
        self.assertEqual(controller.public_session()["tool_choice"], "auto")

        controller.start_response(
            tools=[public_tool],
            client_tool_bindings=prepared.client_tool_bindings,
            mcp_pipeline_names=prepared.mcp_pipeline_names,
        )
        events = controller.start_function_call(
            call_id="call_response_reserved",
            name=pipeline_name,
            arguments={},
        )
        record = controller.tool_call("call_response_reserved")
        self.assertEqual(record.name, CANCEL_ASYNC_TOOL_NAME)
        self.assertEqual(record.pipeline_name, pipeline_name)
        self.assertEqual(
            next(event for event in events if event["type"] == "response.function_call_arguments.done")["name"],
            CANCEL_ASYNC_TOOL_NAME,
        )

    async def test_response_tool_churn_is_bounded_without_evicting_live_bindings(self) -> None:
        llm = _FakeLLM()
        broker_calls: list[FunctionCallParams] = []

        async def client_broker(params: FunctionCallParams) -> None:
            broker_calls.append(params)

        projection = RealtimeClientToolProjection(scope_id="session-churn")
        projection.configure(llm, [], client_tool_handler=client_broker)  # type: ignore[arg-type]
        with patch("realtime.client_tools._MAX_CLIENT_TOOL_BINDINGS", 2):
            first = projection.project_tools([_tool("first")], "auto")
            second = projection.project_tools([_tool("second")], "auto")
            with self.assertRaises(RealtimeProtocolError) as raised:
                projection.project_tools([_tool("third")], "auto")

        self.assertEqual(raised.exception.code, "client_tool_binding_limit")
        self.assertEqual(raised.exception.param, "tools")
        first_pipeline_name = next(iter(first.bindings))
        second_pipeline_name = next(iter(second.bindings))
        await llm.handlers[None](_params(first_pipeline_name, {}))
        await llm.handlers[None](_params(second_pipeline_name, {}))
        self.assertEqual([call.function_name for call in broker_calls], ["first", "second"])

    async def test_failed_mcp_response_preparation_does_not_consume_client_binding_budget(self) -> None:
        llm = _FakeLLM()
        runtime = RealtimeMCPRuntime(controller=_controller(), emit_batch=AsyncMock())

        async def client_broker(_params: FunctionCallParams) -> None:
            return None

        runtime.client_tool_projection.configure(
            llm,  # type: ignore[arg-type]
            [],
            client_tool_handler=client_broker,
        )
        runtime.bind_llm(llm)  # type: ignore[arg-type]
        undefined_mcp = {"type": "mcp", "server_label": "undefined"}
        with patch("realtime.client_tools._MAX_CLIENT_TOOL_BINDINGS", 1):
            with self.assertRaises(RealtimeProtocolError) as raised:
                await runtime.prepare_tools([_tool("rejected_client"), undefined_mcp], "auto")
            accepted = await runtime.prepare_tools([_tool("accepted_client")], "auto")
            runtime.commit_response_tools(accepted)

        self.assertEqual(raised.exception.code, "mcp_server_not_defined")
        self.assertEqual(set(accepted.client_tool_bindings.values()), {"accepted_client"})
        self.assertEqual(runtime._pending_response_preparations, {})

    async def test_rolled_back_session_preparation_restores_client_binding_registry(self) -> None:
        llm = _FakeLLM()
        runtime = RealtimeMCPRuntime(controller=_controller(), emit_batch=AsyncMock())

        async def client_broker(_params: FunctionCallParams) -> None:
            return None

        runtime.client_tool_projection.configure(
            llm,  # type: ignore[arg-type]
            [],
            client_tool_handler=client_broker,
        )
        runtime.bind_llm(llm)  # type: ignore[arg-type]
        with patch("realtime.client_tools._MAX_CLIENT_TOOL_BINDINGS", 1):
            rejected = await runtime.prepare_session_update([_tool("rejected_client")], "auto")
            runtime.rollback_session_update(rejected)
            accepted = await runtime.prepare_session_update([_tool("accepted_client")], "auto")

        self.assertEqual(set(accepted.client_tool_bindings.values()), {"accepted_client"})
        runtime.commit_session_update(accepted)

    async def test_client_arguments_are_forwarded_without_server_schema_execution(self) -> None:
        llm = _FakeLLM()
        broker_calls: list[FunctionCallParams] = []

        async def client_broker(params: FunctionCallParams) -> None:
            broker_calls.append(params)

        projection = RealtimeClientToolProjection(scope_id="session-arguments")
        schema = _tool(
            "lookup",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        )
        projection.configure(llm, [schema], client_tool_handler=client_broker)  # type: ignore[arg-type]
        prepared = projection.project_tools([schema], "auto")
        pipeline_name = next(iter(prepared.bindings))
        arguments = {"value": "client-validates-this"}

        await llm.handlers[None](_params(pipeline_name, arguments))

        self.assertEqual(len(broker_calls), 1)
        self.assertEqual(broker_calls[0].function_name, "lookup")
        self.assertIs(broker_calls[0].arguments, arguments)


if __name__ == "__main__":
    unittest.main()
