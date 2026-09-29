# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

from __future__ import annotations

import asyncio
import copy
import json
import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from mcp.types import CallToolResult, Tool
from pipecat.adapters.schemas.tools_schema import AdapterType
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallParams

from realtime.controller import RealtimeSessionController
from realtime.frames import RealtimeDeferredResponseCreateFrame, RealtimeResponseCreateFrame
from realtime.mcp import (
    MCPExecutionResult,
    MCPToolBinding,
    MCPToolPreparationPoisonedError,
    RealtimeMCPRuntime,
    _http_headers,
    _validate_server_url,
)
from realtime.protocol import RealtimeProtocolError
from realtime.serializer import RealtimeFrameSerializer
from realtime.session import AudioFormatCapability, RealtimeSessionCapabilities
from realtime.transport import RealtimeManualResponseGate

MODEL = "nvidia/nemotron-realtime"
VOICE = "Magpie-Multilingual.EN-US.Aria"
PIPELINE_NAME = "private_fixture_echo"
SERVER_LABEL = "fixture"
TOOL_NAME = "echo"
CALL_ID = "call_mcp_1"
SERVER_URL = "https://mcp.example.test/rpc"


class _FakeLLM:
    def __init__(self) -> None:
        self._functions: dict[str, tuple[Any, bool]] = {}

    def register_function(self, name: str, handler: Any, *, cancel_on_interruption: bool) -> None:
        if name in self._functions:
            raise ValueError(f"duplicate function {name}")
        self._functions[name] = (handler, cancel_on_interruption)

    def unregister_function(self, name: str) -> None:
        self._functions.pop(name)


class _FakeMCPWorker:
    def __init__(self, tools: list[Tool], result: CallToolResult | None = None) -> None:
        self.tools = tools
        self.result = result or CallToolResult.model_validate(
            {
                "content": [{"type": "text", "text": "fixture result"}],
                "structuredContent": {"answer": 42},
                "isError": False,
            }
        )
        self.started = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    @property
    def dead(self) -> bool:
        return False

    async def start(self) -> None:
        self.started += 1

    async def list_tools(self) -> list[Tool]:
        return list(self.tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        self.calls.append((name, copy.deepcopy(arguments)))
        return self.result

    def shutdown(self) -> None:
        self.closed = True


def _mcp_tool_definition(*, require_approval: Any = "never") -> dict[str, Any]:
    return {
        "type": "mcp",
        "server_label": SERVER_LABEL,
        "server_url": SERVER_URL,
        "authorization": "private-bearer",
        "headers": {"X-Tenant": "private-tenant"},
        "allowed_tools": [TOOL_NAME],
        "require_approval": require_approval,
    }


def _mcp_controller(tools: list[dict[str, Any]]) -> RealtimeSessionController:
    controller = RealtimeSessionController(
        model=MODEL,
        voice=VOICE,
        runtime_config={},
        capabilities=RealtimeSessionCapabilities(
            voices=frozenset({VOICE}),
            function_tools=True,
            mcp_tools=True,
        ),
    )
    controller.apply_session_update({"tools": tools})
    return controller


def _advertised_tools() -> list[Tool]:
    return [
        Tool.model_validate(
            {
                "name": TOOL_NAME,
                "description": "Echo one value",
                "inputSchema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
                "annotations": {"readOnlyHint": True},
            }
        ),
        Tool.model_validate(
            {
                "name": "write_value",
                "description": "A filtered write operation",
                "inputSchema": {"type": "object"},
                "annotations": {"readOnlyHint": False},
            }
        ),
    ]


def _controller_with_mcp_call() -> RealtimeSessionController:
    controller = RealtimeSessionController(
        model=MODEL,
        voice=VOICE,
        runtime_config={},
        capabilities=RealtimeSessionCapabilities(
            voices=frozenset({VOICE}),
            function_tools=True,
            mcp_tools=True,
        ),
    )
    controller.register_mcp_tool_binding(
        pipeline_name=PIPELINE_NAME,
        server_label=SERVER_LABEL,
        name=TOOL_NAME,
    )
    controller.bind_session_mcp_tools(frozenset({PIPELINE_NAME}))
    controller.start_response(mcp_pipeline_names=frozenset({PIPELINE_NAME}))
    controller.start_function_call(
        call_id=CALL_ID,
        name=PIPELINE_NAME,
        arguments={"value": "hello"},
    )
    return controller


def _install_pending_approval(
    runtime: RealtimeMCPRuntime,
    controller: RealtimeSessionController,
) -> tuple[str, asyncio.Future[Any]]:
    request_id, _ = controller.request_mcp_approval(CALL_ID)
    future = asyncio.get_running_loop().create_future()
    runtime._approvals[request_id] = future
    runtime._approval_calls[request_id] = CALL_ID
    return request_id, future


@contextmanager
def _use_mcp_worker(worker: _FakeMCPWorker):
    with (
        patch.dict(
            os.environ,
            {"REALTIME_MCP_ALLOWED_SERVER_URLS": json.dumps([SERVER_URL])},
            clear=False,
        ),
        patch("realtime.mcp._MCPServerWorker", return_value=worker),
    ):
        yield


def _call_params(
    result_callback: Any,
    *,
    function_name: str = PIPELINE_NAME,
    call_id: str = CALL_ID,
    arguments: dict[str, Any] | None = None,
) -> FunctionCallParams:
    return FunctionCallParams(
        function_name=function_name,
        tool_call_id=call_id,
        arguments={"value": "hello"} if arguments is None else arguments,
        llm=None,
        pipeline_worker=None,
        context=None,
        result_callback=result_callback,
    )


def _serializer_for_mcp(
    controller: RealtimeSessionController,
    runtime: RealtimeMCPRuntime,
    emit_batch: Any,
) -> RealtimeFrameSerializer:
    async def emit(event: dict[str, Any]) -> None:
        await emit_batch([event])

    serializer = RealtimeFrameSerializer(controller=controller)
    serializer.set_emit(emit, emit_batch)
    serializer.set_mcp_runtime(runtime)
    return serializer


def _approval_response_event(
    request_id: str,
    *,
    item_id: str,
    approve: bool,
    event_id: str | None = None,
    reason: str | None = None,
) -> str:
    item: dict[str, Any] = {
        "id": item_id,
        "type": "mcp_approval_response",
        "approval_request_id": request_id,
        "approve": approve,
    }
    if reason is not None:
        item["reason"] = reason
    event = {"type": "conversation.item.create", "item": item}
    if event_id is not None:
        event["event_id"] = event_id
    return json.dumps(event)


class MCPConfigurationAndHappyPathTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _binding_policy(
        runtime: RealtimeMCPRuntime,
        prepared: Any,
    ) -> dict[str, bool]:
        return {
            runtime._bindings[pipeline_name].name: runtime._bindings[pipeline_name].require_approval
            for pipeline_name in prepared.mcp_pipeline_names
        }

    def test_unknown_pipeline_binding_does_not_reserve_a_response(self) -> None:
        controller = _mcp_controller([])

        with self.assertRaisesRegex(ValueError, "has no binding"):
            controller.start_response(mcp_pipeline_names=frozenset({"missing"}))

        self.assertFalse(controller.response_in_progress)
        self.assertIsNone(controller.active_response_id)

    async def _prepare_runtime(
        self,
        *,
        require_approval: Any = "never",
        result: CallToolResult | None = None,
    ) -> tuple[
        RealtimeSessionController,
        RealtimeMCPRuntime,
        _FakeMCPWorker,
        _FakeLLM,
        list[dict[str, Any]],
        Any,
    ]:
        controller = _mcp_controller([_mcp_tool_definition(require_approval=require_approval)])
        emitted: list[dict[str, Any]] = []

        async def emit_batch(events: list[dict[str, Any]]) -> None:
            emitted.extend(copy.deepcopy(events))

        worker = _FakeMCPWorker(_advertised_tools(), result=result)
        llm = _FakeLLM()
        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=emit_batch)
        with _use_mcp_worker(worker):
            prepared = await runtime.prepare_session(llm)  # type: ignore[arg-type]
        return controller, runtime, worker, llm, emitted, prepared

    async def test_server_url_and_headers_enforce_the_connection_egress_boundary(self) -> None:
        with patch.dict(
            os.environ,
            {"REALTIME_MCP_ALLOWED_SERVER_URLS": json.dumps([SERVER_URL])},
            clear=False,
        ):
            await _validate_server_url(SERVER_URL)
            cases = (
                ("/relative", "invalid_mcp_server_url"),
                ("https://user:secret@mcp.example.test/rpc", "invalid_mcp_server_url"),
                ("https://mcp.example.test/rpc#fragment", "invalid_mcp_server_url"),
                ("https://unlisted.example.test/rpc", "mcp_server_not_allowed"),
            )
            for url, code in cases:
                with self.subTest(url=url), self.assertRaises(RealtimeProtocolError) as raised:
                    await _validate_server_url(url)
                self.assertEqual(raised.exception.code, code)

        self.assertEqual(
            _http_headers(
                {
                    "authorization": "private-bearer",
                    "headers": {"X-Tenant": "tenant-1"},
                }
            ),
            {"X-Tenant": "tenant-1", "Authorization": "Bearer private-bearer"},
        )
        for header in ("Host", "Content-Length", "Proxy-Authorization", "Upgrade"):
            with self.subTest(header=header), self.assertRaises(RealtimeProtocolError) as raised:
                _http_headers({"headers": {header: "client-controlled"}})
            self.assertEqual(raised.exception.code, "invalid_mcp_header")
            self.assertEqual(raised.exception.param, "tools.headers")

    async def test_prepare_session_discovers_filters_and_registers_native_mcp_tools(self) -> None:
        client_tool = {
            "type": "function",
            "name": TOOL_NAME,
            "description": "A client function with the same public name",
            "parameters": {"type": "object"},
        }
        definition = _mcp_tool_definition(require_approval="always")
        controller = _mcp_controller([client_tool, definition])
        emitted: list[dict[str, Any]] = []

        async def emit_batch(events: list[dict[str, Any]]) -> None:
            emitted.extend(copy.deepcopy(events))

        worker = _FakeMCPWorker(_advertised_tools())
        llm = _FakeLLM()
        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=emit_batch)

        async def client_handler(_params: FunctionCallParams) -> None:
            return None

        runtime.client_tool_projection.configure(
            llm,  # type: ignore[arg-type]
            [client_tool],
            client_tool_handler=client_handler,
        )
        with _use_mcp_worker(worker):
            prepared = await runtime.prepare_session(llm)  # type: ignore[arg-type]

        self.assertEqual(
            [event["type"] for event in emitted],
            [
                "conversation.item.added",
                "mcp_list_tools.in_progress",
                "conversation.item.done",
                "mcp_list_tools.completed",
            ],
        )
        list_item_id = emitted[0]["item"]["id"]
        self.assertEqual({event.get("item_id") for event in (emitted[1], emitted[3])}, {list_item_id})
        listed_item = emitted[2]["item"]
        self.assertEqual(listed_item["type"], "mcp_list_tools")
        self.assertNotIn("status", listed_item)
        self.assertEqual([tool["name"] for tool in listed_item["tools"]], [TOOL_NAME])
        self.assertNotIn("private-bearer", json.dumps(emitted))
        self.assertNotIn("private-tenant", json.dumps(emitted))

        client_pipeline_name = prepared.pipeline_tools[0]["name"]
        self.assertEqual(client_pipeline_name, TOOL_NAME)
        self.assertEqual(prepared.client_tool_bindings, {client_pipeline_name: TOOL_NAME})
        self.assertEqual(
            {key: value for key, value in prepared.pipeline_tools[0].items() if key != "name"},
            {key: value for key, value in client_tool.items() if key != "name"},
        )
        self.assertEqual(len(prepared.pipeline_tools), 2)
        private_name = prepared.pipeline_tools[1]["name"]
        self.assertNotEqual(private_name, TOOL_NAME)
        self.assertTrue(private_name.startswith("mcp_echo_"))
        self.assertEqual(prepared.pipeline_tools[1]["parameters"], _advertised_tools()[0].inputSchema)
        self.assertEqual(prepared.mcp_pipeline_names, frozenset({private_name}))
        handler, cancel_on_interruption = llm._functions[private_name]
        self.assertEqual(handler, runtime.handle_tool_call)
        self.assertFalse(cancel_on_interruption)

        controller.start_response(mcp_pipeline_names=prepared.mcp_pipeline_names)
        call_events = controller.start_function_call(
            call_id=CALL_ID,
            name=private_name,
            arguments={"value": "hello"},
        )
        self.assertIn("response.mcp_call_arguments.done", [event["type"] for event in call_events])
        native_item = next(event["item"] for event in call_events if event["type"] == "conversation.item.added")
        self.assertEqual(native_item["type"], "mcp_call")
        self.assertEqual(native_item["server_label"], SERVER_LABEL)
        self.assertEqual(native_item["name"], TOOL_NAME)

        forced = await runtime.prepare_tools(
            [client_tool, {"type": "mcp", "server_label": SERVER_LABEL}],
            {"type": "mcp", "server_label": SERVER_LABEL, "name": TOOL_NAME},
        )
        runtime.commit_response_tools(forced)
        self.assertEqual([tool["name"] for tool in forced.pipeline_tools], [private_name])
        self.assertEqual(
            forced.pipeline_tool_choice,
            {"type": "function", "function": {"name": private_name}},
        )
        self.assertEqual(worker.started, 1)

    async def test_mcp_allocation_avoids_historical_client_provider_names(self) -> None:
        controller = _mcp_controller([])
        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=AsyncMock())
        llm = _FakeLLM()

        async def client_handler(_params: FunctionCallParams) -> None:
            return None

        runtime.client_tool_projection.configure(
            llm,
            [],
            client_tool_handler=client_handler,  # type: ignore[arg-type]
        )
        runtime.bind_llm(llm)  # type: ignore[arg-type]
        historical_name = "mcp_historical_client"
        runtime.client_tool_projection.project_tools(
            [{"type": "function", "name": historical_name, "parameters": {"type": "object"}}],
            "auto",
        )
        worker = _FakeMCPWorker(_advertised_tools())
        with (
            _use_mcp_worker(worker),
            patch("realtime.mcp._pipeline_name", side_effect=[historical_name, "mcp_safe_name"]),
        ):
            prepared = await runtime.prepare_tools([_mcp_tool_definition()], "auto")

        self.assertEqual(prepared.mcp_pipeline_names, frozenset({"mcp_safe_name"}))
        runtime.rollback_response_tools(prepared)

    async def test_committed_session_policy_replaces_label_reference_default(self) -> None:
        controller = _mcp_controller([_mcp_tool_definition()])
        worker = _FakeMCPWorker(_advertised_tools())
        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=AsyncMock())
        llm = _FakeLLM()
        updated_definition = {
            **_mcp_tool_definition(require_approval="always"),
            "allowed_tools": ["write_value"],
        }
        with _use_mcp_worker(worker):
            await runtime.prepare_session(llm)  # type: ignore[arg-type]
            prepared_update = await runtime.prepare_session_update([updated_definition], "auto")
            runtime.commit_session_update(prepared_update)
            referenced = await runtime.prepare_tools(
                [{"type": "mcp", "server_label": SERVER_LABEL}],
                "auto",
            )
            runtime.commit_response_tools(referenced)

        self.assertEqual(self._binding_policy(runtime, referenced), {"write_value": True})

    async def test_rolled_back_session_policy_does_not_change_label_reference_default(self) -> None:
        controller = _mcp_controller([_mcp_tool_definition()])
        worker = _FakeMCPWorker(_advertised_tools())
        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=AsyncMock())
        llm = _FakeLLM()
        updated_definition = {
            **_mcp_tool_definition(require_approval="always"),
            "allowed_tools": ["write_value"],
        }
        with _use_mcp_worker(worker):
            await runtime.prepare_session(llm)  # type: ignore[arg-type]
            prepared_update = await runtime.prepare_session_update([updated_definition], "auto")
            rolled_back_names = prepared_update.mcp_pipeline_names
            runtime.rollback_session_update(prepared_update)
            referenced = await runtime.prepare_tools(
                [{"type": "mcp", "server_label": SERVER_LABEL}],
                "auto",
            )
            runtime.commit_response_tools(referenced)

        self.assertEqual(self._binding_policy(runtime, referenced), {TOOL_NAME: False})
        self.assertTrue(rolled_back_names.isdisjoint(runtime._bindings))
        self.assertTrue(rolled_back_names.isdisjoint(llm._functions))

    async def test_response_local_policy_does_not_replace_session_label_reference(self) -> None:
        controller = _mcp_controller([_mcp_tool_definition()])
        worker = _FakeMCPWorker(_advertised_tools())
        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=AsyncMock())
        llm = _FakeLLM()
        response_definition = {
            **_mcp_tool_definition(require_approval="always"),
            "allowed_tools": ["write_value"],
        }
        with _use_mcp_worker(worker):
            await runtime.prepare_session(llm)  # type: ignore[arg-type]
            response_tools = await runtime.prepare_tools([response_definition], "auto")
            runtime.commit_response_tools(response_tools)
            referenced = await runtime.prepare_tools(
                [{"type": "mcp", "server_label": SERVER_LABEL}],
                "auto",
            )
            runtime.commit_response_tools(referenced)

        self.assertEqual(self._binding_policy(runtime, response_tools), {"write_value": True})
        self.assertEqual(self._binding_policy(runtime, referenced), {TOOL_NAME: False})

    async def test_committed_session_tool_replacement_removes_old_label_default(self) -> None:
        controller = _mcp_controller([_mcp_tool_definition()])
        worker = _FakeMCPWorker(_advertised_tools())
        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=AsyncMock())
        llm = _FakeLLM()
        with _use_mcp_worker(worker):
            await runtime.prepare_session(llm)  # type: ignore[arg-type]
            prepared_update = await runtime.prepare_session_update([], "auto")
            runtime.commit_session_update(prepared_update)
            with self.assertRaises(RealtimeProtocolError) as raised:
                await runtime.prepare_tools(
                    [{"type": "mcp", "server_label": SERVER_LABEL}],
                    "auto",
                )

        self.assertEqual(raised.exception.code, "mcp_server_not_defined")

    async def test_serializer_session_update_accepts_committed_label_reference(self) -> None:
        controller = _mcp_controller([_mcp_tool_definition()])
        emitted: list[dict[str, Any]] = []

        async def emit_batch(events: list[dict[str, Any]]) -> None:
            emitted.extend(copy.deepcopy(events))

        worker = _FakeMCPWorker(_advertised_tools())
        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=emit_batch)
        llm = _FakeLLM()
        reference = {"type": "mcp", "server_label": SERVER_LABEL}
        with _use_mcp_worker(worker):
            await runtime.prepare_session(llm)  # type: ignore[arg-type]
            context = LLMContext([])
            serializer = _serializer_for_mcp(controller, runtime, emit_batch)
            serializer.set_response_gate(RealtimeManualResponseGate(controller=controller))
            serializer.bind_context(context)
            emitted.clear()

            frame = await serializer.deserialize(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {"tools": [reference]},
                    }
                )
            )
            referenced = await runtime.prepare_tools([reference], "auto")
            runtime.commit_response_tools(referenced)

        self.assertIsNone(frame)
        self.assertEqual([event["type"] for event in emitted], ["session.updated"])
        self.assertEqual(emitted[0]["session"]["tools"], [reference])
        self.assertNotIn("private-bearer", json.dumps(emitted))
        self.assertNotIn("private-tenant", json.dumps(emitted))
        self.assertEqual(controller.public_session()["tools"], [reference])
        self.assertEqual(runtime._session_definitions[SERVER_LABEL], _mcp_tool_definition())
        self.assertEqual(self._binding_policy(runtime, referenced), {TOOL_NAME: False})
        self.assertEqual(context.tool_choice, "auto")
        projected_tools = context.tools.custom_tools[AdapterType.OPENAI]
        self.assertEqual(len(projected_tools), 1)
        self.assertIn(projected_tools[0]["function"]["name"], referenced.mcp_pipeline_names)
        runtime.shutdown()

    async def test_rollback_removes_uncommitted_server_identity(self) -> None:
        controller = _mcp_controller([])
        worker = _FakeMCPWorker(_advertised_tools())
        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=AsyncMock())
        llm = _FakeLLM()
        runtime.bind_llm(llm)  # type: ignore[arg-type]
        with _use_mcp_worker(worker):
            prepared_update = await runtime.prepare_session_update([_mcp_tool_definition()], "auto")
            runtime.rollback_session_update(prepared_update)
            with self.assertRaises(RealtimeProtocolError) as raised:
                await runtime.prepare_tools(
                    [{"type": "mcp", "server_label": SERVER_LABEL}],
                    "auto",
                )

        self.assertEqual(raised.exception.code, "mcp_server_not_defined")
        self.assertNotIn(SERVER_LABEL, runtime._servers)
        self.assertTrue(worker.closed)

    async def test_failed_serializer_commit_rolls_back_prepared_mcp_policy(self) -> None:
        controller = _mcp_controller([_mcp_tool_definition()])
        emitted: list[dict[str, Any]] = []

        async def emit_batch(events: list[dict[str, Any]]) -> None:
            emitted.extend(copy.deepcopy(events))

        worker = _FakeMCPWorker(_advertised_tools())
        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=emit_batch)
        llm = _FakeLLM()
        updated_definition = {
            **_mcp_tool_definition(require_approval="always"),
            "allowed_tools": ["write_value"],
        }
        with _use_mcp_worker(worker):
            await runtime.prepare_session(llm)  # type: ignore[arg-type]
            context = LLMContext([])
            serializer = _serializer_for_mcp(controller, runtime, emit_batch)
            serializer.set_response_gate(RealtimeManualResponseGate(controller=controller))
            serializer.bind_context(context)
            emitted.clear()

            with patch.object(controller, "apply_session_update", side_effect=RuntimeError("commit failed")):
                frame = await serializer.deserialize(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {"tools": [updated_definition]},
                        }
                    )
                )
            referenced = await runtime.prepare_tools(
                [{"type": "mcp", "server_label": SERVER_LABEL}],
                "auto",
            )
            runtime.commit_response_tools(referenced)

        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["type"], "error")
        self.assertEqual(emitted[-1]["error"]["code"], "event_processing_failed")
        self.assertEqual(self._binding_policy(runtime, referenced), {TOOL_NAME: False})
        self.assertFalse(any(binding.name == "write_value" for binding in runtime._bindings.values()))
        original_public_definition = _mcp_tool_definition()
        original_public_definition.pop("authorization")
        original_public_definition.pop("headers")
        self.assertEqual(controller.public_session()["tools"], [original_public_definition])

    async def test_successful_call_completes_after_context_and_allows_response_b(self) -> None:
        native_result = CallToolResult.model_validate(
            {
                "content": [
                    {
                        "type": "text",
                        "text": "fixture result",
                        "_meta": {"part_secret": "native-only"},
                    }
                ],
                "structuredContent": {
                    "visible": {"answer": 42, "_meta": {"nested_secret": "native-only"}},
                    "_meta": {"structured_secret": "native-only"},
                },
                "_meta": {"top_secret": "native-only"},
                "isError": False,
            }
        )
        controller, runtime, worker, _llm, emitted, prepared = await self._prepare_runtime(result=native_result)
        emitted.clear()
        private_name = next(iter(prepared.mcp_pipeline_names))
        response_a = controller.start_response(mcp_pipeline_names=prepared.mcp_pipeline_names)[0]["response"]["id"]
        controller.start_function_call(
            call_id=CALL_ID,
            name=private_name,
            arguments={"value": "hello"},
        )
        runtime.notify_call_announced(CALL_ID)
        controller.finish_response(status="completed")
        captured: dict[str, Any] = {}

        async def result_callback(result: Any, *, properties: Any) -> None:
            captured["result"] = result
            captured["properties"] = properties

        await runtime.handle_tool_call(_call_params(result_callback, function_name=private_name))

        self.assertEqual(worker.calls, [(TOOL_NAME, {"value": "hello"})])
        self.assertEqual([event["type"] for event in emitted], ["response.mcp_call.in_progress"])
        self.assertEqual(dict(captured["result"])["structuredContent"], {"visible": {"answer": 42}})
        self.assertNotIn("_meta", json.dumps(dict(captured["result"])))
        self.assertFalse(captured["properties"].run_llm)

        await captured["properties"].on_context_updated()

        self.assertEqual(
            [event["type"] for event in emitted],
            [
                "response.mcp_call.in_progress",
                "response.mcp_call.completed",
                "conversation.item.done",
                "response.output_item.done",
            ],
        )
        done_item = emitted[-1]["item"]
        self.assertEqual(done_item["type"], "mcp_call")
        self.assertIn("_meta", done_item["output"])
        self.assertTrue(controller.tool_call(CALL_ID).completed)
        response_b = controller.start_response()[0]["response"]["id"]
        self.assertNotEqual(response_a, response_b)

    async def test_approval_response_controls_execution_through_the_public_item_path(self) -> None:
        for approve in (True, False):
            with self.subTest(approve=approve):
                controller, runtime, worker, _llm, emitted, prepared = await self._prepare_runtime(
                    require_approval="always"
                )
                emitted.clear()
                private_name = next(iter(prepared.mcp_pipeline_names))
                controller.start_response(mcp_pipeline_names=prepared.mcp_pipeline_names)
                controller.start_function_call(
                    call_id=CALL_ID,
                    name=private_name,
                    arguments={"value": "hello"},
                )
                runtime.notify_call_announced(CALL_ID)
                captured: dict[str, Any] = {}

                async def result_callback(
                    result: Any,
                    *,
                    properties: Any,
                    _captured: dict[str, Any] = captured,
                ) -> None:
                    _captured["result"] = result
                    _captured["properties"] = properties

                call_task = asyncio.create_task(
                    runtime.handle_tool_call(_call_params(result_callback, function_name=private_name))
                )
                for _ in range(20):
                    approval = next(
                        (
                            event["item"]
                            for event in emitted
                            if event["type"] == "conversation.item.done"
                            and event["item"]["type"] == "mcp_approval_request"
                        ),
                        None,
                    )
                    if approval is not None:
                        break
                    await asyncio.sleep(0)
                else:
                    self.fail("MCP approval request was not published")

                async def emit_batch(
                    events: list[dict[str, Any]],
                    _emitted: list[dict[str, Any]] = emitted,
                ) -> None:
                    _emitted.extend(copy.deepcopy(events))

                serializer = _serializer_for_mcp(controller, runtime, emit_batch)
                await serializer.deserialize(
                    _approval_response_event(
                        approval["id"],
                        item_id=f"item_approval_{str(approve).lower()}",
                        approve=approve,
                        reason=None if approve else "not authorized",
                    )
                )
                await asyncio.wait_for(call_task, timeout=1)
                await captured["properties"].on_context_updated()

                event_types = [event["type"] for event in emitted]
                approval_request_done = next(
                    index
                    for index, event in enumerate(emitted)
                    if event["type"] == "conversation.item.done" and event["item"]["type"] == "mcp_approval_request"
                )
                approval_response_done = next(
                    index
                    for index, event in enumerate(emitted)
                    if event["type"] == "conversation.item.done" and event["item"]["type"] == "mcp_approval_response"
                )
                terminal_type = "response.mcp_call.completed" if approve else "response.mcp_call.failed"
                self.assertLess(approval_request_done, approval_response_done)
                self.assertLess(approval_response_done, event_types.index(terminal_type))
                self.assertEqual(len(worker.calls), 1 if approve else 0)
                self.assertEqual(captured["result"].failed, not approve)


_SESSION_FUNCTION_TOOL = {
    "type": "function",
    "name": "lookup",
    "description": "Look up one value",
    "parameters": {"type": "object", "properties": {}},
}


class ResponseToolPreparationTransactionTests(unittest.IsolatedAsyncioTestCase):
    async def _function_fixture(
        self,
        *,
        deferred: bool,
    ) -> tuple[
        RealtimeSessionController,
        RealtimeFrameSerializer,
        RealtimeManualResponseGate,
        RealtimeMCPRuntime,
        _FakeLLM,
        list[dict[str, Any]],
    ]:
        controller = RealtimeSessionController(
            model=MODEL,
            voice=VOICE,
            runtime_config={},
            capabilities=RealtimeSessionCapabilities(
                voices=frozenset({VOICE}),
                function_tools=True,
                mcp_tools=True,
                supports_manual_input=True,
            ),
        )
        controller.apply_session_update({"output_modalities": ["text"]})
        emitted: list[dict[str, Any]] = []

        async def emit_batch(events: list[dict[str, Any]]) -> None:
            emitted.extend(copy.deepcopy(events))

        async def client_tool_handler(_params: FunctionCallParams) -> None:
            return None

        llm = _FakeLLM()
        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=emit_batch)
        runtime.client_tool_projection.configure(
            llm,  # type: ignore[arg-type]
            [],
            client_tool_handler=client_tool_handler,
        )
        runtime.bind_llm(llm)  # type: ignore[arg-type]
        gate = RealtimeManualResponseGate(controller=controller)
        serializer = _serializer_for_mcp(controller, runtime, emit_batch)
        serializer.set_response_gate(gate)
        if deferred:
            controller.apply_session_update({"audio": {"input": {"turn_detection": None}}})
            gate.register_commit()
            controller.start_response(output_modalities=["text"])
        return controller, serializer, gate, runtime, llm, emitted

    @staticmethod
    def _response_create(tool: dict[str, Any]) -> str:
        return json.dumps(
            {
                "type": "response.create",
                "event_id": "response_tool_transaction",
                "response": {"tools": [tool]},
            }
        )

    async def test_function_projection_rolls_back_when_admission_rejects(self) -> None:
        for deferred, admission_method in (
            (False, "validate_response_preparation"),
            (True, "register_deferred_response"),
        ):
            with self.subTest(deferred=deferred):
                controller, serializer, gate, runtime, _llm, emitted = await self._function_fixture(deferred=deferred)
                active_response_id = controller.active_response_id
                before = runtime.client_tool_projection.snapshot()

                with patch.object(
                    gate,
                    admission_method,
                    side_effect=RuntimeError("injected admission failure"),
                ):
                    frame = await serializer.deserialize(self._response_create(_SESSION_FUNCTION_TOOL))

                self.assertIsNone(frame)
                self.assertEqual(runtime.client_tool_projection.snapshot(), before)
                self.assertEqual(runtime._pending_response_preparations, {})
                self.assertEqual(controller.active_response_id, active_response_id)
                self.assertIsNone(gate._pending_deferred_response_id)
                self.assertIsNone(gate._response_preparation_id)
                self.assertFalse(serializer.connection_closed)
                self.assertEqual(emitted[-1]["error"]["code"], "event_processing_failed")

    async def test_immediate_and_deferred_admission_commit_function_projection(self) -> None:
        for deferred in (False, True):
            with self.subTest(deferred=deferred):
                _controller, serializer, gate, runtime, _llm, _emitted = await self._function_fixture(deferred=deferred)
                before = runtime.client_tool_projection.snapshot()

                frame = await serializer.deserialize(self._response_create(_SESSION_FUNCTION_TOOL))

                expected_type = RealtimeDeferredResponseCreateFrame if deferred else RealtimeResponseCreateFrame
                self.assertIsInstance(frame, expected_type)
                self.assertNotEqual(runtime.client_tool_projection.snapshot(), before)
                self.assertEqual(runtime._pending_response_preparations, {})
                self.assertIsNone(gate._response_preparation_id)
                if deferred:
                    self.assertEqual(gate._pending_deferred_response_id, frame.request_id)
                runtime.shutdown()

    async def test_commit_failure_after_admission_retires_connection(self) -> None:
        controller, serializer, gate, runtime, _llm, emitted = await self._function_fixture(deferred=False)

        with patch.object(
            runtime,
            "commit_response_tools",
            side_effect=RuntimeError("injected response tool commit failure"),
        ):
            frame = await serializer.deserialize(self._response_create(_SESSION_FUNCTION_TOOL))

        self.assertIsNone(frame)
        self.assertTrue(serializer.connection_closed)
        self.assertFalse(controller.response_in_progress)
        self.assertIsNone(gate._pending_response_marker_id)
        self.assertEqual(runtime._pending_response_preparations, {})
        self.assertEqual(emitted[-1]["error"]["code"], "event_processing_failed")

    async def test_unprovable_response_rollback_retires_connection(self) -> None:
        _controller, serializer, gate, runtime, _llm, emitted = await self._function_fixture(deferred=False)

        def mutate_projection_then_reject(_preparation_id: str) -> None:
            runtime.client_tool_projection.project_tools(
                [
                    {
                        "type": "function",
                        "name": "concurrent_tool",
                        "description": "A conflicting later projection",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
                "auto",
            )
            raise RuntimeError("injected admission failure after unexpected mutation")

        with patch.object(gate, "validate_response_preparation", side_effect=mutate_projection_then_reject):
            frame = await serializer.deserialize(self._response_create(_SESSION_FUNCTION_TOOL))

        self.assertIsNone(frame)
        self.assertTrue(serializer.connection_closed)
        self.assertEqual(runtime._pending_response_preparations, {})
        self.assertEqual(emitted[-1]["error"]["code"], "tool_preparation_state_inconsistent")

    async def test_mcp_bindings_and_server_roll_back_when_admission_rejects(self) -> None:
        controller, serializer, gate, runtime, llm, emitted = await self._function_fixture(deferred=False)
        worker = _FakeMCPWorker(_advertised_tools())
        llm_functions_before = dict(llm._functions)
        with (
            _use_mcp_worker(worker),
            patch.object(
                gate,
                "validate_response_preparation",
                side_effect=RuntimeError("injected admission failure"),
            ),
        ):
            frame = await serializer.deserialize(self._response_create(_mcp_tool_definition()))

        self.assertIsNone(frame)
        self.assertEqual(runtime._pending_response_preparations, {})
        self.assertEqual(runtime._servers, {})
        self.assertEqual(runtime._bindings, {})
        self.assertEqual(runtime._binding_names, {})
        self.assertEqual(runtime._pipeline_identities, {})
        self.assertEqual(runtime._registered_handlers, set())
        self.assertEqual(controller._mcp_tool_bindings, {})
        self.assertEqual(llm._functions, llm_functions_before)
        self.assertTrue(worker.closed)
        self.assertFalse(serializer.connection_closed)
        self.assertEqual(emitted[-1]["error"]["code"], "event_processing_failed")

    async def test_ordinary_preparation_rejection_rolls_back_and_keeps_connection_open(self) -> None:
        _controller, serializer, _gate, runtime, _llm, emitted = await self._function_fixture(deferred=False)
        close_connection = AsyncMock()
        serializer.set_connection_failure_handler(close_connection)
        before = runtime.client_tool_projection.snapshot()
        event = json.dumps(
            {
                "type": "response.create",
                "response": {
                    "tools": [
                        _SESSION_FUNCTION_TOOL,
                        {"type": "mcp", "server_label": "undefined"},
                    ]
                },
            }
        )

        frame = await serializer.deserialize(event)

        self.assertIsNone(frame)
        self.assertFalse(serializer.connection_closed)
        close_connection.assert_not_awaited()
        self.assertEqual(runtime.client_tool_projection.snapshot(), before)
        self.assertEqual(runtime._pending_response_preparations, {})
        self.assertEqual(emitted[-1]["error"]["code"], "mcp_server_not_defined")

    async def test_response_preparation_rollback_failure_closes_physical_connection(self) -> None:
        _controller, serializer, _gate, runtime, _llm, emitted = await self._function_fixture(deferred=False)
        close_connection = AsyncMock()
        serializer.set_connection_failure_handler(close_connection)
        event = json.dumps(
            {
                "type": "response.create",
                "event_id": "poisoned_response_tools",
                "response": {
                    "tools": [
                        _SESSION_FUNCTION_TOOL,
                        {"type": "mcp", "server_label": "undefined"},
                    ]
                },
            }
        )

        with patch.object(
            runtime.client_tool_projection,
            "restore",
            side_effect=RuntimeError("injected projection rollback failure"),
        ):
            frame = await serializer.deserialize(event)

        self.assertIsNone(frame)
        self.assertTrue(serializer.connection_closed)
        self.assertEqual(emitted[-1]["error"]["code"], "tool_preparation_state_inconsistent")
        self.assertEqual(emitted[-1]["error"]["event_id"], "poisoned_response_tools")
        close_connection.assert_awaited_once_with("realtime state transition failed")
        event_count = len(emitted)
        self.assertIsNone(await serializer.deserialize('{"type":"response.create"}'))
        self.assertEqual(len(emitted), event_count)

    async def test_live_session_preparation_rollback_failure_closes_physical_connection(self) -> None:
        _controller, serializer, _gate, runtime, _llm, emitted = await self._function_fixture(deferred=False)
        serializer.bind_context(
            LLMContext([{"role": "system", "content": ""}]),
            instructions_renderer=lambda instructions: [{"role": "system", "content": instructions}],
        )
        close_connection = AsyncMock()
        serializer.set_connection_failure_handler(close_connection)
        event = json.dumps(
            {
                "type": "session.update",
                "event_id": "poisoned_session_tools",
                "session": {
                    "tools": [
                        _SESSION_FUNCTION_TOOL,
                        {"type": "mcp", "server_label": "undefined"},
                    ]
                },
            }
        )

        with patch.object(
            runtime.client_tool_projection,
            "restore",
            side_effect=RuntimeError("injected projection rollback failure"),
        ):
            frame = await serializer.deserialize(event)

        self.assertIsNone(frame)
        self.assertTrue(serializer.connection_closed)
        self.assertEqual(emitted[-1]["error"]["code"], "tool_preparation_state_inconsistent")
        self.assertEqual(emitted[-1]["error"]["event_id"], "poisoned_session_tools")
        close_connection.assert_awaited_once_with("realtime state transition failed")

    async def test_nested_binding_rollback_failure_poison_is_not_downgraded(self) -> None:
        controller, _serializer, _gate, runtime, llm, _emitted = await self._function_fixture(deferred=False)
        worker = _FakeMCPWorker(_advertised_tools())

        def reject_mcp_handler(name: str | None, handler: Any, **options: Any) -> None:  # noqa: ARG001
            if name is None:
                raise AssertionError("The client projection handler is already installed")
            raise RuntimeError("injected LLM registration failure")

        with (
            _use_mcp_worker(worker),
            patch.object(llm, "register_function", side_effect=reject_mcp_handler),
            patch.object(
                controller,
                "unregister_mcp_tool_binding",
                side_effect=RuntimeError("injected controller rollback failure"),
            ),
            self.assertRaises(MCPToolPreparationPoisonedError),
        ):
            await runtime.prepare_tools([_mcp_tool_definition()], "auto")

        self.assertEqual(runtime._pending_response_preparations, {})
        self.assertEqual(runtime._servers, {})
        self.assertTrue(worker.closed)


class LiveSessionTransactionTests(unittest.IsolatedAsyncioTestCase):
    async def _fixture(
        self,
    ) -> tuple[
        RealtimeSessionController,
        RealtimeFrameSerializer,
        RealtimeManualResponseGate,
        LLMContext,
        RealtimeMCPRuntime,
        list[dict[str, Any]],
    ]:
        controller = RealtimeSessionController(
            model=MODEL,
            voice=VOICE,
            runtime_config={"prompt_content": "old instructions", "tool_choice": "auto"},
            instructions="old instructions",
            server_tools=["lookup"],
            trusted_tool_schemas=[_SESSION_FUNCTION_TOOL],
            capabilities=RealtimeSessionCapabilities(
                voices=frozenset({VOICE}),
                input_formats=frozenset(
                    {
                        AudioFormatCapability("audio/pcm", 24_000),
                        AudioFormatCapability("audio/pcmu"),
                    }
                ),
                trusted_function_tools=frozenset({"lookup"}),
            ),
        )
        emitted: list[dict[str, Any]] = []

        async def emit_batch(events: list[dict[str, Any]]) -> None:
            emitted.extend(copy.deepcopy(events))

        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=emit_batch)
        await runtime.prepare_session(_FakeLLM())  # type: ignore[arg-type]
        gate = RealtimeManualResponseGate(controller=controller)
        serializer = _serializer_for_mcp(controller, runtime, emit_batch)
        serializer.set_response_gate(gate)
        context = LLMContext([{"role": "system", "content": "old instructions"}])
        serializer.bind_context(
            context,
            instructions_renderer=lambda instructions: [{"role": "system", "content": instructions}],
        )
        emitted.clear()
        return controller, serializer, gate, context, runtime, emitted

    @staticmethod
    def _fail_once_after(original: Any) -> Any:
        failed = False

        def fail(*args: Any, **kwargs: Any) -> Any:
            nonlocal failed
            result = original(*args, **kwargs)
            if not failed:
                failed = True
                raise RuntimeError("injected session transaction failure")
            return result

        return fail

    async def test_each_commit_boundary_rolls_back_every_session_owner(self) -> None:
        update = json.dumps(
            {
                "type": "session.update",
                "event_id": "transaction_failure",
                "session": {
                    "instructions": "new instructions",
                    "tool_choice": "required",
                    "audio": {"input": {"format": {"type": "audio/pcmu"}}},
                },
            }
        )
        for boundary in (
            "instructions",
            "context_tools",
            "context_tool_choice",
            "session",
            "runtime_projection",
            "audio",
            "mcp_commit",
        ):
            with self.subTest(boundary=boundary):
                controller, serializer, gate, context, runtime, emitted = await self._fixture()
                controller_state = controller.snapshot_session_runtime_state()
                messages = tuple(context.get_messages())
                tools = context.tools
                tool_choice = context.tool_choice
                audio_state = (
                    serializer._client_in_format,
                    serializer._client_in_rate,
                    serializer._client_out_format,
                    serializer._client_out_rate,
                    serializer._resampler,
                    serializer._output_audio_generation,
                )
                definitions = copy.deepcopy(runtime._session_definitions)

                target, attribute = {
                    "instructions": (gate, "commit_session_instructions"),
                    "context_tools": (context, "set_tools"),
                    "context_tool_choice": (context, "set_tool_choice"),
                    "session": (controller, "apply_session_update"),
                    "runtime_projection": (controller, "bind_session_runtime_projection"),
                    "audio": (serializer._resampler, "for_format_transition"),
                    "mcp_commit": (runtime, "commit_session_update"),
                }[boundary]
                original = getattr(target, attribute)
                if boundary == "mcp_commit":

                    def side_effect(*_args: Any, **_kwargs: Any) -> None:
                        raise RuntimeError("injected MCP commit failure")

                else:
                    side_effect = self._fail_once_after(original)
                with patch.object(target, attribute, side_effect=side_effect):
                    frame = await serializer.deserialize(update)

                self.assertIsNone(frame)
                self.assertEqual(emitted[-1]["type"], "error")
                self.assertEqual(emitted[-1]["error"]["code"], "event_processing_failed")
                self.assertEqual(controller.snapshot_session_runtime_state(), controller_state)
                self.assertEqual(tuple(context.get_messages()), messages)
                self.assertTrue(
                    all(current is previous for current, previous in zip(context.get_messages(), messages, strict=True))
                )
                self.assertIs(context.tools, tools)
                self.assertIs(context.tool_choice, tool_choice)
                self.assertEqual(
                    (
                        serializer._client_in_format,
                        serializer._client_in_rate,
                        serializer._client_out_format,
                        serializer._client_out_rate,
                        serializer._resampler,
                        serializer._output_audio_generation,
                    ),
                    audio_state,
                )
                self.assertEqual(runtime._session_definitions, definitions)
                self.assertEqual(runtime._pending_session_updates, {})
                self.assertFalse(serializer.connection_closed)

                emitted.clear()
                await serializer.deserialize(json.dumps({"type": "session.update", "session": {"type": "realtime"}}))
                self.assertEqual(emitted[-1]["type"], "session.updated")
                runtime.shutdown()

    async def test_mcp_commit_copy_failure_preserves_pending_update_for_rollback(self) -> None:
        _controller, _serializer, _gate, _context, runtime, _emitted = await self._fixture()
        prepared = await runtime.prepare_session_update([_SESSION_FUNCTION_TOOL], "auto")
        update_id = prepared.session_update_id
        self.assertIsNotNone(update_id)
        definitions = copy.deepcopy(runtime._session_definitions)

        with (
            patch("realtime.mcp.copy.deepcopy", side_effect=RuntimeError("injected copy failure")),
            self.assertRaisesRegex(RuntimeError, "injected copy failure"),
        ):
            runtime.commit_session_update(prepared)

        self.assertEqual(runtime._session_definitions, definitions)
        self.assertIn(update_id, runtime._pending_session_updates)
        runtime.rollback_session_update(prepared)
        self.assertNotIn(update_id, runtime._pending_session_updates)
        runtime.shutdown()

    async def test_rollback_failure_retires_connection(self) -> None:
        _controller, serializer, _gate, context, _runtime, emitted = await self._fixture()
        close_connection = AsyncMock()
        serializer.set_connection_failure_handler(close_connection)
        original = context.set_tools

        def fail_after_mutation(value: Any) -> None:
            original(value)
            raise RuntimeError("persistent context setter failure")

        with patch.object(context, "set_tools", side_effect=fail_after_mutation):
            await serializer.deserialize(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {"tool_choice": "required"},
                    }
                )
            )

        self.assertTrue(serializer.connection_closed)
        self.assertEqual(emitted[-1]["error"]["code"], "event_processing_failed")
        close_connection.assert_awaited_once_with("realtime state transition failed")
        event_count = len(emitted)
        await serializer.deserialize(json.dumps({"type": "session.update", "session": {"type": "realtime"}}))
        self.assertEqual(len(emitted), event_count)


class MCPCancellationLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_native_result_metadata_is_also_removed_from_model_error(self) -> None:
        controller = _controller_with_mcp_call()

        async def emit_batch(events: list[dict[str, Any]]) -> None:  # noqa: ARG001
            return None

        class Worker:
            dead = False

            async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:  # noqa: ARG002
                return CallToolResult.model_validate(
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": "fixture failure",
                                "_meta": {"part_secret": "private"},
                            }
                        ],
                        "structuredContent": {
                            "visible": {"answer": 42, "_meta": {"nested_secret": "private"}},
                            "_meta": {"structured_secret": "private"},
                        },
                        "_meta": {"top_secret": "private"},
                        "isError": True,
                    }
                )

        async def result_callback(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
            return None

        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=emit_batch)
        runtime._bindings[PIPELINE_NAME] = MCPToolBinding(
            pipeline_name=PIPELINE_NAME,
            server_label=SERVER_LABEL,
            name=TOOL_NAME,
            description="",
            input_schema={"type": "object"},
            annotations=None,
            require_approval=False,
        )
        runtime._servers[SERVER_LABEL] = SimpleNamespace(worker=Worker())
        runtime.notify_call_announced(CALL_ID)

        result = await runtime._execute_tool_call(_call_params(result_callback))

        self.assertIn("_meta", result.error["message"])
        self.assertIn("nested_secret", result.error["message"])
        self.assertNotIn("_meta", result["error"]["message"])
        self.assertNotIn("nested_secret", result["error"]["message"])
        self.assertIn("fixture failure", result["error"]["message"])

    async def test_cancel_allows_exactly_one_scheduled_pipecat_context_callback(self) -> None:
        controller = _controller_with_mcp_call()
        captured_callback: Any = None
        emitted: list[dict[str, Any]] = []

        async def emit_batch(events: list[dict[str, Any]]) -> None:
            emitted.extend(copy.deepcopy(events))

        async def result_callback(result: Any, *, properties: Any) -> None:  # noqa: ARG001
            nonlocal captured_callback
            captured_callback = properties.on_context_updated

        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=emit_batch)
        result = MCPExecutionResult(output='{"value":"done"}', error=None)
        with patch.object(runtime, "_execute_tool_call", AsyncMock(return_value=result)):
            await runtime.handle_tool_call(_call_params(result_callback))

        self.assertIsNotNone(captured_callback)
        self.assertTrue(runtime.cancel_call_events(CALL_ID))
        await captured_callback()

        self.assertFalse(runtime._closed)
        self.assertFalse(runtime._fatal_error_emitted)
        self.assertEqual(emitted, [])
        with self.assertRaisesRegex(RuntimeError, "cannot continue safely"):
            await captured_callback()
        self.assertTrue(runtime._closed)
        self.assertTrue(runtime._fatal_error_emitted)
        self.assertIn("mcp_lifecycle_error", [event.get("error", {}).get("code") for event in emitted])

    async def test_uncorrelated_missing_context_result_remains_fatal(self) -> None:
        controller = _controller_with_mcp_call()
        emitted: list[dict[str, Any]] = []
        captured_callback: Any = None

        async def emit_batch(events: list[dict[str, Any]]) -> None:
            emitted.extend(copy.deepcopy(events))

        async def result_callback(result: Any, *, properties: Any) -> None:  # noqa: ARG001
            nonlocal captured_callback
            captured_callback = properties.on_context_updated

        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=emit_batch)
        with patch.object(
            runtime,
            "_execute_tool_call",
            AsyncMock(return_value=MCPExecutionResult(output='{"value":"done"}', error=None)),
        ):
            await runtime.handle_tool_call(_call_params(result_callback))
        runtime._pending_results.pop(CALL_ID)

        with self.assertRaisesRegex(RuntimeError, "cannot continue safely"):
            await captured_callback()

        self.assertTrue(runtime._closed)
        self.assertTrue(runtime._fatal_error_emitted)
        self.assertIn("mcp_lifecycle_error", [event.get("error", {}).get("code") for event in emitted])

    async def test_cancelled_context_callback_allowances_are_bounded(self) -> None:
        controller = _controller_with_mcp_call()

        async def emit_batch(events: list[dict[str, Any]]) -> None:  # noqa: ARG001
            return None

        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=emit_batch)
        for index in range(600):
            runtime._remember_cancelled_context_callback(f"call_{index}")

        self.assertEqual(len(runtime._cancelled_context_callbacks), 512)
        self.assertNotIn("call_0", runtime._cancelled_context_callbacks)
        self.assertIn("call_599", runtime._cancelled_context_callbacks)

    async def test_handler_cancel_during_approval_publication_is_ordered_and_idempotent(self) -> None:
        controller = _controller_with_mcp_call()
        emitted: list[dict[str, Any]] = []
        approval_send_entered = asyncio.Event()
        release_approval_send = asyncio.Event()

        async def emit_batch(events: list[dict[str, Any]]) -> None:
            if any(
                event.get("type") == "conversation.item.added"
                and event.get("item", {}).get("type") == "mcp_approval_response"
                for event in events
            ):
                approval_send_entered.set()
                await release_approval_send.wait()
            emitted.extend(copy.deepcopy(events))

        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=emit_batch)
        request_id, future = _install_pending_approval(runtime, controller)
        serializer = _serializer_for_mcp(controller, runtime, emit_batch)
        approval_item_id = "item_client_approval_1"

        deserialize = asyncio.create_task(
            serializer.deserialize(_approval_response_event(request_id, item_id=approval_item_id, approve=True))
        )
        await asyncio.wait_for(approval_send_entered.wait(), timeout=1)

        # Pipecat cancels the function handler before its cancellation frame
        # reaches the output-side lifecycle observer.
        runtime._mark_approval_waiter_stopped(request_id, cancelled=True)
        runtime._approvals.pop(request_id).cancel()
        runtime._approval_calls.pop(request_id)
        self.assertEqual(runtime.cancel_call_events(CALL_ID), [])

        release_approval_send.set()
        await asyncio.wait_for(deserialize, timeout=1)

        event_types = [event["type"] for event in emitted]
        approval_done_index = next(
            index
            for index, event in enumerate(emitted)
            if event["type"] == "conversation.item.done" and event["item"]["id"] == approval_item_id
        )
        failure_index = event_types.index("response.mcp_call.failed")
        self.assertLess(approval_done_index, failure_index)
        self.assertNotIn("error", event_types)
        self.assertTrue(future.cancelled())
        self.assertTrue(controller.tool_call(CALL_ID).completed)
        self.assertNotIn(request_id, runtime._approval_response_claims)
        self.assertEqual(runtime.cancel_call_events(CALL_ID), [])

    async def test_approval_decision_is_not_released_before_item_publication(self) -> None:
        controller = _controller_with_mcp_call()
        approval_send_entered = asyncio.Event()
        release_approval_send = asyncio.Event()

        async def emit_batch(events: list[dict[str, Any]]) -> None:
            if any(
                event.get("type") == "conversation.item.added"
                and event.get("item", {}).get("type") == "mcp_approval_response"
                for event in events
            ):
                approval_send_entered.set()
                await release_approval_send.wait()

        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=emit_batch)
        request_id, future = _install_pending_approval(runtime, controller)
        serializer = _serializer_for_mcp(controller, runtime, emit_batch)
        deserialize = asyncio.create_task(
            serializer.deserialize(_approval_response_event(request_id, item_id="item_approved", approve=True))
        )

        await asyncio.wait_for(approval_send_entered.wait(), timeout=1)
        self.assertFalse(future.done())
        release_approval_send.set()
        await asyncio.wait_for(deserialize, timeout=1)

        self.assertTrue(future.result().approve)
        self.assertNotIn(request_id, runtime._approval_response_claims)

    async def test_cancel_before_approval_claim_rejects_without_journaling(self) -> None:
        controller = _controller_with_mcp_call()
        emitted: list[dict[str, Any]] = []

        async def emit_batch(events: list[dict[str, Any]]) -> None:
            emitted.extend(copy.deepcopy(events))

        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=emit_batch)
        request_id, _ = _install_pending_approval(runtime, controller)
        runtime.cancel_call_events(CALL_ID)
        serializer = _serializer_for_mcp(controller, runtime, emit_batch)
        approval_item_id = "item_late_approval"

        await serializer.deserialize(
            _approval_response_event(
                request_id,
                item_id=approval_item_id,
                approve=True,
                event_id="evt_late_approval",
            )
        )

        self.assertNotIn(approval_item_id, controller.conversation.ordered_item_ids())
        self.assertEqual(emitted[-1]["type"], "error")
        self.assertEqual(emitted[-1]["error"]["code"], "mcp_approval_not_found")
        self.assertEqual(emitted[-1]["error"]["event_id"], "evt_late_approval")

    async def test_approval_publication_failure_cannot_strand_waiter_or_call(self) -> None:
        controller = _controller_with_mcp_call()
        emitted: list[dict[str, Any]] = []
        fail_next_send = True

        async def emit_batch(events: list[dict[str, Any]]) -> None:
            nonlocal fail_next_send
            if fail_next_send:
                fail_next_send = False
                raise ConnectionError("fixture wire failure")
            emitted.extend(copy.deepcopy(events))

        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=emit_batch)
        request_id, future = _install_pending_approval(runtime, controller)
        serializer = _serializer_for_mcp(controller, runtime, emit_batch)

        await serializer.deserialize(
            _approval_response_event(request_id, item_id="item_failed_publication", approve=True)
        )

        self.assertTrue(future.cancelled())
        self.assertTrue(controller.tool_call(CALL_ID).completed)
        self.assertNotIn(request_id, runtime._approval_response_claims)
        self.assertIn("response.mcp_call.failed", [event["type"] for event in emitted])
        self.assertEqual(emitted[-1]["type"], "error")
        self.assertEqual(emitted[-1]["error"]["code"], "event_processing_failed")

    async def test_closed_wire_cannot_release_an_unpublished_approval(self) -> None:
        controller = _controller_with_mcp_call()
        serializer = RealtimeFrameSerializer(controller=controller)

        async def emit_batch(events: list[dict[str, Any]]) -> None:  # noqa: ARG001
            serializer.notify_connection_closed()

        runtime = RealtimeMCPRuntime(controller=controller, emit_batch=emit_batch)
        request_id, future = _install_pending_approval(runtime, controller)
        serializer = _serializer_for_mcp(controller, runtime, emit_batch)

        await serializer.deserialize(
            _approval_response_event(request_id, item_id="item_closed_publication", approve=True)
        )

        self.assertTrue(future.cancelled())
        self.assertTrue(controller.tool_call(CALL_ID).completed)
        self.assertNotIn(request_id, runtime._approval_response_claims)
