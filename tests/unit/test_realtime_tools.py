# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

from __future__ import annotations

import json
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from pipecat.frames.frames import (
    FunctionCallFromLLM,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.llm import NvidiaLLMService, NvidiaLLMSettings
from realtime_helpers import FakeWebSocket

from realtime.client_tools import RealtimeClientTools, RealtimeToolResultPolicy, normalize_client_tools
from realtime.conversation import ConversationState
from realtime.gateway import handle_realtime_websocket
from realtime.llm import fold_async_tool_messages
from realtime.observer import RealtimeLifecycleObserver
from realtime.serializer import RealtimeFrameSerializer
from realtime.session import DEFAULT_REALTIME_MODEL, map_session_update_to_flat_config

WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Weather",
    "parameters": {
        "type": "object",
        "properties": {"city": {"$ref": "#/$defs/City"}},
        "required": ["city"],
        "$defs": {"City": {"type": "string"}},
    },
}


def _session_update(session: dict[str, Any]) -> str:
    return json.dumps({"type": "session.update", "event_id": "e1", "session": session})


class NormalizeClientToolsTests(unittest.TestCase):
    def test_session_tools_map_to_client_tools(self) -> None:
        flat = map_session_update_to_flat_config({"tools": [WEATHER_TOOL]})
        self.assertEqual(flat["client_tools"], [WEATHER_TOOL])

    def test_omitted_tools_leave_client_tools_unset(self) -> None:
        self.assertNotIn("client_tools", map_session_update_to_flat_config({}))
        self.assertEqual(map_session_update_to_flat_config({"tools": []})["client_tools"], [])

    def test_chat_completions_shape_is_flattened(self) -> None:
        tools = normalize_client_tools([{"type": "function", "function": {"name": "ping"}}])
        self.assertEqual(
            tools,
            [{"type": "function", "name": "ping", "parameters": {"type": "object", "properties": {}}}],
        )

    def test_provider_projection_keeps_full_schema(self) -> None:
        client_tools = RealtimeClientTools([WEATHER_TOOL], scope_id="sess_test", output_timeout_secs=5)
        llm = NvidiaLLMService(api_key="test", settings=NvidiaLLMSettings(model="test-model"))
        schema = client_tools.configure(llm, None, ())
        projected = [tool["function"] for tools in schema.custom_tools.values() for tool in tools]
        self.assertEqual(projected[0]["name"], "get_weather")
        self.assertEqual(projected[0]["parameters"], WEATHER_TOOL["parameters"])

    def test_invalid_tools_are_rejected(self) -> None:
        for bad in (
            {"tools": "x"},
            {"tools": [{"type": "mcp", "name": "a"}]},
            {"tools": [{"type": "function", "name": "bad name"}]},
            {"tools": [{"type": "function", "name": "a"}, {"type": "function", "name": "a"}]},
            {"tools": [{"type": "function", "name": "a", "parameters": {"type": "string"}}]},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                map_session_update_to_flat_config(bad)


class GatewayClientToolsTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, session: dict[str, Any], **kwargs: Any) -> tuple[FakeWebSocket, dict]:
        captured: dict = {}

        async def start_bot(ws, config, session_view):  # noqa: ARG001
            captured["config"] = config
            captured["session"] = session_view

        ws = FakeWebSocket([_session_update(session)])
        ws.query_params = {"model": "nvidia/nemotron-voice-agent"}
        with patch("realtime.gateway.resolve_realtime_tts_voice", return_value=None):
            await handle_realtime_websocket(
                ws,
                sanitize_session_config=lambda data, **_: dict(data),
                start_bot=start_bot,
                resolve_server_tools=lambda _config: ["set_memory"],
                **kwargs,
            )
        return ws, captured

    async def test_client_tools_pass_to_pipeline_and_echo(self) -> None:
        ws, captured = await self._run({"tools": [WEATHER_TOOL]}, supports_client_tools=lambda _config: True)
        self.assertEqual(captured["config"]["client_tools"], [WEATHER_TOOL])
        self.assertEqual(captured["session"]["tools"], [WEATHER_TOOL])
        self.assertEqual(captured["session"]["model"], "nvidia/nemotron-voice-agent")
        self.assertEqual(ws.sent[0]["session"]["model"], "nvidia/nemotron-voice-agent")

    async def test_default_model_is_advertised_without_a_request(self) -> None:
        ws = FakeWebSocket([])
        await handle_realtime_websocket(ws, sanitize_session_config=lambda data, **_: dict(data))
        self.assertEqual(ws.sent[0]["session"]["model"], DEFAULT_REALTIME_MODEL)

    async def test_pipeline_without_capability_rejects_client_tools(self) -> None:
        ws, captured = await self._run({"tools": [WEATHER_TOOL]})
        self.assertNotIn("config", captured)
        self.assertEqual(ws.sent[-1]["type"], "error")
        self.assertIn("does not run client tools", ws.sent[-1]["error"]["message"])

    async def test_client_tool_names_must_not_shadow_server_tools(self) -> None:
        tool = {**WEATHER_TOOL, "name": "set_memory"}
        ws, captured = await self._run({"tools": [tool]}, supports_client_tools=lambda _config: True)
        self.assertNotIn("config", captured)
        self.assertIn("conflict with server tools", ws.sent[-1]["error"]["message"])

    async def test_forced_tool_choice_is_rejected_with_client_tools(self) -> None:
        for tool_choice in ("required", {"type": "function", "name": "get_weather"}):
            with self.subTest(tool_choice=tool_choice):
                ws, captured = await self._run(
                    {"tools": [WEATHER_TOOL], "tool_choice": tool_choice}, supports_client_tools=lambda _config: True
                )
                self.assertNotIn("config", captured)
                self.assertIn("tool_choice", ws.sent[-1]["error"]["message"])


class ClientToolFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            self.emitted.append(event)

        self.conversation = ConversationState(assistant_has_responded=True)
        self.client_tools = RealtimeClientTools([WEATHER_TOOL], scope_id="sess_test", output_timeout_secs=5)
        self.llm = NvidiaLLMService(api_key="test", settings=NvidiaLLMSettings(model="test-model"))
        self.client_tools.configure(self.llm, None, ())
        self.observer = RealtimeLifecycleObserver(
            emit=emit, conversation=self.conversation, client_tools=self.client_tools
        )
        self.serializer = RealtimeFrameSerializer(conversation=self.conversation)
        self.serializer.set_emit(emit)
        self.serializer.set_client_tools(self.client_tools)

    async def _push(self, frame: Any) -> None:
        await self.observer.on_push_frame(
            FramePushed(
                source=MagicMock(),
                destination=MagicMock(),
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )

    async def _client_call_response(self, call_id: str = "call_1") -> None:
        call = FunctionCallFromLLM(
            function_name="get_weather", tool_call_id=call_id, arguments={"city": "Paris"}, context=None
        )
        await self._push(LLMFullResponseStartFrame())
        await self._push(FunctionCallsStartedFrame(function_calls=[call]))
        await self._push(LLMFullResponseEndFrame())

    async def _send_output(self, call_id: str, output: Any) -> None:
        item = {"type": "function_call_output", "call_id": call_id, "output": output}
        await self.serializer.deserialize(json.dumps({"type": "conversation.item.create", "item": item}))

    def _types(self) -> list[str]:
        return [event["type"] for event in self.emitted]

    async def test_tool_only_response_emits_call_and_completes(self) -> None:
        await self._client_call_response()

        self.assertEqual(
            self._types(),
            [
                "response.created",
                "response.output_item.added",
                "conversation.item.created",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.done",
            ],
        )
        args_done = self.emitted[4]
        self.assertEqual(args_done["call_id"], "call_1")
        self.assertEqual(args_done["name"], "get_weather")
        self.assertEqual(json.loads(args_done["arguments"]), {"city": "Paris"})
        response = self.emitted[-1]["response"]
        self.assertEqual(response["status"], "completed")
        self.assertEqual([item["type"] for item in response["output"]], ["function_call"])
        self.assertNotIn("nvidia.tool.started", self._types())

    async def test_output_is_acknowledged_and_response_create_waits_for_it(self) -> None:
        await self._client_call_response()
        self.emitted.clear()

        rejected = await self.serializer.deserialize(json.dumps({"type": "response.create"}))
        self.assertIsNone(rejected)
        self.assertEqual(self.emitted[-1]["error"]["code"], "tool_output_pending")

        await self._send_output("call_1", '{"temp": 21}')
        ack = self.emitted[-1]
        self.assertEqual(ack["type"], "conversation.item.created")
        self.assertEqual(ack["item"]["type"], "function_call_output")
        self.assertEqual(ack["item"]["output"], '{"temp": 21}')

    async def test_invalid_outputs_are_rejected(self) -> None:
        await self._client_call_response()
        for call_id, output, code in (
            ("call_missing", "{}", "invalid_item"),
            ("call_1", {"temp": 21}, "invalid_item"),
        ):
            with self.subTest(call_id=call_id, output=output):
                await self._send_output(call_id, output)
                self.assertEqual(self.emitted[-1]["error"]["code"], code)

    async def test_duplicate_output_is_rejected(self) -> None:
        await self._client_call_response()
        await self._send_output("call_1", "first")
        await self._send_output("call_1", "second")
        self.assertEqual(self.emitted[-1]["error"]["code"], "duplicate_tool_output")

    async def test_output_for_call_of_cancelled_response_is_rejected(self) -> None:
        call = FunctionCallFromLLM(function_name="get_weather", tool_call_id="call_1", arguments={}, context=None)
        await self._push(LLMFullResponseStartFrame())
        await self._push(FunctionCallsStartedFrame(function_calls=[call]))
        await self.serializer.deserialize(json.dumps({"type": "response.cancel"}))

        await self._send_output("call_1", "late")
        self.assertEqual(self.emitted[-1]["error"]["code"], "client_tool_call_cancelled")


class AsyncToolMessageFoldingTests(unittest.TestCase):
    def test_late_result_replaces_the_placeholder_and_developer_message_is_dropped(self) -> None:
        from pipecat.processors.aggregators import async_tool_messages

        placeholder = async_tool_messages.build_started_message("call_1")
        final = async_tool_messages.build_final_result_message("call_1", '{"temp": 21}')
        messages = [
            {"role": "system", "content": "prompt"},
            {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
            {**placeholder},
            {"role": "user", "content": "anything else?"},
            {**final},
        ]

        folded = fold_async_tool_messages(messages)

        self.assertEqual([message["role"] for message in folded], ["system", "assistant", "tool", "user"])
        self.assertEqual(folded[2]["content"], '{"temp": 21}')

    def test_messages_without_async_tools_are_unchanged(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        self.assertIs(fold_async_tool_messages(messages), messages)


class ToolResultPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_server_result_in_a_round_with_a_client_call_does_not_run_the_llm(self) -> None:
        client_tools = RealtimeClientTools([WEATHER_TOOL], scope_id="sess_test", output_timeout_secs=5)
        client_tools.configure(NvidiaLLMService(api_key="test", settings=NvidiaLLMSettings(model="m")), None, ())
        policy = RealtimeToolResultPolicy(client_tools)
        pushed: list[Any] = []

        async def capture(frame: Any, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
            pushed.append(frame)

        policy.push_frame = capture  # type: ignore[method-assign]
        calls = [
            FunctionCallFromLLM(function_name="get_weather", tool_call_id="client", arguments={}, context=None),
            FunctionCallFromLLM(function_name="calculate_bmi", tool_call_id="server", arguments={}, context=None),
        ]
        await policy.process_frame(FunctionCallsStartedFrame(function_calls=calls), FrameDirection.DOWNSTREAM)
        result = FunctionCallResultFrame(function_name="calculate_bmi", tool_call_id="server", arguments={}, result={})
        await policy.process_frame(result, FrameDirection.DOWNSTREAM)

        self.assertFalse(pushed[-1].properties.run_llm)


if __name__ == "__main__":
    unittest.main()
