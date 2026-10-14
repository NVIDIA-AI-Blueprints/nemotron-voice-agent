# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from pipecat.frames.frames import (
    FunctionCallFromLLM,
    FunctionCallResultProperties,
    FunctionCallsStartedFrame,
    LLMFullResponseEndFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from realtime_helpers import FakeWebSocket

from realtime.client_tools import ClientToolBridge, normalize_client_tools, to_openai_tools
from realtime.conversation import ConversationState
from realtime.gateway import handle_realtime_websocket
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

    def test_openai_projection_keeps_full_schema(self) -> None:
        projected = to_openai_tools([WEATHER_TOOL])
        self.assertEqual(projected[0]["function"]["parameters"], WEATHER_TOOL["parameters"])

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


class ClientToolFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            self.emitted.append(event)

        self.emit = emit
        self.conversation = ConversationState(assistant_has_responded=True)
        self.bridge = ClientToolBridge(tools=[WEATHER_TOOL], emit=emit)
        self.observer = RealtimeLifecycleObserver(emit=emit, conversation=self.conversation, client_tools=self.bridge)
        self.serializer = RealtimeFrameSerializer(conversation=self.conversation)
        self.serializer.set_emit(emit)
        self.serializer.set_client_tool_bridge(self.bridge)

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

    def _types(self) -> list[str]:
        return [event["type"] for event in self.emitted]

    async def test_tool_only_response_emits_call_and_completes(self) -> None:
        call = FunctionCallFromLLM(
            function_name="get_weather", tool_call_id="call_1", arguments={"city": "Paris"}, context=None
        )
        await self._push(FunctionCallsStartedFrame(function_calls=[call]))
        await self._push(LLMFullResponseEndFrame())

        self.assertEqual(
            self._types(),
            [
                "response.created",
                "response.output_item.added",
                "conversation.item.created",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.done",
            ],
        )
        args_done = self.emitted[3]
        self.assertEqual(args_done["call_id"], "call_1")
        self.assertEqual(args_done["name"], "get_weather")
        self.assertEqual(json.loads(args_done["arguments"]), {"city": "Paris"})
        response = self.emitted[-1]["response"]
        self.assertEqual(response["status"], "completed")
        self.assertEqual([item["type"] for item in response["output"]], ["function_call"])
        self.assertNotIn("nvidia.tool.started", self._types())

    async def test_output_is_acknowledged_after_context_update(self) -> None:
        call = FunctionCallFromLLM(function_name="get_weather", tool_call_id="call_1", arguments={}, context=None)
        await self._push(FunctionCallsStartedFrame(function_calls=[call]))
        self.emitted.clear()

        results: list[tuple[Any, FunctionCallResultProperties]] = []

        async def result_callback(result: Any, *, properties: FunctionCallResultProperties) -> None:
            results.append((result, properties))

        params = MagicMock(tool_call_id="call_1", result_callback=result_callback)
        handler = asyncio.create_task(self.bridge.handle_call(params))
        await asyncio.sleep(0)
        frame = await self.serializer.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "function_call_output", "call_id": "call_1", "output": '{"temp": 21}'},
                }
            )
        )
        await handler

        self.assertIsNone(frame)
        self.assertEqual(results[0][0], '{"temp": 21}')
        properties = results[0][1]
        self.assertFalse(properties.run_llm)
        self.assertEqual(self.emitted, [])
        await properties.on_context_updated()
        self.assertEqual(self.emitted[0]["type"], "conversation.item.created")
        self.assertEqual(self.emitted[0]["item"]["type"], "function_call_output")
        self.assertEqual(self.emitted[0]["item"]["call_id"], "call_1")

    async def test_output_for_settled_call_is_still_acknowledged(self) -> None:
        self.bridge.expect("call_1")
        params = MagicMock(tool_call_id="call_1")
        handler = asyncio.create_task(self.bridge.handle_call(params))
        await asyncio.sleep(0)
        handler.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await handler

        await self.serializer.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "function_call_output", "call_id": "call_1", "output": "late"},
                }
            )
        )
        self.assertEqual(self.emitted[-1]["type"], "conversation.item.created")

    async def test_unknown_call_id_is_rejected(self) -> None:
        await self.serializer.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "function_call_output", "call_id": "call_missing", "output": "{}"},
                }
            )
        )
        self.assertEqual(self.emitted[-1]["type"], "error")
        self.assertEqual(self.emitted[-1]["error"]["code"], "invalid_item")


if __name__ == "__main__":
    unittest.main()
