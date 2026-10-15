# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Generic Assistant sessions over ``WS /v1/realtime``, end to end.

The real FastAPI app, Realtime gateway, Generic pipeline, Pipecat aggregators,
and output transport run in-process. Only the network-bound services are
replaced: ASR passes frames through, TTS voices speakable text as a short
silent clip, and the LLM replays scripted chat-completion chunks.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

import uvicorn
import websockets
from openai.types.chat import ChatCompletionChunk
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.nvidia.llm import NvidiaLLMService
from pipecat.utils.text.base_text_aggregator import AggregationType

import examples.generic.pipeline as generic_pipeline
import server

EVENT_TIMEOUT_SECS = 20.0


def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "scripted",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
    )


def text_turn(text: str) -> list[ChatCompletionChunk]:
    return [chunk({"role": "assistant", "content": text}), chunk({}, "stop")]


def tool_turn(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> list[ChatCompletionChunk]:
    tool_call = {
        "index": 0,
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    return [chunk({"role": "assistant", "tool_calls": [tool_call]}), chunk({}, "tool_calls")]


class PassthroughSTT(FrameProcessor):
    def __init__(self, **_kwargs: Any) -> None:
        super().__init__()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class SilentClipTTS(FrameProcessor):
    """Voice a reply as one short clip; like NVIDIA TTS, skip text with nothing to say."""

    def __init__(self, **_kwargs: Any) -> None:
        super().__init__()
        self._text = ""
        self._contexts = 0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMTextFrame):
            self._text += frame.text
            return
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMFullResponseEndFrame):
            text, self._text = self._text, ""
            if any(char.isalnum() for char in text):
                self._contexts += 1
                context_id = f"ctx-{self._contexts}"
                await self.push_frame(TTSStartedFrame(context_id=context_id))
                await self.push_frame(
                    TTSAudioRawFrame(audio=b"\x00\x00" * 2205, sample_rate=22050, num_channels=1, context_id=context_id)
                )
                await self.push_frame(
                    TTSTextFrame(text=text, aggregated_by=AggregationType.SENTENCE, context_id=context_id)
                )
                await self.push_frame(TTSStoppedFrame(context_id=context_id))
        await self.push_frame(frame, direction)


class ScriptedLLM(NvidiaLLMService):
    """Replay scripted completions instead of calling an endpoint."""

    script: list[Any] = []
    requests: list[list[Any]] = []

    async def get_chat_completions(self, context):
        ScriptedLLM.requests.append(list(context.get_messages()))
        turn = ScriptedLLM.script.pop(0) if ScriptedLLM.script else text_turn("Fallback reply.")
        if isinstance(turn, Exception):
            raise turn

        async def chunks():
            for item in turn:
                yield item

        return chunks()


class GenericRealtimeEndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        ScriptedLLM.script = []
        ScriptedLLM.requests = []
        patches = contextlib.ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(
            patch.dict(
                os.environ,
                {
                    "SERVICE_RECIPE": "cloud",
                    "NVIDIA_API_KEY": "test-key",
                    "USE_SILERO_VAD_TURN_DETECTION": "true",
                    # Realtime sessions must not greet even when the welcome message is on.
                    "ENABLE_WELCOME_MESSAGE": "true",
                },
            )
        )
        patches.enter_context(patch.object(generic_pipeline, "NvidiaForceEouSTTService", PassthroughSTT))
        patches.enter_context(patch.object(generic_pipeline, "NvidiaTTSService", SilentClipTTS))
        patches.enter_context(patch.object(generic_pipeline, "NvidiaLLMService", ScriptedLLM))
        patches.enter_context(
            patch.object(server, "_ensure_services_ready_for_connection", AsyncMock(return_value=None))
        )
        patches.enter_context(patch("realtime.gateway.resolve_realtime_tts_voice", return_value=None))

        config = uvicorn.Config(server.create_app(), host="127.0.0.1", port=0, log_level="warning")
        self.server = uvicorn.Server(config)
        self.server_task = asyncio.create_task(self.server.serve())
        for _ in range(200):
            if self.server.started:
                break
            await asyncio.sleep(0.05)
        self.assertTrue(self.server.started, "uvicorn did not start")
        self.port = self.server.servers[0].sockets[0].getsockname()[1]

    async def asyncTearDown(self) -> None:
        self.server.should_exit = True
        await asyncio.wait_for(self.server_task, timeout=EVENT_TIMEOUT_SECS)

    async def connect(self, session: dict[str, Any] | None = None):
        ws = await websockets.connect(f"ws://127.0.0.1:{self.port}/v1/realtime")
        self.addAsyncCleanup(ws.close)
        created = await self.events_until(ws, "session.created")
        self.assertEqual(len(created), 1)
        await self.send(ws, {"type": "session.update", "session": {"type": "realtime", **(session or {})}})
        events = await self.events_until(ws, "session.updated")
        self.assertNotIn("error", [event["type"] for event in events], events)
        return ws

    async def send(self, ws, event: dict[str, Any]) -> None:
        await ws.send(json.dumps(event))

    async def send_user_text(self, ws, text: str) -> None:
        item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
        await self.send(ws, {"type": "conversation.item.create", "item": item})

    async def events_until(self, ws, event_type: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        loop = asyncio.get_running_loop()
        deadline = loop.time() + EVENT_TIMEOUT_SECS
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                self.fail(f"no {event_type} within {EVENT_TIMEOUT_SECS}s; got {[e['type'] for e in events]}")
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining))
            events.append(event)
            if event["type"] == event_type:
                return events

    async def events_within(self, ws, seconds: float) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(seconds):
                while True:
                    events.append(json.loads(await ws.recv()))
        return events

    async def test_session_waits_for_the_client_and_answers_a_text_turn(self) -> None:
        ScriptedLLM.script = [text_turn("Hello! How can I help?")]
        ws = await self.connect()

        self.assertEqual(await self.events_within(ws, 1.0), [])
        self.assertEqual(ScriptedLLM.requests, [])

        await self.send_user_text(ws, "Hi")
        await self.send(ws, {"type": "response.create"})
        events = await self.events_until(ws, "response.done")

        types = [event["type"] for event in events]
        self.assertNotIn("error", types)
        self.assertEqual(types[:2], ["conversation.item.created", "response.created"])
        self.assertIn("response.output_audio.delta", types)
        response = events[-1]["response"]
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["output"][0]["content"][0]["transcript"], "Hello! How can I help?")

    async def test_reply_without_speech_completes_and_the_next_response_is_accepted(self) -> None:
        ScriptedLLM.script = [text_turn(" \n "), text_turn("Second reply.")]
        ws = await self.connect()
        await self.send_user_text(ws, "Hi")
        await self.send(ws, {"type": "response.create"})

        first = (await self.events_until(ws, "response.done"))[-1]["response"]
        self.assertEqual(first["status"], "completed")
        self.assertEqual(first["output"], [])

        await self.send(ws, {"type": "response.create"})
        events = await self.events_until(ws, "response.done")
        self.assertNotIn("error", [event["type"] for event in events])
        self.assertEqual(events[-1]["response"]["output"][0]["content"][0]["transcript"], "Second reply.")

    async def test_llm_failure_fails_the_response(self) -> None:
        ScriptedLLM.script = [RuntimeError("upstream returned 500")]
        ws = await self.connect()
        await self.send_user_text(ws, "Hi")
        await self.send(ws, {"type": "response.create"})

        events = await self.events_until(ws, "response.done")
        self.assertIn("error", [event["type"] for event in events])
        self.assertEqual(events[-1]["response"]["status"], "failed")

        await self.send(ws, {"type": "response.create"})
        retry = await self.events_until(ws, "response.done")
        self.assertEqual(retry[-1]["response"]["status"], "completed")

    async def test_idle_response_cancel_returns_an_error(self) -> None:
        ws = await self.connect()
        await self.send(ws, {"type": "response.cancel", "event_id": "evt_cancel"})

        error = (await self.events_until(ws, "error"))[-1]["error"]
        self.assertEqual(error["code"], "response_cancel_not_active")
        self.assertEqual(error["event_id"], "evt_cancel")

    async def test_catalog_tool_round_is_one_response(self) -> None:
        ScriptedLLM.script = [
            tool_turn("calculate_bmi", {"weight_kg": 70, "height_m": 1.75}),
            text_turn("Your BMI is about 22.9."),
        ]
        ws = await self.connect({"prompt": {"id": "generic_assistant"}})
        await self.send_user_text(ws, "I weigh 70 kilograms and I am 1.75 meters tall. What is my BMI?")
        await self.send(ws, {"type": "response.create"})

        events = await self.events_until(ws, "response.done")
        types = [event["type"] for event in events]
        self.assertNotIn("error", types)
        self.assertEqual(types.count("response.created"), 1)
        self.assertIn("nvidia.tool.completed", types)
        response = events[-1]["response"]
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["output"][0]["content"][0]["transcript"], "Your BMI is about 22.9.")
        self.assertEqual(len(ScriptedLLM.requests), 2)
        user_messages = [m for m in ScriptedLLM.requests[0] if isinstance(m, dict) and m.get("role") == "user"]
        self.assertIn("What is my BMI?", user_messages[-1]["content"])
        tool_messages = [m for m in ScriptedLLM.requests[1] if isinstance(m, dict) and m.get("role") == "tool"]
        self.assertIn("22.86", tool_messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
