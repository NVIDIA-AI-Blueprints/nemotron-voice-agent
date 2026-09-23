# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import asyncio
import json

from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.llm import NvidiaLLMSettings

from examples.shared.nvidia_streaming_llm import (
    NvidiaStreamingLLMService,
    StreamingLLMInputFrame,
    StreamingLLMUserAggregator,
)
from examples.shared.vllm_streaming_input.vllm_patch import patch_scheduler_source

URL = "ws://nvidia-llm-vllm:8000/v1/streaming-session"


class _FakeWebSocket:
    """Records client events and answers each commit with one delta and done."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self._inbox: asyncio.Queue[str] = asyncio.Queue()

    async def send(self, raw: str) -> None:
        event = json.loads(raw)
        self.sent.append(event)
        if event["type"] == "input_text_buffer.commit":
            chunk = {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "delta": {"content": "Hello!"}, "finish_reason": None}],
            }
            for reply in (
                {"type": "response.delta", "response_id": "stale", "chunk": chunk},
                {"type": "response.delta", "response_id": event["response_id"], "chunk": chunk},
                {"type": "response.done", "response_id": event["response_id"]},
            ):
                self._inbox.put_nowait(json.dumps(reply))

    async def recv(self) -> str:
        return await self._inbox.get()

    async def close(self) -> None:
        pass


def _service() -> tuple[NvidiaStreamingLLMService, _FakeWebSocket]:
    service = NvidiaStreamingLLMService(api_key="test", base_url=URL, settings=NvidiaLLMSettings(model="m"))
    websocket = _FakeWebSocket()
    service._websocket = websocket
    return service, websocket


def _events(websocket: _FakeWebSocket) -> list[tuple[str, str]]:
    return [(event["type"], event.get("text", "")) for event in websocket.sent]


def test_service_serves_chat_completions_from_the_same_server() -> None:
    service, _ = _service()

    assert str(service._client.base_url) == "http://nvidia-llm-vllm:8000/v1/"


def test_service_appends_only_new_text_and_clears_on_asr_revision() -> None:
    service, websocket = _service()
    request = {"model": "m", "messages": [{"role": "system", "content": "Be brief."}]}

    async def run() -> None:
        for text in ("", "Hi", "Hi there", "Hi there", "Hey there"):
            await service._sync_session(request, text)

    asyncio.run(run())

    assert _events(websocket) == [
        ("session.update", ""),
        ("input_text_buffer.append", "Hi"),
        ("input_text_buffer.append", " there"),
        ("input_text_buffer.clear", ""),
        ("input_text_buffer.append", "Hey there"),
    ]


def test_service_commits_the_user_turn_and_streams_its_chunks() -> None:
    service, websocket = _service()
    context = LLMContext([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}])

    async def run() -> list[str]:
        service._reset_response_state()
        stream = await service.get_chat_completions(context)
        return [chunk.choices[0].delta.content async for chunk in stream]

    assert asyncio.run(run()) == ["Hello!"]
    update, append, commit = websocket.sent
    assert update["request"]["messages"] == [{"role": "system", "content": "Be brief."}]
    assert append == {"type": "input_text_buffer.append", "text": "Hi"}
    assert commit["type"] == "input_text_buffer.commit"


def test_user_aggregator_streams_the_turn_so_far_before_aggregating() -> None:
    aggregator = StreamingLLMUserAggregator(LLMContext())
    pushed: list[Frame] = []

    async def record_push(frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        pushed.append(frame)

    async def aggregate(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)

    aggregator.push_frame = record_push
    original = LLMUserAggregator.process_frame
    LLMUserAggregator.process_frame = aggregate
    try:
        for frame in (
            InterimTranscriptionFrame(text="Hi", user_id="", timestamp=""),
            TranscriptionFrame(text="Hi there.", user_id="", timestamp=""),
            InterimTranscriptionFrame(text="How", user_id="", timestamp=""),
        ):
            asyncio.run(aggregator.process_frame(frame, FrameDirection.DOWNSTREAM))
    finally:
        LLMUserAggregator.process_frame = original

    assert [frame.text for frame in pushed if isinstance(frame, StreamingLLMInputFrame)] == [
        "Hi",
        "Hi there.",
        "Hi there. How",
    ]


def test_patch_scheduler_source_is_idempotent() -> None:
    source = (
        "        session.sampling_params = update.sampling_params\n"
        "        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:\n"
    )
    patched, changed = patch_scheduler_source(source)
    again, changed_again = patch_scheduler_source(patched)

    assert changed is True
    assert "        session.max_tokens = update.max_tokens\n" in patched
    assert changed_again is False
    assert again == patched
