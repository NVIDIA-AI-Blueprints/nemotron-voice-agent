# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import asyncio
import json
from copy import deepcopy

import pytest
from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.llm import NvidiaLLMSettings

import examples.shared.nvidia_streaming_llm as streaming_llm
from examples.shared.nvidia_streaming_llm import (
    NvidiaStreamingLLMService,
    StreamingLLMInputFrame,
    StreamingLLMUserAggregator,
)
from examples.shared.vllm_streaming_input import realtime

URL = "ws://nvidia-llm-vllm:8000/v1/realtime"
MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
CALL = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Pune"}'}}


def _reply(text: str = "", calls: tuple = (), usage: dict | None = None) -> list[dict]:
    """Return the chat completion chunks vLLM streams for one answer."""
    chunks = [{"choices": [{"index": 0, "delta": {"content": text}}]}] if text else []
    for index, (call_id, name, arguments) in enumerate(calls):
        start = {"index": index, "id": call_id, "type": "function", "function": {"name": name, "arguments": ""}}
        chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": [start]}}]})
        for part in (arguments[:5], arguments[5:]):
            chunks.append(
                {
                    "choices": [
                        {"index": 0, "delta": {"tool_calls": [{"index": index, "function": {"arguments": part}}]}}
                    ]
                }
            )
    finish_reason = "tool_calls" if calls else "stop"
    chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}], "usage": usage})
    return chunks


class _FakeServer:
    """Serves realtime text sessions with the wrapper's protocol and answers from scripted replies.

    A reply is a list of chat completion chunks, or ``None`` to stream one
    delta and wait for ``response.cancel``.
    """

    def __init__(self) -> None:
        self.connections: list[_FakeWebSocket] = []
        self.replies: list = []

    async def connect(self, url: str, **_) -> "_FakeWebSocket":
        assert url == URL
        self.connections.append(_FakeWebSocket(self))
        return self.connections[-1]


class _FakeWebSocket:
    def __init__(self, server: _FakeServer) -> None:
        self._server = server
        self.session = realtime.new_session(MODEL)
        self.items: list[dict] = []
        self.requests: list[dict] = []
        self.sent: list[dict] = []
        self._text = ""
        self._response: realtime.ResponseEvents | None = None
        self._inbox: asyncio.Queue[str] = asyncio.Queue()
        self._emit(realtime.event("session.created", session=self.session))

    def _emit(self, *events: dict) -> None:
        for event in events:
            self._inbox.put_nowait(json.dumps(event))

    def _add(self, item: dict) -> None:
        self._emit(realtime.event("conversation.item.done", previous_item_id=None, item=item))
        self.items.append(item)

    async def send(self, raw: str) -> None:
        event = json.loads(raw)
        self.sent.append(event)
        event_type = event["type"]
        if event_type in ("session.update", "session.chat_completion_options.update"):
            update = event.get("session") or {"chat_completion_options": event["chat_completion_options"]}
            self.session = {**self.session, **update}
            realtime.validate_session(self.session)
        elif event_type == "conversation.item.create":
            self._add(realtime.normalize_item(event["item"]))
        elif event_type == "conversation.item.delete":
            self.items = [item for item in self.items if item["id"] != event["item_id"]]
        elif event_type == "input_text.append":
            self._text += event["text"]
        elif event_type == "input_text.clear":
            self._text = ""
        elif event_type == "input_text.commit":
            self._add(realtime.user_item(self._text))
            self._text = ""
        elif event_type == "response.create":
            self.requests.append(
                {**realtime.chat_request_fields(self.session), "messages": realtime.chat_messages(self.items, "")}
            )
            self._response = realtime.ResponseEvents(self.items[-1]["id"] if self.items else None)
            self._emit(realtime.event("response.output_text.delta", response_id="resp_old", delta="stale"))
            self._emit(*self._response.started())
            reply = self._server.replies.pop(0)
            for chunk in reply or [{"choices": [{"index": 0, "delta": {"content": "Long"}}]}]:
                self._emit(*self._response.chunk(chunk))
            if reply is not None:
                self._finish("completed")
        elif event_type == "response.cancel":
            if self._response is None:
                self._emit(realtime.error_event("no_active_response", "there is no active response", event))
            else:
                self._finish("cancelled")

    def _finish(self, status: str) -> None:
        events, items = self._response.finished(status)
        self._response = None
        self.items += items
        self._emit(*events)

    async def recv(self) -> str:
        return await self._inbox.get()

    async def close(self) -> None:
        pass


@pytest.fixture
def server(monkeypatch) -> _FakeServer:
    fake = _FakeServer()
    monkeypatch.setattr(streaming_llm, "connect", fake.connect)
    return fake


def _service() -> NvidiaStreamingLLMService:
    return NvidiaStreamingLLMService(api_key="test", base_url=URL, settings=NvidiaLLMSettings(model=MODEL))


async def _answer(service: NvidiaStreamingLLMService, context: LLMContext) -> list:
    service._reset_response_state()
    stream = await service.get_chat_completions(context)
    return [chunk async for chunk in stream]


def _types(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


def _conversation(websocket: _FakeWebSocket) -> list[tuple]:
    return [streaming_llm._item_key(item) for item in websocket.items]


def test_service_serves_chat_completions_from_the_same_server() -> None:
    assert str(_service()._client.base_url) == "http://nvidia-llm-vllm:8000/v1/"


def test_sampling_and_response_format_go_in_chat_completion_options() -> None:
    session = _service()._session
    fields, options = session._session_for(
        {
            "model": MODEL,
            "messages": [],
            "temperature": 0.3,
            "seed": None,
            "chat_template_kwargs": {"enable_thinking": True},
            "response_format": {"type": "json_object"},
            "max_completion_tokens": 64,
            "logprobs": True,
        }
    )

    assert fields["max_output_tokens"] == 64
    assert options == {
        "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": True},
        "response_format": {"type": "json_object"},
    }
    assert session._dropped_keys == {"logprobs"}


@pytest.mark.parametrize(("max_tokens", "expected"), [(4096, 4096), (4097, "inf"), (None, "inf")])
def test_output_token_limit_stays_within_the_realtime_range(max_tokens, expected) -> None:
    fields, _ = _service()._session._session_for({"model": MODEL, "messages": [], "max_tokens": max_tokens})

    assert fields["max_output_tokens"] == expected


def test_turn_prefills_text_and_streams_tool_calls(server) -> None:
    server.replies.append(
        _reply(
            "Checking.", [("call_1", "get_weather", '{"city":"Pune"}')], {"prompt_tokens": 20, "completion_tokens": 5}
        )
    )
    tools = ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="get_weather",
                description="Get the weather.",
                properties={"city": {"type": "string"}},
                required=["city"],
            )
        ]
    )
    context = LLMContext([{"role": "system", "content": "Be brief."}], tools=tools)
    service = _service()

    async def run() -> list:
        for text in ("Weather", "Weather in", "Whether in"):
            await service._session.sync(await service._session_request_for(context), text)
        context.add_message({"role": "user", "content": "Whether in Pune?"})
        return await _answer(service, context)

    chunks = asyncio.run(run())

    websocket = server.connections[0]
    assert _types(websocket.sent) == [
        "session.update",
        "session.chat_completion_options.update",
        "conversation.item.create",
        "input_text.append",
        "input_text.append",
        "input_text.clear",
        "input_text.append",
        "input_text.append",
        "input_text.commit",
        "response.create",
    ]
    assert [event["text"] for event in websocket.sent if event["type"] == "input_text.append"] == [
        "Weather",
        " in",
        "Whether in",
        " Pune?",
    ]
    assert websocket.requests[0]["tools"][0]["function"]["name"] == "get_weather"
    assert websocket.requests[0]["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Whether in Pune?"},
    ]
    deltas = [chunk.choices[0].delta for chunk in chunks if chunk.choices]
    assert deltas[0].content == "Checking."
    assert deltas[1].tool_calls[0].id == "call_1"
    assert deltas[1].tool_calls[0].function.name == "get_weather"
    assert "".join(delta.tool_calls[0].function.arguments for delta in deltas[1:]) == '{"city":"Pune"}'
    assert chunks[-1].usage.prompt_tokens == 20


def test_tool_result_turn_appends_only_the_result(server) -> None:
    server.replies += [_reply("Let me check.", [("call_1", "get_weather", '{"city":"Pune"}')]), _reply("Sunny.")]
    context = LLMContext([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather?"}])
    service = _service()

    async def run() -> None:
        await _answer(service, context)
        context.add_messages(
            [
                {"role": "assistant", "content": "Let me check."},
                {"role": "assistant", "tool_calls": [CALL]},
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
            ]
        )
        await _answer(service, context)

    asyncio.run(run())

    websocket = server.connections[0]
    second_turn = websocket.sent[_types(websocket.sent).index("response.create") + 1 :]
    assert _types(second_turn) == ["conversation.item.create", "response.create"]
    assert second_turn[0]["item"]["type"] == "function_call_output"
    assert websocket.requests[1]["messages"][2:] == [
        {"role": "assistant", "content": "Let me check.", "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
    ]


def test_spoken_answer_replaces_the_generated_one(server) -> None:
    server.replies += [_reply("Hello there, how are you today?"), _reply("Bye.")]
    context = LLMContext([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}])
    service = _service()

    async def run() -> None:
        await _answer(service, context)
        context.add_messages([{"role": "assistant", "content": "Hello there,"}, {"role": "user", "content": "Bye"}])
        await _answer(service, context)

    asyncio.run(run())

    assert len(server.connections) == 1
    websocket = server.connections[0]
    second_turn = websocket.sent[_types(websocket.sent).index("response.create") + 1 :]
    assert _types(second_turn) == [
        "conversation.item.delete",
        "conversation.item.create",
        "input_text.append",
        "input_text.commit",
        "response.create",
    ]
    assert _conversation(websocket) == [
        ("message", "system", "Be brief."),
        ("message", "user", "Hi"),
        ("message", "assistant", "Hello there,"),
        ("message", "user", "Bye"),
        ("message", "assistant", "Bye."),
    ]


def test_abandoned_answer_is_cancelled_and_reconciled(server) -> None:
    server.replies += [None, _reply("Sure.")]
    context = LLMContext([{"role": "user", "content": "Tell me a story"}])
    service = _service()

    async def run() -> None:
        first_chunk = asyncio.Event()

        async def consume() -> None:
            service._reset_response_state()
            async for _ in await service.get_chat_completions(context):
                first_chunk.set()

        task = asyncio.create_task(consume())
        await first_chunk.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        context.add_messages([{"role": "assistant", "content": "Lo"}, {"role": "user", "content": "Shorter"}])
        await _answer(service, context)

    asyncio.run(run())

    websocket = server.connections[0]
    cancel = next(event for event in websocket.sent if event["type"] == "response.cancel")
    assert cancel["response_id"].startswith("resp_")
    assert _conversation(websocket) == [
        ("message", "user", "Tell me a story"),
        ("message", "assistant", "Lo"),
        ("message", "user", "Shorter"),
        ("message", "assistant", "Sure."),
    ]


def test_update_cut_short_starts_a_new_session(server) -> None:
    server.replies += [_reply("Two."), _reply("Buenos Aires.")]
    context = LLMContext([{"role": "user", "content": "One plus one?"}])
    service = _service()
    summary = [{"role": "user", "content": "Summary: the user asked a sum."}]

    async def run() -> None:
        await _answer(service, context)
        websocket = server.connections[0]
        send = websocket.send
        deleting = asyncio.Event()

        async def stall_after_delete(raw: str) -> None:
            await send(raw)
            if json.loads(raw)["type"] == "conversation.item.delete":
                deleting.set()
                await asyncio.Event().wait()

        websocket.send = stall_after_delete
        task = asyncio.create_task(service._session.sync({"model": MODEL, "messages": summary}, "Stop"))
        await deleting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        context.set_messages([*summary, {"role": "user", "content": "Capital of Argentina?"}])
        await _answer(service, context)

    asyncio.run(run())

    assert len(server.connections) == 2
    assert _conversation(server.connections[1]) == [
        ("message", "user", "Summary: the user asked a sum."),
        ("message", "user", "Capital of Argentina?"),
        ("message", "assistant", "Buenos Aires."),
    ]


def test_late_confirmation_of_a_deleted_item_does_not_track_it_again(server) -> None:
    server.replies += [_reply("Paris."), _reply("Lima.")]
    service = _service()

    async def run() -> None:
        for system in ("Be brief.", "Be kind."):
            await service._session.sync({"model": MODEL, "messages": [{"role": "system", "content": system}]}, "")
        context = LLMContext([{"role": "system", "content": "Be kind."}, {"role": "user", "content": "France?"}])
        await _answer(service, context)
        context.add_messages([{"role": "assistant", "content": "Paris."}, {"role": "user", "content": "Peru?"}])
        await _answer(service, context)

    asyncio.run(run())

    websocket = server.connections[0]
    deleted = [event["item_id"] for event in websocket.sent if event["type"] == "conversation.item.delete"]
    assert len(deleted) == len(set(deleted)) == 1
    assert _conversation(websocket) == [
        ("message", "system", "Be kind."),
        ("message", "user", "France?"),
        ("message", "assistant", "Paris."),
        ("message", "user", "Peru?"),
        ("message", "assistant", "Lima."),
    ]


def test_unread_answer_is_cancelled_before_the_next_turn(server) -> None:
    server.replies += [None, _reply("Sure.")]
    context = LLMContext([{"role": "user", "content": "Tell me a story"}])
    service = _service()

    async def run() -> list:
        service._reset_response_state()
        await service.get_chat_completions(context)
        context.add_messages([{"role": "assistant", "content": "Lo"}, {"role": "user", "content": "Shorter"}])
        return await _answer(service, context)

    chunks = asyncio.run(run())

    websocket = server.connections[0]
    assert {"type": "response.cancel"} in websocket.sent
    assert [chunk.choices[0].delta.content for chunk in chunks if chunk.choices] == ["Sure."]
    assert _conversation(websocket)[-2:] == [("message", "user", "Shorter"), ("message", "assistant", "Sure.")]


def test_server_error_raises_and_resets_the_session(server) -> None:
    server.replies.append(_reply())
    context = LLMContext([{"role": "user", "content": "Hi"}])
    service = _service()

    async def run() -> None:
        service._reset_response_state()
        stream = await service.get_chat_completions(context)
        websocket = server.connections[0]
        websocket._inbox = asyncio.Queue()
        websocket._emit(realtime.error_event("processing_error", "boom"))
        async for _ in stream:
            pass

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(run())
    assert service._session._websocket is None


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


def test_queued_prefill_keeps_history_stable_when_the_user_turn_is_committed(server) -> None:
    server.replies.append(_reply("Hello!"))
    context = LLMContext([{"role": "system", "content": "Be brief."}])
    aggregator = StreamingLLMUserAggregator(context)
    queued = []
    service = _service()

    async def enqueue(frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, StreamingLLMInputFrame):
            queued.append(frame)

    async def run():
        aggregator.push_frame = enqueue
        await aggregator.process_frame(
            InterimTranscriptionFrame(text="Hi there", user_id="", timestamp=""), FrameDirection.DOWNSTREAM
        )
        # The final aggregation runs before the downstream processor drains its
        # queued prefill frame, appending the user turn to the shared history.
        context.add_message({"role": "user", "content": "Hi there"})
        live = queued[0]
        await service._session.sync(await service._session_request_for(live.context), live.text)
        await _answer(service, deepcopy(context))

    asyncio.run(run())

    websocket = server.connections[0]
    assert _types(websocket.sent).count("conversation.item.create") == 1
    assert websocket.requests[0]["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi there"},
    ]
