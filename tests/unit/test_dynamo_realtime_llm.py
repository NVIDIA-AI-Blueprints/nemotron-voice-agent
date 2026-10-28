# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import asyncio
import json

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.nvidia.llm import NvidiaLLMSettings

import examples.shared.nvidia_streaming_llm as streaming_llm
from examples.shared.nvidia_streaming_llm import NvidiaStreamingLLMService

URL = "ws://dynamo-frontend:8000/v1/realtime"
MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
CALL = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Pune"}'}}


def _message(role: str, text: str, item_id: str) -> dict:
    content_type = "output_text" if role == "assistant" else "input_text"
    return {"id": item_id, "type": "message", "role": role, "content": [{"type": content_type, "text": text}]}


class _FakeRealtimeServer:
    """Keeps a Dynamo-style conversation per connection and answers ``response.create`` from scripted replies.

    A reply is the answer text plus optional ``(call_id, name, arguments)`` calls,
    or ``None`` to stream one delta and wait for ``response.cancel``.
    """

    def __init__(self) -> None:
        self.connections: list[_FakeRealtimeWebSocket] = []
        self.replies: list = []

    async def connect(self, url: str, **_) -> "_FakeRealtimeWebSocket":
        assert url == URL
        self.connections.append(_FakeRealtimeWebSocket(self))
        return self.connections[-1]


class _FakeRealtimeWebSocket:
    def __init__(self, server: _FakeRealtimeServer) -> None:
        self._server = server
        self.sent: list[dict] = []
        self.items: list[dict] = []
        self._buffer = ""
        self._responses = 0
        self._inbox: asyncio.Queue[str] = asyncio.Queue()
        self._emit({"type": "session.created", "session": {}})

    def _emit(self, event: dict) -> None:
        self._inbox.put_nowait(json.dumps(event))

    def _add(self, item: dict) -> None:
        self.items.append(item)
        self._emit({"type": "conversation.item.done", "item": item})

    async def send(self, raw: str) -> None:
        event = json.loads(raw)
        self.sent.append(event)
        if event["type"] == "conversation.item.create":
            self._add(event["item"])
        elif event["type"] == "conversation.item.delete":
            self.items = [item for item in self.items if item["id"] != event["item_id"]]
        elif event["type"] == "input_text.append":
            self._buffer += event["text"]
        elif event["type"] == "input_text.clear":
            self._buffer = ""
        elif event["type"] == "input_text.commit":
            self._add(_message("user", self._buffer, f"srv_user_{len(self.items)}"))
            self._buffer = ""
        elif event["type"] == "response.create":
            self._responses += 1
            response_id = f"resp_{self._responses}"
            self._emit({"type": "response.output_text.delta", "response_id": "resp_old", "delta": "stale"})
            self._emit({"type": "response.created", "response": {"id": response_id}})
            reply = self._server.replies.pop(0)
            if reply is None:
                self._emit({"type": "response.output_text.delta", "response_id": response_id, "delta": "Long"})
                self._cancellable = response_id
                return
            self._answer(response_id, *reply)
        elif event["type"] == "response.cancel":
            response_id = event.get("response_id", f"resp_{self._responses}")
            self._add(_message("assistant", "Long", f"srv_asst_{len(self.items)}"))
            self._emit({"type": "response.done", "response": {"id": response_id, "status": "cancelled"}})

    def _answer(self, response_id: str, text: str, calls: list = (), usage: dict | None = None) -> None:
        if text:
            self._emit({"type": "response.output_text.delta", "response_id": response_id, "delta": text})
        for call_id, name, arguments in calls:
            item = {"id": f"fc_{call_id}", "type": "function_call", "call_id": call_id, "name": name}
            self._emit({"type": "response.output_item.added", "response_id": response_id, "item": item})
            for part in (arguments[:5], arguments[5:]):
                delta = {"call_id": call_id, "item_id": item["id"], "delta": part}
                self._emit({"type": "response.function_call_arguments.delta", "response_id": response_id, **delta})
        self._add(_message("assistant", text, f"srv_asst_{len(self.items)}"))
        for call_id, name, arguments in calls:
            self._add(
                {
                    "id": f"fc_{call_id}",
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                }
            )
        self._emit({"type": "response.done", "response": {"id": response_id, "status": "completed", "usage": usage}})

    async def recv(self) -> str:
        return await self._inbox.get()

    async def close(self) -> None:
        pass


@pytest.fixture
def server(monkeypatch) -> _FakeRealtimeServer:
    fake = _FakeRealtimeServer()
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


def _conversation(websocket: _FakeRealtimeWebSocket) -> list[tuple]:
    return [streaming_llm._item_key(item) for item in websocket.items]


def test_realtime_url_selects_the_dynamo_session() -> None:
    service = _service()

    assert isinstance(service._session, streaming_llm._RealtimeSession)
    assert str(service._client.base_url) == "http://dynamo-frontend:8000/v1/"


def test_reasoning_maps_to_effort_and_sampling_is_dropped() -> None:
    session = _service()._session._session_for(
        {
            "model": MODEL,
            "messages": [],
            "temperature": 0.3,
            "chat_template_kwargs": {"enable_thinking": True},
            "max_completion_tokens": 64,
        }
    )

    assert session["max_output_tokens"] == 64
    assert session["reasoning"] == {"effort": "medium"}
    assert "temperature" not in session


def test_reasoning_off_maps_to_minimal_effort() -> None:
    session = _service()._session._session_for(
        {"model": MODEL, "messages": [], "chat_template_kwargs": {"enable_thinking": False}}
    )

    assert session["reasoning"] == {"effort": "minimal"}


def test_turn_prefills_text_and_maps_tool_calls(server) -> None:
    server.replies.append(
        ("Checking.", [("call_1", "get_weather", '{"city":"Pune"}')], {"input_tokens": 20, "output_tokens": 5})
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
            await service._session.sync(service._session_request_for(context), text)
        context.add_message({"role": "user", "content": "Whether in Pune?"})
        return await _answer(service, context)

    chunks = asyncio.run(run())

    sent = server.connections[0].sent
    assert _types(sent) == [
        "session.update",
        "conversation.item.create",
        "input_text.append",
        "input_text.append",
        "input_text.clear",
        "input_text.append",
        "input_text.append",
        "input_text.commit",
        "response.create",
    ]
    assert sent[0]["session"]["model"] == MODEL
    assert sent[0]["session"]["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        }
    ]
    assert [event["text"] for event in sent if event["type"] == "input_text.append"] == [
        "Weather",
        " in",
        "Whether in",
        " Pune?",
    ]
    deltas = [chunk.choices[0].delta for chunk in chunks if chunk.choices]
    assert deltas[0].content == "Checking."
    assert deltas[1].tool_calls[0].id == "call_1"
    assert deltas[1].tool_calls[0].function.name == "get_weather"
    assert "".join(delta.tool_calls[0].function.arguments for delta in deltas[1:]) == '{"city":"Pune"}'
    assert chunks[-1].usage.prompt_tokens == 20
    assert all(chunk.id == "resp_1" for chunk in chunks)


def test_matching_answer_keeps_the_server_items(server) -> None:
    server.replies += [("Let me check.", [("call_1", "get_weather", '{"city":"Pune"}')]), ("Sunny.",)]
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
    assert _conversation(websocket)[-4:-1] == [
        ("message", "assistant", "Let me check."),
        ("function_call", "call_1", "get_weather", '{"city":"Pune"}'),
        ("function_call_output", "call_1", "Sunny"),
    ]


def test_spoken_answer_replaces_the_generated_one(server) -> None:
    server.replies += [("Hello there, how are you today?",), ("Bye.",)]
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
    server.replies += [None, ("Sure.",)]
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
    assert {"type": "response.cancel", "response_id": "resp_1"} in websocket.sent
    assert _conversation(websocket) == [
        ("message", "user", "Tell me a story"),
        ("message", "assistant", "Lo"),
        ("message", "user", "Shorter"),
        ("message", "assistant", "Sure."),
    ]


def test_unread_answer_is_cancelled_before_the_next_turn(server) -> None:
    server.replies += [None, ("Sure.",)]
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


def test_failed_response_raises_and_resets_the_session(server) -> None:
    server.replies.append(("",))
    context = LLMContext([{"role": "user", "content": "Hi"}])
    service = _service()

    async def fail(*_) -> None:
        raise RuntimeError("Realtime response failed: {}")

    async def run() -> None:
        service._reset_response_state()
        stream = await service.get_chat_completions(context)
        websocket = server.connections[0]
        websocket._inbox = asyncio.Queue()
        websocket._emit({"type": "error", "error": {"code": "server_error", "message": "boom"}})
        async for _ in stream:
            pass

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(run())
    assert service._session._websocket is None
