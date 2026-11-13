# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""Shared doubles of the gateway's tests. They load no Pipecat module and nothing of the example."""

import asyncio
import json

from starlette.websockets import WebSocketState

from gateways.live.ports import SessionParts
from gateways.live.protocol import SessionConfig


def response(response_id, *items, end="response.completed"):
    """Return one response's events, numbered from 1: created, each item done, then ``end`` unless it is None."""
    ended = {"id": response_id, "status": "completed", "output": list(items)}
    events = [
        {"type": "response.created", "response": {"id": response_id}},
        *({"type": "response.output_item.done", "item": item} for item in items),
        *([{"type": end, "response": ended}] if end else []),
    ]
    return [{**event, "sequence_number": number} for number, event in enumerate(events, 1)]


class Stream:
    """A response stream: its events, then (once ``until`` is set) ``then``; then it ends, or raises ``fail``."""

    def __init__(self, events=(), until=None, then=(), fail=None):
        self.events, self.until, self.then, self.fail, self.closed = list(events), until, list(then), fail, False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events and self.until is not None:
            await self.until.wait()
            self.events, self.until = self.then, None
        if self.events:
            return self.events.pop(0)
        if self.fail is not None:
            raise self.fail
        raise StopAsyncIteration

    async def aclose(self):
        self.closed = True


class FakeAdapter:
    """A frontend adapter and a delegation core in one, noting what the gateway asks of them.

    As the core, it lets a later call attach to a conversation in ``kept``; while ``keeps`` is set, a delete keeps its
    conversation there and answers ``deleted: false``, as work that outlives the call does.
    """

    CONVERSATION = "conv_fake"

    def __init__(self, peer=None, streams=()):
        self.peer, self.streams, self.hold, self.fail = peer, list(streams), {}, {}
        self.session = self.delegation = self.reader = None
        self.started = self.closed = self.muted = self.keeps = False
        self.audio, self.appended, self.typed_texts, self.kept = [], [], [], set()
        self.log, self.requests, self.checked, self.heard, self.raw = [], [], [], [], []
        self.models_rejected, self.fails_on, self.deleted = {"bad-model"}, None, asyncio.Event()

    def parts(self):
        return SessionParts(frontend=self)

    def core(self):
        return self

    async def step(self, name, noted=True):
        """Run step ``name``: wait for its ``hold`` event, then raise its ``fail`` error or its list's next error."""
        if noted:
            self.log.append(name)
        if name in self.hold:
            await self.hold[name].wait()
        failure = self.fail.get(name)
        if isinstance(failure, list):
            failure = failure.pop(0) if failure else None
        if failure is not None:
            raise failure

    async def start(self, session, delegation):
        await self.step("start")
        self.session, self.delegation, self.started = session, delegation, True
        if self.peer is not None:
            await self.peer.connect()
        if session.socket is not None:
            self.reader = asyncio.create_task(self.read(session.socket))
        delegation.listen(self)

    async def read(self, socket):
        """Hand the socket each frame the client sends, as the transport's serializer does, until the client leaves."""
        while (frame := await socket.receive())["type"] != "websocket.disconnect":
            audio = await socket.handle(frame["text"] if frame.get("text") is not None else frame.get("bytes"))
            if audio:
                self.audio.append(audio)

    async def close(self):
        await self.step("close")
        self.closed = True
        if self.reader is not None and self.reader is not asyncio.current_task():
            self.reader.cancel()
            await asyncio.wait({self.reader})

    async def set_muted(self, muted):
        self.muted = muted

    async def append_instructions(self, text, client_event_id=None, delegation_id=None):
        self.appended.append(("instructions", text, client_event_id, delegation_id))

    async def append_thinking(self, text, client_event_id=None, delegation_id=None):
        self.appended.append(("thinking", text, client_event_id, delegation_id))

    async def append_commentary(self, text, client_event_id=None, delegation_id=None):
        self.appended.append(("commentary", text, client_event_id, delegation_id))

    def typed(self, text):
        self.typed_texts.append(text)

    def on_delegation_event(self, delegation_id, event):
        self.heard.append((delegation_id, event["type"]))
        self.raw.append(event)
        if event["type"] == self.fails_on:
            raise RuntimeError("the frontend failed on this event")

    def on_delegation_lost(self, delegation_id):
        self.heard.append((delegation_id, "lost"))

    async def create_conversation(self, metadata):
        await self.step("create")
        self.metadata = metadata
        return {"id": self.CONVERSATION, "object": "conversation", "created_at": 0, "metadata": dict(metadata)}

    async def update_conversation(self, conversation_id, metadata):
        await self.step("update")
        if conversation_id not in self.kept:
            raise LookupError("Conversation not found")
        self.kept.discard(conversation_id)
        self.metadata = metadata
        return {"id": conversation_id, "object": "conversation", "created_at": 0, "metadata": dict(metadata)}

    async def delete_conversation(self, conversation_id):
        await self.step("delete")
        self.deleted.set()
        if self.keeps:
            self.kept.add(conversation_id)
        return {"id": conversation_id, "object": "conversation.deleted", "deleted": not self.keeps}

    async def retrieve_model(self, model):
        if model in self.models_rejected:
            raise ValueError("model not served")
        return {"id": model, "object": "model", "created": 0, "owned_by": "test"}

    async def create_response(self, params):
        self.requests.append(params)
        await self.step("response", noted=False)
        stream = self.streams.pop(0) if self.streams else response(f"resp_{len(self.requests)}")
        return stream if isinstance(stream, Stream) else Stream(stream)

    async def list_conversation_items(self, conversation_id):
        return {"object": "list", "data": [], "first_id": None, "last_id": None, "has_more": False}

    async def input_tokens(self, params):
        """Check a request's settings: a model in ``models_rejected`` is refused with its ``param``, as by the core."""
        await self.step("check", noted=False)
        self.checked.append(params)
        if params.get("model") in self.models_rejected:
            refused = ValueError(f"The model {params['model']!r} is not served")
            refused.param = "model"
            raise refused
        return {"object": "response.input_tokens", "input_tokens": 1}

    async def aclose(self):
        await self.step("aclose")


class Collector:
    """A subscriber that keeps every event it receives."""

    def __init__(self):
        self.events: list[dict] = []
        self._arrived = asyncio.Event()

    async def send(self, message):
        self.events.append(message)
        self._arrived.set()

    async def wait_for(self, kind, timeout=2.0):
        async with asyncio.timeout(timeout):
            while True:
                for event in self.events:
                    if event["type"] == kind:
                        return event
                self._arrived.clear()
                await self._arrived.wait()

    def kinds(self):
        return [event["type"] for event in self.events]


def responses_config(conversation=None, **kwargs):
    """Return a responses-mode session configuration; ``conversation`` names a kept one to continue."""
    responses = {"model": "org/org/model", "reasoning": {"effort": "low", "summary": "auto"}}
    if conversation is not None:
        responses["conversation"] = conversation
    return SessionConfig(model="live-1", delegation={"type": "responses", "responses": responses}, **kwargs)


class ClientSocket:
    """The server side of a primary WebSocket whose client keeps its events in a :class:`Collector`, and sends none."""

    client_state = WebSocketState.CONNECTED

    def __init__(self, collector: Collector):
        self.send_json = collector.send

    async def receive(self):
        await asyncio.Event().wait()


class FakeWebSocket:
    """The server side of a primary WebSocket, driven by the test; it notes what it had sent when first closed."""

    headers: dict = {"host": "localhost"}  # noqa: RUF012

    def __init__(self):
        self.to_server: asyncio.Queue = asyncio.Queue()
        self.from_server: list[dict] = []
        self._arrived = asyncio.Event()
        self.closed = False
        self.sent_at_close: list[str] | None = None
        self.client_state = WebSocketState.CONNECTED

    async def accept(self):
        return None

    async def receive(self):
        """Return the next frame as Starlette's ``WebSocket.receive()`` does: a text frame, or the client's close."""
        sent = await self.to_server.get()
        if isinstance(sent, int):
            self.client_state = WebSocketState.DISCONNECTED
            return {"type": "websocket.disconnect", "code": sent}
        return {"type": "websocket.receive", "text": sent}

    async def send_json(self, message):
        self.from_server.append(message)
        self._arrived.set()

    async def close(self, code=1000):
        if not self.closed:
            self.sent_at_close = [event["type"] for event in self.from_server]
        self.closed = True

    def send(self, message):
        self.to_server.put_nowait(json.dumps(message))

    def leave(self, code=1000):
        """Close the client's side of the socket with ``code``."""
        self.to_server.put_nowait(code)

    async def wait_for(self, kind, *, after=0, timeout=10.0, where=None):
        async with asyncio.timeout(timeout):
            while True:
                for event in self.from_server[after:]:
                    if event["type"] == kind and (where is None or where(event)):
                        return event
                self._arrived.clear()
                await self._arrived.wait()

    def kinds(self, skip_audio=True):
        return [e["type"] for e in self.from_server if not (skip_audio and e["type"] == "session.output_audio.delta")]
