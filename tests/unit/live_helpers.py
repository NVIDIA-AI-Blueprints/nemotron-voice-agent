# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Shared doubles of the gateway's tests. They load no Pipecat module and nothing of the example.

The frontend adapter's doubles are in ``tests/unit/cascade_helpers.py``, so an import error in one box does not stop
the gateway's tests.
"""

import asyncio
import json

from live.ports import SessionParts
from live.protocol import SessionConfig


class FakeAdapter:
    """A frontend adapter and a delegation core in one, which record what the gateway asks of them.

    As the core, it is the scripted ``ResponsesClient`` the session builds its server delegation on: it
    opens one conversation, refuses the models in ``models_rejected`` (also when it checks settings), and answers each
    request with a stream that has no event, noting the request's input items.
    """

    CONVERSATION = "conv_fake"

    def __init__(self, fail_start=False):
        self.fail_start = fail_start
        self.session = self.delegation = None
        self.started = self.closed = False
        self.muted = False
        self.audio: list[bytes] = []
        self.appended: list[tuple] = []
        self.items: list[dict] = []  # the input items of the requests the server delegation sent
        self.creates = 0  # how many requests it sent
        self.models_rejected = {"bad-model"}
        self.checked: list[dict] = []  # the settings it was asked to check (input_tokens)

    def parts(self):
        """The parts of a session: this fake is its frontend adapter."""
        return SessionParts(frontend=self)

    def core(self):
        """The gateway's core hook: this fake is also the one delegation core."""
        return self

    async def start(self, session, delegation):
        if self.fail_start:
            raise RuntimeError("pipeline failed")
        self.session, self.delegation, self.started = session, delegation, True

    async def close(self):
        self.closed = True

    # The delegation core's official endpoints (live.server_delegation.ResponsesClient).
    async def create_conversation(self, metadata):
        return {"id": self.CONVERSATION, "object": "conversation", "created_at": 0, "metadata": dict(metadata)}

    async def delete_conversation(self, conversation_id):
        return {"id": conversation_id, "object": "conversation.deleted", "deleted": True}

    async def retrieve_model(self, model):
        if model in self.models_rejected:
            raise ValueError("model not served")
        return {"id": model, "object": "model", "created": 0, "owned_by": "test"}

    async def create_response(self, params):
        self.items.extend(params["input"])
        self.creates += 1
        return NoEvents()

    async def list_conversation_items(self, conversation_id):
        return {"object": "list", "data": [], "first_id": None, "last_id": None, "has_more": False}

    async def input_tokens(self, params):
        """Check a request's settings: a model in ``models_rejected`` is refused with its ``param``, as by the core."""
        self.checked.append(params)
        if params.get("model") in self.models_rejected:
            refused = ValueError(f"The model {params['model']!r} is not served")
            refused.param = "model"
            raise refused
        return {"object": "response.input_tokens", "input_tokens": 1}

    async def aclose(self):
        """The client's own close, which the gateway runs at server shutdown: it holds nothing to release."""

    async def set_muted(self, muted):
        self.muted = muted

    async def feed_audio(self, pcm, sample_rate):
        self.audio.append(pcm)

    async def append_instructions(self, text, client_event_id=None, delegation_id=None):
        self.appended.append(("instructions", text))

    async def append_thinking(self, text, client_event_id=None, delegation_id=None):
        self.appended.append(("thinking", text))

    async def append_commentary(self, text, client_event_id=None, delegation_id=None):
        self.appended.append(("commentary", text, client_event_id, delegation_id))


class NoEvents:
    """A response stream that ends at once."""

    def __aiter__(self):
        """The stream is its own iterator."""
        return self

    async def __anext__(self):
        """Nothing: the stream has ended."""
        raise StopAsyncIteration


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


def responses_config(**kwargs):
    """A responses-mode session configuration."""
    return SessionConfig(
        model="live-1",
        delegation={
            "type": "responses",
            "responses": {"model": "org/org/model", "reasoning": {"effort": "low", "summary": "auto"}},
        },
        **kwargs,
    )


class FakeWebSocket:
    """The server side of a primary WebSocket, driven by the test."""

    def __init__(self):
        self.to_server: asyncio.Queue = asyncio.Queue()
        self.from_server: list[dict] = []
        self._arrived = asyncio.Event()
        self.closed = False

    # What the gateway calls ---------------------------------------------------------------
    headers: dict = {}  # noqa: RUF012

    async def accept(self):
        return None

    async def receive(self):
        """The next frame as Starlette's ``WebSocket.receive()`` gives it: a text frame, here always."""
        return {"type": "websocket.receive", "text": await self.to_server.get()}

    async def receive_json(self):
        return json.loads(await self.to_server.get())

    async def receive_text(self):
        return await self.to_server.get()

    async def send_json(self, message):
        self.from_server.append(message)
        self._arrived.set()

    async def close(self, code=1000):
        self.closed = True

    # What the test calls --------------------------------------------------------------------
    def send(self, message):
        self.to_server.put_nowait(json.dumps(message))

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
