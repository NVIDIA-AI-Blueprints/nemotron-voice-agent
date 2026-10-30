# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""Doubles for the backend LLM's tests: scripted model clients, providers, tool sources and an IM-like task."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any, ClassVar

from openai.types.chat import ChatCompletionChunk

from delegation_core.backend.llm import LlmAdapter, LlmSettings, LlmSource
from delegation_core.backend.port import TERMINALS, BoundaryInput
from delegation_core.backend.providers import Backend, ProviderLimits
from delegation_core.config import ModelEndpoint
from tests.unit.contracts.backend import MENU_FUNCTION, FakeTaskContext

ENDPOINT = ModelEndpoint(model="test/model", base_url="http://localhost:1/v1")
LIMITS = ProviderLimits(provider_retries=0, first_event_timeout_seconds=2.0, idle_timeout_seconds=2.0)
FAST = LlmSettings(
    first_event_timeout_seconds=0.5,
    idle_timeout_seconds=0.5,
    server_tool_timeout_seconds=1.0,
    stop_timeout_seconds=0.5,
    provider_retries=0,
)


# --------------------------------------------------------------------------------------------- model clients
class FakeStream:
    """A model client's stream of chunks or events; a :class:`Sleep` in it is a pause between two of them."""

    def __init__(self, items):
        self.items, self.closed = list(items), False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for item in self.items:
            if isinstance(item, Sleep):
                await asyncio.sleep(item.seconds)
            else:
                yield item

    async def close(self):
        self.closed = True


class FakeChat:
    """``chat.completions.create``, scripted: each call streams the next list of chunks, or raises an exception."""

    def __init__(self, *script):
        self.script, self.requests = list(script), []
        self.chat = self.completions = self

    async def create(self, **request):
        self.requests.append(request)
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return FakeStream(step)

    async def close(self):
        pass


class Event:
    """A Responses stream event as the OpenAI client gives it: a model with ``model_dump``."""

    def __init__(self, data):
        self.data = data

    def model_dump(self, **_):
        return json.loads(json.dumps(self.data))


class FakeResponses:
    """``responses.create``, scripted: each call streams the next round of events."""

    def __init__(self, *rounds):
        self.rounds, self.requests = list(rounds), []
        self.responses = self

    async def create(self, **request):
        self.requests.append(request)
        return FakeStream(Event(event) for event in self.rounds.pop(0))

    async def close(self):
        pass


def chunk(content=None, tool_calls=None, finish=None):
    delta = {"role": "assistant"}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    body = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "test/model"}
    body["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
    return ChatCompletionChunk.model_validate(body)


def tool_fragment(index, call_id=None, name=None, arguments=None):
    function = {key: value for key, value in (("name", name), ("arguments", arguments)) if value is not None}
    fragment = {"index": index, "type": "function", "function": function}
    if call_id:
        fragment["id"] = call_id
    return fragment


# --------------------------------------------------------------------------------------------- providers
class Sleep:
    """A pause in a scripted round."""

    def __init__(self, seconds):
        self.seconds = seconds


class Hang:
    """A pause in a scripted round that ignores being cancelled: a provider that does not let go (a stop's limit)."""

    def __init__(self, seconds):
        self.seconds = seconds


class ScriptedProvider(Backend):
    """A provider that plays one script per round: events, exceptions, :class:`Sleep`, :class:`Hang` or gates."""

    name = "scripted"

    def __init__(self, *rounds, limits: ProviderLimits = LIMITS, endpoint: ModelEndpoint = ENDPOINT):
        super().__init__(endpoint, limits)
        self.rounds, self.payloads, self.opened, self.closed = list(rounds), [], 0, False
        self.read_after_terminal = False

    async def _events(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        self.opened += 1
        ended = False
        for step in self.rounds.pop(0):
            self.read_after_terminal = self.read_after_terminal or ended
            ended = ended or (isinstance(step, dict) and step.get("type") in TERMINALS)
            if isinstance(step, BaseException):
                raise step
            if isinstance(step, Sleep):
                await asyncio.sleep(step.seconds)
            elif isinstance(step, Hang):
                await _hang(step.seconds)
            elif isinstance(step, asyncio.Event):
                await step.wait()
            else:
                yield step

    async def close(self):
        self.closed = True


async def _hang(seconds):
    task = asyncio.ensure_future(asyncio.sleep(seconds))
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue


# --------------------------------------------------------------------------------------------- tool sources
class RecordingTools:
    """A server tool source whose tools answer ``{"ok": true}``, slowly when asked, and record their calls."""

    label: ClassVar[str] = "shop"

    def __init__(self, names: Sequence[str] = ("lookup",), *, delay: float = 0.0, fail: Sequence[str] = ()):
        self.names, self.delay, self.fail, self.calls = list(names), delay, set(fail), []

    def tools(self):
        return [
            {"type": "function", "name": name, "parameters": {"type": "object", "properties": {}}}
            for name in self.names
        ]

    async def execute(self, call_id, name, arguments):
        self.calls.append((call_id, name, arguments))
        if self.delay:
            await asyncio.sleep(self.delay)
        if name in self.fail:
            raise RuntimeError(f"{name} broke")
        return json.dumps({"ok": True, "tool": name})


class WorkTools:
    """The IM's work tools as a test sees them: ``show`` and ``cancel_work``, which marks the task cancelled."""

    label: ClassVar[str] = "delegation_core"

    def __init__(self):
        self.shown, self.cancelled = [], False

    def tools(self):
        empty = {"type": "object", "properties": {}}
        show = {"type": "object", "properties": {"display": {"type": "string"}}, "required": ["display"]}
        return [
            {"type": "function", "name": "show", "parameters": show},
            {"type": "function", "name": "cancel_work", "parameters": empty},
        ]

    async def execute(self, call_id, name, arguments):
        if name == "show":
            self.shown.append(json.loads(arguments)["display"])
            return json.dumps({"shown": True})
        self.cancelled = True
        return json.dumps({"stopping": "after this step"})


# --------------------------------------------------------------------------------------------- the task
class IMTask(FakeTaskContext):
    """A task context that keeps the conversation as the IM does and refuses calls after its own ``cancel_work``."""

    def __init__(self, *, work_tools: WorkTools | None = None, **kwargs: Any):
        kwargs.setdefault("declared", [MENU_FUNCTION])
        super().__init__(tools=work_tools, **kwargs)
        self.requests: list[dict] = []
        self.owed: dict[str, str] = {}
        self.calls_seen: list[str] = []

    def request(self) -> dict:
        request = super().request()
        self.requests.append(request)
        return request

    def route(self, call):
        if getattr(self.work_tools, "cancelled", False):
            return "refused"
        return super().route(call)

    def tool_outcome(self, call_id, output, outcome):
        super().tool_outcome(call_id, output, outcome)
        self.owed[call_id] = output

    def take_boundary_input(self) -> BoundaryInput:
        joined = [
            {"type": "function_call_output", "call_id": call_id, "output": self.owed.pop(call_id)}
            for call_id in self.calls_seen
            if call_id in self.owed
        ]
        self.items += joined
        boundary = super().take_boundary_input()
        return BoundaryInput((*joined, *boundary.outputs), boundary.steering)

    def see(self, event: dict) -> None:
        if event["type"] == "response.completed":
            output = event["response"]["output"]
            self.items += output
            self.calls_seen += [item["call_id"] for item in output if item.get("type") == "function_call"]


async def drive(run, task: IMTask) -> list[dict]:
    """Read a run's events as the IM would, showing each to ``task``."""
    events = []
    async for event in run.events():
        task.see(event)
        events.append(event)
    return events


def source(
    *providers: Backend,
    settings: LlmSettings = FAST,
    tools: Sequence | None = None,
    endpoints=None,
) -> LlmSource:
    """An ``LlmSource`` whose providers are ``providers``, made in turn, one per model a session asks for."""
    made = list(providers)
    return LlmSource(
        settings,
        endpoints=endpoints or (lambda model: ModelEndpoint(model=model or ENDPOINT.model, base_url=ENDPOINT.base_url)),
        tools=lambda: list(tools or ()),
        make_provider=lambda endpoint: made.pop(0),
    )


def adapter(*providers: Backend, **kwargs: Any) -> LlmAdapter:
    """One IM session's adapter on ``providers`` (see :func:`source`)."""
    return source(*providers, **kwargs).new()


def items_of(events: list[dict], item_type: str) -> list[dict]:
    return [e["item"] for e in events if e["type"] == "response.output_item.done" and e["item"]["type"] == item_type]


def kinds(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]
