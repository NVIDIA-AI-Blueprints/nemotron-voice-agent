# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""The IM's test doubles: scripted adapters and conductors, the core's settings and texts, and the IM's test case."""

from __future__ import annotations

import asyncio
import json
import unittest
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from unittest import mock

from examples.frontend_backend_live.delegation_core.backend.port import (
    Conductor,
    ConductorInput,
    Decision,
    Place,
    Start,
    Steer,
    StopOutcome,
    TaskContext,
)
from examples.frontend_backend_live.delegation_core.config import CoreSettings, Texts
from examples.frontend_backend_live.delegation_core.im.input import BackendTask
from examples.frontend_backend_live.delegation_core.im.session import IMSession
from examples.frontend_backend_live.delegation_core.im.stream import ResponseStream, ResponseWriter
from examples.frontend_backend_live.delegation_core.items import message_text
from tests.unit.contracts.endings import normalized
from tests.unit.core_helpers import CORE_BLOCK, CORE_TEXTS

# The core's own settings, as the registry's block sets them: the IM's settings and its texts.
CORE_SETTINGS = CoreSettings.parse(CORE_BLOCK, CORE_TEXTS)
TEXTS = CORE_SETTINGS.texts
DEFAULTS = BackendTask("test/model", "Do the task.", [])
MENU = {"type": "function", "name": "get_menu", "parameters": {"type": "object", "properties": {}}}
# The lines the endings fixture marks by name ("[handover_line]" ...).
FIXTURE_LINES = {key: getattr(TEXTS, key) for key in ("handover_line", "cancelled_line", "unconfirmed_line")}


def is_handover_line(text: str) -> bool:
    """Whether ``text`` is ``texts.handover_line`` with a taker's delegation id in its ``{delegation}`` field."""
    head, _, tail = TEXTS.handover_line.partition("{delegation}")
    return len(text) > len(head) + len(tail) and text.startswith(head) and text.endswith(tail)


def ask(text: str) -> list[dict]:
    """A request: one user message."""
    return [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}]


def message_item(text: str, phase: str, item_id: str = "msg") -> dict:
    item = {"id": item_id, "type": "message", "role": "assistant", "status": "completed", "phase": phase}
    return {**item, "content": [{"type": "output_text", "text": text, "annotations": []}]}


def call_item(name: str, arguments: dict, call_id: str) -> dict:
    item = {"id": f"fc_{call_id}", "type": "function_call", "status": "completed", "call_id": call_id, "name": name}
    return {**item, "arguments": json.dumps(arguments)}


async def read_all(stream: AsyncIterator[dict]) -> list[dict]:
    """Every event of a response stream, until it ends."""
    return [event async for event in stream]


def finished_items(events: list[dict]) -> list[dict]:
    """The items a response's events finished, in order."""
    return [event["item"] for event in events if event["type"] == "response.output_item.done"]


def app_outputs(events: list[dict], output: str = "{}") -> list[dict]:
    """The app's output for every function call a response's events carry."""
    calls = [item for item in finished_items(events) if item["type"] == "function_call"]
    return [{"type": "function_call_output", "call_id": call["call_id"], "output": output} for call in calls]


def said(events: list[dict]) -> list[tuple[str, str]]:
    """The phase and text of each message a response's events finished, in order."""
    return [(item["phase"], message_text(item)) for item in finished_items(events) if item["type"] == "message"]


def texts(events: list[dict]) -> list[str]:
    """The text of each message a response's events finished, in order."""
    return [text for _, text in said(events)]


def function_calls(events: list[dict]) -> list[str]:
    """The call id of each of the app's function calls a response's events finished, in order."""
    return [item["call_id"] for item in finished_items(events) if item["type"] == "function_call"]


def records_of(message: dict) -> list[dict]:
    """The work records a request's message carries, one per line of its records block."""
    head = TEXTS.records_block.split("{records}")[0]
    (block,) = [part["text"] for part in message["content"] if part["text"].startswith(head)]
    return [json.loads(line) for line in block.splitlines()[1:]]


class Wire:
    """Everything the IM writes, in order, as ``(response_id, event)`` with ``response_id`` added to each event."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    @contextmanager
    def recording(self) -> Iterator[Wire]:
        original = ResponseWriter._emit

        def emit(writer, kind, **fields):
            original(writer, kind, **fields)
            self.events.append((writer.id, {"type": kind, **deepcopy(fields), "response_id": writer.id}))

        with mock.patch.object(ResponseWriter, "_emit", emit):
            yield self


# The steps of a scripted task -------------------------------------------------------------------------------------
class ScriptStep:
    """One step of a scripted task. ``ends``: the task's events stop after it (an answer, or the app's calls)."""

    ends = False

    async def play(self, run: ScriptedRun) -> AsyncIterator[dict]:
        if False:
            yield {}


@dataclass
class Say(ScriptStep):
    """Commentary in the task's next round."""

    text: str

    async def play(self, run):
        run.said.append(self.text)
        if False:
            yield {}


@dataclass
class _Answer(ScriptStep):
    text: str
    usage: dict | None = None
    hold: asyncio.Event | None = None
    ends = True

    async def play(self, run):
        for event in run.round_events([run.message(self.text, "final_answer")], usage=self.usage):
            yield event
        if self.hold is not None:
            await self.hold.wait()


@dataclass
class _Checkpoint(ScriptStep):
    async def play(self, run):
        run.boundary()
        if False:
            yield {}


@dataclass
class _AppCall(ScriptStep):
    name: str
    arguments: dict
    call_id: str | None = None
    ends = True

    async def play(self, run):
        call = call_item(self.name, self.arguments, self.call_id or f"call_{run.task.task_id}_{run.round + 1}")
        for event in run.round_events([call]):
            yield event
        run.routes.append(run.task.route(call))


@dataclass
class OwnCall(ScriptStep):
    """A round that calls the backend's own tool ``name``, which the run runs and reports."""

    name: str = "lookup"
    output: str = '{"ok": true}'
    fails: bool = False
    label: str = "cafe"
    call_id: str | None = None

    async def play(self, run):
        call = call_item(self.name, {}, self.call_id or f"call_{run.task.task_id}_{run.round + 1}")
        for event in run.round_events([call]):
            yield event
        route = run.task.route(call)
        run.routes.append(route)
        if route == "backend":
            run.task.tool_started(call, self.label)
            run.task.tool_outcome(call["call_id"], self.output, "failed" if self.fails else "done")


@dataclass
class Show(ScriptStep):
    """A round that calls the IM's ``show`` work tool, which the run runs through the task's tools."""

    display: str

    async def play(self, run):
        call = call_item("show", {"display": self.display}, f"call_{run.task.task_id}_show_{run.round + 1}")
        for event in run.round_events([call]):
            yield event
        route = run.task.route(call)
        run.routes.append(route)
        if route == "backend":
            tools = run.task.tools()
            run.task.tool_started(call, tools.label)
            output = await tools.execute(call["call_id"], "show", call["arguments"])
            run.task.tool_outcome(call["call_id"], output, "done")


@dataclass
class _Wait(ScriptStep):
    event: asyncio.Event

    async def play(self, run):
        released, stopping = asyncio.ensure_future(self.event.wait()), asyncio.ensure_future(run.stopping.wait())
        await asyncio.wait({released, stopping}, return_when=asyncio.FIRST_COMPLETED)
        for waiter in (released, stopping):
            waiter.cancel()
        if False:
            yield {}


@dataclass
class CancelOwnWork(ScriptStep):
    """A round that calls the work tool cancel_work, run through the task's tools; then a round of ``later`` calls."""

    later: tuple[dict, ...] = ()

    async def play(self, run):
        call = call_item("cancel_work", {}, "call_cancel")
        for event in run.round_events([call]):
            yield event
        run.routes.append(run.task.route(call))
        tools = run.task.tools()
        run.task.tool_started(call, tools.label)
        run.task.tool_outcome("call_cancel", await tools.execute("call_cancel", "cancel_work", "{}"), "done")
        if self.later:
            for event in run.round_events(list(self.later)):
                yield event
            run.routes += [run.task.route(later) for later in self.later]


@dataclass
class Fail(ScriptStep):
    """The run raises."""

    error: Exception = field(default_factory=lambda: RuntimeError("The backend failed"))

    async def play(self, run):
        raise self.error
        yield {}


@dataclass
class Cut(ScriptStep):
    """An attempt cut short (``response.incomplete`` with ``reason``), then the run raises."""

    reason: str = "max_output_tokens"

    async def play(self, run):
        details = {"incomplete_details": {"reason": self.reason}}
        for event in run.round_events([run.message("Half an answ", "final_answer")], terminal=("incomplete", details)):
            yield event
        raise RuntimeError(f"The round was cut: {self.reason}")


@dataclass
class FailedAttempt(ScriptStep):
    """An attempt that streams a message and fails; the run goes on with its next step (a retry)."""

    code: str = "server_error"

    async def play(self, run):
        error = {"error": {"code": self.code, "message": "The attempt failed."}}
        for event in run.round_events([run.message("Half an answ", "final_answer")], terminal=("failed", error)):
            yield event


@dataclass
class Progress(ScriptStep):
    """The run reports progress that is not model text."""

    text: str

    async def play(self, run):
        run.task.progress(self.text)
        if False:
            yield {}


@dataclass
class Empty(ScriptStep):
    """A round with no items: the task's events end with no answer."""

    ends = True

    async def play(self, run):
        for event in run.round_events([]):
            yield event


def answer(text: str = "Done.", usage: dict | None = None, hold: asyncio.Event | None = None) -> ScriptStep:
    """A round that answers ``text``; with ``hold``, the task's events end only once it is set."""
    return _Answer(text, usage, hold)


def checkpoint() -> ScriptStep:
    """A safe point with no round: the run takes the boundary input (and settles any steering)."""
    return _Checkpoint()


def app_call(name: str = "get_menu", arguments: dict | None = None, call_id: str | None = None) -> ScriptStep:
    """A round that calls the app's function ``name``; the task's events end until resumed."""
    return _AppCall(name, arguments or {}, call_id)


def wait(event: asyncio.Event) -> ScriptStep:
    """Pause until ``event`` is set or the run is stopped."""
    return _Wait(event)


class ScriptedRun:
    """One task's run: it plays its steps; each round starts at a safe point (``take_boundary_input``)."""

    def __init__(self, adapter: ScriptedAdapter, task: TaskContext, steps: Sequence[ScriptStep]):
        self.adapter, self.task, self.steps = adapter, task, list(steps)
        self.round = 0
        self.said: list[str] = []
        self.stopping = asyncio.Event()
        self.steers = self.resumes = self.stops = 0
        self.requests: list[dict] = []
        self.histories: list[list[dict]] = []
        self.boundaries: list = []
        self.applied: list = []  # the steering it applied
        self.routes: list[str] = []

    def message(self, text: str, phase: str) -> dict:
        return message_item(text, phase, f"msg_{self.task.task_id}_{self.round + 1}_{len(text)}")

    def boundary(self) -> None:
        given = self.task.take_boundary_input()
        self.boundaries.append(given)
        if given.steering is not None:
            self.task.settle(given.steering, self.adapter.delivery)
            if self.adapter.delivery == "applied":
                self.applied.append(given.steering)

    def round_events(
        self, items: list[dict], *, usage: dict | None = None, terminal=None, boundary: bool = True
    ) -> list[dict]:
        """One attempt of a round: the commentary said before it, then ``items``, after a safe point if ``boundary``."""
        if boundary:
            self.boundary()
        self.round += 1
        self.histories.append(self.task.history())
        self.requests.append(self.task.request())
        said = [message_item(text, "commentary", f"msg_said_{self.round}_{n}") for n, text in enumerate(self.said)]
        self.said = []
        items = [*said, *items]
        response_id = f"resp_{self.task.task_id}_{self.round}"
        events = [{"type": "response.created", "response": {"id": response_id, "status": "in_progress"}}]
        for index, item in enumerate(items):
            events.append({"type": "response.output_item.added", "output_index": index, "item": item})
            events.append({"type": "response.output_item.done", "output_index": index, "item": item})
        kind, fields = terminal or ("completed", {"usage": usage} if usage else {})
        response = {"id": response_id, "status": kind, "output": items, **fields}
        events.append({"type": f"response.{kind}", "response": response})
        return events

    async def events(self) -> AsyncIterator[dict]:
        while self.steps and not self.stopping.is_set():
            step = self.steps.pop(0)
            async for event in step.play(self):
                if self.stopping.is_set():
                    return
                yield event
            if step.ends:
                return

    def steer(self) -> None:
        self.steers += 1

    async def resume(self) -> None:
        self.resumes += 1
        self.boundary()

    async def stop(self) -> StopOutcome:
        self.stops += 1
        self.stopping.set()
        if self.adapter.stop_delay:
            await asyncio.sleep(self.adapter.stop_delay)
        return self.adapter.stop_outcome


class ScriptedAdapter:
    """A backend adapter whose tasks each play the next script (steps, or a function of the task) in a ``run``."""

    def __init__(
        self,
        *scripts: Sequence[ScriptStep] | Callable[[TaskContext], Sequence[ScriptStep]],
        max_running: int = 1,
        conductor: Conductor | None = None,
        delivery: str = "applied",
        stop_outcome: StopOutcome = "stopped",
        stop_delay: float = 0.0,
        run: type[ScriptedRun] = ScriptedRun,
    ):
        self.scripts, self.run = list(scripts), run
        self.max_running = max_running
        self._conductor = conductor if conductor is not None else SteerNewest()
        self.delivery, self.stop_outcome, self.stop_delay = delivery, stop_outcome, stop_delay
        self.runs: list[ScriptedRun] = []
        self.conductors = 0
        self.closed = False

    def conductor(self) -> Conductor:
        self.conductors += 1
        return self._conductor

    def start(self, task: TaskContext) -> ScriptedRun:
        script = self.scripts.pop(0) if self.scripts else [answer()]
        run = self.run(self, task, script(task) if callable(script) else script)
        self.runs.append(run)
        return run

    async def close(self) -> None:
        self.closed = True


# Conductors ---------------------------------------------------------------------------------------------------------
class SteerNewest:
    """The backend LLM's ``steer`` rule: the arrivals steer into the newest open Work, or start one."""

    async def decide(self, given: ConductorInput, place: Place) -> None:
        delegations = tuple(arrival.delegation_id for arrival in given.arrivals)
        if given.works:
            await place(Steer(given.works[-1].work_id, delegations))
        else:
            await place(Start(delegations))


class StartEach:
    """A conductor that starts a Work of its own for every arrival, which waits its turn for a slot."""

    async def decide(self, given: ConductorInput, place: Place) -> None:
        for arrival in given.arrivals:
            await place(Start((arrival.delegation_id,)))


def planned(*plans: Callable[[ConductorInput, Place], Awaitable[None]]) -> ScriptedConductor:
    """A conductor that runs one plan per ``decide()``, in order."""
    waiting = list(plans)

    async def plan(given: ConductorInput, place: Place) -> None:
        await waiting.pop(0)(given, place)

    return ScriptedConductor(plan)


def placing(*decisions: Decision) -> Callable[[ConductorInput, Place], Awaitable[None]]:
    """A plan that places ``decisions`` in order, whatever it is given."""

    async def plan(given: ConductorInput, place: Place) -> None:
        for decision in decisions:
            await place(decision)

    return plan


@dataclass
class ScriptedConductor:
    """A conductor that hands every ``decide()`` to ``plan(given, place)``; it keeps what it was given and told."""

    plan: Callable[[ConductorInput, Place], Awaitable[None]]
    given: list[ConductorInput] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)

    async def decide(self, given: ConductorInput, place: Place) -> None:
        self.given.append(given)

        async def recording(decision: Decision) -> str:
            outcome = await place(decision)
            self.outcomes.append(outcome)
            return outcome

        await self.plan(given, recording)


class IMCase(unittest.IsolatedAsyncioTestCase):
    """An IM test: the wire recorded, and every session it starts closed at the end."""

    def setUp(self):
        self.wire = self.enterContext(Wire().recording())
        self.traces: list[tuple[str, dict]] = []

    def session(self, adapter: ScriptedAdapter, *, texts: Texts = TEXTS, **changes: Any) -> IMSession:
        """A started IM session on ``adapter``: the core's settings, which ``changes`` override, and ``texts``.

        ``changes`` may also give the session's ``result_format`` and ``output_steps``.
        """
        formats = ("result_format", "output_steps")
        options = {name: changes.pop(name) for name in formats if name in changes}
        changes.setdefault("close_grace_seconds", 0.01)  # a test's last close does not wait for runs it left hanging
        settings = CORE_SETTINGS.delegation.model_copy(update=changes)
        observe = lambda kind, **fields: self.traces.append((kind, fields))  # noqa: E731
        im = IMSession(adapter, DEFAULTS, settings, texts, **options, observe=observe)
        im.start()
        self.addAsyncCleanup(im.close)
        return im

    async def turns(self, count: int = 30) -> None:
        """Let the event loop run ``count`` times, so everything ready runs."""
        for _ in range(count):
            await asyncio.sleep(0)

    async def until(self, predicate, what: str, timeout: float = 4.0) -> None:
        """Wait until ``predicate()`` holds."""
        try:
            async with asyncio.timeout(timeout):
                while not predicate():
                    await asyncio.sleep(0.01)
        except TimeoutError:
            raise AssertionError(f"timed out waiting for {what}") from None

    async def admit_each(self, im: IMSession, *requests: str, settings: dict | None = None) -> list[ResponseStream]:
        """Admit one request per text as D1, D2 ..., letting everything ready run after each; their streams."""
        streams = []
        for number, text in enumerate(requests, 1):
            streams.append(im.admit(ask(text), delegation_id=f"D{number}", settings=settings).stream)
            await self.turns()
        return streams

    async def answer_calls(self, im: IMSession, events: list[dict], output: str = "{}", **admission: Any) -> list[dict]:
        """The app's ``output`` for each call ``events`` ended with, as D1's continuation unless told: its events."""
        admission.setdefault("delegation_id", "D1")
        continuation = im.admit_client_request(app_outputs(events, output), **admission)
        return await asyncio.wait_for(read_all(continuation.stream), 2)

    def timeline(self) -> list[list[str]]:
        """The wire, as the endings fixture writes it: a handover line is marked whichever taker it names."""
        events = []
        for response_id, event in self.wire.events:
            item = event.get("item") or {}
            if item.get("type") == "message" and is_handover_line(message_text(item)):
                event = {**event, "item": message_item(TEXTS.handover_line, item["phase"])}
            events.append((response_id, event))
        return normalized(events, FIXTURE_LINES)
