# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""The core's test doubles: rounds and their rules, a task's context, scripted models, adapters and conductors."""

from __future__ import annotations

import asyncio
import json
import unittest
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from unittest import mock

from examples.frontend_backend_live.delegation_core.backend.conductors import SteerOpenWork
from examples.frontend_backend_live.delegation_core.backend.llm import LlmSettings, LlmSource
from examples.frontend_backend_live.delegation_core.backend.port import (
    BoundaryInput,
    Request,
    Start,
    Steering,
    TaskSummary,
    WorkSummary,
    message_phase,
)
from examples.frontend_backend_live.delegation_core.backend.providers import Backend, ProviderLimits
from examples.frontend_backend_live.delegation_core.config import CoreSettings
from examples.frontend_backend_live.delegation_core.im.input import BackendTask
from examples.frontend_backend_live.delegation_core.im.session import IMSession
from examples.frontend_backend_live.delegation_core.im.stream import ResponseWriter
from examples.frontend_backend_live.delegation_core.items import message_text
from examples.shared.delegation_contract import TERMINALS, ModelEndpoint
from tests.unit.core_helpers import CORE_BLOCK, CORE_TEXTS, Sleep, normalized

CORE_SETTINGS = CoreSettings.parse(CORE_BLOCK, CORE_TEXTS)
TEXTS = CORE_SETTINGS.texts
DEFAULTS = BackendTask("test/model", "Do the task.", [])
MENU = {"type": "function", "name": "get_menu", "parameters": {"type": "object", "properties": {}}}
STEERING = [{"type": "message", "role": "user", "content": "Make it two."}]
ENDPOINT = ModelEndpoint(model="test/model", base_url="http://localhost:1/v1")
LIMITS = ProviderLimits(provider_retries=1, first_event_timeout_seconds=2.0, idle_timeout_seconds=2.0)
NO_RETRY = ProviderLimits(provider_retries=0, first_event_timeout_seconds=2.0, idle_timeout_seconds=2.0)
FAST = LlmSettings(
    first_event_timeout_seconds=0.5,
    idle_timeout_seconds=0.5,
    server_tool_timeout_seconds=1.0,
    stop_timeout_seconds=0.5,
    provider_retries=0,
)
TABLE_LINES = {key: getattr(TEXTS, key) for key in ("handover_line", "cancelled_line", "unconfirmed_line")}


def is_handover_line(text):
    """Return whether ``text`` is the handover line with a taker's delegation id in its ``{delegation}`` field."""
    head, _, tail = TEXTS.handover_line.partition("{delegation}")
    return len(text) > len(head) + len(tail) and text.startswith(head) and text.endswith(tail)


def ask(text):
    return [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}]


def message_item(text, phase, item_id="msg"):
    item = {"id": item_id, "type": "message", "role": "assistant", "status": "completed", "phase": phase}
    return {**item, "content": [{"type": "output_text", "text": text, "annotations": []}]}


def call_item(name, arguments, call_id):
    item = {"id": f"fc_{call_id}", "type": "function_call", "status": "completed", "call_id": call_id, "name": name}
    return {**item, "arguments": json.dumps(arguments)}


async def read_all(stream):
    return [event async for event in stream]


def kinds(events):
    return [event["type"] for event in events]


def finished_items(events):
    return [event["item"] for event in events if event["type"] == "response.output_item.done"]


def items_of(events, item_type):
    return [item for item in finished_items(events) if item["type"] == item_type]


def app_outputs(events, output="{}"):
    calls = items_of(events, "function_call")
    return [{"type": "function_call_output", "call_id": call["call_id"], "output": output} for call in calls]


def said(events):
    """Return the phase and text of each message the events finished, in order."""
    return [(item["phase"], message_text(item)) for item in items_of(events, "message")]


def texts(events):
    return [text for _, text in said(events)]


def answers(events, phase="final_answer"):
    return [text for kind, text in said(events) if kind == phase]


def function_calls(events):
    return [item["call_id"] for item in items_of(events, "function_call")]


def answer_text(event):
    """Return the last ``final_answer`` text of a ``response.completed`` with no function call, or ``None``."""
    output = (event.get("response") or {}).get("output") or []
    if event.get("type") != "response.completed" or any(item.get("type") == "function_call" for item in output):
        return None
    messages = [
        item
        for item in output
        if item.get("type") == "message"
        and item.get("status") == "completed"
        and item.get("phase", "final_answer") == "final_answer"
    ]
    return message_text(messages[-1]) if messages else None


def records_of(message):
    """Return the work records a request's message carries, one per line of its records block."""
    head = TEXTS.records_block.split("{records}")[0]
    (block,) = [part["text"] for part in message["content"] if part["text"].startswith(head)]
    return [json.loads(line) for line in block.splitlines()[1:]]


def message_events(text, *, item_id="msg_1", index=0, phase=None):
    """Build the events of one whole assistant message: announced, its text, finished (``phase`` only when given)."""
    item = {"id": item_id, "type": "message", "status": "completed", "role": "assistant"}
    item["content"] = [{"type": "output_text", "text": text, "annotations": []}]
    if phase is not None:
        item["phase"] = phase
    opened = {**item, "status": "in_progress", "content": []}
    delta = {"item_id": item_id, "output_index": index, "content_index": 0, "delta": text}
    return [
        {"type": "response.output_item.added", "output_index": index, "item": opened},
        {"type": "response.output_text.delta", **delta},
        {"type": "response.output_item.done", "output_index": index, "item": item},
    ]


def call_events(name, arguments, *, call_id="call_1", index=0):
    item = call_item(name, arguments, call_id)
    return [
        {"type": "response.output_item.added", "output_index": index, "item": {**item, "status": "in_progress"}},
        {"type": "response.output_item.done", "output_index": index, "item": item},
    ]


def a_round(*parts, response_id="resp_1"):
    events = [{"type": "response.created", "response": {"id": response_id, "status": "in_progress"}}]
    for part in parts:
        events += part
    completed = {"id": response_id, "status": "completed", "output": finished_items(events)}
    return [*events, {"type": "response.completed", "response": completed}]


def text_round(text, response_id="resp_1"):
    """Build a round that answers ``text`` with no phase, as a chat model gives it."""
    return a_round(message_events(text), response_id=response_id)


def call_round(name, arguments, call_id="call_1", *, said=None):
    parts = [message_events(said, index=0)] if said else []
    parts.append(call_events(name, arguments, call_id=call_id, index=len(parts)))
    return a_round(*parts, response_id=f"resp_{call_id}")


def check_events(events):
    """Return the port's event rules that one task's events break: attempts, items, phases, calls last, endings."""
    problems, attempt, ended = [], None, False
    for index, event in enumerate(events):
        kind = event.get("type", "")
        if kind == "response.created":
            if attempt is not None and not ended:
                problems.append(f"event {index} opens an attempt before the last one ended")
            attempt, ended = [], False
        elif attempt is None or ended:
            problems.append(f"event {index} ({kind}) is outside an attempt")
        else:
            attempt.append(event)
            if kind in TERMINALS:
                ended = True
                problems += _attempt_problems(attempt)
    if attempt is not None and not ended:
        problems.append("the events ended inside an attempt")
    if events and events[-1].get("type") != "response.completed":
        problems.append(f"the events ended after {events[-1].get('type')}, not after a completed round")
    return problems


def _attempt_problems(attempt):
    added = {event.get("item", {}).get("id") for event in attempt if event["type"] == "response.output_item.added"}
    items = finished_items(attempt)
    problems = [f"item {item.get('id')} was never announced" for item in items if item.get("id") not in added]
    calls_seen = False
    for item in items:
        if item.get("type") == "function_call":
            calls_seen = True
        elif item.get("type") == "message":
            if calls_seen:
                problems.append("a message follows a call in its round")
            if item.get("phase") not in ("commentary", "final_answer"):
                problems.append(f"message {item.get('id')} has no phase")
    if calls_seen and any(item.get("type") == "message" and message_phase(item) == "final_answer" for item in items):
        problems.append("a round with calls carries a final_answer")
    return problems


class NoTools:
    label = "delegation_core"

    def tools(self):
        return []

    async def execute(self, call_id, name, arguments):
        return json.dumps({"error": f"Unknown tool {name}"})


class FakeTaskContext:
    """A ``TaskContext`` that serves a request and a history, and records every report.

    The app's functions are those the request declares; every other call is the backend's. The test hands over the
    app's outputs with :meth:`give_outputs` and steering with :meth:`steer_with`; :meth:`see` reads a run's event as
    the IM does, so the outputs of the calls it saw join at the next safe point.
    """

    def __init__(
        self,
        *,
        instructions="COMPOSED-INSTRUCTIONS",
        history=None,
        declared=(MENU,),
        tools=None,
        model="test/model",
        work_id="W1",
        task_id="W1.1",
    ):
        self.work_id, self.task_id, self.model, self.texts = work_id, task_id, model, TEXTS
        self.instructions, self.declared = instructions, list(declared)
        self.items = list(history or [{"type": "message", "role": "user", "content": "Do the task."}])
        self.work_tools = tools or NoTools()
        self.refused, self.calls_seen, self.requests, self.traces = set(), [], [], []
        self.started, self.outcomes, self.order, self.settled = [], {}, [], []
        self._owed, self._outputs, self._steering = {}, [], deque()
        self.tools_closed = False  # rule E9: the IM never closed this task's tools

    def give_outputs(self, outputs):
        self._outputs += list(outputs)

    def steer_with(self, items, steering_id="steer_1"):
        """Queue steering for the next safe point, behind any batch waiting; the test then calls ``steer()``."""
        self._steering.append(Steering(steering_id, tuple(items)))
        return self._steering[-1]

    def see(self, event):
        if event["type"] == "response.completed":
            output = event["response"]["output"]
            self.items += output
            self.calls_seen += [item["call_id"] for item in output if item.get("type") == "function_call"]

    def request(self):
        tools = list(self.declared)
        self.requests.append(
            {"model": self.model, "instructions": self.instructions, "tools": tools, "tool_choice": "auto"}
        )
        return self.requests[-1]

    def history(self):
        return list(self.items)

    def tools(self):
        return self.work_tools

    def route(self, call):
        if call.get("call_id") in self.refused:
            return "refused"
        return "app" if call.get("name") in {tool.get("name") for tool in self.declared} else "backend"

    def refusal(self, call_id):
        return self.texts.not_run_output

    def tool_started(self, call, source):
        self.started.append((call.get("call_id", ""), source))
        self.order.append(f"started:{call.get('call_id')}")

    def tool_outcome(self, call_id, output, outcome):
        self.outcomes[call_id] = (output, outcome)
        self.order.append(f"outcome:{call_id}")
        self._owed[call_id] = output

    def progress(self, text):
        pass

    def take_boundary_input(self):
        owed = [call_id for call_id in self.calls_seen if call_id in self._owed]
        joined = [
            {"type": "function_call_output", "call_id": call_id, "output": self._owed.pop(call_id)} for call_id in owed
        ]
        outputs, self._outputs = (*joined, *self._outputs), []
        self.items += list(outputs)
        return BoundaryInput(outputs, self._steering.popleft() if self._steering else None)

    def settle(self, steering, delivery):
        self.settled.append((steering.id, delivery))
        if delivery == "applied":
            self.items += list(steering.items)

    def trace(self, kind, **fields):
        self.traces.append((kind, fields))


class FakeDecisionTarget:
    def __init__(self):
        self.decisions, self.works = [], 0

    async def start(self, decision):
        self.decisions.append(decision)
        self.works += 1
        return f"Started W{self.works} with {max(1, len(decision.tasks))} task(s)."

    async def steer(self, decision):
        self.decisions.append(decision)
        return f"Sent to {decision.work_id}; it applies at the next step."

    async def cancel(self, decision):
        self.decisions.append(decision)
        return f"Stopping {decision.work_id}."

    async def answer(self, decision):
        self.decisions.append(decision)
        return f"Answered {decision.delegation_id}."

    async def place(self, decision):
        return await decision.apply(self)


async def decide(conductor, given):
    target = FakeDecisionTarget()
    await asyncio.wait_for(conductor.decide(given, target.place), 5)
    return target.decisions


def a_request(delegation_id, text):
    return Request(delegation_id, ({"type": "message", "role": "user", "content": text},))


def an_open_work(work_id="W1", delegation_id="d_old"):
    task = TaskSummary(f"{work_id}.1", "running", None, ("get_menu: done",), True, "Looking it up.", 3.0)
    return WorkSummary(work_id, "running", delegation_id, (a_request(delegation_id, "Find teas."),), (task,), 3.0)


def an_ended_work(work_id="W0", delegation_id="d_done"):
    task = TaskSummary(f"{work_id}.1", "answered", None, ("book_table: done",), True, None, 9.0)
    return WorkSummary(work_id, "answered", delegation_id, (a_request(delegation_id, "Book a table."),), (task,), 9.0)


class Hang:
    """A pause in a scripted round that ignores being cancelled: a provider that does not let go."""

    def __init__(self, seconds):
        self.seconds = seconds


class ScriptedProvider(Backend):
    """A provider that plays one script per round, or ``respond(payload)``'s: events, exceptions, pauses or gates."""

    name = "scripted"

    def __init__(self, *rounds, respond=None, limits=NO_RETRY, endpoint=ENDPOINT):
        super().__init__(endpoint, limits)
        self.rounds, self.respond, self.payloads = list(rounds), respond, []
        self.opened = self.closed = 0
        self.read_after_terminal = False

    async def _events(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        self.opened += 1
        ended = False
        for step in self.respond(payload) if self.respond is not None else self.rounds.pop(0):
            self.read_after_terminal = self.read_after_terminal or ended
            ended = ended or (isinstance(step, dict) and step.get("type") in TERMINALS)
            if isinstance(step, BaseException):
                raise step
            if isinstance(step, Sleep):
                await asyncio.sleep(step.seconds)
            elif isinstance(step, Hang):
                hang = asyncio.ensure_future(asyncio.sleep(step.seconds))
                while not hang.done():
                    try:
                        await asyncio.shield(hang)
                    except asyncio.CancelledError:
                        continue
            elif isinstance(step, asyncio.Event):
                await step.wait()
            else:
                yield step

    async def close(self):
        self.closed += 1


def gated(gate, round_):
    """Return ``round_`` held at ``gate`` after its first event, streaming until the test sets the gate."""
    return [round_[0], gate, *round_[1:]]


class RecordingTools:
    """A server tool source whose tools answer ``{"ok": true}``, slowly when asked, and record their calls."""

    label = "shop"

    def __init__(self, names=("lookup",), *, delay=0.0, fail=()):
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


async def drive(run, task):
    """Read a run's events as the IM would, showing each to ``task``."""
    events = []
    async for event in run.events():
        task.see(event)
        events.append(event)
    return events


def source(*providers, settings=FAST, tools=None):
    """Build an ``LlmSource`` whose providers are ``providers``, made in turn, one per model a session asks for."""
    made = list(providers)
    return LlmSource(
        settings,
        endpoints=lambda model: ModelEndpoint(model=model or ENDPOINT.model, base_url=ENDPOINT.base_url),
        tools=lambda: list(tools or ()),
        make_provider=lambda endpoint: made.pop(0),
    )


def adapter(*providers, **kwargs):
    return source(*providers, **kwargs).new()


class ScriptStep:
    """One step of a scripted task. ``ends``: the task's events stop after it (an answer, or the app's calls)."""

    ends = False

    async def play(self, run):
        if False:
            yield {}


@dataclass
class Act(ScriptStep):
    """A step with no round: ``act(run)`` is done when the run comes to it."""

    act: object

    async def play(self, run):
        self.act(run)
        if False:
            yield {}


@dataclass
class Attempt(ScriptStep):
    """An attempt of a round of final ``answers`` that ends ``terminal``; then the run raises ``raises``, if given.

    An attempt that is retried (``ends=False``) leaves the task's events open.
    """

    answers: tuple[str, ...] = ()
    terminal: tuple[str, dict] | None = None
    raises: Exception | None = None
    usage: dict | None = None
    ends: bool = True

    async def play(self, run):
        items = [run.message(text, "final_answer") for text in self.answers]
        for event in run.round_events(items, usage=self.usage, terminal=self.terminal):
            yield event
        if self.raises is not None:
            raise self.raises


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
    """A round that calls the backend's own tool ``name``, which the run runs and reports (once ``gate`` is set)."""

    name: str = "lookup"
    output: str = '{"ok": true}'
    fails: bool = False
    label: str = "cafe"
    call_id: str | None = None
    gate: asyncio.Event | None = None

    async def play(self, run):
        call = call_item(self.name, {}, self.call_id or f"call_{run.task.task_id}_{run.round + 1}")
        for event in run.round_events([call]):
            yield event
        route = run.task.route(call)
        run.routes.append(route)
        if route == "backend":
            run.task.tool_started(call, self.label)
            if self.gate is not None:
                await self.gate.wait()
            run.task.tool_outcome(call["call_id"], self.output, "failed" if self.fails else "done")


@dataclass
class WorkTool(ScriptStep):
    """A round that calls the IM's work tool ``name``, run through the task's tools; then a round of ``later`` calls."""

    name: str
    arguments: dict
    later: tuple[dict, ...] = ()

    async def play(self, run):
        call = call_item(self.name, self.arguments, f"call_{run.task.task_id}_{self.name}_{run.round + 1}")
        for event in run.round_events([call]):
            yield event
        run.routes.append(run.task.route(call))
        tools = run.task.tools()
        run.task.tool_started(call, tools.label)
        output = await tools.execute(call["call_id"], self.name, call["arguments"])
        run.task.tool_outcome(call["call_id"], output, "done")
        if self.later:
            for event in run.round_events(list(self.later)):
                yield event
            run.routes += [run.task.route(later) for later in self.later]


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
class Fail(ScriptStep):
    error: Exception = field(default_factory=lambda: RuntimeError("The backend failed"))

    async def play(self, run):
        raise self.error
        yield {}


def answer(text="Done.", usage=None):
    return Attempt((text,), usage=usage)


def say(text):
    """Return a step that says ``text`` as commentary in the task's next round."""
    return Act(lambda run: run.said.append(text))


def progress(text):
    """Return a step whose run reports ``text`` as progress, which is not model text."""
    return Act(lambda run: run.task.progress(text))


def checkpoint():
    """Return a safe point with no round, where the run takes the boundary input and settles any steering."""
    return Act(lambda run: run.boundary())


def app_call(name="get_menu", arguments=None, call_id=None):
    """Return a round that calls the app's function ``name``; the task's events end until it is resumed."""
    return _AppCall(name, arguments or {}, call_id)


def wait(event):
    """Return a step that pauses until ``event`` is set or the run is stopped."""
    return _Wait(event)


class ScriptedRun:
    def __init__(self, adapter, task, steps):
        self.adapter, self.task, self.steps = adapter, task, list(steps)
        self.round, self.said, self.stopping = 0, [], asyncio.Event()
        self.resumes = self.stops = 0
        self.requests, self.histories, self.boundaries, self.routes = [], [], [], []

    def message(self, text, phase):
        return message_item(text, phase, f"msg_{self.task.task_id}_{self.round + 1}_{len(text)}")

    def boundary(self):
        given = self.task.take_boundary_input()
        self.boundaries.append(given)
        if given.steering is not None:
            self.task.settle(given.steering, self.adapter.delivery)

    def round_events(self, items, *, usage=None, terminal=None, boundary=True):
        """Build one attempt of a round: its commentary, then ``items``, after a safe point if ``boundary``."""
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
        events.append(
            {"type": f"response.{kind}", "response": {"id": response_id, "status": kind, "output": items, **fields}}
        )
        return events

    async def events(self):
        while self.steps and not self.stopping.is_set():
            step = self.steps.pop(0)
            async for event in step.play(self):
                if self.stopping.is_set():
                    return
                yield event
            if step.ends:
                return

    def steer(self):
        pass

    async def resume(self):
        self.resumes += 1
        self.boundary()

    async def stop(self):
        self.stops += 1
        self.stopping.set()
        if self.adapter.stop_delay:
            await asyncio.sleep(self.adapter.stop_delay)
        return self.adapter.stop_outcome


class ScriptedAdapter:
    """A backend adapter whose tasks each play the next script (steps, or a function of the task) in a ``run``."""

    def __init__(
        self,
        *scripts,
        max_running=1,
        conductor=None,
        delivery="applied",
        stop_outcome="stopped",
        stop_delay=0.0,
        run=ScriptedRun,
    ):
        self.scripts, self.run, self.max_running = list(scripts), run, max_running
        self._conductor = conductor if conductor is not None else SteerOpenWork()
        self.delivery, self.stop_outcome, self.stop_delay = delivery, stop_outcome, stop_delay
        self.runs, self.closed = [], False

    def conductor(self):
        return self._conductor

    def start(self, task):
        script = self.scripts.pop(0) if self.scripts else [answer()]
        self.runs.append(self.run(self, task, script(task) if callable(script) else script))
        return self.runs[-1]

    async def close(self):
        self.closed = True


class StartEach:
    """A conductor that starts a Work of its own for every arrival, which waits its turn for a slot."""

    async def decide(self, given, place):
        for arrival in given.arrivals:
            await place(Start((arrival.delegation_id,)))


@dataclass
class ScriptedConductor:
    """A conductor that hands every ``decide()`` to ``plan(given, place)``; it keeps what it was given and told."""

    plan: object
    given: list = field(default_factory=list)
    outcomes: list = field(default_factory=list)

    async def decide(self, given, place):
        self.given.append(given)

        async def recording(decision):
            self.outcomes.append(await place(decision))
            return self.outcomes[-1]

        await self.plan(given, recording)


def planned(*plans):
    waiting = list(plans)

    async def plan(given, place):
        await waiting.pop(0)(given, place)

    return ScriptedConductor(plan)


def placing(*decisions):

    async def plan(given, place):
        for decision in decisions:
            await place(decision)

    return plan


class Wire:
    """Everything the IM writes, in order, as ``(response_id, event)`` with ``response_id`` added to each event."""

    def __init__(self):
        self.events = []

    @contextmanager
    def recording(self):
        original = ResponseWriter._emit

        def emit(writer, kind, **fields):
            original(writer, kind, **fields)
            self.events.append((writer.id, {"type": kind, **deepcopy(fields), "response_id": writer.id}))

        with mock.patch.object(ResponseWriter, "_emit", emit):
            yield self


class IMCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.wire = self.enterContext(Wire().recording())
        self.traces = []

    def session(self, adapter, *, texts=TEXTS, **changes):
        """Start an IM session on ``adapter`` with the core's settings, which ``changes`` override, and ``texts``.

        ``changes`` may also give the session's ``result_format`` and ``output_steps``.
        """
        options = {name: changes.pop(name) for name in ("result_format", "output_steps") if name in changes}
        # A test's last close does not wait for the runs it left hanging.
        changes.setdefault("close_grace_seconds", 0.01)
        settings = CORE_SETTINGS.delegation.model_copy(update=changes)
        observe = lambda kind, **fields: self.traces.append((kind, fields))  # noqa: E731
        im = IMSession(adapter, DEFAULTS, settings, texts, **options, observe=observe)
        im.start()
        self.addAsyncCleanup(im.close)
        return im

    async def turns(self, count=30):
        for _ in range(count):
            await asyncio.sleep(0)

    async def until(self, predicate, what, timeout=4.0):
        try:
            async with asyncio.timeout(timeout):
                while not predicate():
                    await asyncio.sleep(0.01)
        except TimeoutError:
            raise AssertionError(f"timed out waiting for {what}") from None

    async def admit_each(self, im, *requests, settings=None):
        """Admit one request per text as D1, D2 ..., letting everything ready run after each; return their streams."""
        streams = []
        for number, text in enumerate(requests, 1):
            streams.append(im.admit(ask(text), delegation_id=f"D{number}", settings=settings).stream)
            await self.turns()
        return streams

    async def answer_calls(self, im, events, output="{}", **admission):
        """Answer each call ``events`` ended with by ``output``, as D1's continuation unless told; return its events."""
        admission.setdefault("delegation_id", "D1")
        continuation = im.admit_client_request(app_outputs(events, output), **admission)
        return await asyncio.wait_for(read_all(continuation.stream), 2)

    def timeline(self):
        """Return the wire as the endings table writes it, a handover line marked whichever taker it names."""
        events = []
        for response_id, event in self.wire.events:
            item = event.get("item") or {}
            if item.get("type") == "message" and is_handover_line(message_text(item)):
                event = {**event, "item": message_item(TEXTS.handover_line, item["phase"])}
            events.append((response_id, event))
        return normalized(events, TABLE_LINES)
