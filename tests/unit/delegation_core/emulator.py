# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A test-only backend adapter with a scripted conductor and scripted workers, for testing the IM without a model.

It implements the backend port and passes both of its contracts (``tests/unit/contracts/backend.py``). A test builds it
with ``EmulatorSource.from_settings(...)`` or injects it into ``build_core(..., adapters={"emulator": ...})``.

**Scripts.** A request's text carries directives in square brackets; the words around them are the request.

The conductor reads each arrival's text and places the arrival:

* ``[start]``: a new Work for it, which is also what an arrival with no placing directive gets;
* ``[start:A|B]``: a new Work with one task per brief; a brief may hold worker directives: ``[start:[own:a]|[slow:1]]``;
* ``[steer]``: join the newest open Work, or start one when none is open; ``[steer:W1]`` joins every open task of W1;
  ``[steer:W1.2]`` joins its task W1.2;
* ``[cancel:W1]`` stops Work W1, ``[cancel:W1.1]`` its task W1.1; neither places the arrival itself;
* ``[answer:text]``: answer the arrival from the records;
* ``[skip]``: place nothing for it (the IM starts it once ``decide()`` returns);
* ``[think:N]``: wait N seconds first (the busy-conductor rule: one ``decide()`` at a time);
* ``[raise]``: ``decide()`` raises.

A task's run reads its script when it starts: the block's ``script``, then the directives of the first text of the
newest user message of its history (the task's brief when the IM gives it as a message of its own, otherwise the newest
request):

* ``[client:fn {json}]`` calls ``fn`` with these arguments, ``[own:tool]`` calls ``tool``, ``[own-fail:tool]`` calls
  ``tool``, which fails, and ``[show:text]`` calls the IM's ``show`` work tool. ``route()`` decides who runs each call:
  ``app`` ends the round and the iteration, so the task waits for the app; ``backend`` runs it here (one of the IM's
  work tools through ``tools()``, any other as the emulator's own tool, under the block's ``label``); ``refused`` is not
  run. Consecutive calls share a round;
* ``[say:text]``: a ``commentary`` message in the round, before its calls;
* ``[progress:text]``: ``progress(text)``, a note that is not model text;
* ``[slow:N]``: the round waits N seconds after it opened; ``release()``, ``steer()`` and ``stop()`` end the wait;
* ``[fail]``: the run raises; ``[fail:429]`` raises an error whose ``status_code`` is 429;
* ``[empty]``: the last round completes with no items;
* ``[reply:text]``: the answer's words. Without it the answer is ``Done: <request>``, the newest request's words.

**Rounds.** Each round opens at a safe point: ``take_boundary_input()`` hands over the app's outputs and waiting
steering, which the run settles as the block's ``delivery`` says (an applied steering's own directives join the
script); then it reads ``request()`` and ``history()``. A ``steer()`` during a ``[slow]`` wait, before the round ran
a call, ends that attempt ``response.incomplete`` (reason ``steered``), and the round runs again with the steering;
the dropped attempt's items go with it. Messages carry ``phase``; calls come last; every call it runs is reported
before the round's terminal; call ids are unique per source (``call_1``, ``call_2`` ...). ``stop()`` ends any wait at
once, nothing follows it, and it returns the block's ``stop_outcome`` after ``stop_delay_seconds``.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import aclosing
from dataclasses import dataclass, field, fields
from itertools import count
from types import MappingProxyType
from typing import Any, get_args

from delegation_core.backend.port import (
    Answer,
    BoundaryInput,
    Cancel,
    ConductorInput,
    ControlTools,
    Decision,
    Delivery,
    EndpointLookup,
    OpenState,
    Place,
    Request,
    Start,
    Steer,
    Steering,
    StopOutcome,
    TaskContext,
    ToolSources,
)

# One directive: ``[name]`` or ``[name:argument]``; an argument may hold bracketed directives one level deep.
DIRECTIVE = re.compile(r"\[([a-z][a-z-]*)(?::((?:[^\[\]]|\[[^\[\]]*\])*))?\]")
# The worker's directives that are calls, and the directives a worker reads (the conductor's are skipped).
CALLS = frozenset({"client", "own", "own-fail", "show"})
WORKER = CALLS | {"say", "progress", "slow", "fail", "empty", "reply"}
# The step a script ends with: the task's answer.
ANSWER = ("answer", None)


def directives(text: str) -> list[tuple[str, str | None]]:
    """The directives in ``text``, in order, as ``(name, argument)``; ``argument`` is ``None`` without a colon."""
    return [(match[1], match[2]) for match in DIRECTIVE.finditer(text)]


def words(text: str) -> str:
    """``text`` without its directives, its spaces collapsed."""
    return " ".join(DIRECTIVE.sub(" ", text).split())


def texts(item: dict) -> list[str]:
    """The texts of an input item's content, in order: a string content, or its text parts."""
    content = item.get("content")
    if isinstance(content, str):
        return [content]
    return [part["text"] for part in content or [] if isinstance(part, dict) and isinstance(part.get("text"), str)]


def is_user_message(item: dict) -> bool:
    """Whether an input item is a user message (a message may leave its ``type`` out)."""
    return item.get("type", "message") == "message" and item.get("role") == "user"


def newest_request(items: list[dict] | tuple[dict, ...]) -> str:
    """The first text of the newest user message: the newest request, or a task's brief; ``""`` when there is none.

    Only the first text counts, so the records block the IM adds as a second part is never read as a script.
    """
    for item in reversed(items):
        if is_user_message(item):
            return next(iter(texts(item)), "")
    return ""


def briefs(argument: str) -> tuple[str, ...]:
    """A ``start`` argument split at its top-level ``|``: one brief per task."""
    parts, depth, current = [], 0, ""
    for character in argument:
        depth += {"[": 1, "]": -1}.get(character, 0)
        if character == "|" and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += character
    return (*parts, current)


class EmulatedFailure(RuntimeError):
    """The failure ``[fail]`` raises; ``[fail:429]`` gives it the upstream status code a provider error carries."""

    def __init__(self, message: str, status_code: int | None = None):
        """Fail with ``message``, and ``status_code`` when the script names one."""
        super().__init__(message)
        self.status_code = status_code


# ----------------------------------------------------------------------------------------------- the source
@dataclass(frozen=True)
class EmulatorSettings:
    """The emulator's settings block, as a core's settings would hold it (``backend.emulator``)."""

    max_running: int = 1  # tasks that may run at once in one IM session
    delivery: Delivery = "applied"  # how a run settles the steering it takes
    stop_delay_seconds: float = 0.0  # how long a run takes to stop
    stop_outcome: StopOutcome = "stopped"  # what a stop confirms once that time has passed
    script: str = ""  # directives every task runs before its own
    label: str = "emulator"  # the server label of the calls it runs itself
    models: tuple[str, ...] = ()  # the models it serves; empty: any
    unsupported: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))  # param -> why

    @classmethod
    def parse(cls, block: Mapping[str, Any]) -> EmulatorSettings:
        """Validate the block; unknown keys and wrong values are refused, as every adapter's block is.

        Raises:
            ValueError: An unknown key or a value of the wrong kind.
        """
        unknown = set(block) - {each.name for each in fields(cls)}
        if unknown:
            raise ValueError(f"emulator has unknown keys: {', '.join(sorted(map(str, unknown)))}")
        values = dict(block)
        try:
            if "models" in values:
                values["models"] = tuple(values["models"])
            if "unsupported" in values:
                values["unsupported"] = MappingProxyType(dict(values["unsupported"]))
        except (TypeError, ValueError) as error:
            raise ValueError(f"emulator.models must be a list and emulator.unsupported a mapping: {error}") from None
        settings = cls(**values)
        running = settings.max_running
        if isinstance(running, bool) or not isinstance(running, int) or running < 1:
            raise ValueError("emulator.max_running must be a whole number of at least 1")
        if settings.delivery not in get_args(Delivery):
            raise ValueError(f"emulator.delivery must be one of {', '.join(get_args(Delivery))}")
        if settings.stop_outcome not in get_args(StopOutcome):
            raise ValueError(f"emulator.stop_outcome must be one of {', '.join(get_args(StopOutcome))}")
        delay = settings.stop_delay_seconds
        if isinstance(delay, bool) or not isinstance(delay, int | float) or delay < 0:
            raise ValueError("emulator.stop_delay_seconds must be a non-negative number of seconds")
        named = [
            settings.script,
            settings.label,
            *settings.models,
            *settings.unsupported,
            *settings.unsupported.values(),
        ]
        if not all(isinstance(each, str) for each in named):
            raise ValueError("emulator.script, label, models and unsupported hold text")
        return settings


class EmulatorSource:
    """An ``AdapterSource`` and a ``OneShotSource`` for tests; its runs never retry, as one-shot sessions need."""

    def __init__(self, settings: EmulatorSettings):
        """Serve IM sessions with these settings."""
        self.settings = settings
        self.adapters: list[EmulatorAdapter] = []  # every adapter it gave, in order
        self.released = asyncio.Event()
        self.closed = False
        self._calls = count(1)
        self._responses = count(1)

    @classmethod
    def from_settings(
        cls,
        block: Mapping[str, Any],
        *,
        endpoints: EndpointLookup,
        tools: ToolSources,
        vocabulary: ControlTools,
    ) -> EmulatorSource:
        """Build it from its block; it uses no endpoint, tool source or control vocabulary.

        Raises:
            ValueError: The block is refused (:meth:`EmulatorSettings.parse`).
        """
        return cls(EmulatorSettings.parse(block))

    def describe(self) -> dict:
        """What it serves, as plain JSON."""
        settings = self.settings
        return {"adapter": "emulator", "max_running": settings.max_running, "models": list(settings.models)}

    async def check_model(self, model: str) -> None:
        """Raise ``ValueError`` for a model outside ``models``, when the block lists any."""
        if self.settings.models and model not in self.settings.models:
            raise ValueError(f"The emulator serves {', '.join(self.settings.models)}; not {model!r}")

    def unsupported(self, settings: Mapping[str, Any]) -> list[tuple[str, str]]:
        """The request settings the block says it cannot honor, as ``(param, message)``, in the request's order."""
        return [(param, self.settings.unsupported[param]) for param in settings if param in self.settings.unsupported]

    def new(self) -> EmulatorAdapter:
        """The backend of a new IM session."""
        adapter = EmulatorAdapter(self)
        self.adapters.append(adapter)
        return adapter

    def new_one_shot(self) -> EmulatorAdapter:
        """The backend of one stateless request's one-shot session: the same adapter, whose runs never retry."""
        return self.new()

    async def aclose(self) -> None:
        """Release what it holds: nothing, so it only notes the close."""
        self.closed = True

    def release(self) -> None:
        """End every ``[slow]`` wait now, and every later one at once."""
        self.released.set()

    def call_id(self) -> str:
        """A call id no other call of this source has."""
        return f"call_{next(self._calls)}"

    def response_id(self) -> str:
        """A backend response id for one attempt."""
        return f"resp_emulator_{next(self._responses)}"


class EmulatorAdapter:
    """One IM session's emulated backend: its scripted conductor and one scripted run per task."""

    def __init__(self, source: EmulatorSource):
        """Run tasks with ``source``'s settings."""
        self.source = source
        self.max_running = source.settings.max_running
        self.runs: list[EmulatorRun] = []  # every run it started, in order
        self.closed = False
        self._conductor = EmulatorConductor()

    def conductor(self) -> EmulatorConductor:
        """The session's conductor: the same one on every call."""
        return self._conductor

    def start(self, task: TaskContext) -> EmulatorRun:
        """Run one task; nothing runs before ``events()``."""
        run = EmulatorRun(self.source, task)
        self.runs.append(run)
        return run

    async def close(self) -> None:
        """Note the close."""
        self.closed = True


# --------------------------------------------------------------------------------------------- the conductor
def _arrival_text(arrival: Request) -> str:
    return " ".join(text for item in arrival.input if is_user_message(item) for text in texts(item))


def _start(arrival: Request, argument: str | None, given: ConductorInput) -> Decision:
    return Start((arrival.delegation_id,), briefs(argument) if argument else ())


def _steer(arrival: Request, argument: str | None, given: ConductorInput) -> Decision:
    if argument is None:  # the newest open Work: the summaries list the open Works, then recently ended ones
        open_works = [work for work in given.works if work.state in get_args(OpenState)]
        if not open_works:
            return Start((arrival.delegation_id,))
        return Steer(open_works[-1].work_id, (arrival.delegation_id,))
    work_id, _, task = argument.partition(".")
    return Steer(work_id, (arrival.delegation_id,), argument if task else None)


def _cancel(arrival: Request, argument: str | None, given: ConductorInput) -> Decision:
    work_id, _, task = (argument or "").partition(".")
    return Cancel(work_id, argument if task else None)


def _answer(arrival: Request, argument: str | None, given: ConductorInput) -> Decision:
    return Answer(arrival.delegation_id, argument or "")


# Each directive that makes a decision, and whether that decision places the arrival.
DECISIONS: dict[str, tuple[Callable[[Request, str | None, ConductorInput], Decision], bool]] = {
    "start": (_start, True),
    "steer": (_steer, True),
    "cancel": (_cancel, False),
    "answer": (_answer, True),
}


class EmulatorConductor:
    """The scripted conductor: it reads each arrival's directives and places the arrival through ``place``."""

    def __init__(self):
        """Start with nothing given and nothing placed."""
        self.given: list[ConductorInput] = []  # what each decide() call was given, in order
        self.outcomes: list[str] = []  # each outcome line place() returned, in order

    async def decide(self, given: ConductorInput, place: Place) -> None:
        """Place each arrival as its directives say; an arrival with no placing directive is started.

        Raises:
            RuntimeError: An arrival says ``[raise]``.
        """
        self.given.append(given)
        for arrival in given.arrivals:
            placed = False
            for name, argument in directives(_arrival_text(arrival)):
                if name == "think":
                    await asyncio.sleep(float(argument or 0))
                elif name == "raise":
                    raise RuntimeError(f"The emulator's conductor failed on {arrival.delegation_id}")
                elif name == "skip":
                    placed = True
                elif name in DECISIONS:
                    make, places = DECISIONS[name]
                    self.outcomes.append(await place(make(arrival, argument, given)))
                    placed = placed or places
            if not placed:
                self.outcomes.append(await place(Start((arrival.delegation_id,))))


# ------------------------------------------------------------------------------------------------- a task's run
def _message_events(item_id: str, index: int, text: str, phase: str) -> list[dict]:
    """One whole assistant message with its ``phase``, as the official events stream it."""
    done_part = {"type": "output_text", "text": text, "annotations": [], "logprobs": []}
    opened = {"id": item_id, "type": "message", "status": "in_progress", "role": "assistant", "phase": phase}
    where = {"item_id": item_id, "output_index": index, "content_index": 0}
    return [
        {"type": "response.output_item.added", "output_index": index, "item": {**opened, "content": []}},
        {"type": "response.content_part.added", **where, "part": {**done_part, "text": ""}},
        {"type": "response.output_text.delta", **where, "delta": text, "logprobs": []},
        {"type": "response.output_text.done", **where, "text": text, "logprobs": []},
        {"type": "response.content_part.done", **where, "part": done_part},
        {
            "type": "response.output_item.done",
            "output_index": index,
            "item": {**opened, "status": "completed", "content": [done_part]},
        },
    ]


def _call_events(call: dict, index: int) -> list[dict]:
    """One whole function call, as the official events stream it."""
    where = {"item_id": call["id"], "output_index": index}
    return [
        {"type": "response.output_item.added", "output_index": index, "item": {**call, "status": "in_progress"}},
        {"type": "response.function_call_arguments.delta", **where, "delta": call["arguments"]},
        {"type": "response.function_call_arguments.done", **where, "arguments": call["arguments"]},
        {"type": "response.output_item.done", "output_index": index, "item": call},
    ]


@dataclass
class _Round:
    """The attempt of a round that runs now."""

    response_id: str
    history: list[dict]  # what the task's history() gave at its start
    output: list[dict] = field(default_factory=list)  # its items, in order
    calls: bool = False  # it holds calls, so what follows them is the next round
    for_app: bool = False  # the app runs one of its calls: the iteration ends with it
    ran: bool = False  # it ran a call, so it is never run again

    def snapshot(self, status: str, **fields: Any) -> dict:
        return {"id": self.response_id, "object": "response", "status": status, "output": list(self.output), **fields}


async def _first_of(events: tuple[asyncio.Event, ...], seconds: float) -> None:
    """Wait until one of ``events`` is set, or ``seconds`` have passed."""
    waiters = [asyncio.ensure_future(event.wait()) for event in events]
    try:
        await asyncio.wait(waiters, timeout=seconds, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for waiter in waiters:
            waiter.cancel()


class EmulatorRun:
    """One task's scripted run: a ``BackendRun`` that keeps the port's event rules."""

    def __init__(self, source: EmulatorSource, task: TaskContext):
        """Run ``task`` with ``source``'s settings; the script is read when ``events()`` first runs."""
        self.source, self.task = source, task
        self.round = 0
        self.requests: list[dict] = []  # what request() gave at each round's start, redone attempts included
        self.settled: list[tuple[str, Delivery]] = []  # each steering batch it settled, in order
        self.resumes = 0
        self.stopped = False
        self.answered = False
        self.reply: str | None = None
        self._steps: deque[tuple[str, str | None]] | None = None
        self._round: _Round | None = None
        self._steering_waits = False
        self._interrupt = asyncio.Event()

    # ---------------------------------------------------------------------------------- the port
    async def events(self) -> AsyncIterator[dict]:
        """The task's events, round by round: until its answer, until a round of the app's calls, or by raising."""
        if self._steps is None:
            script = self.source.settings.script + " " + newest_request(self.task.history())
            self._steps = deque(step for step in directives(script) if step[0] in WORKER)
        while not (self.stopped or self.answered):
            name, argument = self._steps[0] if self._steps else ANSWER
            round_ = self._round
            if round_ is not None and round_.calls and name not in CALLS:
                yield self._terminal("response.completed")
                if round_.for_app:
                    return  # the task waits for the app's outputs: resume(), then events() again
                continue
            async with aclosing(self._step(name, argument)) as step_events:
                async for event in step_events:
                    if self.stopped:
                        return  # nothing follows a stop
                    yield event
            if self._steps and not self.stopped:
                self._steps.popleft()

    def steer(self) -> None:
        """Steering waits: a ``[slow]`` wait ends now, and the next safe point takes it."""
        self._steering_waits = True
        self._interrupt.set()

    async def resume(self) -> None:
        """The app's outputs are in; the next round takes them at its safe point."""
        self.resumes += 1

    async def stop(self) -> StopOutcome:
        """Stop now: any wait ends and nothing more is written; confirm after the block's delay."""
        self.stopped = True
        self._interrupt.set()
        if self.source.settings.stop_delay_seconds:
            await asyncio.sleep(self.source.settings.stop_delay_seconds)
        return self.source.settings.stop_outcome

    # ---------------------------------------------------------------------------------- the steps
    def _note(self, name: str, argument: str | None) -> None:
        """A step that writes no event: a note, the answer's words, or a failure."""
        if name == "progress":
            self.task.progress(argument or "")
        elif name == "reply":
            self.reply = argument or ""
        else:
            status = int(argument) if argument else None
            raise EmulatedFailure(f"Emulated failure of task {self.task.task_id}", status)

    async def _step(self, name: str, argument: str | None) -> AsyncIterator[dict]:
        """One step: a note writes no event; any other writes into the open round, opening one when none is open."""
        if name in ("progress", "reply", "fail"):
            self._note(name, argument)
            return
        if self._round is None:
            yield self._open()
        if name == "slow":
            async for event in self._slow(float(argument or 0)):
                yield event
        elif name == "say":
            for event in self._message(argument or "", "commentary"):
                yield event
        elif name in CALLS:
            for event in await self._call(name, argument):
                yield event
        else:  # the answer, or [empty]: the task's last round
            if name == "answer":
                for event in self._message(self._answer_text(), "final_answer"):
                    yield event
            self.answered = True
            yield self._terminal("response.completed")

    def _open(self, redo: bool = False) -> dict:
        """Open a round (or ``redo`` its attempt) at a safe point: take outputs and steering, read the task."""
        boundary: BoundaryInput = self.task.take_boundary_input()
        self._steering_waits = False
        if boundary.steering is not None:
            self._settle(boundary.steering)
        if not redo:
            self.round += 1
        self.requests.append(self.task.request())
        self._round = _Round(self.source.response_id(), self.task.history())
        return {"type": "response.created", "response": self._round.snapshot("in_progress")}

    def _settle(self, steering: Steering) -> None:
        delivery = self.source.settings.delivery
        self.task.settle(steering, delivery)
        self.settled.append((steering.id, delivery))
        if delivery == "applied":
            steps = [step for step in directives(newest_request(steering.items)) if step[0] in WORKER]
            self._steps.extend(steps)

    async def _slow(self, seconds: float) -> AsyncIterator[dict]:
        """Wait in the open round; steering that comes first, before the round ran a call, redoes its attempt."""
        if not (self.stopped or self._steering_waits):
            self._interrupt.clear()
            await _first_of((self._interrupt, self.source.released), seconds)
        if self.stopped or not self._steering_waits or self._round.ran:
            return
        yield self._terminal("response.incomplete", incomplete_details={"reason": "steered"})
        yield self._open(redo=True)

    def _message(self, text: str, phase: str) -> list[dict]:
        index, item_id = len(self._round.output), f"msg_{self._round.response_id}_{len(self._round.output)}"
        events = _message_events(item_id, index, text, phase)
        self._round.output.append(events[-1]["item"])
        return events

    async def _call(self, name: str, argument: str | None) -> list[dict]:
        """Write one call into the round, then run it when the IM routes it to the backend."""
        tool, _, arguments = (argument or "").partition(" ")
        if name == "show":
            tool, arguments = "show", json.dumps({"display": argument or ""})
        call_id = self.source.call_id()
        call = {
            "id": f"fc_{call_id}",
            "type": "function_call",
            "status": "completed",
            "call_id": call_id,
            "name": tool,
            "arguments": arguments.strip() or "{}",
        }
        round_ = self._round
        events = _call_events(call, len(round_.output))
        round_.output.append(call)
        round_.calls = True
        route = self.task.route(call)
        if route == "app":
            round_.for_app = True
        elif route == "backend":
            await self._run_call(call, fails=name == "own-fail")
            round_.ran = True
        return events

    async def _run_call(self, call: dict, *, fails: bool) -> None:
        """Run a call the backend runs, reported before and after: an IM work tool, or the emulator's own tool."""
        work_tools = self.task.tools()
        if call["name"] in {tool.get("name") for tool in work_tools.tools()}:
            self.task.tool_started(call, work_tools.label)
            try:
                output, outcome = await work_tools.execute(call["call_id"], call["name"], call["arguments"]), "done"
            except Exception as error:
                output, outcome = json.dumps({"error": str(error)}), "failed"
        else:
            self.task.tool_started(call, self.source.settings.label)
            if fails:
                output, outcome = json.dumps({"error": f"{call['name']} failed (emulated)"}), "failed"
            else:
                output, outcome = json.dumps({"tool": call["name"], "result": "done"}), "done"
        self.task.tool_outcome(call["call_id"], output, outcome)

    def _answer_text(self) -> str:
        if self.reply is not None:
            return self.reply
        request = words(newest_request(self._round.history))
        return f"Done: {request}" if request else "Done."

    def _terminal(self, kind: str, **fields: Any) -> dict:
        """End the open attempt with ``kind``; nothing of it follows."""
        round_, self._round = self._round, None
        status = kind.removeprefix("response.")
        return {"type": kind, "response": round_.snapshot(status, **fields)}


def emulator_source(block: Mapping[str, Any] | None = None) -> EmulatorSource:
    """An emulator source from ``block``, for a test that builds no core: it uses no endpoint, tool or vocabulary."""
    return EmulatorSource.from_settings(block or {}, endpoints=_no_endpoint, tools=tuple, vocabulary=None)


def _no_endpoint(model: str | None) -> Any:
    raise LookupError("The emulator needs no model endpoint")
