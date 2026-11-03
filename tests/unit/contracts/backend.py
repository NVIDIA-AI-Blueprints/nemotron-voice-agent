# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D102, D103, D107

"""The backend port's contract (``delegation_core/backend/port.py``): its checker, doubles and contract test classes.

* :func:`check_events`: the port's event rules on one task's events, as a list of violations;
* round builders (:func:`text_round`, :func:`call_round`) and :class:`ScriptedBackend`, a model provider that plays
  scripted rounds, for the adapters' and the IM's tests;
* :class:`FakeTaskContext`: a ``TaskContext`` that records every report, for an adapter's tests;
* :class:`BackendAdapterContract`: each backend adapter's test subclasses it (``make_adapter(scenario)``);
* :class:`FakeDecisionTarget` and :class:`ConductorContract`: each conductor's test subclasses it;
* :class:`ProviderContract`: each model provider's test subclasses it.

Only the delegation core and the standard library are imported, like the core's own tests.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from delegation_core.backend.port import (
    TERMINALS,
    Answer,
    BackendAdapter,
    BoundaryInput,
    Cancel,
    Conductor,
    ConductorInput,
    Decision,
    Delivery,
    Outcome,
    Request,
    Route,
    Start,
    Steer,
    Steering,
    StopOutcome,
    TaskSummary,
    ToolSource,
    WorkSummary,
    message_phase,
)
from delegation_core.backend.providers import Backend, ProviderLimits
from delegation_core.config import CoreSettings, ModelEndpoint
from tests.unit.contracts.base import ContractCase, collect

# The core's texts as core.yaml words them (TaskContext.texts): a run's error outputs, not-run outputs and notes.
TEXTS = CoreSettings.load(Path(__file__).resolve().parents[3] / "src/examples/frontend_backend_live/core.yaml").texts

# ----------------------------------------------------------------------------------------------- the event rules


def check_events(events: Sequence[dict], *, phases: bool = True) -> list[str]:
    """The violations of the port's event rules in one task's events up to where its iteration ended (empty: none).

    It checks attempts, items, phases, calls last and how the iteration ends. ``phases`` checks that every message
    carries a phase and that a round with calls carries no ``final_answer`` (a model provider's own events have no
    phase: the run labels them).
    """
    problems: list[str] = []
    attempt: list[dict] | None = None
    ended = False  # the current attempt has its terminal
    for index, event in enumerate(events):
        kind = event.get("type", "")
        if kind == "response.created":
            if attempt is not None and not ended:
                problems.append(f"E1: event {index} opens an attempt before the last one ended")
            attempt, ended = [], False
            continue
        if attempt is None:
            problems.append(f"E1: event {index} ({kind}) comes before any response.created")
            continue
        if ended:
            problems.append(f"E1: event {index} ({kind}) follows its attempt's terminal")
            continue
        attempt.append(event)
        if kind in TERMINALS:
            ended = True
            problems += _attempt_problems(attempt, phases)
    if attempt is not None and not ended:
        problems.append("E5: the iteration ended inside an attempt")
    if events and events[-1].get("type") != "response.completed":
        problems.append(f"E5: the iteration ended after {events[-1].get('type')}, not after a completed round")
    return problems


def _attempt_problems(attempt: list[dict], phases: bool) -> list[str]:
    problems = []
    added = {event.get("item", {}).get("id") for event in attempt if event["type"] == "response.output_item.added"}
    items = [event["item"] for event in attempt if event["type"] == "response.output_item.done"]
    for item in items:
        if item.get("id") not in added:
            problems.append(f"E2: item {item.get('id')} is finished but was never announced")
    calls_seen = False
    for item in items:
        if item.get("type") == "function_call":
            calls_seen = True
        elif item.get("type") == "message":
            if calls_seen:
                problems.append("E4: a message follows a call in its round")
            if phases and item.get("phase") not in ("commentary", "final_answer"):
                problems.append(f"E3: message {item.get('id')} has no phase")
    if (
        phases
        and calls_seen
        and any(item.get("type") == "message" and message_phase(item) == "final_answer" for item in items)
    ):
        problems.append("E4: a round with calls carries a final_answer")
    return problems


# -------------------------------------------------------------------------------- rounds and a scripted provider
ENDPOINT = ModelEndpoint(model="test/model", base_url="http://localhost:1/v1")
LIMITS = ProviderLimits(provider_retries=1, first_event_timeout_seconds=2.0, idle_timeout_seconds=2.0)


def message_events(text: str, *, item_id: str = "msg_1", index: int = 0, phase: str | None = None) -> list[dict]:
    """One whole assistant message: announced, its text, finished (``phase`` only when given)."""
    item = {"id": item_id, "type": "message", "status": "completed", "role": "assistant"}
    item["content"] = [{"type": "output_text", "text": text, "annotations": []}]
    if phase is not None:
        item["phase"] = phase
    opened = {**item, "status": "in_progress", "content": []}
    return [
        {"type": "response.output_item.added", "output_index": index, "item": opened},
        {
            "type": "response.output_text.delta",
            "item_id": item_id,
            "output_index": index,
            "content_index": 0,
            "delta": text,
        },
        {"type": "response.output_item.done", "output_index": index, "item": item},
    ]


def call_events(name: str, arguments: dict, *, call_id: str = "call_1", index: int = 0) -> list[dict]:
    """One whole function call: announced, then finished."""
    item = {"id": f"fc_{call_id}", "type": "function_call", "status": "completed", "call_id": call_id, "name": name}
    item["arguments"] = json.dumps(arguments)
    return [
        {"type": "response.output_item.added", "output_index": index, "item": {**item, "status": "in_progress"}},
        {"type": "response.output_item.done", "output_index": index, "item": item},
    ]


def a_round(*parts: list[dict], response_id: str = "resp_1") -> list[dict]:
    """One attempt that completes: ``response.created``, the parts' events, ``response.completed``."""
    events = [{"type": "response.created", "response": {"id": response_id, "status": "in_progress"}}]
    for part in parts:
        events += part
    output = [event["item"] for event in events if event["type"] == "response.output_item.done"]
    return [
        *events,
        {"type": "response.completed", "response": {"id": response_id, "status": "completed", "output": output}},
    ]


def text_round(text: str, response_id: str = "resp_1") -> list[dict]:
    """A round that answers ``text``, with no phase (as a chat model gives it)."""
    return a_round(message_events(text), response_id=response_id)


def call_round(name: str, arguments: dict, call_id: str = "call_1", *, said: str | None = None) -> list[dict]:
    """A round that calls ``name``; ``said`` is a message before the call (no phase)."""
    parts = [message_events(said, index=0)] if said else []
    parts.append(call_events(name, arguments, call_id=call_id, index=len(parts)))
    return a_round(*parts, response_id=f"resp_{call_id}")


class ScriptedBackend(Backend):
    """A model provider that plays rounds of Responses events, one round per ``stream()`` call.

    An ``Exception`` in a round raises there; an ``asyncio.Event`` waits there until it is set. ``payloads`` keeps
    every request it was sent.
    """

    def __init__(self, rounds, endpoint: ModelEndpoint = ENDPOINT, limits: ProviderLimits = LIMITS):
        super().__init__(endpoint, limits)
        self.rounds = list(rounds)
        self.payloads: list[dict] = []
        self.gate: asyncio.Event | None = None

    async def stream(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        if self.gate is not None:
            await self.gate.wait()
        step = self.rounds.pop(0)
        for event in step:
            if isinstance(event, Exception):
                raise event
            if isinstance(event, asyncio.Event):
                await event.wait()
                continue
            yield event


# ------------------------------------------------------------------------------------------------ a task's context
class NoTools:
    """A tool source with no tools: what a task gets when the IM offers no work tool."""

    label = "delegation_core"

    def tools(self) -> list[dict]:
        return []

    async def execute(self, call_id: str, name: str, arguments: str) -> str:
        return json.dumps({"error": f"Unknown tool {name}"})


class FakeTaskContext:
    """A ``TaskContext`` for an adapter's tests: it serves a request and a history, and records every report.

    The app's functions are those the request declares (``declared``); every other call is the backend's, and a call it
    refuses gets ``texts.not_run_output``. Its texts are core.yaml's. The test hands the app's outputs over with
    :meth:`give_outputs` and steering with :meth:`steer_with`.
    """

    def __init__(
        self,
        *,
        instructions: str = "COMPOSED-INSTRUCTIONS",
        history: list[dict] | None = None,
        declared: Sequence[dict] = (),
        tools: ToolSource | None = None,
        work_id: str = "W1",
        task_id: str = "W1.1",
    ):
        self.work_id, self.task_id = work_id, task_id
        self.texts = TEXTS
        self.instructions, self.declared = instructions, list(declared)
        self.items = list(history or [{"type": "message", "role": "user", "content": "Do the task."}])
        self.work_tools = tools or NoTools()
        self.refused: set[str] = set()
        self.started: list[tuple[str, str]] = []  # (call_id, source), in order
        self.outcomes: dict[str, tuple[str, Outcome]] = {}
        self.notes: list[str] = []
        self.settled: list[tuple[str, Delivery]] = []
        self.traces: list[tuple[str, dict]] = []
        self.order: list[str] = []  # "started:<id>", "outcome:<id>", in order
        self._outputs: list[dict] = []
        self._steering: deque[Steering] = deque()  # the Work's one steering queue, oldest first

    # -------------------------------------------------------------------------- what the test does
    def give_outputs(self, outputs: Sequence[dict]) -> None:
        """The app's outputs are in; the next ``take_boundary_input()`` hands them over."""
        self._outputs += list(outputs)

    def steer_with(self, items: Sequence[dict], steering_id: str = "steer_1") -> Steering:
        """Steering waits for the next safe point, behind any batch already waiting; the test then calls ``steer()``."""
        steering = Steering(steering_id, tuple(items))
        self._steering.append(steering)
        return steering

    # -------------------------------------------------------------------------- the port
    def request(self) -> dict:
        return {
            "model": "test/model",
            "instructions": self.instructions,
            "tools": list(self.declared),
            "tool_choice": "auto",
        }

    def history(self) -> list[dict]:
        return list(self.items)

    def position(self) -> int:
        return len(self.items)

    def since(self, position: int) -> list[dict]:
        return list(self.items[position:])

    def tools(self) -> ToolSource:
        return self.work_tools

    def route(self, call: dict) -> Route:
        if call.get("call_id") in self.refused:
            return "refused"
        names = {tool.get("name") for tool in self.declared}
        return "app" if call.get("name") in names else "backend"

    def refusal(self, call_id: str) -> str:
        return self.texts.not_run_output

    def tool_started(self, call: dict, source: str) -> None:
        self.started.append((call.get("call_id", ""), source))
        self.order.append(f"started:{call.get('call_id')}")

    def tool_outcome(self, call_id: str, output: str, outcome: Outcome) -> None:
        self.outcomes[call_id] = (output, outcome)
        self.order.append(f"outcome:{call_id}")

    def progress(self, text: str) -> None:
        self.notes.append(text)

    def take_boundary_input(self) -> BoundaryInput:
        outputs, self._outputs = tuple(self._outputs), []
        self.items += list(outputs)
        steering = self._steering.popleft() if self._steering else None
        return BoundaryInput(outputs, steering)

    def settle(self, steering: Steering, delivery: Delivery) -> None:
        self.settled.append((steering.id, delivery))
        if delivery == "applied":
            self.items += list(steering.items)

    def trace(self, kind: str, **fields: Any) -> None:
        self.traces.append((kind, fields))


# ------------------------------------------------------------------------------------------ the adapter contract
MENU_FUNCTION = {"type": "function", "name": "get_menu", "parameters": {"type": "object", "properties": {}}}
STEERING = [{"type": "message", "role": "user", "content": "Make it two."}]


class BackendAdapterContract(ContractCase):
    """What every backend adapter does with one task, whatever its kind (the port's event rules).

    A subclass is an ``IsolatedAsyncioTestCase`` that implements:

    * ``make_adapter(scenario) -> BackendAdapter``, its backend scripted for the scenario:

      - ``answer``: one round that answers "Done.";
      - ``own_call``: a round that calls the backend's own tool ``lookup``, then a round that answers;
      - ``tool_fails``: the same, but ``lookup`` fails;
      - ``app_call``: a round that says a line and calls the app's ``get_menu``, then (after the outputs) one that
        answers;
      - ``slow``: a first round that waits after it opened until :meth:`release`, then answers "Old."; a second round
        that answers "Done.";
      - ``tail_error``: a round that answers and completes, then fails after its terminal;
    * ``release()``: let the ``slow`` round go on;
    * ``sent_instructions() -> list[str]``: the instructions its backend was sent, every model call.

    The harness binds it to a real runtime (Hermes over ACP, in ``test_harness_real_hermes.py``) whose model endpoint
    is scripted for each scenario.
    """

    def make_adapter(self, scenario: str) -> BackendAdapter:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError

    def sent_instructions(self) -> list[str]:
        raise NotImplementedError

    def task(self, **kwargs: Any) -> FakeTaskContext:
        return FakeTaskContext(declared=[MENU_FUNCTION], **kwargs)

    async def run_task(self, scenario: str, task: FakeTaskContext | None = None) -> tuple[FakeTaskContext, list[dict]]:
        task = task or self.task()
        run = self.make_adapter(scenario).start(task)
        return task, await collect(run.events())

    async def test_events_keep_rounds_and_attempts(self):
        _, events = await self.run_task("answer")
        self.assertEqual(check_events(events, phases=False), [])

    async def test_every_message_carries_its_phase(self):
        _, events = await self.run_task("answer")
        self.assertEqual(check_events(events), [])
        answers = [
            e for e in events if e["type"] == "response.output_item.done" and e["item"].get("phase") == "final_answer"
        ]
        self.assertEqual(len(answers), 1)

    async def test_a_round_with_calls_has_no_final_answer(self):
        _, events = await self.run_task("app_call")
        self.assertEqual(check_events(events), [])

    async def test_app_calls_end_the_events_until_resume(self):
        task = self.task()
        run = self.make_adapter("app_call").start(task)
        first = await collect(run.events())
        self.assertEqual(first[-1]["type"], "response.completed")
        calls = [
            e["item"]
            for e in first
            if e["type"] == "response.output_item.done" and e["item"]["type"] == "function_call"
        ]
        self.assertEqual([call["name"] for call in calls], ["get_menu"])
        task.give_outputs([{"type": "function_call_output", "call_id": calls[0]["call_id"], "output": "{}"}])
        await run.resume()
        rest = await collect(run.events())
        texts = [e["item"] for e in rest if e["type"] == "response.output_item.done" and e["item"]["type"] == "message"]
        self.assertTrue(texts)
        self.assertEqual(rest[-1]["type"], "response.completed")

    async def test_own_calls_are_reported_before_and_after(self):
        task, events = await self.run_task("own_call")
        self.assertEqual(check_events(events), [])
        self.assertEqual(task.order, ["started:call_1", "outcome:call_1"])
        self.assertEqual(task.outcomes["call_1"][1], "done")

    async def test_a_failed_tool_keeps_its_output_and_the_task_goes_on(self):
        task, events = await self.run_task("tool_fails")
        output, outcome = task.outcomes["call_1"]
        self.assertEqual(outcome, "failed")
        self.assertTrue(output)
        self.assertEqual(events[-1]["type"], "response.completed")

    async def test_steering_is_settled_once(self):
        task = self.task()
        run = self.make_adapter("slow").start(task)
        events = run.events()
        await anext(events)  # the slow round is open
        steering = task.steer_with(STEERING)
        run.steer()
        self.release()
        await collect(events)
        self.assertEqual([steering_id for steering_id, _ in task.settled], [steering.id])

    async def test_a_stop_is_confirmed_and_nothing_follows_it(self):
        task = self.task()
        run = self.make_adapter("slow").start(task)
        events = run.events()
        await anext(events)
        outcome: StopOutcome = await asyncio.wait_for(run.stop(), 5)
        self.assertIn(outcome, ("stopped", "unconfirmed"))
        self.release()
        later = await collect(events)
        self.assertEqual([e for e in later if e["type"] == "response.output_item.done"], [])

    async def test_an_attempt_that_delivered_an_item_is_never_retried(self):
        # the tail_error round answered (it delivered an item) and completed before it failed: no retry.
        task = self.task()
        run = self.make_adapter("tail_error").start(task)
        events = []
        try:
            async for event in run.events():
                events.append(event)
        except Exception:
            pass  # a run may fail here; it may never run the delivered round again
        self.assertEqual(sum(e["type"] == "response.created" for e in events), 1)

    async def test_the_task_reads_the_composed_instructions(self):
        await self.run_task("answer", self.task(instructions="COMPOSED-INSTRUCTIONS voice note"))
        self.assertIn("COMPOSED-INSTRUCTIONS voice note", self.sent_instructions())

    async def test_the_adapter_offers_a_conductor_and_a_run_limit(self):
        adapter = self.make_adapter("answer")
        self.assertGreaterEqual(adapter.max_running, 1)
        self.assertTrue(callable(getattr(adapter.conductor(), "decide", None)))


# ------------------------------------------------------------------------------------------ the conductor contract
class FakeDecisionTarget:
    """A ``DecisionTarget`` that records each decision and returns an outcome text, as the IM session would."""

    def __init__(self):
        self.decisions: list[Decision] = []
        self.works = 0

    async def start(self, decision: Start) -> str:
        self.decisions.append(decision)
        self.works += 1
        return f"Started W{self.works} with {max(1, len(decision.tasks))} task(s)."

    async def steer(self, decision: Steer) -> str:
        self.decisions.append(decision)
        return f"Sent to {decision.work_id}; it applies at the next step."

    async def cancel(self, decision: Cancel) -> str:
        self.decisions.append(decision)
        return f"Stopping {decision.work_id}."

    async def answer(self, decision: Answer) -> str:
        self.decisions.append(decision)
        return f"Answered {decision.delegation_id}."

    async def place(self, decision: Decision) -> str:
        """The ``place`` callback a conductor gets: apply by double dispatch."""
        return await decision.apply(self)


def a_request(delegation_id: str, text: str) -> Request:
    return Request(delegation_id, ({"type": "message", "role": "user", "content": text},))


def an_open_work(work_id: str = "W1", delegation_id: str = "d_old") -> WorkSummary:
    task = TaskSummary(f"{work_id}.1", "running", None, ("get_menu: done",), True, "Looking it up.", 3.0)
    return WorkSummary(work_id, "running", delegation_id, (a_request(delegation_id, "Find teas."),), (task,), 3.0)


def an_ended_work(work_id: str = "W0", delegation_id: str = "d_done") -> WorkSummary:
    task = TaskSummary(f"{work_id}.1", "answered", None, ("book_table: done",), True, None, 9.0)
    return WorkSummary(work_id, "answered", delegation_id, (a_request(delegation_id, "Book a table."),), (task,), 9.0)


class ConductorContract(ContractCase):
    """What every conductor does with one ``decide()`` call (the rule conductors, the LLM agent's, the harness's).

    A subclass is an ``IsolatedAsyncioTestCase`` that implements ``make_conductor() -> Conductor``.
    """

    def make_conductor(self) -> Conductor:
        raise NotImplementedError

    def given(self, *arrivals: Request, works: tuple[WorkSummary, ...] = ()) -> ConductorInput:
        conversation = tuple(item for arrival in arrivals for item in arrival.input)
        return ConductorInput(tuple(arrivals), works, conversation, 0)

    async def test_it_places_only_through_place_and_names_only_what_it_was_shown(self):
        # It is shown an open Work and a recently ended one (the work records); the IM re-checks a decision that names
        # the ended one.
        target = FakeDecisionTarget()
        given = self.given(a_request("d_new", "Also a latte."), works=(an_open_work(), an_ended_work()))
        await asyncio.wait_for(self.make_conductor().decide(given, target.place), 30)
        for decision in target.decisions:
            with self.subTest(decision=decision):
                if isinstance(decision, Start):
                    self.assertLessEqual(set(decision.delegations), {"d_new"})
                elif isinstance(decision, Steer):
                    self.assertIn(decision.work_id, ("W1", "W0"))
                    self.assertLessEqual(set(decision.delegations), {"d_new"})
                    self.assertIn(decision.task_id, (None, f"{decision.work_id}.1"))
                elif isinstance(decision, Cancel):
                    self.assertIn(decision.work_id, ("W1", "W0"))
                else:
                    self.assertEqual(decision.delegation_id, "d_new")

    async def test_it_places_each_arrival_at_most_once(self):
        target = FakeDecisionTarget()
        given = self.given(a_request("d_1", "Find teas."), a_request("d_2", "And the hours."))
        await asyncio.wait_for(self.make_conductor().decide(given, target.place), 30)
        placed = [d for decision in target.decisions for d in _delegations(decision)]
        self.assertEqual(len(placed), len(set(placed)))

    async def test_a_decide_cancelled_before_it_places_leaves_it_usable(self):
        conductor = self.make_conductor()
        first = asyncio.create_task(
            conductor.decide(self.given(a_request("d_1", "Find teas.")), FakeDecisionTarget().place)
        )
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        target = FakeDecisionTarget()
        given = self.given(a_request("d_1", "Find teas."), a_request("d_2", "Make it green tea."))
        await asyncio.wait_for(conductor.decide(given, target.place), 30)
        placed = {d for decision in target.decisions for d in _delegations(decision)}
        self.assertLessEqual(placed, {"d_1", "d_2"})


def _delegations(decision: Decision) -> tuple[str, ...]:
    return (decision.delegation_id,) if isinstance(decision, Answer) else getattr(decision, "delegations", ())


# ------------------------------------------------------------------------------------------ the provider contract
class ProviderContract(ContractCase):
    """What every model provider's ``stream()`` gives the adapter: rounds by the port's event rules, phases aside.

    A subclass is an ``IsolatedAsyncioTestCase`` that implements ``make_provider(script) -> Backend``, scripted so that
    ``text`` answers "Hello." and ``call`` calls ``get_menu`` once.
    """

    def make_provider(self, script: str) -> Backend:
        raise NotImplementedError

    def payload(self) -> dict:
        return {"model": "test/model", "instructions": "Be brief.", "input": [], "tools": [MENU_FUNCTION]}

    async def test_a_text_round_keeps_the_round_rules(self):
        events = await collect(self.make_provider("text").stream(self.payload()))
        self.assertEqual(check_events(events, phases=False), [])

    async def test_a_call_round_keeps_the_round_rules(self):
        events = await collect(self.make_provider("call").stream(self.payload()))
        self.assertEqual(check_events(events, phases=False), [])
        calls = [e for e in events if e["type"] == "response.output_item.done" and e["item"]["type"] == "function_call"]
        self.assertEqual(len(calls), 1)
