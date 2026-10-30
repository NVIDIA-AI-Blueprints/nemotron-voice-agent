# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Works and tasks: what runs, what it reads, and how it reaches the wire.

One request, or several joined requests, is one Work (:class:`WorkRun`), run by one or more tasks (:class:`TaskRun`):
each task is one ``BackendRun`` that reads and reports through the task's ``TaskContext``. Nothing is written until a
round completes, so a failed attempt writes nothing. A Work's response belongs to its carrier, the newest request that
landed in it; while that response waits for the app's outputs, the other tasks' items are held in order.

The commit gate refuses every call a task routes while a newer request waits to land in it, or waits for the
conductor's decision, so the backend re-plans with the newer request; a call that already runs is never cut. With
``delegation.hold_answers_while_deciding``, an answer a task gives while a request waits for the conductor's decision
waits for that decision too: steered into the task, the answer is progress and the task runs one more round with the
request, in a new run of its adapter; placed elsewhere, it is written then; a stop drops it.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from delegation_core.backend.port import (
    BackendRun,
    BoundaryInput,
    Delivery,
    Outcome,
    Request,
    Route,
    Steering,
    StopOutcome,
    TaskSummary,
    WorkSummary,
    message_phase,
)
from delegation_core.im.conversation import TaskHistory
from delegation_core.im.input import settings_for, task_request, user_message, with_parts
from delegation_core.im.results import CallReports, ControlCalls
from delegation_core.im.stop import AtBackstop, AtDeadline, StopReason
from delegation_core.im.stream import ResponseWriter, message_text
from delegation_core.im.tools import WorkTools
from delegation_core.ledger.records import Runner, Task, Work
from delegation_core.ledger.records import Steering as SteeringRecord

if TYPE_CHECKING:
    from delegation_core.config import Texts
    from delegation_core.im.session import IMSession

# The reasons a cut attempt ends with that the IM passes on as they are.
OFFICIAL_REASONS = frozenset({"max_output_tokens", "content_filter"})
# An upstream "too many requests" (an error's ``status_code``), which ends a task ``rate_limit_exceeded``.
TOO_MANY_REQUESTS = 429
# How much of a request's text the work records show: its newest words.
RECORD_TEXT_CHARS = 300
# The display tool always shows as an mcp_call item; the IM's control tools (cancel_work) never reach the wire.
DISPLAY_REPORTS = CallReports()
CONTROL_REPORTS = ControlCalls()


@dataclass(frozen=True)
class Failure:
    """How a failed task ends its Work's response.

    ``response.failed`` with an official error code, or ``response.incomplete`` with an official reason.
    """

    kind: str
    fields: dict

    @classmethod
    def of(cls, error: BaseException, terminal: dict | None) -> Failure:
        """The failure an error gives, read from the task's last attempt when it ended with one.

        An upstream 429 that ended the task is ``rate_limit_exceeded`` even after a failed attempt, whose terminal the
        run may have closed with its own ``server_error`` before it retried; otherwise the attempt's own code counts.
        """
        response = (terminal or {}).get("response") or {}
        reason = (response.get("incomplete_details") or {}).get("reason")
        if reason in OFFICIAL_REASONS:
            return cls("response.incomplete", {"incomplete_details": {"reason": reason}})
        detail = f"{type(error).__name__}: {error}"[:300]
        limited = getattr(error, "status_code", None) == TOO_MANY_REQUESTS
        code = "rate_limit_exceeded" if limited else (response.get("error") or {}).get("code") or "server_error"
        return cls("response.failed", {"error": {"code": code, "message": detail}})


# A task whose run ended with neither an answer nor the app's calls (an empty answer is unclear once it acted).
NO_ANSWER = Failure("response.failed", {"error": {"code": "server_error", "message": "The backend gave no answer."}})


@dataclass(eq=False)
class Batch:
    """Requests steered into a Work, from the conductor's decision until they land (a steering batch).

    A batch steered into every open task waits in each of them, as one ``Steering`` record per task; it lands once,
    when the first task applies it.
    """

    delegations: list[str]
    items: list[dict]
    brings: dict | None  # the settings an explicit request brings (a typed run or continuation); None keeps the Work's
    landed: bool = False


@dataclass(eq=False)
class Waiting:
    """A batch waiting in one task: the batch, and that task's ``Steering`` record of it."""

    batch: Batch
    record: SteeringRecord


@dataclass(eq=False)
class Parked:
    """A response that ended with the app's calls, and the tasks that wait for the calls' outputs."""

    work: WorkRun
    delegation_id: str
    calls: dict[str, tuple[TaskRun, dict]]  # call_id: (the task, the call), in call order
    timer: asyncio.TimerHandle | None = field(default=None, repr=False)


def request_text(items: list[dict]) -> str:
    """The newest words of a request's user messages, as the work records show them."""
    texts = []
    for item in items:
        content = item.get("content") if item.get("role") == "user" else None
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts += [part.get("text", "") for part in content if isinstance(part, dict) and part.get("text")]
    return " ".join(texts)[-RECORD_TEXT_CHARS:]


class TaskRun:
    """One task: one worker's run, and the ``TaskContext`` it reads and reports through (``port.TaskContext``)."""

    def __init__(self, work: WorkRun, record: Task):
        """A task of ``work``, run for its ``Task`` record; its brief is the conductor's, ``None`` for the requests."""
        self.work, self.session, self.record = work, work.session, record
        self.work_id, self.task_id, self.brief = work.id, record.id, record.brief
        self.ended = False
        self.run: BackendRun | None = None
        self.view: TaskHistory | None = None  # its history, made when it starts
        self.pending: list[dict] = []  # requests that joined before it started: they join its history then
        self.steering: deque[Waiting] = deque()  # steering batches waiting for its next safe point
        self.taken: dict[str, Waiting] = {}  # steering handed to the adapter, by its record's id, until settled
        self.outputs: dict[str, dict] = {}  # outputs that join at the next safe point, by call_id
        self.messages: list[dict] = []  # typed messages that came with the app's outputs: they join after them
        self.call_order: list[str] = []  # its last round's calls, in order
        self.routes: dict[str, Route] = {}  # who runs each of its calls, decided once
        self.refusals: dict[str, str] = {}  # each refused call's not-run output, which says why
        self.app_calls: dict[str, dict] = {}  # its last round's calls the app runs
        self.started_calls: dict[str, CallReports] = {}  # the backend calls reported started, and how they show
        self.reported: set[str] = set()  # the backend calls whose outcome was reported
        self.steps: dict[str, list[str]] = {}  # each call: its name and its outcome so far
        self.shown: list[dict] = []  # what its show calls put on the screen, as the items of its result
        self.attempt: list[dict] = []  # the items of the attempt being read
        self.last_terminal: dict | None = None  # the terminal of an attempt that failed or was cut
        self.answered = False  # its last round gave an answer
        self.answers = 0
        self.held = False  # its round's calls were refused while the conductor decides: it waits for the decision
        self.held_answer: dict | None = None  # its answer, given while the conductor decides: it waits for the decision
        self.completed = False  # its run ended as it should: with an answer, or (one-shot) with the app's calls
        self.stop_reason: StopReason | None = None  # a stop was requested: nothing more of it is written
        self.stopped: StopOutcome | None = None
        self.own_cancel = False  # its own cancel_work came: later calls are refused
        self.failure: Failure | None = None
        self.grant: asyncio.Future | None = None  # resolved by the scheduler when it has a slot
        self.outputs_in: asyncio.Future | None = None  # resolved when the app's outputs are in
        self.driver: asyncio.Task | None = None
        shown = self.session.result_format.tools()
        self.displays = frozenset(shown)  # its work tools whose items are the display
        names = tuple(dict.fromkeys((*shown, *self.session.settings.work_tools)))
        self.work_tools = WorkTools(names, self.texts, self.cancel_own, self.session.settings.display_label)
        self._readers: dict[str, Callable[[dict], None]] = {
            "response.created": self._attempt_opened,
            "response.output_item.done": self._item_done,
            "response.completed": self._round_completed,
            "response.failed": self._attempt_failed,
            "response.incomplete": self._attempt_failed,
        }

    # ------------------------------------------------------------------------------------------ the driver
    def begin(self) -> None:
        """Start the task's driver: it waits for a slot, then runs."""
        self.driver = asyncio.create_task(self._drive(), name=f"task-{self.task_id}")

    async def _drive(self) -> None:
        try:
            await self._slot()
            self.run = self.work.task_started(self)
            while True:
                await self._read_events()
                if self.held_answer is not None and await self._decided_on_answer():
                    self.run = self.session.backend.start(self)  # one more round, with the newer request
                    continue
                if self.stop_reason is not None or not self.app_calls:
                    break
                if self._gated():
                    await self.session.decided()  # a conductor deciding on the newer input: wait for its decision
                    if self.stop_reason is not None:
                        break  # the decision stopped this task: no next round starts (the stop door confirms it)
                    await self.run.resume()  # their not-run outputs join at the next safe point, with the newer input
                    continue
                if not self.session.waits_for_app(self):
                    break
                await self._wait_for_app()
                await self.run.resume()
            if self.stop_reason is None:
                self.completed = self.answered or bool(self.app_calls)
                self.failure = None if self.completed else NO_ANSWER
        except asyncio.CancelledError:
            if self.stop_reason is None:
                raise
        except Exception as exc:
            if self.stop_reason is None:
                logger.warning(f"Task {self.task_id} failed: {type(exc).__name__}: {exc}")
                self.failure = Failure.of(exc, self.last_terminal)
                self.trace("backend.failed", error_type=type(exc).__name__, ending=self.failure.kind)
        finally:
            self.session.release(self)
            if self.stop_reason is None:
                self._end()

    @property
    def running(self) -> bool:
        """Whether its run holds a slot now (its record is ``running``)."""
        return self.record.state == "running"

    async def _slot(self) -> None:
        """Wait for a slot, then run: ``queued`` (or ``waiting_client``, after the app's outputs) to ``running``."""
        self.grant = self.session.request_slot(self)
        await self.grant
        self.record.move("running")

    async def _read_events(self) -> None:
        self.answered = False
        events = self.run.events()
        try:
            async for raw in events:
                reader = self._readers.get(raw.get("type", ""))
                if reader is not None:
                    reader(raw)
                if self.held:  # the round's calls were refused while the conductor decides: the run reads on after it
                    self.held = False
                    await self.session.decided()
                    if self.stop_reason is not None:
                        return  # the decision stopped this task: no next round starts (the stop door confirms it)
        finally:
            close = getattr(events, "aclose", None)
            if close is not None:
                await close()

    def _gated(self) -> bool:
        """The commit gate as the round's app calls would go out; ``True`` when it refused them.

        Their route was decided when the round completed, and the round's backend calls ran since: newer input that
        came meanwhile waits to land, or for the conductor's decision, so none of the app's calls is dispatched.
        """
        if not self.newer_input_waits():
            return False
        for call_id, call in self.app_calls.items():
            self.routes[call_id] = "refused"
            self.refusals[call_id] = self.session.texts.not_run_output
            self._refuse(call)
        self.app_calls = {}
        return True

    async def _decided_on_answer(self) -> bool:
        """Wait for the conductor's decision, then release the held answer; ``True`` when the task runs on.

        Newer input the decision steered into this task makes the answer progress, and the task runs one more round
        with it; otherwise the answer is written now. A stop meanwhile has dropped it (``request_stop``).
        """
        await self.session.decided()
        if self.held_answer is None:
            return False
        if self.unapplied():
            self.session.commentary(self, self._release_answer("commentary"))
            return True
        self._answer(self._release_answer("written"))
        return False

    def _release_answer(self, how: str) -> dict | None:
        """The held answer leaves the hold ``how``: written, commentary or dropped (``None``: none was held)."""
        answer, self.held_answer = self.held_answer, None
        if answer is not None:
            self.trace("im.answer.released", how=how)
        return answer

    async def _wait_for_app(self) -> None:
        """Wait, holding no slot, for the app's outputs; then for a slot again."""
        self.session.release(self)
        self.record.move("waiting_client")
        self.outputs_in = asyncio.get_running_loop().create_future()
        self.record.acted = True
        self.work.park(self, list(self.app_calls.values()))
        await self.outputs_in
        self.outputs_in = None
        await self._slot()
        self.work.resumed()

    def give(self, outputs: list[dict], messages: list[dict]) -> None:
        """The app's outputs for this task's calls are in (and the typed messages that came with them)."""
        for output in outputs:
            self.outputs[output["call_id"]] = output
            self._step(output["call_id"], "done")
        self.messages += messages
        if self.outputs_in is not None and not self.outputs_in.done():
            self.outputs_in.set_result(None)

    def _end(self) -> None:
        """The task ended: its record moves to how it ended, with whether anything it did changed something."""
        if self.ended:
            return
        self.ended = True
        record = self.record
        state = "failed" if self.failure is not None else "answered" if self.completed else "cancelled"
        record.move(state)
        record.ended_at = time.monotonic()
        record.outcome = ("done" if state == "answered" else "unclear") if record.acted else "none"
        self.work.task_ended(self)

    # ------------------------------------------------------------------------------------------ the stop door
    def request_stop(self, reason: StopReason) -> bool:
        """Ask the task to stop for ``reason``: nothing more of it is written, and its calls are refused.

        ``False`` when it had ended, or a stop was asked already.
        """
        if self.ended or self.stop_reason is not None:
            return False
        self.stop_reason = reason
        self.record.stop.move("stop_requested")
        self.session.unschedule(self)
        self._release_answer("dropped")
        return True

    async def confirm_stop(self) -> StopOutcome:
        """Stop the run, bounded by ``stop_timeout_seconds``; then write the reason's line and end the task."""
        outcome: StopOutcome = "stopped"
        if self.run is not None:
            try:
                async with asyncio.timeout(self.session.settings.stop_timeout_seconds):
                    outcome = await self.run.stop()
            except Exception as exc:  # a stop that times out or raises is not confirmed
                logger.warning(f"Task {self.task_id} did not confirm its stop: {type(exc).__name__}: {exc}")
                outcome = "unconfirmed"
        self.stopped = outcome
        self.record.stop.move(outcome)
        self.record.stop.outcome = "unclear" if self.record.acted else "none"
        if self.driver is not None and not self.driver.done():
            self.driver.cancel()
            await asyncio.gather(self.driver, return_exceptions=True)
        self.work.stop_answer(self.stop_reason.answer(self.session.texts, outcome))
        self._end()
        return outcome

    def cancel_own(self) -> str:
        """The task's own ``cancel_work``: it stops after this step; later calls are refused, its next answer counts."""
        self.own_cancel = True
        return self.session.texts.outcomes["cancel_requested"].format(work=self.work_id, tasks=self.task_id)

    # ------------------------------------------------------------------------------------------ the round reader
    def _attempt_opened(self, raw: dict) -> None:
        self.attempt, self.last_terminal = [], None

    def _item_done(self, raw: dict) -> None:
        item = raw.get("item")
        if isinstance(item, dict):
            self.attempt.append(item)

    def _attempt_failed(self, raw: dict) -> None:
        self.attempt, self.last_terminal = [], raw  # a failed or cut attempt writes nothing

    def _round_completed(self, raw: dict) -> None:
        items, self.attempt, self.last_terminal = self.attempt, [], None
        usage = (raw.get("response") or {}).get("usage")
        writer = self.session.book.writer(self.work.carrier)
        if usage and writer is not None:
            writer.usage = usage  # the response's terminal reports its newest round's usage
        self.view.join(items)
        calls = [item for item in items if item.get("type") == "function_call"]
        messages = [item for item in items if item.get("type") == "message"]
        finals = [] if calls else [item for item in messages if message_phase(item) == "final_answer"]
        answer = finals[-1] if finals else None
        for message in messages:
            if message is not answer:
                self.session.commentary(self, message)
        self.call_order = [call.get("call_id", "") for call in calls]
        self.app_calls = {}
        for call in calls:
            if self.route(call) == "app":  # decided once: an adapter may have routed it before passing the round on
                self.app_calls[call.get("call_id", "")] = call
                self.session.action(self, call)
        refused = any(self.routes.get(call.get("call_id", "")) == "refused" for call in calls)
        self.held = refused and self.session.waits_for_decision and self.stop_reason is None
        if answer is not None and self._holds_answers():
            self.held_answer = answer
            self.trace("im.answer.held")
        elif answer is not None:
            self._answer(answer)

    def _holds_answers(self) -> bool:
        """Whether its answer waits for the conductor's decision now (``delegation.hold_answers_while_deciding``)."""
        holds = self.session.settings.hold_answers_while_deciding
        return holds and self.session.waits_for_decision and self.stop_reason is None

    def _answer(self, message: dict) -> None:
        """Its answer: the round answered, and the Work writes it."""
        self.answered = True
        self.answers += 1
        self.trace("im.answer.written")
        self.work.answer(self, message)

    # ------------------------------------------------------------------------------------------ TaskContext
    @property
    def texts(self) -> Texts:
        """The core's texts: they word the error outputs, not-run outputs and notes its run adds."""
        return self.session.texts

    def request(self) -> dict:
        """The settings of the task's next model call, its instructions composed."""
        return self.work.request()

    def history(self) -> list[dict]:
        """The task's view of the conversation, within the budget."""
        return self.view.history()

    def position(self) -> int:
        """The position after the last item of the task's view."""
        return self.view.position()

    def since(self, position: int) -> list[dict]:
        """The task's items from ``position`` on, trimmed ones included."""
        return self.view.since(position)

    def tools(self) -> WorkTools:
        """The IM's work tools for this task."""
        return self.work_tools

    def route(self, call: dict) -> Route:
        """Who runs ``call``: decided once per call, when the adapter or the round reader first asks."""
        call_id = call.get("call_id", "")
        if call_id not in self.routes:
            name = call.get("name", "")
            refusal = self._refusal(call)
            self.routes[call_id] = "refused" if refusal is not None else self._route(name)
            self._step(call_id, "started", name)
            if refusal is not None:
                self.refusals[call_id] = refusal
                self._refuse(call)
            else:
                self.session.call_routed(call)
        return self.routes[call_id]

    def refusal(self, call_id: str) -> str:
        """The not-run output the IM gave a call it refused, which says why."""
        return self.refusals[call_id]

    def _refusal(self, call: dict) -> str | None:
        """Why the IM will not run ``call`` now, as the not-run output that says so; ``None``: it runs.

        Once its stop was asked or after its own ``cancel_work``, ``texts.stopped_output``; for a function the app and
        a work tool both name, ``texts.clash_output``; while newer input waits, ``texts.not_run_output`` (the commit
        gate); in a Work that took one request, for an app call that repeats the newest call that ran, which this task
        made and the app answered, with no request since, that answer (``texts.repeated_call_output``).
        """
        texts, name = self.session.texts, call.get("name", "")
        declared = any(tool.get("type") == "function" and tool.get("name") == name for tool in self.work.settings.tools)
        if self.stop_reason is not None or self.own_cancel:
            return texts.stopped_output
        if declared and name in self.work_tools.names:  # the app and a work tool name the same function
            return texts.clash_output
        if self.newer_input_waits():
            return texts.not_run_output
        if not declared or len(self.work.requests) > 1:  # a Work that took another request may ask for it again
            return None
        return self.session.repeated_output(call, self.task_id)

    def _route(self, name: str) -> Route:
        """Who runs a call the IM lets run: the app, for a function the request declared; else the backend."""
        declared = any(tool.get("type") == "function" and tool.get("name") == name for tool in self.work.settings.tools)
        return "app" if declared else "backend"

    def newer_input_waits(self) -> bool:
        """Whether newer input waits: to land in this task, or for the conductor's decision (it may concern any Work).

        The backend re-plans with it first: steering that has not landed, or a request the conductor has not placed.
        """
        return bool(self.steering or self.taken) or self.session.waits_for_decision

    def runner(self, name: str) -> Runner:
        """Who runs a call of ``name``, as its action records it: the IM (its work tools), the app or the backend.

        A refused call keeps the runner it would have had.
        """
        if name in self.work_tools.names:
            return "im"
        declared = any(tool.get("type") == "function" and tool.get("name") == name for tool in self.work.settings.tools)
        return "app" if declared else "backend"

    def _refuse(self, call: dict) -> None:
        """A call the IM does not run: its action is ``not_run``, and the not-run output that says why is its output."""
        call_id = call.get("call_id", "")
        output = self.refusals[call_id]
        self.outputs[call_id] = {"type": "function_call_output", "call_id": call_id, "output": output}
        self.session.action(self, call, "not_run")
        self._step(call_id, "not_run", call.get("name", ""))

    def tool_started(self, call: dict, source: str) -> None:
        """A call the backend runs starts: reported as ``delegation.backend_calls`` says."""
        call_id = call.get("call_id", "")
        if call_id in self.started_calls or self.routes.get(call_id) == "refused":
            return
        self.started_calls[call_id] = self._reports(call.get("name", ""))
        self.record.acted = True
        self.session.action(self, call)
        self._step(call_id, "started", call.get("name", ""))
        if self.stop_reason is None:
            self.started_calls[call_id].started(self, call, source)

    def tool_outcome(self, call_id: str, output: str, outcome: Outcome) -> None:
        """A call the backend ran ended; its output joins at the next safe point, once."""
        if call_id in self.reported:
            return
        self.reported.add(call_id)
        if outcome == "unclear":  # it may have run: the backend is told so, and checks before it tries again
            self.trace("tools.unclear", call_id=call_id, detail=output[:300])
            output = self.session.texts.owed_output
        self.session.call_ended(call_id, output, outcome)
        self.outputs[call_id] = {"type": "function_call_output", "call_id": call_id, "output": output}
        self._step(call_id, outcome)
        if outcome == "done" and self.steps[call_id][0] == "show" and "show" in self.work_tools.names:
            label = self.session.settings.display_label
            self.shown.append({"type": "mcp_call", "server_label": label, "name": "show", "output": output})
        reports = self.started_calls.get(call_id) or self._reports(self.steps[call_id][0])
        if self.stop_reason is None:
            reports.ended(self, call_id, output, outcome)

    def _reports(self, name: str) -> CallReports:
        """How a call reaches the wire: the display always as an item; the IM's control tools never; others as set."""
        if name in self.displays:
            return DISPLAY_REPORTS
        return CONTROL_REPORTS if name in self.work_tools.names else self.session.call_reports

    def progress(self, text: str) -> None:
        """Activity that is not model text: a progress note (never history)."""
        if text.strip():
            self.record.progress = text
            self.session.progress(self, text)

    def take_boundary_input(self) -> BoundaryInput:
        """A safe point: the outputs (then typed messages) join the history; waiting steering is handed over."""
        order = {call_id: index for index, call_id in enumerate(self.call_order)}
        outputs = sorted(self.outputs.values(), key=lambda item: order.get(item.get("call_id"), len(order)))
        joined, self.outputs, self.messages = [*outputs, *self.messages], {}, []
        if joined:
            self.view.join(joined)
        steering = None
        if self.steering and self.stop_reason is None:
            waiting = self.steering.popleft()
            waiting.record.move("taken")
            steering = Steering(waiting.record.id, tuple(self.work.steering_items(waiting.batch)))
            self.taken[steering.id] = waiting
        return BoundaryInput(tuple(outputs), steering)

    def settle(self, steering: Steering, delivery: Delivery) -> None:
        """What became of a batch the adapter took: ``applied`` lands it; ``not_applied`` keeps it for later."""
        waiting = self.taken.pop(steering.id, None)
        if waiting is None:
            return
        waiting.record.move(delivery)
        if delivery == "applied":
            self.view.join(list(steering.items))
            self.work.land(waiting.batch)
        else:
            self.steering.appendleft(waiting)

    def unapplied(self) -> list[Waiting]:
        """The batches still waiting in this task or taken and not settled, oldest first."""
        return [*self.taken.values(), *self.steering]

    def trace(self, kind: str, **fields: Any) -> None:
        """Note a trace of this task, with its Work, the task and the delegation that carries it."""
        where = {"work_id": self.work_id, "task_id": self.task_id, "delegation_id": self.work.carrier}
        self.session.observe(kind, **where, **fields)

    # ------------------------------------------------------------------------------------------ what it shows
    def _step(self, call_id: str, outcome: str, name: str | None = None) -> None:
        step = self.steps.setdefault(call_id, [name or "", outcome])
        step[1] = outcome
        if name:
            step[0] = name

    def step_line(self, call_id: str) -> str:
        """One call's line: its name and its outcome so far."""
        name, outcome = self.steps[call_id]
        return f"{name}: {outcome}"

    def step_lines(self) -> tuple[str, ...]:
        """One line per call it made, with the call's outcome so far."""
        return tuple(self.step_line(call_id) for call_id in self.steps)

    def summary(self) -> TaskSummary:
        """The task as a conductor and the work records read it, from its record; an ended task says how it ended."""
        record = self.record
        end = record.ended_at if record.ended_at is not None else time.monotonic()
        seconds = end - record.started_at if record.started_at is not None else 0.0
        return TaskSummary(
            record.id, record.state, record.brief, self.step_lines(), record.acted, record.progress, seconds
        )


class WorkRun:
    """One Work: its tasks, its carrier's response, its settings, its steering, its ending."""

    def __init__(self, session: IMSession, record: Work):
        """Run ``record``: its requests (oldest first; the newest carries it) and one task per ``Task`` record."""
        self.session, self.record, self.id = session, record, record.id
        # What its starting requests asked, as they came; requests that join before it starts add theirs.
        self.requested = [item for delegation in record.requests for item in session.request_items(delegation)]
        self.settings = settings_for(record.settings, session.defaults)  # the record's settings, resolved
        self.tasks = [TaskRun(self, task) for task in record.tasks]
        self.input: list[dict] = []  # its input as the tasks read it (records included), made when it starts
        self.batches: list[Batch] = []  # steering that has not landed
        self.deferred: list[str] = []  # takers whose handover waits for the carrier's next response
        self.held: list[Callable[[ResponseWriter], None]] = []  # writes held while its response is closed
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.stop_reason: StopReason | None = None  # a stop of the whole Work
        self.ended = False
        self.done = asyncio.Event()  # set once the Work has ended
        self._backstop: asyncio.TimerHandle | None = None

    @property
    def carrier(self) -> str:
        """The delegation whose response carries the Work's output now."""
        return self.record.carrier

    @property
    def requests(self) -> list[str]:
        """The requests it carries, oldest first, handed-over ones included (its record's)."""
        return self.record.requests

    def take_settings(self, settings: dict) -> None:
        """An explicit request brought its own settings: the Work's response is written with them from now."""
        self.record.settings = deepcopy(settings)
        self.settings = settings_for(self.record.settings, self.session.defaults)

    def open_tasks(self) -> list[TaskRun]:
        """The tasks that have not ended."""
        return [task for task in self.tasks if not task.ended]

    def begin(self) -> None:
        """Start every task's driver; each waits for a slot."""
        for task in self.tasks:
            task.begin()

    # ------------------------------------------------------------------------------------------ starting
    def task_started(self, task: TaskRun) -> BackendRun:
        """A task got its first slot: the Work starts with it if it has not, then the task's run starts."""
        if self.started_at is None:
            self.started_at = time.monotonic()
            self.record.move("running")
            carrier = self.session.ledger.delegations[self.carrier]
            if not carrier.ended:
                carrier.move("running")
            self.input = self.session.work_input(self)
            self.start_response()
            limit = self.session.settings.work_timeout_seconds
            if limit is not None:
                self._backstop = asyncio.get_running_loop().call_later(limit, self.stop_soon, AtBackstop())
            self.session.observe("backend.dequeue", delegation_id=self.carrier, work_id=self.id)
        task.record.started_at = time.monotonic()
        task.view = self.session.history_for(self, task)
        return self.session.backend.start(task)

    def request(self) -> dict:
        """The settings of the next model call of any of its tasks, the instructions composed."""
        return task_request(self.settings, *self.session.instruction_notes())

    def steering_items(self, batch: Batch) -> list[dict]:
        """A batch's requests as a task takes them: the records block and the closing line in the newest one."""
        return with_parts(batch.items, *self.session.records_parts())

    # ------------------------------------------------------------------------------------------ writing
    def write(self, action: Callable[[ResponseWriter], None]) -> None:
        """Write into the carrier's response now, or hold it, in order, until a response of it is in progress."""
        writer = self.session.book.running(self.carrier)
        if writer is None or self.held:
            self.held.append(action)
            return
        action(writer)

    def flush(self) -> None:
        """Write what was held, in order, while the carrier's response is in progress."""
        while self.held:
            writer = self.session.book.running(self.carrier)
            if writer is None:
                return
            self.held.pop(0)(writer)

    def answer(self, task: TaskRun, message: dict) -> None:
        """A task's answer: its own ``final_answer`` item, written at once (or held); never merged with another.

        The task's result is ``final`` with its first answer (its shown displays, then the answer's message), and
        ``pushed`` once that is written into the carrier's response.
        """
        result = task.record.result
        if result.state == "none":
            result.items = [*task.shown, message]
            result.move("final")
        if task.stop_reason is None:
            text = message_text(message)

            def written(writer: ResponseWriter) -> None:
                writer.message(text, "final_answer")
                if result.state == "final":
                    result.move("pushed")

            self.write(written)

    def stop_answer(self, line: str) -> None:
        """A stopped task's answer (``texts.cancelled_line`` or ``unconfirmed_line``), a ``final_answer``.

        It goes on the carrier's open response, after what the other tasks wrote while it was closed; none while that
        response waits for the app, ended with its calls.
        """
        if line and self.session.book.writer(self.carrier) is not None:
            self.start_response()
            self.session.book.running(self.carrier).message(line, "final_answer")

    def start_response(self) -> None:
        """The carrier's open response goes ``in_progress`` (once); what was held while it was closed follows."""
        self.session.book.start(self.carrier)
        self.flush()

    # ------------------------------------------------------------------------------------------ the app's calls
    def park(self, task: TaskRun, calls: list[dict]) -> None:
        """A task's round ended with the app's calls: they end the response (now, or once the held items are out)."""
        self.write(lambda writer: self._send_calls(task, calls, writer))

    def _send_calls(self, task: TaskRun, calls: list[dict], writer: ResponseWriter) -> None:
        if task.stop_reason is not None:
            return  # held while another call waited, and stopped since: nothing of it reaches the app
        for call in calls:
            writer.function_call(call)
        carrier = self.carrier
        self.session.notes.drop(carrier)
        self.session.book.end(carrier, "response.completed")
        self.record.move("waiting_client")
        self.session.ledger.delegations[carrier].move("waiting_client")
        parked = Parked(self, carrier, {call.get("call_id", ""): (task, call) for call in calls})
        self.session.park(parked)

    def continue_with(
        self, parked: Parked, outputs: dict[str, dict], messages: list[dict], brings: dict | None
    ) -> None:
        """A continuation brought the outputs of ``parked``'s calls: the tasks go on in the carrier's next response.

        The continuation is an explicit request: its settings become the Work's. Each waiting task gets its
        calls' outputs, in call order, and the typed messages after them.
        """
        if brings is not None:
            self.take_settings(brings)
        if not self.ended:
            self.record.move("running")
            self.session.ledger.delegations[parked.delegation_id].move("running")
        by_task: dict[TaskRun, list[dict]] = {}
        for call_id, (task, _call) in parked.calls.items():
            by_task.setdefault(task, []).append(outputs[call_id])
        for task, given in by_task.items():
            task.give(given, messages)
        if self.held or any(task.running for task in self.tasks):
            self.resumed()  # held items go out now, also when the task whose calls these were has stopped

    def resumed(self) -> None:
        """A task runs again after the app's outputs: the carrier's response goes in progress, held items follow."""
        self.session.book.start(self.carrier)
        for taker in self.deferred[:]:
            self.deferred.remove(taker)
            self._take_over(taker)
        self.flush()

    # ------------------------------------------------------------------------------------------ steering
    def steer(self, delegations: list[str], task: TaskRun | None, brings: dict | None) -> None:
        """These requests join the Work's open tasks, or one task; a running task takes them at its next safe point."""
        items = [item for delegation in delegations for item in self.session.request_items(delegation)]
        batch = Batch(list(delegations), items, brings)
        if self.started_at is None:
            self.requested.extend(items)
            self.land(batch)
            return
        self.batches.append(batch)
        for target in [task] if task is not None else self.open_tasks():
            if target.view is None:
                target.pending.extend(items)
                self.land(batch)
            else:
                record = self.session.ledger.new_steering(target.task_id, batch.delegations, items)
                target.steering.append(Waiting(batch, record))
                if target.running:
                    target.run.steer()

    def land(self, batch: Batch) -> None:
        """A batch landed: each of its requests takes the Work over in turn, and the Work keeps or takes settings."""
        if batch.landed:
            return
        batch.landed = True
        if batch in self.batches:
            self.batches.remove(batch)
        for taker in batch.delegations:
            self.requests.append(taker)
            self._take_over(taker)
        if batch.brings is not None:
            self.take_settings(batch.brings)

    def take_over_at_start(self, takers: list[str]) -> None:
        """The newer requests of a new Work take it over in turn, while it is queued; each brings its input."""
        for taker in takers:
            self.requests.append(taker)
            self.requested.extend(self.session.request_items(taker))
            self._take_over(taker)

    def _take_over(self, taker: str) -> None:
        """``taker`` carries the Work from now on: the handover that names it, written in order.

        While the carrier's response waits for the app (it ended with the app's calls), the handover waits for the
        carrier's next response, which the app's outputs open (``resumed``), or for the client-tool deadline.
        """
        if self.session.book.writer(self.carrier) is None:
            self.deferred.append(taker)
            return
        self._hand_over(taker)

    def _hand_over(self, taker: str) -> None:
        """Write the handover: the taker's response goes in progress naming the carrier's newest response.

        An older response still open then ends with ``texts.handover_line``, which names the taker, as ``commentary``;
        one that ended with the app's calls (at the client-tool deadline) gets nothing more.
        """
        book, ledger, older = self.session.book, self.session.ledger, self.carrier
        book.start(taker, previous=older)
        if book.writer(older) is not None:
            book.start(older)
            book.running(older).message(self.session.texts.handover_line.format(delegation=taker), "commentary")
            self.session.notes.drop(older)
            book.end(older, "response.completed")
        older_record = ledger.delegations[older]
        older_record.handed_to = taker
        older_record.move("handed_over")
        self.record.carrier = taker
        if self.started_at is not None:
            ledger.delegations[taker].move("running")
        self.session.observe("backend.handed_over", delegation_id=older, taker=taker, work_id=self.id)

    # ------------------------------------------------------------------------------------------ stopping
    def stop_soon(self, reason: StopReason, task: TaskRun | None = None) -> None:
        """Ask the stop door now (nothing more is written, calls are refused); confirm it in the background."""
        asked = self.ask_stop(reason, task)
        if asked:
            self.session.spawn(self.confirm_stop(asked))

    async def stop(self, reason: StopReason, task: TaskRun | None = None) -> None:
        """The stop door for the whole Work, or for one task of it (a ``Cancel`` may name one)."""
        await self.confirm_stop(self.ask_stop(reason, task))

    def ask_stop(self, reason: StopReason, task: TaskRun | None = None) -> list[TaskRun]:
        """Ask the tasks to stop for ``reason``; return those that were still running and are stopping now.

        Once no task of the Work goes on, its responses waiting for the app stop waiting and later outputs are late;
        while another task goes on they keep waiting, since only the app's outputs open the carrier's next response.
        """
        targets = [task] if task is not None else self.open_tasks()
        whole = all(each in targets for each in self.open_tasks())
        asked = [each for each in targets if each.request_stop(reason)]
        if asked and whole:
            self.stop_reason = reason
        if all(each.stop_reason is not None for each in self.open_tasks()):
            for parked in {parked for parked in self.session.parked.values() if parked.work is self}:
                self.session.unpark(parked)
        return asked

    async def confirm_stop(self, asked: list[TaskRun]) -> None:
        """Wait for each asked task to stop (``stopped`` or ``unconfirmed``)."""
        await asyncio.gather(*(each.confirm_stop() for each in asked))

    def park_timer(self, parked: Parked) -> None:
        """Start the client-tool deadline of a parked response (``None``: it waits until the close)."""
        limit = self.session.settings.client_tool_timeout_seconds
        if limit is not None:
            parked.timer = asyncio.get_running_loop().call_later(limit, self.deadline, parked)

    def deadline(self, parked: Parked) -> None:
        """The app never answered ``parked``'s calls in time: the Work stops, unless a newer request carries it.

        With no newer request landed in the Work, the whole Work stops and fails (``AtDeadline``), writing nothing. A
        request that landed meanwhile in another task takes the Work over instead, and the stuck task stops with the
        outcome-unknown line, so nothing the request asked runs twice.
        """
        if self.ended:
            return
        if not self.deferred:
            self.stop_soon(AtDeadline())
            return
        self.session.unpark(parked)  # the outputs that come from now on are late
        self.record.move("running")
        for taker in self.deferred[:]:
            self.deferred.remove(taker)
            self._hand_over(taker)
        self.flush()
        waiting = dict.fromkeys(task for task, _ in parked.calls.values())  # the tasks whose calls it holds
        stuck = [task for task in waiting if task.request_stop(AtDeadline())]  # these stop; the Work goes on
        if stuck:
            self.session.spawn(self.confirm_stop(stuck))

    # ------------------------------------------------------------------------------------------ ending
    def task_ended(self, task: TaskRun) -> None:
        """A task ended; the Work ends with its last task."""
        if not self.open_tasks():
            self._finish()

    def _finish(self) -> None:
        if self.ended:
            return
        self.ended, self.ended_at = True, time.monotonic()
        if self._backstop is not None:
            self._backstop.cancel()
        session, book, carrier = self.session, self.session.book, self.carrier
        failure = next((task.failure for task in self.tasks if task.failure is not None), None)
        completed = any(task.completed for task in self.tasks)
        session.notes.drop(carrier)
        if self.stop_reason is not None:
            state = self.stop_reason.work_state
            self.stop_reason.end(book, carrier)
        elif failure is not None:
            state = "failed"
            self.start_response()
            book.end(carrier, failure.kind, **failure.fields)
        else:
            state = "answered" if completed else "cancelled"
            self.start_response()
            book.end(carrier, "response.completed")
        self.record.move(state)
        record = session.ledger.delegations.get(carrier)
        if record is not None and not record.ended:
            record.move(state)
        for task in self.tasks:  # steering no task applied: its requests start a follow-up Work
            for waiting in task.unapplied():
                if not waiting.batch.landed:
                    waiting.record.move("follow_up")
        leftovers = [d for batch in self.batches for d in batch.delegations] + self.deferred
        session.work_ended(self, leftovers)
        self.done.set()

    def segment(self) -> list[dict]:
        """What the Work adds to the shared conversation once it ended: its requests, then each task's own items.

        A call left without an output in a task's view gets the output the task holds for it, which no safe point took
        (a call that ended while its task stopped), or ``texts.owed_output`` when the task holds none.
        """
        items = list(self.requested)
        for task in self.tasks:
            if task.view is not None:
                task.view.join(task.view.owed(self.session.texts.owed_output, task.outputs))
                items += task.view.own()
        return items

    # ------------------------------------------------------------------------------------------ what it shows
    def summary(self) -> WorkSummary:
        """The Work, open or ended, as a conductor reads it: its requests and every task, ended ones included."""
        requests = tuple(Request(d, tuple(self.session.request_items(d))) for d in self.requests)
        tasks = tuple(task.summary() for task in self.tasks)
        end = self.ended_at if self.ended_at is not None else time.monotonic()
        return WorkSummary(self.id, self.record.state, self.carrier, requests, tasks, end - self.record.queued_at)

    def record_line(self) -> dict:
        """The Work as the records block shows it to the backend: one JSON object."""
        steps = [line for task in self.tasks for line in task.step_lines()]
        notes = [task.record.progress for task in self.tasks if task.record.progress]
        texts = [request_text(self.session.request_items(d)) for d in self.requests]
        line: dict[str, Any] = {"work": self.id, "state": self.record.state, "requests": texts, "steps": steps}
        if notes:
            line["progress"] = notes[-1]
        return line


def brief_message(template: str, brief: str) -> dict:
    """A conductor's brief for one task, as ``texts.task_line`` in a user message of its own."""
    return user_message(template.format(task=brief))
