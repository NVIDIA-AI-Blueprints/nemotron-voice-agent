# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Works and tasks: what runs, what it reads, and how it reaches the wire."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from examples.frontend_backend_live.delegation_core.backend.port import (
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
    allowed_tools,
    listed,
    message_phase,
)
from examples.frontend_backend_live.delegation_core.im.conversation import TaskHistory
from examples.frontend_backend_live.delegation_core.im.input import settings_for, task_request, user_message, with_parts
from examples.frontend_backend_live.delegation_core.im.results import CallReports, ControlCalls
from examples.frontend_backend_live.delegation_core.im.stop import AtBackstop, AtDeadline, StopReason
from examples.frontend_backend_live.delegation_core.im.stream import ResponseWriter
from examples.frontend_backend_live.delegation_core.im.tools import WorkTools
from examples.frontend_backend_live.delegation_core.items import message_text
from examples.frontend_backend_live.delegation_core.ledger.records import Runner, Task, Work
from examples.frontend_backend_live.delegation_core.ledger.records import Steering as SteeringRecord

if TYPE_CHECKING:
    from examples.frontend_backend_live.delegation_core.config import Texts
    from examples.frontend_backend_live.delegation_core.im.session import IMSession

# The reasons of a cut attempt that end the response incomplete; any other failure ends it failed.
OFFICIAL_REASONS = frozenset({"max_output_tokens", "content_filter"})
TOO_MANY_REQUESTS = 429
RECORD_TEXT_CHARS = 300
ITEM_REPORTS = CallReports()
CONTROL_REPORTS = ControlCalls()


@dataclass(frozen=True)
class Failure:
    """How a failed task ends its Work's response: ``response.failed`` or ``response.incomplete``."""

    kind: str
    fields: dict

    @classmethod
    def of(cls, error: BaseException, terminal: dict | None) -> Failure:
        """Return the failure ``error`` gives, read from the task's last attempt when it ended with one."""
        response = (terminal or {}).get("response") or {}
        reason = (response.get("incomplete_details") or {}).get("reason")
        if reason in OFFICIAL_REASONS:
            return cls("response.incomplete", {"incomplete_details": {"reason": reason}})
        detail = f"{type(error).__name__}: {error}"[:300]
        # A 429 wins over the attempt's code, which the run may have set to server_error before it retried.
        limited = getattr(error, "status_code", None) == TOO_MANY_REQUESTS
        code = "rate_limit_exceeded" if limited else (response.get("error") or {}).get("code") or "server_error"
        return cls("response.failed", {"error": {"code": code, "message": detail}})


NO_ANSWER = Failure("response.failed", {"error": {"code": "server_error", "message": "The backend gave no answer."}})


@dataclass(eq=False)
class Batch:
    """Requests steered into a Work: one ``Steering`` record per task, landed once by the first task to apply it."""

    delegations: list[str]
    items: list[dict]
    brings: dict | None
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
    calls: dict[str, tuple[TaskRun, dict]]
    timer: asyncio.TimerHandle | None = field(default=None, repr=False)


def request_text(items: list[dict]) -> str:
    """Return the newest words of a request's user messages, as the work records show them."""
    texts = []
    for item in items:
        content = item.get("content") if item.get("role") == "user" else None
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts += [part.get("text", "") for part in content if isinstance(part, dict) and part.get("text")]
    return " ".join(texts)[-RECORD_TEXT_CHARS:]


class TaskRun:
    """One task: one worker's run, and the ``TaskContext`` it reads and reports through."""

    def __init__(self, work: WorkRun, record: Task):
        """Make a task of ``work``, run for its ``Task`` record."""
        self.work, self.session, self.record = work, work.session, record
        self.work_id, self.task_id, self.brief = work.id, record.id, record.brief
        self.ended = False
        self.run: BackendRun | None = None
        self.view: TaskHistory | None = None
        self.pending: list[dict] = []
        self.steering: deque[Waiting] = deque()
        self.taken: dict[str, Waiting] = {}
        self.outputs: dict[str, dict] = {}
        self.call_order: list[str] = []
        self.routes: dict[str, Route] = {}
        self.refusals: dict[str, str] = {}
        self.app_calls: dict[str, dict] = {}
        self.started_calls: dict[str, CallReports] = {}
        self.reported: set[str] = set()
        self.steps: dict[str, list[str]] = {}
        self.attempt: list[dict] = []
        self.last_terminal: dict | None = None
        self.answered = False
        self.held = False
        self.completed = False
        self.stop_reason: StopReason | None = None
        self.stopped: StopOutcome | None = None
        self.own_cancel = False
        self.failure: Failure | None = None
        self.grant: asyncio.Future | None = None
        self.outputs_in: asyncio.Future | None = None
        self.driver: asyncio.Task | None = None
        shown = self.session.result_format.tools()
        self.displays = frozenset(shown)
        names = tuple(dict.fromkeys((*shown, *self.session.work_tools)))
        self.work_tools = WorkTools(names, self.texts, self.cancel_own, self.session.settings.display_label)
        self._readers: dict[str, Callable[[dict], None]] = {
            "response.created": self._attempt_opened,
            "response.output_item.done": self._item_done,
            "response.completed": self._round_completed,
            "response.failed": self._attempt_failed,
            "response.incomplete": self._attempt_failed,
        }

    def begin(self) -> None:
        """Start the task's driver: it waits for a slot, then runs."""
        self.driver = asyncio.create_task(self._drive(), name=f"task-{self.task_id}")

    async def _drive(self) -> None:
        try:
            await self._slot()
            self.run = self.work.task_started(self)
            while True:
                await self._read_events()
                if self.stop_reason is not None or not self.app_calls:
                    break
                if self._gated():
                    # The gate refused the app's calls: the run goes on with the newer input once it is placed.
                    await self.session.decided()
                    if self.stop_reason is not None:
                        break
                    await self.run.resume()
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
        """Whether its run holds a slot."""
        return self.record.state == "running"

    async def _slot(self) -> None:
        """Wait for a slot, then move to ``running``."""
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
                # The round's calls were refused while the conductor decides: read on once it has.
                if self.held:
                    self.held = False
                    await self.session.decided()
                    if self.stop_reason is not None:
                        return
        finally:
            close = getattr(events, "aclose", None)
            if close is not None:
                await close()

    def _gated(self) -> bool:
        """Apply the commit gate to the round's app calls as they would go out; return whether it refused them.

        Their route was decided when the round completed, and newer input may have come while its backend calls ran.
        """
        if not self.newer_input_waits():
            return False
        for call_id, call in self.app_calls.items():
            self.routes[call_id] = "refused"
            self.refusals[call_id] = self.session.texts.not_run_output
            self._refuse(call)
        self.app_calls = {}
        return True

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

    def give(self, outputs: list[dict]) -> None:
        """Take the app's outputs for this task's calls."""
        for output in outputs:
            self.outputs[output["call_id"]] = output
            self._step(output["call_id"], "done")
        if self.outputs_in is not None and not self.outputs_in.done():
            self.outputs_in.set_result(None)

    def _end(self) -> None:
        """End the task: its record moves to how it ended."""
        if self.ended:
            return
        self.ended = True
        record = self.record
        state = "failed" if self.failure is not None else "answered" if self.completed else "cancelled"
        record.move(state)
        record.ended_at = time.monotonic()
        # No response's end waits for calls it never reported ended.
        self.session.book.drop(self.unreported())
        self.work.task_ended(self)

    def request_stop(self, reason: StopReason) -> bool:
        """Ask the task to stop for ``reason``; ``False`` when it had ended or a stop was asked already."""
        if self.ended or self.stop_reason is not None:
            return False
        self.stop_reason = reason
        self.record.stop.move("stop_requested")
        self.session.unschedule(self)
        self.session.book.drop(self.unreported())
        return True

    def unreported(self) -> list[str]:
        """Return the backend calls it reported started and not ended."""
        return [call_id for call_id in self.started_calls if call_id not in self.reported]

    async def confirm_stop(self) -> StopOutcome:
        """Stop the run within ``stop_timeout_seconds``, then write the reason's line and end the task."""
        outcome: StopOutcome = "stopped"
        if self.run is not None:
            try:
                async with asyncio.timeout(self.session.settings.stop_timeout_seconds):
                    outcome = await self.run.stop()
            except Exception as exc:
                logger.warning(f"Task {self.task_id} did not confirm its stop: {type(exc).__name__}: {exc}")
                outcome = "unconfirmed"
        self.stopped = outcome
        self.record.stop.move(outcome)
        if self.driver is not None and not self.driver.done():
            self.driver.cancel()
            await asyncio.gather(self.driver, return_exceptions=True)
        self.work.stop_answer(self.stop_reason.answer(self.session.texts, outcome))
        self._end()
        return outcome

    def cancel_own(self) -> str:
        """Stop after this step, for the task's own ``cancel_work``: later calls are refused, its next answer counts."""
        self.own_cancel = True
        return self.session.texts.outcomes["cancel_requested"].format(work=self.work_id, tasks=self.task_id)

    def _attempt_opened(self, raw: dict) -> None:
        self.attempt, self.last_terminal = [], None

    def _item_done(self, raw: dict) -> None:
        item = raw.get("item")
        if isinstance(item, dict):
            self.attempt.append(item)

    def _attempt_failed(self, raw: dict) -> None:
        # A failed or cut attempt writes nothing.
        self.attempt, self.last_terminal = [], raw

    def _round_completed(self, raw: dict) -> None:
        items, self.attempt, self.last_terminal = self.attempt, [], None
        usage = (raw.get("response") or {}).get("usage")
        writer = self.session.book.writer(self.work.carrier)
        if usage and writer is not None:
            writer.usage = usage
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
            if self.route(call) == "app":
                self.app_calls[call.get("call_id", "")] = call
                self.session.action(self, call)
        refused = any(self.routes.get(call.get("call_id", "")) == "refused" for call in calls)
        self.held = refused and self.session.waits_for_decision and self.stop_reason is None
        if answer is not None:
            self._answer(answer)

    def _answer(self, message: dict) -> None:
        """Mark the round answered, and have the Work write ``message``."""
        self.answered = True
        self.trace("im.answer.written")
        self.work.answer(self, message)

    @property
    def texts(self) -> Texts:
        """The core's texts."""
        return self.session.texts

    def request(self) -> dict:
        """Return the settings of the task's next model call, its instructions composed."""
        return self.work.request()

    def history(self) -> list[dict]:
        """Return the task's view of the conversation, within the budget."""
        return self.view.history()

    def tools(self) -> WorkTools:
        """Return the IM's work tools for this task."""
        return self.work_tools

    def route(self, call: dict) -> Route:
        """Return who runs ``call``, decided once per call, when the adapter or the round reader first asks."""
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
        """Return the not-run output the IM gave a call it refused."""
        return self.refusals[call_id]

    def _refusal(self, call: dict) -> str | None:
        """Return the not-run output that says why the IM will not run ``call``, or ``None`` when it runs.

        The reasons, in order: a stop or the task's own ``cancel_work``; a function the app and a work tool both name;
        an app function the ``allowed_tools`` tool choice leaves out; newer input waiting (the commit gate); an app
        call that repeats the newest call that ran, which this task made and the app answered, with no request since.
        """
        texts, name, settings = self.session.texts, call.get("name", ""), self.work.settings
        declared = [tool for tool in settings.tools if tool.get("type") == "function" and tool.get("name") == name]
        if self.stop_reason is not None or self.own_cancel:
            return texts.stopped_output
        if declared and name in self.work_tools.names:
            return texts.clash_output
        allowed = allowed_tools(settings.tool_choice)
        if declared and not any(listed(tool, allowed) for tool in declared):
            return texts.unlisted_output
        if self.newer_input_waits():
            return texts.not_run_output
        # A Work that took another request may ask for the same call again.
        if not declared or len(self.work.requests) > 1:
            return None
        return self.session.repeated_output(call, self.task_id)

    def _route(self, name: str) -> Route:
        """Return who runs a call the IM lets run: the app for a function the request declared, else the backend."""
        declared = any(tool.get("type") == "function" and tool.get("name") == name for tool in self.work.settings.tools)
        return "app" if declared else "backend"

    def newer_input_waits(self) -> bool:
        """Return whether newer input waits: to land in this task, or for the conductor's decision on any Work.

        This is the commit gate: the backend re-plans with that input first, and a call already running is never cut.
        """
        return bool(self.steering or self.taken) or self.session.waits_for_decision

    def runner(self, name: str) -> Runner:
        """Return who runs a call of ``name``, as its action records it, also for a refused call."""
        if name in self.work_tools.names:
            return "im"
        declared = any(tool.get("type") == "function" and tool.get("name") == name for tool in self.work.settings.tools)
        return "app" if declared else "backend"

    def _refuse(self, call: dict) -> None:
        """Refuse ``call``: its action is ``not_run``, and its output is the not-run output that says why."""
        call_id = call.get("call_id", "")
        output = self.refusals[call_id]
        self.outputs[call_id] = {"type": "function_call_output", "call_id": call_id, "output": output}
        self.session.action(self, call, "not_run")
        self._step(call_id, "not_run", call.get("name", ""))

    def tool_started(self, call: dict, source: str) -> None:
        """Report a backend call as it starts: an ``mcp_call`` item, unless it is one of the IM's control tools."""
        call_id = call.get("call_id", "")
        if call_id in self.started_calls or self.routes.get(call_id) == "refused":
            return
        if call_id not in self.routes:
            self.routes[call_id] = "backend"
            self.session.call_routed(call)
        self.started_calls[call_id] = self._reports(call.get("name", ""))
        self.record.acted = True
        self.session.action(self, call)
        self._step(call_id, "started", call.get("name", ""))
        if self.stop_reason is None:
            self.started_calls[call_id].started(self, call, source)

    def tool_outcome(self, call_id: str, output: str, outcome: Outcome) -> None:
        """Report that a backend call ended; its output joins at the next safe point, once."""
        if call_id in self.reported:
            return
        self.reported.add(call_id)
        if outcome == "unclear":
            self.trace("tools.unclear", call_id=call_id, detail=output[:300])
            output = self.session.texts.owed_output
        self.session.call_ended(call_id, output, outcome)
        self.outputs[call_id] = {"type": "function_call_output", "call_id": call_id, "output": output}
        self._step(call_id, outcome)
        reports = self.started_calls.get(call_id) or self._reports(self.steps[call_id][0])
        if self.stop_reason is None:
            reports.ended(self, call_id, output, outcome)

    def _reports(self, name: str) -> CallReports:
        """Return how a call reaches the wire: as an item, except the IM's control tools, which never do."""
        return CONTROL_REPORTS if name in self.work_tools.names and name not in self.displays else ITEM_REPORTS

    def progress(self, text: str) -> None:
        """Report activity that is not model text: a progress note, never history."""
        if text.strip():
            self.record.progress = text
            self.session.progress(self, text)

    def take_boundary_input(self) -> BoundaryInput:
        """Join the outputs to the history and hand over waiting steering: a safe point."""
        order = {call_id: index for index, call_id in enumerate(self.call_order)}
        outputs = sorted(self.outputs.values(), key=lambda item: order.get(item.get("call_id"), len(order)))
        self.outputs = {}
        if outputs:
            self.view.join(outputs)
        steering = None
        if self.steering and self.stop_reason is None:
            waiting = self.steering.popleft()
            waiting.record.move("taken")
            steering = Steering(waiting.record.id, tuple(self.work.steering_items(waiting.batch)))
            self.taken[steering.id] = waiting
        return BoundaryInput(tuple(outputs), steering)

    def settle(self, steering: Steering, delivery: Delivery) -> None:
        """Settle a batch the adapter took: ``applied`` lands it, ``not_applied`` keeps it for later."""
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
        """Return the batches still waiting in this task, or taken and not settled, oldest first."""
        return [*self.taken.values(), *self.steering]

    def trace(self, kind: str, **fields: Any) -> None:
        """Note a trace of this task, with its Work, the task and the delegation that carries it."""
        where = {"work_id": self.work_id, "task_id": self.task_id, "delegation_id": self.work.carrier}
        self.session.observe(kind, **where, **fields)

    def _step(self, call_id: str, outcome: str, name: str | None = None) -> None:
        step = self.steps.setdefault(call_id, [name or "", outcome])
        step[1] = outcome
        if name:
            step[0] = name

    def step_line(self, call_id: str) -> str:
        """Return one call's line: its name and its outcome so far."""
        name, outcome = self.steps[call_id]
        return f"{name}: {outcome}"

    def step_lines(self) -> tuple[str, ...]:
        """Return one line per call it made, with the call's outcome so far."""
        return tuple(self.step_line(call_id) for call_id in self.steps)

    def summary(self) -> TaskSummary:
        """Return the task as a conductor and the work records read it; an ended task says how it ended."""
        record = self.record
        end = record.ended_at if record.ended_at is not None else time.monotonic()
        seconds = end - record.started_at if record.started_at is not None else 0.0
        return TaskSummary(
            record.id, record.state, record.brief, self.step_lines(), record.acted, record.progress, seconds
        )


class WorkRun:
    """One Work: its tasks, its carrier's response, its settings, its steering, its ending."""

    def __init__(self, session: IMSession, record: Work):
        """Make the run of ``record``, with one task per ``Task`` record."""
        self.session, self.record, self.id = session, record, record.id
        self.requested = [item for delegation in record.requests for item in session.request_items(delegation)]
        self.settings = settings_for(record.settings, session.defaults)
        self.tasks = [TaskRun(self, task) for task in record.tasks]
        self.input: list[dict] = []
        self.batches: list[Batch] = []
        self.deferred: list[str] = []
        self.held: list[Callable[[ResponseWriter], None]] = []
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.stop_reason: StopReason | None = None
        self.ended = False
        self.done = asyncio.Event()
        self._backstop: asyncio.TimerHandle | None = None

    @property
    def carrier(self) -> str:
        """The delegation whose response carries the Work's output now."""
        return self.record.carrier

    @property
    def requests(self) -> list[str]:
        """The requests it carries, in the order they joined it, handed-over ones included."""
        return self.record.requests

    def take_settings(self, settings: dict) -> None:
        """Take an explicit request's own settings: the Work's response is written with them from then on."""
        self.record.settings = deepcopy(settings)
        self.settings = settings_for(self.record.settings, self.session.defaults)

    def open_tasks(self) -> list[TaskRun]:
        """Return the tasks that have not ended."""
        return [task for task in self.tasks if not task.ended]

    def parked(self) -> set[Parked]:
        """Return its responses that ended with the app's calls and wait for their outputs."""
        return {parked for parked in self.session.parked.values() if parked.work is self}

    def begin(self) -> None:
        """Start every task's driver; each waits for a slot."""
        for task in self.tasks:
            task.begin()

    def task_started(self, task: TaskRun) -> BackendRun:
        """Start ``task``'s run on its first slot, and the Work with it if the Work has not started."""
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
        """Return the settings of the next model call of any of its tasks, the instructions composed."""
        return task_request(self.settings, *self.session.instruction_notes())

    def steering_items(self, batch: Batch) -> list[dict]:
        """Return a batch's requests as a task takes them: the records block and the closing line in the newest one."""
        if not batch.delegations:
            return list(batch.items)
        return with_parts(batch.items, *self.session.records_parts())

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
        """Write a task's answer as its own ``final_answer`` item, at once or held; it is never merged with another."""
        result = task.record.result
        if result.state == "none":
            result.move("final")
        if task.stop_reason is None:
            text = message_text(message)

            def written(writer: ResponseWriter) -> None:
                writer.message(text, "final_answer")
                if result.state == "final":
                    result.move("pushed")

            self.write(written)

    def stop_answer(self, line: str) -> None:
        """Write a stopped task's answer as a ``final_answer`` on the carrier's open response, after what was held."""
        if line and self.session.book.writer(self.carrier) is not None:
            self.start_response()
            self.session.book.running(self.carrier).message(line, "final_answer")

    def start_response(self) -> None:
        """Put the carrier's open response in progress (once); what was held while it was closed follows."""
        self.session.book.start(self.carrier)
        self.flush()

    def park(self, task: TaskRun, calls: list[dict]) -> None:
        """Write a task's app calls, at once or after the held items; the response ends once its backend calls end."""
        self.write(lambda writer: self._send_calls(task, calls, writer))

    def _send_calls(self, task: TaskRun, calls: list[dict], writer: ResponseWriter) -> None:
        # Held while another call waited, and stopped since: nothing of it reaches the app.
        if task.stop_reason is not None:
            return
        for call in calls:
            writer.function_call(call)
        carrier = self.carrier
        self.session.notes.drop(carrier)
        self.record.move("waiting_client")
        self.session.ledger.delegations[carrier].move("waiting_client")
        parked = Parked(self, carrier, {call.get("call_id", ""): (task, call) for call in calls})
        self.session.park(parked)
        self.session.book.end_after_calls(carrier, "response.completed", ended=lambda: self.park_timer(parked))

    def continue_with(
        self, parked: Parked, outputs: dict[str, dict], messages: list[dict], brings: dict | None
    ) -> None:
        """Go on with ``parked``'s tasks in the carrier's next response: their outputs, then the typed messages."""
        if brings is not None:
            self.take_settings(brings)
        if not self.ended:
            self.record.move("running")
            self.session.ledger.delegations[parked.delegation_id].move("running")
        by_task: dict[TaskRun, list[dict]] = {}
        for call_id, (task, _call) in parked.calls.items():
            by_task.setdefault(task, []).append(outputs[call_id])
        for task, given in by_task.items():
            task.give(given)
        if messages:
            self.take_typed(list(by_task), messages)
        if self.held or any(task.running for task in self.tasks):
            self.resumed()
        if not self.open_tasks():
            self._finish()

    def resumed(self) -> None:
        """Reopen the carrier's response for a task that runs again after the app's outputs; held items follow."""
        self.session.book.start(self.carrier)
        for taker in self.deferred[:]:
            self.deferred.remove(taker)
            self._take_over(taker)
        self.flush()

    def steer(self, delegations: list[str], task: TaskRun | None, brings: dict | None) -> None:
        """Join these requests to the Work's open tasks, or to one; a running task takes them at its next safe point."""
        items = [item for delegation in delegations for item in self.session.request_items(delegation)]
        self._steer(Batch(list(delegations), items, brings), [task] if task is not None else self.open_tasks())

    def take_typed(self, waiting: list[TaskRun], messages: list[dict]) -> None:
        """Steer the open tasks with typed messages that came with outputs; the ``waiting`` tasks take them first.

        They are no request: no records block, handover or follow-up Work; untaken, they join the conversation last.
        """
        batch = Batch([], list(messages), None)
        first = [task for task in waiting if task.stop_reason is None and not task.ended]
        self._steer(batch, [task for task in self.open_tasks() if task not in first])
        for task in first:
            record = self.session.ledger.new_steering(task.task_id)
            task.steering.appendleft(Waiting(batch, record))

    def _steer(self, batch: Batch, targets: list[TaskRun]) -> None:
        """Join ``batch`` to ``targets``, or to the Work's input before the Work starts."""
        if self.started_at is None:
            self.requested.extend(batch.items)
            self.land(batch)
            return
        self.batches.append(batch)
        for target in targets:
            if target.view is None:
                target.pending.extend(batch.items)
                self.land(batch)
            else:
                record = self.session.ledger.new_steering(target.task_id)
                target.steering.append(Waiting(batch, record))
                if target.running:
                    target.run.steer()

    def land(self, batch: Batch) -> None:
        """Land a batch: each of its requests takes the Work over in turn, and the Work keeps or takes settings."""
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
        """Let the newer requests of a queued Work take it over in turn, each bringing its input."""
        for taker in takers:
            self.requests.append(taker)
            self.requested.extend(self.session.request_items(taker))
            self._take_over(taker)

    def _take_over(self, taker: str) -> None:
        """Make ``taker`` carry the Work: hand over at once, or later if the carrier's response waits for the app."""
        if self.session.book.writer(self.carrier) is None:
            self.deferred.append(taker)
            return
        self._hand_over(taker)

    def _hand_over(self, taker: str) -> None:
        """Hand the Work to ``taker``: its response starts, and the carrier's open one ends with the handover line."""
        book, ledger, older = self.session.book, self.session.ledger, self.carrier
        book.start(taker, previous=older)
        if book.writer(older) is not None:
            book.start(older)
            line = self.session.texts.handover_line.format(delegation=taker)
            self.session.notes.drop(older)
            book.end_after_calls(older, "response.completed", last=lambda writer: writer.message(line, "commentary"))
        ledger.delegations[older].move("handed_over")
        self.record.carrier = taker
        if self.started_at is not None:
            ledger.delegations[taker].move("running")
        self.session.observe("backend.handed_over", delegation_id=older, taker=taker, work_id=self.id)

    def stop_soon(self, reason: StopReason, task: TaskRun | None = None) -> None:
        """Ask the stop door at once (nothing more is written, calls are refused); confirm it in the background."""
        asked = self.ask_stop(reason, task)
        if asked:
            self.session.spawn(self.confirm_stop(asked))

    async def stop(self, reason: StopReason, task: TaskRun | None = None) -> None:
        """Stop the whole Work, or one task of it, through the stop door."""
        await self.confirm_stop(self.ask_stop(reason, task))

    def ask_stop(self, reason: StopReason, task: TaskRun | None = None) -> list[TaskRun]:
        """Ask the tasks to stop for ``reason``; return those that were still running and are stopping now.

        Once no task goes on, its responses stop waiting for the app, a request that landed meanwhile takes the Work
        over, and the Work ends at once when no task is open; while another task goes on, they keep waiting.
        """
        targets = [task] if task is not None else self.open_tasks()
        whole = all(each in targets for each in self.open_tasks())
        asked = [each for each in targets if each.request_stop(reason)]
        if asked and whole:
            self.stop_reason = reason
        parked = self.parked()
        if parked and all(each.stop_reason is not None for each in self.open_tasks()):
            if self.stop_reason is None:
                self.stop_reason = reason
            for each in parked:
                self.session.unpark(each)
            for taker in self.deferred[:]:
                self.deferred.remove(taker)
                self._hand_over(taker)
            self.flush()
            if not self.open_tasks():
                self._finish()
        return asked

    async def confirm_stop(self, asked: list[TaskRun]) -> None:
        """Wait for each asked task to stop (``stopped`` or ``unconfirmed``)."""
        await asyncio.gather(*(each.confirm_stop() for each in asked))

    def park_timer(self, parked: Parked) -> None:
        """Start a parked response's client-tool deadline once it ended, unless its calls' outputs came already."""
        limit = self.session.settings.client_tool_timeout_seconds
        owed = any(self.session.parked.get(call_id) is parked for call_id in parked.calls)
        if limit is not None and owed:
            parked.timer = asyncio.get_running_loop().call_later(limit, self.deadline, parked)

    def deadline(self, parked: Parked) -> None:
        """Stop the Work when the app never answered ``parked``'s calls in time, unless another request carries it.

        A request that landed meanwhile takes the Work over, and the stuck tasks stop, so nothing it asked runs twice.
        """
        if self.ended:
            return
        if not self.deferred:
            self.stop_soon(AtDeadline())
            return
        self.session.unpark(parked)
        self.record.move("running")
        for taker in self.deferred[:]:
            self.deferred.remove(taker)
            self._hand_over(taker)
        self.flush()
        waiting = dict.fromkeys(task for task, _ in parked.calls.values())
        stuck = [task for task in waiting if task.request_stop(AtDeadline())]
        if stuck:
            self.session.spawn(self.confirm_stop(stuck))
        if not self.open_tasks():
            self._finish()

    def task_ended(self, task: TaskRun) -> None:
        """End the Work with its last task, unless held items wait for the response the app's outputs open next."""
        if not self.open_tasks() and not (self.held and self.parked()):
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
        for task in self.tasks:
            for waiting in task.unapplied():
                if not waiting.batch.landed:
                    waiting.record.move("follow_up")
        leftovers = [d for batch in self.batches for d in batch.delegations] + self.deferred
        session.work_ended(self, leftovers)
        self.done.set()

    def segment(self) -> list[dict]:
        """Return what the ended Work adds to the conversation: its requests, then each task's own items.

        A call left without an output gets the one its task holds, else ``texts.owed_output``; untaken typed messages
        come last.
        """
        items = list(self.requested)
        for task in self.tasks:
            if task.view is not None:
                task.view.join(task.view.owed(self.session.texts.owed_output, task.outputs))
                items += task.view.own()
        return items + [item for batch in self.batches if not batch.delegations for item in batch.items]

    def summary(self) -> WorkSummary:
        """Return the Work, open or ended, as a conductor reads it: its requests and every task."""
        requests = tuple(Request(d, tuple(self.session.request_items(d))) for d in self.requests)
        tasks = tuple(task.summary() for task in self.tasks)
        end = self.ended_at if self.ended_at is not None else time.monotonic()
        return WorkSummary(self.id, self.record.state, self.carrier, requests, tasks, end - self.record.queued_at)

    def record_line(self) -> dict:
        """Return the Work as the records block shows it to the backend: one JSON object."""
        steps = [line for task in self.tasks for line in task.step_lines()]
        notes = [task.record.progress for task in self.tasks if task.record.progress]
        texts = [request_text(self.session.request_items(d)) for d in self.requests]
        line: dict[str, Any] = {"work": self.id, "state": self.record.state, "requests": texts, "steps": steps}
        if notes:
            line["progress"] = notes[-1]
        return line


def brief_message(template: str, brief: str) -> dict:
    """Return a conductor's brief for one task, as ``texts.task_line`` in a user message of its own."""
    return user_message(template.format(task=brief))
