# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One IM session: admission, placement through the conductor, the scheduler, close."""

from __future__ import annotations

import asyncio
import contextvars
import json
from collections import deque
from collections.abc import Callable, Coroutine, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from examples.frontend_backend_live.delegation_core.backend.port import (
    Answer,
    AppOutputStep,
    BackendAdapter,
    Cancel,
    Conductor,
    ConductorInput,
    Decision,
    Outcome,
    Request,
    Start,
    Steer,
)
from examples.frontend_backend_live.delegation_core.config import DelegationSettings, Texts
from examples.frontend_backend_live.delegation_core.im.conversation import Conversation, TaskHistory, with_ids
from examples.frontend_backend_live.delegation_core.im.input import (
    BackendTask,
    compose,
    records_parts,
    settings_for,
    with_parts,
)
from examples.frontend_backend_live.delegation_core.im.notes import ProgressNotes
from examples.frontend_backend_live.delegation_core.im.results import ResultFormat
from examples.frontend_backend_live.delegation_core.im.run import Parked, TaskRun, WorkRun, brief_message
from examples.frontend_backend_live.delegation_core.im.stop import AtClose, ForNewerRequest
from examples.frontend_backend_live.delegation_core.im.stream import ResponseBook, ResponseStream
from examples.frontend_backend_live.delegation_core.items import is_message, message_text
from examples.frontend_backend_live.delegation_core.ledger.records import WorkLedger
from examples.shared.ids import uid

FUNCTION_CALL_OUTPUTS_REQUIRED = "function_call_outputs_required"
NO_CALL = "Session waits for a later call: its call has ended"


class DelegationError(Exception):
    """Raised when the IM refuses a request, before anything changed; ``code`` is the official code, if any."""

    def __init__(self, message: str, code: str | None = None):
        """Refuse a request with ``message``, and with ``code`` when an official code names the refusal."""
        super().__init__(message)
        self.code = code


def call_arguments(call: dict) -> str | None:
    """Return a call's arguments as JSON with sorted keys, so calls compare as data; ``None`` when not JSON."""
    try:
        return json.dumps(json.loads(call.get("arguments") or "{}"), sort_keys=True)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Admission:
    """How a request was accepted: its delegation, and the stream of its one response."""

    delegation_id: str
    stream: ResponseStream = field(compare=False, repr=False)


@dataclass(frozen=True)
class Incoming:
    """What an accepted request brings to the Work it joins; its input is its ``Delegation`` record's."""

    # A typed run's or continuation's own settings; None for a delegation, which keeps the Work's.
    brings: dict | None


def _noop(kind: str, **fields: Any) -> None:
    """Ignore observations."""


def _logged(task: asyncio.Task) -> None:
    """Log the error of a close that runs in the background, so it is never left unretrieved."""
    if not task.cancelled() and task.exception() is not None:
        logger.warning(f"A close in the background failed: {type(task.exception()).__name__}: {task.exception()}")


class Decisions:
    """The IM session as the conductor's ``port.DecisionTarget``, apart since the session's ``start()`` differs."""

    def __init__(self, session: IMSession):
        """Apply decisions to ``session``."""
        self.session = session

    async def start(self, decision: Start) -> str:
        """Start a new Work for these requests, with its tasks."""
        return self.session.apply_start(decision)

    async def steer(self, decision: Steer) -> str:
        """Join these requests to an open Work (one task, or every open task)."""
        return self.session.apply_steer(decision)

    async def cancel(self, decision: Cancel) -> str:
        """Stop a Work, or one task of it, through the stop door; return once the stop is asked."""
        return self.session.apply_cancel(decision)

    async def answer(self, decision: Answer) -> str:
        """End a request with this answer from the records; no Work changes."""
        return self.session.apply_answer(decision)


class IMSession:
    """The interaction manager of one conversation: admission, placement, Works and their tasks, records, close."""

    def __init__(
        self,
        backend: BackendAdapter,
        defaults: BackendTask,
        settings: DelegationSettings,
        texts: Texts,
        *,
        result_format: ResultFormat | None = None,
        output_steps: Sequence[AppOutputStep] = (),
        observe: Callable[..., None] = _noop,
        renew: Callable[[], BackendAdapter] | None = None,
        room: Callable[[IMSession], bool] = lambda session: True,
        on_closed: Callable[[IMSession], None] = lambda session: None,
    ):
        """Wire the session; nothing runs until :meth:`start`.

        ``renew`` makes a new backend, for a later call once the one before closed; a one-shot session, which never
        waits for a later call, has none. ``room`` says whether fewer than ``max_after_call_conversations`` other
        conversations have work open that outlives their call, and ``on_closed`` runs once the session has closed.
        """
        self.backend, self.defaults, self.settings, self.texts = backend, defaults, settings, texts
        self.result_format = result_format if result_format is not None else ResultFormat()
        self.observe = observe
        self.caller_note, self.voice_note = texts.caller_note, texts.voice_note
        self.output_steps = tuple(output_steps)
        self.ledger = WorkLedger(uid("conv"))
        self.book = ResponseBook(self.ledger)
        self.notes = ProgressNotes(self.book, settings.progress_note_seconds)
        self.conversation = Conversation(settings.context_tokens, settings.tool_output_chars, texts.cut_output)
        self.incoming: dict[str, Incoming] = {}
        self.arrivals: list[str] = []
        self.works: dict[str, WorkRun] = {}
        self.ended_works: deque[WorkRun] = deque(maxlen=settings.records_max_works)
        self.ready: deque[TaskRun] = deque()
        self.running: set[TaskRun] = set()
        self.parked: dict[str, Parked] = {}
        self.admitted = 0
        # The newest call let run: its name, arguments, call_id, and the requests admitted when it ran.
        self.newest_call: tuple[str, str | None, str, int] | None = None
        self.conductor: Conductor | None = None
        self.decisions = Decisions(self)
        self.placer: asyncio.Task | None = None
        self.closed = False
        self._wake = asyncio.Event()
        self._deciding: asyncio.Task | None = None
        self._call: object | None = None
        self._given: set[str] = set()
        self._placed_any = False
        self._may_restart = False
        self._restart = False
        self._cursor = 0
        self._background: set[asyncio.Task] = set()
        self._closing: asyncio.Task | None = None
        self._decided = asyncio.Event()
        self._decided.set()
        self.renew, self.room, self.on_closed = renew, room, on_closed
        # Set by the service: the conversation opted in to keep work after its call (a Live session's store: true).
        self.continuable = False
        self._leaving: asyncio.Task | None = None
        # The keep time of a session that waits for a later call, and the close of its backend once no Work is open.
        self._keep: asyncio.TimerHandle | None = None
        self._backend_closing: asyncio.Task | None = None

    def start(self) -> None:
        """Ask the adapter for the session's conductor, once, and start placing."""
        self.conductor = self.backend.conductor()
        self.placer = asyncio.create_task(self._place(), name="im-placer")

    @property
    def work_tools(self) -> tuple[str, ...]:
        """The IM's work tools of every task: ``continue_after_call`` too when the work can outlive the call.

        It can in a conversation that opted in, on a backend that runs more than one task at once, so a task slot stays
        free for the call.
        """
        tools = ("cancel_work", "ask_caller")
        return (*tools, "continue_after_call") if self.continuable and self.backend.max_running > 1 else tools

    @property
    def detached(self) -> bool:
        """Whether the session waits for a later call: its call ended while its records list work outliving it."""
        return self.ledger.session.state == "detached"

    @property
    def runs_after_call(self) -> bool:
        """Whether one of its open Works outlives the call."""
        return any(work.record.after_call for work in self.works.values())

    @property
    def waits_for_decision(self) -> bool:
        """Whether a request waits for the conductor's decision; it may concern any Work, so the gate holds them all."""
        return not self._decided.is_set()

    async def decided(self) -> None:
        """Wait until no request waits for the conductor's decision (at once when none does, or after the close)."""
        await self._decided.wait()

    @property
    def history(self) -> list[dict]:
        """The session's conversation, oldest first: ended Works and closed exchanges, then the open Works' items."""
        _, items = self.conversation.window()
        for work in self.works.values():
            items += deepcopy(work.requested)
            items += [item for task in work.tasks if task.view is not None for item in task.view.own()]
        return items

    def spawn(self, work: Coroutine) -> None:
        """Run ``work`` in the background; the close waits for it."""
        task = asyncio.create_task(work)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def admit(
        self, input_items: list[dict], *, delegation_id: str | None = None, settings: dict | None = None
    ) -> Admission:
        """Accept a delegation in one step, or refuse it with ``DelegationError`` before anything changes."""
        return self._accept(input_items, delegation_id, settings, None)

    def _accept(
        self, input_items: list[dict], delegation_id: str | None, settings: dict | None, brings: dict | None
    ) -> Admission:
        if self.closed:
            raise DelegationError("Session is closing")
        if self.detached:
            raise DelegationError(NO_CALL)
        if delegation_id is not None and delegation_id in self.ledger.delegations:
            raise DelegationError(f"Delegation {delegation_id} already exists")
        waiting = sum(1 for record in self.ledger.delegations.values() if record.state == "queued")
        if waiting >= self.settings.max_pending_delegations:
            raise DelegationError("Delegation queue is full")
        settings, delegation_id = deepcopy(settings or {}), delegation_id or uid("item")
        # Its messages carry its id, which every task's view of the conversation knows them by.
        delegation = self.ledger.new_delegation(
            delegation_id, input_items=with_ids(input_items, delegation_id), settings=settings
        )
        stream = ResponseStream()
        self.book.open(delegation.id, stream, *self.response_settings(settings))
        self.incoming[delegation.id] = Incoming(brings)
        self.arrivals.append(delegation.id)
        self.admitted += 1
        self._decided.clear()
        self.observe("backend.queued", delegation_id=delegation.id, waiting=waiting + 1)
        if self._deciding is not None and not self._deciding.done() and not self._placed_any and self._may_restart:
            # A decide() that placed nothing starts again with every arrival.
            self._restart, self._may_restart = True, False
            self._deciding.cancel()
        self._wake.set()
        return Admission(delegation.id, stream)

    def request_items(self, delegation_id: str) -> list[dict]:
        """Return a request's input as the ledger recorded it; none for a request the ledger forgot."""
        record = self.ledger.delegations.get(delegation_id)
        return deepcopy(record.input) if record is not None else []

    async def _place(self) -> None:
        """Run one ``decide()`` at a time; no ending of one call stops the placer."""
        while True:
            while not self.arrivals:
                self._wake.clear()
                await self._wake.wait()
            try:
                await self._decide()
            except Exception:
                logger.exception("Placing the arrivals failed")

    async def _decide(self) -> None:
        """Call ``decide()``, restarted once by an arrival before it placed anything; then place what it left.

        The question of a task that had ended when the input was built is read by this decision, then cleared.
        """
        restarted = False
        while True:
            given_ids = list(self.arrivals)
            self._given, self._placed_any, self._restart = set(given_ids), False, False
            self._may_restart = not restarted
            given, cursor = self._conductor_input(given_ids)
            asked = [(task, task.question) for task in self.ledger.tasks.values() if task.ended and task.question]
            call = self._call = object()
            deciding = self._deciding = asyncio.create_task(self.conductor.decide(given, self._place_for(call)))
            try:
                await asyncio.wait({deciding})
            finally:
                if not deciding.done():
                    deciding.cancel()
                self._deciding, self._call = None, None
            if not (deciding.cancelled() and self._restart):
                break
            restarted = True
            self.observe("im.decide.restarted", arrivals=len(self.arrivals))
        self._given = set()
        error = asyncio.CancelledError() if deciding.cancelled() else deciding.exception()
        if error is None:
            self._cursor = cursor
        else:
            logger.warning(
                f"The conductor failed; the IM's fallback rule places what it left: {type(error).__name__}: {error}"
            )
        leftovers = [delegation for delegation in given_ids if delegation in self.arrivals]
        if leftovers:
            await self._place_by_rule(leftovers)
        for task, question in asked:
            # A question written while it decided is not read yet; the next decision reads it.
            if task.question == question:
                task.question = None

    async def _place_by_rule(self, delegations: list[str]) -> None:
        """Place what a ``decide()`` left by the IM's fallback rule: a failed conductor starts no work beside open work.

        A Work that outlives the call is the exception: the rule never steers into it, and its tasks leave a slot free.
        """
        newest = next((work for work in reversed(self.works.values()) if not work.record.after_call), None)
        decision = Start(tuple(delegations)) if newest is None else Steer(newest.id, tuple(delegations))
        self._given = set(delegations)
        try:
            outcome = await decision.apply(self.decisions)
        finally:
            self._given = set()
        self.observe("im.placed_by_rule", decision=type(decision).__name__, outcome=outcome)

    def _conductor_input(self, given_ids: list[str]) -> tuple[ConductorInput, int]:
        """Return what one ``decide()`` reads, and the conversation position it reads up to."""
        start, items = self.conversation.window()
        new_from = max(0, min(len(items), self._cursor - start))
        arrivals = tuple(Request(d, tuple(self.request_items(d))) for d in given_ids)
        works = tuple(work.summary() for work in [*self.works.values(), *self.ended_works])
        return ConductorInput(arrivals, works, tuple(items), new_from), start + len(items)

    def _place_for(self, call: object) -> Callable[[Decision], Coroutine[Any, Any, str]]:
        """Return the ``place`` one ``decide()`` call gets, which refuses what it places after that call ended.

        A runtime that kept a cancelled or finished turn may still call it, also while a later ``decide()`` runs.
        """

        async def place(decision: Decision) -> str:
            if call is not self._call:
                return self._refused("ended")
            return await self.place(decision)

        return place

    async def place(self, decision: Decision) -> str:
        """Apply one decision of the conductor and return its outcome line."""
        self._placed_any = True
        outcome = await decision.apply(self.decisions)
        self.observe("im.placed", decision=type(decision).__name__, outcome=outcome)
        return outcome

    def _take(self, delegations: Iterable[str], given: set[str] | None = None) -> list[str]:
        """Take the unplaced arrivals among ``delegations``, and among ``given`` when it is named, oldest first."""
        wanted = set(delegations) if given is None else set(delegations) & given
        taken = [delegation for delegation in self.arrivals if delegation in wanted]
        for delegation in taken:
            self.arrivals.remove(delegation)
        if not self.arrivals and not self._decided.is_set():
            self._decided.set()
            self.observe("im.decided", delegation_ids=taken)
        return taken

    def _refused(self, reason: str, **values: str) -> str:
        """Return the outcome of a refused decision, with the reason ``texts.refusal_reasons`` words for it."""
        return self.texts.refusal(reason, **values)

    def _not_placeable(self, delegations: Iterable[str]) -> str:
        """Return the outcome for requests none of which waits to be placed.

        A request that joined a Work is already placed: ``texts.outcomes.already_placed`` names that Work and the tasks
        it joined. The unplaced refusal names each other request, one that joined no Work, such as an unknown one.
        """
        placed, lines, others = self.texts.outcomes["already_placed"], [], []
        for delegation in dict.fromkeys(delegations):
            record = self.ledger.delegations.get(delegation)
            if record is None or not record.work:
                others.append(delegation)
            else:
                lines.append(placed.format(delegation=delegation, work=record.work, tasks=", ".join(record.tasks)))
        if others or not lines:
            lines.append(self._refused("unplaced", delegations=", ".join(others) or self.texts.none))
        return " ".join(lines)

    def _placeable(self, delegations: Iterable[str]) -> bool:
        return any(d in self.arrivals and d in self._given for d in delegations)

    def _over_limit(self, tasks: int) -> bool:
        """Return whether ``tasks`` more would put the running and queued tasks over a ``max_running`` above one."""
        limit = self.backend.max_running
        active = sum(task.record.state in ("queued", "running") for work in self.works.values() for task in work.tasks)
        return limit > 1 and active + tasks > limit

    def room_after_call(self, work: WorkRun) -> bool:
        """Return whether ``work`` may outlive the call: the call is on, and both limits leave room.

        The open tasks of the Works that outlive the call, ``work``'s included, stay below ``max_running``, so a slot
        stays free for the call; and ``room`` says that fewer than ``max_after_call_conversations`` other
        conversations have such work open.
        """
        if self.closed or self.detached:
            return False
        marked = sum(len(each.open_tasks()) for each in self.works.values() if each.record.after_call or each is work)
        return marked < self.backend.max_running and self.room(self)

    def apply_start(self, decision: Start) -> str:
        """Start a new Work for these requests, one task per brief, within the tasks the backend runs at once."""
        if not self._placeable(decision.delegations):
            return self._not_placeable(decision.delegations)
        if self._over_limit(len(decision.tasks) or 1):
            return self._refused("over_limit", max_running=str(self.backend.max_running))
        work = self._start_work(self._take(decision.delegations, self._given), decision.tasks)
        tasks = ", ".join(task.task_id for task in work.tasks)
        return self.texts.outcomes["started"].format(work=work.id, tasks=tasks)

    def apply_steer(self, decision: Steer) -> str:
        """Join these requests to an open Work, one task or all; an ended Work or task gives a new Work instead."""
        if not self._placeable(decision.delegations):
            return self._not_placeable(decision.delegations)
        work = self.works.get(decision.work_id)
        if work is None and decision.work_id not in self.ledger.works:
            return self._refused("no_work", work=decision.work_id)
        task = None
        if decision.task_id is not None:
            if decision.task_id not in {each.id for each in self.ledger.works[decision.work_id].tasks}:
                return self._refused("no_task", task=decision.task_id, work=decision.work_id)
            if work is not None:
                task = next(each for each in work.tasks if each.task_id == decision.task_id)
        if work is None or (task is not None and task.ended):
            if self._over_limit(1):
                return self._refused("over_limit", max_running=str(self.backend.max_running))
            ended = decision.task_id or decision.work_id
            new = self._start_work(self._take(decision.delegations, self._given), (), ended)
            return self.texts.outcomes["restarted"].format(work=ended, new=new.id)
        taken = self._take(decision.delegations, self._given)
        brings = next((self.incoming[d].brings for d in reversed(taken) if self.incoming[d].brings is not None), None)
        tasks = tuple(each.task_id for each in ([task] if task is not None else work.open_tasks()))
        for delegation in taken:
            joined = self.ledger.delegations[delegation]
            joined.work, joined.tasks = work.id, tasks
        work.steer(taken, task, brings)
        return self.texts.outcomes["steered"].format(work=work.id, tasks=", ".join(tasks))

    def apply_cancel(self, decision: Cancel) -> str:
        """Stop an open Work, or one task, through the stop door; return once the stop is asked."""
        work = self.works.get(decision.work_id)
        ended = self.texts.outcomes["ended"]
        if work is None:
            known = decision.work_id in self.ledger.works
            return ended.format(work=decision.work_id) if known else self._refused("no_work", work=decision.work_id)
        task = None
        if decision.task_id is not None:
            task = next((each for each in work.tasks if each.task_id == decision.task_id), None)
            if task is None:
                return self._refused("no_task", task=decision.task_id, work=decision.work_id)
            if task.ended:
                return ended.format(work=decision.task_id)
        stopping = [task] if task is not None else work.open_tasks()
        work.stop_soon(ForNewerRequest(), task)
        tasks = ", ".join(each.task_id for each in stopping)
        return self.texts.outcomes["cancel_requested"].format(work=work.id, tasks=tasks)

    def apply_answer(self, decision: Answer) -> str:
        """End this request with ``text`` from the records; with no Work open it is refused, and the request waits."""
        if not self.works:
            return self._refused("no_open_work")
        taken = self._take([decision.delegation_id], self._given)
        if not taken:
            return self._not_placeable([decision.delegation_id])
        delegation = taken[0]
        self.book.start(delegation)
        item = self.book.running(delegation).message(decision.text, "final_answer")
        self.book.end(delegation, "response.completed")
        self.ledger.delegations[delegation].move("answered")
        self.conversation.join([*self.request_items(delegation), item])
        return self.texts.outcomes["answered"].format(delegation=delegation)

    def _start_work(self, delegations: list[str], briefs: Iterable[str], continues: str | None = None) -> WorkRun:
        """Start a new queued Work for these requests, oldest first: the newest carries it, the others hand over.

        ``continues`` names the ended Work or task whose thread it continues; ``None`` begins a new thread.
        """
        newest = self.ledger.delegations[delegations[-1]]
        record = self.ledger.new_work(delegations[:1], deepcopy(newest.settings), tuple(briefs))
        record.continues = continues
        work = self.works[record.id] = WorkRun(self, record)
        tasks = tuple(task.id for task in record.tasks)
        for delegation in delegations:
            joined = self.ledger.delegations[delegation]
            joined.work, joined.tasks = record.id, tasks
        work.take_over_at_start(delegations[1:])
        work.begin()
        return work

    def work_input(self, work: WorkRun) -> list[dict]:
        """Return a starting Work's input as its tasks read it: its requests, the work records in the newest."""
        return with_parts(work.requested, *self.records_parts(exclude=work))

    def records_parts(
        self, exclude: WorkRun | None = None, joining: Mapping[str, Sequence[str]] | None = None
    ) -> tuple[str, ...]:
        """Return the work records block and the closing line, as a request's extra parts; none with no record."""
        return records_parts(self.texts.records_block, self.texts.closing_line, self.records(exclude, joining))

    def history_for(self, work: WorkRun, task: TaskRun) -> TaskHistory:
        """Return a starting task's view: the conversation's window, its Work's input, its brief, what joined before."""
        _, window = self.conversation.window()
        view = TaskHistory(
            [*window, *work.input], self.settings.context_tokens, self.settings.tool_output_chars, self.texts.cut_output
        )
        if task.brief:
            view.join([brief_message(self.texts.task_line, task.brief)])
        view.join(task.pending)
        return view

    def instruction_notes(self) -> tuple[str, ...]:
        """Return what the IM adds to every task's instructions, in order: the caller, voice and format notes."""
        return (self.caller_note, self.voice_note, self.result_format.note)

    def response_settings(self, settings: dict | None) -> tuple[str, str | None]:
        """Return what a request's response reports: its model, and its instructions as the IM composes them."""
        task = settings_for(settings or {}, self.defaults)
        return task.model, compose(task.instructions, *self.instruction_notes())

    def records(self, exclude: WorkRun | None = None, joining: Mapping[str, Sequence[str]] | None = None) -> list[dict]:
        """Return the work records: the newest ended Works, then every open one.

        ``joining`` names, by Work, the requests that join it with the steer being read.
        """
        works = [work for work in self.works.values() if work is not exclude]
        return [work.record_line((joining or {}).get(work.id, ())) for work in [*self.ended_works, *works]]

    def commentary(self, task: TaskRun, message: dict) -> None:
        """Write a backend message that is not the task's answer as a progress note."""
        text = message_text(message)
        if text.strip():
            task.record.progress = text
            if task.stop_reason is None:
                self.notes.note(task.work.carrier, text)

    def progress(self, task: TaskRun, text: str) -> None:
        """Write a task's activity that is not model text as a progress note."""
        if task.stop_reason is None:
            self.notes.note(task.work.carrier, text)

    def waits_for_app(self, task: TaskRun) -> bool:
        """Return whether a task whose round ended with the app's calls waits for their outputs: always, here."""
        return True

    def work_ended(self, work: WorkRun, leftovers: list[str]) -> None:
        """Close an ended Work: its records close, its segment joins the conversation, its leftover requests go on.

        A leftover request whose call has ended is cancelled instead. While the session waits for a later call, every
        leftover request is cancelled, and once no Work is open, its keep time starts again and its backend closes.
        """
        self.works.pop(work.id, None)
        for parked in {parked for parked in self.parked.values() if parked.work is work}:
            self.unpark(parked)
        tasks = {task.task_id: task for task in work.tasks}
        for action in self.ledger.actions.values():
            if action.work == work.id and not action.ended:
                action.move("unclear")
                tasks[action.task].steps[action.id][1] = "unclear"
        self.conversation.join(work.segment())
        self.ended_works.append(work)
        pending = [d for d in leftovers if not self.ledger.delegations[d].ended]
        waiting = self.detached and not self.closed
        # Nobody hears a follow-up Work while the session waits for a later call, or for a request whose call ended.
        gone = [d for d in pending if waiting or (not self.closed and self.book.writer(d) is None)]
        for delegation in gone:
            self.book.cancelled(delegation)
            self.ledger.delegations[delegation].move("cancelled")
        pending = [d for d in pending if d not in gone]
        if pending and not self.closed:
            # A follow-up serves the current call: it starts in a copy of the placer's context, as a placed Work does.
            context = self.placer.get_context() if self.placer is not None else contextvars.copy_context()
            context.copy().run(self._follow_up, work, pending)
        self.observe("backend.ended", work_id=work.id, state=work.record.state, delegation_id=work.carrier)
        self._prune()
        if waiting and not self.works:
            self._keep_from_now()
            self._release_backend()

    def _follow_up(self, work: WorkRun, pending: list[str]) -> None:
        """Start the Work that continues ``work`` for its leftover requests, oldest first."""
        order = list(self.incoming)
        self._start_work(sorted(pending, key=order.index), (), work.id)
        self.observe("im.follow_up", work_id=work.id, delegation_ids=pending)

    def _prune(self) -> None:
        """Forget the oldest closed records and the requests that no record names."""
        for forgotten in self.ledger.prune():
            for task in forgotten.tasks:
                if task.result.state == "pushed":
                    task.result.move("expired")
        for delegation in [d for d in self.incoming if d not in self.ledger.delegations]:
            del self.incoming[delegation]

    def request_slot(self, task: TaskRun) -> asyncio.Future:
        """Return a slot for ``task``, granted once fewer than ``max_running`` tasks run, in the order tasks asked."""
        grant = asyncio.get_running_loop().create_future()
        self.ready.append(task)
        task.grant = grant
        self._schedule()
        return grant

    def release(self, task: TaskRun) -> None:
        """Give ``task``'s slot to the next ready task."""
        self.running.discard(task)
        self._schedule()

    def unschedule(self, task: TaskRun) -> None:
        """Take ``task`` off the queue before it got a slot."""
        if task in self.ready:
            self.ready.remove(task)

    def _schedule(self) -> None:
        while self.ready and len(self.running) < self.backend.max_running:
            task = self.ready.popleft()
            if task.grant is not None and not task.grant.done():
                self.running.add(task)
                task.grant.set_result(None)

    def action(self, task: TaskRun, call: dict, state: str | None = None) -> None:
        """Record a call of ``task`` with who runs it, and move it to ``state`` when given."""
        call_id, name = call.get("call_id", ""), call.get("name", "")
        action = self.ledger.actions.get(call_id)
        if action is None:
            action = self.ledger.new_action(call_id, task.task_id, name, task.runner(name))
        if state is not None and not action.ended:
            action.move(state)

    def call_ended(self, call_id: str, output: str | None, outcome: Outcome) -> None:
        """Record that a call ended with ``outcome``, once."""
        action = self.ledger.actions.get(call_id)
        if action is not None and not action.ended:
            action.output = output
            action.move(outcome)

    def call_routed(self, call: dict) -> None:
        """Note ``call``, which runs (the app's or the backend's), as the newest call that ran."""
        self.newest_call = (call.get("name", ""), call_arguments(call), call.get("call_id", ""), self.admitted)

    def repeated_output(self, call: dict, task_id: str) -> str | None:
        """Return the app's answer, worded by ``texts.repeated_call_output``, when ``call`` repeats the newest call."""
        if self.newest_call is None:
            return None
        name, arguments, call_id, admitted = self.newest_call
        if admitted != self.admitted:
            return None
        if arguments is None or (call.get("name", ""), call_arguments(call)) != (name, arguments):
            return None
        action = self.ledger.actions.get(call_id)
        answered = action is not None and action.task == task_id and action.runner == "app" and action.state == "done"
        if not answered or not isinstance(action.output, str):
            return None
        return self.texts.repeated_call_output.format(output=action.output)

    def park(self, parked: Parked) -> None:
        """Owe the outputs of a response's app calls, under the client-tool deadline from its end."""
        for call_id in parked.calls:
            self.parked[call_id] = parked
        self.observe("tools.wait.started", delegation_id=parked.delegation_id, call_ids=list(parked.calls))

    def unpark(self, parked: Parked) -> None:
        """Stop waiting for these calls' outputs."""
        for call_id in parked.calls:
            self.parked.pop(call_id, None)
        if parked.timer is not None:
            parked.timer.cancel()

    def admit_client_request(
        self, input_items: list[dict], *, delegation_id: str | None = None, settings: dict | None = None
    ) -> Admission:
        """Accept a request the client made itself, or refuse it before anything changes.

        Function outputs continue the response that ended with their calls, and must answer every call of it; outputs
        whose Work has ended are recorded only. Anything else is a typed run, which steers running work as a spoken
        request does, where the Live API queues it.
        """
        if self.closed:
            raise DelegationError("Session is closing")
        if self.detached:
            raise DelegationError(NO_CALL)
        items = deepcopy(input_items)
        results = [item for item in items if str(item.get("type", "")).endswith("_output")]
        messages = [item for item in items if is_message(item)]
        if not results:
            return self._accept(items, delegation_id, settings, deepcopy(settings or {}))
        outputs: dict[str, dict] = {}
        for item in results:
            call_id = item.get("call_id")
            if item.get("type") != "function_call_output" or not isinstance(call_id, str) or not call_id:
                raise ValueError("A result must be a function_call_output with the call_id of a function call")
            if "output" not in item:
                raise ValueError(f"The result for call {call_id} has no output")
            if call_id in outputs:
                raise ValueError(f"Duplicate result for call {call_id}")
            outputs[call_id] = item
        parked = {self.parked[call_id] for call_id in outputs if call_id in self.parked}
        if not parked:
            return self._late(delegation_id, outputs, messages, settings)
        if len(parked) > 1 or any(call_id not in self.parked for call_id in outputs):
            raise ValueError("The results answer calls of more than one response")
        (waiting,) = parked
        if any(call_id not in outputs for call_id in waiting.calls):
            raise DelegationError(
                "Return the output of every function call before response.create", FUNCTION_CALL_OUTPUTS_REQUIRED
            )
        self.unpark(waiting)
        stream = ResponseStream()
        brings = deepcopy(settings or {})
        self.book.open(waiting.delegation_id, stream, *self.response_settings(brings))
        for call_id, output in outputs.items():
            self.call_ended(call_id, output.get("output"), "done")
        # The ledger keeps the app's outputs; the tasks read what the steps leave.
        for step in self.output_steps:
            calls = [(call, outputs[call_id]) for call_id, (_task, call) in waiting.calls.items()]
            outputs = step.outputs(calls, settings_for(brings, self.defaults).tools)
        # Typed messages with the outputs count as a new request.
        if messages:
            self.admitted += 1
        waiting.work.continue_with(waiting, outputs, messages, brings)
        self.observe("tools.wait.completed", delegation_id=waiting.delegation_id, call_ids=list(waiting.calls))
        return Admission(waiting.delegation_id, stream)

    def _late(
        self, delegation_id: str | None, outputs: dict[str, dict], messages: list[dict], settings: dict | None
    ) -> Admission:
        """Record outputs whose Work has ended; answer with an empty ``completed``, or run the typed messages."""
        for call_id, output in outputs.items():
            action = self.ledger.actions.get(call_id)
            if action is not None and action.output is None:
                action.output = output.get("output")
        if messages:
            logger.info(f"Typed messages came with late function outputs for {delegation_id}: they run as a typed run")
            return self._accept(messages, None, settings, deepcopy(settings or {}))
        delegation = delegation_id or uid("item")
        record = self.ledger.delegations.get(delegation)
        stream = ResponseStream()
        self.book.late(delegation, stream, *self.response_settings(record.settings if record is not None else settings))
        return Admission(delegation, stream)

    async def leave(self) -> bool:
        """End the call; return whether the session is kept for a later call, else it closed as :meth:`close` does.

        It is kept while its records list a Work that outlives the call, open or among the newest ended. Then the
        session waits for a later call: the placer stops, the other Works stop as at a close (a Work that outlives the
        call stops too while it waits for the app), every open response ends ``cancelled``, and every open request that
        no such Work carries is cancelled and never placed. The keep time starts, the backend closes when no Work is
        open, and else releases its idle sessions. It runs as a task of its own, which a cancelled caller leaves
        running; safe to call twice.
        """
        if self._leaving is None:
            self._leaving = asyncio.create_task(self._leave(), name="im-leave")
        return await asyncio.shield(self._leaving)

    async def _leave(self) -> bool:
        if self.closed or not any(work.record.after_call for work in [*self.works.values(), *self.ended_works]):
            await self.close()
            return False
        self.ledger.session.move("detached")
        self._decided.set()
        try:
            await self._end_call(lambda work: work.record.after_call)
            # Only unplaced requests wait here, and no Work carries them.
            self.arrivals.clear()
        finally:
            self.notes.close()
            self._keep_from_now()
            if self.works:
                self.backend.release_idle()
            else:
                # The backend closes in the background: attach() and close() wait for it.
                self._release_backend()
        return True

    async def attach(self) -> None:
        """Serve a later call of a session that waits for one; ``DelegationError`` when it has ended or a call holds it.

        It waits for a :meth:`leave` and a backend's close still running. When the backend closed, it builds a new one
        first, whose conductor reads the whole conversation, so a failed build leaves the session waiting; a kept
        conductor keeps what it read. Then it stops the keep time, moves the session to active and places this call's
        requests.
        """
        while True:
            running = [task for task in (self._leaving, self._backend_closing) if task is not None and not task.done()]
            if not running:
                break
            await asyncio.wait(running)
        if self.closed:
            raise DelegationError("Conversation has ended")
        if not self.detached:
            raise DelegationError("Conversation serves another call")
        if self._backend_closing is not None:
            backend = self.renew()
            self.backend, self.conductor, self._cursor = backend, backend.conductor(), 0
            self._backend_closing = None
        if self._keep is not None:
            self._keep.cancel()
            self._keep = None
        self.ledger.session.move("active")
        self._leaving = None
        self.placer = asyncio.create_task(self._place(), name="im-placer")
        works = [*self.ended_works, *self.works.values()]
        self.observe("im.attached", after_call={work.id: work.requests[0] for work in works if work.record.after_call})

    def _keep_from_now(self) -> None:
        """Start the keep time again: ``after_call_seconds`` from now with no call attached, the session closes."""
        if self._keep is not None:
            self._keep.cancel()
        self._keep = asyncio.get_running_loop().call_later(self.settings.after_call_seconds, self._kept_too_long)

    def _kept_too_long(self) -> None:
        """Close the session whose keep time ended, as :meth:`close` does: work still running stops."""
        self._keep = None
        if self._closing is None:
            # Closing from now on, so no call attaches before the close runs.
            self.closed = True
            self._closing = asyncio.create_task(self._close(), name="im-close")
            self._closing.add_done_callback(_logged)

    def _release_backend(self) -> asyncio.Task:
        """Start closing the backend of a session that waits for a later call with no Work open; return the close.

        :meth:`attach` builds a new one.
        """
        if self._backend_closing is None:
            self._backend_closing = asyncio.create_task(self.backend.close(), name="im-backend-close")
            self._backend_closing.add_done_callback(_logged)
        return self._backend_closing

    async def close(self) -> None:
        """End the session; safe to call twice.

        Queued and app-waiting Works stop at once and running ones get ``close_grace_seconds``; then every open response
        ends ``response.incomplete`` (``cancelled``) in acceptance order, and the backend adapter closes last.
        """
        if self._closing is None:
            self._closing = asyncio.create_task(self._close(), name="im-close")
        await asyncio.shield(self._closing)

    async def _close(self) -> None:
        # A leave() that keeps the session ends first; one that closes it waits for this close.
        if self.detached and self._leaving is not None and not self._leaving.done():
            await asyncio.wait({self._leaving})
        self.closed = True
        self._decided.set()
        if self._keep is not None:
            self._keep.cancel()
            self._keep = None
        try:
            await self._end_call(lambda work: False)
            if self._background:
                await asyncio.gather(*self._background, return_exceptions=True)
        finally:
            self.notes.close()
            if not self.ledger.session.ended:
                self.ledger.session.move("ended")
            try:
                if self._backend_closing is None:
                    await self.backend.close()
                else:
                    await asyncio.wait({self._backend_closing})
            finally:
                self.on_closed(self)

    async def _end_call(self, keep: Callable[[WorkRun], bool]) -> None:
        """Stop the call's work as :meth:`close` does; a running Work that ``keep`` names runs on, with its requests."""
        if self.placer is not None:
            self.placer.cancel()
            await asyncio.gather(self.placer, return_exceptions=True)
        # Responses whose end waits are cancelled first, before a stop ends them otherwise, and after the grace.
        self.book.cancel_closing()
        at_once = [work for work in self.works.values() if work.record.state in ("queued", "waiting_client")]
        await asyncio.gather(*(work.stop(AtClose()) for work in at_once))
        running = [work for work in self.works.values() if not keep(work)]
        if running:
            try:
                async with asyncio.timeout(self.settings.close_grace_seconds):
                    await asyncio.gather(*(work.done.wait() for work in running))
            except TimeoutError:
                pass
        self.book.cancel_closing()
        asked = [(work, work.ask_stop(AtClose())) for work in list(self.works.values()) if not keep(work)]
        for delegation in self.book.open_delegations():
            self.book.cancelled(delegation)
        kept = {work.id for work in self.works.values() if keep(work)}
        for record in self.ledger.delegations.values():
            if not record.ended and record.work not in kept:
                record.move("cancelled")
        await asyncio.gather(*(work.confirm_stop(tasks) for work, tasks in asked))
