# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One IM session: admission, placement through the conductor, the scheduler, close.

Admission is one step: a request is refused before anything changes, or its delegation is recorded and its response
opens ``queued``. The conductor places arrivals one ``decide()`` call at a time, and the session re-checks and answers
every decision; what a ``decide()`` leaves unplaced joins the newest open Work, or starts one when none is open, so a
conductor's failure never starts work beside open work. While an arrival waits for its decision, the commit gate holds
every open Work's new calls, since any of them may be the one the caller changed, and with
``delegation.hold_answers_while_deciding`` their answers too. A typed run steers running work as a spoken request does,
a deliberate deviation from GPT-Live, which queues it.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Callable, Coroutine, Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from delegation_core.backend.port import (
    Answer,
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
from delegation_core.config import DelegationSettings, Texts
from delegation_core.ids import uid
from delegation_core.im.conversation import Conversation, TaskHistory
from delegation_core.im.input import BackendTask, compose, records_parts, settings_for, with_parts
from delegation_core.im.notes import ProgressNotes
from delegation_core.im.results import CallReports, ResultFormat
from delegation_core.im.run import Parked, TaskRun, WorkRun, brief_message
from delegation_core.im.stop import AtClose, ForNewerRequest
from delegation_core.im.stream import ResponseBook, ResponseStream, message_text
from delegation_core.ledger.records import WorkLedger

# The official code of a continuation that leaves function calls without their outputs.
FUNCTION_CALL_OUTPUTS_REQUIRED = "function_call_outputs_required"


class DelegationError(Exception):
    """Raised when the IM refuses a request, before anything changed; ``code`` is the official code, if any."""

    def __init__(self, message: str, code: str | None = None):
        """Refuse a request with ``message``, and with ``code`` when an official code names the refusal."""
        super().__init__(message)
        self.code = code


def is_message(item: dict) -> bool:
    """Whether an input item is a message (a typed one may leave its ``type`` out)."""
    return item.get("type", "message") == "message"


def call_arguments(call: dict) -> str | None:
    """A call's arguments as parsed JSON, written one way (keys sorted) so calls compare as data; ``None``: not JSON."""
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
    """An accepted request's settings for a Work it joins (an explicit request's); its input is its ``Delegation``'s."""

    brings: dict | None  # a typed run's or continuation's own settings; None: a delegation, which keeps the Work's


def _noop(kind: str, **fields: Any) -> None:
    """Ignore observations."""


class Decisions:
    """The IM session as the conductor's ``port.DecisionTarget``: each decision applied, re-checked and answered.

    A class of its own because the session's own ``start()`` starts the session.
    """

    def __init__(self, session: IMSession):
        """Apply decisions to ``session``."""
        self.session = session

    async def start(self, decision: Start) -> str:
        """A new Work for these requests, with its tasks."""
        return self.session.apply_start(decision)

    async def steer(self, decision: Steer) -> str:
        """These requests join an open Work (one task, or every open task)."""
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
        call_reports: CallReports | None = None,
        observe: Callable[..., None] = _noop,
    ):
        """Wire the session; nothing runs until :meth:`start`.

        Args:
            backend: The session's backend adapter (``AdapterSource.new()``).
            defaults: What a request leaves out: the deployment's model and instructions.
            settings: The IM's settings (``CoreSettings.delegation``).
            texts: The core's texts (``CoreSettings.texts``): the lines the IM and the backend write or add to a
                model's input.
            result_format: The class ``delegation.result_format`` names, built with its note; the core's builder
                picks it from ``im/results.RESULT_FORMATS`` (default ``plain``).
            call_reports: The class ``delegation.backend_calls`` names, picked from ``im/results.CALL_REPORTS``
                (default ``items``).
            observe: Optional ``observe(kind, **fields)`` hook for traces.
        """
        self.backend, self.defaults, self.settings, self.texts = backend, defaults, settings, texts
        self.result_format = result_format if result_format is not None else ResultFormat()
        self.call_reports = call_reports if call_reports is not None else CallReports()
        self.observe = observe
        self.voice_note = texts.voice_notes[settings.voice_note]
        self.ledger = WorkLedger(uid("conv"))
        self.book = ResponseBook(self.ledger)
        self.notes = ProgressNotes(self.book, settings.progress_note_seconds)
        self.conversation = Conversation(settings.context_tokens, settings.tool_output_chars, texts.cut_output)
        self.incoming: dict[str, Incoming] = {}  # every accepted request whose record is kept, by delegation
        self.arrivals: list[str] = []  # the requests no decision has placed yet, oldest first
        self.works: dict[str, WorkRun] = {}  # the open Works
        self.ended_works: deque[WorkRun] = deque(maxlen=settings.records_max_works)
        self.ready: deque[TaskRun] = deque()  # tasks waiting for a slot, in the order they became ready
        self.running: set[TaskRun] = set()  # tasks holding a slot
        self.parked: dict[str, Parked] = {}  # each call the app owes an output for, by call_id
        self.admitted = 0  # the requests admitted so far, steering and typed messages with the app's outputs included
        # The newest call let run: its name, arguments, call_id, and the requests admitted when it ran.
        self.newest_call: tuple[str, str | None, str, int] | None = None
        self.conductor: Conductor | None = None
        self.decisions = Decisions(self)
        self.placer: asyncio.Task | None = None
        self.closed = False
        self._wake = asyncio.Event()
        self._deciding: asyncio.Task | None = None  # the decide() call in flight
        self._call: object | None = None  # its token: only its own place() applies decisions
        self._given: set[str] = set()  # the arrivals given to it
        self._placed_any = False  # it placed something
        self._may_restart = False  # an arrival may still restart it (once)
        self._restart = False  # an arrival restarted it
        self._cursor = 0  # how far into the conversation the conductor has read
        self._background: set[asyncio.Task] = set()
        self._closing: asyncio.Task | None = None
        self._decided = asyncio.Event()  # set while no arrival waits for a decision (and once the session closes)
        self._decided.set()

    # ------------------------------------------------------------------------------------------ the session
    def start(self) -> None:
        """Ask the adapter for the session's conductor, once, and start placing."""
        self.conductor = self.backend.conductor()
        self.placer = asyncio.create_task(self._place(), name="im-placer")

    @property
    def busy(self) -> bool:
        """Whether a request waits to be placed or a Work is open."""
        return bool(self.arrivals or self.works)

    @property
    def waits_for_decision(self) -> bool:
        """Whether a request waits for the conductor's decision: the commit gate holds every open Work's calls.

        With ``delegation.hold_answers_while_deciding``, the answers the open Works' tasks give wait for it too.
        """
        return not self._decided.is_set()

    async def decided(self) -> None:
        """Return once no request waits for the conductor's decision (at once when none does, or after the close)."""
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

    # ------------------------------------------------------------------------------------------ admission
    def admit(
        self, input_items: list[dict], *, delegation_id: str | None = None, settings: dict | None = None
    ) -> Admission:
        """Accept a delegation in one step, or refuse it before anything changes.

        Args:
            input_items: The request, as input items.
            delegation_id: The delegation's id (``metadata.live_delegation_id``); the IM makes one when none is given.
            settings: What the request set (``im/input.py``, ``request_settings``); the IM's defaults fill the rest.

        Raises:
            DelegationError: The session is closing, the delegation id is in use, or the room is full.
        """
        return self._accept(input_items, delegation_id, settings, None)

    def _accept(
        self, input_items: list[dict], delegation_id: str | None, settings: dict | None, brings: dict | None
    ) -> Admission:
        if self.closed:
            raise DelegationError("Session is closing")
        if delegation_id is not None and delegation_id in self.ledger.delegations:
            raise DelegationError(f"Delegation {delegation_id} already exists")
        waiting = sum(1 for record in self.ledger.delegations.values() if record.state == "queued")
        if waiting >= self.settings.max_pending_delegations:
            raise DelegationError("Delegation queue is full")
        settings = deepcopy(settings or {})
        delegation = self.ledger.new_delegation(
            delegation_id or uid("item"), input_items=deepcopy(input_items), settings=settings
        )
        stream = ResponseStream()
        self.book.open(delegation.id, stream, *self.response_settings(settings))
        self.incoming[delegation.id] = Incoming(brings)
        self.arrivals.append(delegation.id)
        self.admitted += 1
        self._decided.clear()
        self.observe("backend.queued", delegation_id=delegation.id, waiting=waiting + 1)
        if self._deciding is not None and not self._deciding.done() and not self._placed_any and self._may_restart:
            self._restart, self._may_restart = True, False
            self._deciding.cancel()  # the busy rule: a decide() that placed nothing starts again with all arrivals
        self._wake.set()
        return Admission(delegation.id, stream)

    def request_items(self, delegation_id: str) -> list[dict]:
        """A request's input, as it came (its ``Delegation`` record's); none for a request the ledger forgot."""
        record = self.ledger.delegations.get(delegation_id)
        return deepcopy(record.input) if record is not None else []

    # ------------------------------------------------------------------------------------------ placement
    async def _place(self) -> None:
        """The placer: one ``decide()`` at a time; no ending of one call stops it."""
        while True:
            while not self.arrivals:
                self._wake.clear()
                await self._wake.wait()
            try:
                await self._decide()
            except Exception:  # an unexpected error must not stop placement: log it, and go on with what is left
                logger.exception("Placing the arrivals failed")

    async def _decide(self) -> None:
        """One ``decide()`` call, restarted once by an arrival before it placed anything; then the unplaced arrivals."""
        restarted = False
        while True:
            given_ids = list(self.arrivals)
            self._given, self._placed_any, self._restart = set(given_ids), False, False
            self._may_restart = not restarted
            given, cursor = self._conductor_input(given_ids)
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
            logger.warning(f"The conductor failed; the steer rule places what it left: {type(error).__name__}: {error}")
        leftovers = [delegation for delegation in given_ids if delegation in self.arrivals]
        if leftovers:
            await self._place_by_rule(leftovers)

    async def _place_by_rule(self, delegations: list[str]) -> None:
        """What a ``decide()`` left unplaced joins the newest open Work, or starts one when none is open.

        The steer rule (``backend/conductors.py``, ``SteerOpenWork``), applied by the IM through the same re-checks as
        any decision, for a conductor that failed, timed out, was cancelled or skipped a request.
        """
        newest = next(reversed(self.works.values()), None)
        decision = Start(tuple(delegations)) if newest is None else Steer(newest.id, tuple(delegations))
        self._given = set(delegations)
        try:
            outcome = await decision.apply(self.decisions)
        finally:
            self._given = set()
        self.observe("im.placed_by_rule", decision=type(decision).__name__, outcome=outcome)

    def _conductor_input(self, given_ids: list[str]) -> tuple[ConductorInput, int]:
        """What one ``decide()`` reads, and the conversation position it reads up to."""
        start, items = self.conversation.window()
        new_from = max(0, min(len(items), self._cursor - start))
        arrivals = tuple(Request(d, tuple(self.request_items(d))) for d in given_ids)
        works = tuple(work.summary() for work in [*self.works.values(), *self.ended_works])
        return ConductorInput(arrivals, works, tuple(items), new_from), start + len(items)

    def _place_for(self, call: object) -> Callable[[Decision], Coroutine[Any, Any, str]]:
        """The ``place`` one ``decide()`` call gets: after that call ended, what it places is refused (the port).

        A runtime that kept a cancelled or finished turn may still call it, also while a later ``decide()`` runs.
        """

        async def place(decision: Decision) -> str:
            if call is not self._call:
                return self._refused("ended")
            return await self.place(decision)

        return place

    async def place(self, decision: Decision) -> str:
        """Apply one decision of the conductor (double dispatch) and return its outcome line."""
        self._placed_any = True
        outcome = await decision.apply(self.decisions)
        self.observe("im.placed", decision=type(decision).__name__, outcome=outcome)
        return outcome

    def _take(self, delegations: Iterable[str], given: set[str] | None = None) -> list[str]:
        """The arrivals among ``delegations`` that no decision placed yet, oldest first; they are placed now.

        A ``decide()`` may place only the arrivals it was ``given``.
        """
        wanted = set(delegations) if given is None else set(delegations) & given
        taken = [delegation for delegation in self.arrivals if delegation in wanted]
        for delegation in taken:
            self.arrivals.remove(delegation)
        if not self.arrivals and not self._decided.is_set():
            self._decided.set()  # the commit gate lets the open Works' calls (and held answers) go again
            self.observe("im.decided", delegation_ids=taken)
        return taken

    def _refused(self, reason: str, **values: str) -> str:
        """The outcome of a refused decision, with the reason ``texts.refusal_reasons`` words for it."""
        return self.texts.refusal(reason, **values)

    def _not_placeable(self, delegations: Iterable[str]) -> str:
        return self._refused("unplaced", delegations=", ".join(delegations) or self.texts.none)

    def _placeable(self, delegations: Iterable[str]) -> bool:
        return any(d in self.arrivals and d in self._given for d in delegations)

    def _over_limit(self, tasks: int) -> bool:
        """Whether ``tasks`` new tasks and the running and queued tasks of open Works are more than ``max_running``.

        Checked only when more than one task may run at once: with one, a rule places (the steer rule, or the backend
        LLM's own, whose ``start_each`` queues Works on purpose).
        """
        limit = self.backend.max_running
        active = sum(task.record.state in ("queued", "running") for work in self.works.values() for task in work.tasks)
        return limit > 1 and active + tasks > limit

    # ------------------------------------------------------------------------------------------ the decisions
    def apply_start(self, decision: Start) -> str:
        """A new Work for these requests, one task per brief, within the tasks the backend runs at once."""
        if not self._placeable(decision.delegations):
            return self._not_placeable(decision.delegations)
        if self._over_limit(len(decision.tasks) or 1):
            return self._refused("over_limit", max_running=str(self.backend.max_running))
        work = self._start_work(self._take(decision.delegations, self._given), decision.tasks)
        tasks = ", ".join(task.task_id for task in work.tasks)
        return self.texts.outcomes["started"].format(work=work.id, tasks=tasks)

    def apply_steer(self, decision: Steer) -> str:
        """These requests join an open Work, one task or all; an ended Work or task gives a new Work instead.

        That new Work is held to the tasks the backend runs at once, as a Start is.
        """
        if not self._placeable(decision.delegations):
            return self._not_placeable(decision.delegations)
        work = self.works.get(decision.work_id)
        if work is None and decision.work_id not in self.ledger.works:
            return self._refused("no_work", work=decision.work_id)
        task = None
        if work is not None and decision.task_id is not None:
            task = next((each for each in work.tasks if each.task_id == decision.task_id), None)
            if task is None:
                return self._refused("no_task", task=decision.task_id, work=decision.work_id)
        if work is None or (task is not None and task.ended):
            if self._over_limit(1):
                return self._refused("over_limit", max_running=str(self.backend.max_running))
            new = self._start_work(self._take(decision.delegations, self._given), ())
            ended = decision.task_id or decision.work_id
            return self.texts.outcomes["restarted"].format(work=ended, new=new.id)
        taken = self._take(decision.delegations, self._given)
        brings = next((self.incoming[d].brings for d in reversed(taken) if self.incoming[d].brings is not None), None)
        for delegation in taken:
            self.ledger.delegations[delegation].work = work.id
        work.steer(taken, task, brings)
        tasks = ", ".join(each.task_id for each in ([task] if task is not None else work.open_tasks()))
        return self.texts.outcomes["steered"].format(work=work.id, tasks=tasks)

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
        """End this request with ``text``, from the records; no Work changes.

        An answer is about open work only: with no Work open, it is refused, and the request waits for its place.
        """
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

    # ------------------------------------------------------------------------------------------ Works
    def _start_work(self, delegations: list[str], briefs: Iterable[str]) -> WorkRun:
        """A new queued Work for these requests (oldest first): the newest carries it, the others hand over in turn."""
        newest = self.ledger.delegations[delegations[-1]]
        record = self.ledger.new_work(delegations[:1], deepcopy(newest.settings), tuple(briefs))
        work = self.works[record.id] = WorkRun(self, record)
        for delegation in delegations:
            self.ledger.delegations[delegation].work = record.id
        work.take_over_at_start(delegations[1:])
        work.begin()
        return work

    def work_input(self, work: WorkRun) -> list[dict]:
        """A starting Work's input as its tasks read it: its requests, the work records in the newest."""
        return with_parts(work.requested, *self.records_parts(exclude=work))

    def records_parts(self, exclude: WorkRun | None = None) -> tuple[str, ...]:
        """The work records block and the closing line, as a request's extra parts; none with no record."""
        return records_parts(self.texts.records_block, self.texts.closing_line, self.records(exclude))

    def history_for(self, work: WorkRun, task: TaskRun) -> TaskHistory:
        """A starting task's view: the conversation's window, its Work's input, its brief, what joined before it."""
        _, window = self.conversation.window()
        view = TaskHistory(
            [*window, *work.input], self.settings.context_tokens, self.settings.tool_output_chars, self.texts.cut_output
        )
        if task.brief:
            view.join([brief_message(self.texts.task_line, task.brief)])
        view.join(task.pending)
        return view

    def instruction_notes(self) -> tuple[str, ...]:
        """What the IM adds to every task's instructions, in order: the voice note, then the result format's note."""
        return (self.voice_note, self.result_format.note)

    def response_settings(self, settings: dict | None) -> tuple[str, str | None]:
        """What a request's response reports: its model, and its instructions as the IM composes them.

        The request's instructions or the default, then the notes (:meth:`instruction_notes`), as every task of its
        Work runs them; a client reads them on the response object, and Live's ``/history`` shows them.
        """
        task = settings_for(settings or {}, self.defaults)
        return task.model, compose(task.instructions, *self.instruction_notes())

    def records(self, exclude: WorkRun | None = None) -> list[dict]:
        """The work records: every open Work, then the newest ended ones."""
        works = [work for work in self.works.values() if work is not exclude]
        return [work.record_line() for work in [*self.ended_works, *works]]

    def commentary(self, task: TaskRun, message: dict) -> None:
        """A message of the backend that is not the task's answer: a progress note."""
        text = message_text(message)
        if text.strip():
            task.record.progress = text
            if task.stop_reason is None:
                self.notes.note(task.work.carrier, text)

    def progress(self, task: TaskRun, text: str) -> None:
        """A task's activity that is not model text: a progress note."""
        if task.stop_reason is None:
            self.notes.note(task.work.carrier, text)

    def waits_for_app(self, task: TaskRun) -> bool:
        """Whether a task whose round ended with the app's calls waits for their outputs (here: always)."""
        return True

    def work_ended(self, work: WorkRun, leftovers: list[str]) -> None:
        """A Work ended: its records close, its segment joins the conversation, its leftover requests go on."""
        self.works.pop(work.id, None)
        for parked in {parked for parked in self.parked.values() if parked.work is work}:
            self.unpark(parked)
        for action in self.ledger.actions.values():
            if action.work == work.id and not action.ended:
                action.move("unclear")  # it never reported an end: whether it ran is not known
        self.conversation.join(work.segment())
        self.ended_works.append(work)
        pending = [d for d in leftovers if not self.ledger.delegations[d].ended]
        if pending and not self.closed:
            order = list(self.incoming)
            self._start_work(sorted(pending, key=order.index), ())  # unapplied steering becomes a follow-up Work
            self.observe("im.follow_up", work_id=work.id, delegation_ids=pending)
        self.observe("backend.ended", work_id=work.id, state=work.record.state, delegation_id=work.carrier)
        self._prune()

    def _prune(self) -> None:
        """Forget the oldest closed records (``WorkLedger.prune``) and the requests no record names any more."""
        for forgotten in self.ledger.prune():
            for task in forgotten.tasks:
                if task.result.state == "pushed":
                    task.result.move("expired")
        for delegation in [d for d in self.incoming if d not in self.ledger.delegations]:
            del self.incoming[delegation]

    # ------------------------------------------------------------------------------------------ the scheduler
    def request_slot(self, task: TaskRun) -> asyncio.Future:
        """A slot for ``task`` when fewer than ``max_running`` tasks run, in the order tasks asked."""
        grant = asyncio.get_running_loop().create_future()
        self.ready.append(task)
        task.grant = grant
        self._schedule()
        return grant

    def release(self, task: TaskRun) -> None:
        """``task`` stopped running: its slot goes to the next ready task."""
        self.running.discard(task)
        self._schedule()

    def unschedule(self, task: TaskRun) -> None:
        """``task`` stops before it got a slot."""
        if task in self.ready:
            self.ready.remove(task)

    def _schedule(self) -> None:
        while self.ready and len(self.running) < self.backend.max_running:
            task = self.ready.popleft()
            if task.grant is not None and not task.grant.done():
                self.running.add(task)
                task.grant.set_result(None)

    # ------------------------------------------------------------------------------------------ records
    def action(self, task: TaskRun, call: dict, state: str | None = None) -> None:
        """Record a call of ``task`` (started), with who runs it, and move it to ``state`` when given."""
        call_id, name = call.get("call_id", ""), call.get("name", "")
        action = self.ledger.actions.get(call_id)
        if action is None:
            action = self.ledger.new_action(call_id, task.task_id, name, task.runner(name))
        if state is not None and not action.ended:
            action.move(state)

    def call_ended(self, call_id: str, output: str | None, outcome: Outcome) -> None:
        """A call ended with ``outcome``; recorded once."""
        action = self.ledger.actions.get(call_id)
        if action is not None and not action.ended:
            action.output = output
            action.move(outcome)

    def call_routed(self, call: dict) -> None:
        """``call`` runs (the app's or the backend's): it is the newest call that ran."""
        self.newest_call = (call.get("name", ""), call_arguments(call), call.get("call_id", ""), self.admitted)

    def repeated_output(self, call: dict, task_id: str) -> str | None:
        """The output of ``task_id``'s app call with the name and arguments of the newest call that ran, which it made.

        The app's answer to that call, as ``texts.repeated_call_output`` words it, which the IM gives instead of
        sending the call again; ``None`` for any other call, when the newest call got no text output from the app (it
        never answered), or when a request was admitted after it ran (a new request, steering too, may ask for the
        same call again).
        """
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

    # ------------------------------------------------------------------------------------------ the app's calls
    def park(self, parked: Parked) -> None:
        """A response ended with the app's calls: their outputs are owed, under the client-tool deadline."""
        for call_id in parked.calls:
            self.parked[call_id] = parked
        parked.work.park_timer(parked)
        self.observe("tools.wait.started", delegation_id=parked.delegation_id, call_ids=list(parked.calls))

    def unpark(self, parked: Parked) -> None:
        """Nothing waits for these calls' outputs any more."""
        for call_id in parked.calls:
            self.parked.pop(call_id, None)
        if parked.timer is not None:
            parked.timer.cancel()

    def admit_client_request(
        self, input_items: list[dict], *, delegation_id: str | None = None, settings: dict | None = None
    ) -> Admission:
        """Accept a request the client made itself, or refuse it before anything changes.

        * Function outputs for calls a response ended with continue it: the delegation's next response opens on this
          request, and the waiting tasks go on with the outputs, then the typed messages that came with them. The
          outputs must answer every call of that response (``function_call_outputs_required`` otherwise).
        * Outputs for calls whose Work has ended (after the client-tool deadline, or calls the IM does not know) are
          recorded only, and the request gets an empty ``response.completed``; typed messages that came with them run
          as a typed run instead.
        * Anything else is a typed run: an arrival that brings its own settings, even with no items.

        Raises:
            DelegationError: The session is closing, or outputs are missing (with its official ``code``).
            ValueError: A result is not a ``function_call_output``, has no ``call_id``, comes twice, or the results
                answer calls of more than one response.
        """
        if self.closed:
            raise DelegationError("Session is closing")
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
        if messages:  # the caller's typed messages with the outputs ask anew
            self.admitted += 1
        waiting.work.continue_with(waiting, outputs, messages, brings)
        self.observe("tools.wait.completed", delegation_id=waiting.delegation_id, call_ids=list(waiting.calls))
        return Admission(waiting.delegation_id, stream)

    def _late(
        self, delegation_id: str | None, outputs: dict[str, dict], messages: list[dict], settings: dict | None
    ) -> Admission:
        """Outputs whose Work has ended: recorded on their actions; an empty ``completed``, or a typed run."""
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

    # ------------------------------------------------------------------------------------------ close
    async def close(self) -> None:
        """End the session; safe to call twice.

        Queued Works and those waiting for the app stop at once, and running Works get
        ``delegation.close_grace_seconds`` (``None``: until they end). Then every open task stops and every open
        response ends ``response.incomplete`` with status ``cancelled``, in the order its request was accepted. The
        backend adapter closes last, whatever happens.
        """
        if self._closing is None:
            self._closing = asyncio.create_task(self._close(), name="im-close")
        await asyncio.shield(self._closing)

    async def _close(self) -> None:
        self.closed = True
        self._decided.set()  # no decision comes any more: nothing waits for one
        try:
            if self.placer is not None:
                self.placer.cancel()
                await asyncio.gather(self.placer, return_exceptions=True)
            at_once = [work for work in self.works.values() if work.record.state in ("queued", "waiting_client")]
            await asyncio.gather(*(work.stop(AtClose()) for work in at_once))
            running = list(self.works.values())
            if running:
                try:
                    async with asyncio.timeout(self.settings.close_grace_seconds):
                        await asyncio.gather(*(work.done.wait() for work in running))
                except TimeoutError:
                    pass
            asked = [(work, work.ask_stop(AtClose())) for work in list(self.works.values())]
            for delegation in self.book.open_delegations():
                self.book.cancelled(delegation)
            for record in self.ledger.delegations.values():
                if not record.ended:
                    record.move("cancelled")
            await asyncio.gather(*(work.confirm_stop(tasks) for work, tasks in asked))
            if self._background:
                await asyncio.gather(*self._background, return_exceptions=True)
        finally:
            self.notes.close()
            if not self.ledger.session.ended:
                self.ledger.session.move("ended")
            await self.backend.close()
