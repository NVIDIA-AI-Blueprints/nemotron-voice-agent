# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One IM session: admission, placement through the conductor, the scheduler, close."""

from __future__ import annotations

import asyncio
import contextvars
import json
from collections import deque
from collections.abc import Callable, Coroutine, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from loguru import logger

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    AppOutputStep,
    BackendAdapter,
    CallerLine,
    CallerMessage,
    Cancel,
    CompleteDecision,
    Conductor,
    ConductorInput,
    Decision,
    DecisionRejected,
    Outcome,
    PlacementExpired,
    Request,
    Start,
    Steer,
    Steering,
    TaskSpec,
    TaskSummary,
    WorkSummary,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.config import DelegationSettings, Texts
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.conversation import (
    Conversation,
    TaskHistory,
    owed_outputs,
    with_ids,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.decisions import (
    DecisionExecutor,
    RoutingTargetUnavailable,
    WorkerCapacityUnavailable,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.input import (
    BackendTask,
    ClarificationReceipt,
    InputScope,
    caller_lines,
    compose,
    is_user_message,
    records_parts,
    scoped_items,
    settings_for,
    with_parts,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.notes import ProgressNotes
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.ownership import ResponseOwnership
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.results import ResultFormat
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.run import (
    RECORD_STEPS,
    RECORD_TEXT_CHARS,
    Parked,
    TaskRun,
    WorkRun,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.stop import (
    AtAttachmentLoss,
    AtClose,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.stream import (
    ResponseBook,
    ResponseStream,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.items import (
    call_arguments,
    is_message,
    message_text,
    validate_output_content,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.ledger.records import (
    MAX_KEPT_CLOSED,
    ControlAction,
    Placement,
    Runner,
    Task,
    WorkLedger,
)
from nemotron_voice_agent.pipecat.shared.delegation_contract import (
    AppCallDelivery,
    AppResultState,
    AppResultWindow,
    CallerContext,
    RetainedAppCall,
    SettleCallerTurn,
)
from nemotron_voice_agent.pipecat.shared.ids import approx_tokens, uid

FUNCTION_CALL_OUTPUTS_REQUIRED = "function_call_outputs_required"
NO_CALL = "Session waits for a later call: its call has ended"


class DelegationError(Exception):
    """Raised when the IM refuses a request, before anything changed; ``code`` is the official code, if any."""

    def __init__(self, message: str, code: str | None = None):
        """Refuse a request with ``message``, and with ``code`` when an official code names the refusal."""
        super().__init__(message)
        self.code = code


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

    async def submit(self, decision: CompleteDecision) -> str:
        """Admit a complete source-bound decision through the Core executor."""
        return self.session.admit_decision(decision)

    async def start(self, decision: Start) -> str:
        """Submit the deterministic policy assignment as one complete decision."""
        return await self.submit(CompleteDecision((decision,)))

    async def steer(self, decision: Steer) -> str:
        """Submit the deterministic policy update through the same executor."""
        return await self.submit(CompleteDecision((decision,)))

    async def cancel(self, decision: Cancel) -> str:
        """Submit a scoped stop through the common control executor."""
        return await self.submit(CompleteDecision((decision,)))

    async def caller_message(self, decision: CallerMessage) -> str:
        """Submit a routing question or operational status with its source ownership."""
        return await self.submit(CompleteDecision((decision,)))


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
        self.conversation_ref = uid("conversation_trace")
        self.ownership = ResponseOwnership(self)
        self.book = ResponseBook(self.ledger, self.ownership.provenance)
        self.notes = ProgressNotes(self.book, settings.progress_note_seconds)
        self.conversation = Conversation(settings.context_tokens, settings.tool_output_chars, texts.cut_output)
        self.incoming: dict[str, Incoming] = {}
        self.arrivals: list[str] = []
        self.works: dict[str, WorkRun] = {}
        self.ended_works: deque[WorkRun] = deque(maxlen=settings.records_max_works)
        self.ready: deque[TaskRun] = deque()
        self.running: set[TaskRun] = set()
        self._capacity_change: asyncio.Future[None] | None = None
        self.parked: dict[str, Parked] = {}
        self._app_result_window: AppResultWindow | None = None
        self.admitted = 0
        self._input_order: dict[str, int] = {}
        self._caller_context: dict[str, CallerContext] = {}
        self._last_input_order = 0
        self._failed_frontend_input_order = 0
        self._call_generation = 0
        # The newest call let run: its name, arguments, call_id, and the requests admitted when it ran.
        self.newest_call: tuple[str, str | None, str, int] | None = None
        self.conductor: Conductor | None = None
        self.decisions = Decisions(self)
        self.executor = DecisionExecutor(self)
        self._admitting_decision = False
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
        self.continuable = False
        self._leaving: asyncio.Task | None = None
        self._attaching_call = False
        self._keep: asyncio.TimerHandle | None = None
        self._backend_closing: asyncio.Task | None = None
        self.thread_history: dict[str, list[dict]] = {}
        self.last_ended_work: str | None = None
        self.clarifications: dict[str, dict] = {}
        self._decision_epoch = 0
        self._placement_serial = 0
        self._decision_subjects: list[dict] = []
        self.caller_attached = True
        self._attachment_epoch = 0
        self._attachment = asyncio.Event()
        self._attachment.set()
        self._caller_turns: set[object] = set()
        self._caller_turns_settled = asyncio.Event()
        self._caller_turns_settled.set()

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
        tools = ("cancel_work", "set_work_state", "caller_message", "resolve_questions")
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

    async def decided(self, *, require_attachment: bool = True) -> None:
        """Wait until no request waits for the conductor's decision (at once when none does, or after the close)."""
        while not self._decided.is_set():
            await self._decided.wait()
        if require_attachment:
            await self._attachment.wait()

    @property
    def caller_turn_pending(self) -> bool:
        """Whether finalized caller input is still being routed by the frontend."""
        return bool(self._caller_turns)

    def begin_caller_turn(self) -> SettleCallerTurn | None:
        """Hold dispatch for one exact turn, including input restored before the listener reattaches."""
        if self.closed or self.detached:
            return None
        turn = object()
        self._caller_turns.add(turn)
        self._caller_turns_settled.clear()

        def settle(*, routing_failed: bool = False) -> None:
            if turn not in self._caller_turns:
                return
            if routing_failed:
                self._last_input_order += 1
                self._failed_frontend_input_order = self._last_input_order
                self._fence_existing_work(self._failed_frontend_input_order)
            self._caller_turns.remove(turn)
            if not self._caller_turns:
                self._caller_turns_settled.set()

        return settle

    def _clear_caller_turns(self) -> None:
        self._caller_turns.clear()
        self._caller_turns_settled.set()

    async def dispatch_placement(self, *, require_attachment: bool = False) -> None:
        """Wait for frontend routing and Core placement before the next dispatch check."""
        await self._caller_turns_settled.wait()
        await self.decided(require_attachment=require_attachment)

    async def wait_attached(self) -> None:
        """Wait for a listening attachment without changing any Work or effect deadline."""
        await self._attachment.wait()

    @property
    def attachment_ready(self) -> bool:
        """Whether the caller attachment has finished reopening its admission gate."""
        return not self.closed and self.caller_attached and self._attachment.is_set()

    async def _placement_attached(self) -> bool:
        """Hold admitted placement while the listener is absent, without reissuing an accepted model turn."""
        while not self.caller_attached and not self.closed:
            await self._attachment.wait()
        return not self.closed

    async def set_attachment(self, attached: bool) -> dict:
        """Fence attachment-owned actions independently of long after-call continuation."""
        if self.closed:
            raise DelegationError("Session is closing")
        control_attachment = getattr(self.conductor, "set_attachment", None)
        if attached:
            epoch = self._attachment_epoch
            if callable(control_attachment):
                await control_attachment(True)
            if epoch == self._attachment_epoch and not self.closed:
                self.caller_attached = True
                self._attachment.set()
                for work in self.works.values():
                    for task in work.open_tasks():
                        if task.control.has_input():
                            task.wake.set()
        else:
            self.caller_attached = False
            self._attachment_epoch += 1
            self._attachment.clear()
            self._clear_caller_turns()
            fence = getattr(self.conductor, "fence_attachment_sources", None)
            if callable(fence):
                self._uncertain_control_sources(fence())
            capability = getattr(self.backend, "can_run_without_attachment", None)
            qualified = capability() if callable(capability) else False
            # Only adapters explicitly qualifying their complete tool surface may run unattended.
            # In-process LLM tools all pass TaskContext.route; unknown/native tools fail closed.
            stops = [control_attachment(False)] if callable(control_attachment) else []
            stops += [
                work.stop(AtAttachmentLoss())
                for work in list(self.works.values())
                if not work.record.after_call
                and (not qualified or any(tool.get("type") != "function" for tool in work.settings.tools))
            ]
            await asyncio.gather(*stops)
        result = {"attached": self.attachment_ready}
        if attached and result["attached"]:
            result["pending_calls"] = [
                call_id
                for call_id in self.parked
                if (action := self.ledger.actions.get(call_id)) is not None
                and action.runner == "app"
                and action.state == "started"
                and not self.parked[call_id].work.ended
            ]
        return result

    def _uncertain_control_sources(self, sources: Sequence[str]) -> None:
        """Close dispatched, still-unplaced native sources with uncertainty instead of replaying possible effects."""
        for delegation in sources:
            if delegation not in self.arrivals:
                continue
            self._take([delegation])
            self.book.start(delegation)
            line = self.texts.owed_output
            item = self.book.running(delegation).message(line, "final_answer", source_item=True)
            self.book.failed(delegation, "server_error", line)
            self.ledger.delegations[delegation].move("failed")
            self.conversation.join([*self.request_items(delegation), item])
            self.observe("im.controls.uncertain", delegation_id=delegation, control_receipt_retained=False)

    @property
    def history(self) -> list[dict]:
        """The session's conversation, oldest first: ended Works and closed exchanges, then the open Works' items."""
        _, items = self.conversation.window()
        for work in self.works.values():
            items += deepcopy(work.requested)
            for task in work.tasks:
                if task.view is not None:
                    items += task.record.without_superseded_answer(task.view.own())
        return items

    def spawn(self, work: Coroutine) -> asyncio.Task:
        """Run ``work`` in the background; the close waits for it."""
        task = asyncio.create_task(work)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        task.add_done_callback(_logged)
        return task

    def admit(
        self,
        input_items: list[dict],
        *,
        delegation_id: str | None = None,
        settings: dict | None = None,
        caller_excerpt: str | None = None,
        caller_context: CallerContext | None = None,
    ) -> Admission:
        """Accept a delegation in one step, or refuse it with ``DelegationError`` before anything changes."""
        return self._accept(
            input_items, delegation_id, settings, None, caller_excerpt=caller_excerpt, caller_context=caller_context
        )

    def _accept(
        self,
        input_items: list[dict],
        delegation_id: str | None,
        settings: dict | None,
        brings: dict | None,
        *,
        caller_excerpt: str | None = None,
        caller_context: CallerContext | None = None,
    ) -> Admission:
        if self.closed:
            raise DelegationError("Session is closing")
        if not self.caller_attached:
            raise DelegationError("The call waits for a listening attachment")
        if self.detached:
            raise DelegationError(NO_CALL)
        deferred_source = delegation_id is not None and any(
            delegation_id in row["source_delegation_ids"]
            for row in self.clarifications.values()
            if row["state"] == "waiting"
        )
        if delegation_id is not None and (delegation_id in self.ledger.delegations or deferred_source):
            raise DelegationError(f"Delegation {delegation_id} already exists")
        pending = {record.id for record in self.ledger.delegations.values() if record.state == "queued"}
        waiting = len(pending | self.ownership.pending_sources())
        if waiting >= self.settings.max_pending_delegations:
            raise DelegationError("Delegation queue is full")
        settings, delegation_id = deepcopy(settings or {}), delegation_id or uid("item")
        delegation = self.ledger.new_delegation(
            delegation_id, input_items=with_ids(input_items, delegation_id), settings=settings
        )
        delegation.call_generation = self._call_generation
        delegation.caller_excerpt = caller_excerpt
        if caller_context is not None and delegation.input and is_user_message(delegation.input[0]):
            self._caller_context[delegation.input[0]["id"]] = deepcopy(caller_context)
        self.remember_input_order(delegation.input)
        stream = ResponseStream()
        self.book.open(delegation.id, stream, *self.response_settings(settings))
        self.incoming[delegation.id] = Incoming(brings)
        self.arrivals.append(delegation.id)
        self.admitted += 1
        self._decided.clear()
        self.observe("backend.queued", delegation_id=delegation.id, waiting=waiting + 1)
        if self._deciding is not None and not self._deciding.done() and not self._placed_any and self._may_restart:
            # Invalidate placement before asking the old invocation to cancel. Its cleanup may still call place().
            self._restart, self._may_restart = True, False
            self._call = None
            self._deciding.cancel(PlacementExpired)
        self._wake.set()
        return Admission(delegation.id, stream)

    @property
    def call_generation(self) -> int:
        """Identify the call that currently owns the conversation."""
        return self._call_generation

    def caller_order(self, message_id: str) -> int | None:
        """Return a caller message's admission order when retained."""
        return self._input_order.get(message_id)

    def remember_input_order(self, items: Sequence[dict]) -> None:
        """Record caller chronology on admission, independently of later per-task application order."""
        for item in items:
            identity = item.get("id")
            if item.get("role") == "user" and isinstance(identity, str) and identity not in self._input_order:
                self._last_input_order += 1
                self._input_order[identity] = self._last_input_order

    def latest_input_order(self, items: Iterable[dict]) -> int:
        """Return the newest admitted caller-message order in the original input items, or zero."""
        return max(
            (
                self._input_order.get(identity, 0)
                for item in items
                if is_user_message(item) and isinstance(identity := item.get("id"), str)
            ),
            default=0,
        )

    def request_items(self, delegation_id: str) -> list[dict]:
        """Return a request's input as the ledger recorded it; none for a request the ledger forgot."""
        record = self.ledger.delegations.get(delegation_id)
        return deepcopy(record.input) if record is not None else []

    def request_view(self, delegation_id: str, *, pending: bool = False) -> Request:
        """Return the full input and its separately retained caller excerpt for a job record."""
        record = self.ledger.delegations.get(delegation_id)
        return Request(
            delegation_id,
            tuple(self.request_items(delegation_id)),
            "pending" if pending else "landed",
            record.caller_excerpt if record is not None else None,
        )

    def bind_worker_input(
        self,
        text: str,
        items: Iterable[dict],
        task_ref: str | None,
        *,
        selections: tuple[CallerLine, ...] = (),
        require_lines: bool = False,
        include_bound_lines: bool = False,
    ) -> InputScope:
        """Bind a worker's excerpt against retained caller and task playback evidence."""
        return InputScope.bind(
            text,
            items,
            self._caller_context,
            self._playback_owners(task_ref),
            selections=selections,
            require_lines=require_lines,
            include_bound_lines=include_bound_lines,
        )

    def _playback_owners(self, reference: str | None) -> frozenset[str]:
        """Return retained task identities whose publications belong to this exact continuation."""
        work = self.ledger.works.get(reference) if reference is not None else None
        if work is not None and len(work.tasks) == 1:
            reference = work.tasks[0].id
        owners: set[str] = set()
        while reference is not None and reference not in owners:
            task = self.ledger.tasks.get(reference)
            if task is None:
                break
            owners.add(task.id)
            reference = task.continues
        return frozenset(owners)

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

        Normal leftovers get one retry; questions remain until explicit resolution or ownership transfer.
        """
        restarted, normal_rounds = False, 0
        decision_rejection = None
        self._placed_any = False
        while True:
            if not await self._placement_attached():
                return
            given_ids = list(self.arrivals)
            self._given, self._restart = set(given_ids), False
            self._may_restart = not restarted
            self._decision_epoch += 1
            try:
                given, cursor = self._conductor_input(given_ids, attempt=normal_rounds + 1)
            except Exception as failure:
                error = failure
                break
            if decision_rejection is not None:
                given = replace(given, decision_rejection=decision_rejection)
            call = self._call = object()
            attachment_epoch = self._attachment_epoch
            self.observe("im.routing.started", delegation_ids=given_ids)
            deciding = self._deciding = asyncio.create_task(self.conductor.decide(given, self._place_for(call)))
            try:
                await asyncio.wait({deciding})
            finally:
                if not deciding.done():
                    deciding.cancel()
                self._deciding, self._call = None, None
            error = asyncio.CancelledError() if deciding.cancelled() else deciding.exception()
            if attachment_epoch != self._attachment_epoch:
                # Detachment already settled uncertain native inputs. Only arrivals still pending can be routed again.
                self._given = set()
                if not self.arrivals:
                    self._decided.set()
                return
            if self._restart:
                restarted = True
                self.observe("im.decide.restarted", arrivals=len(self.arrivals))
                continue
            normal_rounds += 1
            remaining = [d for d in given_ids if d in self.arrivals]
            if error is None:
                self._cursor = cursor
            repairable = error is None or isinstance(error, DecisionRejected) and not self._placed_any
            if repairable and remaining and normal_rounds < 2 and not self.closed:
                decision_rejection = str(error) if isinstance(error, DecisionRejected) else None
                self.observe(
                    "im.decide.retry",
                    delegation_ids=remaining,
                    reason="unplaced" if error is None else "decision_rejected",
                )
                continue
            break
        self._given = set()
        capacity = isinstance(error, WorkerCapacityUnavailable)
        unavailable = isinstance(error, RoutingTargetUnavailable)
        if error is not None and not (capacity or unavailable):
            logger.warning("The conductor failed; unresolved routing: {}: {}", type(error).__name__, error)
        leftovers = [delegation for delegation in given_ids if delegation in self.arrivals]
        if leftovers:
            await self._fail_unplaced(
                leftovers,
                capacity=capacity,
                unavailable=unavailable,
                rejected=isinstance(error, DecisionRejected) and not self._placed_any,
                execution_failed=error is not None and not isinstance(error, DecisionRejected),
            )
        if not self.arrivals:
            self._decided.set()

    async def _fail_unplaced(
        self,
        delegations: list[str],
        *,
        capacity: bool = False,
        unavailable: bool = False,
        rejected: bool = False,
        execution_failed: bool = False,
    ) -> None:
        """Close unresolved routing with the outcome established by the last decision attempt."""
        if not await self._placement_attached():
            return
        self.fence_unresolved_inputs(delegations)
        self._given = set(delegations)
        self._take(delegations, self._given)
        self._given = set()
        failure_output = self.texts.conductor_failed_output if execution_failed else self.texts.routing_failed_output
        for delegation in delegations:
            self.book.start(delegation)
            if capacity or unavailable or rejected:
                notice = (
                    self.texts.capacity_unavailable_output
                    if capacity
                    else self.texts.routing_failed_output
                    if unavailable
                    else self.texts.routing_rejected_output
                )
                self.routing_notice(delegation, notice, remember_input=True)
                self.book.end_after_calls(delegation, "response.completed")
                self.ledger.delegations[delegation].move("answered")
            else:
                self.book.running(delegation).message(failure_output, "final_answer", source_item=True)
                self.book.failed(delegation, "server_error", failure_output)
                self.ledger.delegations[delegation].move("failed")
        if capacity:
            self.observe("im.routing.capacity_unavailable", delegation_ids=delegations)
        else:
            self.observe("im.routing.unresolved", delegation_ids=delegations)

    def fence_unresolved_inputs(self, delegations: Iterable[str], *, tasks: Iterable[Task] | None = None) -> int:
        """Fence older work and return the input order for any work still to be allocated."""
        order = self.latest_input_order(
            item for delegation in delegations for item in self.ledger.delegations[delegation].input
        )
        if order and tasks is None:
            self._fence_existing_work(order)
        elif order:
            for task in dict.fromkeys(self.latest_task(record) for record in tasks):
                work = self.works.get(task.work)
                run = next((run for run in work.tasks if run.task_id == task.id), None) if work is not None else None
                if run is not None and not run.ended:
                    run.fence_unresolved_input(order)
                elif task.successor_task_id is None:
                    task.record_unresolved_input(order)
        return order

    def _fence_existing_work(self, order: int) -> None:
        for work in self.works.values():
            for task in work.tasks:
                task.fence_unresolved_input(order)

    def routing_notice(self, delegation: str, text: str, *, remember_input: bool) -> None:
        """Send a response-scoped notice and retain its conversation context."""
        writer = self.book.running(delegation)
        if writer is not None:
            item = writer.message(text, "final_answer", source_item=True)
            items = self.request_items(delegation) if remember_input else []
            self.conversation.join([*items, item])

    def _conductor_input(self, given_ids: list[str], *, attempt: int = 1) -> tuple[ConductorInput, int]:
        """Return what one ``decide()`` reads, and the conversation position it reads up to."""
        items, new_from, read_through = self.conversation.read(self._cursor)
        arrivals = tuple(
            replace(request, caller_lines=caller_lines(request.input, self._caller_context))
            for request in map(self.request_view, given_ids)
        )
        works = self.work_summaries()
        self._decision_subjects = [
            {
                "work_id": work.id,
                "state": work.record.state,
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "state": task.record.state,
                        "input_revision": task.input_revision,
                        "questions": [
                            {
                                "question_id": question.id,
                                "input_revision": question.input_revision,
                                "dispatch": question.dispatch,
                            }
                            for question in task.record.questions
                            if question.open and question.owner_task_id == task.task_id
                        ],
                    }
                    for task in work.tasks
                ],
            }
            for work in [*self.works.values(), *self.ended_works]
        ]
        pending = tuple(
            {**deepcopy(row), "source_lines": caller_lines(row["input"], self._caller_context)}
            for row in self.clarifications.values()
            if row["state"] == "waiting"
        )
        return ConductorInput(
            arrivals,
            works,
            tuple(items),
            new_from,
            self.last_ended_work,
            pending,
            attempt,
            conversation_ref=self.conversation_ref,
            last_ended_task=self._last_ended_task(),
            control_actions=self.control_view(),
            available_worker_slots=self.available_worker_slots(),
        ), read_through

    def control_view(self) -> tuple[dict, ...]:
        """Keep active and explicitly referenced obligations alongside recent control outcomes."""
        values = list(self.ledger.controls.values())
        needed = {
            key
            for row in self.clarifications.values()
            if row["state"] == "waiting"
            for key in row.get("accepted_action_ids", ())
        }
        recent = {control.id for control in values[-MAX_KEPT_CLOSED:]}
        return tuple(
            asdict(control)
            for control in values
            if control.id in needed or control.id in recent or control.state in ("pending", "unknown")
        )

    def work_summaries(self) -> tuple[WorkSummary, ...]:
        """The same operational view for either routing producer."""
        works = tuple(work.summary() for work in [*self.works.values(), *self.ended_works])
        return (*works, *self.retained_question_works({work.work_id for work in works}))

    def task_results(self, task_id: str) -> tuple[tuple[dict, ...], int]:
        """Expose bounded actual operation evidence for status without restarting the business worker."""
        actions = [
            action for action in self.ledger.actions.values() if action.task == task_id and action.runner != "im"
        ]
        selected = [
            action
            for action in actions
            if action in actions[-RECORD_STEPS:] or action.state == "unclear" and action.late_outcome is None
        ]

        def excerpt(value):
            if value is None:
                return None, False
            text = value if isinstance(value, str) else json.dumps(value)
            if len(text) <= RECORD_TEXT_CHARS:
                return deepcopy(value), False
            half = RECORD_TEXT_CHARS // 2
            return text[:half] + self.texts.cut_output.format(chars=len(text) - 2 * half) + text[-half:], True

        rows = []
        for action in selected:
            fact = action.execution_fact()
            output, cut = excerpt(fact["output"])
            arguments, arguments_cut = excerpt(fact["arguments"])
            rows.append(
                {
                    "call_id": fact["call_id"],
                    "task_id": fact["task_id"],
                    "work_id": action.work,
                    "name": fact["name"],
                    "runner": fact["runner"],
                    "arguments": arguments,
                    "arguments_truncated": arguments_cut,
                    "outcome": fact["outcome"],
                    "has_result": fact["has_result"],
                    "output": output,
                    "output_truncated": cut,
                    "late_evidence": action.late_outcome is not None,
                }
            )
        return tuple(rows), len(actions) - len(selected)

    def retained_question_works(self, shown: set[str]):
        """Expose unresolved thread questions beyond the recent-result presentation window."""
        for work in self.ledger.works.values():
            if work.id in shown:
                continue
            tasks = []
            for task in work.tasks:
                questions = tuple(
                    {
                        "question_id": q.id,
                        "text": q.text,
                        "dispatch": q.dispatch,
                        "resolution": q.resolution,
                        "owner_task_id": q.owner_task_id,
                    }
                    for q in task.questions
                    if q.open and q.owner_task_id == task.id
                )
                if questions:
                    results, omitted = self.task_results(task.id)
                    tasks.append(
                        TaskSummary(
                            task.id,
                            task.state,
                            task.brief,
                            (),
                            task.acted,
                            task.progress,
                            0.0,
                            task.question,
                            (task.retained_answer or "")[:RECORD_TEXT_CHARS] or None,
                            questions=questions,
                            stop=task.stop.state,
                            answer_publication=task.answer_publication,
                            tool_results=results,
                            tool_results_omitted=omitted,
                            continues=task.continues,
                            successor_task_id=task.successor_task_id,
                            successor_work_id=task.successor_work_id,
                            steering=self.task_steering_state(task),
                        )
                    )
            if tasks:
                yield WorkSummary(work.id, work.state, work.carrier, (), tuple(tasks), 0.0, work.after_call)

    def latest_task(self, task: Task) -> Task:
        """Follow retained successors of an exact task."""
        while task.successor_task_id in self.ledger.tasks:
            task = self.ledger.tasks[task.successor_task_id]
        return task

    def task_steering_state(self, task: Task) -> Literal["active", "continuable", "continued", "unavailable"]:
        """Describe exact current eligibility without selecting another thread or mutating records."""
        if task.successor_task_id is not None:
            return "continued"
        if task.stop.state != "none" or task.continuation_cancelled_by is not None:
            return "unavailable"
        work = self.works.get(task.work)
        live = next((run for run in work.tasks if run.task_id == task.id), None) if work is not None else None
        if live is not None and not live.ended and live.stop_reason is None and not live.own_cancel:
            return "active"
        if task.state in ("answered", "failed") and task.id in self.thread_history:
            return "continuable"
        return "unavailable"

    def cancel_ended_questions(self, task: Task, action_id: str) -> list[str]:
        """Close the ended thread's own unresolved questions without claiming any physical stop."""
        if not task.ended or task.successor_task_id is not None:
            return []
        closed = task.close_questions("cancelled")
        if not closed:
            return []
        task.continuation_cancelled_by = action_id
        if task.answer_publication == "pending":
            self.supersede_answer(task)
        for question in closed:
            self.observe(
                "im.question.closed",
                question_id=question.id,
                task_id=task.id,
                work_id=task.work,
                disposition="cancelled",
                dispatch=question.dispatch,
                input_ref=question.input_ref,
                action_id=action_id,
            )
        return [question.id for question in closed]

    def supersede_answer(self, task: Task) -> None:
        """Retain the task's evidence while withdrawing an obsolete draft from shared history."""
        task.answer_publication = "superseded"
        if task.answer_item_id is not None:
            self.conversation.discard(task.answer_item_id)

    def _last_ended_task(self) -> str | None:
        work = self.ledger.works.get(self.last_ended_work)
        if work is None or len(work.tasks) != 1:
            return None
        task = work.tasks[0]
        if self.task_steering_state(task) != "continuable":
            return None
        return task.id

    def _place_for(self, call: object) -> Callable[[Decision], Coroutine[Any, Any, str]]:
        """Return the ``place`` one ``decide()`` call gets, which refuses what it places after that call ended.

        A runtime that kept a cancelled or finished turn may still call it, also while a later ``decide()`` runs.
        """

        def current() -> bool:
            return call is self._call and self._deciding is not None and not self._deciding.done()

        async def place(decision: Decision) -> str:
            if not current():
                return PlacementExpired(self._refused("ended"))
            if not await self._placement_attached() or not current():
                return PlacementExpired(self._refused("ended"))
            return await self.place(decision)

        return place

    async def place(self, decision: Decision) -> str:
        """Apply one decision of the conductor and return its outcome line."""
        if not await self._placement_attached():
            return PlacementExpired(self._refused("ended"))
        self._placement_serial += 1
        complete = decision if isinstance(decision, CompleteDecision) else CompleteDecision((decision,))
        outcome = await complete.apply(self.decisions)
        self.observe("im.placed", decision=type(decision).__name__, outcome=outcome)
        return outcome

    def admit_decision(self, decision: CompleteDecision) -> str:
        """Submit against the current conductor invocation's still-pending caller sources."""
        queued = tuple(source for source in self.arrivals if source in self._given)
        return self.executor.submit(decision, epoch=self._decision_epoch, queued=queued)

    @contextmanager
    def applying_decision(self, sources: tuple[str, ...]) -> Iterator[None]:
        """Delay closing responses and releasing the routing frontier until the complete batch is applied."""
        self._admitting_decision = True
        try:
            yield
        finally:
            self._admitting_decision = False
            if not self.arrivals:
                self._decided.set()
            for source in sources:
                self.ownership.finish_owner(self.ownership.owner(source))
            self._prune()

    def take_decision_sources(self, sources: tuple[str, ...]) -> None:
        """Consume only requests bound to the current placement invocation."""
        self._take(sources, self._given)

    def control_settled(self, sources: tuple[str, ...]) -> None:
        """Revise each owner and complete ready responses outside a batch admission."""
        for source in sources:
            owner = self.ownership.owner(source)
            owner.revision += 1
            if not self._admitting_decision:
                self.ownership.finish_owner(owner)

    def can_publish_lifecycle(self, source: str, *, retain_for_reconnect: bool) -> bool:
        """Check whether a lifecycle update still belongs to this call and an open response."""
        record = self.ledger.delegations.get(source)
        call_open = not self.detached if retain_for_reconnect else self.caller_attached
        return (
            not self.closed
            and call_open
            and record is not None
            and not record.ended
            and record.call_generation == self._call_generation
        )

    def _take(self, delegations: Iterable[str], given: set[str] | None = None) -> list[str]:
        """Take the unplaced arrivals among ``delegations``, and among ``given`` when it is named, oldest first."""
        wanted = set(delegations) if given is None else set(delegations) & given
        taken = [delegation for delegation in self.arrivals if delegation in wanted]
        if taken:
            self._placed_any = True
        for delegation in taken:
            self.arrivals.remove(delegation)
        if not self._admitting_decision and not self.arrivals and not self._decided.is_set():
            self._decided.set()
            self.observe("im.decided", delegation_ids=taken)
        return taken

    def _refused(self, reason: str, **values: str) -> str:
        """Return the outcome of a refused decision, with the reason ``texts.refusal_reasons`` words for it."""
        return self.texts.refusal(reason, **values)

    def available_worker_slots(self) -> int:
        """Count open scheduler slots from running tasks and queued reservations."""
        active = len(self.running) + sum(
            task not in self.running and task.record.state == "queued"
            for work in self.works.values()
            for task in work.tasks
            if not task.ended
        )
        return max(0, self.backend.max_running - active)

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

    def clarification_input(self, ids: Sequence[str]) -> list[dict] | None:
        """Read retained clarification input and its accepted action receipts."""
        rows = [self.clarifications.get(key) for key in ids]
        if any(row is None or row["state"] != "waiting" for row in rows) or len(set(ids)) != len(ids):
            return None
        items = [deepcopy(item) for row in rows for item in row["input"]]
        for row in rows:
            ids = row.get("accepted_action_ids", ())
            accepted = [asdict(self.ledger.controls[key]) for key in ids if key in self.ledger.controls]
            if accepted:
                items.append(
                    ClarificationReceipt(
                        type="message",
                        role="developer",
                        id=f"{row['id']}.accepted",
                        content=[
                            {
                                "type": "input_text",
                                "text": json.dumps({"private_backend_state": {"accepted_actions": accepted}}),
                            }
                        ],
                    )
                )
        return items

    def clarification_sources(self, identities: Sequence[str]) -> tuple[str, ...]:
        """Return the original source IDs from the validated clarification records."""
        return tuple(
            source for identity in identities for source in self.clarifications[identity]["source_delegation_ids"]
        )

    def consume_clarifications(
        self,
        identities: tuple[str, ...],
        control: ControlAction,
        *,
        superseded_by: str | None = None,
    ) -> None:
        """Record each action that uses a clarification in its validated input snapshot."""
        for key in identities:
            row = self.clarifications[key]
            row["state"] = "consumed"
            row.setdefault("consumed_by_actions", []).append(control.id)
            self.observe(
                "im.clarification.consumed",
                clarification_id=key,
                action_id=control.id,
                delegation_ids=list(control.sources),
                source_delegation_ids=list(row["source_delegation_ids"]),
            )
        if superseded_by is not None:
            for key in identities:
                self.clarifications[key].update(state="superseded", superseded_by=superseded_by)

    def open_clarification(self, action: CallerMessage, control: ControlAction, extra: Sequence[dict]) -> str:
        """Retain the routing question's original sources, context and placement coordinates."""
        waiting = sum(row["state"] == "waiting" for row in self.clarifications.values())
        if waiting - len(action.clarifications) >= self.settings.max_pending_delegations:
            raise ValueError("Too many unresolved routing questions")
        clarification_id = uid("clarification")
        prior_sources = self.clarification_sources(action.clarifications)
        accepted_ids = {
            key for source in action.delegations for key in self.ownership.owner(source).controls if key != control.id
        }
        accepted_ids.update(
            key for old in action.clarifications for key in self.clarifications[old].get("accepted_action_ids", ())
        )
        row = {
            "id": clarification_id,
            "action_id": control.id,
            "state": "waiting",
            "question": action.message,
            "source_delegation_ids": list(dict.fromkeys([*prior_sources, *action.delegations])),
            "reply_delegation_ids": list(action.delegations),
            "supersedes": list(action.clarifications),
            "source_epoch": self._decision_epoch,
            "source_placement": self._placement_serial,
            "carrier": action.delegations[-1],
            "subjects": self._decision_subjects,
            "input": [*extra, *[item for source in action.delegations for item in self.request_items(source)]],
            "accepted_action_ids": sorted(accepted_ids),
        }
        if approx_tokens(json.dumps(row)) > self.settings.context_tokens:
            raise ValueError("Routing clarification exceeds context budget")
        self.clarifications[clarification_id] = row
        return clarification_id

    def clarification_written(self, identity: str, item: dict) -> None:
        """Retain the question that actually reached the caller's response."""
        row = self.clarifications[identity]
        row["input"].append(item)
        self.observe(
            "im.clarification.created",
            clarification_id=identity,
            delegation_ids=row["reply_delegation_ids"],
            source_delegation_ids=row["source_delegation_ids"],
            supersedes=row["supersedes"],
            source_epoch=row["source_epoch"],
            source_placement=row["source_placement"],
        )

    def start_work(
        self,
        delegations: list[str],
        briefs: Iterable[str | TaskSpec],
        continues: str | None = None,
        *,
        extra: Sequence[dict] = (),
        settings: dict | None = None,
        input_items: Sequence[dict] | None = None,
        start_barrier: asyncio.Future[None] | None = None,
        worker_scopes: tuple[tuple[InputScope, ...], ...] = (),
        unresolved_input_order: int = 0,
    ) -> WorkRun:
        """Start a new queued Work for these requests, oldest first: the newest carries it, the others hand over.

        ``continues`` names the ended Work or task whose thread it continues; ``None`` begins a new thread.
        ``input_items`` retains an exact unapplied payload, including clarification and control evidence.
        """
        newest = self.ledger.delegations[delegations[-1]]
        specs = [spec if isinstance(spec, TaskSpec) else TaskSpec(spec) for spec in briefs]
        for spec in specs:
            if spec.verifies is not None and (spec.continues is not None or spec.verifies not in self.thread_history):
                raise ValueError("Verification needs an exact retained task and cannot resume its goal")
        if continues is None and len(specs) == 1:
            continues = specs[0].continues
        predecessors = []
        for reference in [spec.continues for spec in specs] if specs else [continues]:
            if reference is None:
                predecessors.append(None)
                continue
            source = self.ledger.works.get(reference)
            if source is not None and len(source.tasks) == 1:
                reference = source.tasks[0].id
            predecessor = self.ledger.tasks.get(reference)
            if predecessor is None or reference not in self.thread_history:
                raise ValueError(f"Task {reference} has no retained exact continuation history")
            if predecessor.continuation_cancelled_by is not None:
                raise ValueError(f"Task {reference} was closed to continuation by a stop")
            if predecessor.successor_task_id is not None:
                raise ValueError(
                    json.dumps(
                        {
                            "reason": "continued",
                            "task_id": predecessor.id,
                            "work_id": predecessor.work,
                            "successor_task_id": predecessor.successor_task_id,
                            "successor_work_id": predecessor.successor_work_id,
                        }
                    )
                )
            predecessors.append(predecessor)
        ids = [task.id for task in predecessors if task is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("A thread can have only one successor in a Work")
        record = self.ledger.new_work(
            delegations[:1],
            deepcopy(newest.settings if settings is None else settings),
            tuple(spec.brief for spec in specs),
        )
        if len(predecessors) == 1 and predecessors[0] is not None:
            continues = predecessors[0].id
        record.continues = continues
        focused = self.backend.max_running > 1 and getattr(self.conductor, "model_placement", True)
        for index, task in enumerate(record.tasks):
            task.history_mode = "focused" if focused or (specs and specs[index].verifies is not None) else "shared"
            predecessor = predecessors[index]
            task.continues = predecessor.id if predecessor is not None else None
            task.verifies = (
                specs[index].verifies
                if specs and specs[index].verifies is not None
                else predecessor.verifies
                if predecessor is not None
                else None
            )
            if task.verifies is not None:
                task.history_mode = "focused"
        work = self.works[record.id] = WorkRun(self, record)
        work.start_barrier = start_barrier
        work.requested[:0] = deepcopy(list(extra))
        for task, source_task in zip(work.tasks, predecessors, strict=True):
            task.initial_requested[:0] = deepcopy(list(extra))
            if source_task is not None:
                source_task.successor_task_id, source_task.successor_work_id = task.task_id, work.id
                if source_task.answer_publication == "pending":
                    self.supersede_answer(source_task)
                task.presentation.inherit(source_task)
                task.record.unresolved_input_order = source_task.unresolved_input_order
                for question in source_task.questions:
                    if question.open and question.owner_task_id == source_task.id:
                        if question.dispatch == "pending":
                            question.dispatch = "dropped"
                        question.previous_owners = (*question.previous_owners, question.owner_task_id)
                        question.owner_task_id = task.task_id
                        task.record.questions.append(question)
                        task.presentation.refresh_question_summary()
                        source_task.question = None
                        task.trace(
                            "im.question.transferred",
                            question_id=question.id,
                            origin_task_id=source_task.id,
                            owner_task_id=task.task_id,
                        )
        tasks = tuple(task.id for task in record.tasks)
        for delegation in delegations:
            joined = self.ledger.delegations[delegation]
            if joined.placements and continues is not None:
                joined.placements = tuple(
                    Placement(record.id, tasks) if target.work == continues or continues in target.tasks else target
                    for target in joined.placements
                )
                joined.work, joined.tasks = joined.placements[0].work, joined.placements[0].tasks
            else:
                joined.placements = (*joined.placements, Placement(record.id, tasks))
                joined.work, joined.tasks = joined.placements[0].work, joined.placements[0].tasks
        work.take_over_at_start(delegations[1:])
        if input_items is not None:
            work.requested = deepcopy(list(input_items))
            for task in work.tasks:
                task.initial_requested = deepcopy(list(input_items))
        if worker_scopes:
            for task, scopes in zip(work.tasks, worker_scopes, strict=True):
                if scopes:
                    task.initial_scopes = scopes
                    task.history_mode = task.record.history_mode = "focused"
        fence_order = max(self._failed_frontend_input_order, unresolved_input_order)
        if fence_order:
            for task in work.tasks:
                task.fence_unresolved_input(fence_order)
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

    def retained_thread(self, task_id: str | None) -> list[dict] | None:
        """Return model-facing thread evidence, omitting a superseded final by its original item identity."""
        if task_id is None:
            return None
        history = self.thread_history.get(task_id)
        record = self.ledger.tasks.get(task_id)
        return record.without_superseded_answer(history) if record is not None and history is not None else history

    def history_for(self, work: WorkRun, task: TaskRun) -> TaskHistory:
        """Return a starting task's view: the conversation's window, its Work's input, its brief, what joined before."""
        _, window = self.conversation.window()
        ancestor = self.retained_thread(task.continues or task.record.verifies) or []
        initial = scoped_items(task.initial_requested, task.initial_scopes)
        pending = scoped_items(task.pending, task.pending_scopes)
        if task.history_mode == "focused":
            window = []
            records = json.dumps(work.record_line(task=task), ensure_ascii=False)
            prefix = [*ancestor, *with_parts(initial, self.texts.records_block.format(records=records))]
        else:
            prefix = [*window, *work.input]
        view = TaskHistory(prefix, self.settings.context_tokens, self.settings.tool_output_chars, self.texts.cut_output)
        task.thread_items = deepcopy([*ancestor, *initial])
        task.ancestor_size = len(ancestor)
        if task.brief and not task.initial_scopes:
            brief = task.assignment_message()
            view.join([brief])
            task.thread_items.append(deepcopy(brief))
        view.join(pending)
        task.thread_items.extend(deepcopy(pending))
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
                self.notes.note(
                    task.work.carrier,
                    text,
                    self.ownership.attribution(task.work, task_ref=task.task_id),
                    hold=self.ownership.managed(task.work) and not task.work.unheard(),
                    eligible=task.publication_guard(),
                )

    def progress(self, task: TaskRun, text: str) -> None:
        """Write a task's activity that is not model text as a progress note."""
        if task.stop_reason is None:
            self.notes.note(
                task.work.carrier,
                text,
                self.ownership.attribution(task.work, task_ref=task.task_id),
                hold=self.ownership.managed(task.work) and not task.work.unheard(),
                eligible=task.publication_guard(),
            )

    def waits_for_app(self, task: TaskRun) -> bool:
        """Return whether a task whose round ended with the app's calls waits for their outputs: always, here."""
        return True

    def task_ended(self, task: TaskRun, *, continue_pending: bool = False) -> None:
        """Save the ended task's thread and carry its unapplied input to one exact successor."""
        history = Conversation(self.settings.context_tokens, self.settings.tool_output_chars, self.texts.cut_output)
        history.join(
            [
                *task.thread_items,
                *owed_outputs(task.thread_items, self.texts.owed_output, task.outputs),
                *task.control_items,
            ]
        )
        self.thread_history[task.task_id] = history.window()[1]
        pending = task.unapplied() if continue_pending else []
        stale = [
            waiting
            for waiting in pending
            if waiting.batch.caller_input and waiting.batch.call_generation != self._call_generation
        ]
        if stale:
            stale_sources = tuple(dict.fromkeys(source for waiting in stale for source in waiting.batch.delegations))
            for waiting in stale:
                waiting.record.move("cancelled")
                task.taken.pop(waiting.record.id, None)
                if waiting in task.steering:
                    task.steering.remove(waiting)
            for source in stale_sources:
                record = self.ledger.delegations.get(source)
                if record is not None and not record.ended:
                    self.book.cancelled(source)
                    record.move("cancelled")
            self.ownership.discard_pending(task.work, stale_sources, tasks=(task.task_id,))
            self.executor.delivered(task.task_id, stale_sources, "cancelled")
            pending = [waiting for waiting in pending if waiting not in stale]
        for waiting in pending:
            if waiting.record.state == "taken":
                task.settle(Steering(waiting.record.id, tuple(waiting.batch.worker_items())), "unknown")
        pending = [waiting for waiting in pending if waiting.record.state != "unknown"]
        if not pending:
            return
        discard = (
            self.closed
            or self.detached
            or (task.stop_reason is not None and task.stop_reason.discard_pending_requests)
            or task.stopped == "unconfirmed"
        )
        sources = list(dict.fromkeys(source for waiting in pending for source in waiting.batch.delegations))
        for waiting in pending:
            waiting.record.move("cancelled" if discard else "follow_up")
        task.steering.clear()
        if discard:
            self.ownership.discard_pending(task.work, sources, tasks=(task.task_id,))
            self.executor.delivered(task.task_id, tuple(sources), "cancelled")
            return
        unapplied_items = [deepcopy(item) for waiting in pending for item in waiting.batch.items]
        unapplied_scopes = tuple(scope for waiting in pending for scope in waiting.batch.scopes)
        sources = sorted(sources or [task.work.carrier], key=list(self.incoming).index)
        context = self.placer.get_context() if self.placer is not None else contextvars.copy_context()
        successor = context.copy().run(
            self.start_work,
            sources,
            (TaskSpec(task.brief or "", task.task_id),),
            task.task_id,
            settings=next(
                (waiting.batch.brings for waiting in reversed(pending) if waiting.batch.brings is not None),
                task.work.record.settings,
            ),
            input_items=unapplied_items,
            worker_scopes=(unapplied_scopes,),
        )
        self.executor.transfer(task.task_id, successor.tasks[0].task_id, tuple(sources))
        self.observe("im.task.follow_up", task_id=task.task_id, delegation_ids=sources)

    def work_ended(self, work: WorkRun, leftovers: list[str]) -> None:
        """Close an ended Work: its records close, its segment joins the conversation, its leftover requests go on.

        A leftover request whose call has ended is cancelled instead. While the session waits for a later call, every
        leftover request is cancelled, and once no Work is open, its keep time starts again and its backend closes.
        """
        self.last_ended_work = work.id
        self.works.pop(work.id, None)
        for parked in {parked for parked in self.parked.values() if parked.work is work}:
            self.unpark(parked, stop_state=work.record.state)
        tasks = {task.task_id: task for task in work.tasks}
        for action in self.ledger.actions.values():
            if action.work == work.id and not action.ended:
                unsent = action.runner == "app" and (not action.app_published or action.app_delivered is False)
                outcome = "not_run" if unsent else "unclear"
                action.move(outcome)
                tasks[action.task].steps[action.id][1] = outcome
        self.conversation.join(work.segment())
        self.ended_works.append(work)
        pending = [d for d in leftovers if not self.ledger.delegations[d].ended]
        waiting = self.detached and not self.closed
        gone = [
            d
            for d in pending
            if waiting or (not self.closed and self.book.writer(d) is None and not self.ownership.listening(d))
        ]
        for delegation in gone:
            self.book.cancelled(delegation)
            self.ledger.delegations[delegation].move("cancelled")
        pending = [d for d in pending if d not in gone]
        if pending and not self.closed:
            context = self.placer.get_context() if self.placer is not None else contextvars.copy_context()
            context.copy().run(self._follow_up, work, pending)
        self.observe("backend.ended", work_id=work.id, state=work.record.state, delegation_id=work.carrier)
        self._prune()
        if waiting and not self.works:
            self._keep_from_now()
            if not self._attaching_call:
                self._release_backend()

    def _follow_up(self, work: WorkRun, pending: list[str]) -> None:
        """Start the Work that continues ``work`` for its leftover requests, oldest first."""
        order = list(self.incoming)
        self.start_work(sorted(pending, key=order.index), (), work.id)
        self.observe("im.follow_up", work_id=work.id, delegation_ids=pending)

    def _prune(self) -> None:
        """Forget the oldest closed records and the requests that no record names."""
        needed = self.ownership.protected_works() | {
            work.id
            for work in self.ledger.works.values()
            if any(q.open and q.owner_task_id == task.id for task in work.tasks for q in task.questions)
        }
        execution_owners = {
            owner for work in self.works.values() for task in work.tasks for owner in task.execution_owners
        }
        execution_owners.update(owner for task in self.running for owner in task.execution_owners)
        needed.update(task.work for key, task in self.ledger.tasks.items() if key in execution_owners)
        needed.update(task.work_id for task in self.running)
        sources = {
            source
            for row in self.clarifications.values()
            if row["state"] == "waiting"
            for key in row.get("accepted_action_ids", ())
            if key in self.ledger.controls
            for source in self.ledger.controls[key].sources
        }
        for forgotten in self.ledger.prune(needed, sources):
            for task in forgotten.tasks:
                self.thread_history.pop(task.id, None)
                if task.result.state == "pushed":
                    task.result.move("expired")
        for delegation in [d for d in self.incoming if d not in self.ledger.delegations]:
            del self.incoming[delegation]
        self.ownership.prune()
        needed_inputs = {item.get("id") for record in self.ledger.delegations.values() for item in record.input}
        needed_inputs.update(
            item.get("id") for row in self.clarifications.values() if row["state"] == "waiting" for item in row["input"]
        )
        self._caller_context = {key: value for key, value in self._caller_context.items() if key in needed_inputs}
        needed_inputs.update(
            identity
            for task in self.ledger.tasks.values()
            for receipt in task.input_receipts.values()
            for identity in receipt.source_message_ids
        )
        for work in self.works.values():
            for task in work.open_tasks():
                needed_inputs.update(
                    item.get("id")
                    for waiting in task.unapplied()
                    if waiting.batch.caller_input
                    for item in waiting.batch.items
                    if is_user_message(item)
                )
                if task.view is None:
                    needed_inputs.update(
                        item.get("id") for item in (*task.initial_requested, *task.pending) if is_user_message(item)
                    )
        self._input_order = {key: value for key, value in self._input_order.items() if key in needed_inputs}
        self.prune_clarifications()

    def prune_clarifications(self) -> None:
        """Bound closed routing records even while an unrelated Work stays open."""
        closed = [key for key, row in self.clarifications.items() if row["state"] != "waiting"]
        for key in closed[:-MAX_KEPT_CLOSED]:
            del self.clarifications[key]

    def request_slot(self, task: TaskRun) -> asyncio.Future:
        """Return a slot for ``task``, granted once fewer than ``max_running`` tasks run, in the order tasks asked."""
        grant = asyncio.get_running_loop().create_future()
        self.ready.append(task)
        task.grant = grant
        self._schedule()
        return grant

    def release(self, task: TaskRun) -> None:
        """Give ``task``'s slot to the next ready task."""
        if (task.stopped == "unconfirmed" and not task.stop_quiescent) or task.inference_unconfirmed:
            return
        held_slot = task in self.running
        self.running.discard(task)
        self._schedule()
        if held_slot:
            self._notify_capacity_change()

    def unschedule(self, task: TaskRun) -> None:
        """Drop the task's reservation, including if it has yet to request a slot."""
        if task in self.ready:
            self.ready.remove(task)
        self._notify_capacity_change()

    def capacity_change(self) -> asyncio.Future[None]:
        """Let pending assignments recheck actual slots and queued reservations on the next change."""
        if self._capacity_change is None:
            self._capacity_change = asyncio.get_running_loop().create_future()
        return self._capacity_change

    def _notify_capacity_change(self) -> None:
        changed, self._capacity_change = self._capacity_change, None
        if changed is not None:
            changed.set_result(None)

    def _schedule(self) -> None:
        if self.closed:
            return
        while self.ready and len(self.running) < self.backend.max_running:
            task = self.ready.popleft()
            if task.grant is not None and not task.grant.done():
                self.running.add(task)
                task.grant.set_result(None)

    def action(self, task: TaskRun, call: dict, state: str | None = None, *, runner: Runner | None = None) -> None:
        """Record a call of ``task`` with who runs it, and move it to ``state`` when given."""
        call_id, name = call.get("call_id", ""), call.get("name", "")
        action = self.ledger.actions.get(call_id)
        if action is None:
            action = self.ledger.new_action(call_id, task.task_id, name, runner or task.runner(name))
        elif action.task != task.task_id or action.work != task.work_id or action.name != name:
            raise ValueError("A tool identity cannot be rebound to another task or tool")
        if runner is not None and action.runner != runner:
            raise ValueError("A tool observation cannot replace a different execution owner")
        if action.arguments is None and isinstance(call.get("arguments"), str):
            action.arguments = call["arguments"]
        if state is not None and not action.ended:
            action.move(state)

    def call_ended(self, call_id: str, output: str | list[dict] | None, outcome: Outcome | Literal["not_run"]) -> None:
        """Record that a call ended with ``outcome``, once."""
        action = self.ledger.actions.get(call_id)
        if action is None:
            return
        late = action.state == "unclear" and outcome != "unclear" and action.late_outcome is None
        if not action.ended:
            action.output = output
            action.move(outcome)
        elif late:
            action.late_output, action.late_outcome = output, outcome
            self.observe("tools.late_evidence", call_id=call_id, task_id=action.task, outcome=outcome)
        else:
            return
        record = self.ledger.tasks.get(action.task)
        if not late and (record is None or not record.ended or outcome == "unclear"):
            return
        task = next(
            (
                task
                for work in [*self.works.values(), *self.ended_works]
                for task in work.tasks
                if task.task_id == action.task
            ),
            None,
        )
        if task is not None:
            task.record_step(call_id, outcome)
            item = {"type": "function_call_output", "call_id": call_id, "output": "" if output is None else output}
            task.outputs[call_id] = item
            for index, old in enumerate(task.thread_items):
                if old.get("type") == "function_call_output" and old.get("call_id") == call_id:
                    task.thread_items[index] = item
                    break
            else:
                if any(
                    old.get("type") == "function_call" and old.get("call_id") == call_id for old in task.thread_items
                ):
                    task.thread_items.append(item)
            if task.ended:
                self.task_ended(task)

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

    def track_app_call_delivery(self, on_window: AppResultWindow) -> AppCallDelivery:
        """Use exact delivery receipts for this call's application result windows."""
        self._app_result_window = on_window
        return AppCallDelivery(self._app_call_delivered, self._app_window_state)

    def _app_window_state(self, response_id: str) -> AppResultState | None:
        record = self.ledger.responses.get(response_id)
        return record.app_result_window if record is not None else None

    def _app_call_delivered(self, call_id: str) -> None:
        parked = self.parked.get(call_id)
        if parked is not None and self.ledger.actions[call_id].app_delivered is False:
            self.ledger.actions[call_id].app_delivered = True
            self._open_app_result_window(parked)

    def park(self, parked: Parked) -> None:
        """Retain the outputs owed by this response's batch of app calls."""
        for call_id in parked.calls:
            self.parked[call_id] = parked
            if self._app_result_window is not None:
                action = self.ledger.actions[call_id]
                action.app_delivered = False
                action.app_response_id = parked.response_id
        self.observe("tools.wait.started", delegation_id=parked.delegation_id, call_ids=list(parked.calls))

    def app_calls_ended(self, parked: Parked) -> None:
        """The source response ended. Open the result window when delivery is also complete."""
        parked.response_ended = True
        self._open_app_result_window(parked)

    def _open_app_result_window(self, parked: Parked) -> None:
        if (
            parked.window_opened
            or not parked.response_ended
            or not parked.calls
            or any(
                self.parked.get(key) is not parked or self.ledger.actions[key].app_delivered is False
                for key in parked.calls
            )
        ):
            return
        parked.window_opened = True
        loop = asyncio.get_running_loop()
        started = loop.time()
        if (limit := self.settings.client_tool_timeout_seconds) is not None:
            parked.timer = loop.call_at(started + limit, parked.work.deadline, parked)
        self._set_app_window(parked, started)

    def _set_app_window(self, parked: Parked, state: AppResultState) -> None:
        if self._app_result_window is not None:
            self.ledger.responses[parked.response_id].app_result_window = state
            self._app_result_window(parked.delegation_id, parked.response_id, state)

    def unpark(self, parked: Parked, *, stop_state: str | None = None) -> None:
        """Stop waiting for these calls' outputs."""
        for call_id in parked.calls:
            self.parked.pop(call_id, None)
        if parked.timer is not None:
            parked.timer.cancel()
        if stop_state is not None and not parked.window_opened:
            unsent = all(self.ledger.actions[key].app_delivered is False for key in parked.calls)
            self._set_app_window(parked, "cancelled" if stop_state == "cancelled" and unsent else "closed")

    def check_retained_app_call(self, call_id: str) -> tuple[RetainedAppCall, Admission | None]:
        """Revalidate a Core publication that the transport confirms it has never queued for delivery."""
        if self.closed or self.detached or not self.attachment_ready:
            return "withhold", None
        if self.caller_turn_pending or self.waits_for_decision:
            return "pending", None
        parked = self.parked.get(call_id)
        action = self.ledger.actions.get(call_id)
        if action is None or action.runner != "app" or not action.app_published:
            return "withhold", None
        if parked is None or parked.work.ended:
            self.call_ended(call_id, self.texts.not_run_output, "not_run")
            return "withhold", self._resume_cancelled_app_frame(action.app_response_id, action.work)
        if action.ended and action.state != "unclear":
            return "withhold", None
        task, call = parked.calls[call_id]
        refusal = ("not_run", self.texts.not_run_output) if action.ended else task.call_refusal(call)
        if refusal is None and task.input_revision != parked.input_revision:
            refusal = "not_run", self.texts.not_run_output
        if refusal is None:
            return "publish", None
        self.call_ended(call_id, refusal[1], "not_run")
        task.remember_refusal(call, *refusal)
        self.parked.pop(call_id)
        del parked.calls[call_id]
        if parked.calls:
            self._open_app_result_window(parked)
            return "withhold", None

        self.unpark(parked)
        work, delegation_id = parked.work, parked.delegation_id
        continuation = self._open_app_continuation(delegation_id)
        task.give([])
        work.resumed()
        self.observe("tools.wait.completed", delegation_id=delegation_id, call_ids=[call_id])
        return "withhold", continuation

    def _resume_cancelled_app_frame(self, response_id: str | None, work_id: str) -> Admission | None:
        """Release an unaffected sibling when a cancelled frame owed no app results."""
        response = self.ledger.responses.get(response_id) if response_id is not None else None
        if response is None or response.app_result_window != "cancelled":
            return None
        delegation_id = response.delegation
        record = self.ledger.delegations[delegation_id]
        if record.state != "waiting_client" or record.responses[-1] != response_id:
            return None
        if not self.ownership.has_unsettled_sibling(delegation_id, work_id):
            return None
        continuation = self._open_app_continuation(delegation_id)
        self.book.start(delegation_id)
        self.ownership.flush(delegation_id)
        return continuation

    def _open_app_continuation(self, delegation_id: str) -> Admission | None:
        record = self.ledger.delegations[delegation_id]
        continuation = None
        if self.book.writer(delegation_id) is None and not record.ended:
            stream = ResponseStream()
            self.book.open(delegation_id, stream, *self.response_settings(record.settings))
            continuation = Admission(delegation_id, stream)
        if record.state == "waiting_client":
            record.move("running")
        return continuation

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
        outputs = self._function_outputs(results)
        expired = {self.ownership.expired[call_id] for call_id in outputs if call_id in self.ownership.expired}
        recovering = [frame for frame in expired if self.ownership.owed_output(frame.delegation_id)]
        if recovering and messages:
            raise ValueError(
                "Send outputs-only for the expired composite response first, then send fresh messages separately"
            )
        parked = {self.parked[call_id] for call_id in outputs if call_id in self.parked}
        if not parked:
            if recovering:
                if len(expired) != 1 or any(call_id not in self.ownership.expired for call_id in outputs):
                    raise ValueError("The results answer calls of more than one response")
                (frame,) = expired
                if set(outputs) != set(frame.calls):
                    raise DelegationError(
                        "Return the output of every function call before response.create",
                        FUNCTION_CALL_OUTPUTS_REQUIRED,
                    )
                if frame not in self.ownership.expired_consumed:
                    self._record_late_app_outputs(outputs)
                    self.ownership.expired_consumed.add(frame)
                    stream = ResponseStream()
                    record = self.ledger.delegations[frame.delegation_id]
                    self.book.open(frame.delegation_id, stream, *self.response_settings(record.settings))
                    if record.state == "waiting_client":
                        record.move("running")
                    self.book.start(frame.delegation_id)
                    self.ownership.flush(frame.delegation_id)
                    return Admission(frame.delegation_id, stream)
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
        # Keep the app's facts in the ledger and isolate each optional worker-facing transformation.
        for step in self.output_steps:
            calls = [(call, outputs[call_id]) for call_id, (_task, call) in waiting.calls.items()]
            tools = settings_for(brings, self.defaults).tools
            try:
                rendered = step.outputs(deepcopy(calls), deepcopy(tools))
                if rendered.keys() != outputs.keys() or self._function_outputs(list(rendered.values())) != rendered:
                    raise ValueError("App output step changed the call identities")
            except Exception as error:
                logger.warning("App output step {} failed ({})", type(step).__name__, type(error).__name__)
                continue
            outputs = rendered
        # Typed messages with the outputs count as a new request.
        if messages:
            self.admitted += 1
        waiting.work.continue_with(waiting, outputs, messages, brings)
        self.observe("tools.wait.completed", delegation_id=waiting.delegation_id, call_ids=list(waiting.calls))
        return Admission(waiting.delegation_id, stream)

    @staticmethod
    def _function_outputs(items: list[dict]) -> dict[str, dict]:
        """Index a batch of app results only if each call has one valid output."""
        outputs: dict[str, dict] = {}
        for item in items:
            call_id = item.get("call_id")
            if item.get("type") != "function_call_output" or not isinstance(call_id, str) or not call_id:
                raise ValueError("A result must be a function_call_output with the call_id of a function call")
            if "output" not in item:
                raise ValueError(f"The result for call {call_id} has no output")
            output = item["output"]
            if not isinstance(output, (str, list)):
                raise ValueError(f"The result for call {call_id} must be text or a list of output content items")
            try:
                validate_output_content(output)
            except ValueError as exc:
                raise ValueError(f"Invalid result for call {call_id}: {exc}") from None
            if call_id in outputs:
                raise ValueError(f"Duplicate result for call {call_id}")
            outputs[call_id] = item
        return outputs

    def _record_late_app_outputs(self, outputs: dict[str, dict]) -> None:
        """Validate the whole batch against original app publications before joining any evidence."""
        for call_id, item in outputs.items():
            action = self.ledger.actions.get(call_id)
            if (
                action is None
                or action.runner != "app"
                or not action.app_published
                or "not_run" in (action.state, action.late_outcome)
                or call_id in self.parked
            ):
                raise DelegationError("Late tool evidence must match a published application call that no longer waits")
            output = item["output"]
            if not isinstance(output, (str, list)):
                raise DelegationError("Late tool evidence requires text or a list of output content items")
            known = (
                action.late_output
                if action.late_outcome is not None
                else action.output
                if action.state in ("done", "failed")
                else None
            )
            if known is not None and known != output:
                raise DelegationError("Late tool evidence conflicts with the operation's known result")
        for call_id, item in outputs.items():
            self.call_ended(call_id, item["output"], "done")

    def reconcile_tool_outputs(self, outputs: list[dict]) -> dict:
        """Accept exact late app evidence without resuming a worker or admitting a caller source."""
        if (
            not isinstance(outputs, list)
            or not outputs
            or any(not isinstance(item, dict) or not isinstance(item.get("output"), str) for item in outputs)
        ):
            raise DelegationError("Late tool evidence must be a nonempty list of text outputs")
        try:
            results = self._function_outputs(outputs)
        except ValueError as error:
            raise DelegationError(str(error)) from None
        self._record_late_app_outputs(results)
        return {"recorded_call_ids": list(results), "resumed": False}

    def _late(
        self, delegation_id: str | None, outputs: dict[str, dict], messages: list[dict], settings: dict | None
    ) -> Admission:
        """Record outputs whose Work has ended; answer with an empty ``completed``, or run the typed messages."""
        self._record_late_app_outputs(outputs)
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
        self._clear_caller_turns()
        self._decided.set()
        try:
            await self._end_call(lambda work: work.record.after_call)
            self.arrivals.clear()
        finally:
            self._app_result_window = None
            self.notes.close()
            self._keep_from_now()
            if self.works:
                self.backend.release_idle()
            else:
                self._release_backend()
        return True

    async def attach(self) -> None:
        """Serve a later call of a session that waits for one; ``DelegationError`` when it has ended or a call holds it.

        It waits for a :meth:`leave` and a backend's close still running. When the backend closed, it builds a new one
        first, whose conductor reads the whole conversation, so a failed build leaves the session waiting; a kept
        conductor keeps what it read. Restore any listener fence left by the previous call before ending retention
        and placing this call's requests.
        """
        while True:
            running = [task for task in (self._leaving, self._backend_closing) if task is not None and not task.done()]
            if not running:
                break
            await asyncio.wait(running)
        if self.closed:
            raise DelegationError("Conversation has ended")
        if not self.detached or self._attaching_call:
            raise DelegationError("Conversation serves another call")
        if self._backend_closing is not None:
            backend = self.renew()
            self.backend, self.conductor, self._cursor = backend, backend.conductor(), 0
            self._backend_closing = None
        if not self.attachment_ready:
            # Activation can await the retained conductor. Only this call may complete the conversation claim.
            self._attaching_call = True
            try:
                if not (await self.set_attachment(True))["attached"]:
                    raise DelegationError("Conversation is not ready for another call")
            except BaseException:
                if not self.closed and not self.works:
                    self._release_backend()
                raise
            finally:
                self._attaching_call = False
        if self._keep is not None:
            self._keep.cancel()
            self._keep = None
        self._call_generation += 1
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
        if self.detached and self._leaving is not None and not self._leaving.done():
            await asyncio.wait({self._leaving})
        self.closed = True
        self._clear_caller_turns()
        self._attachment.set()
        for row in self.clarifications.values():
            if row["state"] == "waiting":
                row["state"] = "cancelled"
        self._decided.set()
        if self._keep is not None:
            self._keep.cancel()
            self._keep = None
        try:
            await self._end_call(lambda work: False)
            if self._background:
                await asyncio.gather(*self._background, return_exceptions=True)
        finally:
            self._app_result_window = None
            self.notes.close()
            if not self.ledger.session.ended:
                self.ledger.session.move("ended")
            try:
                if self._backend_closing is None:
                    await self.backend.close()
                else:
                    await asyncio.wait({self._backend_closing})
            finally:
                for task in tuple(self.running):
                    task.clear_quiescence_observers()
                self.on_closed(self)

    async def _end_call(self, keep: Callable[[WorkRun], bool]) -> None:
        """Stop the call's work as :meth:`close` does; a running Work that ``keep`` names runs on, with its requests."""
        if self.placer is not None:
            self.placer.cancel()
            await asyncio.gather(self.placer, return_exceptions=True)
        self.book.cancel_closing()
        at_once = [
            work
            for work in self.works.values()
            if work.record.state in ("queued", "waiting_client", "pausing", "paused")
        ]
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
            target_works = {target.work for target in record.placements} if record.placements else {record.work}
            if not record.ended and not target_works.intersection(kept):
                record.move("cancelled")
        await asyncio.gather(*(work.confirm_stop(tasks) for work, tasks in asked))
        self.ownership.end_call(kept)
