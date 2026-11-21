# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One task's run on the harness (``port.BackendRun``): a worker session of the agent, read as the IM's rounds."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from examples.frontend_backend_live.delegation_core.backend.port import (
    BoundaryInput,
    Steering,
    StopOutcome,
    TaskContext,
)
from examples.shared.ids import uid

from . import usage
from .bridge import WorkerDoor
from .control import mode_item
from .protocol import (
    AdmissionReady,
    AppCall,
    CallRunning,
    CallSettled,
    CancelDispatch,
    HarnessSession,
    HeldCall,
    Mailbox,
    MessageChunk,
    NativeCompletion,
    NativeSteerReceipt,
    PlanUpdate,
    PromptDispatched,
    PromptEnded,
    RefusedReply,
    Runtime,
    SessionEvent,
    ThoughtChunk,
    ToolCard,
    ToolCardEnd,
    UserText,
    Wake,
    refused_reply,
    usage_total,
)
from .render import output_text
from .runtime import cancel_in_background, give_back
from .sessions import KeptSession, keys, unread

if TYPE_CHECKING:
    from .adapter import HarnessAdapter

OPEN_ATTEMPTS = 2
LOGGED_TEXT_CHARS = 300


class HarnessTurnError(RuntimeError):
    """The agent's turn gave no answer the run can pass on (its text is logged, never spoken)."""


@dataclass
class _Message:
    """One assistant message of the agent: its text so far, and whether it is the agent's own."""

    id: str
    parts: list[str] = field(default_factory=list)
    own: bool = False

    @property
    def text(self) -> str:
        return "".join(self.parts).strip()


@dataclass
class _NativeInput:
    """A taken batch associated with one exact native invocation."""

    steering: Steering
    invocation_id: str
    receipt: asyncio.Future[NativeSteerReceipt]


@dataclass
class _Turn:
    """One turn of the session: from a prompt until every prompt of the session has returned."""

    main: int
    total_before: int | None
    usage_before: Mapping[str, Any] | None = None
    session_id: str | None = None
    started_round: int = 0
    started: float = field(default_factory=time.monotonic)
    # Seconds spent reading the turn: the wait for the app does not count toward its limit.
    used: float = 0.0
    main_open: bool = True
    stop: str | None = None
    usage: Mapping[str, Any] | None = None
    error: str | None = None
    cancelled_by_us: bool = False
    cancelling: asyncio.Task[CancelDispatch] | None = None
    seen: bool = False
    reused: bool = False
    ended: PromptEnded | None = None
    usage_reported: bool = False
    invocation_id: str | None = None
    call_ids: set[str] = field(default_factory=set)
    native_input: _NativeInput | None = None
    uncertain_input: bool = False

    @property
    def cancel_dispatch(self) -> CancelDispatch:
        """Keep unclear transmission distinct from a cancel that never left the client."""
        if not self.cancelled_by_us:
            return "not_dispatched"
        sending = self.cancelling
        if sending is None or not sending.done() or sending.cancelled() or sending.exception() is not None:
            return "unknown"
        return sending.result()

    def reusable_completion(self, completion: NativeCompletion) -> bool:
        """A confirmed terminal allows reuse when cancellation could not reach a later prompt."""
        dispatch = self.cancel_dispatch
        return (dispatch, completion.outcome) in {("not_dispatched", "succeeded"), ("dispatched", "cancelled")}


class _TimedOut:
    """The turn ran past its limit (the run's own event)."""

    def apply(self, reader: WorkerRun) -> None:
        """Hand the event to ``reader``."""
        reader.timed_out()


class WorkerRun:
    """One task's run on the harness (``port.BackendRun``)."""

    def __init__(self, task: TaskContext, host: HarnessAdapter):
        """Run ``task`` on ``host``'s agent; nothing is opened before :meth:`events`."""
        self.task, self.host = task, host
        self.execution_ref = task.execution_ref
        self.sources = task.tool_sources(host.tools)
        self.role = host.workers
        self.settings, self.agent = host.settings, host.settings.agent
        self.round = 0
        self.done = False
        self.answered = False
        self.stopped = False
        self.suspended = False
        self.waits = False
        self.runtime: Runtime | None = None
        self.session: HarnessSession | None = None
        self.door: WorkerDoor | None = None
        self.mailbox: Mailbox = asyncio.Queue()
        self.turn: _Turn | None = None
        self.instructions = ""
        # The keys of every item the session was shown or wrote, in the IM's order.
        self.shown: list[tuple] = []
        self.served: list[str] = []
        # Prompts whose end the run has not read, in mailbox order: the session's own set runs ahead of the mailbox.
        self.open_prompts: set[int] = set()
        self.total: int | None = None
        self.usage: Mapping[str, Any] | None = None
        self.tool_rows = False
        self.next_prompts: list[Steering] = []
        self.prompt_inputs: dict[int, tuple[Steering, ...]] = {}
        self.app_queue: deque[HeldCall] = deque()
        self.forwarded: list[HeldCall] = []
        self.placement_refusals: list[tuple[HeldCall, RefusedReply]] = []
        self.running: set[str] = set()
        self.cards_open: set[str] = set()
        self.runtime_calls: dict[str, dict] = {}
        self.boundary: Callable[[], AsyncIterator[dict]] | None = None
        self.pending: SessionEvent | _TimedOut | None = None
        self.response_id = ""
        self._stopping: asyncio.Future | None = None
        self._input_recovery: tuple[_Turn, WorkerDoor] | None = None
        self._released = asyncio.Event()
        self._reading_finished = asyncio.Event()
        self._reading_finished.set()
        self._suspend_pause: object | None = None
        self._suspend_reusable = False
        self._suspend_outputs: dict[str, str] = {}
        self._borrowed = None
        self._observed_owners: dict[str, TaskContext] = {}
        self._quiet_callbacks: dict[object, Callable[[], None]] = {}
        self._quiet_subscribed = False
        self._history_mode = task.history_mode
        conversation = getattr(task, "conversation_ref", None)
        if not isinstance(conversation, str) or not conversation:
            if task.continues is not None:
                raise ValueError("A harness continuation requires a nonempty conversation_ref")
            conversation = host
        # This reference only isolates ownership names; exact Core continuation is the dependency, never authority
        # to attach a kept native session or expand the task's permission.
        self.retirement = host.retirements.register(
            conversation,
            self.execution_ref,
            task.work_id,
            (task.continues,) if task.continues is not None else (),
        )
        self._new_round()

    async def events(self) -> AsyncIterator[dict]:
        """Yield the task's Responses events round by round; call it again after :meth:`resume`.

        Raises:
            HarnessTurnError: The turn gave no answer the run can pass on, or ran past its limit.
            Exception: The session could not be opened: the agent did not start or did not list its tools.
        """
        self.waits = False
        if self.done or self.stopped or self.suspended or self._released.is_set():
            return
        self._reading_finished.clear()
        try:
            if self.session is None:
                await self._open()
            if self._released.is_set():
                return
            while not (self.done or self.stopped or self.suspended or self.waits or self._released.is_set()):
                async for event in self._round():
                    yield event
        except BaseException:
            if self.suspended or self._borrowed is not None:
                raise
            self.done = not self.stopped
            self._observe_dropped_turn("stop_requested" if self.stopped else "incomplete")
            if not self.stopped:
                # The turn would run on with no reader; a stop cancels its own.
                self._cancel_turn()
            await self.release()
            raise
        finally:
            self._reading_finished.set()
            if self._borrowed is not None:
                self._borrowed.check_ready()
            self._notify_quiescent()
        if self.done and not self.suspended and self._borrowed is None:
            await self.release()

    def on_quiescent(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Notify only after positive whole-owner retirement and its final reader exit; never infer it from cancel."""
        token = object()
        self._quiet_callbacks[token] = callback
        if not self._quiet_subscribed:
            self._quiet_subscribed = True
            self.retirement.ready.add_done_callback(lambda _: self._notify_quiescent())
        self._notify_quiescent()
        return lambda: self._quiet_callbacks.pop(token, None)

    def _notify_quiescent(self) -> None:
        if not (self.retirement.qualified and self._reading_finished.is_set()):
            return
        for token in tuple(self._quiet_callbacks):

            def notify(selected=token):
                callback = self._quiet_callbacks.pop(selected, None)
                if callback is not None:
                    callback()

            asyncio.get_running_loop().call_soon(notify)

    def steer(self) -> None:
        """Wake the reader, which takes the waiting steering at once."""
        if not (self.done or self.stopped or self.suspended):
            if self.door is not None:
                self.door.admission_changed()
            self.mailbox.put_nowait(Wake())

    async def resume(self) -> None:
        """Queue waiting steering, then answer the held app call without claiming native input ingestion."""
        if self.stopped or self._released.is_set():
            raise RuntimeError("A stopped native worker cannot resume")
        if self.suspended:
            if await self.suspend() != "stopped":
                raise RuntimeError("The native worker still owns an unsettled suspended invocation")
            if self.stopped or self._released.is_set():
                raise RuntimeError("A stopped native worker cannot resume")
            await self._resume_suspended()
            return
        self.task.trace("tools.completed", round=self.round)
        if self.door is not None:
            self.door.admission_changed()
        boundary = self._boundary()
        outputs = {item.get("call_id"): item for item in boundary.outputs}
        steering = boundary.steering
        while self._steering_ready(steering) and steering is not None:
            self._deliver(steering)
            steering = self._boundary().steering
        for held in self.forwarded:
            text = output_text(outputs.get(held.call_id))
            text = self.task.texts.owed_output if text is None else text
            held.reply(self.door.qualify_refusal(refused_reply(self.task, held.call_id, fallback=text)))
        self.forwarded = []

    async def suspend(self) -> StopOutcome:
        """Suspend business inference while retaining genuine effects and the same native conversation."""
        if self._borrowed is not None:
            return await self._borrowed.business_suspension()
        return await self._suspend_foreground()

    async def prepare_restricted_control(self) -> None:
        """Settle the old native invocation before interpreting input in a separate restricted model call."""
        if await self.suspend() != "stopped":
            raise RuntimeError("The original native foreground did not confirm suspension")
        for batch in reversed(self.next_prompts):
            self.task.settle(batch, "not_applied")
        self.next_prompts = []

    async def _suspend_foreground(self, *, reading: bool = False) -> StopOutcome:
        """Confirm only this foreground; an already-dispatched Core business operation may still be pending."""
        self.suspended = True
        self._drop_native_input()
        if self.door is not None and self._suspend_pause is None:
            self._suspend_pause = self.door.pause_admissions(self.task.texts.not_run_output)
        self.mailbox.put_nowait(Wake())
        outcome: StopOutcome = "unconfirmed"
        try:
            async with asyncio.timeout(self.settings.stop_timeout_seconds):
                if not reading:
                    await self._reading_finished.wait()
                session, door = self.session, self.door
                if session is None:
                    outcome = "stopped"
                    return outcome
                if door is None:
                    return outcome
                if self._suspend_pause is None:
                    self._suspend_pause = door.pause_admissions(self.task.texts.not_run_output)
                if self._suspend_pause is None:
                    return outcome
                turn = self.turn
                cancel = self._cancel_turn()
                if cancel is not None:
                    await asyncio.shield(cancel)
                if turn is not None:
                    await session.ended(turn.main)
                await session.idle()
                await door.drain_admissions()
                self._drain_recovery_events()
                while self.app_queue:
                    held = self.app_queue[0]
                    if held.answer.done():
                        self.app_queue.popleft()
                    elif self.task.dispatch(held.call) == "refused":
                        self.app_queue.popleft()
                        self._refuse(held)
                    else:
                        return outcome
                for held, reply in self.placement_refusals:
                    held.reply(reply)
                self.placement_refusals = []
                retained = frozenset(held.call_id for held in self.forwarded)
                while pending := door.managed_calls(retained):
                    await asyncio.gather(*(asyncio.shield(call) for call in pending), return_exceptions=True)
                self._drain_recovery_events()
                if (
                    self.stopped
                    or self._released.is_set()
                    or not session.quiescent
                    or self.running
                    or self.runtime_calls
                ):
                    return outcome
                terminal = session.prompt_state(turn.main).terminal if turn is not None else None
                unsent = terminal is not None and terminal.dispatch == "not_dispatched"
                if turn is not None and not unsent and (terminal is None or not terminal.terminal_confirmed):
                    return outcome
                completion = self.agent.completion(terminal) if terminal is not None else None
                self._suspend_reusable = (
                    turn is None or unsent or (completion is not None and turn.reusable_completion(completion))
                )
                if not self._suspend_reusable:
                    return outcome
                self._observe_dropped_turn("suspended")
                self.pending = self.boundary = None
                outcome = "stopped"
                return outcome
        except (TimeoutError, RuntimeError):
            return outcome
        finally:
            self.task.trace("backend.suspended", round=self.round, outcome=outcome)

    async def _resume_suspended(self) -> None:
        """Start a fresh invocation only after real outputs and the old foreground turn settled."""
        batches: list[Steering] = []
        try:
            self._collect(batches)
            door, session, turn = self.door, self.session, self.turn
            pending = [held for held in self.forwarded if not held.answer.done()]
            if any(held.call_id not in self._suspend_outputs for held in pending):
                raise RuntimeError("The original application operation still has no actual result")
            for held in pending:
                held.reply(self._suspend_outputs.pop(held.call_id))
            self.forwarded = [held for held in self.forwarded if not held.answer.done()]
            if door is not None and door.pending_calls:
                async with asyncio.timeout(self.settings.stop_timeout_seconds):
                    await asyncio.gather(*(asyncio.shield(call) for call in tuple(door.pending_calls)))
            if self.stopped or self._released.is_set() or not self.task.can_dispatch():
                return
            self.task.refresh_history(unconfirmed_calls=tuple(turn.call_ids) if turn is not None else ())
            mode = mode_item(self.task)
            extra = [item for batch in batches for item in batch.items]
            history = self.task.history(extra=[*extra, mode])
            seen = set(self.shown)
            # This is the same logical task, not pooled-session adoption. Its narrow control projection can
            # contain private pending-call facts absent from the restored business view. Append new revisions
            # (including real late outcomes) without requiring that projection's last item as an ancestry anchor.
            unseen = [item for item, key in zip(history, keys(history), strict=True) if key not in seen]
            instructions = self._instructions()
            reusable = bool(
                self._suspend_reusable
                and session is not None
                and session.alive
                and session.capabilities.continuation
                and self._history_mode == self.task.history_mode
                and self.instructions == instructions
                and door is not None
                and door.resume_admissions(self._suspend_pause, self.task, self.sources)
            )
            self.suspended = self.done = self.answered = False
            self._suspend_pause = None
            if reusable:
                self.shown += keys([item for item in unseen if item is not mode])
                scheduled = tuple(batches)
                batches.clear()
                self._prompt(self.host.renderer.transcript(unseen), scheduled)
            elif session is None:
                for steering in reversed(batches):
                    self.task.settle(steering, "not_applied")
                batches.clear()
            else:
                self.next_prompts.extend(batches)
                batches.clear()
                await self._reopen("a suspended invocation requires a fresh native session", instructions=instructions)
        finally:
            for steering in reversed(batches):
                self.task.settle(steering, "not_applied")

    async def stop(self) -> StopOutcome:
        """Stop the task: cancel a running prompt, retire the door and the session, and wait for the end."""
        if self._stopping is None:
            self.stopped = True
            self._stopping = asyncio.ensure_future(self._stop())
        return await asyncio.shield(self._stopping)

    async def _stop(self) -> StopOutcome:
        self._drop_native_input()
        if self.door is not None:
            self.door.stop_admissions(self.task.texts.stopped_output)
        self._close_door(self.task.texts.stopped_output)
        self._observe_dropped_turn("stop_requested")
        self.mailbox.put_nowait(Wake())
        outcome: StopOutcome = "unconfirmed"
        session = self.session
        try:
            async with asyncio.timeout(self.settings.stop_timeout_seconds):
                if session is not None:
                    cancel = self._cancel_turn()
                    if cancel is not None:
                        await asyncio.shield(cancel)
                    await session.idle()
                await self.release()
                # Stop can be called by the event consumer while the generator is suspended at its yield.
                # Its exit is still required for the quiescence callback, but cannot own this stop receipt.
                if not self.retirement.local_ready:
                    await asyncio.wait((self.retirement.ready,))
                if self.retirement.local_ready:
                    outcome = "stopped"
        except Exception:
            outcome = "unconfirmed"
        finally:
            self._close_door(self.task.texts.stopped_output)
            await self.release()
        self.task.trace("backend.stopped", round=self.round, outcome=outcome)
        return outcome

    async def release(self) -> None:
        """Pool the session for a later task if it can serve one, else retire it; either way give back its place.

        A session that the pool's bound evicts to make room is retired, and so is one released while the pool is
        closed, or by a task whose Work outlived the call: no later task's history holds that Work's steps.
        """
        self._released.set()
        self.mailbox.put_nowait(Wake())
        self._drop_native_input()
        self._observe_dropped_turn("stop_requested" if self.stopped else "incomplete")
        kept = self._kept()
        if kept is not None:
            self.runtime = self.session = self.door = None
            kept.runtime.leave()
            self.retirement.finish()
            if self.task.after_call:
                self._retire_kept(kept, "a marked task")
                return
            evicted = self.host.keep(kept)
            if evicted is kept:
                self._retire_kept(kept, "the pool closed")
                return
            self.task.trace("harness.session_kept", server=kept.door.name, session=kept.session.id)
            if evicted is not None:
                self._retire_kept(evicted, "evicted")
            return
        self._close_door(self.task.texts.stopped_output)
        if self.door is not None:
            await self.door.drain_admissions()
        waiting, self.next_prompts = self.next_prompts, []
        for steering in reversed(waiting):
            # These batches were never attached to a native invocation.
            self.task.settle(steering, "not_applied")
        self._input_recovery = None
        session = self.session
        if session is not None:
            session.retire()
            for prompt, batches in self.prompt_inputs.items():
                state = session.prompt_state(prompt)
                delivery = "unknown" if state.dispatched else "not_applied"
                for steering in reversed(batches):
                    self.task.settle(steering, delivery)
            self.prompt_inputs.clear()
        runtime, self.runtime = self.runtime, None
        if runtime is not None:
            pending = list(self.door.pending_calls) if self.door is not None else []
            if self.turn is not None and self.turn.cancelling is not None:
                pending.append(self.turn.cancelling)
            self.retirement.finish(
                give_back(runtime, session, pending_calls=pending),
                runtime=runtime,
                finalize=self._drain_retirement_events,
            )
        else:
            self.retirement.finish()

    def _close_door(self, text: str) -> None:
        if self.door is not None:
            try:
                self._answer_forwarded()
            finally:
                self.host.bridge.close(self.door, text)

    def _answer_forwarded(self) -> None:
        """Preserve returned app facts while releasing old held calls without applying caller steering."""
        pending = [held for held in self.forwarded if not held.answer.done()]
        try:
            if not pending:
                return
            boundary = self.task.take_boundary_input()
            if boundary.steering is not None:
                self.task.settle(boundary.steering, "not_applied")
            outputs = {item.get("call_id"): item for item in boundary.outputs}
            for held in pending:
                output = self._suspend_outputs.pop(held.call_id, output_text(outputs.get(held.call_id)))
                reply = refused_reply(
                    self.task, held.call_id, fallback=self.task.texts.owed_output if output is None else output
                )
                # Handed to Core: it may have reached the app. Only a bound refusal proves otherwise.
                if output is None and isinstance(reply, RefusedReply):
                    reply = refused_reply(self.task, held.call_id)
                held.reply(reply)
        finally:
            for held in pending:
                held.reply(self.task.texts.owed_output)

    def _pause_door(self) -> object | None:
        """Fence this old invocation without retiring a native conversation that may safely continue."""
        token = self.door.pause_admissions(self.task.texts.not_run_output)
        if token is None:
            return None
        try:
            self._answer_forwarded()
        finally:
            for held in list(self.door.held):
                held.reply(self.task.texts.not_run_output)
        return token

    def _kept(self) -> KeptSession | None:
        """Return the session for a later task: only once its task answered and nothing of the task is left in it."""
        session, door, runtime = self.session, self.door, self.runtime
        if (
            self._borrowed is not None
            or session is None
            or door is None
            or door.replay_exhausted
            or runtime is None
            or not self.answered
            or self.stopped
            or not self.task.can_dispatch()
        ):
            return None
        if self.turn is not None and self.turn.cancel_dispatch != "not_dispatched":
            return None
        left = (session.outstanding, self.next_prompts, self.prompt_inputs, self.forwarded, self.app_queue)
        lost = not (runtime.client.alive and session.alive and session.quiescent and door.admission_open)
        if (
            any(left)
            or self.running
            or self.cards_open
            or door.held
            or door.admitting
            or door.app_admissions
            or door.pending_calls
            or lost
        ):
            return None
        served = [*self.served, self.task.work_id, self.task.task_id]
        return KeptSession(
            runtime,
            session,
            door,
            self.mailbox,
            self.instructions,
            list(self.shown),
            self.total,
            self.usage,
            self.tool_rows,
            served,
            self.task.task_id,
            time.monotonic(),
            history_mode=self.task.history_mode,
        )

    async def _open(self) -> None:
        """Take the first idle session of its thread that fits (:meth:`_adopt`), else open a fresh one, and prompt it.

        The history is read before the walk, so one over the budget fails the task and leaves every idle session as it
        was. The walk never awaits, so two tasks never take the same session, and its checks (:meth:`_misfit`) retire
        only a lost session.
        """
        await self._wait_for_predecessors()
        if self._released.is_set():
            return
        self.task.refresh_history()
        history = self.task.history()
        instructions = self._instructions()
        why = []
        for kept in () if self.runtime is not None else self.host.candidates(self.task):
            misfit = self._misfit(kept, instructions, history)
            if misfit is None:
                misfit = self._adopt(kept, instructions)
                if misfit is None:
                    return
            why.append(misfit)
        if self.runtime is None:
            self.runtime = await self.host.shared(self.role)
        await self._open_session(instructions, why or ["no idle session of its thread"])

    async def _wait_for_predecessors(self) -> None:
        """Let this run stop without cancelling the exact older owners whose settlement it awaits."""
        if not self.retirement.ancestors:
            return
        waiting = asyncio.create_task(self.retirement.wait())
        released = asyncio.create_task(self._released.wait())
        try:
            await asyncio.wait((waiting, released), return_when=asyncio.FIRST_COMPLETED)
            if not self._released.is_set():
                await waiting
        finally:
            for task in (waiting, released):
                if not task.done():
                    task.cancel()
            await asyncio.gather(waiting, released, return_exceptions=True)

    def _instructions(self) -> str:
        """Compose the worker's stable native policy separately from untrusted tool results and task input."""
        instructions = self.task.request().get("instructions") or ""
        if not self.settings.steering.cooperative_end:
            return instructions
        note = self.settings.prompts.worker_protocol
        return "\n\n".join(part for part in (instructions, note) if part)

    def _misfit(self, kept: KeptSession, instructions: str, history: list[dict]) -> str | None:
        """Return why the idle session ``kept`` cannot serve the task; else ``None``.

        A session whose process or session is lost is retired on the way; any other stays idle as it was.
        """
        if not (kept.runtime.client.alive and kept.session.alive):
            why = "session lost" if kept.runtime.client.alive else "process lost"
            self.host.take_kept(kept)
            self._retire_kept(kept, why)
            return why
        if kept.door.replay_exhausted:
            why = "native request receipt capacity exhausted"
            self.host.take_kept(kept)
            self._retire_kept(kept, why)
            return why
        if not kept.session.quiescent:
            return "native invocation remains unconfirmed"
        if kept.instructions != instructions:
            return "other instructions"
        if kept.history_mode != self.task.history_mode:
            return "other history mode"
        if not kept.door.serves(self.task, self.sources):
            return "another toolset"
        return unread(kept.shown, history).why

    def _adopt(self, kept: KeptSession, instructions: str) -> str | None:
        """Take over ``kept``, which fits: a turn place, waiting steering, its door, then the items it has not read.

        Return ``None`` once it is prompted. A full process, or a history that the steering left it unfit for, returns
        why, and a failure is raised; either way the session stays idle as it was, its place given back.
        """
        if not self.host.take_place(kept.runtime):
            return "process full"
        batches: list[Steering] = []
        adopted = False
        try:
            self._collect(batches)
            history = self.task.history(extra=[item for batch in batches for item in batch.items])
            missed = unread(kept.shown, history)
            if missed.items is None or not kept.door.rebind(self.task, self.sources):
                return missed.why or "another toolset"
            self.host.take_kept(kept)
            self.runtime, self.session, self.door, self.mailbox = kept.runtime, kept.session, kept.door, kept.mailbox
            adopted = True
            self.door.can_cooperatively_end = self._can_cooperatively_end
            self.door.owned_input_refused = self._request_input_recovery
            while not self.mailbox.empty():
                self.mailbox.get_nowait()
            self.instructions, self.served = instructions, list(kept.served)
            self.shown = [*kept.shown, *keys(missed.items)]
            self.total, self.usage, self.tool_rows = kept.total, kept.usage, kept.tool_rows
            self.task.trace(
                "harness.session_reused",
                server=kept.door.name,
                session=kept.session.id,
                items=len(missed.items),
                continues=self.task.continues is not None and self.task.continues in kept.served,
                idle=round(time.monotonic() - kept.idle_since, 3),
                before=missed.before,
                absent=missed.absent,
                last_task=kept.last_task,
            )
            text = self.host.renderer.transcript(missed.items)
            scheduled = tuple(batches)
            batches.clear()
            self._prompt(text, scheduled)
            self.turn.reused = True
            return None
        finally:
            for batch in reversed(batches):
                self.task.settle(batch, "not_applied")
            if not adopted:
                kept.runtime.leave()

    def _retire_kept(self, kept: KeptSession, why: str) -> None:
        """Retire ``kept``, an idle session out of the pool that holds no place, and say why."""
        closed = kept.retire(self.host.bridge)
        self.task.trace(
            "harness.session_retired",
            server=kept.door.name,
            session=kept.session.id,
            last_task=kept.last_task,
            why=why,
            closed=closed,
        )

    async def _open_session(self, instructions: str, why: list[str], batches: tuple[Steering, ...] = ()) -> None:
        """Keep taken input owned until scheduling, returning every certainly-unsent batch on an early exit."""
        waiting = list(batches)
        try:
            await self._prepare_session(instructions, why, waiting)
        finally:
            for steering in reversed(waiting):
                self.task.settle(steering, "not_applied")

    async def _prepare_session(
        self, instructions: str, why: list[str], batches: list[Steering], *, prompt: bool = True
    ) -> tuple[str, ...] | None:
        """Open a fresh session in the run's runtime and prompt it with the history, ``batches`` applied.

        ``why`` says why the task opens a fresh session: one reason per idle session it tried, or why it left its own.
        """
        runtime, started = self.runtime, time.monotonic()
        for attempt in range(1, OPEN_ATTEMPTS + 1):
            name = runtime.server_name(self.role.prefix)
            mailbox: Mailbox = asyncio.Queue()
            door = self.host.own_door(
                WorkerDoor(
                    name,
                    self.task,
                    self.sources,
                    mailbox,
                    tool_timeout_seconds=self.settings.tool_timeout_seconds,
                    can_cooperatively_end=self._can_cooperatively_end,
                    owned_input_refused=self._request_input_recovery,
                    invocation_active=lambda: False,
                ),
            )
            server, session = self.host.bridge.open(door), None
            try:
                folder = runtime.session_folder(f"{self.execution_ref}-{name}")
                blocks = self.agent.prepare(folder, instructions)
                session = await runtime.client.new_session(folder, server, mailbox, role="worker")
                listed = await self._listed(door)
            except BaseException:
                if session is not None:
                    session.retire()
                self.host.bridge.close(door, self.task.texts.stopped_output)
                raise
            if listed:
                break
            session.retire()
            self.host.bridge.close(door, self.task.texts.stopped_output)
            logger.warning(f"harness.session_not_ready task={self.task.task_id} server={name} attempt={attempt}")
        else:
            raise RuntimeError("The agent did not list the session's tools: its MCP connection to the bridge failed")
        self.mailbox, self.door, self.session = mailbox, door, session
        self.total, self.usage = 0, dict.fromkeys(usage.COUNTERS, 0)
        self.instructions, self.served = instructions, []
        seconds = round(time.monotonic() - started, 3)
        self.task.trace("harness.session_opened", server=name, session=session.id, seconds=seconds, why=why)
        if self.stopped or self._released.is_set():
            await self.release()
            return
        if not prompt:
            return blocks
        if self.suspended:
            return
        self._collect(batches)
        history = self.task.history(extra=[item for batch in batches for item in batch.items])
        self.shown = keys(history)
        text = "\n\n".join([*blocks, self.host.renderer.worker_input(history)])
        scheduled = tuple(batches)
        batches.clear()  # _prompt owns both successful scheduling and rejection before dispatch.
        self._prompt(text, scheduled)

    async def _open_for_control(self) -> tuple[str, ...]:
        """Open only a never-opened logical worker, without starting a business prompt or another control runtime."""
        await self._wait_for_predecessors()
        if self.stopped or self._released.is_set():
            raise RuntimeError("The native logical worker is no longer available")
        if self.session is not None:
            return ()
        if self.runtime is None:
            self.runtime = await self.host.shared(self.role)
        blocks = await self._prepare_session(self._instructions(), ["first logical worker control"], [], prompt=False)
        if self.session is None or self.door is None:
            raise RuntimeError("The first native worker session did not open")
        return blocks or ()

    def _prompt(self, text: str, batches: tuple[Steering, ...] = ()) -> None:
        """Bind immutable input batches to one scheduled native invocation."""
        try:
            prompt = self.session.prompt(text, may_dispatch=self._can_dispatch)
        except BaseException:
            for batch in reversed(batches):
                self.task.settle(batch, "not_applied")
            raise
        self.prompt_inputs[prompt] = batches
        self._start_turn(prompt)

    def _can_dispatch(self) -> bool:
        return bool(
            not (self.stopped or self.suspended or self.done)
            and self.task.can_dispatch()
            and self.door is not None
            and self.door.admission_open
            and self.door.task is self.task
        )

    async def _listed(self, door: WorkerDoor) -> bool:
        try:
            async with asyncio.timeout(self.settings.ready_timeout_seconds):
                await door.listed.wait()
            return True
        except TimeoutError:
            return False

    def _start_turn(self, prompt: int) -> None:
        self._input_recovery = None
        self.open_prompts.add(prompt)
        session = self.session
        try:
            session_id = session.id
        except Exception:
            session_id = None
        self.turn = _Turn(prompt, self.total, self.usage, session_id=session_id, started_round=self.round)
        turn = self.turn
        self.door.invocation = self.turn
        self.door.invocation_active = lambda: session.prompt_state(prompt).phase == "dispatched"
        self.door.native_admission = None
        if (steering := session.native_steering) is not None:

            def admit_native_tool(name: str, arguments: dict):
                if self.turn is not turn or not self._can_dispatch():
                    return None
                verified = steering.admit_tool(name, arguments)
                if verified is not None and (pending := turn.native_input) is not None:
                    self._settle_native_input(turn, pending)
                return verified

            self.door.native_admission = admit_native_tool
        self.door.admission_changed()
        self.task.trace("harness.prompt_sent", round=self.round, prompt=prompt)
        if self._borrowed is not None:
            self._borrowed.prompt_started()

    def _new_round(self) -> None:
        self.messages: dict[str, _Message] = {}
        self.calls: list[dict] = []
        self.activity = False
        self.activity_since_message = False

    async def _round(self) -> AsyncIterator[dict]:
        """Yield one round: ``response.created``, the session's events up to a boundary, then the boundary's ending.

        A round ends when a model message begins after tool activity, when an app call is forwarded, when steering needs
        a safe point, when every prompt has returned, or on a stop or a timeout.
        """
        self.round += 1
        self._new_round()
        self.response_id = uid("resp")
        yield {"type": "response.created", "response": {"id": self.response_id, "status": "in_progress", "output": []}}
        if self.stopped or self.suspended or self._released.is_set():
            return
        self._ready_app()
        while self.boundary is None:
            if self.stopped or self.suspended or self._released.is_set():
                return
            if self._ready_to_reprompt():
                self._schedule_input_recovery(self.door)
                self.boundary = self._finish_input_recovery
                break
            event, self.pending = self.pending, None
            if event is None:
                event = await self._next()
            self._apply_event(event)
        finish, self.boundary = self.boundary, None
        async for event in finish():
            yield event

    async def _next(self) -> SessionEvent | _TimedOut:
        """Return the session's next event, or a :class:`_TimedOut` once the turn's reading time is up.

        A task whose Work outlives the call reads with no time limit: the keep time and the stop door bound it.
        """
        turn = self.turn
        if turn is None or self.task.after_call:
            return await self.mailbox.get()
        started = time.monotonic()
        door = self.door
        paused_before = door.placement_elapsed() if door is not None else 0.0
        try:
            # GLOBAL placement already paused reading time at Core's refusal boundary. Keep that budget while
            # holding before first admission, without pausing mailbox reads or the native invocation's wall timer.
            limit = (
                None
                if door is not None and door.placement_waiting
                else max(self.settings.turn_timeout_seconds - turn.used, 0.001)
            )
            async with asyncio.timeout(limit):
                return await self.mailbox.get()
        except TimeoutError:
            if self.task.after_call:
                return await self.mailbox.get()
            return _TimedOut()
        finally:
            paused = (door.placement_elapsed() - paused_before) if door is not None else 0.0
            turn.used += max(0.0, time.monotonic() - started - paused)

    def _has_items(self) -> bool:
        return bool(self.calls) or any(message.text for message in self.messages.values())

    async def _close(self, final: _Message | None, usage: Mapping[str, Any] | None = None) -> AsyncIterator[dict]:
        """End the round: messages (``final`` is the answer), then calls; the agent's own text is only logged."""
        items = []
        for message in self.messages.values():
            # Never commentary either, which may be spoken as progress.
            if message.own:
                logger.warning(
                    f"harness.text_dropped task={self.task.task_id} text={message.text[:LOGGED_TEXT_CHARS]!r}"
                )
                self.task.trace("harness.text_dropped", round=self.round)
            elif message.text and not self._input_prevents_output():
                phase = "final_answer" if message is final else "commentary"
                items.append(
                    {
                        "id": uid("msg"),
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "phase": phase,
                        "content": [{"type": "output_text", "text": message.text, "annotations": []}],
                    }
                )
        items += [dict(call) for call in self.calls]
        self.shown += keys(items)
        for index, item in enumerate(items):
            opening = {**item, "status": "in_progress"}
            if item["type"] == "message":
                opening["content"] = []
            yield {"type": "response.output_item.added", "output_index": index, "item": opening}
            yield {"type": "response.output_item.done", "output_index": index, "item": item}
        response = {"id": self.response_id, "status": "completed", "output": items}
        if usage:
            response["usage"] = usage
        yield {"type": "response.completed", "response": response}
        self._new_round()

    def _input_prevents_output(self) -> bool:
        turn = self.turn
        return bool(
            self.task.pending_input_ready()
            or self.next_prompts
            or self.prompt_inputs
            or (turn is not None and (turn.native_input is not None or turn.uncertain_input))
        )

    async def _fail(self, reason: str, text: str = "", usage: Mapping[str, Any] | None = None) -> AsyncIterator[dict]:
        """End the round ``response.failed`` and raise :class:`HarnessTurnError`; the agent's text is only logged."""
        logger.warning(
            f"harness.turn_refused task={self.task.task_id} reason={reason!r} text={text[:LOGGED_TEXT_CHARS]!r}"
        )
        self.task.trace("harness.turn_refused", round=self.round, reason=reason)
        error = {"code": "server_error", "message": reason}
        response = {"id": self.response_id, "status": "failed", "output": [], "error": error}
        if usage is not None:
            response["usage"] = usage
        yield {"type": "response.failed", "response": response}
        self.done = True
        raise HarnessTurnError(reason)

    def _steering_ready(self, current: Steering | None = None) -> bool:
        """Return ineligible taken input for Core follow-up, preserving its original order."""
        if self.task.pending_input_ready():
            return True
        waiting, self.next_prompts = self.next_prompts, []
        if current is not None:
            waiting.append(current)
        # not_applied puts each batch back at the front of the task's queue.
        for steering in reversed(waiting):
            self.task.settle(steering, "not_applied")
        return False

    def _between_rounds(self) -> None:
        """Join owed outputs and hand eligible waiting batches to the steering strategy."""
        if self.suspended:
            return
        steering = self._boundary().steering
        while self._steering_ready(steering) and steering is not None:
            self._deliver(steering)
            steering = self._boundary().steering

    def _deliver(self, steering: Steering) -> None:
        """Hand ``steering`` to the agent class's strategy (``steering.py``); a batch with no text applies at once."""
        if text := self.host.renderer.steering(steering):
            if self.next_prompts or not self._submit_native_input(steering, text):
                self.settings.steering.deliver(self, steering)
            return
        self._applied(steering)
        self.task.trace("harness.steer_applied", round=self.round, steering=steering.id, how="no_text")

    def _submit_native_input(self, steering: Steering, text: str) -> bool:
        """Use verified native steering only for the current dispatched foreground."""
        session, turn = self.session, self.turn
        if session is None or (channel := session.native_steering) is None or turn is None:
            return False
        if (
            self._borrowed is not None
            or not turn.main_open
            or turn.cancelled_by_us
            or turn.native_input is not None
            or turn.uncertain_input
            or not self._can_dispatch()
        ):
            return False
        state = session.prompt_state(turn.main)
        if state.phase != "dispatched":
            return False
        try:
            receipt = channel.submit(turn.main, state.invocation_id, steering.id, text)
        except Exception as error:
            self.task.settle(steering, "unknown")
            turn.uncertain_input = True
            logger.warning(f"harness.steer_submit_failed error_type={type(error).__name__}")
            return True
        if receipt is None:
            return False
        pending = _NativeInput(steering, state.invocation_id, receipt)
        turn.native_input = pending
        self.messages.clear()
        receipt.add_done_callback(lambda _: self._settle_native_input(turn, pending))
        return True

    def _settle_native_input(self, turn: _Turn, pending: _NativeInput) -> None:
        """Apply only model-backed evidence for the exact input and original invocation."""
        if self.turn is not turn or turn.native_input is not pending:
            return
        future = pending.receipt
        if not future.done():
            return
        turn.native_input = None
        receipt = future.result() if not future.cancelled() and future.exception() is None else None
        steering = pending.steering
        if isinstance(receipt, NativeSteerReceipt) and (receipt.invocation_id, receipt.input_id) == (
            pending.invocation_id,
            steering.id,
        ):
            if receipt.delivery == "not_applied":
                self.next_prompts.insert(0, steering)
                self.mailbox.put_nowait(Wake())
                return
            if receipt.delivery == "applied" and receipt.request_id:
                self._applied(steering)
                self.task.trace(
                    "harness.steer_applied",
                    round=self.round,
                    steering=steering.id,
                    invocation_id=pending.invocation_id,
                    model_consumed=True,
                )
                if self._pending_delivery():
                    waiting = self.next_prompts.pop(0)
                    if not self._submit_native_input(waiting, self.host.renderer.steering(waiting)):
                        self.next_prompts.insert(0, waiting)
                        self.mailbox.put_nowait(Wake())
                return
        self.task.settle(steering, "unknown")
        turn.uncertain_input = True

    def _drop_native_input(self) -> None:
        """Release an observer without retrying a correction that may have reached the agent."""
        turn = self.turn
        if turn is not None and (pending := turn.native_input) is not None:
            pending.receipt.cancel()
            self._settle_native_input(turn, pending)

    async def _finish_next_call(self) -> AsyncIterator[dict]:
        async for event in self._close(None):
            yield event
        self._between_rounds()

    async def _finish_woken(self) -> AsyncIterator[dict]:
        async for event in self._close(None):
            yield event
        self._between_rounds()

    async def _finish_placement_refusal(self) -> AsyncIterator[dict]:
        """Let Core settle global placement before the native call gets its original non-owned refusal."""
        async for event in self._close(None):
            yield event
        self._between_rounds()
        waiting, self.placement_refusals = self.placement_refusals, []
        if not self.stopped:
            for held, reply in waiting:
                held.reply(reply)

    async def _finish_app(self) -> AsyncIterator[dict]:
        async for event in self._close(None):
            yield event
        self.waits = True
        self.task.trace("tools.waiting", round=self.round, call_ids=[held.call_id for held in self.forwarded])

    async def _finish_stopped(self) -> AsyncIterator[dict]:
        return
        yield

    async def _finish_timeout(self) -> AsyncIterator[dict]:
        """Fail the task whose turn ran past ``turn_timeout_seconds``; :meth:`events` cancels the turn."""
        async for event in self._fail(f"the turn ran past {self.settings.turn_timeout_seconds:g} s"):
            yield event

    async def _finish_turn(self) -> AsyncIterator[dict]:
        """End the turn with the answer, the next prompt, a fresh session, or a failure.

        The answer is the last message after the turn's last tool activity, when the turn ended ``end_turn`` and
        the agent reports a successful native completion. Eligible input first goes to the same native
        session; only the refreshed answer ends the task. A turn without an answer continues with input or fails
        the task.
        """
        turn = self.turn
        if self.stopped or self.suspended:
            return
        if (pending := turn.native_input) is not None:
            await asyncio.wait((pending.receipt,), timeout=self.settings.stop_timeout_seconds)
            self._drop_native_input()
        if self.stopped or self.suspended or self._released.is_set():
            return
        turn_usage = self._turn_usage(turn)
        if turn.uncertain_input:
            async for event in self._fail("native steering delivery is unknown", usage=turn_usage):
                yield event
        candidate = self._candidate()
        failure = self._turn_failure(turn)
        reason = failure or self._unparsed(turn, candidate)
        if reason is None:
            # The native invocation is idle. Eligible queued input must run before this task's final answer;
            # typed input accompanying tool outputs has no delegation to re-place after the Work ends.
            self._between_rounds()
            if self.next_prompts:
                async for event in self._close(None, turn_usage):
                    yield event
                await self._prompt_next_now()
                return
            self.task.trace("harness.answered", round=self.round, seconds=round(time.monotonic() - turn.started, 3))
            async for event in self._close(candidate, turn_usage):
                yield event
            self.done = self.answered = True
            return
        if (
            failure is not None
            and turn.reused
            and not turn.seen
            and turn.ended is not None
            and turn.ended.dispatch == "not_dispatched"
        ):
            async for event in self._close(None, turn_usage):
                yield event
            await self._renew()
            return
        if failure is not None:
            async for event in self._fail(failure, self._last_text(), turn_usage):
                yield event
        if not self.calls:
            # Safe: the open round holds no call whose output could join before it.
            self._between_rounds()
        else:
            self._steering_ready()
        waiting = self.next_prompts
        if waiting:
            async for event in self._close(None, turn_usage):
                yield event
            await self._prompt_next_now()
            return
        async for event in self._fail(reason, self._last_text(), turn_usage):
            yield event

    def _turn_failure(self, turn: _Turn) -> str | None:
        completion = self.agent.completion(turn.ended or PromptEnded(turn.main, turn.stop, turn.usage, turn.error))
        return None if completion.outcome == "succeeded" else completion.error or f"native turn {completion.outcome}"

    def _candidate(self) -> _Message | None:
        if self.activity_since_message or not self.messages:
            return None
        return list(self.messages.values())[-1]

    def _unparsed(self, turn: _Turn, candidate: _Message | None) -> str | None:
        """Return why the turn's last message is no answer the run may pass on, or ``None``."""
        if candidate is None or not candidate.text:
            return "the turn ended without a message after its last tool call"
        return None

    def _last_text(self) -> str:
        texts = [message.text for message in self.messages.values() if message.text]
        return texts[-1] if texts else ""

    def _turn_usage(self, turn: _Turn) -> dict | None:
        """Return only counters actually known for this invocation or cumulative interval."""
        invocation = turn.ended.invocation_usage if turn.ended is not None else None
        counts = usage.round_counters(turn.usage, turn.usage_before, invocation, agent=self.agent)
        self._report_turn_usage(turn, counts, ending="error" if turn.error is not None else turn.stop)
        return usage.responses_usage(counts)

    def _observe_dropped_turn(self, ending: str) -> None:
        """Observe only a prompt actually sent; without its terminal, no earlier or partial counters are reused."""
        turn = self.turn
        if turn is None or turn.usage_reported:
            return
        if turn.ended is not None:
            self._turn_usage(turn)
        else:
            self._report_turn_usage(turn, None, ending=ending)

    def _report_turn_usage(self, turn: _Turn, counts: Mapping[str, Any] | None, *, ending: str | None) -> None:
        """Report a native prompt once; telemetry failures cannot prevent its physical cleanup."""
        if turn.usage_reported:
            return
        turn.usage_reported = True
        metadata = {
            "backend": "harness",
            "protocol": self.settings.protocol_name,
            "model": None,
            "model_source": "unknown",
        }
        with suppress(Exception):
            metadata.update(self.host.usage_identity(self.task))
        invocation = turn.ended.invocation_usage if turn.ended is not None else None
        try:
            fields = {
                **metadata,
                "role": "worker",
                "conversation_ref": self.task.conversation_ref,
                "round": self.round,
                "prompt": turn.main,
                "invocation_id": turn.invocation_id,
                "session": turn.session_id,
                "prompt_started_round": turn.started_round,
                "reported_counters": counts or None,
                "usage_missing": not counts,
                "usage_scope": "invocation" if invocation is not None else "cumulative_delta",
                "native_terminal_observed": turn.ended is not None,
                "usage_authority": "worker_round_mirror",
                "ending": ending,
                **usage.observation(self.agent, turn.usage, invocation),
            }
            self.task.trace(
                "harness.round.usage",
                **fields,
            )
        except Exception:
            logger.warning("harness.usage_observation_failed task={} prompt={}", self.task.task_id, turn.main)

    def _collect(self, batches: list[Steering]) -> list[Steering]:
        """Let the owed outputs join at a safe point, then add every waiting batch to ``batches`` after them."""
        steering = self._boundary().steering
        while steering is not None:
            batches.append(steering)
            steering = self._boundary().steering
        return batches

    def _boundary(self) -> BoundaryInput:
        """Take the boundary input at a safe point; the outputs that join are items the session has read."""
        boundary = self.task.take_boundary_input()
        if self.suspended:
            retained = {held.call_id for held in self.forwarded}
            for item in boundary.outputs:
                if item.get("call_id") in retained:
                    text = output_text(item)
                    if text is not None:
                        self._suspend_outputs[item["call_id"]] = text
            # The cancelled native prompt may never have read these real results. Keep them in the
            # task's history so the next safe prompt can include them, even on the same session.
        else:
            self.shown += keys(boundary.outputs)
        return boundary

    def _applied(self, steering: Steering) -> None:
        self.task.settle(steering, "applied")
        self.shown += keys(steering.items)

    async def _prompt_next_now(self) -> None:
        """Send every batch the turn did not take as the next prompt, oldest first; each is ``applied`` once sent."""
        if self.stopped or self.suspended or self.done:
            return
        if self.door.replay_exhausted:
            self.task.refresh_history()
            await self._reopen("native request receipt capacity exhausted")
            return
        batches, self.next_prompts = self.next_prompts, []
        try:
            self._collect(batches)
            extra = [item for batch in batches for item in batch.items]
            self.task.history(extra=extra)
            items = [*extra, *self.task.backend_state(extra)]
            text = self.host.renderer.transcript(items)
            self.task.trace("harness.steer_prompted", round=self.round, batches=len(batches))
            scheduled = tuple(batches)
            batches.clear()
            self._prompt(text, scheduled)
        finally:
            for batch in reversed(batches):
                self.task.settle(batch, "not_applied")

    async def _renew(self) -> None:
        """Retire a session taken from the pool whose first prompt failed before any update; go on in a fresh one.

        Nothing of the task reached it, so the fresh session takes the whole history with the batches left waiting.
        """
        session = self.session
        self.session = None
        closed = session.retire()
        why = "a failed first prompt"
        self.task.trace("harness.session_retired", server=self.door.name, session=session.id, why=why, closed=closed)
        await self._reopen(why)

    async def _reopen(self, why: str, *, instructions: str | None = None) -> None:
        """Open a fresh session in place of the run's, prompted with the history and the batches, saying ``why``.

        The previous invocation has settled or did not dispatch.
        """
        self._drop_native_input()
        batches, self.next_prompts = self.next_prompts, []
        self._close_door(self.task.texts.not_run_output)
        if self.session is not None:
            self.session.retire()
        self.next_prompts, self.prompt_inputs, self.app_queue, self.forwarded = [], {}, deque(), []
        self.open_prompts = set()
        self.running, self.cards_open, self.runtime_calls = set(), set(), {}
        self.pending = self.boundary = None
        self.tool_rows = False
        await self._open_session(self._instructions() if instructions is None else instructions, [why], tuple(batches))

    def _seen(self) -> None:
        turn = self.turn
        if turn is None or turn.seen:
            return
        turn.seen = True

    def prompt_next(self, steering: Steering) -> None:
        """Queue the batch as the next prompt, sent once the turn ends."""
        self.next_prompts.append(steering)

    def _cancel_turn(self) -> asyncio.Task[CancelDispatch] | None:
        """Cancel the running turn once, in the background, within ``stop_timeout_seconds``."""
        turn, session = self.turn, self.session
        if turn is None:
            return None
        if turn.cancelling is not None:
            return turn.cancelling
        if turn.cancelled_by_us or session is None or not session.outstanding:
            return None
        turn.cancelled_by_us = True
        turn.cancelling = cancel_in_background(session, self.settings.stop_timeout_seconds)
        return turn.cancelling

    def _request_input_recovery(self, door: WorkerDoor, reply: RefusedReply) -> None:
        """Request recovery only for this invocation's trusted owned-input refusal; never wait in its MCP call."""
        if not self._can_recover_owned_input(reply):
            return
        if self._schedule_input_recovery(door):
            self.mailbox.put_nowait(Wake())

    def _schedule_input_recovery(self, door: WorkerDoor) -> bool:
        turn = self.turn
        if (
            turn is None
            or door is not self.door
            or turn.cancelled_by_us
            or turn.native_input is not None
            or turn.uncertain_input
            or self._input_recovery is not None
        ):
            return False
        self._input_recovery = (turn, door)
        self.task.trace("harness.input_recovery_requested", prompt=turn.main, invocation_id=turn.invocation_id)
        return True

    def _ready_to_reprompt(self) -> bool:
        """For cancel-and-reprompt, wait for active calls before interrupting the old native invocation."""
        session, turn, door = self.session, self.turn, self.door
        return bool(
            session is not None
            and session.capabilities.delivery == "cancel_reprompt"
            and door is not None
            and turn is not None
            and not turn.cancelled_by_us
            and turn.native_input is None
            and not turn.uncertain_input
            and self._input_recovery is None
            and self._pending_delivery()
            and self.task.can_dispatch()
            and not (self.running or self.cards_open or self.forwarded or door.calls_pending_for(turn))
            and session.prompt_state(turn.main).dispatched
        )

    def _drain_recovery_events(self) -> None:
        """Retain old invocation events and real tool outcomes before replacing their mailbox."""
        while self.pending is not None or not self.mailbox.empty():
            event, self.pending = self.pending, None
            if event is None:
                event = self.mailbox.get_nowait()
            self._apply_event(event)
            if self.pending is event:
                # Text deferred behind this old round's calls is stale. ToolCard.ended, in contrast,
                # is a distinct event and must still settle the actual tool fact before SID reuse.
                self.pending = None
        self.boundary = None

    def _drain_retirement_events(self) -> None:
        """Join late native tool facts before releasing a continuation; publish no obsolete model output."""
        while self.pending is not None or not self.mailbox.empty():
            event, self.pending = self.pending, None
            if event is None:
                event = self.mailbox.get_nowait()
            if isinstance(event, (ToolCard, ToolCardEnd, CallRunning, CallSettled, PromptEnded)):
                self._apply_event(event)
        self.pending = self.boundary = None
        # A released reader may still be inside Queue.get; draining its old terminal must not consume its stop wake.
        self.mailbox.put_nowait(Wake())

    async def _finish_input_recovery(self) -> AsyncIterator[dict]:
        """End a fenced native invocation within the existing stop bound, then refresh the same logical task."""
        recovery = self._input_recovery
        if recovery is None:
            return
        turn, door = recovery
        session, runtime = self.session, self.runtime
        if (
            turn is not self.turn
            or door is not self.door
            or session is None
            or not self._pending_delivery()
            or not self.task.can_dispatch()
        ):
            self._input_recovery = None
            return
        # A cooperative end already settled before this boundary needs no cancellation or replacement.
        terminal = session.prompt_state(turn.main).terminal
        if terminal is not None and terminal.terminal_confirmed and session.quiescent:
            self._drain_recovery_events()
            if not (door.pending_calls or self.running or self.runtime_calls or self.forwarded or self.cards_open):
                self._input_recovery = None
                async for event in self._finish_turn():
                    yield event
                return

        # Pause first admission while preserving the native conversation. Permanent stop/retirement owns its own
        # irreversible fence and invalidates this token, so its completion cannot revive a stopped task.
        pause = self._pause_door()
        failure = None
        cancel = None
        try:
            async with asyncio.timeout(self.settings.stop_timeout_seconds):
                if pause is None:
                    raise RuntimeError("the old invocation's admission cannot be paused")
                cancel = self._cancel_turn()
                if cancel is not None:
                    await asyncio.shield(cancel)
                await session.ended(turn.main)
                await session.idle()
                await door.drain_admissions()
                self._drain_recovery_events()
                while door.pending_calls:
                    await asyncio.gather(*(asyncio.shield(call) for call in tuple(door.pending_calls)))
                self._drain_recovery_events()
                while self.running:
                    self._apply_event(await self.mailbox.get())
                self._drain_recovery_events()
                terminal = session.prompt_state(turn.main).terminal
                if terminal is None or not terminal.terminal_confirmed or not session.quiescent:
                    raise RuntimeError("the native invocation's terminal is unconfirmed")
                completion = self.agent.completion(terminal)
                if completion.outcome not in ("succeeded", "cancelled"):
                    raise RuntimeError(completion.error or f"native turn {completion.outcome}")
                if self.runtime_calls or self.forwarded:
                    raise RuntimeError("the old invocation still has unsettled tool activity")
        except (TimeoutError, RuntimeError) as error:
            detail = str(error) or "native cancellation, terminal or tool settlement exceeded the stop bound"
            failure = f"owned-input recovery failed: {type(error).__name__}: {detail}"
        finally:
            self._input_recovery = None
        if self.stopped or self.done or self.session is not session or self.runtime is not runtime:
            return
        if failure is not None:
            for steering in reversed(self.next_prompts):
                self.task.settle(steering, "not_applied")
            self.next_prompts = []
            async for event in self._fail(failure, usage=self._turn_usage(turn)):
                yield event
            return
        if not (self.task.can_dispatch() and self.task.pending_input_ready()):
            self._steering_ready()
            async for event in self._fail(
                "owned-input recovery no longer has eligible input", usage=self._turn_usage(turn)
            ):
                yield event
            return
        async for event in self._close(None, self._turn_usage(turn)):
            yield event
        self._between_rounds()
        if self.stopped or self.done or self._released.is_set() or not self.task.can_dispatch():
            return
        reuse = bool(
            turn.reusable_completion(completion)
            and session.alive
            and session.capabilities.continuation
            and self._history_mode == self.task.history_mode
            and self.instructions == self._instructions()
            and door.resume_admissions(pause, self.task, self.sources)
        )
        self.task.refresh_history(unconfirmed_calls=tuple(turn.call_ids) if reuse else ())
        self.task.trace(
            "harness.input_recovery_ready",
            prompt=turn.main,
            invocation_id=turn.invocation_id,
            same_session=reuse,
        )
        if reuse:
            await self._prompt_next_now()
            return
        why = "a settled old invocation could not safely continue its native session"
        await self._reopen(why, instructions=self.instructions)

    def _model_output(self, event: SessionEvent, message_id: str | None) -> bool:
        """Note that model output begins; return whether a boundary waits for it, the event read again next round."""
        new_message = message_id is not None and message_id not in self.messages
        if new_message and self.activity and self._has_items():
            self.boundary, self.pending = self._finish_next_call, event
            return True
        return False

    def _apply_event(self, event: SessionEvent | _TimedOut) -> None:
        if self._borrowed is not None and self._borrowed.old_event(event):
            return
        event.apply(self)

    def _begin_message(self, message_id: str) -> _Message:
        message = self.messages.get(message_id)
        if message is None:
            message = self.messages[message_id] = _Message(message_id)
            self.activity_since_message = False
        return message

    def _current_event(self, event: SessionEvent) -> bool:
        """Keep publication bound to the event's originating native invocation."""
        return self.turn is not None and event.prompt in (None, self.turn.main)

    def _native_input_blocks_text(self) -> bool:
        turn = self.turn
        if (pending := turn.native_input) is not None:
            self._settle_native_input(turn, pending)
        return turn.native_input is not None or turn.uncertain_input

    def message_chunk(self, event: MessageChunk) -> None:
        """Read model text; a chunk with no id is the agent's own reply, such as an ack, and is never output."""
        if not self._current_event(event):
            return
        self._seen()
        if self._native_input_blocks_text():
            return
        if not isinstance(event.message_id, str) or not event.message_id or self._model_output(event, event.message_id):
            return
        message = self._begin_message(event.message_id)
        message.parts.append(event.text)

    def thought_chunk(self, event: ThoughtChunk) -> None:
        """Read reasoning: its id begins the message its text will join."""
        if not self._current_event(event):
            return
        self._seen()
        if self._native_input_blocks_text():
            return
        if not isinstance(event.message_id, str) or not event.message_id or self._model_output(event, event.message_id):
            return
        self._begin_message(event.message_id)

    def tool_card(self, event: ToolCard) -> None:
        """Read a tool call's start; the bridge reports its own calls, and the agent's own are reported here."""
        if not self._current_event(event):
            return
        self._seen()
        if self._model_output(event, None):
            return
        self.activity = self.activity_since_message = True
        self.cards_open.add(event.call_id)
        self.pending = event.ended
        if event.bridged:
            return
        if self.turn is not None:
            self.turn.call_ids.add(event.call_id)
        call = {
            "type": "function_call",
            "id": uid("fc"),
            "call_id": event.call_id,
            "name": event.name,
            "arguments": json.dumps(dict(event.arguments or {})),
            "status": "completed",
        }
        # Reported as it runs: the commit gate cannot hold the agent's own tools.
        self.task.tool_observed(call, self.settings.runtime_label)
        self._observed_owners[event.call_id] = self.task
        self.runtime_calls[event.call_id] = call
        self.calls.append(call)

    def tool_card_end(self, event: ToolCardEnd) -> None:
        """Read a tool call's end: the transcript holds a tool row, and an agent's own call reports its outcome."""
        if not self._current_event(event):
            self.cards_open.discard(event.call_id)
            call = self.runtime_calls.pop(event.call_id, None)
            if call is not None:
                owner = self._observed_owners.pop(event.call_id, self.task)
                owner.tool_outcome(event.call_id, event.output, "failed" if event.failed else "done")
            self._ready_app()
            return
        self._seen()
        self.activity = self.activity_since_message = True
        self.tool_rows = True
        self.cards_open.discard(event.call_id)
        call = self.runtime_calls.pop(event.call_id, None)
        if call is not None:
            owner = self._observed_owners.pop(event.call_id, self.task)
            owner.tool_outcome(event.call_id, event.output, "failed" if event.failed else "done")
        self._ready_app()

    def plan(self, event: PlanUpdate) -> None:
        """Read the agent's plan as a progress note, never history."""
        if not self._current_event(event):
            return
        self._seen()
        if event.entries and not self.task.pending_input_ready():
            self.task.progress("; ".join(event.entries))

    def user_text(self, event: UserText) -> None:
        """Read a prompt the agent drains or replays itself: a boundary between model calls."""
        if not self._current_event(event):
            return
        self._seen()
        self.activity = self.activity_since_message = True

    def prompt_dispatched(self, event: PromptDispatched) -> None:
        """Apply only the immutable input attached to this actual native wire handoff."""
        if self._borrowed is not None:
            self._borrowed.dispatched(event)
        batches = self.prompt_inputs.pop(event.prompt, ())
        for steering in batches:
            self._applied(steering)
            self.task.trace(
                "harness.steer_applied",
                round=self.round,
                prompt=event.prompt,
                invocation_id=event.invocation_id,
                steering=steering.id,
                delivery="local_wire_handoff",
                model_consumed=None,
            )
        if self.turn is not None and self.turn.main == event.prompt:
            self.turn.invocation_id = event.invocation_id

    def prompt_ended(self, event: PromptEnded) -> None:
        """Read the invocation terminal; tokens and native command acknowledgements never settle caller input."""
        turn = self.turn
        batches = self.prompt_inputs.pop(event.prompt, ())
        if event.dispatch == "not_dispatched":
            for steering in reversed(batches):
                self.task.settle(steering, "not_applied")
        elif batches:
            for steering in batches:
                self.task.settle(steering, "unknown")
            self.task.trace(
                "harness.input_delivery_unknown",
                prompt=event.prompt,
                invocation_id=event.invocation_id,
                steering_ids=[steering.id for steering in batches],
            )
        if turn is None or event.prompt != turn.main:
            self.open_prompts.discard(event.prompt)
            return
        total = usage_total(event.usage)
        if total is not None:
            self.total = total
        self.usage = usage.snapshot(event.usage)
        if turn is not None and event.prompt == turn.main:
            turn.ended = event
            turn.invocation_id = event.invocation_id
            turn.main_open, turn.stop, turn.error = False, event.stop_reason, event.error
            turn.usage = event.usage
            if self.door is not None:
                self.door.admission_changed()
            self.task.trace("harness.turn_ended", round=self.round, stop=event.stop_reason, error=event.error)
        self.open_prompts.discard(event.prompt)
        if not self.open_prompts and turn is not None and not turn.main_open:
            self.boundary = self._finish_turn

    def app_call(self, event: AppCall) -> None:
        """Read an app call the bridge holds; it goes to the app once nothing else of the batch runs."""
        self._seen()
        if self.turn is not None:
            self.turn.call_ids.add(event.held.call_id)
        self.activity = self.activity_since_message = True
        self.app_queue.append(event.held)
        self._ready_app()

    def admission_ready(self, event: AdmissionReady) -> None:
        """Recheck queued calls after placement waiting changed, without making a model or response boundary."""
        self._ready_app()

    def call_running(self, event: CallRunning) -> None:
        """Read a call the bridge took, refused or run by the bridge; either is a call of this round."""
        self._seen()
        if self.turn is not None:
            self.turn.call_ids.add(event.call["call_id"])
        self.activity = self.activity_since_message = True
        self.calls.append(event.call)
        if not event.refused:
            self.running.add(event.call["call_id"])

    def call_settled(self, event: CallSettled) -> None:
        """Read the end of a call the bridge ran."""
        self.running.discard(event.call_id)
        self._ready_app()

    def wake(self, event: Wake) -> None:
        """End the round on a stop; else take waiting steering, closing the round first only when it holds calls."""
        if self.stopped or self.suspended or self._released.is_set():
            self.boundary = self._finish_stopped
        elif self._borrowed is not None and self._borrowed.steer_requested:
            self.boundary = self._borrowed.finish_steer
        elif self._input_recovery is not None and self._input_recovery[0] is self.turn:
            self.boundary = self._finish_input_recovery
        elif self.calls:
            self.boundary = self._finish_woken
        else:
            self._between_rounds()

    def timed_out(self) -> None:
        """End the round for a turn that ran past its limit."""
        self.boundary = self._finish_timeout

    def _ready_app(self) -> None:
        """Forward the oldest held app call once no other call of the batch runs (one per round)."""
        if self.suspended:
            return
        while self.app_queue and not self.forwarded and self.boundary is None:
            held = self.app_queue[0]
            if held.answer.done():
                self.app_queue.popleft()
                continue
            if self.door is None or self.door.task is not self.task or not self.door.admission_open or self.stopped:
                self.app_queue.popleft()
                held.reply(self.task.texts.stopped_output)
                continue
            if not self.door.call_current(held.call_id):
                self.app_queue.popleft()
                held.reply(self.task.texts.not_run_output)
                continue
            held_count = len(self.door.held) if self.door is not None else 0
            if self.running or len(self.cards_open) > held_count:
                return
            if self.door.placement_pending(held.call):
                self.door.defer_admission(held)
                return
            held = self.app_queue.popleft()
            {"app": self._forward, "refused": self._refuse, "backend": self._misrouted}[self.task.dispatch(held.call)](
                held
            )

    def _forward(self, held: HeldCall) -> None:
        self.calls.append(held.call)
        self.forwarded = [held]
        self.boundary = self._finish_app

    def _refuse(self, held: HeldCall) -> None:
        self.calls.append(held.call)
        reply = refused_reply(self.task, held.call_id)
        if isinstance(reply, RefusedReply) and reply.details.input_scope == "global":
            self.placement_refusals.append((held, reply))
            self.boundary = self._finish_placement_refusal
        else:
            held.reply(self.door.qualify_refusal(reply))

    def _can_cooperatively_end(self, reply: RefusedReply) -> bool:
        """Offer optional model advice only when its explanatory worker guidance was configured."""
        return bool(self.settings.prompts.worker_protocol and self._can_recover_owned_input(reply))

    def _can_recover_owned_input(self, reply: RefusedReply) -> bool:
        """Qualify mechanical recovery from trusted current ownership, independently of prompt prose."""
        return bool(
            self.settings.steering.cooperative_end
            and self.session is not None
            and self.session.capabilities.delivery == "next_prompt"
            and self._pending_delivery()
            and reply.details.task_id == self.task.task_id
            and reply.details.work_id == self.task.work_id
            and reply.details.owned_input_pending
        )

    def _pending_delivery(self) -> bool:
        return bool(
            self._borrowed is None
            and not (self.stopped or self.suspended or self.done or self._released.is_set())
            and self.door is not None
            and self.door.admission_open
            and self.door.task is self.task
            and self.next_prompts
            and self.task.pending_input_ready()
        )

    def _misrouted(self, held: HeldCall) -> None:
        """Fail an app function the IM routes to the backend, answering with ``texts.misrouted_output``."""
        output = json.dumps({"error": self.task.texts.misrouted_output.format(name=held.call["name"])})
        self.calls.append(held.call)
        self.task.tool_started(held.call, self.settings.runtime_label)
        self.task.tool_outcome(held.call_id, output, "failed")
        held.reply(output)
