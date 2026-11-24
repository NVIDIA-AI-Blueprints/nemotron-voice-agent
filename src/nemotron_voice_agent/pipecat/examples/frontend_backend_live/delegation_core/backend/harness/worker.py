# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One task's run on the harness (``port.BackendRun``): a worker session of the agent, read as the IM's rounds."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    BoundaryInput,
    Steering,
    StopOutcome,
    TaskContext,
    ToolSource,
)
from nemotron_voice_agent.pipecat.shared.ids import uid

from . import usage
from .bridge import WorkerDoor
from .control import mode_item
from .input_recovery import InputRecovery
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
)
from .render import output_text
from .runtime import cancel_in_background, give_back
from .sessions import KeptSession, keys, unread

if TYPE_CHECKING:
    from .adapter import HarnessAdapter
    from .control import BorrowedControl

OPEN_ATTEMPTS = 2
LOGGED_TEXT_CHARS = 300


class HarnessTurnError(RuntimeError):
    """The agent's turn gave no answer the run can pass on (its text is logged, never spoken)."""


@dataclass
class _Message:
    """The accumulated text for one assistant message."""

    parts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(self.parts).strip()


@dataclass
class _Turn:
    """One turn of the session: from a prompt until every prompt of the session has returned."""

    main: int
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


@dataclass
class Foreground:
    """All reader state for one business or borrowed control foreground of a native session."""

    task: TaskContext
    sources: tuple[ToolSource, ...] = ()
    turn: _Turn | None = None
    shown: list[tuple] = field(default_factory=list)
    forwarded: list[HeldCall] = field(default_factory=list)
    app_queue: deque[HeldCall] = field(default_factory=deque)
    placement_refusals: list[tuple[HeldCall, RefusedReply]] = field(default_factory=list)
    running: set[str] = field(default_factory=set)
    cards_open: set[str] = field(default_factory=set)
    runtime_calls: dict[str, dict] = field(default_factory=dict)
    next_prompts: list[Steering] = field(default_factory=list)
    prompt_inputs: dict[int, tuple[Steering, ...]] = field(default_factory=dict)
    open_prompts: set[int] = field(default_factory=set)
    done: bool = False
    answered: bool = False
    waits: bool = False
    suspended: bool = False
    suspend_pause: object | None = None
    suspend_reusable: bool = False
    suspend_outputs: dict[str, str] = field(default_factory=dict)
    pending: SessionEvent | _TimedOut | None = None
    boundary: Callable[[], AsyncIterator[dict]] | None = None
    response_id: str = ""
    messages: dict[str, _Message] = field(default_factory=dict)
    calls: list[dict] = field(default_factory=list)
    activity: bool = False
    activity_since_message: bool = False

    def reset_round(self) -> None:
        """Clear the output accumulated for the previous Responses round."""
        self.messages = {}
        self.calls = []
        self.activity = self.activity_since_message = False


class WorkerRun:
    """One task's run on the harness (``port.BackendRun``)."""

    def __init__(self, task: TaskContext, host: HarnessAdapter):
        """Run ``task`` on ``host``'s agent; nothing is opened before :meth:`events`."""
        self.host = host
        self.foreground = Foreground(task, task.tool_sources(host.tools))
        self.execution_ref = task.execution_ref
        self.role = host.workers
        self.settings, self.agent = host.settings, host.settings.agent
        self.round = 0
        self.usage_report = usage.WorkerUsage(host)
        self.stopped = False
        self.runtime: Runtime | None = None
        self.session: HarnessSession | None = None
        self.door: WorkerDoor | None = None
        self.mailbox: Mailbox = asyncio.Queue()
        self._first_prompt_blocks: tuple[str, ...] | None = None
        self.instructions = ""
        self.served: list[str] = []
        self.usage: Mapping[str, Any] | None = None
        self.tool_rows = False
        self._stopping: asyncio.Future | None = None
        self._released = asyncio.Event()
        self._reading_finished = asyncio.Event()
        self._reading_finished.set()
        self._borrowed: BorrowedControl | None = None
        self._observed_owners: dict[str, TaskContext] = {}
        self._quiet_callbacks: dict[object, Callable[[], None]] = {}
        self._quiet_subscribed = False
        self._history_mode = task.history_mode
        conversation = getattr(task, "conversation_ref", None)
        if not isinstance(conversation, str) or not conversation:
            if task.continues is not None:
                raise ValueError("A harness continuation requires a nonempty conversation_ref")
            conversation = host
        self.retirement = host.retirements.register(
            conversation,
            self.execution_ref,
            task.work_id,
            (task.continues,) if task.continues is not None else (),
        )

    @property
    def needs_initial_prompt(self) -> bool:
        """Whether the current native session has yet to receive its first prompt."""
        return self._first_prompt_blocks is not None

    @property
    def task(self) -> TaskContext:
        """The immutable context of the currently active foreground."""
        return self.foreground.task

    def borrow(self, lease: BorrowedControl, task: TaskContext) -> Foreground:
        """Move the original foreground aside intact for one Core-only control invocation."""
        self._borrowed = lease
        original = self.foreground
        self.foreground = Foreground(task)
        return original

    def restore(self, original: Foreground) -> None:
        """Restore the original reader and keep the canonical history revisions seen by the native session."""
        original.shown.extend(self.foreground.shown)
        original.reset_round()
        self.foreground = original
        self._borrowed = None

    async def events(self) -> AsyncIterator[dict]:
        """Yield the task's Responses events round by round; call it again after :meth:`resume`.

        Raises:
            HarnessTurnError: The turn gave no answer the run can pass on, or ran past its limit.
            Exception: The session could not be opened: the agent did not start or did not list its tools.
        """
        self.foreground.waits = False
        if self.foreground.done or self.stopped or self.foreground.suspended or self._released.is_set():
            return
        self._reading_finished.clear()
        try:
            if self.session is None:
                await self._open()
            if self._released.is_set():
                return
            while not (
                self.foreground.done
                or self.stopped
                or self.foreground.suspended
                or self.foreground.waits
                or self._released.is_set()
            ):
                async for event in self._round():
                    yield event
        except BaseException:
            if self.foreground.suspended or self._borrowed is not None:
                raise
            self.foreground.done = not self.stopped
            self._observe_dropped_turn("stop_requested" if self.stopped else "incomplete")
            if not self.stopped:
                # The turn would run on with no reader; a stop cancels its own.
                self.cancel_turn()
            await self.release()
            raise
        finally:
            self._reading_finished.set()
            if self._borrowed is not None:
                self._borrowed.check_ready()
            self._notify_quiescent()
        if self.foreground.done and not self.foreground.suspended and self._borrowed is None:
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
        if not (self.foreground.done or self.stopped or self.foreground.suspended):
            if self.door is not None:
                self.door.admission_changed()
            self.mailbox.put_nowait(Wake())

    async def resume(self) -> None:
        """Queue waiting steering, then answer the held app call without claiming native input ingestion."""
        if self.stopped or self._released.is_set():
            raise RuntimeError("A stopped native worker cannot resume")
        if self.foreground.suspended:
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
        while self.steering_ready(steering) and steering is not None:
            self._deliver(steering)
            steering = self._boundary().steering
        for held in self.foreground.forwarded:
            text = output_text(outputs.get(held.call_id))
            text = self.task.texts.owed_output if text is None else text
            held.reply(refused_reply(self.task, held.call_id, fallback=text))
        self.foreground.forwarded = []

    async def suspend(self) -> StopOutcome:
        """Suspend business inference while retaining its task context and genuine effects."""
        if self._borrowed is not None:
            return await self._borrowed.business_suspension()
        return await self._suspend_foreground()

    async def prepare_restricted_control(self) -> None:
        """Settle the old native invocation before interpreting input in a separate restricted model call."""
        if await self.suspend() != "stopped":
            raise RuntimeError("The original native foreground did not confirm suspension")
        self.return_next_prompts()

    def return_next_prompts(self) -> None:
        """Return batches that were never attached to a native invocation to the Core queue."""
        waiting, self.foreground.next_prompts = self.foreground.next_prompts, []
        for batch in reversed(waiting):
            self.task.settle(batch, "not_applied")

    async def _suspend_foreground(self, *, reading: bool = False) -> StopOutcome:
        """Confirm only this foreground; an already-dispatched Core business operation may still be pending."""
        self.foreground.suspended = True
        if self.door is not None and self.foreground.suspend_pause is None:
            self.foreground.suspend_pause = self.door.pause_admissions(self.task.texts.not_run_output)
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
                if self.foreground.suspend_pause is None:
                    self.foreground.suspend_pause = door.pause_admissions(self.task.texts.not_run_output)
                if self.foreground.suspend_pause is None:
                    return outcome
                turn = self.foreground.turn
                cancel = self.cancel_turn()
                if cancel is not None:
                    await asyncio.shield(cancel)
                if turn is not None:
                    await session.ended(turn.main)
                await session.idle()
                await door.drain_admissions()
                self.drain_recovery_events()
                while self.foreground.app_queue:
                    held = self.foreground.app_queue[0]
                    if held.answer.done():
                        self.foreground.app_queue.popleft()
                    elif self.task.dispatch(held.call) == "refused":
                        self.foreground.app_queue.popleft()
                        self._refuse(held)
                    else:
                        return outcome
                for held, reply in self.foreground.placement_refusals:
                    held.reply(reply)
                self.foreground.placement_refusals = []
                retained = frozenset(held.call_id for held in self.foreground.forwarded)
                while pending := door.managed_calls(retained):
                    await asyncio.gather(*(asyncio.shield(call) for call in pending), return_exceptions=True)
                self.drain_recovery_events()
                if (
                    self.stopped
                    or self._released.is_set()
                    or not session.quiescent
                    or self.foreground.running
                    or self.foreground.runtime_calls
                ):
                    return outcome
                terminal = session.prompt_state(turn.main).terminal if turn is not None else None
                unsent = terminal is not None and terminal.dispatch == "not_dispatched"
                if turn is not None and not unsent and (terminal is None or not terminal.terminal_confirmed):
                    return outcome
                completion = self.agent.completion(terminal) if terminal is not None else None
                self.foreground.suspend_reusable = (
                    turn is None or unsent or (completion is not None and turn.reusable_completion(completion))
                )
                self._observe_dropped_turn("suspended")
                self.foreground.pending = self.foreground.boundary = None
                outcome = "stopped"
                return outcome
        except (TimeoutError, RuntimeError):
            return outcome
        finally:
            self.task.trace("backend.suspended", round=self.round, outcome=outcome)

    def can_recover_input(self) -> bool:
        """Whether this business foreground still owns eligible input for another prompt."""
        return bool(
            self._borrowed is None
            and not (self.stopped or self.foreground.suspended or self.foreground.done or self.released)
            and self.door is not None
            and self.door.admission_open
            and self.door.task is self.task
            and self.foreground.next_prompts
            and self.task.pending_input_ready()
        )

    def resume_admissions(self, pause: object | None) -> bool:
        """Resume tools on this session when its context and continuation capability still apply."""
        session, door = self.session, self.door
        return bool(
            session is not None
            and session.alive
            and session.capabilities.continuation
            and self._history_mode == self.task.history_mode
            and self.instructions == self._instructions()
            and door is not None
            and door.resume_admissions(pause, self.task, self.foreground.sources)
        )

    @property
    def released(self) -> bool:
        """Whether this worker has permanently surrendered its native owner."""
        return self._released.is_set()

    async def _resume_suspended(self) -> None:
        """Resume after pending business tool outputs and the old native turn settle.

        A restricted control foreground may have seen a different history view. Compare exact history
        revisions for this same task instead of requiring a pooled-session ancestry suffix.
        """
        self.return_next_prompts()
        batches: list[Steering] = []
        try:
            self._collect(batches)
            door, session, turn = self.door, self.session, self.foreground.turn
            pending = [held for held in self.foreground.forwarded if not held.answer.done()]
            if any(held.call_id not in self.foreground.suspend_outputs for held in pending):
                raise RuntimeError("The original application operation still has no actual result")
            for held in pending:
                held.reply(self.foreground.suspend_outputs.pop(held.call_id))
            self.foreground.forwarded = [held for held in self.foreground.forwarded if not held.answer.done()]
            if door is not None and door.pending_calls:
                async with asyncio.timeout(self.settings.stop_timeout_seconds):
                    await asyncio.gather(*(asyncio.shield(call) for call in tuple(door.pending_calls)))
            if self.stopped or self._released.is_set() or not self.task.can_dispatch():
                return
            self.task.refresh_history(unconfirmed_calls=tuple(turn.call_ids) if turn is not None else ())
            mode = mode_item(self.task)
            extra = [item for batch in batches for item in batch.items]
            history = self.task.history(extra=[*extra, mode])
            seen = set(self.foreground.shown)
            unseen = [item for item, key in zip(history, keys(history), strict=True) if key not in seen]
            instructions = self._instructions()
            reusable = self.foreground.suspend_reusable and self.resume_admissions(self.foreground.suspend_pause)
            self.foreground.suspended = self.foreground.done = self.foreground.answered = False
            self.foreground.suspend_pause = None
            if reusable:
                if self._first_prompt_blocks is None:
                    self.foreground.shown += keys([item for item in unseen if item is not mode])
                    text = self.host.renderer.transcript(unseen)
                else:
                    self.foreground.shown = keys([item for item in history if item is not mode])
                    text = self.host.renderer.worker_input(history)
                scheduled = tuple(batches)
                batches.clear()
                self._prompt(text, scheduled)
            elif session is None:
                for steering in reversed(batches):
                    self.task.settle(steering, "not_applied")
                batches.clear()
            else:
                self.foreground.next_prompts.extend(batches)
                batches.clear()
                await self.reopen("a suspended invocation requires a fresh native session", instructions=instructions)
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
                    cancel = self.cancel_turn()
                    if cancel is not None:
                        await asyncio.shield(cancel)
                    await session.idle()
                await self.release()
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
        self.return_next_prompts()
        session = self.session
        if session is not None:
            session.retire()
            for prompt, batches in self.foreground.prompt_inputs.items():
                state = session.prompt_state(prompt)
                delivery = "unknown" if state.dispatched else "not_applied"
                for steering in reversed(batches):
                    self.task.settle(steering, delivery)
            self.foreground.prompt_inputs.clear()
        runtime, self.runtime = self.runtime, None
        if runtime is not None:
            pending = list(self.door.pending_calls) if self.door is not None else []
            if self.foreground.turn is not None and self.foreground.turn.cancelling is not None:
                pending.append(self.foreground.turn.cancelling)
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
        pending = [held for held in self.foreground.forwarded if not held.answer.done()]
        try:
            if not pending:
                return
            boundary = self.task.take_boundary_input()
            if boundary.steering is not None:
                self.task.settle(boundary.steering, "not_applied")
            outputs = {item.get("call_id"): item for item in boundary.outputs}
            for held in pending:
                output = self.foreground.suspend_outputs.pop(held.call_id, output_text(outputs.get(held.call_id)))
                reply = refused_reply(
                    self.task, held.call_id, fallback=self.task.texts.owed_output if output is None else output
                )
                if output is None and isinstance(reply, RefusedReply):
                    reply = refused_reply(self.task, held.call_id)
                held.reply(reply)
        finally:
            for held in pending:
                held.reply(self.task.texts.owed_output)

    def pause_admissions(self) -> object | None:
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
            or runtime is None
            or not self.foreground.answered
            or self.stopped
            or not self.task.can_dispatch()
        ):
            return None
        if self.foreground.turn is not None and self.foreground.turn.cancel_dispatch != "not_dispatched":
            return None
        left = (
            session.outstanding,
            self.foreground.next_prompts,
            self.foreground.prompt_inputs,
            self.foreground.forwarded,
            self.foreground.app_queue,
        )
        lost = not (runtime.client.alive and session.alive and session.quiescent and door.admission_open)
        if (
            any(left)
            or self.foreground.running
            or self.foreground.cards_open
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
            list(self.foreground.shown),
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
        return self.task.request().get("instructions") or ""

    def _misfit(self, kept: KeptSession, instructions: str, history: list[dict]) -> str | None:
        """Return why the idle session ``kept`` cannot serve the task; else ``None``.

        A session whose process or session is lost is retired on the way; any other stays idle as it was.
        """
        if not (kept.runtime.client.alive and kept.session.alive):
            why = "session lost" if kept.runtime.client.alive else "process lost"
            self.host.take_kept(kept)
            self._retire_kept(kept, why)
            return why
        if not kept.session.quiescent:
            return "native invocation remains unconfirmed"
        if kept.instructions != instructions:
            return "other instructions"
        if kept.history_mode != self.task.history_mode:
            return "other history mode"
        if not kept.door.serves(self.task, self.foreground.sources):
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
            if missed.items is None or not kept.door.rebind(self.task, self.foreground.sources):
                return missed.why or "another toolset"
            self.host.take_kept(kept)
            self.runtime, self.session, self.door, self.mailbox = kept.runtime, kept.session, kept.door, kept.mailbox
            adopted = True
            while not self.mailbox.empty():
                self.mailbox.get_nowait()
            self.instructions, self.served = instructions, list(kept.served)
            self.foreground.shown = [*kept.shown, *keys(missed.items)]
            self.usage, self.tool_rows = kept.usage, kept.tool_rows
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
            self.foreground.turn.reused = True
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
    ) -> None:
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
                    self.foreground.sources,
                    mailbox,
                    tool_timeout_seconds=self.settings.tool_timeout_seconds,
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
        self.foreground.turn = None
        self.foreground.shown = []
        self._first_prompt_blocks = blocks
        self.usage = dict.fromkeys(usage.COUNTERS, 0)
        self.instructions, self.served = instructions, []
        seconds = round(time.monotonic() - started, 3)
        self.task.trace("harness.session_opened", server=name, session=session.id, seconds=seconds, why=why)
        if self.stopped or self._released.is_set():
            await self.release()
            return
        if not prompt or self.foreground.suspended:
            return
        self._collect(batches)
        history = self.task.history(extra=[item for batch in batches for item in batch.items])
        self.foreground.shown = keys(history)
        text = self.host.renderer.worker_input(history)
        scheduled = tuple(batches)
        batches.clear()  # _prompt owns both successful scheduling and rejection before dispatch.
        self._prompt(text, scheduled)

    async def _open_for_control(self) -> None:
        """Open only a never-opened logical worker, without starting a business prompt or another control runtime."""
        await self._wait_for_predecessors()
        if self.stopped or self._released.is_set():
            raise RuntimeError("The native logical worker is no longer available")
        if self.session is not None:
            return
        if self.runtime is None:
            self.runtime = await self.host.shared(self.role)
        await self._prepare_session(self._instructions(), ["first logical worker control"], [], prompt=False)
        if self.session is None or self.door is None:
            raise RuntimeError("The first native worker session did not open")

    def _prompt(self, text: str, batches: tuple[Steering, ...] = ()) -> None:
        """Bind immutable input and any unsent startup instructions to one scheduled native invocation."""
        if self._first_prompt_blocks:
            text = "\n\n".join((*self._first_prompt_blocks, text))
        try:
            prompt = self.session.prompt(text, may_dispatch=self._can_dispatch)
        except BaseException:
            for batch in reversed(batches):
                self.task.settle(batch, "not_applied")
            raise
        self.foreground.prompt_inputs[prompt] = batches
        self._start_turn(prompt)

    def _can_dispatch(self) -> bool:
        return bool(
            not (self.stopped or self.foreground.suspended or self.foreground.done)
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
        self.foreground.open_prompts.add(prompt)
        session = self.session
        try:
            session_id = session.id
        except Exception:
            session_id = None
        self.foreground.turn = _Turn(prompt, self.usage, session_id=session_id, started_round=self.round)
        self.door.invocation = self.foreground.turn
        self.door.invocation_active = lambda: session.prompt_state(prompt).phase == "dispatched"
        self.door.admission_changed()
        self.task.trace("harness.prompt_sent", round=self.round, prompt=prompt)
        if self._borrowed is not None:
            self._borrowed.prompt_started()

    async def _round(self) -> AsyncIterator[dict]:
        """Yield one round: ``response.created``, the session's events up to a boundary, then the boundary's ending.

        A round ends when a model message begins after tool activity, when an app call is forwarded, when steering needs
        a safe point, when every prompt has returned, or on a stop or a timeout.
        """
        self.round += 1
        self.foreground.reset_round()
        self.foreground.response_id = uid("resp")
        yield {
            "type": "response.created",
            "response": {"id": self.foreground.response_id, "status": "in_progress", "output": []},
        }
        if self.stopped or self.foreground.suspended or self._released.is_set():
            return
        self._ready_app()
        while self.foreground.boundary is None:
            if self.stopped or self.foreground.suspended or self._released.is_set():
                return
            if recovery := InputRecovery.when_ready(self):
                self.foreground.boundary = recovery.finish
                break
            event, self.foreground.pending = self.foreground.pending, None
            if event is None:
                event = await self._next()
            self.apply_event(event)
        finish, self.foreground.boundary = self.foreground.boundary, None
        async for event in finish():
            yield event

    async def _next(self) -> SessionEvent | _TimedOut:
        """Return the session's next event, or a :class:`_TimedOut` once the turn's reading time is up.

        Core placement pauses only the reading budget. A task continuing after the call uses its Work's
        keep deadline and stop boundary.
        """
        turn = self.foreground.turn
        if turn is None or self.task.after_call:
            return await self.mailbox.get()
        started = time.monotonic()
        door = self.door
        paused_before = door.placement_elapsed() if door is not None else 0.0
        try:
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
        return bool(self.foreground.calls) or any(message.text for message in self.foreground.messages.values())

    async def close_round(self, final: _Message | None, usage: Mapping[str, Any] | None = None) -> AsyncIterator[dict]:
        """End the round: messages (``final`` is the answer), then calls."""
        items = []
        for message in self.foreground.messages.values():
            if message.text and not self._input_prevents_output():
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
        items += [dict(call) for call in self.foreground.calls]
        self.foreground.shown += keys(items)
        for index, item in enumerate(items):
            opening = {**item, "status": "in_progress"}
            if item["type"] == "message":
                opening["content"] = []
            yield {"type": "response.output_item.added", "output_index": index, "item": opening}
            yield {"type": "response.output_item.done", "output_index": index, "item": item}
        response = {"id": self.foreground.response_id, "status": "completed", "output": items}
        if usage:
            response["usage"] = usage
        yield {"type": "response.completed", "response": response}
        self.foreground.reset_round()

    def _input_prevents_output(self) -> bool:
        return bool(self.task.pending_input_ready() or self.foreground.next_prompts or self.foreground.prompt_inputs)

    async def fail_round(
        self, reason: str, text: str = "", usage: Mapping[str, Any] | None = None
    ) -> AsyncIterator[dict]:
        """End the round ``response.failed`` and raise :class:`HarnessTurnError`; the agent's text is only logged."""
        logger.warning(
            f"harness.turn_refused task={self.task.task_id} reason={reason!r} text={text[:LOGGED_TEXT_CHARS]!r}"
        )
        self.task.trace("harness.turn_refused", round=self.round, reason=reason)
        error = {"code": "server_error", "message": reason}
        response = {"id": self.foreground.response_id, "status": "failed", "output": [], "error": error}
        if usage is not None:
            response["usage"] = usage
        yield {"type": "response.failed", "response": response}
        self.foreground.done = True
        raise HarnessTurnError(reason)

    def steering_ready(self, current: Steering | None = None) -> bool:
        """Return ineligible taken input for Core follow-up, preserving its original order."""
        if self.task.pending_input_ready():
            return True
        waiting, self.foreground.next_prompts = self.foreground.next_prompts, []
        if current is not None:
            waiting.append(current)
        for steering in reversed(waiting):
            self.task.settle(steering, "not_applied")
        return False

    def between_rounds(self) -> None:
        """Join owed outputs and hand eligible waiting batches to the steering strategy."""
        if self.foreground.suspended:
            return
        steering = self._boundary().steering
        while self.steering_ready(steering) and steering is not None:
            self._deliver(steering)
            steering = self._boundary().steering

    def _deliver(self, steering: Steering) -> None:
        """Queue text for the next native prompt. Apply input without text immediately."""
        if self.host.renderer.steering(steering):
            self.foreground.next_prompts.append(steering)
            return
        self._applied(steering)
        self.task.trace("harness.steer_applied", round=self.round, steering=steering.id, how="no_text")

    async def _finish_next_call(self) -> AsyncIterator[dict]:
        async for event in self.close_round(None):
            yield event
        self.between_rounds()

    async def _finish_woken(self) -> AsyncIterator[dict]:
        async for event in self.close_round(None):
            yield event
        self.between_rounds()

    async def _finish_placement_refusal(self) -> AsyncIterator[dict]:
        """Let Core settle global placement before the native call gets its original non-owned refusal."""
        async for event in self.close_round(None):
            yield event
        self.between_rounds()
        waiting, self.foreground.placement_refusals = self.foreground.placement_refusals, []
        if not self.stopped:
            for held, reply in waiting:
                held.reply(reply)

    async def _finish_app(self) -> AsyncIterator[dict]:
        async for event in self.close_round(None):
            yield event
        self.foreground.waits = True
        self.task.trace(
            "tools.waiting", round=self.round, call_ids=[held.call_id for held in self.foreground.forwarded]
        )

    async def _finish_stopped(self) -> AsyncIterator[dict]:
        return
        yield

    async def _finish_timeout(self) -> AsyncIterator[dict]:
        """Fail the task whose turn ran past ``turn_timeout_seconds``; :meth:`events` cancels the turn."""
        async for event in self.fail_round(f"the turn ran past {self.settings.turn_timeout_seconds:g} s"):
            yield event

    async def finish_turn(self) -> AsyncIterator[dict]:
        """End the turn with the answer, the next prompt, a fresh session, or a failure.

        The answer is the last message after the turn's last tool activity, when the turn ended ``end_turn`` and
        the agent reports a successful native completion. Eligible input first goes to the same native
        session; only the refreshed answer ends the task. A turn without an answer continues with input or fails
        the task.
        """
        turn = self.foreground.turn
        if self.stopped or self.foreground.suspended or self._released.is_set():
            return
        turn_usage = self.turn_usage(turn)
        candidate = self._candidate()
        failure = self._turn_failure(turn)
        reason = failure or self._unparsed(candidate)
        if reason is None:
            self.between_rounds()
            if self.foreground.next_prompts:
                async for event in self.close_round(None, turn_usage):
                    yield event
                await self.prompt_next_now()
                return
            self.task.trace("harness.answered", round=self.round, seconds=round(time.monotonic() - turn.started, 3))
            async for event in self.close_round(candidate, turn_usage):
                yield event
            self.foreground.done = self.foreground.answered = True
            return
        if (
            failure is not None
            and turn.reused
            and not turn.seen
            and turn.ended is not None
            and turn.ended.dispatch == "not_dispatched"
        ):
            async for event in self.close_round(None, turn_usage):
                yield event
            await self._renew()
            return
        if failure is not None:
            async for event in self.fail_round(failure, self._last_text(), turn_usage):
                yield event
        if not self.foreground.calls:
            # Safe: the open round holds no call whose output could join before it.
            self.between_rounds()
        else:
            self.steering_ready()
        waiting = self.foreground.next_prompts
        if waiting:
            async for event in self.close_round(None, turn_usage):
                yield event
            await self.prompt_next_now()
            return
        async for event in self.fail_round(reason, self._last_text(), turn_usage):
            yield event

    def _turn_failure(self, turn: _Turn) -> str | None:
        completion = self.agent.completion(turn.ended or PromptEnded(turn.main, turn.stop, turn.usage, turn.error))
        return None if completion.outcome == "succeeded" else completion.error or f"native turn {completion.outcome}"

    def _candidate(self) -> _Message | None:
        if self.foreground.activity_since_message or not self.foreground.messages:
            return None
        return list(self.foreground.messages.values())[-1]

    def _unparsed(self, candidate: _Message | None) -> str | None:
        """Return why the turn's last message is no answer the run may pass on, or ``None``."""
        if candidate is None or not candidate.text:
            return "the turn ended without a message after its last tool call"
        return None

    def _last_text(self) -> str:
        texts = [message.text for message in self.foreground.messages.values() if message.text]
        return texts[-1] if texts else ""

    def turn_usage(self, turn: _Turn) -> dict | None:
        """Measure a completed native turn for the worker's response."""
        return self.usage_report.completed(self.task, turn, self.round)

    def _observe_dropped_turn(self, ending: str) -> None:
        self.usage_report.dropped(self.task, self.foreground.turn, self.round, ending)

    def _collect(self, batches: list[Steering]) -> list[Steering]:
        """Let the owed outputs join at a safe point, then add every waiting batch to ``batches`` after them."""
        steering = self._boundary().steering
        while steering is not None:
            batches.append(steering)
            steering = self._boundary().steering
        return batches

    def _boundary(self) -> BoundaryInput:
        """Take input at a safe point and retain unread tool results for a suspended invocation."""
        boundary = self.task.take_boundary_input()
        if self.foreground.suspended:
            retained = {held.call_id for held in self.foreground.forwarded}
            for item in boundary.outputs:
                if item.get("call_id") in retained:
                    text = output_text(item)
                    if text is not None:
                        self.foreground.suspend_outputs[item["call_id"]] = text
        else:
            self.foreground.shown += keys(boundary.outputs)
        return boundary

    def _applied(self, steering: Steering) -> None:
        self.task.settle(steering, "applied")
        self.foreground.shown += keys(steering.items)

    async def prompt_next_now(self) -> None:
        """Send every batch the turn did not take as the next prompt, oldest first; each is ``applied`` once sent."""
        if self.stopped or self.foreground.suspended or self.foreground.done:
            return
        batches, self.foreground.next_prompts = self.foreground.next_prompts, []
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
        await self.reopen(why)

    async def reopen(self, why: str, *, instructions: str | None = None) -> None:
        """Open a fresh session in place of the run's, prompted with the history and the batches, saying ``why``.

        The previous invocation has settled or did not dispatch.
        """
        batches, self.foreground.next_prompts = self.foreground.next_prompts, []
        self._close_door(self.task.texts.not_run_output)
        if self.session is not None:
            self.session.retire()
        (
            self.foreground.next_prompts,
            self.foreground.prompt_inputs,
            self.foreground.app_queue,
            self.foreground.forwarded,
        ) = [], {}, deque(), []
        self.foreground.open_prompts = set()
        self.foreground.running, self.foreground.cards_open, self.foreground.runtime_calls = set(), set(), {}
        self.foreground.pending = self.foreground.boundary = None
        self.tool_rows = False
        await self._open_session(self._instructions() if instructions is None else instructions, [why], tuple(batches))

    def _seen(self) -> None:
        turn = self.foreground.turn
        if turn is None or turn.seen:
            return
        turn.seen = True

    def cancel_turn(self) -> asyncio.Task[CancelDispatch] | None:
        """Cancel the running turn once, in the background, within ``stop_timeout_seconds``."""
        turn, session = self.foreground.turn, self.session
        if turn is None:
            return None
        if turn.cancelling is not None:
            return turn.cancelling
        if turn.cancelled_by_us or session is None or not session.outstanding:
            return None
        turn.cancelled_by_us = True
        turn.cancelling = cancel_in_background(session, self.settings.stop_timeout_seconds)
        return turn.cancelling

    def drain_recovery_events(self) -> None:
        """Settle tool effects from the old invocation and discard any deferred obsolete wording."""
        while self.foreground.pending is not None or not self.mailbox.empty():
            event, self.foreground.pending = self.foreground.pending, None
            if event is None:
                event = self.mailbox.get_nowait()
            self.apply_event(event)
            if self.foreground.pending is event:
                self.foreground.pending = None
        self.foreground.boundary = None

    def _drain_retirement_events(self) -> None:
        """Join late tool facts and wake the stopped reader after consuming its old mailbox events."""
        while self.foreground.pending is not None or not self.mailbox.empty():
            event, self.foreground.pending = self.foreground.pending, None
            if event is None:
                event = self.mailbox.get_nowait()
            if isinstance(event, (ToolCard, ToolCardEnd, CallRunning, CallSettled, PromptEnded)):
                self.apply_event(event)
        self.foreground.pending = self.foreground.boundary = None
        self.mailbox.put_nowait(Wake())

    def _model_output(self, event: SessionEvent, message_id: str | None) -> bool:
        """Note that model output begins; return whether a boundary waits for it, the event read again next round."""
        new_message = message_id is not None and message_id not in self.foreground.messages
        if new_message and self.foreground.activity and self._has_items():
            self.foreground.boundary, self.foreground.pending = self._finish_next_call, event
            return True
        return False

    def apply_event(self, event: SessionEvent | _TimedOut) -> None:
        """Send each native event to the foreground that owns it."""
        if self._borrowed is not None and self._borrowed.old_event(event):
            return
        event.apply(self)

    def _begin_message(self, message_id: str) -> _Message:
        message = self.foreground.messages.get(message_id)
        if message is None:
            message = self.foreground.messages[message_id] = _Message()
            self.foreground.activity_since_message = False
        return message

    def _current_event(self, event: SessionEvent) -> bool:
        """Keep publication bound to the event's originating native invocation."""
        return self.foreground.turn is not None and event.prompt in (None, self.foreground.turn.main)

    def message_chunk(self, event: MessageChunk) -> None:
        """Read model text; a chunk with no id is the agent's own reply, such as an ack, and is never output."""
        if not self._current_event(event):
            return
        self._seen()
        if not isinstance(event.message_id, str) or not event.message_id or self._model_output(event, event.message_id):
            return
        message = self._begin_message(event.message_id)
        message.parts.append(event.text)

    def thought_chunk(self, event: ThoughtChunk) -> None:
        """Read reasoning: its id begins the message its text will join."""
        if not self._current_event(event):
            return
        self._seen()
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
        self.foreground.activity = self.foreground.activity_since_message = True
        self.foreground.cards_open.add(event.call_id)
        self.foreground.pending = event.ended
        if event.bridged:
            return
        if self.foreground.turn is not None:
            self.foreground.turn.call_ids.add(event.call_id)
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
        self.foreground.runtime_calls[event.call_id] = call
        self.foreground.calls.append(call)

    def tool_card_end(self, event: ToolCardEnd) -> None:
        """Read a tool call's end: the transcript holds a tool row, and an agent's own call reports its outcome."""
        if not self._current_event(event):
            self.foreground.cards_open.discard(event.call_id)
            call = self.foreground.runtime_calls.pop(event.call_id, None)
            if call is not None:
                owner = self._observed_owners.pop(event.call_id, self.task)
                owner.tool_outcome(event.call_id, event.output, "failed" if event.failed else "done")
            self._ready_app()
            return
        self._seen()
        self.foreground.activity = self.foreground.activity_since_message = True
        self.tool_rows = True
        self.foreground.cards_open.discard(event.call_id)
        call = self.foreground.runtime_calls.pop(event.call_id, None)
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
        self.foreground.activity = self.foreground.activity_since_message = True

    def prompt_dispatched(self, event: PromptDispatched) -> None:
        """Apply only the immutable input attached to this actual native wire handoff."""
        if self._borrowed is not None:
            self._borrowed.dispatched(event)
        batches = self.foreground.prompt_inputs.pop(event.prompt, ())
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
        if self.foreground.turn is not None and self.foreground.turn.main == event.prompt:
            self._first_prompt_blocks = None
            self.foreground.turn.invocation_id = event.invocation_id

    def prompt_ended(self, event: PromptEnded) -> None:
        """Read the invocation terminal; tokens and native command acknowledgements never settle caller input."""
        turn = self.foreground.turn
        batches = self.foreground.prompt_inputs.pop(event.prompt, ())
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
            self.foreground.open_prompts.discard(event.prompt)
            return
        self.usage = usage.snapshot(event.usage)
        if turn is not None and event.prompt == turn.main:
            turn.ended = event
            turn.invocation_id = event.invocation_id
            turn.main_open, turn.stop, turn.error = False, event.stop_reason, event.error
            turn.usage = event.usage
            if self.door is not None:
                self.door.admission_changed()
            self.task.trace("harness.turn_ended", round=self.round, stop=event.stop_reason, error=event.error)
        self.foreground.open_prompts.discard(event.prompt)
        if not self.foreground.open_prompts and turn is not None and not turn.main_open:
            self.foreground.boundary = self.finish_turn

    def app_call(self, event: AppCall) -> None:
        """Read an app call the bridge holds; it goes to the app once nothing else of the batch runs."""
        self._seen()
        if self.foreground.turn is not None:
            self.foreground.turn.call_ids.add(event.held.call_id)
        self.foreground.activity = self.foreground.activity_since_message = True
        self.foreground.app_queue.append(event.held)
        self._ready_app()

    def admission_ready(self, event: AdmissionReady) -> None:
        """Recheck queued calls after placement waiting changed, without making a model or response boundary."""
        self._ready_app()

    def call_running(self, event: CallRunning) -> None:
        """Read a call the bridge took, refused or run by the bridge; either is a call of this round."""
        self._seen()
        if self.foreground.turn is not None:
            self.foreground.turn.call_ids.add(event.call["call_id"])
        self.foreground.activity = self.foreground.activity_since_message = True
        self.foreground.calls.append(event.call)
        if not event.refused:
            self.foreground.running.add(event.call["call_id"])

    def call_settled(self, event: CallSettled) -> None:
        """Read the end of a call the bridge ran."""
        self.foreground.running.discard(event.call_id)
        self._ready_app()

    def wake(self, event: Wake) -> None:
        """End the round on a stop; else take waiting steering, closing the round first only when it holds calls."""
        if self.stopped or self.foreground.suspended or self._released.is_set():
            self.foreground.boundary = self._finish_stopped
        elif self._borrowed is not None and self._borrowed.steer_requested:
            self.foreground.boundary = self._borrowed.finish_steer
        elif self.foreground.calls:
            self.foreground.boundary = self._finish_woken
        else:
            self.between_rounds()

    def timed_out(self) -> None:
        """End the round for a turn that ran past its limit."""
        self.foreground.boundary = self._finish_timeout

    def _ready_app(self) -> None:
        """Forward the oldest held app call once no other call of the batch runs (one per round)."""
        if self.foreground.suspended:
            return
        while self.foreground.app_queue and not self.foreground.forwarded and self.foreground.boundary is None:
            held = self.foreground.app_queue[0]
            if held.answer.done():
                self.foreground.app_queue.popleft()
                continue
            if self.door is None or self.door.task is not self.task or not self.door.admission_open or self.stopped:
                self.foreground.app_queue.popleft()
                held.reply(self.task.texts.stopped_output)
                continue
            if not self.door.call_current(held.call_id):
                self.foreground.app_queue.popleft()
                held.reply(self.task.texts.not_run_output)
                continue
            held_count = len(self.door.held) if self.door is not None else 0
            if self.foreground.running or len(self.foreground.cards_open) > held_count:
                return
            if self.door.placement_pending(held.call):
                self.door.defer_admission(held)
                return
            held = self.foreground.app_queue.popleft()
            {"app": self._forward, "refused": self._refuse, "backend": self._misrouted}[self.task.dispatch(held.call)](
                held
            )

    def _forward(self, held: HeldCall) -> None:
        self.foreground.calls.append(held.call)
        self.foreground.forwarded = [held]
        self.foreground.boundary = self._finish_app

    def _refuse(self, held: HeldCall) -> None:
        self.foreground.calls.append(held.call)
        reply = refused_reply(self.task, held.call_id)
        if isinstance(reply, RefusedReply) and reply.details.input_scope == "global":
            self.foreground.placement_refusals.append((held, reply))
            self.foreground.boundary = self._finish_placement_refusal
        else:
            held.reply(reply)

    def _misrouted(self, held: HeldCall) -> None:
        """Fail an app function the IM routes to the backend, answering with ``texts.misrouted_output``."""
        output = json.dumps({"error": self.task.texts.misrouted_output.format(name=held.call["name"])})
        self.foreground.calls.append(held.call)
        self.task.tool_started(held.call, self.settings.runtime_label)
        self.task.tool_outcome(held.call_id, output, "failed")
        held.reply(output)
