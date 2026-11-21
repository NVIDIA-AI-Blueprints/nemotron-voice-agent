# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A borrowed foreground of the existing native worker; logical task, runtime, SID and old effects stay owned."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from examples.frontend_backend_live.delegation_core.backend.port import (
    ControlContext,
    Steering,
    StopOutcome,
    TaskContext,
)
from examples.shared.ids import uid

from .protocol import CallSettled, PromptDispatched, SessionEvent, ToolCardEnd, Wake
from .sessions import keys

if TYPE_CHECKING:
    from .worker import WorkerRun


def mode_item(task: TaskContext) -> dict:
    """Private capability/state data, not caller input, delivery evidence or a business outcome."""
    return {
        "type": "message",
        "role": "developer",
        "id": uid("native_foreground"),
        "content": [
            {
                "type": "input_text",
                "text": json.dumps(
                    {
                        "private_native_foreground": {
                            "task_id": task.task_id,
                            "execution_ref": task.execution_ref,
                            "business_access": task.business_access,
                        }
                    }
                ),
            }
        ],
    }


@dataclass
class _BusinessForeground:
    """The inactive business reader's state; callbacks/effects remain attached to their original immutable owner."""

    task: TaskContext
    sources: tuple
    turn: Any
    shown: list
    forwarded: list
    app_queue: deque
    placement_refusals: list
    running: set
    cards_open: set
    runtime_calls: dict
    next_prompts: list
    prompt_inputs: dict
    open_prompts: set
    done: bool
    answered: bool
    waits: bool
    pause: object
    outputs: dict

    @classmethod
    def capture(cls, run: WorkerRun) -> _BusinessForeground:
        return cls(
            run.task,
            run.sources,
            run.turn,
            run.shown,
            run.forwarded,
            run.app_queue,
            run.placement_refusals,
            run.running,
            run.cards_open,
            run.runtime_calls,
            run.next_prompts,
            run.prompt_inputs,
            run.open_prompts,
            run.done,
            run.answered,
            run.waits,
            run._suspend_pause,
            run._suspend_outputs,
        )

    def restore(self, run: WorkerRun, seen: list) -> None:
        run.task, run.sources, run.turn = self.task, self.sources, self.turn
        run.shown = [*self.shown, *seen]
        run.forwarded, run.app_queue, run.placement_refusals = self.forwarded, self.app_queue, self.placement_refusals
        run.running, run.cards_open, run.runtime_calls = self.running, self.cards_open, self.runtime_calls
        run.next_prompts, run.prompt_inputs, run.open_prompts = self.next_prompts, self.prompt_inputs, self.open_prompts
        run.done, run.answered, run.waits = self.done, self.answered, self.waits
        run._suspend_pause, run._suspend_outputs = self.pause, self.outputs
        run.suspended, run._suspend_reusable = True, True
        run.pending = run.boundary = run._input_recovery = None
        run._new_round()


class BorrowedControl:
    """One Core-only inference lease using its parent's native conversation, never another WorkerRun or SID."""

    def __init__(self, parent: WorkerRun, task: ControlContext):
        """Bind the foreground to the logical worker that owns its native conversation."""
        if parent.task is not task.original_context or parent._borrowed is not None:
            raise ValueError("A control foreground requires its exact original native logical worker")
        self.parent, self.task = parent, task
        self.saved: _BusinessForeground | None = None
        self.binding = None
        self.entered = False
        self.finished = False
        self.stop_requested = False
        self.steer_requested = False
        self.business_ready = asyncio.Event()
        self.business_outcome: StopOutcome = "unconfirmed"
        self.stop_owner: asyncio.Task | None = None
        self.callbacks: dict[object, Callable[[], None]] = {}
        self.unsubscribers: list[Callable[[], None]] = []
        self.prefixes: dict[int, list[tuple]] = {}
        self.current_turn = None
        self._reading = False
        self._entering = False
        self._entry_finished = asyncio.Event()
        self._finalizing = False

    async def _enter(self) -> None:
        run = self.parent
        first = run.session is None
        self._entering = True
        try:
            blocks = await run._open_for_control()
            if self.stop_requested or run.stopped or run._released.is_set():
                return
            self.business_outcome = await run._suspend_foreground()
            if self.business_outcome != "stopped":
                raise RuntimeError("The original native foreground did not confirm suspension")
            if self.stop_requested or run.stopped or run._released.is_set():
                return
            for batch in reversed(run.next_prompts):
                run.task.settle(batch, "not_applied")
            run.next_prompts = []
            run.task.refresh_history(unconfirmed_calls=tuple(held.call_id for held in run.forwarded))
            self.saved = _BusinessForeground.capture(run)
            retained = frozenset(held.call_id for held in run.forwarded)
            self.binding = run.door.borrow_foreground(self.task, run._suspend_pause, retained)
            run._borrowed = self
            run.task, run.sources = self.task, ()
            run.shown, run.forwarded, run.app_queue = [], [], deque()
            run.placement_refusals, run.running, run.cards_open, run.runtime_calls = [], set(), set(), {}
            run.next_prompts, run.prompt_inputs, run.open_prompts = [], {}, set()
            run.done = run.answered = run.waits = run.suspended = False
            run._suspend_pause, run._suspend_outputs = None, {}
            run.pending = run.boundary = run._input_recovery = None
            run._new_round()
            self.entered = True
            self.unsubscribers.append(run.door.on_call_settled(self.check_ready))
            self.unsubscribers.append(run.on_quiescent(self._retired))
            self._prompt(first=first, blocks=blocks)
        finally:
            self._entry_finished.set()
            self.business_ready.set()
            self.check_ready()

    def _prompt(self, *, first: bool = False, blocks: tuple[str, ...] = ()) -> None:
        run = self.parent
        batches: list[Steering] = []
        try:
            run._collect(batches)
            extra = [item for batch in batches for item in batch.items]
            mode = mode_item(self.task)
            required = [*extra, mode]
            history = self.task.history(extra=required)
            state = self.task.backend_state(required)
            if first:
                prefix = keys(history[: -(len(required) + len(state))])
                text = run.host.renderer.worker_input(history)
            else:
                text = run.host.renderer.transcript([*extra, *state, mode])
                prefix = []
            text = "\n\n".join(part for part in (*blocks, self.task.texts.control_worker_note, text) if part)
            scheduled = tuple(batches)
            batches.clear()
            run._prompt(text, scheduled)
            self.prefixes[run.turn.main] = prefix
        finally:
            for batch in reversed(batches):
                self.task.settle(batch, "not_applied")

    def prompt_started(self) -> None:
        """Watch the exact control invocation for terminal evidence."""
        run = self.parent
        self.current_turn = run.turn
        self.unsubscribers.append(run.session.on_prompt_terminal(run.turn.main, lambda _: self.check_ready()))

    def dispatched(self, event: PromptDispatched) -> None:
        """Retain the history prefix sent with this prompt."""
        self.parent.shown += self.prefixes.pop(event.prompt, [])

    async def business_suspension(self) -> StopOutcome:
        """Return the original business foreground's suspension outcome."""
        await self.business_ready.wait()
        return self.business_outcome

    async def events(self) -> AsyncIterator[dict]:
        """Read control events while the lease owns the foreground."""
        if self.finished or self.stop_requested:
            return
        if not self.entered:
            await self._enter()
        if not self.entered or self.stop_requested:
            return
        self._reading = True
        try:
            async for event in self.parent.events():
                yield event
        finally:
            self._reading = False
            self.check_ready()

    def steer(self) -> None:
        """Wake the reader to deliver pending input to the control invocation."""
        if self.finished or self.stop_requested:
            return
        self.steer_requested = True
        self.parent.mailbox.put_nowait(Wake())

    async def resume(self) -> None:
        """Reject application outputs on the control lease."""
        raise RuntimeError("A borrowed control foreground does not own the original application's completion")

    async def suspend(self) -> StopOutcome:
        """End the control lease before another foreground takes over."""
        return await self.stop()

    async def stop(self) -> StopOutcome:
        """Fence control calls and await terminal and settlement evidence within the stop bound."""
        self.stop_requested = True
        if self.finished:
            return "stopped"
        if not self.entered:
            if self._entering and not self._entry_finished.is_set():
                try:
                    async with asyncio.timeout(self.parent.settings.stop_timeout_seconds):
                        await self._entry_finished.wait()
                except TimeoutError:
                    return "unconfirmed"
            if self.entered:
                return await self.stop()
            self._mark_ready()
            return "stopped"
        if self.stop_owner is None:
            self.stop_owner = asyncio.create_task(self._stop(), name="native-control-stop")
        return await asyncio.shield(self.stop_owner)

    async def _stop(self) -> StopOutcome:
        run = self.parent
        await run._suspend_foreground()
        self.check_ready()
        return "stopped" if self.finished else "unconfirmed"

    def _positive_terminal(self) -> bool:
        run, turn = self.parent, self.current_turn
        if turn is None:
            return True
        if run.session is None or not run.session.quiescent:
            return False
        terminal = run.session.prompt_state(turn.main).terminal
        if terminal is None:
            return False
        if terminal.dispatch == "not_dispatched":
            return True
        if not terminal.terminal_confirmed:
            return False
        return turn.reusable_completion(run.agent.completion(terminal))

    def check_ready(self) -> None:
        """Restore the original foreground once the control invocation has settled."""
        run = self.parent
        if not self.stop_requested or self.finished or self._finalizing:
            return
        if not self.entered:
            if self._entry_finished.is_set():
                self._mark_ready()
            return
        if self._reading:
            return
        if not run._reading_finished.is_set() or not self._positive_terminal():
            return
        if self.current_turn is not None and run.door.calls_pending_for(self.current_turn):
            return
        self._finalizing = True
        try:
            run._drain_recovery_events()
            if self.current_turn is not None and run.door.calls_pending_for(self.current_turn):
                return
            if run.running or run.runtime_calls:
                return
            for batch in reversed(run.next_prompts):
                self.task.settle(batch, "not_applied")
            run.next_prompts = []
            # The parent's same-task resumption compares full revision keys, not a pooled ancestry anchor.
            # Generated items the caller never committed cannot suppress a different canonical item.
            seen = list(run.shown)
            run.door.return_foreground(self.binding)
            self.saved.restore(run, seen)
            run._borrowed = None
            self._mark_ready()
        finally:
            self._finalizing = False

    def _mark_ready(self) -> None:
        self.finished = True
        for unsubscribe in self.unsubscribers:
            unsubscribe()
        self.unsubscribers.clear()
        for token in tuple(self.callbacks):

            def notify(selected=token):
                callback = self.callbacks.pop(selected, None)
                if callback is not None:
                    callback()

            asyncio.get_running_loop().call_soon(notify)

    def _retired(self) -> None:
        # Whole-owner retirement is a fallback proof after an ambiguous foreground. It never resumes business.
        self._mark_ready()

    def on_quiescent(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Subscribe to the release of the control foreground."""
        token = object()
        self.callbacks[token] = callback
        if self.finished:

            def notify():
                selected = self.callbacks.pop(token, None)
                if selected is not None:
                    selected()

            asyncio.get_running_loop().call_soon(notify)
        return lambda: self.callbacks.pop(token, None)

    def old_event(self, event: SessionEvent) -> bool:
        """Settle late tool events with the original business owner."""
        saved = self.saved
        if saved is None:
            return False
        if isinstance(event, CallSettled) and event.call_id in saved.running:
            saved.running.discard(event.call_id)
            return True
        if isinstance(event, ToolCardEnd) and event.call_id in saved.cards_open:
            saved.cards_open.discard(event.call_id)
            call = saved.runtime_calls.pop(event.call_id, None)
            if call is not None:
                owner = self.parent._observed_owners.pop(event.call_id, saved.task)
                owner.tool_outcome(event.call_id, event.output, "failed" if event.failed else "done")
            return True
        return False

    async def finish_steer(self) -> AsyncIterator[dict]:
        """Settle the previous control invocation before its next prompt."""
        self.steer_requested = False
        outcome = await self.parent._suspend_foreground(reading=True)
        if outcome != "stopped":
            raise RuntimeError("The native control foreground did not confirm interruption")
        async for event in self.parent._close(None, self.parent._turn_usage(self.parent.turn)):
            yield event
        if self.stop_requested or not self.task.can_dispatch():
            return
        pause = self.parent._suspend_pause
        if not self.parent.door.resume_borrowed(self.binding, pause, self.task):
            raise RuntimeError("The same native control foreground cannot resume its Core tool binding")
        self.parent.suspended = self.parent.done = False
        self.parent._suspend_pause = None
        self._prompt()
