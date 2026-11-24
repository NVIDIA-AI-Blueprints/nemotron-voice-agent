# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pause, resume and stop confirmation for one logical task."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING

from loguru import logger

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    BackendRun,
    StopOutcome,
    ToolNotExecuted,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.stop import (
    ForNewerRequest,
    StopReason,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.tools import (
    Invocation,
    WorkStateChange,
)

if TYPE_CHECKING:
    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.run import TaskRun


class TaskLifecycle:
    """Own task suspension and the bounded confirmation of native settlement."""

    def __init__(self, task: TaskRun):
        """Initialize pause and stop state for the logical task."""
        self.task = task
        self.want_paused = False
        self.suspension_active = False
        self.pause_outcome: StopOutcome | None = None
        self.stop_reason: StopReason | None = None
        self.stopped: StopOutcome | None = None
        self.stop_quiescent = False
        self.own_cancel = False
        self._pause_owner: asyncio.Task | None = None
        self._confirmation: asyncio.Task | None = None
        self._quiescence_subscriptions: list[Callable[[], None]] = []

    def observe_quiescence(self, unsubscribe: Callable[[], None]) -> None:
        """Register cleanup for a native settlement subscription."""
        self._quiescence_subscriptions.append(unsubscribe)

    def request_pause(self) -> bool:
        """Fence immediately; a separate owner confirms actual invocation and tool quiescence."""
        task = self.task
        if task.ended or self.stop_reason is not None:
            raise ValueError("An ended or stopping task cannot be paused")
        changed = not self.want_paused
        self.want_paused = self.suspension_active = True
        task.refuse_unpublished_app("paused", task.texts.paused_output)
        if task.record.state not in ("pausing", "paused"):
            task.move("pausing")
        task.session.unschedule(task)
        if task.grant is not None and not task.grant.done():
            task.grant.set_result(None)
        if self._pause_owner is None:
            self._pause_owner = task.session.spawn(self._confirm_pause())
        task.wake.set()
        task.work.execution_changed()
        return changed

    async def _confirm_pause(self) -> StopOutcome:
        task = self.task
        outcome: StopOutcome = "stopped"
        try:
            if task.run is not None:
                async with asyncio.timeout(task.session.settings.stop_timeout_seconds):
                    outcome = await task.run.suspend()
        except Exception as exc:
            outcome = "unconfirmed"
            task.trace("backend.pause_unconfirmed", error_type=type(exc).__name__)
        self.pause_outcome = outcome
        if self.stop_reason is None and not task.ended:
            if outcome == "stopped":
                if task.control.run is None:
                    task.session.release(task)
                if task.record.state != "paused":
                    task.move("paused")
            task.work.execution_changed()
        task.wake.set()
        return outcome

    async def wait_paused(self) -> StopOutcome:
        """Join the current bounded confirmation without claiming an external action was cancelled."""
        task = self.task
        if not self.suspension_active:
            raise ValueError("No suspension was requested")
        if self._pause_owner is None:
            self._pause_owner = task.session.spawn(self._confirm_pause())
        return await asyncio.shield(self._pause_owner)

    def request_resume(self) -> bool:
        """Authorize resumption; the driver still requires actual settlement and a new inference permit."""
        task = self.task
        if task.ended or self.stop_reason is not None:
            raise ValueError("An ended or stopping task cannot resume")
        changed = self.want_paused
        self.want_paused = False
        task.wake.set()
        return changed

    def set_work_state(self, invocation: Invocation) -> str:
        """Interpret lifecycle meaning in the worker, enforce its effects in Core."""
        task = self.task
        state = WorkStateChange.model_validate(invocation.arguments).state
        if task.ended or self.stop_reason is not None or self.own_cancel:
            raise ToolNotExecuted("set_work_state: task has ended or is stopping")
        changed = self.request_pause() if state == "paused" else self.request_resume()
        return json.dumps({"requested": state, "state": task.record.state, "changed": changed})

    async def await_resumption(self) -> None:
        """Keep a paused task addressable, processing controls and genuine settlement wakes only."""
        task = self.task

        def resume_ready() -> bool:
            return (
                not self.want_paused
                and self.pause_outcome == "stopped"
                and task.control.run is None
                and not task.control.unconfirmed
            )

        def evidence() -> tuple:
            return (
                self.want_paused,
                task.presentation.input_revision,
                tuple(waiting.record.id for waiting in task.steering),
                deepcopy(task.outputs),
                frozenset(task.reported),
            )

        observed = evidence()
        await self.wait_paused()
        while self.stop_reason is None:
            if task.control.ready():
                await task.control.process_input()
                continue
            if resume_ready():
                self.suspension_active = False
                self.pause_outcome = None
                self._pause_owner = None
                task.move("queued")
                task.work.execution_changed()
                return
            current = evidence()
            if self.pause_outcome == "unconfirmed" and current != observed:
                observed = current
                self._pause_owner = task.session.spawn(self._confirm_pause())
                await self.wait_paused()
                continue
            observed = current
            task.wake.clear()
            if task.control.ready() or resume_ready() or evidence() != observed:
                continue
            await task.wake.wait()

    def request_stop(self, reason: StopReason) -> bool:
        """Ask the task to stop for ``reason``; ``False`` when it had ended or a stop was asked already."""
        task = self.task
        if task.ended or self.stop_reason is not None:
            return False
        self.stop_reason = reason
        task.refuse_unpublished_app("stopped", task.texts.stopped_output)
        self.want_paused = False
        task.wake.set()
        task.presentation.drop_all("task_stopping")
        task.presentation.close_questions("cancelled")
        if task.record.stop.state == "none":
            task.record.stop.move("stop_requested")
            task.report_state("stopping")
        task.session.unschedule(task)
        task.session.book.drop(task.unreported())
        return True

    async def confirm_stop(self) -> StopOutcome:
        """Join the one stop confirmation owner, including repeated concurrent stop requests."""
        task = self.task
        if self._confirmation is None:
            self._confirmation = asyncio.create_task(self._confirm_stop(), name=f"stop-{task.task_id}")
        return await asyncio.shield(self._confirmation)

    async def _confirm_stop(self) -> StopOutcome:
        """Stop the run within its existing deadline, then settle the logical task."""
        task = self.task
        outcome: StopOutcome = "stopped"
        runs = tuple({id(run): run for run in (task.control.run, task.run) if run is not None}.values())
        try:
            async with asyncio.timeout(task.session.settings.stop_timeout_seconds):
                results = await asyncio.gather(*(run.stop() for run in runs), return_exceptions=True)
                if any(result != "stopped" for result in results):
                    outcome = "unconfirmed"
        except Exception as exc:
            logger.warning(f"Task {task.task_id} did not confirm its stop: {type(exc).__name__}: {exc}")
            outcome = "unconfirmed"
        self.stopped = outcome
        task.record.stop.move(outcome)
        if task.driver is not None and task.driver is not asyncio.current_task() and not task.driver.done():
            task.driver.cancel()
            await asyncio.gather(task.driver, return_exceptions=True)
        task.work.stop_answer(self.stop_reason.answer(task.session.texts, outcome))
        task.end()
        if outcome == "unconfirmed":
            self._observe_stopped_runs(runs)
        return outcome

    def _observe_stopped_runs(self, runs: Sequence[BackendRun]) -> None:
        """Release quarantined capacity only on later positive proof, without repeating cancellation or effects."""
        task = self.task
        pending = {id(run) for run in runs}

        def settled(run: BackendRun) -> None:
            pending.discard(id(run))
            if pending or self.stop_quiescent:
                return
            self.stop_quiescent = task.record.stop.quiescent_later = True
            task.control.unconfirmed = False
            task.session.release(task)
            task.trace("backend.stop_later_confirmed", initial_result=self.stopped)
            task.report_state("cancelled")
            task.wake.set()
            self.clear_quiescence_observers()

        for run in runs:
            observe = getattr(run, "on_quiescent", None)
            if callable(observe):
                self.observe_quiescence(observe(lambda owner=run: settled(owner)))
        if not runs:
            self.stop_quiescent = task.record.stop.quiescent_later = True
            task.session.release(task)

    def clear_quiescence_observers(self) -> None:
        """Detach observation callbacks when resource ownership or the whole session has ended."""
        callbacks, self._quiescence_subscriptions = self._quiescence_subscriptions, []
        for unsubscribe in callbacks:
            unsubscribe()

    def cancel_own(self) -> str:
        """Fence immediately and use the common stop door from a separate owner task."""
        task = self.task
        self.own_cancel = True
        task.work.stop_soon(ForNewerRequest(), task)
        return task.session.texts.outcomes["cancel_requested"].format(work=task.work_id, tasks=task.task_id)
