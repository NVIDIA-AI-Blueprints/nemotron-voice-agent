# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Separate ownership of one backend run from the Talker calls that wait on it.

A run is one backend task. A caller is one ``call_backend`` invocation. A
caller never awaits the task itself: it waits on its own future, so a caller
that is cancelled or displaced cannot cancel work that another caller now
owns. Only the current owner receives the run's outcome.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from contextvars import ContextVar
from typing import Any

ProgressCallback = Callable[[Any], Awaitable[None]]

_CURRENT_RUN_ID: ContextVar[str | None] = ContextVar("frontend_backend_run_id", default=None)


def current_run_id() -> str | None:
    """Return the run that owns the calling task, if any."""
    return _CURRENT_RUN_ID.get()


class DelegationRun:
    """One backend task and the single caller that currently owns its outcome."""

    def __init__(
        self,
        run_id: str,
        query: str,
        work: Callable[[ProgressCallback], Coroutine[Any, Any, dict[str, Any]]],
        on_progress: ProgressCallback | None = None,
    ) -> None:
        """Start ``work`` as a task bound to ``run_id``."""
        self.run_id = run_id
        self.query = query
        self._progress = on_progress
        self._owner: asyncio.Future[dict[str, Any]] | None = None
        token = _CURRENT_RUN_ID.set(run_id)
        try:
            self.task: asyncio.Task[dict[str, Any]] = asyncio.create_task(work(self.progress))
        finally:
            _CURRENT_RUN_ID.reset(token)
        self.task.add_done_callback(self._settle)

    @property
    def running(self) -> bool:
        """Return whether the task is still working."""
        return not self.task.done()

    def attach(self) -> asyncio.Future[dict[str, Any]]:
        """Make a fresh caller the owner and return the future it waits on."""
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._owner = future
        if self.task.done():
            self._settle(self.task)
        return future

    def adopt(self, on_progress: ProgressCallback | None) -> asyncio.Future[dict[str, Any]]:
        """Hand the run to a new caller; the previous owner ends at once as superseded."""
        previous = self._owner
        self._progress = on_progress
        future = self.attach()
        if previous is not None and not previous.done():
            previous.cancel()
        return future

    def owns(self, future: asyncio.Future[dict[str, Any]]) -> bool:
        """Return whether ``future`` belongs to the current owner."""
        return self._owner is future

    async def progress(self, event: Any) -> None:
        """Forward one lifecycle event to whichever caller owns the run now."""
        callback = self._progress
        if callback is not None:
            await callback(event)

    async def wait(self, future: asyncio.Future[dict[str, Any]]) -> dict[str, Any]:
        """Wait for this caller's outcome without exposing the task to its cancellation.

        When the owning caller itself is cancelled (for example at pipeline
        shutdown), the run is cancelled too, which matches a caller awaiting the
        task directly. A displaced caller's cancellation never touches the task.
        """
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling() > 0 and self.owns(future):
                self.task.cancel()
            raise

    def _settle(self, task: asyncio.Task[dict[str, Any]]) -> None:
        exception: BaseException | None = None
        if not task.cancelled():
            # Retrieve the exception even when nobody owns the outcome any more,
            # so a displaced run never logs "exception was never retrieved".
            exception = task.exception()
        owner = self._owner
        if owner is None or owner.done():
            return
        if task.cancelled():
            owner.cancel()
        elif exception is not None:
            owner.set_exception(exception)
        else:
            owner.set_result(task.result())
