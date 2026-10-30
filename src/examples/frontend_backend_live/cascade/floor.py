# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The floor: when the voice may start a line.

A reply or a lead-in waits only for the caller: it never starts while they speak (:meth:`Floor.wait_caller_silent`).
Everything else the voice says on its own waits for ``idle``, so the caller comes first, then the direct answer, then
the rest: the presenter words what is ready at once, and ``idle`` lets it speak. There is no time cap: a turn with no
words still ends at the turn detection's stop timeout. The floor counts the caller's starts, so a decision and a lead-in
can tell whether the caller spoke again since the turns they answer, and it calls back once the caller is silent
(:meth:`Floor.when_caller_silent`), so the turns of a decision dropped for that are decided again with no new turn.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any


class Floor:
    """Who has the floor in one call."""

    def __init__(self, clock: Callable[[], float]):
        """Start idle: the caller is silent, no turn waits, nothing is in flight."""
        self.clock = clock
        self.caller_speaking = False
        self.caller_starts = 0  # how many times the caller started speaking
        self.turns_busy = False
        self.in_flight = 0
        self.idle_since = clock()  # when the floor last became idle
        self._changed = asyncio.Event()
        self._silent: list[Callable[[], None]] = []  # called when the caller next stops speaking

    @property
    def may_prepare(self) -> bool:
        """Whether the caller is silent and no turn waits or is being decided."""
        return not self.caller_speaking and not self.turns_busy

    @property
    def idle(self) -> bool:
        """Whether the voice may start a line now: it may prepare, and nothing is in flight."""
        return self.may_prepare and self.in_flight == 0

    def quiet_for(self) -> float:
        """Seconds the floor has been idle, or 0 while it is not."""
        return self.clock() - self.idle_since if self.idle else 0.0

    def set_caller_speaking(self, speaking: bool) -> None:
        """The caller started or stopped speaking."""
        was_idle = self.idle
        self.caller_speaking = speaking
        if speaking:
            self.caller_starts += 1
        self._moved(was_idle)
        if not speaking:
            calls, self._silent = self._silent, []
            for call in calls:
                call()

    def set_turns_busy(self, busy: bool) -> None:
        """A caller turn waits or is being decided, or none."""
        was_idle = self.idle
        self.turns_busy = busy
        self._moved(was_idle)

    def line_handed(self) -> None:
        """A line was handed to speech."""
        was_idle = self.idle
        self.in_flight += 1
        self._moved(was_idle)

    def line_settled(self) -> None:
        """A line in flight was heard, cut or not heard."""
        was_idle = self.idle
        self.in_flight = max(0, self.in_flight - 1)
        self._moved(was_idle)

    def wait_change(self) -> Coroutine[Any, Any, bool]:
        """Return at the next change of any fact after this call, even one made before the wait first runs."""
        return self._changed.wait()

    async def wait_caller_silent(self) -> None:
        """Return once the caller is silent: at once when they are."""
        while self.caller_speaking:
            await self.wait_change()

    def when_caller_silent(self, call: Callable[[], None]) -> None:
        """Call ``call`` once the caller is silent: at once when they are, else when they stop speaking."""
        if self.caller_speaking:
            self._silent.append(call)
        else:
            call()

    def _moved(self, was_idle: bool) -> None:
        if self.idle and not was_idle:
            self.idle_since = self.clock()
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()
