# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The floor: when the voice may start a line that answers no caller turn.

A reply or a lead-in goes out at once. Everything else the voice says on its own waits, so the caller comes first, then
the direct answer, then the rest: ``may_prepare`` lets the presenter word what is ready, and ``idle`` lets it speak.
There is no time cap: a turn with no words still ends at the turn detection's stop timeout.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable


class Floor:
    """Who has the floor in one call."""

    def __init__(self, clock: Callable[[], float]):
        """Start idle: the caller is silent, no turn waits, nothing is in flight."""
        self.clock = clock
        self.caller_speaking = False
        self.turns_busy = False
        self.in_flight = 0
        self.idle_since = clock()  # when the floor last became idle
        self._changed = asyncio.Event()

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
        self._moved(was_idle)

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

    async def wait_may_prepare(self) -> None:
        """Return once the voice may prepare a line."""
        while not self.may_prepare:
            await self._changed.wait()

    async def wait_idle(self) -> None:
        """Return once the voice may start a line."""
        while not self.idle:
            await self._changed.wait()

    async def wait_change(self) -> None:
        """Return at the next change of any fact."""
        await self._changed.wait()

    def _moved(self, was_idle: bool) -> None:
        if self.idle and not was_idle:
            self.idle_since = self.clock()
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()
