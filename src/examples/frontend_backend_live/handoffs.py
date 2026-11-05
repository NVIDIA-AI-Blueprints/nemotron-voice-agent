# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Handing a decided turn to the session's delegation part, and the lead-in the talker wrote for it.

A ``delegate`` decision reserves its work-note entry before anything awaits, so no decision meanwhile reads "none",
then hands the heard log's input to the delegation part (``gateways.live.ports.DelegationPort``). Accepted, the entry
takes the delegation's id and the display hears of it; refused, the entry goes and the refusal line is said. The
lead-in is said only once the delegation is accepted, so a refused delegation never gets one, as the port requires.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from loguru import logger

if TYPE_CHECKING:
    from examples.frontend_backend_live.display import DisplaySink
    from examples.frontend_backend_live.heard_log import HeardLog
    from examples.frontend_backend_live.worknote import WorkNote
    from gateways.live.ports import DelegationPort


class CallerFloor(Protocol):
    """When the caller speaks: what the lead-in rules read."""

    caller_speaking: bool
    caller_starts: int  # how many times the caller started speaking

    async def wait_caller_silent(self) -> None:
        """Return once the caller is silent: at once when they are."""


class Acknowledgements:
    """The spoken acknowledgement of a delegation: the lead-in the talker wrote in the decision that delegated it.

    The Live API requests an acknowledgement for each delegation and lets its model word it; the talker words it in the
    same decision. A delegation with no lead-in says nothing. The lead-in is said after the part's delay (0: at once),
    once the caller is silent; until then it is pending. It is dropped when the caller started speaking after its
    delegation was decided, since the next decision answers their newer words. A pending one is also dropped when its
    delegation's result arrives or the delegation ends (its own line, or its taker, speaks for it), when the caller
    speaks, or when the application appends commentary or thinking, which drops every delegation's. A lead-in asked for
    is the voice's to say.
    """

    def __init__(self, floor: CallerFloor, say: Callable[[str, str], None], *, observe: Callable[..., None]):
        """Say with ``say(delegation_id, text)``; nothing is pending."""
        self.floor, self.say, self.observe = floor, say, observe
        self.pending: dict[str, asyncio.Task] = {}

    async def acknowledge(self, delegation_id: str, text: str, delay_ms: int, *, decided_at: int) -> None:
        """Acknowledge the delegation with the talker's lead-in ``text``, now or after ``delay_ms``; empty: none.

        ``decided_at`` is the floor's count of the caller's starts when the delegation was decided.
        """
        if not text.strip():
            return
        if self.floor.caller_starts != decided_at:
            self.observe("acknowledgement.dropped", delegation_id=delegation_id)
            return
        if delay_ms <= 0 and not self.floor.caller_speaking:
            self.say(delegation_id, text)
            return
        task = asyncio.create_task(self._later(delegation_id, delay_ms / 1000, text), name="acknowledgement")
        self.pending[delegation_id] = task

    async def _later(self, delegation_id: str, delay: float, text: str) -> None:
        await asyncio.sleep(delay)
        await self.floor.wait_caller_silent()
        self.pending.pop(delegation_id, None)
        self.say(delegation_id, text)

    def drop(self, delegation_id: str) -> None:
        """Drop the delegation's pending acknowledgement."""
        task = self.pending.pop(delegation_id, None)
        if task is not None:
            task.cancel()
            self.observe("acknowledgement.dropped", delegation_id=delegation_id)

    def drop_all(self) -> None:
        """Drop every pending acknowledgement."""
        for delegation_id in list(self.pending):
            self.drop(delegation_id)


class HandOffs:
    """Hands each decided turn to the session's delegation part, then says its lead-in by the rules above."""

    def __init__(
        self,
        delegation: DelegationPort,
        heard_log: HeardLog,
        worknote: WorkNote,
        display: DisplaySink,
        acknowledgements: Acknowledgements,
        *,
        refused: Callable[[], None],
        observe: Callable[..., None],
    ):
        """Wire the parts a hand-off reaches; ``refused()`` says the refusal line."""
        self.delegation, self.heard_log, self.worknote, self.display = delegation, heard_log, worknote, display
        self.acknowledgements, self.refused, self.observe = acknowledgements, refused, observe
        self.tasks: set[asyncio.Task] = set()

    def delegate(self, request: str, lead_in: str, *, decided_at: int) -> None:
        """Hand the turn ``request`` over; nothing awaits before the work note reserves its entry."""
        reservation = self.worknote.reserve(request)
        task = asyncio.create_task(
            self._hand_off(reservation, request, lead_in, decided_at), name="delegation-hand-off"
        )
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _hand_off(self, reservation: str | None, request: str, lead_in: str, decided_at: int) -> None:
        """Send the delegation with the heard log's input; confirm the reserved entry, or release it when refused.

        The input notes an answer that waits to be heard as it is built (:meth:`WorkNote.answer_ready`).
        """
        try:
            delegation_id = await self.heard_log.delegate(self.delegation, unheard=self.worknote.answer_ready())
        except asyncio.CancelledError:
            self.worknote.release(reservation)
            raise
        except Exception as exc:
            self.worknote.release(reservation)
            logger.warning(f"The delegation was refused: {type(exc).__name__}: {exc}")
            self.observe("delegation.refused", error_type=type(exc).__name__)
            self.refused()
            return
        self.worknote.confirm(reservation, delegation_id)
        self.display.started(delegation_id, request)
        self.observe("delegation.handed", delegation_id=delegation_id)
        delay_ms = self.delegation.acknowledgement_delay_ms
        await self.acknowledgements.acknowledge(delegation_id, lead_in.strip(), delay_ms, decided_at=decided_at)

    def close(self) -> None:
        """Stop the hand-offs still running."""
        for task in tuple(self.tasks):
            task.cancel()
