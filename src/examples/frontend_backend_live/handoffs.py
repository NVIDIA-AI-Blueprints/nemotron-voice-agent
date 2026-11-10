# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Handing a decided turn to the session's delegation part, and saying the lead-in the talker wrote for it."""

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
    caller_starts: int

    async def wait_caller_silent(self) -> None:
        """Return once the caller is silent: at once when they are."""


class Acknowledgements:
    """The spoken lead-in of each accepted delegation, said after the part's delay once the caller is silent."""

    def __init__(self, floor: CallerFloor, say: Callable[[str, str], None], *, observe: Callable[..., None]):
        """Say with ``say(delegation_id, text)``; nothing is pending."""
        self.floor, self.say, self.observe = floor, say, observe
        self.pending: dict[str, asyncio.Task] = {}

    async def acknowledge(self, delegation_id: str, text: str, delay_ms: int, *, decided_at: int) -> None:
        """Say ``text`` now or after ``delay_ms``, unless the caller started speaking since ``decided_at``."""
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
        """Drop the delegation's pending acknowledgement, which its result or its end makes moot."""
        task = self.pending.pop(delegation_id, None)
        if task is not None:
            task.cancel()
            self.observe("acknowledgement.dropped", delegation_id=delegation_id)

    def drop_all(self) -> None:
        """Drop every pending acknowledgement, as when the caller speaks or the application appends."""
        for delegation_id in list(self.pending):
            self.drop(delegation_id)


class HandOffs:
    """Hands each decided turn to the session's delegation part, then has its lead-in said."""

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
        self.parked: dict[str, tuple[str | None, str, str, int]] = {}

    def delegate(self, request: str, lead_in: str, *, decided_at: int, call_id: str = "") -> None:
        """Reserve the turn's work-note entry at once; hand it over, or park it for the handler of call ``call_id``."""
        reservation = self.worknote.reserve(request)
        if call_id:
            self.parked[call_id] = (reservation, request, lead_in, decided_at)
        else:
            self._start(reservation, request, lead_in, decided_at)

    async def hand_off(self, call_id: str) -> bool | None:
        """Run the hand-off parked for call ``call_id``; return whether the part took it, ``None`` with none parked."""
        parked = self.parked.pop(call_id, None)
        return None if parked is None else await self._hand_off(*parked)

    def after_response(self) -> None:
        """Hand over, at the response's end, each parked hand-off whose call Pipecat's parse never started.

        Pipecat misses a second call packed into one delta, and a last call with no name. The check runs with
        ``call_soon``: after the first step of each handler Pipecat started (``run_in_parallel``), which takes its
        hand-off before any await.
        """
        if self.parked:
            asyncio.get_running_loop().call_soon(self._unclaimed)

    def _unclaimed(self) -> None:
        for call_id in list(self.parked):
            self._start(*self.parked.pop(call_id))

    def _start(self, reservation: str | None, request: str, lead_in: str, decided_at: int) -> None:
        task = asyncio.create_task(
            self._hand_off(reservation, request, lead_in, decided_at), name="delegation-hand-off"
        )
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _hand_off(self, reservation: str | None, request: str, lead_in: str, decided_at: int) -> bool:
        """Send the heard log's input; confirm the reserved entry and return ``True``, or release it on a refusal."""
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
            return False
        self.worknote.confirm(reservation, delegation_id)
        self.display.started(delegation_id, request)
        self.observe("delegation.handed", delegation_id=delegation_id)
        delay_ms = self.delegation.acknowledgement_delay_ms
        await self.acknowledgements.acknowledge(delegation_id, lead_in.strip(), delay_ms, decided_at=decided_at)
        return True

    def close(self) -> None:
        """Stop the hand-offs it started, and release the entries of those still parked."""
        for task in tuple(self.tasks):
            task.cancel()
        for reservation, *_ in self.parked.values():
            self.worknote.release(reservation)
        self.parked.clear()
