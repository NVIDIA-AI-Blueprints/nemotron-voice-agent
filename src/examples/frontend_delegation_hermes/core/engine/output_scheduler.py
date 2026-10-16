# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What to send next, and what never to send over the user (plan section 10.1).

Pure policy, no I/O: the turn manager pushes items and asks :meth:`OutputQueue.next`
whenever the output lane is free.

1. Function-call batches carry no audio: they go first and are never held.
2. Audio is never released while the user is speaking or while the new turn's
   frontend decision is pending (so that turn's filler goes first).
3. ``hold_max_ms`` only marks an item stale; ``on_stale`` decides per kind whether a
   stale item is still spoken (``keep``) or dropped (``drop``, outcome ``not_heard``).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from examples.frontend_delegation_hermes.core.engine.transcript import Entry

#: Speech item kind -> ``output.on_stale`` key.
STALE_KEY = {
    "filler": "filler",
    "direct": "filler",
    "answer": "backend_answer",
    "replay": "backend_answer",
    "status": "status",
    "apology": "apology",
}


@dataclass(slots=True)
class SpeechItem:
    """One assistant message to speak as its own response."""

    kind: str  # filler | direct | answer | replay | status | apology | greeting
    text: str
    entry: Entry | None = None
    turn_id: int | None = None
    usage: dict[str, int] = field(default_factory=dict)
    queued_mono: float = field(default_factory=time.monotonic)
    on_first_audio: Callable[[], None] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CallsItem:
    """One batch of function calls emitted as one response."""

    batch_id: str
    calls: list[tuple[str, str, str]]  # (call_id, name, arguments_json)
    queued_mono: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class Dropped:
    """A speech item dropped instead of spoken."""

    item: SpeechItem
    reason: str


class OutputQueue:
    """FIFO of pending outputs with the hold and stale rules."""

    def __init__(
        self,
        *,
        hold_max_ms: int,
        on_stale: Mapping[str, str],
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind the policy."""
        self._hold_max = hold_max_ms / 1000.0
        self._on_stale = dict(on_stale)
        self._monotonic = monotonic
        self._calls: list[CallsItem] = []
        self._speech: list[SpeechItem] = []

    def __len__(self) -> int:
        """Queued outputs (calls and speech)."""
        return len(self._calls) + len(self._speech)

    def push(self, item: SpeechItem | CallsItem) -> None:
        """Queue one output."""
        if isinstance(item, CallsItem):
            self._calls.append(item)
        else:
            self._speech.append(item)

    def stale(self, item: SpeechItem) -> bool:
        """Whether ``item`` waited longer than ``hold_max_ms``."""
        return item.kind != "greeting" and self._monotonic() - item.queued_mono > self._hold_max

    def next(
        self, *, user_speaking: bool, deciding: bool, current_turn: int | None
    ) -> tuple[SpeechItem | CallsItem | None, list[Dropped]]:
        """The next output that may start now, and the stale items dropped on the way."""
        if self._calls:
            return self._calls.pop(0), []
        if user_speaking or not self._speech:
            return None, []
        dropped: list[Dropped] = []
        while self._speech:
            index = self._pick(deciding=deciding, current_turn=current_turn)
            if index is None:
                return None, dropped
            item = self._speech.pop(index)
            if self.stale(item) and self._on_stale.get(STALE_KEY.get(item.kind, ""), "keep") == "drop":
                dropped.append(Dropped(item, "dropped_stale"))
                continue
            return item, dropped
        return None, dropped

    def _pick(self, *, deciding: bool, current_turn: int | None) -> int | None:
        # The current turn's own filler/direct reply (or replayed answer) goes first.
        for index, item in enumerate(self._speech):
            if item.kind in ("filler", "direct", "replay") and item.turn_id == current_turn:
                return index
        if deciding:
            return None  # hold backend speech until the new turn's decision (and filler) is in
        return 0

    def drain(self) -> list[SpeechItem]:
        """Remove and return every queued speech item (session close); calls are discarded."""
        items, self._speech, self._calls = self._speech, [], []
        return items
