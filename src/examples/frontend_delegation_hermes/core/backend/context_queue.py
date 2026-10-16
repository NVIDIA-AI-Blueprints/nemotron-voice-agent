# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exactly-once, failure-safe delivery of conversation context to Hermes (plan section 7.3).

Every entry the backend has not produced itself (acks, frontend lines, status
replies, apologies, delivery notes, the delegated user words) passes through this
queue. An entry is ``queued`` until the controller renders the next backend
user-side message; it is then ``in_flight`` for that run's epoch and becomes
``committed`` only when the epoch settles with committed history. A rolled-back
epoch puts its entries back at the front, in their original order, ahead of
anything newer. Duplicate voice ``seq`` values are ignored.

Stdlib only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

QUEUED = "queued"
IN_FLIGHT = "in_flight"
COMMITTED = "committed"

#: Entry kinds that never travel to Hermes when the user did not hear them.
_SPOKEN_KINDS = frozenset({"frontend_speech", "status_speech", "controller_speech"})


@dataclass(slots=True)
class ContextEntry:
    """One piece of context for the backend."""

    seq: int | None
    kind: str
    text: str
    origin: str = "user"
    outcome: str = "heard"
    heard_text: str = ""
    answer_id: str | None = None
    during_run: bool = False
    #: ``True`` for the delegated user words of a run (rendered as the request, not in the block).
    run_input: bool = False
    #: Free-form extras (for example the calls of a side-effect note).
    extra: dict[str, Any] = field(default_factory=dict)
    state: str = QUEUED
    epoch: int | None = None
    #: The steer (or run start) that carried the entry, so a ``pending_steer`` can move it.
    carrier: str | None = None

    @classmethod
    def from_wire(cls, data: Mapping[str, Any], *, during_run: bool) -> ContextEntry:
        """Build from a ``history.append`` entry."""
        return cls(
            seq=int(data["seq"]),
            kind=str(data.get("kind") or "user"),
            text=str(data.get("text") or ""),
            origin=str(data.get("origin") or "user"),
            outcome=str(data.get("outcome") or "heard"),
            heard_text=str(data.get("heard_text") or ""),
            answer_id=data.get("answer_id"),
            during_run=during_run,
        )

    @property
    def silent(self) -> bool:
        """Whether the entry carries nothing for Hermes (a spoken line nobody heard)."""
        if self.kind in _SPOKEN_KINDS:
            return self.outcome == "not_heard" or not (self.heard_text if self.outcome == "partial" else self.text)
        if self.kind == "delivery_note":
            return self.outcome == "heard"
        return False


class ContextQueue:
    """Ordered context entries with delivery states."""

    def __init__(self) -> None:
        """Start empty."""
        self._entries: list[ContextEntry] = []
        self._seen: set[int] = set()
        self._committed_seqs: set[int] = set()
        self._committed_upto = -1

    # -- intake -----------------------------------------------------------------

    def add(self, entry: ContextEntry) -> bool:
        """Append a new entry; ``False`` for a duplicate ``seq``."""
        if entry.seq is not None:
            if entry.seq in self._seen:
                return False
            self._seen.add(entry.seq)
        entry.state, entry.epoch = QUEUED, None
        self._entries.append(entry)
        return True

    def add_run_input(self, entries: Iterable[ContextEntry], epoch: int, carrier: str) -> list[ContextEntry]:
        """Register a run's delegated user words, already in flight for ``epoch``; duplicates are skipped."""
        accepted = []
        for entry in entries:
            if entry.seq is not None and entry.seq in self._seen:
                continue
            if entry.seq is not None:
                self._seen.add(entry.seq)
            entry.run_input = True
            entry.state, entry.epoch, entry.carrier = IN_FLIGHT, epoch, carrier
            self._entries.append(entry)
            accepted.append(entry)
        return accepted

    # -- delivery -----------------------------------------------------------------

    def take(self, epoch: int, carrier: str) -> list[ContextEntry]:
        """All queued entries in order, now in flight for ``epoch``; silent ones commit on the spot."""
        taken: list[ContextEntry] = []
        for entry in self._entries:
            if entry.state != QUEUED:
                continue
            if entry.silent:
                self._commit_entry(entry)
                continue
            entry.state, entry.epoch, entry.carrier = IN_FLIGHT, epoch, carrier
            taken.append(entry)
        self._advance_watermark()
        return taken

    def in_flight(self, epoch: int, carrier: str | None = None) -> list[ContextEntry]:
        """Entries in flight for ``epoch`` (optionally only those of one carrier)."""
        return [
            e
            for e in self._entries
            if e.state == IN_FLIGHT and e.epoch == epoch and (carrier is None or e.carrier == carrier)
        ]

    def move(self, old_epoch: int, new_epoch: int, carriers: Iterable[str]) -> int:
        """Move in-flight entries of ``carriers`` to ``new_epoch`` (a ``pending_steer`` re-run)."""
        wanted = set(carriers)
        moved = 0
        for entry in self._entries:
            if entry.state == IN_FLIGHT and entry.epoch == old_epoch and entry.carrier in wanted:
                entry.epoch = new_epoch
                moved += 1
        return moved

    def requeue_carrier(self, epoch: int, carrier: str) -> list[ContextEntry]:
        """Undo one rejected steer: its context goes back to the front, its user words are returned.

        The returned run-input entries are no longer tracked (their ``seq`` is released),
        so the caller can register them again as the input of a new run or as context.
        """
        mine = [e for e in self._entries if e.state == IN_FLIGHT and e.epoch == epoch and e.carrier == carrier]
        words = [e for e in mine if e.run_input]
        back = [e for e in mine if not e.run_input]
        mine_ids = {id(e) for e in mine}
        rest = [e for e in self._entries if id(e) not in mine_ids]
        for entry in words:
            if entry.seq is not None:
                self._seen.discard(entry.seq)
            entry.state, entry.epoch, entry.carrier, entry.run_input = QUEUED, None, None, False
        for entry in back:
            entry.state, entry.epoch, entry.carrier = QUEUED, None, None
        self._entries = [*back, *rest]
        return words

    def commit(self, epoch: int) -> int:
        """Commit every entry in flight for ``epoch``; returns the new acknowledgement watermark."""
        for entry in self._entries:
            if entry.state == IN_FLIGHT and entry.epoch == epoch:
                self._commit_entry(entry)
        self._advance_watermark()
        self._compact()
        return self._committed_upto

    def requeue(self, epoch: int, notes: Iterable[ContextEntry] = ()) -> list[ContextEntry]:
        """Put ``epoch``'s in-flight entries back at the front in original order, after ``notes``.

        Delegated user words of the rolled-back run come back as ``requeued_request``
        entries, so the next message says they were asked for but not completed.
        """
        returning = [e for e in self._entries if e.state == IN_FLIGHT and e.epoch == epoch]
        if not returning and not notes:
            return []
        rest = [e for e in self._entries if not (e.state == IN_FLIGHT and e.epoch == epoch)]
        for entry in returning:
            entry.state, entry.epoch, entry.carrier = QUEUED, None, None
            if entry.run_input:
                entry.run_input = False
                entry.kind = "requeued_request"
        front = [*notes, *returning]
        for note in notes:
            note.state, note.epoch = QUEUED, None
        committed = [e for e in rest if e.state == COMMITTED]
        pending = [e for e in rest if e.state != COMMITTED]
        self._entries = [*committed, *front, *pending]
        return front

    # -- introspection ----------------------------------------------------------------

    @property
    def committed_upto(self) -> int:
        """Highest voice ``seq`` such that every seen ``seq`` up to it is committed (-1 = none)."""
        return self._committed_upto

    def queued(self) -> list[ContextEntry]:
        """Entries waiting for the next backend message."""
        return [e for e in self._entries if e.state == QUEUED]

    def entries(self) -> list[ContextEntry]:
        """Every entry still tracked (committed ones are compacted away)."""
        return list(self._entries)

    # -- internals -----------------------------------------------------------------

    def _commit_entry(self, entry: ContextEntry) -> None:
        entry.state = COMMITTED
        if entry.seq is not None:
            self._committed_seqs.add(entry.seq)

    def _advance_watermark(self) -> None:
        mark = self._committed_upto
        while (mark + 1) in self._committed_seqs:
            mark += 1
        # Seqs never seen by this queue (for example the voice side's own entries that never
        # travel) must not block the watermark: skip gaps below the smallest uncommitted seq.
        uncommitted = [s for s in self._seen if s not in self._committed_seqs]
        floor = min(uncommitted) - 1 if uncommitted else (max(self._seen) if self._seen else mark)
        self._committed_upto = max(mark, floor)

    def _compact(self) -> None:
        self._entries = [e for e in self._entries if e.state != COMMITTED]
