# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the voice said: each line, what it told the caller about a delegation, and how much of it played."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol
from weakref import WeakKeyDictionary


def transcript_results(results: Iterable[dict]) -> list[dict]:
    """Omit private task references from playback transcripts."""
    return [{key: value for key, value in result.items() if key != "task_ref"} for result in results]


class LineKind(StrEnum):
    """What a line is, for the traces: what a line does follows from what it carries, never from its kind."""

    REPLY = "reply"
    LEAD_IN = "lead_in"
    RESULT = "result"
    COMMENTARY = "commentary"
    STATUS = "status"
    NOTICE = "notice"


class Outcome(StrEnum):
    """Where a line stands."""

    HANDED = "handed"
    PLAYING = "playing"
    HEARD = "heard"
    SYNTHESIS_FAILED = "synthesis_failed"
    CUT = "cut"
    NOT_HEARD = "not_heard"


@dataclass(eq=False)
class Line:
    """One line the voice said: one response of the talker."""

    kind: LineKind
    text: str
    delegation_ids: tuple[str, ...] = ()
    items: tuple[Any, ...] = ()
    results: bool = False
    tells: tuple[str, ...] = ()
    again: bool = True
    number: int = 0
    outcome: Outcome = Outcome.HANDED
    heard: str = ""
    audio_started: bool = False
    retried: bool = False
    work_refs: frozenset[str] = field(default_factory=frozenset, kw_only=True)
    work_tells: frozenset[str] = field(default_factory=frozenset, kw_only=True)
    publication_items: tuple[dict, ...] = field(default=(), kw_only=True)
    handoff_request: tuple[tuple[int, str], ...] = field(default=(), kw_only=True)


@dataclass(frozen=True)
class Cut:
    """An interruption's cut while a line's audio played: its lines settle once the aggregator reports what played."""

    lines: tuple[Line, ...]
    playing: Line


class Settled(Protocol):
    """What the voice does when a line settles."""

    def heard(self, line: Line) -> None:
        """Note that the caller heard ``line``'s recorded words."""

    def not_heard(self, line: Line, *, audio_exhausted: bool = False) -> None:
        """Note that nobody heard ``line`` and whether its audio attempts finished silently."""

    def say_again(self, line: Line) -> None:
        """Say ``line`` once more: it produced no audio."""


class SpokenLines:
    """The lines of one call in flight, settled by their playout, and what each delegation told."""

    def __init__(self, settled: Settled, *, observe: Callable[..., None]):
        """Report each settled line to ``settled``."""
        self.settled, self.observe = settled, observe
        self.lines: dict[int, Line] = {}
        self.told: dict[str, list[Line]] = {}
        self.work_told: dict[str, list[Line]] = {}
        self._work_cutoffs: dict[str, int] = {}
        self._delegation_cutoffs: dict[str, int] = {}
        self._bindings: dict[str, set[str]] = {}
        self._line_work: WeakKeyDictionary[Line, set[str]] = WeakKeyDictionary()
        self._orders: WeakKeyDictionary[Line, int] = WeakKeyDictionary()
        self._sequence = 0
        self.cuts: dict[int, Cut] = {}
        self.ended_lines: dict[int, Line] = {}
        self.presentations: deque[Line] = deque(maxlen=64)

    def handed(self, line: Line) -> None:
        """Note that ``line`` went to speech as response ``line.number``."""
        self.register(line)
        self.lines[line.number] = line
        self.observe("line.handed", line=line.number, line_kind=line.kind.value)

    def in_flight(self) -> bool:
        """Return whether a line is handed and not settled."""
        return bool(self.lines or self.ended_lines or self.cuts)

    def in_flight_lines(self) -> Iterator[Line]:
        """Return lines whose playback evidence has not settled."""
        yield from self.lines.values()
        yield from self.ended_lines.values()
        for cut in self.cuts.values():
            yield from cut.lines

    def export_attachment(self, flights: list[Line]) -> tuple[dict, Iterator[tuple[int, dict]]]:
        """Keep scope cutoffs with playback rows, ordered by retention priority."""
        return {
            "presentation_sequence": self._sequence,
            "presentation_cutoffs": dict(self._work_cutoffs),
            "delegation_cutoffs": dict(self._delegation_cutoffs),
        }, self._attachment_rows(flights)

    def _attachment_rows(self, flights: list[Line]) -> Iterator[tuple[int, dict]]:
        recent = dict.fromkeys([*self.presentations, *flights])
        work_index: dict[Line, list[str]] = {}
        delegation_index: dict[Line, list[str]] = {}
        for reference, lines in self.work_told.items():
            for line in lines:
                if self._eligible(line, reference):
                    work_index.setdefault(line, []).append(reference)
        for reference, lines in self.told.items():
            for line in lines:
                if self.delegation_eligible(line, reference):
                    delegation_index.setdefault(line, []).append(reference)
        indexed = sorted(dict.fromkeys([*work_index, *delegation_index]), key=self._orders.__getitem__)
        older = [line for line in indexed if line not in recent]
        order = {line: index for index, line in enumerate([*older, *recent])}
        for line in dict.fromkeys([*reversed(flights), *reversed(indexed), *reversed(recent)]):
            row = self.presentation(line)
            if line not in recent:
                row["indexed_only"] = True
            if line in work_index:
                row["work_told"] = sorted(work_index[line])
            told = sorted(delegation_index.get(line, ()))
            if told != row["context_delegation_ids"]:
                row["told"] = told
            yield order[line], row

    def restore_attachment(self, snapshot: dict, work_refs: Iterable[str]) -> None:
        """Restore playback evidence after work ownership is known, without resuming audio."""
        self._sequence = max(self._sequence, snapshot.get("presentation_sequence", 0))
        self._work_cutoffs.update(snapshot.get("presentation_cutoffs", {}))
        self._delegation_cutoffs.update(snapshot.get("delegation_cutoffs", {}))
        work_refs = frozenset(work_refs)
        for row in snapshot.get("presentations", ()):
            refs = work_refs.intersection(row.get("context_work_refs", row["work_refs"]))
            line = Line(
                LineKind(row.get("kind", "result")),
                row["text"],
                delegation_ids=tuple(row.get("delegation_ids", ())),
                number=row["line_id"],
                outcome=Outcome(row["state"]),
                heard=row["recorded_sentence_prefix"],
                audio_started=row["audio_started"],
                work_refs=frozenset(row["work_refs"]),
                publication_items=tuple(row.get("results", ())),
            )
            self._orders[line] = row.get("sequence", 0)
            self._line_work[line] = set(refs)
            if not row.get("indexed_only"):
                self.presentations.append(line)
            for reference in row.get("work_told", ()):
                if reference in refs and self._eligible(line, reference):
                    self.work_told.setdefault(reference, []).append(line)
            for reference in row.get("told", row.get("context_delegation_ids", row.get("delegation_ids", ()))):
                self.told.setdefault(reference, []).append(line)

    def told_text(self, delegation_id: str) -> str | None:
        """Return presentation state and recorded words for a delegation, if any."""
        return self.context_text((delegation_id,))

    def heard_handoff(
        self, text: str, delegation_id: str, branch_ref: str | None, request: tuple[tuple[int, str], ...]
    ) -> bool:
        """Match an entire played handoff for the same request and work scope."""
        if not request:
            return False
        refs = (branch_ref,) if branch_ref is not None else ()
        work_refs = frozenset(refs)
        return any(
            line.kind is LineKind.LEAD_IN
            and line.delegation_ids == (delegation_id,)
            and self._work_refs(line) == work_refs
            and line.handoff_request == request
            and line.outcome is Outcome.HEARD
            and line.heard == line.text == text
            for line in self._context_lines(() if refs else (delegation_id,), refs)
        )

    def register(self, line: Line) -> None:
        """Remember creation order before a queued line can be superseded; do not alter its playback state."""
        if line not in self._orders:
            self._sequence += 1
            self._orders[line] = self._sequence
        if not line.work_tells:
            for delegation_id in line.tells:
                self._line_work.setdefault(line, set()).update(self._bindings.get(delegation_id, ()))

    def bind_work(self, branch_ref: str, delegation_id: str) -> None:
        """Associate actual unscoped lead-ins with a host-known branch, without rewriting the Line."""
        self._bindings.setdefault(delegation_id, set()).add(branch_ref)
        for line in tuple(self._orders):
            if not line.work_tells and delegation_id in line.tells:
                self._line_work.setdefault(line, set()).add(branch_ref)
                if line.heard and self._eligible(line, branch_ref):
                    rows = self.work_told.setdefault(branch_ref, [])
                    if line not in rows:
                        rows.append(line)
        self._prune_work_scopes()

    def _prune_work_scopes(self) -> None:
        """Keep cutoff guards only while a remembered line or active association can still reference them."""
        reachable = set(self.work_told)
        for refs in self._bindings.values():
            reachable.update(refs)
        for line in self._orders:
            reachable.update(self._work_refs(line))
        self._work_cutoffs = {ref: cutoff for ref, cutoff in self._work_cutoffs.items() if ref in reachable}
        sources = set(self.told)
        for line in self._orders:
            sources.update((*line.tells, *line.delegation_ids))
        self._delegation_cutoffs = {ref: cutoff for ref, cutoff in self._delegation_cutoffs.items() if ref in sources}

    def inherit_work(self, branch_ref: str, predecessor: str) -> None:
        """Reference actual predecessor lines for the same input obligation, without a transfer graph or replay."""
        if branch_ref == predecessor:
            return
        for line in tuple(self._orders):
            if (
                predecessor in self._work_refs(line)
                and self._eligible(line, predecessor)
                and self._eligible(line, branch_ref)
            ):
                self._line_work.setdefault(line, set()).add(branch_ref)
                if line.heard:
                    rows = self.work_told.setdefault(branch_ref, [])
                    if line not in rows:
                        rows.append(line)

    def _work_refs(self, line: Line) -> frozenset[str]:
        return line.work_refs | line.work_tells | frozenset(self._line_work.get(line, ()))

    def _eligible(self, line: Line, branch_ref: str) -> bool:
        return self._orders.get(line, 0) > self._work_cutoffs.get(branch_ref, 0)

    def delegation_eligible(self, line: Line, delegation_id: str) -> bool:
        """Keep delegation evidence only while it survives both request and branch consumption."""
        refs = self._work_refs(line)
        return self._orders.get(line, 0) > self._delegation_cutoffs.get(delegation_id, 0) and (
            not refs or any(self._eligible(line, ref) for ref in refs)
        )

    def context_text(
        self,
        delegation_ids: Sequence[str] = (),
        work_refs: Sequence[str] = (),
    ) -> str | None:
        """Render each line's playback state and recorded words for the Writer."""
        rows = []
        for line in self._context_lines(delegation_ids, work_refs):
            row = self.presentation(line)
            del row["text"]
            row["results"] = transcript_results(row["results"])
            rows.append(row)
        return json.dumps(rows, ensure_ascii=False) if rows else None

    def _context_lines(self, delegation_ids: Sequence[str], work_refs: Sequence[str]) -> tuple[Line, ...]:
        selected: set[Line] = set()
        candidates = [*self.presentations, *self.in_flight_lines()]
        for delegation_id in delegation_ids:
            candidates.extend(self.told.get(delegation_id, ()))
        for reference in work_refs:
            candidates.extend(self.work_told.get(reference, ()))
        for line in candidates:
            branch_match = any(ref in self._work_refs(line) and self._eligible(line, ref) for ref in work_refs)
            source_match = any(
                (ref in (*line.tells, *line.delegation_ids) or line in self.told.get(ref, ()))
                and self.delegation_eligible(line, ref)
                for ref in delegation_ids
            )
            if branch_match or source_match:
                selected.add(line)
        return tuple(sorted(selected, key=lambda item: self._orders.get(item, 0)))

    def context_stamp(self, delegation_ids: Sequence[str], work_refs: Sequence[str]) -> tuple:
        """Invalidate wording for material playback loss, not normal queued-to-completed advancement."""
        loss = {Outcome.CUT, Outcome.NOT_HEARD, Outcome.SYNTHESIS_FAILED}
        return tuple(
            (
                line.number,
                line.text,
                line.heard
                if line.outcome in loss or line.outcome is Outcome.HEARD and line.heard != line.text
                else None,
                line.outcome in loss,
            )
            for line in self._context_lines(delegation_ids, work_refs)
        )

    def presentation(self, line: Line) -> dict:
        """Preserve one original publication without guessing words for individual result branches."""
        results = []
        for item in line.items:
            if not hasattr(item, "said"):
                continue
            result = {"delegation_id": item.said.delegation_id, "item_id": item.said.item_id}
            if item.said.attribution is not None:
                result["branch_ref"] = item.said.attribution.branch_ref
                if item.said.attribution.task_ref is not None:
                    result["task_ref"] = item.said.attribution.task_ref
            results.append(result)
        return {
            "line_id": line.number,
            "sequence": self._orders.get(line, 0),
            "kind": line.kind.value,
            "state": line.outcome.value,
            "text": line.text,
            "recorded_sentence_prefix": line.heard,
            "audio_started": line.audio_started,
            "complete_conveyance": line.outcome is Outcome.HEARD and line.heard == line.text,
            "work_refs": sorted(line.work_refs),
            "context_work_refs": sorted(self._work_refs(line)),
            "delegation_ids": list(line.delegation_ids),
            "context_delegation_ids": sorted(
                set(line.delegation_ids) | {reference for reference, rows in self.told.items() if line in rows}
            ),
            "results": results or list(line.publication_items),
        }

    def forget_work(self, branch_ref: str) -> None:
        """Consume only this branch's prior context; in-flight old progress cannot reappear when it settles."""
        self._work_cutoffs[branch_ref] = self._sequence
        self.work_told.pop(branch_ref, None)
        for delegation_id, lines in tuple(self.told.items()):
            retained = [line for line in lines if self._work_refs(line) != frozenset((branch_ref,))]
            if retained:
                self.told[delegation_id] = retained
            else:
                self.told.pop(delegation_id, None)
        for delegation_id, refs in tuple(self._bindings.items()):
            refs.discard(branch_ref)
            if not refs:
                self._bindings.pop(delegation_id)
        self._prune_work_scopes()

    def forget(self, delegation_id: str) -> None:
        """Forget what the caller heard about a delegation: worded into its result, or it ended."""
        self.told.pop(delegation_id, None)
        self._delegation_cutoffs[delegation_id] = self._sequence

    def carry_told(self, delegation_id: str, taker: str) -> None:
        """Move what the caller heard about a delegation to the front of ``taker``'s, which took over its work."""
        earlier = list(self.told.pop(delegation_id, ()))
        for line in self.in_flight_lines():
            if delegation_id in (*line.tells, *line.delegation_ids) and line not in earlier:
                earlier.append(line)
        if earlier:
            self.told[taker] = earlier + [line for line in self.told.get(taker, ()) if line not in earlier]

    def started(self, number: int) -> None:
        """Note response ``number``'s first audio."""
        line = self.lines.get(number)
        if line is not None:
            line.outcome = Outcome.PLAYING
            line.audio_started = True
            note = getattr(self.settled, "audio_started", None)
            if callable(note):
                note(line)

    def ended(self, number: int, *, synthesis_failed: bool = False) -> bool:
        """Return whether this response awaits the aggregator's words, settling any silent line."""
        line = self.lines.get(number)
        if line is None:
            return False
        if line.outcome is Outcome.PLAYING:
            if synthesis_failed:
                line.outcome = Outcome.SYNTHESIS_FAILED
            self.ended_lines[number] = self.lines.pop(number)
            return True
        self._silent(line)
        return False

    def end_played(self, number: int, played: str) -> None:
        """Settle an exact ended line with the aggregator's text for that response."""
        line = self.ended_lines.pop(number, None)
        if line is not None:
            line.heard = played
            outcome = Outcome.SYNTHESIS_FAILED if line.outcome is Outcome.SYNTHESIS_FAILED else Outcome.HEARD
            self._settle(line, outcome)

    def cut(self, number: int | None, interruption: int, *, newest: int | None = None) -> None:
        """Drop the lines the talker started before the interruption that reached the output.

        ``number`` heads the playout; when its audio played, the lines settle once the aggregator reports what played.
        """
        mark = newest if newest is not None else max(self.lines, default=0)
        lines = tuple(self.lines.pop(each) for each in list(self.lines) if each <= mark)
        playing = next((line for line in lines if line.number == number and line.outcome is Outcome.PLAYING), None)
        if playing is None:
            self._settle_cut(lines, None, "")
        else:
            self.cuts[interruption] = Cut(lines, playing)

    def cut_played(self, interruption: int, played: str) -> bool:
        """Settle the exact cut with only its recorded words and return whether it was pending."""
        cut = self.cuts.pop(interruption, None)
        if cut is None:
            return False
        self._settle_cut(cut.lines, cut.playing, played)
        return True

    def _settle_cut(self, lines: Sequence[Line], playing: Line | None, heard: str) -> None:
        """Settle a cut's lines in order: ``playing`` cut after ``heard``, when any; the others not heard."""
        for line in lines:
            if line is playing:
                line.heard = heard
                self._settle(line, Outcome.CUT)
            else:
                line.heard = ""
                self._settle(line, Outcome.NOT_HEARD)

    def _silent(self, line: Line) -> None:
        """Say ``line``, which produced no audio, once more, or settle it as not heard."""
        if line.again and not line.retried:
            line.retried, line.outcome = True, Outcome.HANDED
            self.lines.pop(line.number, None)
            self.observe("line.retried", line=line.number)
            self.settled.say_again(line)
        else:
            self._settle(line, Outcome.NOT_HEARD, audio_exhausted=True)

    def _settle(self, line: Line, outcome: Outcome, *, audio_exhausted: bool = False) -> None:
        self.lines.pop(line.number, None)
        line.outcome = outcome
        if line not in self.presentations:
            self.presentations.append(line)
        self.observe(
            "line.settled",
            line=line.number,
            line_kind=line.kind.value,
            outcome=outcome.value,
            audio_started=line.audio_started,
            recorded_sentence_prefix=line.heard,
        )
        if outcome is Outcome.NOT_HEARD:
            self.settled.not_heard(line, audio_exhausted=audio_exhausted)
            return
        for delegation_id in line.tells:
            self.told.setdefault(delegation_id, []).append(line)
        for branch_ref in self._work_refs(line):
            if self._eligible(line, branch_ref):
                self.work_told.setdefault(branch_ref, []).append(line)
        self._prune_work_scopes()
        self.settled.heard(line)
