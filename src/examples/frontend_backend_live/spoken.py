# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the voice said: each line, what it told the caller about a delegation, and how much of it played."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


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
    retried: bool = False


@dataclass(frozen=True)
class Cut:
    """An interruption's cut while a line's audio played: its lines settle once the aggregator reports what played."""

    lines: tuple[Line, ...]
    playing: Line


class Settled(Protocol):
    """What the voice does when a line settles."""

    def heard(self, line: Line, cut: bool) -> None:
        """Note that the caller heard ``line``'s ``heard`` words; ``cut``: an interruption cut it there."""

    def not_heard(self, line: Line) -> None:
        """Note that nobody heard ``line``."""

    def say_again(self, line: Line) -> None:
        """Say ``line`` once more: it produced no audio."""


class SpokenLines:
    """The lines of one call in flight, settled by their playout, and what each delegation told."""

    def __init__(self, settled: Settled, *, reports_sentences: bool = True, observe: Callable[..., None]):
        """Report each settled line to ``settled``; ``reports_sentences``: each sentence's text follows its audio."""
        self.settled, self.reports_sentences, self.observe = settled, reports_sentences, observe
        self.lines: dict[int, Line] = {}
        self.told: dict[str, list[Line]] = {}
        self.marks: deque[int] = deque()
        self.cutting: Cut | None = None
        self.dropping: int | None = None
        self.ended_line: Line | None = None

    def handed(self, line: Line) -> None:
        """Note that ``line`` went to speech as response ``line.number``."""
        self.lines[line.number] = line
        self.observe("line.handed", line=line.number, line_kind=line.kind.value)

    def in_flight(self) -> bool:
        """Return whether a line is handed and not settled."""
        return bool(self.lines) or self.cutting is not None

    def told_text(self, delegation_id: str) -> str | None:
        """Return what the caller heard about a delegation, then what the lines in flight tell; ``None`` for nothing.

        A line in flight counts whole, so a wording made while it plays still holds once it has played.
        """
        lines = [*(self.cutting.lines if self.cutting else ()), *self.lines.values()]
        telling = [line.text for line in lines if delegation_id in line.tells]
        heard = [line.heard for line in self.told.get(delegation_id, ())]
        return " ".join([*heard, *telling]) or None

    def forget(self, delegation_id: str) -> None:
        """Forget what the caller heard about a delegation: worded into its result, or it ended."""
        self.told.pop(delegation_id, None)

    def carry_told(self, delegation_id: str, taker: str) -> None:
        """Move what the caller heard about a delegation to the front of ``taker``'s, which took over its work."""
        earlier = self.told.pop(delegation_id, None)
        if earlier:
            self.told[taker] = earlier + self.told.get(taker, [])

    def interrupted(self, newest: int) -> None:
        """Note an interruption that passed the talker after response ``newest``."""
        self.marks.append(newest)

    def reached(self, number: int) -> None:
        """Note response ``number`` at the playout's head; a cut the aggregator never reported settles, nothing played.

        The aggregator reports no cut when the interruption overtook the start of the cut line's turn.
        """
        self.ended_line = self.dropping = None
        if self.cutting is not None:
            self.cut_played("")

    def started(self, number: int) -> None:
        """Note response ``number``'s first audio."""
        line = self.lines.get(number)
        if line is not None:
            line.outcome = Outcome.PLAYING

    def dropped(self, number: int) -> None:
        """Note that a sentence of response ``number`` got no audio: its line is heard as the aggregator records it."""
        self.dropping = number

    def ended(self, number: int) -> None:
        """Settle response ``number`` at its end: heard when it played, else silent."""
        line = self.lines.get(number)
        if line is None:
            return
        if line.outcome is Outcome.PLAYING:
            line.heard = line.text
            if number == self.dropping:
                self.ended_line = line
            self._settle(line, Outcome.HEARD)
        else:
            self._silent(line)

    def end_played(self, played: str) -> Line | None:
        """Return the last ended line with a silent sentence, set as heard up to ``played``; ``None`` with no words."""
        line, self.ended_line = self.ended_line, None
        if line is None or not played:
            return None
        line.heard = played
        return line

    def cut(self, number: int | None) -> None:
        """Drop the lines the talker started before the interruption that reached the output.

        ``number`` heads the playout; when its audio played, the lines settle once the aggregator reports what played.
        """
        mark = self.marks.popleft() if self.marks else max(self.lines, default=0)
        lines = tuple(self.lines.pop(each) for each in list(self.lines) if each <= mark)
        playing = next((line for line in lines if line.number == number and line.outcome is Outcome.PLAYING), None)
        if playing is None:
            self._settle_cut(lines, None, "")
        else:
            self.cutting = Cut(lines, playing)

    def cut_played(self, played: str) -> str:
        """Settle the waiting cut with ``played``, the words the aggregator recorded; return what the caller heard.

        A stitched line with no words recorded was heard whole, since its words come only at its end.
        """
        cut, self.cutting = self.cutting, None
        if cut is None:
            return played
        heard = played or ("" if self.reports_sentences else cut.playing.text)
        self._settle_cut(cut.lines, cut.playing, heard)
        return heard

    def _settle_cut(self, lines: Sequence[Line], playing: Line | None, heard: str) -> None:
        """Settle a cut's lines in order: ``playing`` cut after ``heard``, when any; the others not heard."""
        for line in lines:
            if line is playing and heard:
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
            self._settle(line, Outcome.NOT_HEARD)

    def _settle(self, line: Line, outcome: Outcome) -> None:
        self.lines.pop(line.number, None)
        line.outcome = outcome
        self.observe("line.settled", line=line.number, line_kind=line.kind.value, outcome=outcome.value)
        if outcome is Outcome.NOT_HEARD:
            self.settled.not_heard(line)
            return
        cut = outcome is Outcome.CUT
        for delegation_id in line.tells:
            self.told.setdefault(delegation_id, []).append(line)
        self.settled.heard(line, cut=cut)
