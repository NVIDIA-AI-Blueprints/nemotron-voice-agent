# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the voice said: each line, what it told the caller about a delegation, and how much of it played.

A line an interruption cut is not said again whole: the voice may say its unheard sentences once, as written
(:func:`unheard_rest`).
"""

from __future__ import annotations

import unicodedata
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from pipecat.utils.string import match_endofsentence
from pipecat.utils.text.alnum_utils import alnum_only

from examples.shared import delegation_contract


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
    # The cut line whose unheard sentences this line says: it gets no rest of its own.
    rest_of: Line | None = None


@dataclass(frozen=True)
class Cut:
    """An interruption's cut while a line's audio played: its lines settle once the aggregator reports what played."""

    lines: tuple[Line, ...]
    playing: Line


def sentences(text: str, *, language: str | None = None) -> list[str]:
    """``text`` split into sentences the way the text-to-speech cuts them.

    The shared rule runs with Pipecat's sentence-end test in ``language``. Each sentence is stripped; a last part with
    no sentence end is a sentence too.
    """
    return delegation_contract.sentences(text, ends=lambda rest: match_endofsentence(rest, language=language))


def unheard_rest(line: Line, *, language: str | None = None) -> str:
    """The sentences of ``line.text`` the caller did not hear, as written and in order, joined by a space; or "".

    A sentence is heard when its letters and digits (``alnum_only``) appear in ``line.heard`` after those of the heard
    sentences before it, starting where a heard sentence starts, so a short sentence is never found inside a longer
    one. The sentence that was playing at a cut is not in ``heard``, so it counts as unheard; a sentence missing from
    the middle (one that got no audio) is found too.
    """
    if line.heard == line.text:
        return ""
    heard, found, rest = alnum_only(line.heard), 0, []
    starts, offset = {0}, 0
    for part in sentences(line.heard, language=language):
        offset += len(alnum_only(part))
        starts.add(offset)
    for sentence in sentences(line.text, language=language):
        letters = alnum_only(sentence)
        if not letters:
            continue
        at = heard.find(letters, found)
        while at >= 0 and at != found and at not in starts:
            at = heard.find(letters, at + 1)
        if at < 0:
            rest.append(sentence)
        else:
            found = at + len(letters)
    return " ".join(rest)


def _spelled_tokens(text: str) -> list[str]:
    """Each token of ``text``: itself when it is spelled, else "".

    Tokens split on whitespace, with leading and trailing punctuation stripped and empty ones left out. A spelled token
    is one digit (``Nd``) or one letter of a script with letter case (``Lu``, ``Ll``).
    """
    tokens = []
    for raw in text.split():
        start, end = 0, len(raw)
        while start < end and unicodedata.category(raw[start]).startswith("P"):
            start += 1
        while end > start and unicodedata.category(raw[end - 1]).startswith("P"):
            end -= 1
        if start < end:
            token = raw[start:end]
            tokens.append(token if len(token) == 1 and unicodedata.category(token) in ("Nd", "Lu", "Ll") else "")
    return tokens


def spelled_runs(text: str) -> list[str]:
    """Each run of two or more single-character tokens in ``text``, joined as heard.

    A token is one digit, or one letter of a script with letter case, once its leading and trailing punctuation is
    stripped. ``"last name e L L I S"`` gives ``["eLLIS"]``; a text with none gives ``[]``. Never raises.
    """
    runs, run = [], ""
    for token in [*_spelled_tokens(text), ""]:
        if token:
            run += token
            continue
        if len(run) >= 2:
            runs.append(run)
        run = ""
    return runs


def ends_spelled(text: str) -> bool:
    """Whether ``text`` ends with such a run, so the caller may still be spelling."""
    tokens = _spelled_tokens(text)
    return len(tokens) >= 2 and bool(tokens[-1]) and bool(tokens[-2])


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

    def __init__(
        self,
        settled: Settled,
        *,
        reports_sentences: bool = True,
        clear_ended_on_cut: bool = False,
        requeued_as_said: bool = False,
        observe: Callable[..., None],
    ):
        """Report each settled line to ``settled``; ``reports_sentences``: each sentence's text follows its audio.

        ``clear_ended_on_cut``: a cut also lets go of a line that ended waiting for its words (``ended_line``), so one
        whose words never came cannot read as unsettled after its turn. ``requeued_as_said``: the lines a cut drops
        unheard settle newest first, so their items, each put back at the front of what is owed, keep their order.
        """
        self.settled, self.reports_sentences, self.observe = settled, reports_sentences, observe
        self.clear_ended_on_cut, self.requeued_as_said = clear_ended_on_cut, requeued_as_said
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

    def unsettled(self) -> tuple[Line, ...]:
        """Return the lines not settled yet, in order.

        A cut's lines waiting for what played, the lines in flight, and the line that ended waiting for the words the
        aggregator recorded (a sentence of it got no audio).
        """
        cutting = self.cutting.lines if self.cutting is not None else ()
        ended = (self.ended_line,) if self.ended_line is not None else ()
        return (*cutting, *self.lines.values(), *ended)

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
        if self.clear_ended_on_cut:
            self.ended_line = None  # its words will not come after this cut: it stays heard whole
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
        """Settle a cut's lines in order: ``playing`` cut after ``heard``, when any; the others not heard.

        With ``requeued_as_said`` the cut line settles first, then the others newest first: each not-heard line's items
        go to the front of what is owed, so they come back in the order the lines were handed.
        """
        if self.requeued_as_said:
            cut = [line for line in lines if line is playing and heard]
            lines = [*cut, *reversed([line for line in lines if not (line is playing and heard)])]
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
