# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the voice said: each line, what it told the caller about a delegation, and how much of it played.

Every line is one response of the talker, numbered by the talker (``examples.frontend_backend_live.talker``). The
assistant aggregator after the output writes the words that played into the talker's context, and reports them as each
turn ends; the edge before it (``examples.frontend_backend_live.media.silent_lines.SilentLines``) reports to
:class:`SpokenLines`, in playout order, what the aggregator cannot tell: whether each response's audio played, and
where an interruption cut the playout. So the voice knows what each line did:

* a line heard, even in part, tells the caller about its delegations (``told``): its words, or what played of a cut
  line or of one a sentence of which got no audio; a line counts whole until the aggregator reports what played. What
  a line heard delivered is not said again;
* a line nobody heard gives back what it delivered, to be said next;
* a line that produced no audio is said once more at its end, which the text-to-speech lets out once the line's audio
  context ends: at the latest when it has had nothing for the service's stop timeout (Pipecat's 3 s).

A line an interruption cut is never said again word for word: the talker's context keeps it as far as it played, with
the cut mark, and the talker decides what to say next.

An interruption drops every response the talker started before it passed the talker and not played yet: the talker
notes the newest response then (:meth:`SpokenLines.interrupted`), and the edge's cut settles those (:meth:`cut`). When
the line at the head of the playout had played audio, they settle as the aggregator reports what of it played
(:meth:`cut_played`).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class LineKind(StrEnum):
    """What a line is, for the traces: what a line does follows from what it carries, never from its kind."""

    REPLY = "reply"  # the talker's reply to a caller turn
    LEAD_IN = "lead_in"  # a delegation's acknowledgement
    RESULT = "result"  # ready results, worded into one line
    COMMENTARY = "commentary"  # the application's commentary, or the greeting, worded
    STATUS = "status"  # a progress line about open work
    NOTICE = "notice"  # a fixed line: an ending's, the refusal, the app-wait line, the holding line


class Outcome(StrEnum):
    """Where a line stands."""

    HANDED = "handed"  # handed to speech, not started
    PLAYING = "playing"
    HEARD = "heard"
    CUT = "cut"  # cut after some words played
    NOT_HEARD = "not_heard"


@dataclass(eq=False)
class Line:
    """One line the voice said: one response of the talker."""

    kind: LineKind
    text: str
    delegation_ids: tuple[str, ...] = ()
    items: tuple[Any, ...] = ()  # what a relay delivered with it: given back when nobody heard it
    results: bool = False  # it says backend results, which the backend holds in its own history
    tells: tuple[str, ...] = ()  # the delegations it tells the caller about: a lead-in, a progress line
    again: bool = True  # may be said once more when it produced no audio
    number: int = 0  # the talker's response that says it
    outcome: Outcome = Outcome.HANDED
    heard: str = ""  # what the caller heard of it: the words the assistant aggregator recorded, or the line whole
    retried: bool = False  # said once more after it produced no audio


@dataclass(frozen=True)
class Cut:
    """An interruption's cut while a line's audio played: its lines settle once the aggregator reports what played."""

    lines: tuple[Line, ...]  # every line the interruption dropped, in the order they were handed
    playing: Line  # the line whose audio played


class Settled(Protocol):
    """What the voice does when a line settles."""

    def heard(self, line: Line, cut: bool) -> None:
        """The caller heard ``line``'s ``heard`` words; ``cut``: an interruption cut it there."""

    def not_heard(self, line: Line) -> None:
        """Nobody heard ``line``."""

    def say_again(self, line: Line) -> None:
        """``line`` produced no audio: say it once more."""


class SpokenLines:
    """The lines of one call in flight, settled by their playout, and what each delegation told."""

    def __init__(self, settled: Settled, *, reports_sentences: bool = True, observe: Callable[..., None]):
        """Report each settled line to ``settled``.

        ``reports_sentences``: each sentence's text follows its own audio (per-sentence speech); without it, a line's
        text comes only at its end, so a line cut while it played was heard, whole.
        """
        self.settled, self.reports_sentences, self.observe = settled, reports_sentences, observe
        self.lines: dict[int, Line] = {}  # the lines in flight, by response, in the order they were handed
        self.told: dict[str, list[Line]] = {}  # by delegation: the lines the caller heard about it, in order
        self.marks: deque[int] = deque()  # the newest response when each interruption passed the talker
        self.cutting: Cut | None = None  # a cut while a line played, until the aggregator reports what played
        self.dropping: int | None = None  # the response at the head of the playout a sentence of which got no audio
        self.ended_line: Line | None = None  # its line once it ended, until the aggregator reports its words

    # ------------------------------------------------------------------------------ handing
    def handed(self, line: Line) -> None:
        """``line`` went to speech as response ``line.number``."""
        self.lines[line.number] = line
        self.observe("line.handed", line=line.number, line_kind=line.kind.value)

    def in_flight(self) -> bool:
        """Whether a line is handed and not settled."""
        return bool(self.lines) or self.cutting is not None

    # ------------------------------------------------------------------------------ told
    def told_text(self, delegation_id: str) -> str | None:
        """What the caller heard about a delegation so far, then what the lines in flight tell them; or ``None``.

        A line in flight counts whole, as it counts once it is heard in full, so a wording made while it plays (the
        voice's early wording) still holds once it played in full; so does a line that ended, or was cut, until what of
        it played is reported.
        """
        lines = [*(self.cutting.lines if self.cutting else ()), *self.lines.values()]
        telling = [line.text for line in lines if delegation_id in line.tells]
        heard = [line.heard for line in self.told.get(delegation_id, ())]
        return " ".join([*heard, *telling]) or None

    def forget(self, delegation_id: str) -> None:
        """Forget what the caller heard about a delegation: worded into its result, or it ended."""
        self.told.pop(delegation_id, None)

    def carry_told(self, delegation_id: str, taker: str) -> None:
        """A newer delegation took over the delegation's work: what the caller heard goes with it, first."""
        earlier = self.told.pop(delegation_id, None)
        if earlier:
            self.told[taker] = earlier + self.told.get(taker, [])

    # ------------------------------------------------------------------------------ the playout facts
    def interrupted(self, newest: int) -> None:
        """An interruption passed the talker after response ``newest``."""
        self.marks.append(newest)

    def reached(self, number: int) -> None:
        """Response ``number`` heads the playout: a cut the aggregator never reported settles, with nothing played.

        The aggregator reports a cut as the interruption reaches it, long before a later response does; it reports none
        when the interruption overtook the start of the cut line's turn. The words it reports from now on are a later
        line's, so the line that ended before keeps its own (:meth:`end_played`).
        """
        self.ended_line = self.dropping = None
        if self.cutting is not None:
            self.cut_played("")

    def started(self, number: int) -> None:
        """Response ``number``'s first audio played."""
        line = self.lines.get(number)
        if line is not None:
            line.outcome = Outcome.PLAYING

    def dropped(self, number: int) -> None:
        """A sentence of response ``number`` got no audio: its line is heard as the aggregator records it."""
        self.dropping = number

    def ended(self, number: int) -> None:
        """Response ``number`` ended: heard when it played, else silent.

        A line a sentence of which got no audio is heard whole until the aggregator reports the words it played
        (:meth:`end_played`).
        """
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
        """The aggregator reported the words of the line that ended last, a sentence of which got no audio: return it.

        That sentence went no further than the output, so the aggregator recorded only the sentences that played: the
        line was heard as ``played``, and tells the caller that much. With none recorded, it stays heard whole, and
        ``None`` is returned.
        """
        line, self.ended_line = self.ended_line, None
        if line is None or not played:
            return None
        line.heard = played
        return line

    def cut(self, number: int | None) -> None:
        """An interruption reached the output: it drops the lines the talker started before it passed the talker.

        ``number`` is the response at the head of the playout, if any; a line the talker started after the interruption
        is said in its turn. When the head's audio played, the lines settle once the assistant aggregator reports what
        played (:meth:`cut_played`). Otherwise nobody heard any of them, and they settle now.
        """
        mark = self.marks.popleft() if self.marks else max(self.lines, default=0)
        lines = tuple(self.lines.pop(each) for each in list(self.lines) if each <= mark)
        playing = next((line for line in lines if line.number == number and line.outcome is Outcome.PLAYING), None)
        if playing is None:
            self._settle_cut(lines, None, "")
        else:
            self.cutting = Cut(lines, playing)

    def cut_played(self, played: str) -> str:
        """The aggregator reported what the cut line played: the cut's lines settle; return what the caller heard.

        ``played`` is the words the aggregator recorded: the line was heard up to them, and not at all with none, but a
        stitched line with none was heard whole, since its words come only at its end. Nobody heard the other lines.
        With no cut waiting, ``played`` is returned as it is.
        """
        cut, self.cutting = self.cutting, None
        if cut is None:
            return played
        heard = played or ("" if self.reports_sentences else cut.playing.text)
        self._settle_cut(cut.lines, cut.playing, heard)
        return heard

    def _settle_cut(self, lines: Sequence[Line], playing: Line | None, heard: str) -> None:
        """Settle the lines a cut dropped, in order: ``playing`` cut after ``heard``, when any; the others not heard."""
        for line in lines:
            if line is playing and heard:
                line.heard = heard
                self._settle(line, Outcome.CUT)
            else:
                line.heard = ""
                self._settle(line, Outcome.NOT_HEARD)

    # ------------------------------------------------------------------------------ settling
    def _silent(self, line: Line) -> None:
        """``line`` produced no audio: said once more, or not heard."""
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
