# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the voice said: each line, what it told the caller about a delegation, and how much of it played.

Every line is one response of the talker, numbered by the talker (``examples.frontend_backend_live.talker``). The
assistant aggregator after the output writes what played into the talker's context; the edge before it
(``examples.frontend_backend_live.media.silent_lines.SilentLines``) reports each response's playout to
:class:`SpokenLines`, in playout order, so the voice knows what each line did:

* a line heard, even in part, tells the caller about its delegations (``told``): its words, or what played of a cut
  line; a line in flight counts whole until it settles. What a line heard delivered is not said again;
* a line nobody heard gives back what it delivered, to be said next;
* a line that produced no audio is said once more (``speech.retry_silent_lines``), as is one whose audio has not
  started ``speech.start_timeout_seconds`` after it headed the playout;
* the newest line an interruption cut keeps its unheard rest, which a turn with no words says once, or the turn whose
  words cut it when the talker waits on that turn, unless a line came since or an instruction cut it on purpose.

An interruption drops every response the talker started before it passed the talker and not played yet: the talker
notes the newest response then (:meth:`SpokenLines.interrupted`), and the edge's cut settles those (:meth:`cut`).

With ``speech.heard_words: estimated``, the talker's context keeps more of a line cut while it played: the words its
audio likely carried past what played (:class:`EstimatedHeard`); the heard log and ``told`` keep what played.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from examples.frontend_backend_live.config_manager.schema import SpeechConfig


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
    again: bool = True  # may be said again: its unheard rest resumed, or once more when silent
    number: int = 0  # the talker's response that says it
    outcome: Outcome = Outcome.HANDED
    heard: str = ""  # the words that played
    retried: bool = False  # said once more after it produced no audio


class CutMemory(Protocol):
    """What the talker's context keeps of a line cut while it played; the other records keep what played.

    ``speech.heard_words`` names one of :data:`CUT_MEMORIES`.
    """

    @classmethod
    def from_settings(cls, speech: SpeechConfig) -> CutMemory:
        """The cut memory as the speech settings set it up, for one call."""

    def played(self, text: str, audio_ms: int) -> None:
        """A sentence of the call played, ``audio_ms`` of audio after the one before it."""

    def kept(self, line: Line, played: str, heard_ms: int, speaking: str) -> str:
        """The words of ``line`` the context keeps once ``played`` played, then ``heard_ms`` of ``speaking``'s audio."""


class AsRecorded:
    """``as_recorded``: the talker's context keeps of a cut line what the other records keep."""

    @classmethod
    def from_settings(cls, speech: SpeechConfig) -> AsRecorded:
        """Nothing to set up: no setting changes what the other records keep."""
        return cls()

    def played(self, text: str, audio_ms: int) -> None:
        """Nothing to measure."""

    def kept(self, line: Line, played: str, heard_ms: int, speaking: str) -> str:
        """The words that played, as the other records keep them: a stitched line cut while it played, whole."""
        return played or line.heard


class EstimatedHeard:
    """``estimated``: the context keeps the words that played, then the words of the rest the audio likely carried.

    The rest is the line's text past the words that played, at most as many words as ``speaking`` holds: the text the
    audio belongs to, the unfinished sentence with per-sentence speech and the whole line with stitched speech. With
    no ``speaking`` named, the rest runs to the line's end. The context keeps as many of its whole words as fit in the
    characters that ``heard_ms`` of audio carries at the call's speaking rate: the characters per millisecond of the
    sentences that played, or ``chars_per_second`` until one has.
    """

    def __init__(self, chars_per_second: float):
        """Take the speaking rate assumed until a sentence of the call has played."""
        self.assumed = chars_per_second / 1000  # characters per millisecond
        self.chars = self.ms = 0  # the characters of the call's played sentences, and their audio

    @classmethod
    def from_settings(cls, speech: SpeechConfig) -> EstimatedHeard:
        """The estimate at ``speech.estimate_chars_per_second`` until a sentence of the call has played."""
        return cls(speech.estimate_chars_per_second)

    def played(self, text: str, audio_ms: int) -> None:
        """Count a played sentence into the call's speaking rate."""
        self.chars += len(text)
        self.ms += audio_ms

    def kept(self, line: Line, played: str, heard_ms: int, speaking: str) -> str:
        """``played``, then the whole words of the rest that ``heard_ms`` of audio carried; never part of a word."""
        carried = heard_ms * (self.chars / self.ms if self.ms else self.assumed)
        rest = line.text.split()[len(played.split()) :]
        heard, size = [], -1
        for word in rest[: len(speaking.split()) or len(rest)]:
            size += 1 + len(word)  # the words so far, with the spaces between them
            if size > carried:
                break
            heard.append(word)
        return " ".join([played, *heard]).strip()


# What the talker's context keeps of a cut line, by the name ``speech.heard_words`` gives.
CUT_MEMORIES: dict[str, type[CutMemory]] = {"as_recorded": AsRecorded, "estimated": EstimatedHeard}


class Settled(Protocol):
    """What the voice does when a line settles."""

    def heard(self, line: Line, cut: bool) -> None:
        """The caller heard ``line``'s ``heard`` words; ``cut``: an interruption cut it there."""

    def not_heard(self, line: Line) -> None:
        """Nobody heard ``line``."""

    def say_again(self, line: Line) -> None:
        """``line`` produced no audio: say it once more."""


class SpokenLines:
    """The lines of one call in flight, settled by their playout; what each delegation told; the line to resume."""

    def __init__(
        self,
        settled: Settled,
        *,
        cut_memory: CutMemory,
        start_timeout_seconds: float,
        retry_silent_lines: bool,
        reports_sentences: bool = True,
        observe: Callable[..., None],
    ):
        """Report each settled line to ``settled``; keep of a cut line what ``cut_memory`` makes of it.

        ``reports_sentences``: each sentence's text follows its own audio (per-sentence speech); without it, a line's
        text comes only at its end, so a line cut while it played was heard, whole.
        """
        self.settled, self.cut_memory, self.observe = settled, cut_memory, observe
        self.reports_sentences = reports_sentences
        self.start_timeout_seconds, self.retry_silent_lines = start_timeout_seconds, retry_silent_lines
        self.lines: dict[int, Line] = {}  # the lines in flight, by response, in the order they were handed
        self.told: dict[str, list[str]] = {}  # by delegation: what the caller heard about it, in order
        self.marks: deque[int] = deque()  # the newest response when each interruption passed the talker
        self.last_cut: Line | None = None  # the newest line an interruption cut, until a later line or a resume
        self.spoken_over: list[Line] = []  # lines the caller's words cut, until a later line
        self.resumes = True  # whether a cut may be resumed: not after an instruction's, until the next interruption
        self._cut: tuple[Line, int, str] | None = None  # the newest cut line, its audio past what played, and its text
        self.head: int | None = None  # the response at the head of the playout
        self._timer: asyncio.TimerHandle | None = None
        self._watched: Line | None = None

    # ------------------------------------------------------------------------------ handing
    def handed(self, line: Line) -> None:
        """``line`` went to speech as response ``line.number``; a line since a cut leaves nothing to resume."""
        self.lines[line.number] = line
        self.last_cut = None
        self.spoken_over.clear()
        self.observe("line.handed", line=line.number, line_kind=line.kind.value)
        if line.number == self.head:
            self._watch(line)

    def in_flight(self) -> bool:
        """Whether a line is handed and not settled."""
        return bool(self.lines)

    # ------------------------------------------------------------------------------ told
    def told_text(self, delegation_id: str) -> str | None:
        """What the caller heard about a delegation so far, then what the lines in flight tell them; or ``None``.

        A line in flight counts whole, as it counts once it is heard in full, so a wording made while it plays (the
        voice's early wording) still holds once it played.
        """
        telling = [line.text for line in self.lines.values() if delegation_id in line.tells]
        return " ".join([*self.told.get(delegation_id, ()), *telling]) or None

    def forget(self, delegation_id: str) -> None:
        """Forget what the caller heard about a delegation: worded into its result, or it ended."""
        self.told.pop(delegation_id, None)

    def carry_told(self, delegation_id: str, taker: str) -> None:
        """A newer delegation took over the delegation's work: what the caller heard goes with it, first."""
        earlier = self.told.pop(delegation_id, None)
        if earlier:
            self.told[taker] = earlier + self.told.get(taker, [])

    # ------------------------------------------------------------------------------ the playout facts
    def interrupted(self, newest: int, *, on_purpose: bool = False) -> None:
        """An interruption passed the talker after response ``newest``; on purpose (an instruction's): no resume."""
        self.marks.append(newest)
        self.resumes = not on_purpose
        if on_purpose:
            self.last_cut = None
            self.spoken_over.clear()

    def reached(self, number: int) -> None:
        """Response ``number`` heads the playout: once it has words, its audio is due."""
        self.head = number
        line = self.lines.get(number)
        if line is not None:
            self._watch(line)

    def started(self, number: int) -> None:
        """Response ``number``'s first audio played."""
        line = self.lines.get(number)
        if line is not None:
            line.outcome = Outcome.PLAYING
            self._stop_watch()

    def played(self, number: int, text: str, audio_ms: int) -> None:
        """A sentence of response ``number`` played, ``audio_ms`` of its audio after the sentence before."""
        line = self.lines.get(number)
        if line is not None and text.strip():
            line.outcome = Outcome.PLAYING
            line.heard = f"{line.heard} {text.strip()}".strip()
            self.cut_memory.played(text.strip(), audio_ms)

    def ended(self, number: int) -> None:
        """Response ``number`` ended: heard when it played, else silent."""
        line = self.lines.get(number)
        if line is None:
            return
        if line.outcome is Outcome.PLAYING:
            self._settle(line, Outcome.HEARD)
        else:
            self._silent(line)

    def cut(self, number: int | None, heard_ms: int = 0, speaking: str = "") -> None:
        """An interruption reached the output: the line playing is cut; every line it drops was not heard.

        ``number`` is the response at the head of the playout, if any. The lines the interruption drops are those the
        talker started before the interruption passed it; a line it started after is said in its turn.
        """
        mark = self.marks.popleft() if self.marks else max(self.lines, default=0)
        older = [line for number_, line in self.lines.items() if number_ <= mark]
        self._cut, self.head = None, None
        for line in older:
            playing = line.number == number and line.outcome is Outcome.PLAYING
            if playing and not line.heard and not self.reports_sentences:
                line.heard = line.text  # a stitched line reports its words only at its end: part of it was heard
            if playing and line.heard:
                self._cut = (line, heard_ms, speaking)
                self._settle(line, Outcome.CUT)
            else:
                line.heard = ""
                self._settle(line, Outcome.NOT_HEARD)
            # A delivered line nobody heard is said again in full instead; a cut an instruction made is not resumed.
            if self.resumes and line.again and (line.outcome is Outcome.CUT or not line.items):
                self.last_cut = line

    def kept(self, played: str) -> str:
        """What the talker's context keeps of the newest cut line, which played ``played`` (``speech.heard_words``).

        The cut's facts are read once: a later turn that ends with no cut keeps what it played.
        """
        if self._cut is None:
            return played
        (line, heard_ms, speaking), self._cut = self._cut, None
        return self.cut_memory.kept(line, played, heard_ms, speaking)

    # ------------------------------------------------------------------------------ resuming
    def caller_spoke(self) -> Line | None:
        """The caller said words: the line they cut, if any, which only the decision on those words may resume."""
        line, self.last_cut = self.last_cut, None
        if line is not None:
            self.spoken_over.append(line)
        return line

    def rest(self, line: Line | None = None) -> Line | None:
        """A cut line's words that did not play, once, as a new line; ``None`` when none is left.

        With no ``line``, the newest cut (a turn with no words came right after it). Else ``line``, which the words of a
        turn the talker waited on cut, while no line came after it.
        """
        if line is None:
            line, self.last_cut = self.last_cut, None
        elif line in self.spoken_over:
            self.spoken_over.remove(line)
        else:
            return None
        if line is None:
            return None
        words = line.text.split()
        rest = " ".join(words[min(len(line.heard.split()), len(words)) :])
        if not rest:
            return None
        return Line(line.kind, rest, line.delegation_ids, results=line.results)

    def close(self) -> None:
        """Stop the start timer."""
        self._stop_watch()

    # ------------------------------------------------------------------------------ settling
    def _silent(self, line: Line) -> None:
        """``line`` produced no audio: said once more, or not heard."""
        if self.retry_silent_lines and line.again and not line.retried:
            line.retried, line.outcome = True, Outcome.HANDED
            self._unwatch(line)
            self.lines.pop(line.number, None)
            self.observe("line.retried", line=line.number)
            self.settled.say_again(line)
        else:
            self._settle(line, Outcome.NOT_HEARD)

    def _settle(self, line: Line, outcome: Outcome) -> None:
        self.lines.pop(line.number, None)
        self._unwatch(line)
        line.outcome = outcome
        self.observe("line.settled", line=line.number, line_kind=line.kind.value, outcome=outcome.value)
        if outcome is Outcome.NOT_HEARD:
            self.settled.not_heard(line)
            return
        cut = outcome is Outcome.CUT
        for delegation_id in line.tells:
            self.told.setdefault(delegation_id, []).append(line.heard if cut else line.text)
        self.settled.heard(line, cut=cut)

    def _watch(self, line: Line) -> None:
        """Start ``line``'s start timer: it heads the playout and has words to say, and has not started."""
        if line.outcome is not Outcome.HANDED:
            return
        self._stop_watch()
        self._watched = line
        self._timer = asyncio.get_running_loop().call_later(self.start_timeout_seconds, self._timed_out, line)

    def _timed_out(self, line: Line) -> None:
        self._timer = self._watched = None
        if self.lines.get(line.number) is line and line.outcome is Outcome.HANDED:
            self.observe("line.start_timeout", line=line.number)
            self._silent(line)

    def _unwatch(self, line: Line) -> None:
        if self._watched is line:
            self._stop_watch()

    def _stop_watch(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = self._watched = None
