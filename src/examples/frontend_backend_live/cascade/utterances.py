# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the cascade says: every line with its kind, what each delegation already told the caller, and what played.

**One clock, corrected by playout.** Every line goes through :meth:`Utterances.hand_off`. In one step, with nothing
awaited in between, it is written into the talker's memory and the heard log, counted in flight on the floor, and
handed to the speech output, whose lock is the only order. Every model call that reads the memory sees each line from
its hand-off. The playout facts then correct the records (:class:`Playout`, reported by the frontend adapter's tracker
in playout order): a line heard stays; a cut line keeps what played and ends ``[cut off]``, and ``told`` keeps only that
part; a line nobody heard leaves the memory and the heard log, ``told`` drops it, and a line the presenter delivered
goes back to its inbox. A host with no tracker settles each line as heard when its hand-off returns.

**Generations.** When an interruption passes the cascade's binding, it calls :meth:`Utterances.interrupted`: the lines
handed before it belong to an older generation. An interruption flushes everything queued ahead of it, so the
tracker's ``cut`` settles the lines of the older generations only: a line handed after the interruption passed is
never taken for cut. A cut with no interruption seen by the binding settles every open line.

**Told.** ``told`` is what the caller was already told about a delegation (its lead-in, a progress line), written at
the hand-off, so a result worded while its lead-in still plays already knows it.

**Resume and retry.** The newest line cut by an interruption keeps its unheard rest, which a turn with no words says
once, unless the caller said words since or an instruction cut it on purpose; a line that produced no audio is said
once more.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from examples.frontend_backend_live.cascade.floor import Floor
from examples.frontend_backend_live.cascade.heard_log import HeardLog
from examples.frontend_backend_live.cascade.memory import TalkerMemory

Say = Callable[[str], Awaitable[None]]
MAX_KEPT_LINES = 32


class LineKind(StrEnum):
    """What a line is, for the traces: what a line does follows from what it carries, never from its kind."""

    REPLY = "reply"  # the talker's reply to a caller turn
    LEAD_IN = "lead_in"  # a delegation's acknowledgement
    RESULT = "result"  # ready results, worded into one line
    COMMENTARY = "commentary"  # the application's commentary, or the greeting, worded
    STATUS = "status"  # a progress line, or the still-working line
    NOTICE = "notice"  # a fixed line: an ending's, the refusal, the app-wait line


class Outcome(StrEnum):
    """Where a line stands."""

    HANDED = "handed"  # handed to speech, not started
    PLAYING = "playing"
    HEARD = "heard"
    CUT = "cut"  # cut after some words played
    NOT_HEARD = "not_heard"


@dataclass
class Line:
    """One line the cascade said."""

    id: int
    kind: LineKind
    text: str
    delegation_ids: tuple[str, ...] = ()
    items: tuple[Any, ...] = ()  # what the presenter delivered with it, to requeue when nobody heard it
    results: bool = False  # it says backend results, which the backend holds in its own history
    generation: int = 0
    outcome: Outcome = Outcome.HANDED
    heard: str = ""  # the words that played, when the host reports them
    retried: bool = False  # said once more after it produced no audio


class Playout(Protocol):
    """What the frontend adapter's playout tracker reports, in playout order: each fact is the oldest open line's."""

    def started(self) -> None:
        """The line's first audio reached the caller."""

    def played(self, text: str) -> None:
        """These words of the line reached the caller."""

    def finished(self) -> None:
        """The line played to its end."""

    def cut(self) -> None:
        """An interruption cut the line; the lines handed before that interruption and not played are not heard."""

    def silent(self) -> None:
        """The line produced no audio."""


class Utterances:
    """Says each line of one call, keeps what each delegation already told the caller, and settles each by playout."""

    def __init__(
        self,
        say: Say,
        heard_log: HeardLog,
        memory: TalkerMemory,
        floor: Floor,
        *,
        observe: Callable[..., None],
        reports_playout: bool = False,
        reports_sentences: bool = False,
        start_timeout_seconds: float = 3.0,
        retry_silent_lines: bool = True,
    ):
        """Wire the speech output and the records each line is written to.

        Args:
            say: Awaited with the text to speak.
            heard_log: Where each line is noted for the next delegation.
            memory: The talker's memory, where each line is written when it is handed to speech.
            floor: Counts the lines in flight.
            observe: ``observe(kind, **fields)`` hook for traces.
            reports_playout: Whether the host reports each line's playout (:class:`Playout`); without it a line is
                heard when its hand-off returns.
            reports_sentences: Whether the playout reports each sentence's words once its audio has played
                (per-sentence speech); without it, a stitched line's words arrive only at its end.
            start_timeout_seconds: A line that heads the queue and has not started this long after counts as silent.
            retry_silent_lines: Whether a line that produced no audio is said once more.
        """
        self.say_text, self.heard_log, self.memory, self.floor, self.observe = say, heard_log, memory, floor, observe
        self.reports_playout, self.reports_sentences = reports_playout, reports_sentences
        self.start_timeout_seconds, self.retry_silent_lines = start_timeout_seconds, retry_silent_lines
        self.requeue: Callable[[Line], None] = lambda line: None  # the presenter takes back its lines nobody heard
        self.told: dict[str, list[tuple[int, str]]] = {}  # by delegation: (line, words) the caller was told about it
        self.recent: deque[Line] = deque(maxlen=MAX_KEPT_LINES)
        self.in_flight: deque[Line] = deque()  # handed and not settled, oldest first
        self.last_cut: Line | None = None  # the newest line an interruption cut, until a later line or a resume
        self.generation = 0  # one more for each interruption that passed the binding
        self.resumes = True  # whether a cut may be resumed: not after an instruction's, until the next interruption
        self._made = 0
        self._head_timer: asyncio.TimerHandle | None = None
        self._retries: set[asyncio.Task] = set()

    # ------------------------------------------------------------------------------ saying
    def line(
        self,
        text: str,
        kind: LineKind,
        *,
        delegation_ids: Sequence[str] = (),
        items: Sequence = (),
        results: bool = False,
    ) -> Line:
        """A new line, numbered, not handed yet; ``items`` and ``results`` say what a delivered line carries."""
        self._made += 1
        return Line(self._made, kind, text, tuple(delegation_ids), tuple(items), results, self.generation)

    async def say(self, text: str, kind: LineKind, *, delegation_ids: Sequence[str] = ()) -> Line | None:
        """Say ``text``, a line of ``kind``; empty text is not said."""
        if not text.strip():
            return None
        line = self.line(text, kind, delegation_ids=delegation_ids)
        await self.hand_off(line)
        return line

    async def say_lead_in(self, delegation_id: str, text: str) -> Line | None:
        """Say a delegation's lead-in, which the caller is now told about that delegation."""
        if not text.strip():
            return None
        line = self.line(text, LineKind.LEAD_IN, delegation_ids=(delegation_id,))
        await self.hand_off(line, tells=(delegation_id,))
        return line

    async def hand_off(self, line: Line, *, tells: Sequence[str] = ()) -> None:
        """Hand ``line`` to speech: its records first, in one step, then the speech output.

        ``tells`` are the delegations the line tells the caller about (a lead-in, a progress line). A hand-off that
        raises takes its records back and raises.
        """
        self.memory.spoken(line.id, line.text)
        self.heard_log.agent(line.id, line.text, result=line.results)
        for delegation_id in tells:
            self.told.setdefault(delegation_id, []).append((line.id, line.text))
        line.generation = self.generation
        self.in_flight.append(line)
        self.recent.append(line)
        self.last_cut = None
        self.floor.line_handed()
        self.observe("line.handed", line=line.id, line_kind=line.kind.value)
        if len(self.in_flight) == 1:
            self._watch_head()
        try:
            await self.say_text(line.text)
        except BaseException:
            if line in self.in_flight:
                self._take_back(line)
                self._settle(line, Outcome.NOT_HEARD, requeue=False)
            raise
        if not self.reports_playout and line in self.in_flight:
            self._settle(line, Outcome.HEARD)

    # ------------------------------------------------------------------------------ told
    def told_text(self, delegation_id: str) -> str | None:
        """What the caller was told about a delegation so far, or ``None``."""
        words = [text for _, text in self.told.get(delegation_id, ()) if text]
        return " ".join(words) or None

    def take_told(self, delegation_id: str) -> str | None:
        """What the caller was told about a delegation whose result is said now; it is forgotten."""
        words = self.told_text(delegation_id)
        self.told.pop(delegation_id, None)
        return words

    def forget(self, delegation_id: str) -> None:
        """Forget what the caller was told about a delegation that ended with no result."""
        self.told.pop(delegation_id, None)

    def carry_told(self, delegation_id: str, taker: str) -> None:
        """A newer delegation took over the delegation's work: what the caller was told goes with it, first."""
        earlier = self.told.pop(delegation_id, None)
        if earlier:
            self.told[taker] = earlier + self.told.get(taker, [])

    # ------------------------------------------------------------------------------ playout (the tracker's facts)
    def interrupted(self) -> None:
        """An interruption passed the binding: the lines handed so far are the ones a cut may settle."""
        self.generation += 1
        self.resumes = True

    def cut_on_purpose(self) -> None:
        """An instruction cut the line in progress: no turn with no words resumes it, or a line cut before it."""
        self.resumes, self.last_cut = False, None

    def caller_spoke(self) -> None:
        """The caller said words: a line cut before them is not resumed (a resume answers a turn with no words)."""
        self.last_cut = None

    def started(self) -> None:
        """The oldest open line started playing."""
        head = self._head()
        if head is not None:
            head.outcome = Outcome.PLAYING
            self._cancel_head_timer()

    def played(self, text: str) -> None:
        """Words of the oldest open line reached the caller."""
        head = self._head()
        if head is not None and text.strip():
            head.outcome = Outcome.PLAYING
            head.heard = f"{head.heard} {text.strip()}".strip()

    def finished(self) -> None:
        """The oldest open line played to its end."""
        head = self._head()
        if head is not None:
            self._settle(head, Outcome.HEARD)

    def cut(self) -> None:
        """An interruption cut the line playing; every line handed before it and not played was not heard.

        Once an interruption has passed the binding, a cut settles only the lines handed before the newest one, and
        none when every line in flight came after it: a reply to an instruction is often handed before the
        instruction's interruption reaches the output. With none counted (a host that reports no interruption), every
        line in flight.
        """
        if self.generation:
            older = [line for line in self.in_flight if line.generation < self.generation]
        else:
            older = list(self.in_flight)
        for index, line in enumerate(older):
            heard = line.heard if index == 0 else ""
            if index == 0 and not heard and line.outcome == Outcome.PLAYING and not self.reports_sentences:
                # A stitched line reports its words only at its end: part of it was heard, so it stays cut, whole,
                # and is not said again. With per-sentence speech, a line cut before a sentence finished was not heard.
                heard = line.heard = line.text
            self._correct(line, heard)
            outcome = Outcome.CUT if heard else Outcome.NOT_HEARD
            self._settle(line, outcome)
            # A delivered line nobody heard is requeued instead; a cut an instruction made is not resumed.
            if self.resumes and (outcome == Outcome.CUT or not line.items):
                self.last_cut = line

    def silent(self) -> None:
        """The oldest open line produced no audio: said once more, or not heard.

        Said once more, it keeps its records and goes behind the lines already queued, as its speech does.
        """
        head = self._head()
        if head is None:
            return
        if self.retry_silent_lines and not head.retried:
            head.retried, head.outcome = True, Outcome.HANDED
            self.in_flight.rotate(-1)
            self.observe("line.retried", line=head.id)
            task = asyncio.create_task(self._say_again(head), name="line-retry")
            self._retries.add(task)
            task.add_done_callback(self._retries.discard)
            self._cancel_head_timer()
            self._watch_head()
        else:
            self._take_back(head)
            self._settle(head, Outcome.NOT_HEARD)

    async def _say_again(self, line: Line) -> None:
        try:
            await self.say_text(line.text)
        except Exception as exc:
            self.observe("line.retry_failed", line=line.id, error_type=type(exc).__name__)
            if line in self.in_flight:
                self._take_back(line)
                self._settle(line, Outcome.NOT_HEARD)

    async def resume(self) -> bool:
        """Say the newest cut line's words that did not play, once (a turn with no words came right after the cut)."""
        line, self.last_cut = self.last_cut, None
        if line is None:
            return False
        words = line.text.split()
        rest = " ".join(words[min(len(line.heard.split()), len(words)) :])
        if not rest:
            return False
        await self.hand_off(self.line(rest, line.kind, delegation_ids=line.delegation_ids, results=line.results))
        return True

    def close(self) -> None:
        """Stop the start timer and any line said once more."""
        self._cancel_head_timer()
        for task in tuple(self._retries):
            task.cancel()

    # ------------------------------------------------------------------------------ settling
    def _head(self) -> Line | None:
        return self.in_flight[0] if self.in_flight else None

    def _correct(self, line: Line, heard: str) -> None:
        """Correct the records of a cut line to what played; with nothing played, take them back."""
        if not heard:
            self._take_back(line)
            return
        self.memory.cut(line.id, heard)
        self.heard_log.cut(line.id, heard)
        for said in self.told.values():
            said[:] = [(each, heard if each == line.id else text) for each, text in said]

    def _take_back(self, line: Line) -> None:
        """Nobody heard the line: it leaves the memory, the heard log and ``told``."""
        self.memory.unsay(line.id)
        self.heard_log.unsay(line.id)
        for delegation_id in list(self.told):
            kept = [(each, text) for each, text in self.told[delegation_id] if each != line.id]
            if kept:
                self.told[delegation_id] = kept
            else:
                del self.told[delegation_id]

    def _settle(self, line: Line, outcome: Outcome, *, requeue: bool = True) -> None:
        was_head = self._head() is line
        if line in self.in_flight:
            self.in_flight.remove(line)
        line.outcome = outcome
        self.floor.line_settled()
        self.observe("line.settled", line=line.id, line_kind=line.kind.value, outcome=outcome.value)
        if requeue and outcome == Outcome.NOT_HEARD and line.items:
            self.requeue(line)
        if was_head:
            self._cancel_head_timer()
            self._watch_head()

    def _watch_head(self) -> None:
        """Start the head line's start timer, when the host reports playout."""
        head = self._head()
        if not self.reports_playout or head is None or head.outcome != Outcome.HANDED:
            return
        self._cancel_head_timer()
        loop = asyncio.get_running_loop()
        self._head_timer = loop.call_later(self.start_timeout_seconds, self._head_timed_out, head)

    def _head_timed_out(self, line: Line) -> None:
        self._head_timer = None
        if self._head() is line and line.outcome == Outcome.HANDED:
            self.observe("line.start_timeout", line=line.id)
            self.silent()

    def _cancel_head_timer(self) -> None:
        if self._head_timer is not None:
            self._head_timer.cancel()
            self._head_timer = None
