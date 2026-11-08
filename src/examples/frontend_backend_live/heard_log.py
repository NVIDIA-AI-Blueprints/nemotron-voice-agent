# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The heard log: what the caller said and heard since the last delegation, which the next delegation carries.

A cut result line names the questions it held that did not play. A caller line that holds single letters or digits in a
row ends with a note that joins them as heard (``[spelled: JOHN]``). The log also keeps what the newest result line left
unheard (``newest``), and the worst of every result line that settled since the last delegation (``since_sent``), which
each delegation states to the core; that state outlives the lines' entries.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from examples.frontend_backend_live.prompts import PromptSet
from examples.frontend_backend_live.spoken import spelled_runs
from examples.shared.delegation_contract import ReplyUnheard, sentences, takes_metadata
from examples.shared.observe import log_observation

MAX_TRANSCRIPT_CHARS = 6000


class Heard(StrEnum):
    """What a line of the transcript is; the prompt set words its label."""

    CALLER = "caller"
    AGENT = "agent"
    CONTEXT = "context"
    INSTRUCTION = "instruction"
    NOTE = "note"  # a note of the frontend's own, read with no label


class Delegate(Protocol):
    """The session's delegation part, as the heard log calls it: hand over a delegation's input."""

    async def delegate(self, input: list[dict], *, metadata: Mapping[str, str] | None = None) -> str:
        """Return the delegation id once the request is accepted; raise if it is refused.

        ``metadata`` is optional: a port written before it is called with the input alone.
        """


class UnheardHow(StrEnum):
    """How the caller missed part of an answer."""

    WAITING = "waiting"  # the answer waits to be said, is being worded, or is playing
    CUT = "cut"  # an interruption cut its line before every sentence had played
    NO_AUDIO = "no_audio"  # a sentence of its line got no audio
    DROPPED = "dropped"  # the voice gave up saying it (the re-queue cap)


@dataclass(frozen=True)
class Unheard:
    """What of the backend's answers the caller has not heard, and how they missed it."""

    reply: ReplyUnheard
    how: UnheardHow | None = None  # None with ReplyUnheard.NONE
    # The turn-start snapshot gave this reading, worse than the reading at hand-off (for the traces only).
    from_turn_start: bool = field(default=False, compare=False)

    def worse(self, other: Unheard) -> Unheard:
        """The worse of the two: ``other`` when its level is higher, else this one."""
        return other if not self.reply.at_least(other.reply) else self


HEARD = Unheard(ReplyUnheard.NONE)


@dataclass
class Noted:
    """One entry of the log."""

    seq: int
    kind: Heard
    text: str
    line: int | None = None
    result: bool = False
    cut: bool = False
    pinned: bool = False
    # A result line: what of it was not heard, once its playout settled.
    lost: ReplyUnheard = ReplyUnheard.NONE
    # The question sentences of a cut result line that did not play.
    unheard: tuple[str, ...] = ()
    # A caller line: each run of single letters and digits it holds, joined as heard.
    spelled: tuple[str, ...] = ()


@dataclass(frozen=True)
class Carried:
    """How far the log reached when a delegation's input was built, counted over every entry ever noted."""

    position: int
    settled: int = 0  # how many result lines had settled: the ones after it count for the next delegation


class HeardLog:
    """What the caller said and heard, and the application's context, that the next delegation carries."""

    def __init__(
        self,
        prompts: PromptSet,
        *,
        spelled_notes: bool = True,
        unheard_questions: bool = True,
        observe: Callable[..., None] = log_observation,
    ):
        """Start empty; the lines are worded with the prompt set's transcript texts and cut mark.

        ``spelled_notes``: a caller line that holds a spelled run ends with the note that joins it
        (``delegation.spelled_notes``). ``unheard_questions``: a cut result line names the questions it held that did
        not play, and a line a sentence of which got no audio is marked as cut (``answers.unheard_questions``).
        """
        self.words, self.cut_mark, self.observe = prompts.transcript, prompts.cut_mark, observe
        self.unheard_questions, self.spelled_notes = unheard_questions, spelled_notes
        self.labels = {
            Heard.CALLER: self.words.caller,
            Heard.AGENT: self.words.agent,
            Heard.CONTEXT: self.words.context,
            Heard.INSTRUCTION: self.words.instruction,
            Heard.NOTE: "",
        }
        self.entries: list[Noted] = []
        self._noted = 0
        self._sent = 0
        # What the newest result line that settled left unheard.
        self.newest: Unheard = HEARD
        # The outcome of each result line settled since the mark, in order, and how many settled before the first.
        self._settled: list[Unheard] = []
        self._settled_before = 0
        # The ports that do not take metadata, observed once each.
        self._unsupported: set[int] = set()

    def caller(self, text: str) -> None:
        """Note a caller turn that ended, keeping the single letters and digits it holds in a row for the note."""
        runs = tuple(spelled_runs(text)) if self.spelled_notes and self.words.spelled is not None else ()
        if runs:
            self.observe("heard.spelled", runs=len(runs), tokens=sum(map(len, runs)))
        self._note(Heard.CALLER, text, spelled=runs)

    def agent(self, line: int, text: str, *, result: bool = False) -> None:
        """Note line ``line`` handed to speech; ``result``: it says backend results."""
        self._note(Heard.AGENT, text, line=line, result=result)

    def context(self, text: str) -> None:
        """Note silent application context (``session.thinking.append``)."""
        self._note(Heard.CONTEXT, text)

    def instruction(self, text: str) -> None:
        """Note an instruction append, which every later delegation carries."""
        self._note(Heard.INSTRUCTION, text, pinned=True)

    def _note(self, kind: Heard, text: str, **fields) -> None:
        self.entries.append(Noted(self._noted, kind, text, **fields))
        self._noted += 1

    def replace_agent(self, line: int, text: str) -> None:
        """Set line ``line``'s text: a streamed reply's words so far, then the whole reply."""
        for entry in self.entries:
            if entry.line == line:
                entry.text = text

    def said_again(self, line: int, number: int) -> None:
        """Renumber line ``line``, said once more as ``number``, keeping its entry in place."""
        for entry in self.entries:
            if entry.line == line:
                entry.line = number

    def cut(self, line: int, heard: str, *, unheard: Sequence[str] = ()) -> None:
        """Keep only ``heard`` of line ``line``, with the cut mark; ``unheard``: the questions that did not play."""
        for entry in self.entries:
            if entry.line == line:
                entry.text, entry.cut = heard.rstrip() + self.cut_mark, True
                entry.unheard = tuple(unheard) if self.unheard_questions else ()

    def partly(self, line: int, heard: str, *, unheard: Sequence[str] = ()) -> None:
        """Keep only ``heard`` of line ``line``, a sentence of which got no audio.

        With ``unheard_questions`` it is marked as cut, with ``unheard``, the questions that did not play.
        """
        if self.unheard_questions:
            self.cut(line, heard, unheard=unheard)
            return
        for entry in self.entries:
            if entry.line == line:
                entry.text, entry.cut = heard, True

    def whole(self, line: int, text: str, *, rest: str = "") -> None:
        """Line ``line``'s rest was heard after all: its entry reads ``text`` again, with no cut mark.

        It is then a result line heard in full (``lost`` is ``NONE``). When a delegation already carried the cut entry,
        a ``NOTE`` entry says instead what the caller has now heard (``transcript.now_heard`` with ``rest``), unless
        the prompt set lacks that text.
        """
        entries = [entry for entry in self.entries if entry.line == line]
        for entry in entries:
            entry.text, entry.cut, entry.lost, entry.unheard = text, False, ReplyUnheard.NONE, ()
        if not entries and self.words.now_heard is not None:
            self._note(Heard.NOTE, self.words.now_heard.format(rest=(rest or text).strip()))

    def not_said(self, answer: str) -> None:
        """An answer the backend gave was dropped unheard: the next delegation reads ``transcript.not_said``.

        Nothing is noted when the prompt set lacks that text.
        """
        if self.words.not_said is not None:
            self._note(Heard.NOTE, self.words.not_said.format(answer=answer.strip()))

    @property
    def since_sent(self) -> Unheard:
        """The worst of every result line that settled since the last delegation was sent.

        So an older cut question is not hidden by a newer line heard in full.
        """
        worst = HEARD
        for outcome in self._settled:
            worst = worst.worse(outcome)
        return worst

    @property
    def question_cut(self) -> bool:
        """Whether the newest result line that settled left a question unheard."""
        return self.newest.reply is ReplyUnheard.QUESTION

    def result_settled(self, line: int, rest: Sequence[str], how: UnheardHow | None) -> Unheard:
        """A line that said backend results settled; ``rest`` is its sentences the caller did not hear.

        ``rest`` is empty when the words that played cover every sentence. The level is ``ReplyUnheard.of(rest)``;
        ``how`` names why the line ended short and is kept for the trace only (it is None once the level is NONE). The
        outcome becomes :attr:`newest`, :attr:`since_sent` takes the worse of itself and the outcome, and the line's
        entry keeps the level as ``lost`` when the log still holds that entry. Returns the outcome.
        """
        level = ReplyUnheard.of(rest)
        outcome = HEARD if level is ReplyUnheard.NONE else Unheard(level, how)
        for entry in self.entries:
            if entry.line == line:
                entry.lost = level
        return self._settle(outcome)

    def result_unsaid(self, said: str) -> Unheard:
        """The voice gave up saying a result whose text is ``said``: nothing of it was heard (``DROPPED``)."""
        level = ReplyUnheard.of(sentences(said))
        return self._settle(HEARD if level is ReplyUnheard.NONE else Unheard(level, UnheardHow.DROPPED))

    def _settle(self, outcome: Unheard) -> Unheard:
        self.newest = outcome
        self._settled.append(outcome)
        return outcome

    def unsay(self, line: int) -> None:
        """Remove line ``line``, which nobody heard, unless a delegation carried it already."""
        self.entries = [entry for entry in self.entries if entry.line != line]

    def lines(self) -> list[tuple[Heard, str]]:
        """Return each line the next delegation carries, oldest first: its kind and its labeled text."""
        return [(entry.kind, self._labeled(entry)) for entry in self._carried()]

    def _carried(self) -> list[Noted]:
        """Return the entries the next delegation carries, oldest first."""
        return [
            entry
            for entry in self.entries
            # The backend holds its own answer: a result heard in full stays out.
            if not (entry.result and not entry.cut)
        ]

    def _labeled(self, entry: Noted) -> str:
        label = self.labels[entry.kind]
        text = entry.text
        note = self.words.cut_question
        if note is not None and entry.unheard and entry.lost is ReplyUnheard.QUESTION:
            text += note.format(questions=" ".join(entry.unheard))
        text += "".join(f" [{self.words.spelled}: {run}]" for run in entry.spelled)
        return f"{label}: {text}" if label else text

    def delegation_input(self, *, unheard: bool = False) -> tuple[list[dict], Carried]:
        """Return a delegation's input, one user message with the transcript, and the mark of the entries it carries."""
        entries = self._carried()
        earlier = [self._labeled(entry) for entry in entries if self._earlier(entry)]
        recent = [entry for entry in entries if not self._earlier(entry)]
        capped = [entry for entry in recent if not entry.pinned]
        kept: set[int] = set()
        size = 0
        for entry in reversed(capped):
            size += len(self._labeled(entry)) + 1
            if size > MAX_TRANSCRIPT_CHARS and kept:
                break
            kept.add(entry.seq)
        omitted = [self.words.omitted] if len(kept) < len(capped) else []
        lines = [self._labeled(entry) for entry in recent if entry.pinned or entry.seq in kept]
        before = [self.words.earlier, *earlier] if earlier else []
        note = [self.words.unheard] if unheard else []
        text = "\n".join([*before, self.words.header, *omitted, *lines, *note])
        items = [{"role": "user", "content": [{"type": "input_text", "text": text}]}]
        return items, Carried(self._noted, self._settled_before + len(self._settled))

    def _earlier(self, entry: Noted) -> bool:
        """Return whether ``entry`` is an instruction an accepted delegation carried: not new to the backend."""
        return entry.pinned and entry.seq < self._sent

    def sent(self, carried: Carried) -> None:
        """Remove the entries a delegation carried, but instructions; entries noted since its input was built stay.

        The result lines that settled after its input was built still count for the next delegation (``since_sent``).
        """
        self.entries = [entry for entry in self.entries if entry.seq >= carried.position or entry.pinned]
        self._sent = max(self._sent, carried.position)
        drop = max(0, carried.settled - self._settled_before)
        self._settled, self._settled_before = self._settled[drop:], self._settled_before + drop

    async def delegate(
        self, delegation: Delegate, *, unheard: bool = False, metadata: Mapping[str, str] | None = None
    ) -> str:
        """Delegate with everything noted so far and return the delegation id; a refusal raises and keeps the entries.

        ``unheard``: an answer the backend gave waits to be heard, which the transcript's last line then says.
        ``metadata``: what the frontend states about the request; a port whose ``delegate`` does not take it is called
        with the input alone (observed once).
        """
        items, carried = self.delegation_input(unheard=unheard)
        if metadata and takes_metadata(delegation.delegate):
            delegation_id = await delegation.delegate(items, metadata=metadata)
        else:
            if metadata and id(delegation) not in self._unsupported:
                self._unsupported.add(id(delegation))
                self.observe("delegation.metadata_unsupported", port=type(delegation).__name__)
            delegation_id = await delegation.delegate(items)
        self.sent(carried)
        return delegation_id
