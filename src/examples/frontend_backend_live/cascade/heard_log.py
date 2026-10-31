# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The heard log: what the caller said and heard since the last delegation, which the next delegation carries.

The log leaves the frontend only as a delegation's input: one user message with a labeled transcript in time order,
which is the request. A caller turn is noted when it ends, before the talker decides it. Playout corrects the voice's
lines: a cut line keeps what played, and a line nobody heard leaves. A line that said backend results appears only when
it was not heard in full, since the backend holds its own answer. A delegation removes exactly the entries its input
carried once its request is accepted, so a refused request keeps them all. An instruction stays: every later delegation
carries it, outside the transcript's size cap, as the talker's memory pins it. The first accepted delegation carries it
in its place among the lines since the last request; later ones carry it first, under the ``earlier`` line, so it never
reads as new. While an answer the backend gave waits to be heard, the transcript ends with the ``unheard`` line, so the
caller's words are not read as a reply to it. The transcript's words are the prompt set's (``transcript``,
``cut_mark``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from examples.frontend_backend_live.prompts import PromptSet

# The transcript's size cap: the newest lines are kept within it.
MAX_TRANSCRIPT_CHARS = 6000


class Heard(StrEnum):
    """What a line of the transcript is; the prompt set words its label."""

    CALLER = "caller"
    AGENT = "agent"
    CONTEXT = "context"
    INSTRUCTION = "instruction"


class Delegate(Protocol):
    """The session's delegation part, as the heard log calls it: hand over a delegation's input."""

    async def delegate(self, input: list[dict]) -> str:
        """Return the delegation id once the request is accepted; raise if it is refused."""


@dataclass
class Noted:
    """One entry of the log."""

    seq: int  # its place among every entry ever noted
    kind: Heard  # what the line is, labeled in the set's words
    text: str
    caller: bool = False  # a caller turn, which a normalizer's rewrite may replace
    line: int | None = None  # the voice's line, for playout's corrections
    result: bool = False  # the line said backend results
    cut: bool = False  # the line was cut: it keeps only what played
    pinned: bool = False  # an instruction: every later delegation carries it, outside the size cap


@dataclass(frozen=True)
class Carried:
    """How far the log reached when a delegation's input was built: every entry before it went with that delegation.

    The position counts every entry ever noted, so a mark stays exact while other delegations remove their own entries
    and playout takes back lines nobody heard.
    """

    position: int


class HeardLog:
    """What the caller said and heard, and the application's context, that the next delegation carries."""

    def __init__(self, prompts: PromptSet):
        """Start empty; the lines are worded with the prompt set's transcript texts and cut mark."""
        self.words, self.cut_mark = prompts.transcript, prompts.cut_mark
        self.labels = {
            Heard.CALLER: self.words.caller,
            Heard.AGENT: self.words.agent,
            Heard.CONTEXT: self.words.context,
            Heard.INSTRUCTION: self.words.instruction,
        }
        self.entries: list[Noted] = []  # oldest first
        self._noted = 0  # how many entries were ever noted
        self._sent = 0  # how far the accepted delegations' inputs reached: the instructions before it were carried

    def caller(self, text: str) -> None:
        """A caller turn ended."""
        self._note(Heard.CALLER, text, caller=True)

    def agent(self, line: int, text: str, *, result: bool = False) -> None:
        """The voice handed line ``line`` to speech; ``result`` when it said backend results."""
        self._note(Heard.AGENT, text, line=line, result=result)

    def context(self, text: str) -> None:
        """Silent application context (``session.thinking.append``)."""
        self._note(Heard.CONTEXT, text)

    def instruction(self, text: str) -> None:
        """An instruction append."""
        self._note(Heard.INSTRUCTION, text, pinned=True)

    def _note(self, kind: Heard, text: str, **fields) -> None:
        self.entries.append(Noted(self._noted, kind, text, **fields))
        self._noted += 1

    def replace_caller(self, original: str, text: str) -> None:
        """Replace the newest caller turn noted as ``original`` with ``text`` (the normalizer rewrote it)."""
        for entry in reversed(self.entries):
            if entry.caller and entry.text == original:
                entry.text = text
                return

    def cut(self, line: int, heard: str) -> None:
        """Line ``line`` was cut after ``heard``."""
        for entry in self.entries:
            if entry.line == line:
                entry.text, entry.cut = heard.rstrip() + self.cut_mark, True

    def unsay(self, line: int) -> None:
        """Nobody heard line ``line``: it leaves the log, unless a delegation carried it already."""
        self.entries = [entry for entry in self.entries if entry.line != line]

    def lines(self) -> list[tuple[Heard, str]]:
        """Each line the next delegation carries, oldest first: its kind, and the labeled line the backend reads."""
        return [(entry.kind, self._labeled(entry)) for entry in self._carried()]

    def _carried(self) -> list[Noted]:
        """The entries the next delegation carries, oldest first."""
        return [
            entry
            for entry in self.entries
            if not (entry.result and not entry.cut)  # a result heard in full: the backend holds its own answer
        ]

    def _labeled(self, entry: Noted) -> str:
        return f"{self.labels[entry.kind]}: {entry.text}"

    def transcript(self) -> list[str]:
        """The labeled lines the next delegation carries, oldest first."""
        return [line for _, line in self.lines()]

    def delegation_input(self, *, unheard: bool = False) -> tuple[list[dict], Carried]:
        """The input of a delegation, one user message with the transcript, and the mark of the entries it carries.

        The instructions an accepted delegation carried come first, under the set's ``earlier`` line. The lines since
        the last request follow the header: the newest within the size cap, and every instruction not carried yet
        besides. With ``unheard`` (an answer the backend gave waits to be heard) the transcript ends with the set's
        ``unheard`` line.
        """
        entries = self._carried()
        earlier = [self._labeled(entry) for entry in entries if self._earlier(entry)]
        recent = [entry for entry in entries if not self._earlier(entry)]
        capped = [entry for entry in recent if not entry.pinned]
        kept: set[int] = set()  # the capped entries kept, by place: the newest within the cap
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
        return items, Carried(self._noted)

    def _earlier(self, entry: Noted) -> bool:
        """Whether ``entry`` is an instruction an accepted delegation carried, so it is not new to the backend."""
        return entry.pinned and entry.seq < self._sent

    def sent(self, carried: Carried) -> None:
        """Remove the entries a delegation carried, but instructions; entries noted after its input was built stay."""
        self.entries = [entry for entry in self.entries if entry.seq >= carried.position or entry.pinned]
        self._sent = max(self._sent, carried.position)

    async def delegate(self, delegation: Delegate, *, unheard: bool = False) -> str:
        """Delegate with everything noted so far, and return the delegation id.

        ``unheard`` tells whether an answer the backend gave waits to be heard as the input is built (the work note's
        ``answer_ready``). The entries the input carried leave the log only once the request is accepted, so a refusal,
        which raises, keeps them all.
        """
        items, carried = self.delegation_input(unheard=unheard)
        delegation_id = await delegation.delegate(items)
        self.sent(carried)
        return delegation_id
