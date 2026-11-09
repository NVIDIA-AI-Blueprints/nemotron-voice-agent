# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The heard log: what the caller said and heard since the last delegation, which the next delegation carries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from examples.frontend_backend_live.prompts import PromptSet

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

    seq: int
    kind: Heard
    text: str
    line: int | None = None
    result: bool = False
    cut: bool = False
    pinned: bool = False


@dataclass(frozen=True)
class Carried:
    """How far the log reached when a delegation's input was built, counted over every entry ever noted."""

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
        self.entries: list[Noted] = []
        self._noted = 0
        self._sent = 0

    def caller(self, text: str) -> None:
        """Note a caller turn that ended."""
        self._note(Heard.CALLER, text)

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

    def cut(self, line: int, heard: str) -> None:
        """Keep only ``heard`` of line ``line``, with the cut mark."""
        for entry in self.entries:
            if entry.line == line:
                entry.text, entry.cut = heard.rstrip() + self.cut_mark, True

    def partly(self, line: int, heard: str) -> None:
        """Keep only ``heard`` of line ``line``, a sentence of which got no audio."""
        for entry in self.entries:
            if entry.line == line:
                entry.text, entry.cut = heard, True

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
        return f"{self.labels[entry.kind]}: {entry.text}"

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
        return items, Carried(self._noted)

    def _earlier(self, entry: Noted) -> bool:
        """Return whether ``entry`` is an instruction an accepted delegation carried: not new to the backend."""
        return entry.pinned and entry.seq < self._sent

    def sent(self, carried: Carried) -> None:
        """Remove the entries a delegation carried, but instructions; entries noted since its input was built stay."""
        self.entries = [entry for entry in self.entries if entry.seq >= carried.position or entry.pinned]
        self._sent = max(self._sent, carried.position)

    async def delegate(self, delegation: Delegate, *, unheard: bool = False) -> str:
        """Delegate with everything noted so far and return the delegation id; a refusal raises and keeps the entries.

        ``unheard``: an answer the backend gave waits to be heard, which the transcript's last line then says.
        """
        items, carried = self.delegation_input(unheard=unheard)
        delegation_id = await delegation.delegate(items)
        self.sent(carried)
        return delegation_id
