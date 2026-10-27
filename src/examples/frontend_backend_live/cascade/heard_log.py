# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The heard log: what the caller said and heard since the last delegation, which the next delegation carries.

The log leaves the frontend only as a delegation's input: one user message with a labeled transcript in time order,
which is the request. A caller turn is noted when it ends, before the talker decides it. Playout corrects the voice's
lines: a cut line keeps what played, and a line nobody heard leaves. A line that said backend results appears only when
it was not heard in full, since the backend holds its own answer. A delegation removes exactly the entries its input
carried once its request is accepted, so a refused request keeps them all.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from examples.frontend_backend_live.cascade.memory import CUT_MARK

HEADER = "Conversation since the last request (what the caller said and heard):"
OMITTED = "(earlier lines left out)"
# The transcript's size cap: the newest lines are kept within it.
MAX_TRANSCRIPT_CHARS = 6000


class Heard(StrEnum):
    """What a line of the transcript is."""

    CALLER = "Caller"
    AGENT = "Agent"
    CONTEXT = "App context"
    INSTRUCTION = "App instruction"


class Delegate(Protocol):
    """The session's delegation part, as the heard log calls it: hand over a delegation's input."""

    async def delegate(self, input: list[dict]) -> str:
        """Return the delegation id once the request is accepted; raise if it is refused."""


@dataclass
class Noted:
    """One entry of the log."""

    seq: int  # its place among every entry ever noted
    kind: Heard  # its label
    text: str
    caller: bool = False  # a caller turn, which a normalizer's rewrite may replace
    line: int | None = None  # the voice's line, for playout's corrections
    result: bool = False  # the line said backend results
    cut: bool = False  # the line was cut: it keeps only what played


@dataclass(frozen=True)
class Carried:
    """How far the log reached when a delegation's input was built: every entry before it went with that delegation.

    The position counts every entry ever noted, so a mark stays exact while other delegations remove their own entries
    and playout takes back lines nobody heard.
    """

    position: int


class HeardLog:
    """What the caller said and heard, and the application's context, that the next delegation carries."""

    def __init__(self):
        """Start empty."""
        self.entries: list[Noted] = []  # oldest first
        self._noted = 0  # how many entries were ever noted

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
        self._note(Heard.INSTRUCTION, text)

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
                entry.text, entry.cut = heard.rstrip() + CUT_MARK, True

    def unsay(self, line: int) -> None:
        """Nobody heard line ``line``: it leaves the log, unless a delegation carried it already."""
        self.entries = [entry for entry in self.entries if entry.line != line]

    def transcript(self) -> list[str]:
        """The labeled lines the next delegation carries, oldest first."""
        return [
            f"{entry.kind.value}: {entry.text}"
            for entry in self.entries
            if not (entry.result and not entry.cut)  # a result heard in full: the backend holds its own answer
        ]

    def delegation_input(self) -> tuple[list[dict], Carried]:
        """The input of a delegation, one user message with the transcript, and the mark of the entries it carries."""
        lines = self.transcript()
        kept: list[str] = []
        size = 0
        for line in reversed(lines):
            size += len(line) + 1
            if size > MAX_TRANSCRIPT_CHARS and kept:
                kept.append(OMITTED)
                break
            kept.append(line)
        text = "\n".join([HEADER, *reversed(kept)])
        items = [{"role": "user", "content": [{"type": "input_text", "text": text}]}]
        return items, Carried(self._noted)

    def sent(self, carried: Carried) -> None:
        """Remove exactly the entries a delegation carried; entries noted after its input was built stay."""
        self.entries = [entry for entry in self.entries if entry.seq >= carried.position]

    async def delegate(self, delegation: Delegate) -> str:
        """Delegate with everything noted so far, and return the delegation id.

        The entries the input carried leave the log only once the request is accepted, so a refusal, which raises,
        keeps them all.
        """
        items, carried = self.delegation_input()
        delegation_id = await delegation.delegate(items)
        self.sent(carried)
        return delegation_id
