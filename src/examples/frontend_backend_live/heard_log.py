# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The heard log: what the caller said and heard since the last delegation, which the next delegation carries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from examples.frontend_backend_live.prompts import PromptSet
from examples.frontend_backend_live.spoken import transcript_results
from examples.shared.delegation_contract import CALLER_CONTEXT, CALLER_EXCERPT, CALLER_EXCERPT_CHARS, CallerContext

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
    presentation: str | None = None
    generated_text: str | None = None
    heard_prefix: str = ""
    audio_started: bool = False
    line_kind: str | None = None
    delegation_ids: tuple[str, ...] = ()
    work_refs: tuple[str, ...] = ()
    results: tuple[dict, ...] = ()


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
        self.omitted_before: int | None = None

    def caller(self, text: str) -> None:
        """Note a caller turn that ended."""
        self._note(Heard.CALLER, text)

    def agent(self, line: int, text: str, *, result: bool = False, evidence: dict | None = None) -> None:
        """Note line ``line`` handed to speech; ``result``: it says backend results."""
        self._note(Heard.AGENT, text, line=line, result=result, presentation="queued", generated_text=text)
        if evidence is not None:
            self.correlate(line, evidence)

    def correlate(self, line: int, evidence: dict) -> None:
        """Copy frontend publication identities without substituting backend content for spoken wording."""
        entries = [entry for entry in self.entries if entry.line == line]
        if not entries:
            # A prior delegation may have carried the queued/playing state. Its later settlement is new evidence.
            kind = evidence.get("kind")
            result = kind == "result" if kind is not None else bool(evidence.get("results"))
            self.agent(line, evidence["text"], result=result)
            entries = [self.entries[-1]]
        for entry in entries:
            self._update(
                entry,
                line_kind=evidence.get("kind"),
                delegation_ids=tuple(evidence.get("delegation_ids", ())),
                work_refs=tuple(evidence.get("context_work_refs", evidence.get("work_refs", ()))),
                results=tuple(dict(item) for item in evidence.get("results", ())),
                audio_started=evidence.get("audio_started", entry.audio_started),
            )

    def context(self, text: str) -> None:
        """Note silent application context (``session.thinking.append``)."""
        self._note(Heard.CONTEXT, text)

    def not_conveyed(self, items: list[dict], reason: str) -> str:
        """Journal exact presentation dispositions without a fabricated speech line or caller answer."""
        text = json.dumps(
            {"frontend_presentation": {"state": "not_conveyed", "reason": reason, "items": transcript_results(items)}},
            ensure_ascii=False,
        )
        self._note(Heard.CONTEXT, text, presentation="not_conveyed", results=tuple(items))
        return text

    def uncertain_after_reconnect(self) -> None:
        """Record uncertain earlier speech without putting the retained drafts in the worker transcript."""
        text = json.dumps({"frontend_presentation": {"state": "unknown", "reason": "reconnect"}})
        self._note(Heard.CONTEXT, text, presentation="unknown")

    def instruction(self, text: str) -> None:
        """Note an instruction append, which every later delegation carries."""
        self._note(Heard.INSTRUCTION, text, pinned=True)

    def _note(self, kind: Heard, text: str, **fields) -> None:
        self.entries.append(Noted(self._noted, kind, text, **fields))
        self._noted += 1

    def _update(self, entry: Noted, **changes) -> None:
        """Keep new playback evidence beyond the input sent by an earlier handoff."""
        if any(getattr(entry, key) != value for key, value in changes.items()):
            for key, value in changes.items():
                setattr(entry, key, value)
            entry.seq = self._noted
            self._noted += 1

    def replace_agent(self, line: int, text: str) -> None:
        """Set line ``line``'s text: a streamed reply's words so far, then the whole reply."""
        for entry in self.entries:
            if entry.line == line:
                self._update(entry, text=text, generated_text=text)

    def said_again(self, line: int, number: int) -> None:
        """Renumber line ``line``, said once more as ``number``, keeping its entry in place."""
        for entry in self.entries:
            if entry.line == line:
                self._update(entry, line=number)

    def cut(self, line: int, heard: str) -> None:
        """Keep only ``heard`` of line ``line``, with the cut mark."""
        for entry in self.entries:
            if entry.line == line:
                self._update(
                    entry, text=heard.rstrip() + self.cut_mark, cut=True, presentation="interrupted", heard_prefix=heard
                )

    def partly(self, line: int, heard: str) -> None:
        """Keep only ``heard`` of line ``line``, a sentence of which got no audio."""
        for entry in self.entries:
            if entry.line == line:
                self._update(entry, text=heard, cut=True, presentation="partial", heard_prefix=heard)

    def audio(self, line: int) -> None:
        """Keep observed audio-start evidence separate from any completed sentence transcript."""
        for entry in self.entries:
            if entry.line == line:
                self._update(entry, audio_started=True, presentation="playing")

    def completed(self, line: int, heard: str) -> None:
        """Record completed frontend playback without treating generated or queued text as heard."""
        for entry in self.entries:
            if entry.line == line:
                self._update(entry, presentation="complete", heard_prefix=heard)

    def not_heard(self, line: int, evidence: dict) -> None:
        """Keep the exact playback attempt when a backend line is interrupted before audio."""
        self.correlate(line, evidence)
        for entry in self.entries:
            if entry.line == line:
                self._update(entry, presentation="not_heard", heard_prefix="", audio_started=False)

    def unsay(self, line: int) -> None:
        """Remove line ``line``, which nobody heard, unless a delegation carried it already."""
        self.entries = [entry for entry in self.entries if entry.line != line]

    def lines(self) -> list[tuple[Heard, str]]:
        """Return local pending lines with their original frontend wording and playback."""
        return [(entry.kind, self._labeled(entry, include_generated=True)) for entry in self._carried()]

    def _carried(self) -> list[Noted]:
        """Return the entries the next delegation carries, oldest first."""
        return sorted(self.entries, key=lambda entry: entry.seq)

    def _playback(self, entry: Noted) -> dict:
        return {
            "line_id": entry.line,
            "kind": entry.line_kind,
            "playback_state": entry.presentation or "unknown",
            "audio_started": entry.audio_started,
            "recorded_sentence_prefix": entry.heard_prefix,
            "complete_conveyance": (
                entry.presentation == "complete"
                and entry.generated_text is not None
                and entry.heard_prefix == entry.generated_text
            ),
            "delegation_ids": list(entry.delegation_ids),
            "work_refs": list(entry.work_refs),
            "results": list(entry.results),
        }

    def _labeled(self, entry: Noted, *, include_generated: bool = False) -> str:
        text = entry.text
        if entry.kind is Heard.AGENT:
            receipt = self._playback(entry)
            receipt["results"] = transcript_results(entry.results)
            if include_generated:
                receipt["generated_text"] = entry.generated_text if entry.generated_text is not None else entry.text
            text = json.dumps(receipt, ensure_ascii=False)
        return f"{self.labels[entry.kind]}: {text}"

    def delegation_input(self, *, unheard: bool = False) -> tuple[list[dict], Carried]:
        """Return the caller transcript with recorded playback only, and its handoff mark."""
        entries = self._carried()
        earlier = [self._labeled(entry) for entry in entries if self._earlier(entry)]
        recent = [entry for entry in entries if not self._earlier(entry)]
        capped = [entry for entry in recent if not entry.pinned and entry.presentation != "not_heard"]
        # The newest caller and latest presentation receipt remain indivisible, even if together over the soft cap.
        kept = {
            entry.seq
            for kind in (Heard.CALLER, Heard.AGENT)
            if (entry := next((entry for entry in reversed(capped) if entry.kind is kind), None)) is not None
        }
        for state in ("not_conveyed", "unknown"):
            if disposition := next(
                (entry for entry in reversed(capped) if entry.kind is Heard.CONTEXT and entry.presentation == state),
                None,
            ):
                kept.add(disposition.seq)
        if not kept and capped:
            kept.add(capped[-1].seq)
        size = sum(len(self._labeled(entry)) + 1 for entry in capped if entry.seq in kept)
        # Keep the latest result's actual playback before optional context or acknowledgments can crowd it out.
        result = next((entry for entry in reversed(capped) if entry.result), None)
        if result is not None and result.seq not in kept:
            cost = len(self._labeled(result)) + 1
            if size + cost <= MAX_TRANSCRIPT_CHARS:
                kept.add(result.seq)
                size += cost
        for entry in reversed(capped):
            if entry.seq in kept:
                continue
            cost = len(self._labeled(entry)) + 1
            if size + cost > MAX_TRANSCRIPT_CHARS:
                break
            kept.add(entry.seq)
            size += cost
        # An unplayed attempt can use remaining space without displacing caller text or played lines.
        for entry in reversed(recent):
            if entry.presentation == "not_heard":
                cost = len(self._labeled(entry)) + 1
                if size + cost <= MAX_TRANSCRIPT_CHARS:
                    kept.add(entry.seq)
                    size += cost
        lines = []
        context: CallerContext = {"caller_turns": [], "playback": []}
        omitting = self.omitted_before is not None
        for entry in recent:
            if not entry.pinned and entry.seq not in kept:
                omitting = omitting or entry.presentation != "not_heard"
                continue
            if omitting:
                lines.append(self.words.omitted)
                omitting = False
            lines.append(self._labeled(entry))
            if entry.kind is Heard.CALLER:
                context["caller_turns"].append((entry.seq, entry.text))
            elif entry.kind is Heard.AGENT or entry.presentation == "not_conveyed":
                receipt = {"sequence": entry.seq, **self._playback(entry)}
                if entry.presentation in ("not_heard", "not_conveyed") and len(entry.results) > 1:
                    # Unspoken results can be reported independently to their jobs.
                    context["playback"].extend({**receipt, "results": [result]} for result in entry.results)
                else:
                    context["playback"].append(receipt)
        if omitting:
            lines.append(self.words.omitted)
        before = [self.words.earlier, *earlier] if earlier else []
        note = [self.words.unheard] if unheard else []
        text = "\n".join([*before, self.words.header, *lines, *note])
        caller_excerpt = "\n".join(entry.text for entry in recent if entry.kind is Heard.CALLER)
        items = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
                CALLER_EXCERPT: caller_excerpt[-CALLER_EXCERPT_CHARS:],
                CALLER_CONTEXT: context,
            }
        ]
        return items, Carried(self._noted)

    def _earlier(self, entry: Noted) -> bool:
        """Return whether ``entry`` is an instruction an accepted delegation carried: not new to the backend."""
        return entry.pinned and entry.seq < self._sent

    def sent(self, carried: Carried) -> None:
        """Remove the entries a delegation carried, but instructions; entries noted since its input was built stay."""
        self.entries = [entry for entry in self.entries if entry.seq >= carried.position or entry.pinned]
        self._sent = max(self._sent, carried.position)
        if self.omitted_before is not None and carried.position >= self.omitted_before:
            self.omitted_before = None

    async def delegate(
        self,
        delegation: Delegate,
        *,
        unheard: bool = False,
    ) -> str:
        """Delegate with everything noted so far and return the delegation id; a refusal raises and keeps the entries.

        ``unheard``: an answer the backend gave waits to be heard, which the transcript's last line then says.
        """
        items, carried = self.delegation_input(unheard=unheard)
        delegation_id = await delegation.delegate(items)
        self.sent(carried)
        return delegation_id
