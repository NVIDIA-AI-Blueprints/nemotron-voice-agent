# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The heard log: what the caller said and heard since the last delegation, which the next delegation carries."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Protocol

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.prompts import PromptSet
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.spoken import Line, Outcome, transcript_results
from nemotron_voice_agent.pipecat.shared.delegation_contract import (
    CALLER_CONTEXT,
    CALLER_EXCERPT,
    CALLER_EXCERPT_CHARS,
    CallerContext,
)

MAX_TRANSCRIPT_CHARS = 6000
_PLAYBACK_STATES = {
    Outcome.HANDED: "queued",
    Outcome.PLAYING: "playing",
    Outcome.SYNTHESIS_FAILED: "partial",
    Outcome.CUT: "interrupted",
    Outcome.NOT_HEARD: "not_heard",
}


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

    def export_attachment(self, in_flight: set[int], max_items: int) -> dict:
        """Keep settled transcript evidence and its handoff cursor for reconnect."""
        entries = [entry for entry in self._carried() if entry.line not in in_flight]
        omitted_before = self.omitted_before
        if len(entries) > max_items:
            omitted_before = max(omitted_before or 0, entries[-max_items - 1].seq + 1)
        return {
            "heard_sent": self._sent,
            "heard": [asdict(entry) for entry in entries[-max_items:]],
            "heard_omitted_before": omitted_before,
        }

    def restore_attachment(self, snapshot: dict) -> None:
        """Restore the transcript and continue its sequence after the retained entries."""
        self._sent = snapshot.get("heard_sent", 0)
        self.omitted_before = snapshot.get("heard_omitted_before")
        self.entries = [Noted(**{**entry, "kind": Heard(entry["kind"])}) for entry in snapshot["heard"]]
        self._noted = max(
            self._sent, self.omitted_before or 0, max((entry.seq for entry in self.entries), default=-1) + 1
        )

    def caller(self, text: str) -> int:
        """Note a caller turn that ended and return its sequence."""
        return self._note(Heard.CALLER, text)

    def presented(self, line: Line, evidence: dict, *, replaces: int | None = None) -> None:
        """Journal the line's playback disposition and publication identities for the next delegation."""
        if line.outcome is Outcome.HEARD:
            state = "complete" if evidence["complete_conveyance"] else "partial"
        else:
            state = _PLAYBACK_STATES[line.outcome]
        text = line.text
        if state in ("interrupted", "partial"):
            text = line.heard.rstrip() + self.cut_mark if state == "interrupted" else line.heard
        fields = {
            "line": line.number,
            "result": line.results,
            "cut": state in ("interrupted", "partial"),
            "presentation": state,
            "generated_text": line.text,
            "heard_prefix": line.heard,
            "audio_started": line.audio_started,
            "line_kind": line.kind.value,
            "delegation_ids": line.delegation_ids,
            "work_refs": tuple(evidence["context_work_refs"]),
            "results": tuple(dict(item) for item in evidence["results"]),
        }
        previous = line.number if replaces is None else replaces
        entry = next((entry for entry in self.entries if entry.line == previous), None)
        if entry is None:
            self._note(Heard.AGENT, text, **fields)
        else:
            self._update(entry, text=text, **fields)

    def context(self, text: str) -> None:
        """Note silent application context (``session.thinking.append``)."""
        self._note(Heard.CONTEXT, text)

    def routing_failed(self, reason: str, caller_notes: set[int]) -> None:
        """Replace any unsent notes for these turns with the latest routing disposition."""
        self.entries = [
            entry
            for entry in self.entries
            if entry.presentation != "routing_failed"
            and (entry.kind is not Heard.CALLER or entry.seq not in caller_notes)
        ]
        text = json.dumps({"frontend_routing": {"state": "not_routed", "reason": reason}})
        self._note(Heard.CONTEXT, text, presentation="routing_failed")

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

    def _note(self, kind: Heard, text: str, **fields) -> int:
        sequence = self._noted
        self.entries.append(Noted(sequence, kind, text, **fields))
        self._noted += 1
        return sequence

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
        for state in ("not_conveyed", "unknown", "routing_failed"):
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
