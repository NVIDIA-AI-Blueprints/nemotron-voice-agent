# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared transcript: one append-only, seq-numbered record per session (plan section 7).

Every entry has an ``origin``, a ``route`` fixed at creation (how it reaches the
backend) and, for spoken entries, a playback ``outcome``. The frontend sees a full
re-projection on every call (:meth:`SharedTranscript.frontend_messages`); the backend
receives ``context`` entries exactly once, in ``seq`` order, once their outcome is
final (:meth:`SharedTranscript.take_context`). Hermes-native answers never travel;
a cut or undelivered one yields a ``delivery_note`` entry instead (plan 7.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Routes (plan 7.1).
RUN_INPUT = "run_input"
CONTEXT = "context"
NATIVE = "native"

#: Playback outcomes.
PENDING = "pending"
HEARD = "heard"
PARTIAL = "partial"
NOT_HEARD = "not_heard"

#: Entry kinds that are spoken assistant text.
SPOKEN_KINDS = ("frontend_speech", "status_speech", "controller_speech", "backend_answer")


@dataclass(slots=True)
class Entry:
    """One transcript entry."""

    seq: int
    kind: str
    origin: str
    text: str
    route: str
    turn_id: int | None = None
    outcome: str = HEARD
    heard_text: str = ""
    reason: str = ""
    answer_id: str = ""
    sent: bool = False
    # M2 (replay an unheard answer); voice-side only, never sent on the wire.
    run_id: str = ""  # the backend run that produced a backend_answer (gateway answer.run_id)
    released_mono: float | None = None  # first output_release of the item that speaks this entry
    replayed: bool = False

    @property
    def final(self) -> bool:
        """Whether the outcome can no longer change."""
        return self.outcome != PENDING

    @property
    def spoken_text(self) -> str:
        """What the user actually heard (full text, the heard prefix, or nothing)."""
        if self.outcome == HEARD:
            return self.text
        if self.outcome == PARTIAL:
            return self.heard_text
        return ""

    def wire(self) -> dict[str, Any]:
        """The ``history.append`` entry for the gateway."""
        data: dict[str, Any] = {
            "seq": self.seq,
            "origin": self.origin,
            "kind": self.kind,
            "text": self.text,
            "outcome": self.outcome,
        }
        if self.outcome == PARTIAL:
            data["heard_text"] = self.heard_text
        if self.answer_id:
            data["answer_id"] = self.answer_id
        if self.reason:
            data["reason"] = self.reason
        return data


class SharedTranscript:
    """Append-only transcript of one session."""

    def __init__(self) -> None:
        """Start empty; ``seq`` 0 is reserved for a seeded greeting."""
        self.entries: list[Entry] = []
        self._next_seq = 1

    def _append(self, entry: Entry) -> Entry:
        self.entries.append(entry)
        return entry

    def _seq(self) -> int:
        seq = self._next_seq
        self._next_seq += 1
        return seq

    # -- writers ---------------------------------------------------------------------------

    def seed_greeting(self, text: str) -> Entry:
        """The client's own greeting (tau2 greets in text): heard, seq 0, context route."""
        return self._append(Entry(seq=0, kind="frontend_speech", origin="frontend", text=text, route=CONTEXT))

    def add_user(self, text: str, *, turn_id: int, delegated: bool) -> Entry:
        """A user turn; the delegated one travels in ``delegate``, the others as context."""
        route = RUN_INPUT if delegated else CONTEXT
        return self._append(Entry(seq=self._seq(), kind="user", origin="user", text=text, route=route, turn_id=turn_id))

    def add_spoken(
        self,
        kind: str,
        origin: str,
        text: str,
        *,
        turn_id: int | None = None,
        answer_id: str = "",
        run_id: str = "",
    ) -> Entry:
        """Assistant speech queued for playback (outcome pending)."""
        if kind not in SPOKEN_KINDS:
            raise ValueError(f"not a spoken kind: {kind!r}")
        route = NATIVE if kind == "backend_answer" else CONTEXT
        return self._append(
            Entry(
                seq=self._seq(),
                kind=kind,
                origin=origin,
                text=text,
                route=route,
                turn_id=turn_id,
                outcome=PENDING,
                answer_id=answer_id,
                run_id=run_id,
            )
        )

    def settle(self, entry: Entry, outcome: str, *, heard_text: str = "", reason: str = "") -> Entry | None:
        """Fix a spoken entry's outcome once; returns the delivery note it creates, if any."""
        if entry.final:
            return None
        if outcome == PARTIAL and not heard_text.strip():
            outcome = NOT_HEARD
        entry.outcome = outcome
        entry.heard_text = heard_text.strip() if outcome == PARTIAL else ""
        entry.reason = reason
        if entry.kind != "backend_answer" or outcome == HEARD:
            return None
        return self._append(
            Entry(
                seq=self._seq(),
                kind="delivery_note",
                origin="derived",
                text=entry.text,
                route=CONTEXT,
                turn_id=entry.turn_id,
                outcome=outcome,
                heard_text=entry.heard_text,
                reason=reason,
                answer_id=entry.answer_id,
            )
        )

    # -- readers ---------------------------------------------------------------------------

    def take_context(self) -> list[Entry]:
        """Unsent ``context`` entries in ``seq`` order, stopping at the first one still pending.

        Not-heard frontend/controller lines are marked sent without travelling: they were
        never said (plan 7.4). Delivery notes always travel.
        """
        out: list[Entry] = []
        for entry in sorted(self.entries, key=lambda item: item.seq):
            if entry.route != CONTEXT or entry.sent:
                continue
            if not entry.final:
                break
            entry.sent = True
            if entry.kind in ("frontend_speech", "status_speech", "controller_speech") and entry.outcome == NOT_HEARD:
                continue
            out.append(entry)
        return out

    def frontend_messages(self, *, max_groups: int) -> list[dict[str, Any]]:
        """User/assistant messages for the frontend model: heard text only, alternating roles."""
        messages: list[dict[str, Any]] = []
        for entry in sorted(self.entries, key=lambda item: item.seq):
            if entry.kind == "user":
                role, text = "user", entry.text
            elif entry.kind in SPOKEN_KINDS:
                if entry.outcome == PENDING:
                    role, text = "assistant", f"(being spoken) {entry.text}"
                else:
                    role, text = "assistant", entry.spoken_text
                    if entry.outcome == PARTIAL:
                        text = f"{text} [cut off by the user]"
            else:
                continue
            if not text.strip():
                continue
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] = f"{messages[-1]['content']} {text}"
            else:
                messages.append({"role": role, "content": text})
        user_indexes = [index for index, message in enumerate(messages) if message["role"] == "user"]
        if len(user_indexes) > max_groups:
            messages = messages[user_indexes[-max_groups] :]
        while messages and messages[0]["role"] != "user":
            messages.pop(0)
        return messages

    def last_heard_assistant(self) -> Entry | None:
        """The newest assistant entry the user heard (fully or partly)."""
        for entry in reversed(self.entries):
            if entry.kind in SPOKEN_KINDS and entry.outcome in (HEARD, PARTIAL):
                return entry
        return None

    def last_backend_answer(self) -> Entry | None:
        """The newest backend answer entry (whatever its outcome)."""
        for entry in reversed(self.entries):
            if entry.kind == "backend_answer":
                return entry
        return None

    def backend_asked_question(self) -> bool:
        """Whether the last heard assistant message was the backend's, ending in a question."""
        entry = self.last_heard_assistant()
        return entry is not None and entry.kind == "backend_answer" and entry.spoken_text.rstrip().endswith("?")
