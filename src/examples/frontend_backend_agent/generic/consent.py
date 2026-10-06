# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Per-call approvals that gate every client write (the consent gate).

The direct-write path only fast-tracks a call the caller has just approved
with a bare "yes". Nothing there stops the Thinker from planning a write
nobody approved. This book does:

* **record**: every consent question delivered to the caller records a pending
  approval for exactly one call, its tool and canonical arguments.
* **grant**: the caller's next turn grants it when it is affirmative with no
  negation, condition or change. Any other turn leaves it ungranted.
* **match**: a planned write is approved only by a granted approval for the
  same tool and identical canonical arguments.
* **consume**: an approval authorizes one dispatch.
* **expire**: another write, a session update, a changed or withdrawn request,
  or two further caller turns retire it.

A consent question whose values were summarized still records an approval:
the call it names is exact even when the spoken words are not, and a write it
blocked could otherwise never be approved. Its reply must have completed, so a
question cut off by the caller approves nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from loguru import logger

#: Leading words that make a reply affirmative.
_AFFIRMATIVE_OPENINGS: tuple[tuple[str, ...], ...] = tuple(
    tuple(phrase.split())
    for phrase in (
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "correct",
        "that's right",
        "thats right",
        "that is right",
        "that's correct",
        "go ahead",
        "do it",
        "please do",
        "please go ahead",
        "okay go ahead",
        "ok go ahead",
        "absolutely",
        "confirmed",
        "i confirm",
        "proceed",
        "please proceed",
    )
)
#: Words that make an affirmative opening conditional, negated or changed.
_RESERVATION_WORDS = frozenset(
    {
        "but",
        "no",
        "not",
        "don't",
        "dont",
        "wait",
        "hold",
        "except",
        "instead",
        "actually",
        "change",
        "different",
        "if",
        "unless",
        "only",
        "rather",
        "never",
        "wrong",
    }
)
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
#: Turns after the consent question during which an approval stays usable.
_APPROVAL_TURNS = 2


def is_affirmative(utterance: str) -> bool:
    """Return whether a reply opens with an affirmative and carries no negation, condition or change."""
    tokens = _WORD_RE.findall(str(utterance or "").casefold().replace("’", "'"))
    if not tokens or not _RESERVATION_WORDS.isdisjoint(tokens):
        return False
    return any(tuple(tokens[: len(opening)]) == opening for opening in _AFFIRMATIVE_OPENINGS)


@dataclass(slots=True)
class Approval:
    """One consent question's approval for exactly one call."""

    tool: str
    fingerprint: str
    run_id: str
    generation: int
    #: The caller's turn count when the question was recorded.
    turn: int
    granted: bool = False
    state: str = "pending"


class ConsentBook:
    """One session's write approvals."""

    def __init__(self) -> None:
        """Create an empty book."""
        self._approvals: list[Approval] = []

    def record(self, *, tool: str, fingerprint: str, run_id: str, generation: int, turn: int) -> None:
        """Record a pending approval for the call a consent question named; it replaces older pending ones."""
        self._retire(lambda approval: approval.state == "pending", "replaced")
        self._approvals.append(
            Approval(tool=tool, fingerprint=fingerprint, run_id=run_id, generation=generation, turn=turn)
        )

    def observe_turn(self, *, turn: int, utterance: str, reply_completed: bool | None, generation: int) -> None:
        """Grant or expire pending approvals as of the caller's latest turn.

        ``reply_completed`` is whether the reply carrying the question ended
        normally; None when the session cannot tell, which is then not held
        against it.
        """
        for approval in self._live():
            since = turn - approval.turn
            if approval.generation != generation:
                self._expire(approval, "session_updated")
            elif since > _APPROVAL_TURNS or (since > 1 and not approval.granted):
                self._expire(approval, "turns")
            elif since == 1 and not approval.granted:
                if reply_completed is False:
                    self._expire(approval, "not_heard")
                elif is_affirmative(utterance):
                    approval.granted = True
                    logger.bind(event="consent_gate", tool=approval.tool, outcome="granted").info(
                        f"Consent granted: tool={approval.tool}"
                    )
                else:
                    self._expire(approval, "not_affirmative")

    def take(self, fingerprint: str, generation: int) -> Approval | None:
        """Return the granted approval for exactly this call without consuming it, or None."""
        for approval in self._live():
            if approval.granted and approval.fingerprint == fingerprint and approval.generation == generation:
                return approval
        return None

    def consume(self, approval: Approval) -> None:
        """Use one approval for its single dispatch; every other approval expires with this write."""
        approval.state = "consumed"
        self._retire(lambda other: other is not approval and other.state == "pending", "other_write")

    def write_dispatched(self) -> None:
        """Expire every unconsumed approval; a write was sent."""
        self._retire(lambda approval: approval.state == "pending", "other_write")

    def expire_all(self, reason: str) -> None:
        """Expire every unconsumed approval (withdrawal, session update)."""
        self._retire(lambda approval: approval.state == "pending", reason)

    def expire_for_new_request(self, utterance: str) -> None:
        """Void approvals across a changed request, unless the new words answer the question affirmatively."""
        if is_affirmative(utterance):
            return
        self._retire(lambda approval: approval.state == "pending", "new_request")

    def _live(self) -> list[Approval]:
        return [approval for approval in self._approvals if approval.state == "pending"]

    def _retire(self, selector, reason: str) -> None:
        for approval in self._approvals:
            if selector(approval):
                self._expire(approval, reason)
        self._approvals = [approval for approval in self._approvals if approval.state == "pending"][-4:]

    @staticmethod
    def _expire(approval: Approval, reason: str) -> None:
        if approval.state != "pending":
            return
        approval.state = "expired"
        logger.bind(event="consent_gate", tool=approval.tool, outcome="expired", reason=reason).info(
            f"Consent expired: tool={approval.tool} reason={reason}"
        )
