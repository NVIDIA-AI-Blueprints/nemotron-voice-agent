# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn router: decide whether a committed transcript is worth a frontend decision at all.

An optional gate in front of the talker (`LIVE_BRIDGE_TURN_ROUTER`). The talker cannot stay quiet,
so a bare "mm-hmm" or a stray "okay" would otherwise be answered or delegated. A dropped turn is
still recorded in the history as a user message, so a wrongly dropped request is recoverable the
next time the caller speaks; it just triggers no talker call, no reply, and no delegation.

The rule is deliberately conservative and deterministic, and it keeps whenever it is unsure: a
wasted delegation costs seconds, dead air costs the call. ASR confidence is not used because it
carries no usable signal here.

A turn is dropped only when every word is an acknowledgment or filler, it is at most
`MAX_WORDS` long, and it cannot be an answer. Answers are never dropped:
  * "yes", "yeah", "right" and similar count as answers when the assistant just asked a question
    or asked the caller to confirm;
  * nothing is dropped before the assistant has said anything;
  * digits, number words, correction words and question marks always keep the turn.
"""

import re
from dataclasses import dataclass

from examples.frontend_backend_live.common.words import continuer_vocabulary, words

MAX_WORDS = 4

# Acknowledgments that are a backchannel after a statement but an answer after a question.
AFFIRMATIONS = frozenset({"yes", "yeah", "yep", "yup", "right", "sure", "okay", "ok", "alright"})
FILLERS = frozenset({"uh", "um", "er", "erm", "ah", "oh", "hm", "hmm", "mm", "mmm", "mhm"})
NUMBER_WORDS = frozenset(
    [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
    ]
)
CORRECTION_WORDS = frozenset(
    {"actually", "sorry", "instead", "wait", "no", "nope", "not", "change", "different", "wrong"}
)


@dataclass(frozen=True)
class RouteVerdict:
    """Whether a turn reaches the talker, and why."""

    keep: bool
    reason: str


# Generic ways to ask for an answer without a question mark. Kept short and domain-neutral on
# purpose: missing one only means a reply is judged an answer anyway when unsure.
_REQUESTS_AN_ANSWER = re.compile(
    r"\b(please (confirm|let me know|tell me|say|reply)|let me know|shall i|should i|"
    r"go ahead|confirm (this|that|it|the)|just (say|confirm)|say yes|do you want)\b",
    re.IGNORECASE,
)


def invites_a_response(assistant_text: str | None) -> bool:
    """True when the assistant's last sentence asks the caller to answer or confirm."""
    if not assistant_text:
        return False
    tail = re.split(r"(?<=[.!?])\s+", assistant_text.strip())[-1]
    return tail.rstrip().endswith("?") or bool(_REQUESTS_AN_ANSWER.search(tail))


class TurnRouter:
    """Decides whether a committed transcript is worth a talker decision at all."""

    def __init__(self, continuers=()):
        """Build the droppable vocabulary from the configured continuer phrases and the fillers."""
        self.droppable = continuer_vocabulary(continuers) | FILLERS

    def verdict(self, text: str, last_assistant_text: str | None) -> RouteVerdict:
        """Return a :class:`RouteVerdict` for ``text`` given the assistant's last words."""
        if not re.search(r"\w", text):
            # Only punctuation survived transcription: there is nothing to act on.
            return RouteVerdict(False, "empty")
        if re.search(r"\d|\?", text):
            return RouteVerdict(True, "digit_or_question")
        spoken = words(text)
        if not spoken:
            return RouteVerdict(True, "unrecognized_script")
        if len(spoken) > MAX_WORDS:
            return RouteVerdict(True, "long")
        if any(w in CORRECTION_WORDS or w in NUMBER_WORDS for w in spoken):
            return RouteVerdict(True, "correction_or_number")
        if last_assistant_text is None:
            return RouteVerdict(True, "no_assistant_turn")
        if not all(w in self.droppable or w in AFFIRMATIONS for w in spoken):
            return RouteVerdict(True, "content")
        if any(w in AFFIRMATIONS for w in spoken) and invites_a_response(last_assistant_text):
            return RouteVerdict(True, "answers_a_question")
        return RouteVerdict(False, "acknowledgment")
