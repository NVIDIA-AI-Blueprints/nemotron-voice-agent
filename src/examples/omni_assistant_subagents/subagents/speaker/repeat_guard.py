# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Detection of verbatim Speaker repeats and the follow-ups that excuse them."""

from __future__ import annotations

import re
from collections import deque

BRIDGE_FILLERS = (
    "Hmm, let me look back over our conversation for a second.",
    "Let me think this through more carefully.",
    "One moment — let me reconsider that.",
)

AFFIRMATION_TOKENS = frozenset(
    {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please", "go", "do", "fine", "alright", "right"}
)

_REPEAT_MIN_WORDS = 4
_REPEAT_HISTORY = 4


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for repeat comparison."""
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


def is_affirmation(transcript: str) -> bool:
    """Whether a user turn is a short agreement (a follow-up, not a "stuck" signal)."""
    words = normalize_text(transcript).split()
    if not words or len(words) > 6:
        return False
    return words[0] in AFFIRMATION_TOKENS or AFFIRMATION_TOKENS.issuperset(set(words))


class RepeatGuard:
    """Detects verbatim response repeats and bridges them with a rotating filler.

    The model can restate the same line even when it believes the turn went fine,
    so repetition is detected deterministically here to force a Thinker escalation.
    """

    def __init__(self) -> None:
        """Start with empty reply history."""
        self._recent: deque[str] = deque(maxlen=_REPEAT_HISTORY)
        self._pending = ""
        self._filler_index = 0
        self.suppressing = False
        self.detected = False
        self.filler = ""
        self.emitted = False

    def bridge_filler(self, text: str) -> str | None:
        """Return a bridging filler when ``text`` is the turn's first chunk and repeats a recent reply."""
        if self.emitted or self.suppressing or not self._is_repeat(text):
            return None
        self.suppressing = True
        self.detected = True
        self.emitted = True
        self.filler = BRIDGE_FILLERS[self._filler_index % len(BRIDGE_FILLERS)]
        self._filler_index += 1
        return self.filler

    def could_be_repeat_prefix(self, text: str) -> bool:
        """Whether an incomplete streamed reply could still become an exact recent repeat."""
        normalized = normalize_text(text)
        if not normalized:
            return bool(self._recent)
        return any(recent.startswith(normalized) for recent in self._recent)

    def note_spoken(self, text: str) -> None:
        """Record a reply that already streamed to TTS.

        Bridging it is no longer possible, but the repeat still has to be seen so
        the turn escalates to the Thinker.
        """
        self.emitted = True
        if self._is_repeat(text):
            self.detected = True

    def _is_repeat(self, text: str) -> bool:
        normalized = normalize_text(text)
        if len(normalized.split()) < _REPEAT_MIN_WORDS:
            return False
        return normalized in self._recent

    def note_reply(self, response: str, *, track: bool) -> None:
        """Hold the model's own reply so the next turn can detect a repeat.

        It stays pending until :meth:`reset` starts the next turn, so a reply
        that has not finished reaching TTS is never compared against itself.
        """
        normalized = normalize_text(response)
        self._pending = normalized if track else ""

    def reset(self) -> None:
        """Start a new turn, committing the previous turn's reply to history."""
        if self._pending:
            self._recent.append(self._pending)
            self._pending = ""
        self.suppressing = False
        self.detected = False
        self.filler = ""
        self.emitted = False
