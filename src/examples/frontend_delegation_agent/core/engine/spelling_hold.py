# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence-gated spelling hold.

A turn is held for ``delegation.spelling_hold.hold_ms`` before the frontend decides it
only if all three conditions hold:

1. **It ends mid-spelling:** the last word is a single character (a letter, one digit
   or a digit word) or a separator word ("underscore"). Punctuation and fillers are ignored.
2. **It has strong spelling evidence**, at least one of:
   ``trailing_run`` (the trailing run of single characters has two or more tokens and a
   letter), ``separator`` (a separator word among the last four words), ``all_spelled``
   (two or more words, all single characters or separators, with a letter) or
   ``accumulator`` (the previous turn was merged by a hold).
3. **The value looks incomplete:** the value at the end of the turn (the last anchored
   span, else the joined trailing run) full-matches neither a configured tool-argument
   rule pattern nor one of ``complete_patterns`` (``schema_patterns``, derived from the
   session's identifier examples, match case-insensitively). Schema-derived rules only
   reject clearly incomplete values, so their patterns are never used here.

If speech starts during the hold, the turn manager's cancel-and-merge path joins the
words to the next turn. Pure: no I/O, no clock.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

from examples.frontend_delegation_agent.core.normalization.arguments import ArgumentNormalizer
from examples.frontend_delegation_agent.core.normalization.rules import Token, tokenize
from examples.frontend_delegation_agent.core.normalization.transcript import (
    SpelledRunSettings,
    TranscriptNormalizer,
    TranscriptSettings,
)

#: Why a turn was not held.
NOT_MID_SPELLING = "not_mid_spelling"
NO_EVIDENCE = "no_evidence"
COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class HoldCheck:
    """The predicate's verdict for one turn."""

    hold: bool
    evidence: str = ""
    value: str = ""
    reason: str = ""


class SpellingHoldPredicate:
    """Decides whether a turn that ends mid-spelling should wait for its continuation."""

    def __init__(
        self,
        transcript: TranscriptSettings,
        *,
        complete_patterns: Sequence[str] = (),
        schema_patterns: Sequence[str] = (),
        arguments: ArgumentNormalizer | None = None,
    ) -> None:
        """``transcript`` supplies the separator words and word lists; ``arguments`` the ID patterns."""
        # Anchored spans only, lower case like the tool-argument rules; spelled runs are joined here.
        self._anchored = TranscriptNormalizer(replace(transcript, enabled=True, spelled_runs=SpelledRunSettings()))
        self._rules = transcript.resolved_ruleset()
        self._separators = {word.lower() for word in transcript.separator_words}
        self._patterns = tuple(re.compile(pattern) for pattern in complete_patterns) + tuple(
            re.compile(pattern, re.IGNORECASE) for pattern in schema_patterns
        )
        self._arguments = arguments

    def check(self, text: str, *, accumulator: bool = False) -> HoldCheck:
        """Hold ``text`` or not, with the evidence (held) or the reason (not held)."""
        tokens = [token for token in tokenize(text) if token.core and not token.is_filler(self._rules)]
        if not tokens or not (self._single(tokens[-1]) or self._separator(tokens[-1])):
            return HoldCheck(hold=False, reason=NOT_MID_SPELLING)
        run = self._trailing_run(tokens)
        evidence = ""
        if len(run) >= 2 and any(self._letter(token) for token in run):
            evidence = "trailing_run"
        elif any(self._separator(token) for token in tokens[-4:]):
            evidence = "separator"
        elif (
            len(tokens) >= 2
            and all(self._single(token) or self._separator(token) for token in tokens)
            and any(self._letter(token) for token in tokens)
        ):
            evidence = "all_spelled"
        elif accumulator:
            evidence = "accumulator"
        if not evidence:
            return HoldCheck(hold=False, reason=NO_EVIDENCE)
        value = self._value(text, tokens, run)
        if value and self._complete(value):
            return HoldCheck(hold=False, evidence=evidence, value=value, reason=COMPLETE)
        return HoldCheck(hold=True, evidence=evidence, value=value)

    # -- pieces --------------------------------------------------------------------------------

    def _single(self, token: Token) -> bool:
        """One letter, one digit, or a digit word."""
        return (len(token.core) == 1 and token.core.isalnum()) or token.word in self._rules.digits

    def _char(self, token: Token) -> str:
        return token.core if len(token.core) == 1 else self._rules.digits.get(token.word, "")

    def _letter(self, token: Token) -> bool:
        return len(token.core) == 1 and token.core.isalpha()

    def _separator(self, token: Token) -> bool:
        return token.word in self._separators

    def _trailing_run(self, tokens: Sequence[Token]) -> list[Token]:
        run: list[Token] = []
        for token in reversed(tokens):
            if not self._single(token):
                break
            run.insert(0, token)
        return run

    def _value(self, text: str, tokens: Sequence[Token], run: Sequence[Token]) -> str:
        """The value at the end of the turn: the last anchored span if it ends the turn, else the trailing run."""
        normalized = self._anchored.normalize(text)
        if normalized.spans and normalized.spans[-1].end >= tokens[-1].end:
            return normalized.spans[-1].written
        return "".join(self._char(token) for token in run)

    def _complete(self, value: str) -> bool:
        if any(pattern.fullmatch(value) for pattern in self._patterns):
            return True
        if self._arguments is None:
            return False
        for rule in self._arguments.settings.rules:
            if (
                rule.source == "config"
                and rule.pattern
                and self._arguments.valid(rule, self._arguments.canonical(rule, value))
            ):
                return True
        return False
