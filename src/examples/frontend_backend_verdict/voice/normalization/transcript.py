# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ASR hook: rewrite spoken identifiers in a transcript into written form.

Inverse text normalization scoped to identifiers. Only a span anchored by a
separator word ("underscore") is rewritten, so ordinary speech ("I need two
passengers") is never touched::

    "my user ID is Jordan underscore Lee underscore eight two."
    -> "my user ID is jordan_lee_82."       (case: lower)

A span is built around a run of anchors: every token between two anchors is part
of the ID; left of the first anchor, the adjacent word plus any spelled letters
before it; right of the last anchor, a run of number words (the digit tail) or
the adjacent word (or a run of spelled letters). Stop words, fillers and
punctuation-only tokens end the outer parts. See
``misc/prototypes/voice/normalization-plan.md`` section 4.2.

Optionally (``spelled_runs``, off by default), a run of spelled single characters in
the text the anchor pass left alone is joined too; the run needs at least one letter,
and any word of two or more letters ends it::

    "X, B, seven, T, P, two and Q, U, E, N" -> "XB7TP2 and QUEN"   (case: keep)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from examples.frontend_backend_verdict.voice.normalization.rules import (
    SHORT_TOKEN_LETTERS,
    Ruleset,
    Token,
    apply_case,
    get_ruleset,
    is_number_word,
    is_short,
    join_part,
    tokenize,
)

CASES = ("keep", "lower", "upper")


@dataclass(frozen=True, slots=True)
class SpelledRunSettings:
    """``normalization.transcript.spelled_runs``: join spelled letters and digits outside anchored spans."""

    enabled: bool = False
    min_tokens: int = 3
    case: str = "keep"


@dataclass(frozen=True, slots=True)
class TranscriptSettings:
    """``normalization.transcript``."""

    enabled: bool = False
    separator_words: Mapping[str, str] = field(default_factory=lambda: {"underscore": "_"})
    ruleset: str = "en"
    number_words: bool = True
    compound_numbers: bool = True
    extra_filler_words: tuple[str, ...] = ()
    extra_stop_words: tuple[str, ...] = ()
    max_part_tokens: int = 6
    case: str = "keep"
    frontend_note_key: str = ""
    spelled_runs: SpelledRunSettings = field(default_factory=SpelledRunSettings)

    def resolved_ruleset(self) -> Ruleset:
        """The named ruleset extended with this config's extra words."""
        return get_ruleset(self.ruleset).extended(fillers=self.extra_filler_words, stop_words=self.extra_stop_words)


@dataclass(frozen=True, slots=True)
class Span:
    """One rewritten identifier: the spoken text and its written form."""

    spoken: str
    written: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class NormalizedText:
    """A transcript after normalization, with what was rewritten."""

    text: str
    spans: tuple[Span, ...] = ()

    @property
    def changed(self) -> bool:
        """Whether any span was rewritten."""
        return bool(self.spans)


class TranscriptNormalizer:
    """Rewrites anchored identifier spans; everything else in the text is kept byte for byte."""

    def __init__(self, settings: TranscriptSettings) -> None:
        """Resolve the ruleset once."""
        self._settings = settings
        self._rules = settings.resolved_ruleset()
        self._separators = {word.lower(): char for word, char in settings.separator_words.items()}

    @property
    def settings(self) -> TranscriptSettings:
        """The settings this normalizer was built from."""
        return self._settings

    def normalize(self, text: str) -> NormalizedText:
        """Rewrite every anchored span in ``text``."""
        tokens = tokenize(text)
        spans: list[Span] = []
        floor = 0
        for group in self._anchor_groups(tokens):
            span, next_floor = self._span(text, tokens, group, floor)
            if span is not None:
                spans.append(span)
                floor = next_floor
        if self._settings.spelled_runs.enabled:
            spans = sorted([*spans, *self._spelled_runs(text, tokens, spans)], key=lambda span: span.start)
        if not spans:
            return NormalizedText(text)
        pieces: list[str] = []
        cursor = 0
        for span in spans:
            pieces.extend((text[cursor : span.start], span.written))
            cursor = span.end
        pieces.append(text[cursor:])
        return NormalizedText("".join(pieces), tuple(spans))

    # -- spelled runs --------------------------------------------------------------

    def spelled_char(self, token: Token) -> str | None:
        """The character a spelled token stands for: a letter, a digit or a digit word (else ``None``)."""
        word = token.word
        if token.spelled:
            return token.core
        if len(token.core) == 1 and token.core.isalnum():
            return token.core
        if self._settings.number_words and word in self._rules.digits and word not in self._rules.run_only_digits:
            return self._rules.digits[word]
        return None

    def _spelled_runs(self, text: str, tokens: Sequence[Token], anchored: Sequence[Span]) -> list[Span]:
        """Runs of spelled characters outside the anchored spans (fillers and commas may sit inside)."""
        settings = self._settings.spelled_runs
        taken = [(span.start, span.end) for span in anchored]
        out: list[Span] = []
        run: list[tuple[Token, str]] = []

        def close() -> None:
            chars = [char for _, char in run]
            if len(chars) >= settings.min_tokens and any(char.isalpha() for char in "".join(chars)):
                start, end = run[0][0].start, run[-1][0].end
                spoken = text[start:end]
                out.append(Span(spoken=spoken, written=apply_case("".join(chars), settings.case), start=start, end=end))
            run.clear()

        for token in tokens:
            if any(start <= token.start < end for start, end in taken):
                close()
                continue
            if not token.core or (run and token.is_filler(self._rules)):
                continue  # commas and "uh" inside a run
            char = self.spelled_char(token)
            if char is None:
                close()
                continue
            run.append((token, char))
        close()
        return out

    # -- span construction -------------------------------------------------------

    def _is_anchor(self, token: Token) -> bool:
        return token.word in self._separators

    def _content(self, tokens: Sequence[Token]) -> int:
        """Tokens counted against ``max_part_tokens``: single spelled letters are free."""
        return sum(1 for token in tokens if token.core and not token.is_filler(self._rules) and len(token.core) > 1)

    def _anchor_groups(self, tokens: Sequence[Token]) -> list[list[int]]:
        groups: list[list[int]] = []
        for index, token in enumerate(tokens):
            if not self._is_anchor(token):
                continue
            if groups and self._content(tokens[groups[-1][-1] + 1 : index]) <= self._settings.max_part_tokens:
                groups[-1].append(index)
            else:
                groups.append([index])
        return groups

    def _stops(self, token: Token) -> bool:
        """Ends an outer part: punctuation-only, a stop word, a filler, or another anchor."""
        word = token.word
        return (
            not token.core or word in self._rules.stop_words or token.is_filler(self._rules) or self._is_anchor(token)
        )

    def _left(self, tokens: Sequence[Token], first: int, floor: int) -> list[int]:
        index = first - 1
        if index < floor or self._stops(tokens[index]):
            return []
        # After a whole word only single letters join it ("A a rav" -> "Aarav"); after a
        # short token, any short tokens do ("J am es" -> "James", "S O P H I A").
        limit = 1 if not is_short(tokens[index].word) else SHORT_TOKEN_LETTERS
        taken = [index]
        index -= 1
        while index >= floor and self._content([tokens[i] for i in taken]) < self._settings.max_part_tokens:
            token = tokens[index]
            if not token.core or self._is_anchor(token) or not token.word.isalpha() or len(token.word) > limit:
                break
            if token.word in self._rules.stop_words and len(token.word) > 1:
                break
            taken.insert(0, index)
            index -= 1
        return taken

    def _right(self, tokens: Sequence[Token], last: int) -> list[int]:
        index = last + 1
        while index < len(tokens) and tokens[index].core and tokens[index].is_filler(self._rules):
            index += 1
        if index >= len(tokens) or self._stops(tokens[index]):
            return []
        compound = self._settings.compound_numbers
        if self._settings.number_words and is_number_word(tokens[index].word, self._rules, compound=compound):
            taken: list[int] = []
            while index < len(tokens):
                token = tokens[index]
                if token.core and is_number_word(token.word, self._rules, compound=compound):
                    taken.append(index)
                elif not (token.core and token.is_filler(self._rules)):
                    break
                index += 1
            return taken
        if not is_short(tokens[index].word):
            return [index]
        taken = [index]
        index += 1
        while index < len(tokens) and self._content([tokens[i] for i in taken]) < self._settings.max_part_tokens:
            token = tokens[index]
            if self._stops(token) or not is_short(token.word):
                break
            taken.append(index)
            index += 1
        return taken

    def _span(self, text: str, tokens: Sequence[Token], group: list[int], floor: int) -> tuple[Span | None, int]:
        left = self._left(tokens, group[0], floor)
        right = self._right(tokens, group[-1])
        parts = [left, *[list(range(a + 1, b)) for a, b in zip(group, group[1:], strict=False)], right]
        number_words, compound = self._settings.number_words, self._settings.compound_numbers
        pieces: list[str] = []
        for position, part in enumerate(parts):
            pieces.append(
                join_part([tokens[i] for i in part], self._rules, number_words=number_words, compound=compound)
            )
            if position < len(group):
                pieces.append(self._separators[tokens[group[position]].word])
        written = "".join(pieces)
        if not any(char.isalnum() for char in written):
            return None, floor
        first = left[0] if left else group[0]
        last = right[-1] if right else group[-1]
        start, end = tokens[first].start, tokens[last].end
        span = Span(spoken=text[start:end], written=apply_case(written, self._settings.case), start=start, end=end)
        return span, last + 1
