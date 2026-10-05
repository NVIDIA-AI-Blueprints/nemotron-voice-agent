# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Spoken-identifier normalization for a user transcript.

A caller spelling an identifier produces a run of spelled tokens: single letters,
digits and digit words, separator words, "double"/"triple" repeats, a leading
"hash"/"pound", and "S as in Sam". :func:`normalize_transcript` writes each run
as the identifier it spells and leaves every other character of the text alone::

    "my code is x b seven t p two"                    -> "my code is XB7TP2"
    "it's jordan underscore lee underscore eight two" -> "it's jordan_lee_82"

A run needs at least three spelled tokens and one that is not the ordinary words
"a" or "I". Whole words join a run only beside "underscore", and the run is then
written in lower-case snake style; otherwise letters are capitals. Teen and tens
words ("eighty two") count only inside such a snake run, so a time such as
"ten twenty five" is never joined.

The helpers for turn boundaries (:func:`join_across_turns`,
:func:`ends_mid_spelling`, :func:`matches_complete_pattern`) let a caller hold
or merge a spelling a pause split across two transcripts.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from examples.frontend_backend_agent.src.normalization.rules import (
    CONTEXT_DIGIT_WORDS,
    DIGIT_WORDS,
    EDGE_PUNCTUATION,
    FILLER_WORDS,
    PREFIX_WORDS,
    REPEAT_WORDS,
    SEPARATOR_WORDS,
    TEEN_WORDS,
    TENS_WORDS,
    is_number_word,
    number_words_to_digits,
    single_char,
)

#: A run needs this many spelled tokens.
MIN_RUN_TOKENS = 3
#: Single letters that are also ordinary words; they never make a run on their own.
ORDINARY_LETTERS = frozenset({"a", "i"})
#: Spelled tokens a trailing character needs before it to count as mid-spelling.
MID_SPELLING_EVIDENCE = 2

_TOKEN = re.compile(r"\S+")
_DOTTED_LETTERS = re.compile(r"[A-Za-z](?:\.[A-Za-z])+")
_SNAKE_SEPARATOR = "_"


@dataclass(frozen=True, slots=True)
class _Token:
    """One whitespace token: its core (edge punctuation removed) and where the core sits."""

    core: str
    start: int
    end: int

    @property
    def word(self) -> str:
        """The lower-cased core for matching spoken words.

        A multi-letter all-capitals token is written letters (a joined run such as
        "TWO" or "DASH"), never a spoken word, so it matches none.
        """
        if len(self.core) > 1 and self.core.isupper():
            return ""
        return self.core.lower()

    @property
    def transparent(self) -> bool:
        """Punctuation or a filler: skipped inside a run without ending it."""
        return not self.core or self.core.lower() in FILLER_WORDS


@dataclass(frozen=True, slots=True)
class _Unit:
    """One spelled element of a run and the tokens it covers (inclusive indices)."""

    text: str
    kind: Literal["char", "word", "separator", "prefix"]
    weight: int
    first: int
    last: int


def _tokenize(text: str) -> tuple[_Token, ...]:
    tokens: list[_Token] = []
    for match in _TOKEN.finditer(text):
        raw = match.group()
        core = raw.strip(EDGE_PUNCTUATION)
        start = match.start() + len(raw) - len(raw.lstrip(EDGE_PUNCTUATION))
        tokens.append(_Token(core=core, start=start, end=start + len(core)))
    return tuple(tokens)


def _next(tokens: Sequence[_Token], index: int) -> int | None:
    for candidate in range(index + 1, len(tokens)):
        if not tokens[candidate].transparent:
            return candidate
    return None


def _previous(tokens: Sequence[_Token], index: int) -> int | None:
    for candidate in range(index - 1, -1, -1):
        if not tokens[candidate].transparent:
            return candidate
    return None


def _word_at(tokens: Sequence[_Token], index: int | None) -> str:
    return tokens[index].word if index is not None else ""


def _is_digit_token(word: str) -> bool:
    return (word.isascii() and word.isdigit()) or (word in DIGIT_WORDS and word not in CONTEXT_DIGIT_WORDS)


def _digit_follows(tokens: Sequence[_Token], index: int) -> bool:
    """A digit (or a repeat word) comes next, past any further context digit words ("oh oh seven")."""
    following = _next(tokens, index)
    while following is not None and tokens[following].word in CONTEXT_DIGIT_WORDS:
        following = _next(tokens, following)
    word = _word_at(tokens, following)
    return _is_digit_token(word) or word in REPEAT_WORDS


def _unit(tokens: Sequence[_Token], index: int, run: Sequence[_Unit]) -> _Unit | None:
    """The spelled element starting at ``index`` given the run so far, or ``None``."""
    token = tokens[index]
    word, core = token.word, token.core
    following = _next(tokens, index)
    following_word = _word_at(tokens, following)
    if word in SEPARATOR_WORDS:
        return _Unit(SEPARATOR_WORDS[word], "separator", 1, index, index)
    if word in PREFIX_WORDS:
        return None if run else _Unit(PREFIX_WORDS[word], "prefix", 1, index, index)
    if word in REPEAT_WORDS and following is not None and (char := single_char(following_word)) is not None:
        count = REPEAT_WORDS[word]
        return _Unit(char * count, "char", count, index, following)
    if _DOTTED_LETTERS.fullmatch(core):
        letters = core.replace(".", "")
        return _Unit(letters, "char", len(letters), index, index)
    if len(core) == 1 and core.isascii() and core.isalpha():
        last = index
        second = _next(tokens, following) if following is not None else None
        third = _next(tokens, second) if second is not None else None
        if following_word == "as" and _word_at(tokens, second) == "in" and _word_at(tokens, third).isalpha():
            last = third  # "S as in Sam"
        return _Unit(core, "char", 1, index, last)
    if _is_digit_token(word):
        return _Unit(DIGIT_WORDS.get(word, core), "char", 1, index, index)
    if word in CONTEXT_DIGIT_WORDS:
        after_digit = bool(run) and run[-1].text[-1:].isdigit()
        if after_digit or _digit_follows(tokens, index):
            return _Unit(DIGIT_WORDS[word], "char", 1, index, index)
        return None
    snake = any(unit.text == _SNAKE_SEPARATOR for unit in run)
    if snake and is_number_word(word):
        words, last = [word], index
        if (
            word in TENS_WORDS
            and following is not None
            and _is_digit_token(following_word)
            and not following_word.isdigit()
        ):
            words, last = [word, following_word], following
        digits = number_words_to_digits(words)
        if digits is not None and (word in TEEN_WORDS or word in TENS_WORDS or "-" in word):
            return _Unit(digits, "char", 1, index, last)
    if core.isalpha() and len(core) > 1:
        beside = {_word_at(tokens, _previous(tokens, index)), following_word}
        if any(SEPARATOR_WORDS.get(neighbour) == _SNAKE_SEPARATOR for neighbour in beside):
            return _Unit(core.lower(), "word", 1, index, index)
    return None


def _scan(tokens: Sequence[_Token]) -> list[tuple[_Unit, ...]]:
    """Every maximal run of spelled elements, of any length."""
    runs: list[tuple[_Unit, ...]] = []
    run: list[_Unit] = []
    index = 0
    while index < len(tokens):
        if tokens[index].transparent:
            index += 1
            continue
        unit = _unit(tokens, index, run)
        if unit is None:
            if run:  # close the run, then let this token start the next one ("1 2 3 hash 4 5 6")
                runs.append(tuple(run))
                run = []
                continue
            index += 1
            continue
        run.append(unit)
        index = unit.last + 1
    if run:
        runs.append(tuple(run))
    return runs


def _weight(units: Sequence[_Unit]) -> int:
    return sum(unit.weight for unit in units)


def _is_spelling(units: Sequence[_Unit]) -> bool:
    """At least three spelled tokens, one of them more than an ordinary "a" or "I"."""
    return _weight(units) >= MIN_RUN_TOKENS and any(
        unit.kind in ("char", "word") and unit.text.lower() not in ORDINARY_LETTERS for unit in units
    )


def _value(units: Sequence[_Unit]) -> str:
    text = "".join(unit.text for unit in units)
    return text.lower() if _SNAKE_SEPARATOR in text else text.upper()


def _span(tokens: Sequence[_Token], units: Sequence[_Unit]) -> tuple[int, int]:
    return tokens[units[0].first].start, tokens[units[-1].last].end


def spelled_runs(text: str) -> list[tuple[int, int, str]]:
    """``(start, end, value)`` for every spelled run in ``text``; ``text[start:end]`` is what was said."""
    tokens = _tokenize(text)
    return [(*_span(tokens, units), _value(units)) for units in _scan(tokens) if _is_spelling(units)]


def normalize_transcript(text: str) -> str:
    """``text`` with every spelled run written as the identifier it spells (idempotent)."""
    pieces: list[str] = []
    cursor = 0
    for start, end, value in spelled_runs(text):
        pieces.extend((text[cursor:start], value))
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _last_solid(tokens: Sequence[_Token]) -> int | None:
    return _previous(tokens, len(tokens))


def _trailing_run(tokens: Sequence[_Token]) -> tuple[_Unit, ...] | None:
    """The run that ends at the text's last token, if the text ends in spelling."""
    last = _last_solid(tokens)
    runs = _scan(tokens)
    if last is None or not runs or runs[-1][-1].last != last:
        return None
    return runs[-1]


def join_across_turns(previous: str, current: str) -> str | None:
    """The identifier spelled across two transcripts, when ``previous`` ends and ``current`` starts mid-run."""
    previous_tokens, current_tokens = _tokenize(previous), _tokenize(current)
    trailing = _trailing_run(previous_tokens)
    first = _next(current_tokens, -1)
    current_runs = _scan(current_tokens)
    if trailing is None or first is None or not current_runs or current_runs[0][0].first != first:
        return None
    head_start, head_end = _span(previous_tokens, trailing)
    tail_start, tail_end = _span(current_tokens, current_runs[0])
    combined = f"{previous[head_start:head_end]} {current[tail_start:tail_end]}"
    found = spelled_runs(combined)
    if len(found) == 1 and found[0][0] == 0 and found[0][1] == len(combined):
        return found[0][2]
    return None


def ends_mid_spelling(text: str) -> bool:
    """Whether ``text`` stops inside a spelling.

    The last token spells one character (a letter, a digit or a digit word) or is
    a separator word, and at least two spelled tokens run right up to it.
    """
    trailing = _trailing_run(_tokenize(text))
    if trailing is None:
        return False
    last = trailing[-1]
    one_char = last.kind == "separator" or (last.kind == "char" and len(last.text) == 1)
    return one_char and _weight(trailing[:-1]) >= MID_SPELLING_EVIDENCE


def matches_complete_pattern(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    """Whether the value that ends the normalized text fully matches one of ``patterns``."""
    if not patterns:
        return False
    tokens = _tokenize(normalize_transcript(text))
    last = _last_solid(tokens)
    if last is None:
        return False
    value = tokens[last].core
    return any(pattern.fullmatch(value) for pattern in patterns)
