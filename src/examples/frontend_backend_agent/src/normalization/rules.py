# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Identifier shapes learned from one quoted schema example, and the spoken-word tables.

An :class:`IdentifierRule` describes the written shape of one argument: the
separator between its segments, its letter case, the characters a dictated value
may carry that the shape never does, a symbol prefix to put back, a
``keep_pattern`` for values already in shape, and a guard for values that are
clearly too short. :func:`canonicalize` turns a dictated value into that shape::

    "x b seven t p two"                          -> "XB7TP2"
    "jordan underscore lee underscore eight two" -> "jordan_lee_82"
    "hash q four eight two one nine three seven" -> "#Q4821937"

A rule only ever comes from a schema example (see :func:`rule_from_example`);
nothing here knows a tool or a domain. The word tables are English and shared
with :mod:`.transcript`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

Case = Literal["lower", "upper", "keep"]

DIGIT_WORDS: Mapping[str, str] = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
#: Digit words that name a digit only next to another number ("four oh seven", never "oh, I see").
CONTEXT_DIGIT_WORDS = frozenset({"oh"})
TEEN_WORDS: Mapping[str, int] = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
TENS_WORDS: Mapping[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
REPEAT_WORDS: Mapping[str, int] = {"double": 2, "triple": 3}
SEPARATOR_WORDS: Mapping[str, str] = {"underscore": "_", "dash": "-", "hyphen": "-"}
#: Spoken symbol prefixes; only the first word of a value or a spelled run may be one.
PREFIX_WORDS: Mapping[str, str] = {"hash": "#", "pound": "#"}
FILLER_WORDS = frozenset({"uh", "um", "er", "ah", "hmm", "uhm", "erm"})
#: Punctuation removed from both ends of a spoken token before it is matched.
EDGE_PUNCTUATION = ".,?!;:\"'()[]{}“”‘’…"
#: Symbol prefixes an identifier example may open with.
PREFIX_SYMBOLS = "#"
#: Characters a dictated value may carry that a rule strips unless its example contains them.
STRIPPABLE = " .,-_"

_TOKEN_SPLIT = re.compile(r"[\s,]+")
_EXAMPLE_SHAPE = re.compile(rf"(?P<prefix>[{re.escape(PREFIX_SYMBOLS)}]?)(?P<body>[A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)*)")
_LETTERS_THEN_DIGITS = re.compile(r"([A-Za-z]+)([0-9]+)")
_MIN_IDENTIFIER_CHARS = 3
_MIN_LETTER_CODE_CHARS = 5


@dataclass(frozen=True, slots=True)
class IdentifierRule:
    """The written shape of one identifier argument, learned from its schema example.

    ``min_length`` and ``max_length`` count letters and digits (prefix and
    separators excluded). ``max_length`` is ``0`` when the example's length is
    not fixed, as with a name-like value (``jordan_lee_82``).
    """

    separator: str | None
    case: Case
    strip_chars: str
    prefix: str
    keep_pattern: re.Pattern[str]
    min_length: int
    max_length: int
    #: Number of separator-delimited segments in the example.
    segments: int = 1
    #: The example ends in a digit, so a value without any digit in its last segment is unfinished.
    digit_tail: bool = False


# -- spoken words -----------------------------------------------------------------------


def edge_stripped(token: str) -> str:
    """``token`` without surrounding punctuation."""
    return token.strip(EDGE_PUNCTUATION)


def single_char(word: str) -> str | None:
    """The one character a lower-case token spells: a digit word, a digit, or a letter."""
    if word in DIGIT_WORDS:
        return DIGIT_WORDS[word]
    if len(word) == 1 and word.isascii() and word.isalnum():
        return word
    return None


def is_number_word(word: str) -> bool:
    """A numeral, digit word, teen or tens word (hyphenated compounds such as ``eighty-two`` included)."""
    parts = [part for part in word.split("-") if part]
    return bool(parts) and all(
        (part.isascii() and part.isdigit()) or part in DIGIT_WORDS or part in TEEN_WORDS or part in TENS_WORDS
        for part in parts
    )


def number_words_to_digits(words: Sequence[str]) -> str | None:
    """``["eighty", "two"]`` -> ``"82"``, ``["four", "oh", "seven"]`` -> ``"407"``; ``None`` if any word is not one.

    Words are lower case. A lone context digit word (``oh``) is not a number.
    """
    flat = [part for word in words for part in word.split("-") if part]
    if not flat or (len(flat) == 1 and flat[0] in CONTEXT_DIGIT_WORDS):
        return None
    out: list[str] = []
    index = 0
    while index < len(flat):
        word = flat[index]
        following = flat[index + 1] if index + 1 < len(flat) else ""
        if word.isascii() and word.isdigit():
            out.append(word)
        elif word in DIGIT_WORDS:
            out.append(DIGIT_WORDS[word])
        elif word in TEEN_WORDS:
            out.append(str(TEEN_WORDS[word]))
        elif word in TENS_WORDS:
            unit = DIGIT_WORDS.get(following, "")
            if unit and unit != "0":
                out.append(str(TENS_WORDS[word] + int(unit)))
                index += 1
            else:
                out.append(str(TENS_WORDS[word]))
        else:
            return None
        index += 1
    return "".join(out)


def _spoken_to_written(value: str) -> str:
    """Join a dictated value's tokens, writing spoken digits, separators, repeats and prefixes as characters.

    A single token is only joined, never translated, so the result of one pass is
    stable under the next ("t w o" -> "two" stays "two").
    """
    raws = [token for token in _TOKEN_SPLIT.split(value) if token]
    if len(raws) <= 1:
        return "".join(raws)
    words = [edge_stripped(token).lower() for token in raws]
    out: list[str] = []
    index = 0
    while index < len(raws):
        word = words[index]
        following = words[index + 1] if index + 1 < len(words) else ""
        if word in FILLER_WORDS and len(raws) > 1:
            index += 1
            continue
        if word in SEPARATOR_WORDS:
            out.append(SEPARATOR_WORDS[word])
        elif word in PREFIX_WORDS and not out:
            out.append(PREFIX_WORDS[word])
        elif word in REPEAT_WORDS and (char := single_char(following)) is not None:
            out.append(char * REPEAT_WORDS[word])
            index += 2
            continue
        elif _is_letter(word) and words[index + 1 : index + 3] == ["as", "in"] and _is_word(words, index + 3):
            out.append(edge_stripped(raws[index]))  # "S as in Sam"
            index += 4
            continue
        elif is_number_word(word) and not word.isdigit():
            run = [word]
            while index + len(run) < len(words) and is_number_word(words[index + len(run)]):
                run.append(words[index + len(run)])
            digits = number_words_to_digits(run)
            if digits is None:
                out.append(raws[index])
            else:
                out.append(digits)
                index += len(run)
                continue
        else:
            out.append(raws[index])
        index += 1
    return "".join(out)


def _collapse_number_segments(text: str, separator: str) -> str:
    """``jordan_lee_eight_two`` -> ``jordan_lee_82``: a run of digit-word segments becomes one digit segment."""
    parts = text.split(separator)
    if len(parts) < 2:
        return text
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        lowered = [part.lower() for part in run]
        digits = number_words_to_digits(lowered) if any(not part.isdigit() for part in lowered) else None
        if digits is None:
            out.extend(run)
        else:
            out.append(digits)
        run.clear()

    for part in parts:
        lowered = part.lower()
        if lowered in DIGIT_WORDS or lowered in TEEN_WORDS or lowered in TENS_WORDS or (lowered.isdigit() and part):
            run.append(part)
            continue
        flush()
        out.append(part)
    flush()
    return separator.join(out)


def _is_letter(word: str) -> bool:
    return len(word) == 1 and word.isascii() and word.isalpha()


def _is_word(words: Sequence[str], index: int) -> bool:
    return index < len(words) and words[index].isalpha()


def apply_case(text: str, case: Case) -> str:
    """``lower`` | ``upper`` | ``keep``."""
    if case == "lower":
        return text.lower()
    if case == "upper":
        return text.upper()
    return text


# -- rules ------------------------------------------------------------------------------


def canonicalize(rule: IdentifierRule, value: str) -> str:
    """The value written in the rule's shape; a value already matching ``keep_pattern`` is returned unchanged.

    Deterministic and idempotent. The result may still not match ``keep_pattern``;
    callers use it only when it does.
    """
    if rule.keep_pattern.fullmatch(value):
        return value
    text = _spoken_to_written(value)
    for char in rule.strip_chars:
        text = text.replace(char, "")
    if separator := rule.separator:
        text = re.sub(f"{re.escape(separator)}{{2,}}", separator, text).strip(separator)
        text = _collapse_number_segments(text, separator)
    text = apply_case(text, rule.case)
    if rule.prefix:
        body = text.lstrip(rule.prefix)
        text = rule.prefix + body if body else ""
    return text


def is_incomplete(rule: IdentifierRule, value: str) -> bool:
    """Whether the value is clearly unfinished for the rule's shape (conservative: unsure means complete).

    Unfinished means empty, fewer letters and digits than the example's shape
    needs, an empty segment (``jordan_lee_``), or no digit where the example
    ends in digits (``jordan_lee``).
    """
    text = canonicalize(rule, value)
    body = text[len(rule.prefix) :] if rule.prefix and text.startswith(rule.prefix) else text
    alnum = sum(1 for char in body if char.isalnum())
    if alnum == 0 or alnum < rule.min_length:
        return True
    last = body
    if rule.separator:
        parts = body.split(rule.separator)
        if any(not part for part in parts):
            return True
        last = parts[-1]
    return rule.digit_tail and not any(char.isdigit() for char in last)


# -- examples ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Shape:
    """An identifier example split into a symbol prefix and segments joined by one separator."""

    prefix: str
    separator: str | None
    segments: tuple[str, ...]
    case: Case

    @property
    def name_like(self) -> bool:
        """A letters-only segment beside others: its length varies from value to value."""
        return len(self.segments) > 1 and any(segment.isalpha() for segment in self.segments)

    @property
    def alnum_length(self) -> int:
        return sum(len(segment) for segment in self.segments)


def _shape(example: str) -> _Shape | None:
    """The example's shape, or ``None`` when it is not identifier-shaped.

    Identifier-shaped means letters and digits in segments joined by one ``_`` or
    ``-``, an optional ``#`` prefix, at least three characters, and either a digit
    or an all-caps letter code of five or more. Names (``Quentin``), dates and
    phone numbers (digit-only segments with separators), amounts, emails and
    free text are not.
    """
    match = _EXAMPLE_SHAPE.fullmatch(example)
    if match is None:
        return None
    prefix, body = match.group("prefix"), match.group("body")
    separators = {char for char in body if not char.isalnum()}
    if len(separators) > 1:
        return None
    separator = separators.pop() if separators else None
    segments = tuple(body.split(separator)) if separator else (body,)
    if sum(len(segment) for segment in segments) < _MIN_IDENTIFIER_CHARS:
        return None
    has_digit = any(char.isdigit() for char in body)
    letter_code = separator is None and body.isalpha() and body.isupper() and len(body) >= _MIN_LETTER_CODE_CHARS
    if not (has_digit or letter_code):
        return None
    if separator and all(segment.isdigit() for segment in segments):
        return None
    letters = [char for char in body if char.isalpha()]
    if not letters:
        case: Case = "keep"
    elif all(char.islower() for char in letters):
        case = "lower"
    elif all(char.isupper() for char in letters):
        case = "upper"
    else:
        case = "keep"
    return _Shape(prefix=prefix, separator=separator, segments=segments, case=case)


def is_identifier_example(example: str) -> bool:
    """Whether a quoted schema example is identifier-shaped (see :func:`rule_from_example`)."""
    return _shape(example) is not None


def _letter_class(case: Case) -> str:
    return {"lower": "[a-z]", "upper": "[A-Z]"}.get(case, "[A-Za-z]")


def _alnum_class(case: Case) -> str:
    return {"lower": "[a-z0-9]", "upper": "[A-Z0-9]"}.get(case, "[A-Za-z0-9]")


def _count(n: int) -> str:
    return "" if n == 1 else f"{{{n}}}"


def _segment_pattern(segment: str, case: Case, *, exact: bool, split_letters: bool = False) -> str:
    if segment.isdigit():
        return "[0-9]" + (_count(len(segment)) if exact else "+")
    if segment.isalpha():
        return _letter_class(case) + (_count(len(segment)) if exact else "+")
    if split_letters and (match := _LETTERS_THEN_DIGITS.fullmatch(segment)):
        return _letter_class(case) + _count(len(match.group(1))) + "[0-9]" + _count(len(match.group(2)))
    return _alnum_class(case) + (_count(len(segment)) if exact else "+")


def rule_from_example(example: str) -> IdentifierRule | None:
    """The rule an identifier-shaped example implies, or ``None`` for any other example.

    A fixed shape (a code such as ``XB7TP2`` or ``#Q4821937``) keeps exactly its
    length and character classes. A name-like shape (``jordan_lee_82``) keeps its
    segment count and classes at any length.
    """
    shape = _shape(example)
    if shape is None:
        return None
    exact = not shape.name_like
    separator = re.escape(shape.separator) if shape.separator else ""
    body = separator.join(_segment_pattern(segment, shape.case, exact=exact) for segment in shape.segments)
    return IdentifierRule(
        separator=shape.separator,
        case=shape.case,
        strip_chars="".join(char for char in STRIPPABLE if char not in example),
        prefix=shape.prefix,
        keep_pattern=re.compile(re.escape(shape.prefix) + body),
        min_length=shape.alnum_length if exact else len(shape.segments),
        max_length=shape.alnum_length if exact else 0,
        segments=len(shape.segments),
        digit_tail=shape.segments[-1][-1].isdigit(),
    )


def complete_pattern_from_example(example: str) -> re.Pattern[str] | None:
    """The exact shape of a fixed identifier example, for deciding that a spelled value is finished.

    ``None`` for name-like or non-identifier examples: their length varies, so one
    example cannot say when a spelling is complete. The prefix is optional and
    case is ignored, because a caller rarely says the symbol and a transcript
    writes spelled letters in capitals.
    """
    shape = _shape(example)
    if shape is None or shape.name_like:
        return None
    separator = re.escape(shape.separator) if shape.separator else ""
    body = separator.join(
        _segment_pattern(segment, shape.case, exact=True, split_letters=True) for segment in shape.segments
    )
    prefix = "".join(f"{re.escape(char)}?" for char in shape.prefix)
    return re.compile(prefix + body, re.IGNORECASE)
