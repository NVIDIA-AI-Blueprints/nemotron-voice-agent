# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure spoken -> written token rules shared by the transcript and tool-argument hooks.

The word lists are language-specific, so they live in a named :class:`Ruleset`.
Only ``en`` ships; a new language is a new ruleset, not a change to the hooks.
Letter names (``em`` -> ``m``) are deliberately not mapped: the ASR writes the
syllables of a name as short tokens ("EM MA" for Emma), and mapping them would
corrupt the name.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

#: Punctuation stripped from both ends of a token before matching.
EDGE_PUNCTUATION = ".,?!;:\"'()[]“”‘’…"
#: A token of at most this many letters is "short": part of a spelled or split word.
SHORT_TOKEN_LETTERS = 2

_TOKEN = re.compile(r"\S+")
#: Letters the ASR wrote with dots ("S.A.N", "H.E.R"): joined without the dots.
_DOTTED_LETTERS = re.compile(r"[A-Za-z](?:\.[A-Za-z])+")


@dataclass(frozen=True, slots=True)
class Ruleset:
    """One language's word lists."""

    name: str
    language: str  # the ASR language_code prefix it matches, e.g. "en"
    digits: Mapping[str, str]
    teens: Mapping[str, int]
    tens: Mapping[str, int]
    repeats: Mapping[str, int]
    fillers: frozenset[str]
    stop_words: frozenset[str]
    #: Digit words valid only inside a run of other number words ("oh" in "four oh seven").
    run_only_digits: frozenset[str] = frozenset()

    def extended(self, *, fillers: Iterable[str] = (), stop_words: Iterable[str] = ()) -> Ruleset:
        """A copy with extra fillers and stop words (lower-cased)."""
        return replace(
            self,
            fillers=self.fillers | {word.lower() for word in fillers},
            stop_words=self.stop_words | {word.lower() for word in stop_words},
        )


EN = Ruleset(
    name="en",
    language="en",
    digits={
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
    },
    teens={
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
    },
    tens={"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90},
    repeats={"double": 2, "triple": 3},
    fillers=frozenset({"uh", "um", "er", "ah", "hmm", "uhm", "erm"}),
    stop_words=frozenset(
        {
            "is",
            "it",
            "its",
            "it's",
            "my",
            "id",
            "me",
            "the",
            "of",
            "so",
            "to",
            "and",
            "or",
            "as",
            "in",
            "on",
            "at",
            "by",
            "be",
            "we",
            "us",
            "an",
            "user",
            "name",
            "number",
            "code",
        }
    ),
    run_only_digits=frozenset({"oh"}),
)

RULESETS: Mapping[str, Ruleset] = {EN.name: EN}


def get_ruleset(name: str) -> Ruleset:
    """The ruleset called ``name`` (raises ``KeyError`` with the known names)."""
    try:
        return RULESETS[name]
    except KeyError:
        raise KeyError(f"unknown normalization ruleset {name!r}; known: {sorted(RULESETS)}") from None


@dataclass(frozen=True, slots=True)
class Token:
    """One whitespace token: its core (edge punctuation removed) and where the core sits in the text."""

    core: str
    start: int
    end: int
    spelled: bool = False  # written as dotted letters: never a filler

    @property
    def word(self) -> str:
        """The lower-cased core, for matching."""
        return self.core.lower()

    def is_filler(self, ruleset: Ruleset) -> bool:
        """A filler word of ``ruleset`` (dotted letters never are: "A.H." is spelled)."""
        return not self.spelled and self.word in ruleset.fillers


def tokenize(text: str) -> tuple[Token, ...]:
    """Whitespace tokens with edge punctuation outside ``start:end`` (a pure-punctuation token has ``core == ""``).

    Dotted letters ("S.A.N.") become one token of letters ("SAN"); ``start:end`` still covers the dots.
    """
    tokens: list[Token] = []
    for match in _TOKEN.finditer(text):
        raw = match.group()
        left = len(raw) - len(raw.lstrip(EDGE_PUNCTUATION))
        core = raw.strip(EDGE_PUNCTUATION)
        start = match.start() + left
        end = start + len(core)
        spelled = _DOTTED_LETTERS.fullmatch(core) is not None
        if spelled:
            core = core.replace(".", "")
        tokens.append(Token(core=core, start=start, end=end, spelled=spelled))
    return tuple(tokens)


def is_short(word: str) -> bool:
    """A 1-2 letter alphabetic token (a spelled letter or a split syllable)."""
    return word.isalpha() and len(word) <= SHORT_TOKEN_LETTERS


def is_number_word(word: str, ruleset: Ruleset, *, compound: bool) -> bool:
    """A numeral, digit word, repeat word, or (with ``compound``) a teen or tens word."""
    parts = word.split("-") if "-" in word else (word,)
    return all(
        part.isdigit()
        or part in ruleset.digits
        or part in ruleset.repeats
        or (compound and (part in ruleset.teens or part in ruleset.tens))
        for part in parts
        if part
    ) and any(parts)


def digits_from_words(words: Sequence[str], ruleset: Ruleset, *, compound: bool) -> str | None:
    """``["four", "three", "double", "nine"]`` -> ``"4399"``; ``None`` if any word is not a number word.

    Words are lower-case. ``forty three`` -> ``43`` and ``nineteen`` -> ``19`` with ``compound``.
    A run-only digit word ("oh") needs at least one other number word in the run.
    """
    flat = [part for word in words for part in word.split("-") if part]
    if not flat:
        return None
    if len(flat) == 1 and flat[0] in ruleset.run_only_digits:
        return None
    out: list[str] = []
    index = 0
    while index < len(flat):
        word = flat[index]
        following = flat[index + 1] if index + 1 < len(flat) else ""
        if word.isdigit():
            out.append(word)
        elif word in ruleset.repeats and (following in ruleset.digits or following.isdigit()):
            digit = ruleset.digits.get(following, following)
            out.append(digit * ruleset.repeats[word])
            index += 1
        elif word in ruleset.digits:
            out.append(ruleset.digits[word])
        elif compound and word in ruleset.teens:
            out.append(str(ruleset.teens[word]))
        elif compound and word in ruleset.tens:
            unit = ruleset.digits.get(following, "")
            if unit and following not in ruleset.run_only_digits and unit != "0":
                out.append(str(ruleset.tens[word] + int(unit)))
                index += 1
            else:
                out.append(str(ruleset.tens[word]))
        else:
            return None
        index += 1
    return "".join(out)


def apply_case(text: str, case: str) -> str:
    """``keep`` | ``lower`` | ``upper``."""
    if case == "lower":
        return text.lower()
    if case == "upper":
        return text.upper()
    return text


def join_part(tokens: Sequence[Token], ruleset: Ruleset, *, number_words: bool, compound: bool) -> str:
    """One ID part: all-number-word parts become digits, anything else is concatenated as written."""
    kept = [token for token in tokens if token.core and not token.is_filler(ruleset)]
    if number_words and kept:
        digits = digits_from_words([token.word for token in kept], ruleset, compound=compound)
        if digits is not None:
            return digits
    return "".join(token.core for token in kept)


def spelled_out(value: str, separator_names: Mapping[str, str]) -> str:
    """``jordan_lee_82`` -> ``j, o, r, d, a, n, underscore, l, e, e, underscore, 8, 2`` (for read-backs)."""
    names = {char: word for word, char in separator_names.items()}
    return ", ".join(names.get(char, char) for char in value if not char.isspace())
