# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A deterministic class for one user turn: liveness, backchannel, hold or content (F1/F2).

fix-plan-f1-f2-f4-f8.md section 2.2. The class depends only on the words of the turn and a
per-language :class:`Lexicon` of whole-token phrases; it never looks at a domain, a tool or the
conversation. A session whose language has no lexicon gets ``unclassified``, which every rule
ignores (today's behaviour).

Matching: the text is lower-cased, Unicode apostrophes and dashes become ASCII, and everything
other than letters, digits, ``'`` and ``-`` becomes a space. Phrases match whole-token runs,
scanning left to right and trying the longest phrase first. Unmatched tokens are content unless
they are softeners; negators, answer words and digit tokens are never softeners.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

LIVENESS = "liveness"
BACKCHANNEL = "backchannel"
HOLD = "hold"
CONTENT = "content"
UNCLASSIFIED = "unclassified"

_PING, _REPEAT, _HOLD, _BACKCHANNEL = "ping", "repeat", "hold", "backchannel"
_QUOTES = str.maketrans({"‘": "'", "’": "'", "ʼ": "'", "‐": "-", "‑": "-", "–": "-"})
_NON_WORD = re.compile(r"[^0-9a-z'\-]+")


@dataclass(frozen=True, slots=True)
class Lexicon:
    """One language's phrase lists (each phrase a whole-token sequence)."""

    ping: tuple[str, ...] = ()
    repeat: tuple[str, ...] = ()
    hold: tuple[str, ...] = ()
    backchannel: tuple[str, ...] = ()
    softeners: tuple[str, ...] = ()
    never_strip: tuple[str, ...] = ()
    question_words: tuple[str, ...] = ()
    auxiliaries: tuple[str, ...] = ()
    subjects: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TurnClass:
    """The class of one turn."""

    kind: str
    repeat: bool = False
    has_question: bool = False


def tokens(text: str) -> list[str]:
    """The normalised whole tokens of ``text``."""
    return [token for token in _NON_WORD.sub(" ", text.lower().translate(_QUOTES)).split() if token.strip("'-")]


def classify_turn(text: str, lexicon: Lexicon | None) -> TurnClass:
    """The class of one user turn (``unclassified`` without a lexicon or for an empty turn)."""
    if lexicon is None:
        return TurnClass(UNCLASSIFIED)
    words = tokens(text)
    if not words:
        return TurnClass(UNCLASSIFIED)
    phrases = _phrases(lexicon)
    found: set[str] = set()
    content: list[str] = []
    position = 0
    while position < len(words):
        for length, phrase, kind in phrases:
            if tuple(words[position : position + length]) == phrase:
                found.add(kind)
                position += length
                break
        else:
            word = words[position]
            if word in lexicon.never_strip or word.isdigit() or word not in lexicon.softeners:
                content.append(word)
            position += 1
    has_question = "?" in text or _question_opening(content, lexicon)
    if content:
        return TurnClass(CONTENT, has_question=has_question)
    if _PING in found or _REPEAT in found:
        return TurnClass(LIVENESS, repeat=_REPEAT in found, has_question=has_question)
    if _HOLD in found:
        return TurnClass(HOLD, has_question=has_question)
    if _BACKCHANNEL in found:
        return TurnClass(BACKCHANNEL, has_question=has_question)
    return TurnClass(UNCLASSIFIED, has_question=has_question)  # softeners only ("um")


def lexicon_for(language: str, lexicons: Mapping[str, Lexicon]) -> tuple[str, Lexicon | None]:
    """The lexicon for a BCP-47 tag: the exact tag, then its base language, then none."""
    tag = language.strip().lower().replace("_", "-")
    for key in (tag, tag.split("-")[0]):
        for name, lexicon in lexicons.items():
            if name.lower() == key:
                return name, lexicon
    return "", None


def lexicon_from_mapping(data: Mapping[str, Sequence[str]]) -> Lexicon:
    """A :class:`Lexicon` from a config mapping (unknown list names are an error)."""
    known = set(Lexicon.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown lexicon lists {sorted(unknown)}; known: {sorted(known)}")
    return Lexicon(**{name: tuple(str(item).lower() for item in data.get(name) or ()) for name in known})


def _phrases(lexicon: Lexicon) -> list[tuple[int, tuple[str, ...], str]]:
    out = []
    for kind, items in (
        (_PING, lexicon.ping),
        (_REPEAT, lexicon.repeat),
        (_HOLD, lexicon.hold),
        (_BACKCHANNEL, lexicon.backchannel),
    ):
        for item in items:
            phrase = tuple(tokens(item))
            if phrase:
                out.append((len(phrase), phrase, kind))
    out.sort(key=lambda entry: -entry[0])  # longest first; stable for equal lengths
    return out


def _question_opening(content: Sequence[str], lexicon: Lexicon) -> bool:
    if not content:
        return False
    if content[0] in lexicon.question_words:
        return True
    return len(content) > 1 and content[0] in lexicon.auxiliaries and content[1] in lexicon.subjects


#: The shipped English lexicon (fix-plan-f1-f2-f4-f8.md section 2.2); config can replace or extend it.
ENGLISH: dict[str, list[str]] = {
    "ping": [
        "hello", "are you there", "are you still there", "you there", "still there", "anyone there",
        "is anyone there", "can you hear me", "any update", "any updates", "anything yet", "did you find anything",
        "are you still working", "still working", "how long", "still waiting", "i'm still here", "i'm still waiting",
        "what's happening", "what's going on",
    ],
    "repeat": [
        "repeat", "repeat that", "can you repeat that", "could you repeat that", "can you say that again",
        "could you say that again", "say that again", "say it again", "come again", "pardon", "what was that",
        "i didn't catch that", "i didn't hear that", "i didn't hear you", "i can't hear you", "you cut out",
        "you're breaking up", "i missed that",
    ],
    "hold": [
        "hold on", "hang on", "wait", "one moment", "one second", "one sec", "a moment", "a second", "second", "sec",
        "moment", "give me a moment", "give me a second", "just a moment", "just a second", "let me check",
        "let me see",
    ],
    "backchannel": [
        "mm-hmm", "mhm", "uh-huh", "uh huh", "hmm", "mm", "okay", "ok", "okay sure", "sure", "alright", "all right",
        "right", "thanks", "thank you", "got it", "cool",
    ],
    "softeners": ["please", "sorry", "um", "uh", "er", "so", "just", "hey", "oh", "well"],
    "never_strip": ["no", "not", "never", "don't", "can't", "won't", "yes", "yeah", "yep"],
    "question_words": ["what", "when", "where", "why", "how", "who", "which"],
    "auxiliaries": ["can", "could", "would", "will", "do", "does", "did", "is", "are"],
    "subjects": ["you", "i"],
}  # fmt: skip


__all__ = [
    "BACKCHANNEL",
    "CONTENT",
    "ENGLISH",
    "HOLD",
    "LIVENESS",
    "UNCLASSIFIED",
    "Lexicon",
    "TurnClass",
    "classify_turn",
    "lexicon_for",
    "lexicon_from_mapping",
    "tokens",
]
