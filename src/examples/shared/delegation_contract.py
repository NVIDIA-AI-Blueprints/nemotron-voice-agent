# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delegation contract constants for Responses requests and events, and a resolved model endpoint.

It also holds the example's one text rule for sentences and questions (:func:`sentences`, :func:`question_marks`,
:func:`ends_with_question`), so the frontend and the core read them the same way in any script that marks them. The rule
reads Unicode character names and properties only, never words.
"""

from __future__ import annotations

import functools
import inspect
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

TERMINALS: Final = frozenset({"response.completed", "response.failed", "response.incomplete"})
# The request options besides the model, the instructions, the tools, the tool choice and the output limit.
OPTIONS = ("parallel_tool_calls", "reasoning", "text", "service_tier")
# Metadata keys: the delegation a request starts or continues, the mark of a typed run, valued "true", and what of the
# backend's answers the caller had not heard when they spoke, which a frontend states on each delegation (ReplyUnheard).
DELEGATION_ID = "live_delegation_id"
TYPED_RUN = "live_typed_run"
REPLY_UNHEARD: Final = "live_reply_unheard"
# A function tool's MCP ToolAnnotations: the app's declaration of what the tool does. No model API takes it.
TOOL_ANNOTATIONS: Final = "annotations"
PROGRESS_PHASE = "commentary"
# The API a model endpoint speaks when its services.yaml entry names none: chat completions.
CHAT_COMPLETIONS: Final = "chat-completions"


def is_progress(item: dict) -> bool:
    """Return whether a message item is progress (phase ``commentary``) rather than a final answer."""
    return item.get("phase") == PROGRESS_PHASE


@functools.cache
def _is_question_mark(char: str) -> bool:
    """Whether ``char`` is a closing question mark, in any script.

    It is punctuation (``Po``) whose Unicode name ends with ``QUESTION MARK`` and holds neither ``INVERTED`` nor
    ``INITIAL``.
    """
    if unicodedata.category(char) != "Po":
        return False
    name = unicodedata.name(char, "")
    return name.endswith("QUESTION MARK") and "INVERTED" not in name and "INITIAL" not in name


@functools.cache
def _is_sentence_mark(char: str) -> bool:
    """Whether ``char`` ends a sentence by the default rule.

    It is punctuation (``Po``) whose Unicode name holds ``FULL STOP``, ``QUESTION MARK`` or ``EXCLAMATION MARK``, not an
    opening (inverted) one.
    """
    if unicodedata.category(char) != "Po":
        return False
    name = unicodedata.name(char, "")
    if "INVERTED" in name:
        return False
    return any(mark in name for mark in ("FULL STOP", "QUESTION MARK", "EXCLAMATION MARK"))


@functools.cache
def _is_closing(char: str) -> bool:
    """Closing punctuation (``Pe``, ``Pf``) or an ASCII quote: it may follow a sentence's end mark."""
    return unicodedata.category(char) in ("Pe", "Pf") or char in "\"'"


@functools.cache
def _is_trailing(char: str) -> bool:
    """Whether ``char`` may follow a question's mark at a sentence's end.

    That is a space, closing punctuation, an ASCII quote, Markdown emphasis (``*``, ``_``), ``!`` or a symbol (``So``,
    such as an emoji).
    """
    return char.isspace() or _is_closing(char) or char in "*_!" or unicodedata.category(char) == "So"


@functools.cache
def _is_wide(char: str) -> bool:
    """A full-width or wide character (East Asian Width ``F`` or ``W``), such as ``。`` or ``？``."""
    return unicodedata.east_asian_width(char) in ("F", "W")


def question_marks(text: str) -> int:
    """How many question marks ``text`` holds, in any script.

    A question mark is punctuation (``Po``) whose Unicode name ends with ``QUESTION MARK`` and holds neither
    ``INVERTED`` nor ``INITIAL``. ``?``, ``？``, ``؟`` and ``⁉`` count once each; an opening ``¿`` does not, so a
    question written with both marks counts once. The Greek question mark normalises to ``;`` and is not counted.
    """
    return sum(1 for char in text if _is_question_mark(char))


def ends_with_question(sentence: str) -> bool:
    """Whether ``sentence`` asks.

    Past trailing spaces, closing punctuation (``Pe``, ``Pf``), ASCII quotes, Markdown emphasis (``*``, ``_``), ``!``
    and symbols (``So``, such as an emoji), its last character is a question mark.
    """
    end = len(sentence)
    while end and _is_trailing(sentence[end - 1]) and not _is_question_mark(sentence[end - 1]):
        end -= 1
    return bool(end) and _is_question_mark(sentence[end - 1])


def _sentence_end(rest: str) -> int:
    """The index just past the first sentence end in ``rest``, or 0 when it has none.

    This is the default rule of :func:`sentence_spans`. Closing punctuation, Markdown emphasis (``*``, ``_``) and more
    end marks may follow the mark.
    """
    for index, char in enumerate(rest):
        if not _is_sentence_mark(char):
            continue
        end = index + 1
        while end < len(rest) and (_is_closing(rest[end]) or rest[end] in "*_" or _is_sentence_mark(rest[end])):
            end += 1
        if end == len(rest) or rest[end].isspace() or _is_wide(char):
            return end
    return 0


def sentence_spans(text: str, *, ends: Callable[[str], int] | None = None) -> list[tuple[int, int]]:
    """The ``[start, end)`` span of each sentence of ``text``, in order, together covering it.

    ``ends(rest)`` gives the index just past the first sentence end in ``rest``, or 0 when it has none. The default
    rule: a mark whose Unicode name holds ``FULL STOP``, ``QUESTION MARK`` or ``EXCLAMATION MARK`` (not an opening
    one), with any closing punctuation or Markdown emphasis after it, then a space or the end; a full-width or wide
    mark (East Asian Width F or W, such as ``。`` or ``？``) ends a sentence with no space after it.
    """
    ends = ends or _sentence_end
    spans, start = [], 0
    while start < len(text):
        end = ends(text[start:]) or len(text) - start
        spans.append((start, start + end))
        start += end
    return spans


def sentences(text: str, *, ends: Callable[[str], int] | None = None) -> list[str]:
    """The sentences of ``text`` by :func:`sentence_spans`, each stripped, the empty ones left out."""
    return [part for start, end in sentence_spans(text, ends=ends) if (part := text[start:end].strip())]


def question_key(text: str) -> str:
    """A question's letters and digits, case-folded: two asks are one question when their keys are equal."""
    return "".join(char for char in text.casefold() if char.isalnum())


class ReplyUnheard(StrEnum):
    """What of the backend's answers since the last delegation the caller had not heard when they spoke.

    Ordered ``NONE`` < ``ANSWER`` < ``QUESTION``.
    """

    NONE = "none"  # every answer was heard in full
    ANSWER = "answer"  # part of an answer was not heard, and that part holds no question
    QUESTION = "question"  # a sentence not heard ends with a question (ends_with_question)

    @classmethod
    def read(cls, metadata: Mapping[str, object]) -> ReplyUnheard | None:
        """The fact a request's metadata states; ``None`` when it states none, or a value the contract does not name."""
        value = metadata.get(REPLY_UNHEARD)
        return cls(value) if isinstance(value, str) and value in cls._value2member_map_ else None

    def at_least(self, other: ReplyUnheard) -> bool:
        """Whether this level is ``other`` or worse."""
        return _LEVELS.index(self) >= _LEVELS.index(other)

    @classmethod
    def of(cls, unheard: Sequence[str]) -> ReplyUnheard:
        """The level of the sentences not heard: NONE for none, QUESTION when one ends with a question, else ANSWER."""
        if not unheard:
            return cls.NONE
        return cls.QUESTION if any(ends_with_question(sentence) for sentence in unheard) else cls.ANSWER


_LEVELS: Final = (ReplyUnheard.NONE, ReplyUnheard.ANSWER, ReplyUnheard.QUESTION)


def takes_metadata(delegate: Callable[..., object]) -> bool:
    """Whether a delegation port's ``delegate`` takes the ``metadata`` keyword: a parameter of that name, or ``**``.

    A port written before the keyword is called with the input alone, so it keeps working.
    """
    try:
        parameters = inspect.signature(delegate).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "metadata" or p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)


def model_tool(tool: dict) -> dict:
    """``tool`` as a model API takes it: a copy without its annotations."""
    return {key: deepcopy(value) for key, value in tool.items() if key != TOOL_ANNOTATIONS}


@dataclass(frozen=True)
class ModelEndpoint:
    """A model endpoint from a ``services.yaml`` entry and the session body; ``extra_params`` reach the client as is."""

    model: str
    base_url: str
    api_key: str = "not-needed"
    extra_params: dict = field(default_factory=dict)
    # The API its server speaks: the entry's ``api``, a provider name of the delegation core's ``BACKENDS``.
    api: str = CHAT_COMPLETIONS
