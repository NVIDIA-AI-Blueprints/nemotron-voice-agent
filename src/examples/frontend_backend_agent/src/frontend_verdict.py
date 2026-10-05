# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Decide whether new speech continues the delegated request already running.

The Talker marks a ``call_backend`` made while earlier work still runs with
``task: continue`` or ``task: new``. Code has the final word, and it errs
towards ``new``: a wrong ``new`` only restarts the work, which is what happens
without this feature, while a wrong ``continue`` would lose what the user
asked for. So ``continue`` requires that the user's latest words are a pure
acknowledgement or progress check and that the Talker kept the query.

One exception: a whole utterance that is a progress or presence phrase ("any
progress?", "one second") continues whatever query the Talker wrote. Such an
utterance cannot carry a new request, so a changed query is only a paraphrase.
Plain acknowledgements ("okay", "thanks") keep the identical-query rule because
they can also answer a question.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Literal

from pipecat.adapters.schemas.tools_schema import ToolsSchema

Decision = Literal["continue", "new"]

EXPLICIT_REPEAT_RE = re.compile(r"\b(?:repeat|refresh|recheck|again|one more time|check again)\b", re.IGNORECASE)

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_NORMALIZE_RE = re.compile(r"[^\w\s]")

#: Closed list of words that only acknowledge, confirm, ask about progress or
#: presence, or are said away from the microphone. Every word of an utterance
#: must be listed here for it to keep the running request.
ACK_PROGRESS_WORDS = frozenset(
    {
        # Acknowledgements and confirmations.
        "okay",
        "ok",
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "alright",
        "all",
        "right",
        "fine",
        "great",
        "good",
        "cool",
        "perfect",
        "sounds",
        "thanks",
        "thank",
        "you",
        "so",
        "much",
        "got",
        "it",
        "mm",
        "hmm",
        "mhm",
        "uh",
        "huh",
        "um",
        "hm",
        "go",
        "ahead",
        "please",
        "that's",
        "it's",
        # Progress and presence checks.
        "hello",
        "hi",
        "hey",
        "still",
        "there",
        "are",
        "is",
        "anyone",
        "any",
        "update",
        "updates",
        "did",
        "do",
        "find",
        "found",
        "anything",
        "how",
        "long",
        "will",
        "take",
        "taking",
        "what",
        "what's",
        "status",
        "yet",
        "done",
        "ready",
        "waiting",
        "i",
        "i'm",
        "am",
        # Words said away from the microphone.
        "hold",
        "on",
        "a",
        "second",
        "sec",
        "minute",
        "moment",
        "just",
        "wait",
        "shh",
        "quiet",
        "sorry",
    }
)

#: Whole-utterance progress and presence phrases, as token sequences after
#: :func:`_phrase_tokens`. None of them can carry a request on its own.
_PROGRESS_PHRASES: frozenset[tuple[str, ...]] = frozenset(
    tuple(phrase.split())
    for phrase in (
        "any progress",
        "any news",
        "any update",
        "any updates",
        "how much longer",
        "how long",
        "how long will it take",
        "how long is it going to take",
        "is it done",
        "are you done",
        "still working",
        "are you still working",
        "still working on it",
        "are you still working on it",
        "hold on",
        "i will be right with you",
        "let me know when you are done",
        *(
            f"{lead} {unit}"
            for lead in ("give me a", "one", "just a", "hold on a", "wait a", "hold on one")
            for unit in ("moment", "second", "sec", "minute")
        ),
    )
)
#: Fillers that may surround a progress phrase. Consent words ("yes") are
#: deliberately absent, so "yes, one second" is never read as a progress check.
_PHRASE_FILLERS = frozenset({"okay", "ok", "so", "um", "uh", "hmm", "mm", "well", "hello", "hi", "hey", "sorry"})
#: Sounds an ASR may transcribe as words; they never change the meaning.
_NOISE_WORDS = frozenset({"cough", "coughs", "coughing", "sneeze", "sneezes", "sniffle", "sniffles"})
_CONTRACTIONS = {
    "i'll": ("i", "will"),
    "you're": ("you", "are"),
    "it's": ("it", "is"),
    "that's": ("that", "is"),
    "what's": ("what", "is"),
    "i'm": ("i", "am"),
}

#: Words that signal a change. Never acknowledgement vocabulary.
CORRECTION_CUE_WORDS = frozenset(
    {"no", "not", "actually", "instead", "change", "wrong", "rather", "different", "other"}
)


@dataclass(frozen=True, slots=True)
class Verdict:
    """One decision about speech that arrived while a delegated run was working."""

    decision: Decision
    reason: str
    model_task: str


def same_request(first: str, second: str) -> bool:
    """Return whether two queries are equal after case-folding and removing punctuation."""
    return _normalize(first) == _normalize(second)


def pure_ack_or_progress(utterance: str) -> bool:
    """Return whether every word is acknowledgement, progress-check, or off-microphone vocabulary."""
    if EXPLICIT_REPEAT_RE.search(utterance):
        return False
    words = _WORD_RE.findall(utterance.casefold().replace("’", "'"))
    if not words:
        return False
    return all(word in ACK_PROGRESS_WORDS for word in words)


def is_progress_phrase(utterance: str) -> bool:
    """Return whether the whole utterance is one progress or presence phrase, optionally with fillers."""
    if EXPLICIT_REPEAT_RE.search(utterance):
        return False
    tokens = _phrase_tokens(utterance)
    start, end = 0, len(tokens)
    while start < end and tokens[start] in _PHRASE_FILLERS:
        start += 1
    while end > start and tokens[end - 1] in _PHRASE_FILLERS:
        end -= 1
    return tuple(tokens[start:end]) in _PROGRESS_PHRASES


def decide(task: object, query: str, running_query: str, utterance: str) -> Verdict:
    """Return ``continue`` for a pure acknowledgement with the query kept, or for a progress phrase."""
    model_task = task if task in {"continue", "new"} else ""
    progress_phrase = is_progress_phrase(utterance)
    if not progress_phrase and not pure_ack_or_progress(utterance):
        return Verdict("new", "substantive", model_task)
    if same_request(query, running_query):
        return Verdict("continue", "model" if model_task == "continue" else "acknowledgement", model_task)
    if progress_phrase:
        return Verdict("continue", "progress_phrase", model_task)
    return Verdict("new", "query_changed", model_task)


TASK_PROPERTY: dict = {
    "type": "string",
    "enum": ["continue", "new"],
    "description": (
        "Only while an earlier call_backend request is still running: continue when the latest words leave that "
        "request unchanged, otherwise new."
    ),
}


def with_task_field(schema: ToolsSchema) -> ToolsSchema:
    """Return a copy of a Talker schema whose call_backend accepts the optional task field."""
    custom_tools: dict = {}
    for adapter, tools in (schema.custom_tools or {}).items():
        copied = copy.deepcopy(tools)
        for tool in copied:
            function = tool.get("function") if isinstance(tool, dict) else None
            definition = function if isinstance(function, dict) else tool
            if isinstance(definition, dict) and definition.get("name") == "call_backend":
                definition["parameters"]["properties"]["task"] = copy.deepcopy(TASK_PROPERTY)
        custom_tools[adapter] = copied
    return ToolsSchema(standard_tools=list(schema.standard_tools), custom_tools=custom_tools)


def _phrase_tokens(text: str) -> list[str]:
    """Tokenize for phrase matching: case-folded, contractions expanded, noise words removed."""
    tokens: list[str] = []
    for word in _WORD_RE.findall(text.casefold().replace("’", "'")):
        if word in _NOISE_WORDS:
            continue
        tokens.extend(_CONTRACTIONS.get(word, (word,)))
    return tokens


def _normalize(text: str) -> str:
    return " ".join(_NORMALIZE_RE.sub(" ", text.casefold()).split())
