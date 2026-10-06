# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""One-time checks of a plan that refuses, hands off, or labels an answer as a refusal.

Everything here reads only the session's own tool schemas, the delegated
request and the plan's form; no tool name, identifier or domain word is built
in. A check never vetoes a plan. It returns a code-written reason, the backend
re-plans once with it, and whatever the Thinker returns then is accepted:

* **answer check**: an ``unsupported_request`` plan after tool results may be an
  answer written in the only form that carries text. The Thinker chooses
  between the final-answer form and a refusal.
* **refusal check**: a refusal of a change request, when an enabled write tool
  plausibly serves it and no policy quote backs the refusal.
* **handoff check**: a handoff the caller did not ask for, when another enabled
  tool plausibly serves the request.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from examples.frontend_backend_agent.src.normalization import MUTATING_NAME_TOKENS

#: The placeholder text of the prompt's ``unsupported_request`` example.
PLACEHOLDER_REFUSAL_TEXT = "Please answer that stable question directly."
#: Change verbs for free text: the write-name tokens without the two too generic for prose.
CHANGE_VERBS = MUTATING_NAME_TOKENS - {"in", "set"}
_HANDOFF_NAME_TOKENS = ("transfer", "escalate", "handoff", "hand_off")
_PERSON_WORDS = frozenset({"human", "humans", "person", "representative", "staff", "operator", "supervisor"})
#: Explicit requests for a person, as token sequences.
_PERSON_REQUESTS: tuple[tuple[str, ...], ...] = tuple(
    tuple(phrase.split())
    for phrase in (
        "human",
        "real person",
        "representative",
        "transfer me",
        "speak to someone",
        "supervisor",
        "manager",
        "operator",
    )
)
_NEGATIONS = frozenset({"don't", "dont", "not", "no", "never", "without"})
#: Words too common to tie a request to a tool.
_STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "her",
        "his",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "one",
        "or",
        "our",
        "please",
        "she",
        "so",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "user",
        "user's",
        "wants",
        "want",
        "was",
        "we",
        "what",
        "whether",
        "which",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SUFFIXES = (
    "ications",
    "ication",
    "ations",
    "ation",
    "ments",
    "ment",
    "ings",
    "ing",
    "ied",
    "ies",
    "ed",
    "es",
    "s",
    "y",
    "e",
)
_FIRST_SENTENCE_RE = re.compile(r"^(.*?)(?:[.!?](?:\s|$)|$)", re.DOTALL)
_COMPARABLE_DROP_RE = re.compile(r"[^\w\s']")
#: A policy quote must be at least this many words, so a fragment cannot match by chance.
_MIN_POLICY_QUOTE_WORDS = 4
_MAX_POLICY_QUOTE_CHARS = 400


@dataclass(frozen=True, slots=True)
class ToolInfo:
    """What a check needs to know about one enabled client tool."""

    name: str
    description: str
    kind: str


def stem(word: str) -> str:
    """Return a crude stem: case-folded, one common suffix removed, a doubled final consonant collapsed."""
    lowered = word.casefold().strip("'")
    for suffix in _SUFFIXES:
        if lowered.endswith(suffix) and len(lowered) - len(suffix) >= 3:
            lowered = lowered[: -len(suffix)]
            break
    if len(lowered) >= 3 and lowered[-1] == lowered[-2] and lowered[-1] not in "aeiou":
        lowered = lowered[:-1]
    return lowered


def content_stems(text: str) -> set[str]:
    """Return the stems of the words in ``text`` that are not stop words."""
    return {stem(word) for word in _WORD_RE.findall(str(text or "").casefold()) if word not in _STOP_WORDS}


_CHANGE_STEMS = frozenset(stem(verb) for verb in CHANGE_VERBS)


def is_change_request(query: str) -> bool:
    """Return whether the request contains a change verb (a stemmed write-name token)."""
    return not _CHANGE_STEMS.isdisjoint(stem(word) for word in _WORD_RE.findall(str(query or "").casefold()))


def _tool_stems(tool: ToolInfo) -> set[str]:
    first = _FIRST_SENTENCE_RE.match(tool.description or "")
    words = {stem(token) for token in tool.name.casefold().split("_") if token and token not in _STOP_WORDS}
    return words | content_stems(first.group(1) if first else "")


def candidate_tools(query: str, tools: Iterable[ToolInfo], *, writes_only: bool) -> list[str]:
    """Return the tools that share a content word with the request, by name token or description opening.

    Lexical matching can miss a tool described in other words, so an empty
    result means "accept the plan", never "block it".
    """
    wanted = content_stems(query)
    if not wanted:
        return []
    return [
        tool.name
        for tool in tools
        if (not writes_only or tool.kind == "write") and not wanted.isdisjoint(_tool_stems(tool))
    ]


def is_handoff_tool(tool: ToolInfo) -> bool:
    """Return whether a tool hands the caller to a person: by its name *and* its description, never a read."""
    if tool.kind == "read":
        return False
    name = tool.name.casefold()
    if not any(token in name for token in _HANDOFF_NAME_TOKENS):
        return False
    return not _PERSON_WORDS.isdisjoint(_WORD_RE.findall(tool.description.casefold()))


def asks_for_person(utterance: str) -> bool:
    """Return whether the caller explicitly asked for a person, ignoring negated forms ("don't transfer me")."""
    tokens = _WORD_RE.findall(str(utterance or "").casefold().replace("’", "'"))
    for phrase in _PERSON_REQUESTS:
        width = len(phrase)
        for start in range(len(tokens) - width + 1):
            if tuple(tokens[start : start + width]) != phrase:
                continue
            before = tokens[max(0, start - 3) : start]
            if _NEGATIONS.isdisjoint(before) and not any(word.endswith("n't") for word in before):
                return True
    return False


def comparable(text: object) -> str:
    """Case-fold, drop punctuation and collapse spaces, for verbatim comparison."""
    return " ".join(_COMPARABLE_DROP_RE.sub(" ", str(text or "").casefold().replace("’", "'")).split())


class MalformedBasisError(ValueError):
    """A plan's ``basis`` field is present but not of the documented shape."""


def policy_quote(plan: Mapping[str, Any]) -> str | None:
    """Return the plan's ``basis.policy_quote``, None without a basis; raise on a malformed basis."""
    basis = plan.get("basis")
    if basis is None:
        return None
    if not isinstance(basis, Mapping) or set(basis) - {"policy_quote"}:
        raise MalformedBasisError("basis must be an object with only policy_quote")
    quote = basis.get("policy_quote")
    if quote is None:
        return None
    if not isinstance(quote, str) or len(quote) > _MAX_POLICY_QUOTE_CHARS:
        raise MalformedBasisError("basis.policy_quote must be a short string")
    return quote


def quote_in_instructions(quote: str | None, instructions: str) -> bool:
    """Return whether a policy quote appears verbatim (after normalization) in the session's instructions."""
    wanted = comparable(quote)
    return len(wanted.split()) >= _MIN_POLICY_QUOTE_WORDS and wanted in comparable(instructions)


def is_refusal(plan: Mapping[str, Any]) -> bool:
    """Return whether a plan is the ``unsupported_request`` response hint."""
    return (
        plan.get("tool") == "response_hint"
        and not plan.get("tool_calls")
        and plan.get("reason") == "unsupported_request"
        and str(plan.get("context") or "general") in {"", "general"}
    )


def completion_text(plan: Mapping[str, Any]) -> str:
    """Return the final-answer text of a ``{"complete":true,"response_text":...}`` plan, or ""."""
    if plan.get("complete") is not True:
        return ""
    text = plan.get("response_text")
    return " ".join(text.split()) if isinstance(text, str) else ""


def planned_calls(plan: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Return ``(tool, params)`` for each executable call of a plan; response hints have none."""
    raw = plan.get("tool_calls")
    if raw is None and plan.get("tool") and plan.get("tool") != "response_hint":
        raw = [plan]
    if not isinstance(raw, list):
        return []
    return [(str(item.get("tool") or ""), item.get("params")) for item in raw if isinstance(item, Mapping)]


ANSWER_CHECK_REASON = (
    "unsupported_request means you are declining the request. If your text answers the request from the "
    'results, return {"complete":true,"response_text":"<the answer>"}. If you are declining, return '
    "unsupported_request again with the reason in response_text."
)
PLACEHOLDER_REASON = (
    "Your response_text copied the example's placeholder. Write your own text, or plan the call the request needs."
)


def refusal_check_reason(candidates: list[str]) -> str:
    """Return the code-written correction for a refusal of a change that a write tool may serve."""
    return (
        f"Before refusing, consider: {', '.join(candidates)}. If one of them can serve the request, plan it, "
        "ask for missing details (params_missing), or ask for consent (confirmation_needed). Otherwise refuse, "
        'citing the caller\'s policy word for word in "basis":{"policy_quote":"..."} if it forbids the request.'
    )


def handoff_check_reason(candidates: list[str]) -> str:
    """Return the code-written correction for a handoff the caller did not ask for."""
    return (
        "Hand off only if the caller asks for a person, the caller's policy requires it, or none of "
        f"{', '.join(candidates)} can serve the request. If the policy requires it, quote it word for word in "
        '"basis":{"policy_quote":"..."}.'
    )
