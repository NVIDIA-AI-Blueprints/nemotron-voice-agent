# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""F8: the facts of an identification so far, appended to a failed identification lookup.

fix-plan-f1-f2-f4-f8.md sections 4.2-4.3. One instance per session. It never decides which
field is wrong and never builds a value: it lists what was tried, what the user spelled (and
where each spelling was sent), the user's own corrections, and the other identification
lookups the session's tools offer.

* **Identification tools** are derived from the session's tool schemas
  (:func:`identification_tools`): a read tool whose name or description says it returns a
  person, that takes no other entity's ID, and that has a required personal-detail or
  person-ID argument. A config list may replace the derived set.
* **Output classes** (:func:`classify_lookup_output`), first match wins: ``not_found``,
  ``error``, ``ambiguous``, ``found``, ``unknown`` (``invalid`` and ``cancelled`` come from the
  relay). Only ``found`` closes an episode.
* **Episodes**: spellings and corrections wait in a pending pool until an identification
  attempt opens an episode; a ``found`` closes it, and a closed episode is never shown again.
  A result arriving after its episode closed gets no ledger note.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from examples.frontend_delegation_hermes.core.normalization.arguments import render_message
from examples.frontend_delegation_hermes.core.normalization.schema_rules import (
    PERSON_WORDS,
    is_identifier_argument,
    is_person_argument,
    is_read_tool,
    tool_function,
)

NOT_FOUND = "not_found"
INVALID = "invalid"
ERROR = "error"
AMBIGUOUS = "ambiguous"
FOUND = "found"
UNKNOWN = "unknown"
CANCELLED = "cancelled"
MISSES = (NOT_FOUND, INVALID)

#: Objects that make a lookup return a person (the existing person words, plus "person").
PERSON_OBJECTS = frozenset((*PERSON_WORDS, *(f"{word}s" for word in PERSON_WORDS), "person", "persons", "people"))
_VERBS = frozenset(
    [
        "get",
        "gets",
        "find",
        "finds",
        "look",
        "looks",
        "lookup",
        "lookups",
        "search",
        "searches",
        "retrieve",
        "retrieves",
        "fetch",
        "fetches",
        "return",
        "returns",
    ]
)
_ARTICLES = frozenset(("a", "an", "the", "up", "for", "of"))
#: Argument-name tokens that make an argument a personal detail.
PERSONAL_DETAIL_TOKENS = frozenset(("name", "email", "phone", "zip", "postal", "birth", "dob", "mobile"))
_SEPARATORS = re.compile(r"[\s._\-]+")
_WORD = re.compile(r"[a-z0-9]+")
_MAX_ID_TOKEN = 64


@dataclass(frozen=True, slots=True)
class LedgerSettings:
    """The ledger's knobs (from ``result_hints``)."""

    max_age_turns: int = 8
    correction_cues: tuple[str, ...] = ("not", "no,", "it's", "with two", "double", "instead", "wrong")
    max_attempts_shown: int = 4
    max_spellings_shown: int = 6
    max_corrections_shown: int = 2
    #: At most this many ledger notes per episode (``result_hints.max_hints``).
    max_notes: int = 6


@dataclass(slots=True)
class Attempt:
    """One identification lookup."""

    episode: int
    call_id: str
    tool: str
    fields: dict[str, str]
    origin: str = "model"
    outcome: str = ""


@dataclass(frozen=True, slots=True)
class Spelling:
    """One spelled value of a user turn."""

    turn_id: int
    written: str
    question: str


@dataclass(frozen=True, slots=True)
class Correction:
    """A user turn with a correction cue, verbatim."""

    turn_id: int
    text: str


@dataclass(slots=True)
class _Episode:
    number: int
    attempts: list[Attempt] = field(default_factory=list)
    spellings: list[Spelling] = field(default_factory=list)
    corrections: list[Correction] = field(default_factory=list)
    closed: bool = False
    notes: int = 0


# -- schema-derived identification tools -------------------------------------------------


def identification_tools(tools: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """The session's identification lookups (fix-plan section 4.3, rules 1-4)."""
    out: list[str] = []
    for tool in tools:
        function = tool_function(tool)
        name = str(function.get("name") or "")
        description = str(function.get("description") or "")
        if not name or not is_read_tool(name, description):
            continue
        if not _returns_person(name, description):
            continue
        required = _required_arguments(function)
        if any(is_identifier_argument(arg) and not is_person_argument(arg) for arg in required):
            continue  # takes another entity's ID (an order, a line, a bill)
        if any(_is_personal(arg) for arg in required):
            out.append(name)
    return tuple(out)


def returns_id(name: str, description: str) -> bool:
    """Whether the tool's name or description says it returns an ID (the token ``id``)."""
    return "id" in _WORD.findall(name.lower().replace("_", " ")) or "id" in _WORD.findall(description.lower())


def _returns_person(name: str, description: str) -> bool:
    by_name = _object_after_verb(name.lower().split("_"))
    if by_name is not None:
        return by_name in PERSON_OBJECTS
    sentence = description.strip().lower().split(".")[0]
    by_description = _object_after_verb(_WORD.findall(sentence))
    return by_description in PERSON_OBJECTS if by_description is not None else False


def _object_after_verb(words: Sequence[str]) -> str | None:
    for index, word in enumerate(words):
        if word in _VERBS:
            for following in words[index + 1 :]:
                if following not in _ARTICLES:
                    return following
            return ""
    return None


def _required_arguments(function: Mapping[str, Any]) -> list[str]:
    parameters = function.get("parameters")
    if not isinstance(parameters, Mapping):
        return []
    return [str(arg) for arg in parameters.get("required") or ()]


def _is_personal(argument: str) -> bool:
    tokens = set(_WORD.findall(argument.lower().replace("_", " ")))
    return bool(tokens & PERSONAL_DETAIL_TOKENS) or is_person_argument(argument)


# -- output classes ----------------------------------------------------------------------------


def classify_lookup_output(output: str, *, failure: re.Pattern[str], returns_an_id: bool) -> str:
    """The class of one identification lookup's output (first match wins; section 4.3)."""
    text = output.strip()
    if failure.search(output) or not text:
        return NOT_FOUND
    parsed, is_json = _parse(text)
    if is_json and parsed in ([], {}, None, ""):
        return NOT_FOUND
    if text.startswith("Error") or (isinstance(parsed, dict) and ("error" in parsed or "errors" in parsed)):
        return ERROR
    if isinstance(parsed, list) and len(parsed) >= 2:
        return AMBIGUOUS
    if isinstance(parsed, dict) and _has_person_id(parsed):
        return FOUND
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict) and _has_person_id(parsed[0]):
        return FOUND
    if not is_json and returns_an_id and len(text) <= _MAX_ID_TOKEN and not any(c.isspace() for c in text):
        return FOUND
    return UNKNOWN


def _parse(text: str) -> tuple[Any, bool]:
    try:
        return json.loads(text), True
    except (json.JSONDecodeError, ValueError):
        return None, False


def _has_person_id(record: Mapping[str, Any]) -> bool:
    for key, value in record.items():
        person_id = is_identifier_argument(str(key)) and is_person_argument(str(key))
        if person_id and isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
            return True
    return False


# -- the ledger ------------------------------------------------------------------------------------


class IdentityLedger:
    """One session's identification evidence and its note."""

    def __init__(
        self,
        identification: Iterable[str],
        *,
        tools: Sequence[Mapping[str, Any]] = (),
        failure_pattern: str,
        template: str,
        settings: LedgerSettings | None = None,
    ) -> None:
        """``identification``: the identification tools; ``tools``: the session's schemas (for argument lists)."""
        self.identification = tuple(identification)
        self._settings = settings or LedgerSettings()
        self._failure = re.compile(failure_pattern) if failure_pattern else re.compile(r"(?!x)x")
        self._template = template
        self._schemas = {str(tool_function(t).get("name") or ""): tool_function(t) for t in tools}
        self._episodes: list[_Episode] = []
        self._pending_spellings: list[Spelling] = []
        self._pending_corrections: list[Correction] = []
        self._attempts: dict[str, Attempt] = {}
        self._last_turn = 0

    # -- inputs ----------------------------------------------------------------------------

    def watches(self, tool: str) -> bool:
        """Whether ``tool`` is an identification lookup."""
        return tool in self.identification

    def add_turn(self, turn_id: int, text: str, spelled: Iterable[str], question: str) -> None:
        """One decided user turn: its spelled values and, with a cue, the turn itself as a correction."""
        self._last_turn = max(self._last_turn, turn_id)
        spellings = [
            Spelling(turn_id, value, " ".join(question.split())[:80])
            for value in spelled
            if len(value) >= 2 and any(c.isalpha() for c in value)
        ]
        corrections = [Correction(turn_id, " ".join(text.split())[:160])] if self._has_cue(text) else []
        episode = self._open_episode()
        if episode is None:
            self._pending_spellings.extend(spellings)
            self._pending_corrections.extend(corrections)
        else:
            episode.spellings.extend(spellings)
            episode.corrections.extend(corrections)

    def on_issue(self, call_id: str, tool: str, arguments: str, *, origin: str = "model") -> None:
        """An identification lookup is sent (or answered locally): it joins, or opens, the episode."""
        if not self.watches(tool):
            return
        episode = self._open_episode()
        if episode is None:
            episode = _Episode(number=len(self._episodes) + 1)
            episode.spellings, self._pending_spellings = self._pending_spellings, []
            episode.corrections, self._pending_corrections = self._pending_corrections, []
            self._episodes.append(episode)
        attempt = Attempt(episode.number, call_id, tool, _scalar_fields(arguments), origin)
        episode.attempts.append(attempt)
        self._attempts[call_id] = attempt

    def on_cancel(self, call_id: str) -> None:
        """The call was cancelled before its output (recorded, never counted or shown)."""
        attempt = self._attempts.get(call_id)
        if attempt is not None and not attempt.outcome:
            attempt.outcome = CANCELLED

    def on_local_invalid(self, call_id: str) -> None:
        """The relay answered the call locally as an incomplete value (a miss without a client output)."""
        attempt = self._attempts.get(call_id)
        if attempt is not None:
            attempt.outcome = INVALID

    def on_output(self, call_id: str, output: str) -> tuple[str, str]:
        """``(class, note)`` for one output of an identification lookup ("" note when none is due)."""
        attempt = self._attempts.get(call_id)
        if attempt is None:
            return "", ""
        schema = self._schemas.get(attempt.tool, {})
        attempt.outcome = classify_lookup_output(
            output,
            failure=self._failure,
            returns_an_id=returns_id(attempt.tool, str(schema.get("description") or "")),
        )
        episode = self._episodes[attempt.episode - 1]
        if episode.closed:
            return attempt.outcome, ""  # a late result: logged by the caller, no ledger note
        if attempt.outcome == FOUND:
            episode.closed = True
            return attempt.outcome, ""
        if attempt.outcome != NOT_FOUND or episode.notes >= self._settings.max_notes:
            return attempt.outcome, ""
        note = self._note(episode)
        if note:
            episode.notes += 1
        return attempt.outcome, note

    @property
    def episode(self) -> int:
        """The number of the open episode (0 when none is open)."""
        episode = self._open_episode()
        return episode.number if episode is not None else 0

    # -- the note ----------------------------------------------------------------------------

    def _note(self, episode: _Episode) -> str:
        settings = self._settings
        misses = sum(1 for a in episode.attempts if a.outcome in MISSES)
        shown = [a for a in episode.attempts if a.outcome and a.outcome != CANCELLED]
        lines: list[str] = []
        if misses >= 2 and shown:
            numbered = list(enumerate(shown, 1))[-settings.max_attempts_shown :]
            tried = "; ".join(
                f"{n}) {a.tool} " + " ".join(f'{k}="{v}"' for k, v in a.fields.items()) for n, a in numbered
            )
            lines.append(f"Lookups tried: {tried}.")
        min_turn = self._last_turn - settings.max_age_turns
        spellings = [s for s in episode.spellings if s.turn_id > min_turn][-settings.max_spellings_shown :]
        if spellings:
            parts = []
            for spelling in spellings:
                asked = f' (asked: "{spelling.question}")' if spelling.question else ""
                parts.append(f'"{spelling.written}"{asked} - {self._status(spelling.written, shown)}')
            lines.append("What the user spelled: " + "; ".join(parts) + ".")
        corrections = [c for c in episode.corrections if c.turn_id > min_turn][-settings.max_corrections_shown :]
        if corrections:
            lines.append("User corrections: " + "; ".join(f'"{c.text}"' for c in corrections) + ".")
        if misses >= 2:
            used = {a.tool for a in episode.attempts}
            others = [self._signature(tool) for tool in self.identification if tool not in used]
            if others:
                lines.append("Other identification lookups your tools offer: " + ", ".join(others) + ".")
        if not lines or not self._template:
            return ""
        return render_message(self._template, lines="\n".join(lines))

    @staticmethod
    def _status(written: str, attempts: Sequence[Attempt]) -> str:
        exact = written.casefold()
        loose = _SEPARATORS.sub("", exact)
        separators = ""
        for number, attempt in enumerate(attempts, 1):
            for name, value in attempt.fields.items():
                if value.casefold() == exact:
                    return f"sent as {name} in lookup {number}"
                if not separators and _SEPARATORS.sub("", value.casefold()) == loose:
                    separators = f"sent as {name} in lookup {number}, with different separators"
        return separators or "not sent"

    def _signature(self, tool: str) -> str:
        return f"{tool}({', '.join(_required_arguments(self._schemas.get(tool, {})))})"

    def _open_episode(self) -> _Episode | None:
        return self._episodes[-1] if self._episodes and not self._episodes[-1].closed else None

    def _has_cue(self, text: str) -> bool:
        lowered = " ".join(text.lower().split())
        for cue in self._settings.correction_cues:
            cue = cue.lower()
            if cue.endswith(","):
                if re.search(rf"(?<![a-z']){re.escape(cue[:-1])}\s*,", lowered):
                    return True
            elif re.search(rf"(?<![a-z']){re.escape(cue)}(?![a-z'])", lowered):
                return True
        return False


def _scalar_fields(arguments: str) -> dict[str, str]:
    try:
        data = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}


__all__ = [
    "AMBIGUOUS",
    "CANCELLED",
    "ERROR",
    "FOUND",
    "INVALID",
    "NOT_FOUND",
    "UNKNOWN",
    "IdentityLedger",
    "LedgerSettings",
    "classify_lookup_output",
    "identification_tools",
    "returns_id",
]
