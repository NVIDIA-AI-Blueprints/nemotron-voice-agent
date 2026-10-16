# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A3: which values of a failed lookup the user spelled, and which were only heard as a word.

``tau3-geval-failure-fixes-plan.md`` section 4 (A3). Information only: the line states facts
and never says which field is wrong. One instance per session; the turn manager records each
decided user turn (:meth:`FieldProvenance.add_turn`) and the relay asks for the line on the
second and later miss of a person lookup (:meth:`FieldProvenance.describe`).

A value is attributed only with a **field association** and a **turn association**; anything
else is ``unknown``:

* the evidence comes from the newest user turn associated with that field: the user named the
  field in the turn (and no other field of the lookup), or named no field of the lookup and the
  assistant's question just before it named exactly that field. An earlier turn never counts,
  even when the newest one does not decide the value (a later correction wins);
* a turn with two or more spelled spans answers several values, so every field is ``unknown``
  for it (a two-person or multi-field turn cannot be split without guessing);
* ``spelled``: one ``spelled_runs`` span of that turn equals the sent value character for
  character, ignoring only letter case;
* ``heard as a word``: the value appears as a whole word in that turn, with no spelled span over it;
* a value of digits only is never attributed (``spelled_runs`` needs a letter to form a run);
* when the turn's one spelled span differs from the value sent, the line also says what the user
  spelled for that field.
"""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

SPELLED = "spelled"
HEARD = "heard as a word"
UNKNOWN = "unknown"
#: Name suffixes a speaker often drops ("zip" for zip_code, "phone" for phone_number).
_DROPPABLE_SUFFIXES = ("number", "code", "address", "id")


@dataclass(frozen=True, slots=True)
class SpelledSpan:
    """A ``spelled_runs`` span of a user turn: its written form and position in the turn text."""

    written: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class TurnEvidence:
    """One decided user turn: its text, its spelled spans, and the assistant question heard before it."""

    text: str
    spelled: tuple[SpelledSpan, ...]
    question: str


def field_labels(argument: str) -> tuple[str, ...]:
    """The spoken names of an argument: ``first_name`` -> ("first name",), ``zip_code`` -> ("zip code", "zip")."""
    words = [word for word in re.split(r"[^a-z0-9]+", argument.lower()) if word]
    if not words:
        return ()
    labels = [" ".join(words)]
    if len(words) > 1 and words[-1] in _DROPPABLE_SUFFIXES:
        labels.append(" ".join(words[:-1]))
    return tuple(labels)


def _mentions(text: str, labels: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])", lowered) for label in labels)


class FieldProvenance:
    """Recent user turns and the attribution of lookup values to them."""

    def __init__(self, max_turns: int = 30) -> None:
        """Keep the newest ``max_turns`` turns."""
        self._turns: deque[TurnEvidence] = deque(maxlen=max_turns)

    def add_turn(self, text: str, spelled: Iterable[SpelledSpan], question: str) -> None:
        """One decided user turn (raw text, its spelled spans, the assistant's preceding question)."""
        self._turns.append(TurnEvidence(text=text, spelled=tuple(spelled), question=question))

    def attribute(self, arguments: Mapping[str, Any]) -> list[tuple[str, str, str, str]]:
        """``(argument, value, class, spelled_other)`` for every scalar argument of a lookup."""
        fields = {
            name: str(value)
            for name, value in arguments.items()
            if isinstance(value, str | int | float) and not isinstance(value, bool)
        }
        labels = {name: field_labels(name) for name in fields}
        out: list[tuple[str, str, str, str]] = []
        for name, value in fields.items():
            turn = self._associated_turn(name, labels)
            kind, other = (UNKNOWN, "") if turn is None else self._classify(value, turn)
            out.append((name, value, kind, other))
        return out

    def describe(self, arguments_json: str) -> str:
        """The factual line for one lookup's arguments ("" when the arguments are not a JSON object)."""
        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError:
            return ""
        if not isinstance(arguments, dict) or not arguments:
            return ""
        parts = []
        for name, value, kind, other in self.attribute(arguments):
            part = f'{name} "{value}" ({kind})'
            if other:
                part += f' (the user spelled "{other}" for the {field_labels(name)[0]})'
            parts.append(part)
        return f"[Fields in this lookup: {', '.join(parts)}.]" if parts else ""

    # -- internals ---------------------------------------------------------------------------

    def _associated_turn(self, name: str, labels: Mapping[str, tuple[str, ...]]) -> TurnEvidence | None:
        others = [label for other, own in labels.items() if other != name for label in own]
        for turn in reversed(self._turns):
            if _mentions(turn.text, labels[name]) and not _mentions(turn.text, others):
                return turn
            if (
                not _mentions(turn.text, [label for own in labels.values() for label in own])
                and _mentions(turn.question, labels[name])
                and not _mentions(turn.question, others)
            ):
                return turn
        return None

    @staticmethod
    def _classify(value: str, turn: TurnEvidence) -> tuple[str, str]:
        if not value or value.isdigit() or len(turn.spelled) > 1:
            return UNKNOWN, ""
        spelled = turn.spelled[0] if turn.spelled else None
        if spelled is not None and spelled.written.lower() == value.lower():
            return SPELLED, ""
        other = spelled.written if spelled is not None else ""
        for match in re.finditer(rf"(?<!\w){re.escape(value)}(?!\w)", turn.text, re.IGNORECASE):
            if spelled is None or match.end() <= spelled.start or match.start() >= spelled.end:
                return HEARD, other
        return UNKNOWN, other
