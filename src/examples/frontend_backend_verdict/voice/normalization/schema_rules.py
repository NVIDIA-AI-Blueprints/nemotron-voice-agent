# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool-argument rules derived from a session's tool schemas (``rules: auto``).

The client's ``session.update`` tools are the only input: names, argument names and
types, and the example quoted in an argument's description ("such as 'ab_cd_12'").
Nothing here knows a tool or a domain by name. From those schemas:

* **identifier rules**: for every *read* tool (see :func:`is_read_tool`; anything else
  counts as a write tool and is never rewritten), each string argument named ``id`` or
  ``*_id`` whose description quotes an example gets an :class:`ArgumentRule` derived
  from that example: separator, case, characters to strip (only spaces, dots and dashes
  the example does not contain), a literal symbol prefix to restore (``#``; never a
  letter or digit), and a loose pattern that rejects only clearly incomplete values
  (fewer segments than the example, an empty segment, or no digit where the example
  ends in digits). A value already in the example's shape is never changed.
* **spelling-hold patterns**: the strict shape (lengths and character classes) of the
  example of every identifier argument (``id``, ``*_id``, ``*_ids``) of any tool. A
  spelled value matching one is complete and not held. Examples of other arguments
  (names, states, dates, codes such as ``'CA'``) are never used, and neither are
  examples with a name-like part (a letters-only segment, as in ``'ab_cd_12'``): their
  length varies from value to value, so one example's digit count cannot tell when a
  spelling is complete. Such values are held (at most ``hold_ms``), never cut short.
* **read tools**: the tools result hints watch (``result_hints.tools: auto``).
* **phone arguments**: string arguments of read tools whose name contains ``phone``
  (reported only; no code acts on them).

Pure: no I/O. See ``tau3-voice-domain-specific-changes-genericization.md`` sections 4.5,
4.7, 4.9 and 5.6.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from examples.frontend_backend_verdict.voice.normalization.arguments import ArgumentRule

#: Tool-name prefixes and description openings that mark a read (lookup) tool; anything else is a write tool.
READ_PREFIXES = ("get_", "find_", "search_", "list_", "lookup_", "check_", "calculate")
READ_DESCRIPTION_OPENINGS = ("get", "find", "search", "list", "look up")
#: Characters a rule may strip, unless the example contains them.
STRIPPABLE = " .-"
#: Words naming a person-type entity (result hints count local invalid answers for such arguments only).
PERSON_WORDS = ("user", "customer", "member", "client", "profile")

_EXAMPLE = re.compile(r"""(?:such as|e\.g\.,?|for example)\s*(?:'([^'\s]+)'|"([^"\s]+)")""", re.IGNORECASE)
_PERSON_ARGUMENT = re.compile(r"(?:^|_)(?:" + "|".join(PERSON_WORDS) + r")(?:_|$)")
_ALNUM = re.compile(r"[A-Za-z0-9]+")
_LETTERS_THEN_DIGITS = re.compile(r"([A-Za-z]+)(\d+)")


@dataclass(frozen=True, slots=True)
class SchemaRules:
    """Everything derived from one tool list."""

    rules: tuple[ArgumentRule, ...] = ()
    complete_patterns: tuple[str, ...] = ()
    read_tools: tuple[str, ...] = ()
    phone_arguments: tuple[str, ...] = ()  # "tool.argument"

    def snapshot(self) -> dict[str, Any]:
        """A JSON-able summary (logged per session, and the reviewed test snapshot)."""
        return {
            "rules": [
                {
                    "tool": rule.tool,
                    "argument": rule.argument,
                    "label": rule.label,
                    "format_hint": rule.format_hint,
                    "strip": rule.strip,
                    "collapse_separators": rule.collapse_separators,
                    "case": rule.case,
                    "restore_prefix": rule.restore_prefix,
                    "keep_pattern": rule.keep_pattern,
                    "pattern": rule.pattern,
                }
                for rule in self.rules
            ],
            "complete_patterns": list(self.complete_patterns),
            "read_tools": list(self.read_tools),
            "phone_arguments": list(self.phone_arguments),
        }


@dataclass(frozen=True, slots=True)
class _Shape:
    """An example split into a symbol prefix and alphanumeric segments joined by one separator."""

    prefix: str
    separator: str
    segments: tuple[str, ...]
    case: str


def tool_function(tool: Mapping[str, Any]) -> Mapping[str, Any]:
    """The function part of a Realtime (flat) or Chat Completions (``function``-wrapped) tool."""
    inner = tool.get("function")
    return inner if isinstance(inner, Mapping) else tool


def is_read_tool(name: str, description: str) -> bool:
    """A lookup by its name or the opening of its description; unknown tools count as write tools."""
    return name.startswith(READ_PREFIXES) or description.lstrip().lower().startswith(READ_DESCRIPTION_OPENINGS)


def is_identifier_argument(name: str, *, plural: bool = False) -> bool:
    """``id`` or ``*_id`` (and ``*_ids`` with ``plural``)."""
    return name == "id" or name.endswith("_id") or (plural and name.endswith("_ids"))


def is_person_argument(name: str) -> bool:
    """An identifier argument naming a person-type entity (``user_id``, ``customer_id``, ...)."""
    return _PERSON_ARGUMENT.search(name.lower()) is not None


def first_example(description: str) -> str:
    """The first quoted single-token example ("such as 'x'", "e.g. 'x'", "for example 'x'"), or ""."""
    match = _EXAMPLE.search(description or "")
    return (match.group(1) or match.group(2)) if match else ""


def tools_sha256(tools: Sequence[Mapping[str, Any]]) -> str:
    """Fingerprint of a session's tool schemas (sorted by name; first 16 hex digits)."""
    ordered = sorted((dict(tool) for tool in tools), key=lambda tool: str(tool_function(tool).get("name") or ""))
    blob = json.dumps(ordered, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def build_schema_rules(tools: Sequence[Mapping[str, Any]]) -> SchemaRules:
    """Derive identifier rules, hold patterns, read tools and phone arguments from ``tools``."""
    rules: list[ArgumentRule] = []
    patterns: list[str] = []
    read_tools: list[str] = []
    phones: list[str] = []
    for tool in tools:
        function = tool_function(tool)
        name = str(function.get("name") or "")
        if not name:
            continue
        read = is_read_tool(name, str(function.get("description") or ""))
        if read:
            read_tools.append(name)
        parameters = function.get("parameters")
        properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
        for argument, spec in (properties or {}).items():
            if not isinstance(spec, Mapping):
                continue
            string = spec.get("type") == "string"
            if read and string and "phone" in argument.lower():
                phones.append(f"{name}.{argument}")
            if not is_identifier_argument(argument, plural=True):
                continue
            shape = _shape(first_example(str(spec.get("description") or "")))
            if shape is None:
                continue
            pattern = _complete_pattern(shape)
            if pattern and pattern not in patterns:
                patterns.append(pattern)
            if read and string and is_identifier_argument(argument):
                rules.append(_rule(name, argument, shape))
    return SchemaRules(
        rules=tuple(rules),
        complete_patterns=tuple(patterns),
        read_tools=tuple(read_tools),
        phone_arguments=tuple(phones),
    )


# -- shapes --------------------------------------------------------------------------------


def _shape(example: str) -> _Shape | None:
    """Split an example; ``None`` when it is not an identifier-like token."""
    body = example.lstrip("".join(ch for ch in example if not ch.isalnum()))
    prefix = example[: len(example) - len(body)]
    if not body or not body.isascii():
        return None
    separators = {ch for ch in body if not ch.isalnum()}
    if len(separators) > 1:
        return None
    separator = separators.pop() if separators else ""
    segments = tuple(body.split(separator)) if separator else (body,)
    if not all(_ALNUM.fullmatch(segment) for segment in segments):
        return None
    letters = [ch for ch in body if ch.isalpha()]
    if letters and all(ch.islower() for ch in letters):
        case = "lower"
    elif letters and all(ch.isupper() for ch in letters):
        case = "upper"
    else:
        case = "keep"
    return _Shape(prefix=prefix, separator=separator, segments=segments, case=case)


def _letter_class(shape: _Shape) -> str:
    return {"lower": "[a-z]", "upper": "[A-Z]"}.get(shape.case, "[A-Za-z]")


def _alnum_class(shape: _Shape) -> str:
    return {"lower": "[a-z0-9]", "upper": "[A-Z0-9]"}.get(shape.case, "[A-Za-z0-9]")


def _count(n: int) -> str:
    return "" if n == 1 else f"{{{n}}}"


def _literal(text: str) -> str:
    return "".join(ch if ch in "#_-@%&~:;,=!/<>'\"" else re.escape(ch) for ch in text)


def _complete_pattern(shape: _Shape) -> str:
    """The strict shape with exact lengths; "" for a name-like example (its length is not fixed)."""
    if len(shape.segments) > 1 and any(segment.isalpha() for segment in shape.segments):
        return ""
    letter, alnum = _letter_class(shape), _alnum_class(shape)
    parts: list[str] = []
    for segment in shape.segments:
        if segment.isdigit():
            parts.append(rf"\d{_count(len(segment))}")
        elif segment.isalpha():
            parts.append(f"{letter}{_count(len(segment))}")
        elif match := _LETTERS_THEN_DIGITS.fullmatch(segment):
            parts.append(f"{letter}{_count(len(match.group(1)))}" + rf"\d{_count(len(match.group(2)))}")
        else:
            parts.append(f"{alnum}{_count(len(segment))}")
    prefix = "".join(f"{_literal(ch)}?" for ch in shape.prefix)
    return "^" + prefix + _literal(shape.separator).join(parts) + "$"


def _keep_pattern(shape: _Shape) -> str:
    """A value already well formed (any lengths, the example's classes and case): never rewritten."""
    letter, alnum = _letter_class(shape), _alnum_class(shape)
    parts: list[str] = []
    for segment in shape.segments:
        if segment.isdigit():
            parts.append(r"\d+")
        elif segment.isalpha():
            parts.append(f"{letter}+")
        elif _LETTERS_THEN_DIGITS.fullmatch(segment):
            parts.append(rf"{letter}+\d+")
        else:
            parts.append(f"{alnum}+")
    return "^" + _literal(shape.prefix) + _literal(shape.separator).join(parts) + "$"


def _incomplete_guard(shape: _Shape) -> str:
    """Full-matches unless the value is clearly incomplete (never over lengths, classes or the prefix)."""
    ends_in_digits = shape.segments[-1].isdigit()
    if not shape.separator:
        return r"^.*\d.*$" if ends_in_digits else r"^.+$"
    sep = _literal(shape.separator)
    other = f"[^{re.escape(shape.separator)}]"
    n = len(shape.segments)
    if ends_in_digits:
        return f"^{other}+(?:{sep}{other}+){{{n - 2},}}{sep}{other}*\\d{other}*$"
    return f"^{other}+(?:{sep}{other}+){{{n - 1},}}$"


def _format_hint(shape: _Shape) -> str:
    words = []
    for segment in shape.segments:
        if segment.isdigit():
            words.append("digits")
        elif segment.isalpha():
            words.append("name" if len(shape.segments) > 1 else "letters")
        else:
            words.append("code")
    return shape.prefix + (shape.separator or "").join(words)


def _rule(tool: str, argument: str, shape: _Shape) -> ArgumentRule:
    example = shape.prefix + shape.separator.join(shape.segments)
    return ArgumentRule(
        tool=tool,
        argument=argument,
        label=argument.replace("_", " ").removesuffix(" id") + " ID" if argument != "id" else "ID",
        format_hint=_format_hint(shape),
        spoken_form=True,
        strip="".join(ch for ch in STRIPPABLE if ch not in example),
        collapse_separators=shape.separator,
        case=shape.case,
        pattern=_incomplete_guard(shape),
        on_invalid="answer_locally",
        restore_prefix=shape.prefix,
        keep_pattern=_keep_pattern(shape),
        source="schema",
    )
