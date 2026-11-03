# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ID recovery: after an app lookup finds nothing, point the backend at the one written form worth trying.

Speech recognition guesses letter case, spacing and symbols, so the backend often sends an id the caller spelled right
in a form the app does not know (``Ethan_Martin_2396`` where the app keeps ``ethan_martin_2396``). When an app call's
output says nothing was found, :class:`ShapeHints` writes the value that failed in the shape the app's ids take, learned
from the tool's parameter description (``such as 'sara_doe_496'``), from values that worked earlier in the session, and
from fields of the same name in earlier outputs. It adds that form to the output the backend reads, once, and only when
the session has not tried it yet. It never changes a letter or a digit, and it never refuses a call.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass

# A quoted example in a parameter's description: 'sara_doe_496' or "#W0000000" (no spaces inside).
_QUOTED = re.compile(r"'([^'\s]+)'|\"([^\"\s]+)\"")
# The separators an id's groups may be joined with, and every character besides letters and digits an id may hold.
_SEPARATORS = "_-. "
_DROPPED = re.compile(r"[\s_\-.()]+")
# The groups of a value: runs of letters and digits between separators.
_GROUPS = re.compile(r"[^\W_]+")
# Values longer than this, or holding more than this many groups, are not ids.
_MAX_ID_CHARS = 64
_MAX_GROUPS = 8


def skeleton(value: str) -> str:
    """``value``'s letters and digits in order, with letter case ignored: every written form of one id shares it."""
    return "".join(_GROUPS.findall(value)).casefold()


@dataclass(frozen=True)
class Shape:
    """How an id is written: a leading mark, the letter case, the separator between groups, digit group sizes."""

    prefix: str  # leading characters that are not letters or digits, such as "#"
    case: str | None  # "lower", "upper", or None when letters of both cases appear
    separator: str | None  # the one separator between groups ("" for none); None when several kinds appear
    has_letters: bool
    has_digits: bool
    digit_groups: tuple[int, ...]  # group sizes of an id of digits only, such as (3, 3, 4)


def shape_of(example: str) -> Shape | None:
    """The shape of ``example``; ``None`` when it does not look like an id."""
    if not example or len(example) > _MAX_ID_CHARS:
        return None
    prefix = example[: len(example) - len(example.lstrip("#@+"))]
    body = example[len(prefix) :]
    groups = _GROUPS.findall(body)
    if not groups or len(groups) > _MAX_GROUPS or "".join(groups) != _DROPPED.sub("", body):
        return None
    between = set(re.sub(r"[^\W_]+", "\0", body).strip("\0").split("\0")) - {""}
    if between - set(_SEPARATORS):
        return None
    letters = [char for char in body if char.isalpha()]
    case = None
    if letters and all(char.islower() for char in letters):
        case = "lower"
    elif letters and all(char.isupper() for char in letters):
        case = "upper"
    has_digits = any(char.isdigit() for char in body)
    digit_groups = tuple(len(group) for group in groups) if not letters and len(groups) > 1 else ()
    separator = "" if not between else between.pop() if len(between) == 1 else None
    return Shape(prefix, case, separator, bool(letters), has_digits, digit_groups)


def apply_shape(value: str, shape: Shape) -> str | None:
    """``value`` written in ``shape``: the same letters and digits in the same order; ``None`` when it does not fit."""
    target = shape_of(value)
    if target is None or (target.has_letters, target.has_digits) != (shape.has_letters, shape.has_digits):
        return None
    groups = _GROUPS.findall(value[len(target.prefix) :])
    if shape.digit_groups:
        digits = "".join(groups)
        if sum(shape.digit_groups) != len(digits):
            return None
        groups, start = [], 0
        for size in shape.digit_groups:
            groups.append(digits[start : start + size])
            start += size
    # With several kinds of separator in the example, only its letter case can be trusted.
    joined = value[len(target.prefix) :] if shape.separator is None else shape.separator.join(groups)
    if shape.case == "lower":
        joined = joined.lower()
    elif shape.case == "upper":
        joined = joined.upper()
    shaped = shape.prefix + joined
    return shaped if shaped != value and skeleton(shaped) == skeleton(value) else None


def _string_arguments(call: Mapping) -> dict[str, str]:
    """A call's arguments that are strings, by name; empty when the arguments are not a JSON object."""
    try:
        arguments = json.loads(call.get("arguments") or "{}")
    except (TypeError, ValueError):
        return {}
    if not isinstance(arguments, dict):
        return {}
    return {name: value for name, value in arguments.items() if isinstance(value, str) and value}


def failed_value(arguments: Mapping[str, str], output: str) -> tuple[str, str] | None:
    """The argument (name, value) a "not found" output is about: the one it quotes, else the call's only string."""
    quoted = [(name, value) for name, value in arguments.items() if value in output]
    if len(quoted) == 1:
        return quoted[0]
    if not quoted and len(arguments) == 1:
        return next(iter(arguments.items()))
    return None


def _function(tool: Mapping) -> Mapping:
    """A function tool's definition, in the Responses form or the chat-completions form."""
    function = tool.get("function")
    return function if isinstance(function, Mapping) else tool


def description_examples(tools: Iterable[Mapping]) -> dict[tuple[str, str], str]:
    """The first quoted example of each tool parameter's description, by (tool name, parameter name)."""
    examples: dict[tuple[str, str], str] = {}
    for tool in tools:
        function = _function(tool)
        properties = (function.get("parameters") or {}).get("properties") or {}
        for parameter, schema in properties.items() if isinstance(properties, Mapping) else ():
            description = schema.get("description") if isinstance(schema, Mapping) else None
            match = _QUOTED.search(description) if isinstance(description, str) else None
            if match:
                examples[(str(function.get("name", "")), parameter)] = match.group(1) or match.group(2)
    return examples


def _fields(value: object) -> Iterator[tuple[str, str]]:
    """Every (key, string value) pair in a parsed JSON value, at any depth."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(item, str):
                yield str(key), item
            else:
                yield from _fields(item)
    elif isinstance(value, list):
        for item in value:
            yield from _fields(item)


class ShapeHints:
    """One session's learned id shapes and tried forms, and the hint a "not found" output gets."""

    def __init__(self, not_found: str, hint: str, observe: Callable[..., None]) -> None:
        """Start with nothing learned.

        Args:
            not_found: The pattern an app output matches when the value it was given names nothing.
            hint: The line added to such an output (``texts.shape_hint``, with ``{value}``).
            observe: ``observe(kind, **fields)`` for traces.
        """
        self.not_found = re.compile(not_found)
        self.hint = hint
        self.observe = observe
        self.worked: dict[tuple[str, str], str] = {}  # (tool, parameter): the newest value that worked
        self.fields: dict[str, str] = {}  # field name: the newest string an app output held for it
        self.tried: dict[str, set[str]] = {}  # skeleton: every form sent in this session

    def outputs(self, calls: Iterable[tuple[dict, dict]], tools: Iterable[Mapping]) -> dict[str, dict]:
        """The outputs of the app's ``calls`` (call, output item) as the backend reads them, by call id.

        Each output is the app's own, except that a "not found" output gets the hint when a shape is known and its form
        is new. Outputs that found something teach the shapes first, so a call that worked in the same batch counts.
        """
        calls = list(calls)
        examples = description_examples(tools)
        failures = []
        for call, output in calls:
            name, arguments, text = call.get("name", ""), _string_arguments(call), output.get("output")
            for value in arguments.values():
                self.tried.setdefault(skeleton(value), set()).add(value)
            if not isinstance(text, str):
                continue
            if self.not_found.search(text):
                failures.append((output, name, arguments, text))
            else:
                self._learn(name, arguments, text)
        hinted = {output["call_id"]: output for _, output in calls}
        for output, name, arguments, text in failures:
            suggestion = self._suggest(name, arguments, text, examples)
            if suggestion is not None:
                hinted[output["call_id"]] = {**output, "output": f"{text}\n{self.hint.format(value=suggestion)}"}
        return hinted

    def _learn(self, name: str, arguments: Mapping[str, str], text: str) -> None:
        """A call that found something: its values worked, and its output's fields are values the app knows."""
        for parameter, value in arguments.items():
            self.worked[(name, parameter)] = value
        try:
            parsed = json.loads(text)
        except ValueError:
            return
        for key, value in _fields(parsed):
            if shape_of(value) is not None:
                self.fields[key] = value

    def _suggest(
        self, name: str, arguments: Mapping[str, str], text: str, examples: Mapping[tuple[str, str], str]
    ) -> str | None:
        """The failed value in a learned shape, when that form is new to the session."""
        failed = failed_value(arguments, text)
        if failed is None:
            return None
        parameter, value = failed
        sources = (
            ("session", self.worked.get((name, parameter))),
            ("session", self.fields.get(parameter)),
            ("description", examples.get((name, parameter))),
        )
        for source, example in sources:
            shape = shape_of(example) if example else None
            suggestion = apply_shape(value, shape) if shape is not None else None
            if suggestion is None:
                continue
            if suggestion in self.tried.get(skeleton(value), set()):
                return None
            self.observe("tools.shape_hint", tool=name, failed=value, suggested=suggestion, source=source)
            return suggestion
        return None
