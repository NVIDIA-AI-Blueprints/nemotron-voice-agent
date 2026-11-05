# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ID recovery: after an app lookup finds nothing, point the backend at the one written form worth trying.

Speech recognition guesses letter case, spacing and symbols, so the backend often sends an id the caller spelled right
in a form the app does not know (``Maya_Ross_3817`` where the app keeps ``maya_ross_3817``). When an app call's
output says nothing was found, :class:`ShapeHints` writes the value that failed in the shape the app's ids take, learned
from the tool's parameter description (``such as 'lee_park_42'``), from values that worked earlier in the session, and
from fields of the same name in earlier outputs. It adds that form to the output the backend reads, once, and only when
the session has not tried it yet. It never changes a letter or a digit, and it never refuses a call. It is the app
output step ``shape_hints`` (``backend.app_output_steps``), whose block names the pattern of an output that found
nothing and the hint's words.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

# A quoted example in a parameter's description: 'lee_park_42' or "#A1234567" (no spaces inside).
_QUOTED = re.compile(r"'([^'\s]+)'|\"([^\"\s]+)\"")
# The separators an id's groups may be joined with, and every character besides letters and digits an id may hold.
_SEPARATORS = "_-. "
_DROPPED = re.compile(r"[\s_\-.()]+")
# The groups of a value: runs of letters and digits between separators.
_GROUPS = re.compile(r"[^\W_]+")
# An app output that reports an error rather than a result (its text starts with "Error" or "Exception", or its JSON
# object has a non-empty "error" or "errors" key): the values it was given did not work.
_ERROR_TEXT = re.compile(r"(?i)^\W*(error|exception)\b")
_ERROR_KEYS = ("error", "errors")
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
    """The argument (name, value) a "not found" output is about: the one it quotes, else the call's only string.

    A value counts as quoted only where it stands as a whole word, so a short value such as ``1`` is not found inside
    ``10``.
    """
    quoted = [(name, value) for name, value in arguments.items() if value and _quotes(output, value)]
    if len(quoted) == 1:
        return quoted[0]
    if not quoted and len(arguments) == 1:
        return next(iter(arguments.items()))
    return None


def _quotes(output: str, value: str) -> bool:
    """Whether ``output`` holds ``value`` with no letter, digit or underscore right before or after it."""
    return re.search(rf"(?<!\w){re.escape(value)}(?!\w)", output) is not None


def _function(tool: Mapping) -> Mapping:
    """A function tool's definition, in the Responses form or the chat-completions form."""
    function = tool.get("function")
    return function if isinstance(function, Mapping) else tool


def description_examples(tools: Iterable[Mapping] | None) -> dict[tuple[str, str], str]:
    """The first quoted example of each tool parameter's description, by (tool name, parameter name).

    A tool, or its parameters, that is not a mapping teaches nothing: no tool list fails the outputs it helps read.
    """
    examples: dict[tuple[str, str], str] = {}
    for tool in tools or ():
        if not isinstance(tool, Mapping):
            continue
        function = _function(tool)
        parameters = function.get("parameters")
        properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
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
            hint: The line added to such an output, with ``{value}``.
            observe: ``observe(kind, **fields)`` for traces.
        """
        self.not_found = re.compile(not_found)
        self.hint = hint
        self.observe = observe
        self.worked: dict[tuple[str, str], str] = {}  # (tool, parameter): the newest value that worked
        self.fields: dict[str, str] = {}  # field name: the newest string an app output held for it
        self.tried: dict[str, set[str]] = {}  # skeleton: every form sent or suggested in this session

    @classmethod
    def from_settings(cls, block: Mapping[str, Any], *, observe: Callable[..., None]) -> ShapeHints:
        """The step of one IM session, from its ``backend.app_output_steps.shape_hints`` block.

        The block sets ``not_found_pattern``, the pattern of an app output that names nothing, and ``hint``, the line
        added to such an output, with ``{value}``.

        Raises:
            ValueError: The block sets another key or leaves one out, the pattern is not a regular expression, or the
                hint does not hold ``{value}`` or names another field.
        """
        where = "backend.app_output_steps.shape_hints"
        if not isinstance(block, Mapping) or set(block) != {"not_found_pattern", "hint"}:
            raise ValueError(f"{where} must set not_found_pattern and hint, and nothing else")
        pattern, hint = block["not_found_pattern"], block["hint"]
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError(f"{where}.not_found_pattern must be non-empty text")
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(f"{where}.not_found_pattern is not a valid regular expression: {error}") from None
        if not isinstance(hint, str) or "{value}" not in hint:
            raise ValueError(f"{where}.hint must hold {{value}}")
        try:
            hint.format(value="")
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(f"{where}.hint names a field it is not given: {exc}") from None
        return cls(pattern, hint, observe)

    def outputs(self, calls: Iterable[tuple[dict, dict]], tools: Iterable[Mapping] | None) -> dict[str, dict]:
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
        """A call that found something: its values worked, and its output's fields are values the app knows.

        An output that reports an error teaches nothing, and an output too deeply nested to read teaches only its
        call's values: learning never fails the outputs it reads.
        """
        if _ERROR_TEXT.match(text):
            return
        try:
            parsed = json.loads(text)
            pairs = list(_fields(parsed))
        except (ValueError, RecursionError):
            parsed, pairs = None, []
        if isinstance(parsed, Mapping) and any(parsed.get(key) for key in _ERROR_KEYS):
            return
        for parameter, value in arguments.items():
            self.worked[(name, parameter)] = value
        for key, value in pairs:
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
            forms = self.tried.setdefault(skeleton(value), set())
            if suggestion in forms:
                return None
            forms.add(suggestion)  # suggested once: a retry of the same failed form gets no second hint
            self.observe("tools.shape_hint", tool=name, failed=value, suggested=suggestion, source=source)
            return suggestion
        return None
