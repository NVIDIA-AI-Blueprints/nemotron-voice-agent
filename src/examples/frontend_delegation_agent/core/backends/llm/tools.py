# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool schemas, tool-name repair, argument checks and failure warnings of the LLM backend.

The rules are in llm-backend-plan.md section 6.4, and the tests check them against recorded backend traces:

* :func:`model_tools` sends the name, the description (the name when empty) and the parameters with
  every object node completed (``properties`` and a ``required`` list naming only existing properties).
* :func:`repair_name` maps a near-miss tool name to a session tool. The fuzzy step uses a 0.7 cutoff,
  which decides whether an invented name runs a real (possibly writing) tool.
* :class:`FailureWarnings` appends one of two loop warnings to failing tool results.

Stdlib only.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import re
from collections.abc import Callable, Iterable
from typing import Any

#: The fuzzy-match cutoff: 0.6 or lower maps invented telecom names onto write tools.
NAME_CUTOFF = 0.7
EXACT_FAILURE_WARN_AFTER = 2
SAME_TOOL_FAILURE_WARN_AFTER = 3

_SCHEMA_CHILDREN = ("items", "additionalProperties", "not", "if", "then", "else", "contains")
_SCHEMA_LISTS = ("anyOf", "oneOf", "allOf", "prefixItems")
_SCHEMA_MAPS = ("properties", "$defs", "definitions", "patternProperties")
_QUOTES_AND_TAGS = ('"', "'", "<", ">")
_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")


def normalize_schema(node: Any) -> Any:
    """A copy of ``node`` whose object nodes all have ``properties`` and a valid ``required`` list."""
    if isinstance(node, list):
        return [normalize_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for key, value in node.items():
        if key in _SCHEMA_MAPS and isinstance(value, dict):
            out[key] = {name: normalize_schema(child) for name, child in value.items()}
        elif (key in _SCHEMA_LISTS and isinstance(value, list)) or (
            key in _SCHEMA_CHILDREN and isinstance(value, dict)
        ):
            out[key] = normalize_schema(value)
        else:
            out[key] = copy.deepcopy(value)
    if out.get("type") == "object":
        if not isinstance(out.get("properties"), dict):
            out["properties"] = {}
        required = out.get("required")
        out["required"] = (
            [name for name in required if isinstance(name, str) and name in out["properties"]]
            if isinstance(required, list)
            else []
        )
    return out


def model_tools(tools: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ``tools`` request field for the session's tools (``configure`` gives name, description, parameters)."""
    out = []
    for tool in tools:
        name = str(tool["name"])
        parameters = tool.get("parameters") or {"type": "object", "properties": {}}
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool.get("description") or name),
                    "parameters": normalize_schema(parameters),
                },
            }
        )
    return out


def repair_name(name: str, valid: Iterable[str]) -> str | None:
    """The session tool ``name`` was meant to be, or ``None``.

    In order: cut at the first quote or angle bracket; lower case; ``-`` and spaces to ``_``;
    CamelCase to snake_case and a ``tool`` suffix removed (twice); else the closest name at
    :data:`NAME_CUTOFF`. Candidates are sorted, so a tie always resolves the same way.
    """
    names = sorted(set(valid))
    for mark in _QUOTES_AND_TAGS:
        cut = name.find(mark)
        if cut > 0:
            name = name[:cut]
    if not name:
        return None
    lowered = name.lower()

    def snake(text: str) -> str:
        return text.lower().replace("-", "_").replace(" ", "_")

    def camel(text: str) -> str:
        return _CAMEL.sub("_", text).lower()

    def without_suffix(text: str) -> str | None:
        lower = text.lower()
        for suffix in ("_tool", "-tool", "tool"):
            if lower.endswith(suffix):
                return text[: -len(suffix)].rstrip("_-")
        return None

    for candidate in (lowered, snake(name)):
        if candidate in names:
            return candidate
    candidates = {name, lowered, snake(name), camel(name)}
    for _ in range(2):
        stripped = {s for s in (without_suffix(c) for c in candidates) if s}
        candidates |= stripped | {snake(s) for s in stripped} | {camel(s) for s in stripped}
    for candidate in sorted(candidates):
        if candidate in names:
            return candidate
    close = difflib.get_close_matches(lowered, names, n=1, cutoff=NAME_CUTOFF)
    return close[0] if close else None


def parse_arguments(raw: Any) -> tuple[dict[str, Any] | None, str]:
    """``(arguments, "")`` or ``(None, error)``; empty arguments are ``{}``, dicts pass through."""
    if isinstance(raw, dict):
        return raw, ""
    text = "" if raw is None else str(raw)
    if not text.strip():
        return {}, ""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(value, dict):
        return None, "Tool arguments must be a JSON object"
    return value, ""


def canonical_arguments(args: Any) -> str:
    """Arguments as they are replayed in the history: sorted keys, no spaces, ASCII escapes."""
    return json.dumps(args, sort_keys=True, separators=(",", ":"))


def is_failure(result: str) -> bool:
    """The failure test for a tool result: starts with ``Error`` or names an ``"error"``/``"failed"`` key."""
    head = result[:500].lower()
    return result.startswith("Error") or '"error"' in head or '"failed"' in head


class FailureWarnings:
    """Per-run counters of failing calls; appends a loop warning to the result that crosses a threshold.

    ``render(key, **vars)`` renders the warning texts (``harness_tool_loop_warning`` and the two
    loop messages of ``llm.yaml``). A successful call clears its counters.
    """

    def __init__(self, render: Callable[..., str]) -> None:
        """No failures yet."""
        self._render = render
        self._exact: dict[tuple[str, str], int] = {}
        self._same: dict[str, int] = {}

    def after(self, name: str, args: dict[str, Any], result: str) -> str:
        """``result``, with a warning appended when this failure repeats."""
        signature = (name, hashlib.sha256(json.dumps(args, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
        if not is_failure(result):
            self._exact.pop(signature, None)
            self._same.pop(name, None)
            return result
        exact = self._exact[signature] = self._exact.get(signature, 0) + 1
        same = self._same[name] = self._same.get(name, 0) + 1
        if exact >= EXACT_FAILURE_WARN_AFTER:
            code, count, key = "repeated_exact_failure_warning", exact, "harness_loop_exact_failure"
        elif same >= SAME_TOOL_FAILURE_WARN_AFTER:
            code, count, key = "same_tool_failure_warning", same, "harness_loop_same_tool_failure"
        else:
            return result
        message = self._render(key, tool_name=name, count=count)
        return result + "\n\n" + self._render("harness_tool_loop_warning", code=code, count=count, message=message)
