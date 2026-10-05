# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Normalization rules derived from a session's tool schemas, and nothing else.

The caller's tool list is the only input: tool names and descriptions, argument
names, types and titles, and the example quoted in an argument's description
("such as 'jordan_lee_82'"). From those schemas:

* **read tools**: tools :func:`.classify.tool_kind` calls reads. Every other tool
  is a write and its arguments are never rewritten.
* **identifier rules**: for each string argument of a read tool whose description
  quotes an identifier-shaped example, the :class:`.rules.IdentifierRule` that
  example implies. An argument without such an example gets no rule.
* **complete patterns**: the exact shape of every fixed identifier example of an
  ``id`` / ``*_id`` / ``*_ids`` argument of any tool. They only decide when a
  spelled value is finished; name-like and non-identifier examples never become
  one.
* **phone arguments**: string arguments of read tools whose name contains ``phone``.
* **labels**: how each argument is said aloud (its ``title``, else its name).

Malformed schemas are skipped, never raised on.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from examples.frontend_backend_agent.src.normalization.classify import tool_kind
from examples.frontend_backend_agent.src.normalization.rules import (
    IdentifierRule,
    complete_pattern_from_example,
    rule_from_example,
)

#: A cue word followed by one quoted example ("such as 'x'", "e.g. \"x\"", "like '#x'").
_EXAMPLE = re.compile(
    r"""(?:such\s+as|e\.\s?g\.|for\s+example|for\s+instance|example|like)\s*[,:]?\s*"""
    r"""(?:'([^'\n]{1,64})'|"([^"\n]{1,64})"|`([^`\n]{1,64})`)""",
    re.IGNORECASE,
)
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class SchemaRules:
    """Everything derived from one tool list."""

    #: ``(tool, argument)`` -> rule; read tools only.
    identifier_rules: Mapping[tuple[str, str], IdentifierRule]
    complete_patterns: tuple[re.Pattern[str], ...]
    read_tools: frozenset[str]
    #: tool -> its string arguments whose name contains ``phone`` (read tools only).
    phone_arguments: Mapping[str, tuple[str, ...]]
    #: ``(tool, argument)`` -> spoken label.
    labels: Mapping[tuple[str, str], str]

    def rule_for(self, tool: str, argument: str) -> IdentifierRule | None:
        """The identifier rule for one argument of one tool, if its schema implies one."""
        return self.identifier_rules.get((tool, argument))

    def snapshot(self) -> dict[str, Any]:
        """A JSON-able summary for logs: counts and rule shapes, never a value from a tool result."""
        return {
            "counts": {
                "read_tools": len(self.read_tools),
                "identifier_rules": len(self.identifier_rules),
                "complete_patterns": len(self.complete_patterns),
                "phone_arguments": sum(len(names) for names in self.phone_arguments.values()),
            },
            "read_tools": sorted(self.read_tools),
            "identifier_rules": [
                {
                    "tool": tool,
                    "argument": argument,
                    "separator": rule.separator,
                    "case": rule.case,
                    "strip_chars": rule.strip_chars,
                    "prefix": rule.prefix,
                    "keep_pattern": rule.keep_pattern.pattern,
                    "min_length": rule.min_length,
                    "max_length": rule.max_length,
                    "segments": rule.segments,
                    "digit_tail": rule.digit_tail,
                }
                for (tool, argument), rule in sorted(self.identifier_rules.items())
            ],
            "complete_patterns": [pattern.pattern for pattern in self.complete_patterns],
            "phone_arguments": {tool: list(names) for tool, names in sorted(self.phone_arguments.items())},
        }


EMPTY_SCHEMA_RULES = SchemaRules(
    identifier_rules=MappingProxyType({}),
    complete_patterns=(),
    read_tools=frozenset(),
    phone_arguments=MappingProxyType({}),
    labels=MappingProxyType({}),
)


def tools_sha256(tools: Sequence[Mapping[str, Any]]) -> str:
    """SHA-256 hex digest of the tool list's canonical JSON, in the order given."""
    blob = json.dumps(list(tools), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def quoted_example(description: object) -> str | None:
    """The first example quoted after a cue word in a description, if any."""
    if not isinstance(description, str):
        return None
    match = _EXAMPLE.search(description)
    if match is None:
        return None
    return next(group for group in match.groups() if group is not None)


def build_schema_rules(tools: Sequence[Mapping[str, Any]]) -> SchemaRules:
    """Derive read tools, identifier rules, complete patterns, phone arguments and labels from ``tools``.

    A tool is a Realtime function schema (``{"name", "description", "parameters"}``)
    or the same wrapped under ``"function"``. A later tool with an already-seen
    name is ignored.
    """
    rules: dict[tuple[str, str], IdentifierRule] = {}
    patterns: dict[str, re.Pattern[str]] = {}
    read_tools: set[str] = set()
    phones: dict[str, tuple[str, ...]] = {}
    labels: dict[tuple[str, str], str] = {}
    seen: set[str] = set()
    for tool in tools if isinstance(tools, Sequence) and not isinstance(tools, str | bytes) else ():
        function = _function(tool)
        if function is None:
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name or name in seen:
            continue
        seen.add(name)
        description = function.get("description")
        read = tool_kind(name, description if isinstance(description, str) else "") == "read"
        if read:
            read_tools.add(name)
        parameters = function.get("parameters")
        properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
        if not isinstance(properties, Mapping):
            continue
        tool_phones: list[str] = []
        for argument, spec in properties.items():
            if not isinstance(argument, str) or not isinstance(spec, Mapping):
                continue
            labels[(name, argument)] = _label(argument, spec)
            string = _is_string(spec)
            if read and string and "phone" in argument.lower():
                tool_phones.append(argument)
                continue
            if _is_identifier_argument(argument):
                for example in _examples(spec):
                    if (pattern := complete_pattern_from_example(example)) is not None:
                        patterns.setdefault(pattern.pattern, pattern)
            example = quoted_example(spec.get("description")) if read and string else None
            if example is not None and (rule := rule_from_example(example)) is not None:
                rules[(name, argument)] = rule
        if tool_phones:
            phones[name] = tuple(tool_phones)
    return SchemaRules(
        identifier_rules=MappingProxyType(rules),
        complete_patterns=tuple(patterns.values()),
        read_tools=frozenset(read_tools),
        phone_arguments=MappingProxyType(phones),
        labels=MappingProxyType(labels),
    )


def _function(tool: object) -> Mapping[str, Any] | None:
    if not isinstance(tool, Mapping):
        return None
    inner = tool.get("function")
    if "name" not in tool and isinstance(inner, Mapping):
        return inner
    return tool


def _is_string(spec: Mapping[str, Any]) -> bool:
    kind = spec.get("type")
    return kind == "string" or (isinstance(kind, list) and "string" in kind)


def _is_identifier_argument(argument: str) -> bool:
    return argument == "id" or argument.endswith(("_id", "_ids"))


def _examples(spec: Mapping[str, Any]) -> list[str]:
    """Examples quoted on the argument and, for an array, on its items."""
    found = [quoted_example(spec.get("description"))]
    items = spec.get("items")
    if isinstance(items, Mapping):
        found.append(quoted_example(items.get("description")))
    return [example for example in found if example]


def _label(argument: str, spec: Mapping[str, Any]) -> str:
    title = spec.get("title")
    if isinstance(title, str) and title.strip():
        return _SPACE.sub(" ", title.strip()).lower()
    return argument.replace("_", " ")
