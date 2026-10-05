# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""The argument screen: put a read tool's dictated identifiers into their schema's shape.

:func:`screen_call` never touches a write tool's arguments: they come back as an
exact copy. For a read tool, each string argument with an identifier rule is
canonicalized, and the canonical value replaces the original only when it lands
on the rule's ``keep_pattern``. A value that still does not fit and is clearly
unfinished is reported as incomplete, so the caller can ask for the rest instead
of looking up a fragment. Screening is idempotent.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from examples.frontend_backend_agent.src.normalization.rules import canonicalize, is_incomplete
from examples.frontend_backend_agent.src.normalization.schema_rules import SchemaRules


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool call as the model emitted it."""

    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ScreenResult:
    """The screened arguments of one call and what the screen did to them."""

    arguments: dict[str, Any]
    #: Arguments rewritten into their rule's shape.
    changed: tuple[str, ...]
    #: Arguments judged clearly unfinished (left as given).
    incomplete: tuple[str, ...]
    kind: Literal["read", "write"]


def screen_call(rules: SchemaRules, call: ToolCall) -> ScreenResult:
    """Canonicalize a read call's identifier arguments; return a write call's arguments untouched."""
    arguments = copy.deepcopy(dict(call.arguments))
    if call.name not in rules.read_tools:
        return ScreenResult(arguments=arguments, changed=(), incomplete=(), kind="write")
    changed: list[str] = []
    incomplete: list[str] = []
    for argument, value in arguments.items():
        rule = rules.rule_for(call.name, argument)
        if rule is None or not isinstance(value, str) or rule.keep_pattern.fullmatch(value):
            continue
        canonical = canonicalize(rule, value)
        if rule.keep_pattern.fullmatch(canonical):
            arguments[argument] = canonical
            changed.append(argument)
        elif is_incomplete(rule, value):
            incomplete.append(argument)
    return ScreenResult(arguments=arguments, changed=tuple(changed), incomplete=tuple(incomplete), kind="read")


def canonical_json(arguments: Mapping[str, Any]) -> str:
    """The canonical JSON of call arguments (the encoding of ``client_call_fingerprint``)."""
    return json.dumps(arguments, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
