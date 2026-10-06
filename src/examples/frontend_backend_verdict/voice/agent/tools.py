# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Realtime tool schema <-> text-prototype ``ToolSpec`` and ``ToolResult``."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from examples.frontend_backend_verdict.text.messages import ToolCall, ToolResult
from examples.frontend_backend_verdict.text.tools import ToolSpec
from examples.frontend_backend_verdict.voice.agent.port import OutgoingCall

_FIRST_SENTENCE = re.compile(r"^(.+?[.!?])(\s|$)", re.DOTALL)


def realtime_tools_to_specs(tools: Sequence[Mapping[str, Any]]) -> tuple[ToolSpec, ...]:
    """Flat Realtime function tools -> ``ToolSpec`` without callables (external execution)."""
    specs: list[ToolSpec] = []
    seen: set[str] = set()
    for tool in tools:
        name = str(tool.get("name") or "").strip()
        if not name:
            raise ValueError("tool without a name")
        if name in seen:
            raise ValueError(f"duplicate tool name {name!r}")
        seen.add(name)
        parameters = tool.get("parameters") or {"type": "object", "properties": {}}
        specs.append(ToolSpec(name=name, description=str(tool.get("description") or ""), parameters=dict(parameters)))
    return tuple(specs)


def to_outgoing(calls: Sequence[ToolCall]) -> tuple[OutgoingCall, ...]:
    """Backend tool calls -> wire calls; the backend's id is used verbatim as ``call_id``."""
    return tuple(OutgoingCall(call_id=call.id, name=call.name, arguments=call.arguments_json) for call in calls)


def to_results(outputs: Mapping[str, str]) -> tuple[ToolResult, ...]:
    """``function_call_output`` items -> ``ToolResult``; outputs are passed verbatim (tau2 errors included)."""
    return tuple(ToolResult(tool_call_id=call_id, content=output) for call_id, output in outputs.items())


def first_sentence(description: str) -> str:
    """The first sentence of a tool description (the whole text when it has no sentence end)."""
    text = " ".join(description.split())
    match = _FIRST_SENTENCE.match(text)
    return match.group(1) if match else text


def capability_lines(specs: Sequence[ToolSpec]) -> tuple[str, ...]:
    """``<name>: <first sentence>`` per tool, declaration order, deduplicated."""
    lines: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.name in seen:
            continue
        seen.add(spec.name)
        summary = first_sentence(spec.description)
        lines.append(f"{spec.name}: {summary}" if summary else spec.name)
    return tuple(lines)
