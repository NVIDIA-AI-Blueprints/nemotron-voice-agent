# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Declarative internal-tool contracts for Frontend/Backend Agent domains."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """Validation and clarification metadata for one planner argument."""

    kind: type
    required: bool = True
    label: str = ""
    bounds: tuple[float, float] | None = None
    choices: frozenset[str] | None = None
    max_len: int = 200
    default: Any = None


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Per-call dependencies; stateless tools ignore both fields."""

    state: Any = None
    backend: Any = None


ToolRunner = Callable[[Mapping[str, Any], ToolContext], Awaitable[dict[str, Any]]]
ToolSpeaker = Callable[[dict[str, Any], dict[str, Any]], str]
ToolValidator = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Single source of truth for one internal planner tool."""

    name: str
    contract: str
    params: dict[str, ParamSpec]
    run: ToolRunner
    speak: ToolSpeaker | None = None
    validate: ToolValidator | None = None
    mutates: bool = False
    timeout_s: float = 12.0
    capability: str = ""


def validate_arguments(spec: ToolSpec, arguments: Mapping[str, Any]) -> list[str]:
    """Return missing required names and reject every invalid supplied value."""
    unexpected = set(arguments) - set(spec.params)
    if unexpected:
        raise ValueError(f"unexpected params: {sorted(unexpected)}")

    missing = [name for name, param in spec.params.items() if param.required and arguments.get(name) in (None, "")]
    if missing:
        return missing

    for name, param in spec.params.items():
        if name not in arguments:
            continue
        value = arguments[name]
        if param.kind is str:
            if not isinstance(value, str) or not 0 < len(value.strip()) <= param.max_len:
                raise ValueError(f"invalid {name}")
            if param.choices and value.casefold() not in {choice.casefold() for choice in param.choices}:
                raise ValueError(f"invalid {name}")
            continue
        if param.kind is bool:
            if not isinstance(value, bool):
                raise ValueError(f"invalid {name}")
        elif param.kind is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"invalid {name}")
        elif param.kind is float:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"invalid {name}")
        elif not isinstance(value, param.kind):
            raise ValueError(f"invalid {name}")
        if param.bounds:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"invalid {name}")
            if not param.bounds[0] <= float(value) <= param.bounds[1]:
                raise ValueError(f"invalid {name}")
    if spec.validate is not None:
        spec.validate(arguments)
    return []


def _client_tool_contract(tool: Mapping[str, Any]) -> tuple[str, str, Mapping[str, Any]]:
    """Validate the canonical client schema fields used by the Thinker prompt."""
    name = tool.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Client planner tools require a non-empty name")
    description = tool.get("description", "")
    if not isinstance(description, str):
        raise ValueError(f"Client planner tool {name!r} description must be text")
    parameters = tool.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ValueError(f"Client planner tool {name!r} parameters must be an object")
    return name.strip(), description.strip() or "Client-executed capability.", parameters


def render_tool_block(
    specs: Sequence[ToolSpec],
    client_tools: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Render enabled server and client tool contracts into the Thinker prompt."""
    lines = ["\n\nAvailable tools (only these names are permitted):"]
    if not specs and not client_tools:
        lines.append("- none.")
        return "\n".join(lines)
    names: set[str] = set()
    for spec in specs:
        if spec.name in names:
            raise ValueError(f"Duplicate planner tool name {spec.name!r}")
        names.add(spec.name)
        required = [name for name, param in spec.params.items() if param.required]
        optional = [name for name, param in spec.params.items() if not param.required]
        lines.append(f"- {spec.name} [server-executed]: {spec.contract}")
        lines.append(
            f"  Required params: {', '.join(required) or 'none'}. Optional params: {', '.join(optional) or 'none'}."
        )
    for raw_tool in client_tools:
        name, description, parameters = _client_tool_contract(raw_tool)
        if name in names:
            raise ValueError(f"Duplicate planner tool name {name!r}")
        names.add(name)
        required = parameters.get("required", [])
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise ValueError(f"Client planner tool {name!r} required fields must be a string list")
        properties = parameters.get("properties", {})
        property_names = list(properties) if isinstance(properties, Mapping) else []
        optional = [item for item in property_names if item not in set(required)]
        lines.append(f"- {name} [client-executed]: {description}")
        lines.append(
            f"  Required params: {', '.join(required) or 'none'}. Optional params: {', '.join(optional) or 'none'}."
        )
        lines.append(
            "  JSON Schema: "
            + json.dumps(parameters, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        )
    return "\n".join(lines)
