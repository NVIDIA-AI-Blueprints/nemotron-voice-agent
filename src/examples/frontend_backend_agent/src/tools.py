# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Declarative internal-tool contracts for Frontend/Backend Agent domains."""

from __future__ import annotations

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


def render_tool_block(specs: Sequence[ToolSpec]) -> str:
    """Render only enabled tool contracts into the hidden Thinker prompt."""
    lines = ["\n\nAvailable internal tools (only these names are permitted):"]
    if not specs:
        lines.append("- none.")
        return "\n".join(lines)
    for spec in specs:
        required = [name for name, param in spec.params.items() if param.required]
        optional = [name for name, param in spec.params.items() if not param.required]
        lines.append(f"- {spec.name}: {spec.contract}")
        lines.append(
            f"  Required params: {', '.join(required) or 'none'}. Optional params: {', '.join(optional) or 'none'}."
        )
    return "\n".join(lines)
