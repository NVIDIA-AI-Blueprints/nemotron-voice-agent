# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The conductor's control tools: their schemas and instructions from the core's texts, and each call's decision.

A model conductor places decisions by calling ``start_work``, ``steer_work``, ``cancel_work`` and ``answer_status``.
Their descriptions come from ``texts.control_tools``; the parameter names are this module's contract with the decisions
in ``backend/port.py``, so a schema that leaves one out is refused when the core is built. The worker's ``cancel_work``
(``texts.work_tools``) is a different tool.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from delegation_core.backend.port import Answer, Cancel, Decision, Start, Steer


def _name(given: dict, key: str) -> str:
    value = given.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_name(given: dict, key: str) -> str | None:
    return None if given.get(key) in (None, "") else _name(given, key)


def _names(given: dict, key: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    value = given.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{key} must be a list of non-empty strings")
    if not value and not allow_empty:
        raise ValueError(f"{key} must name at least one")
    return tuple(value)


def _text(given: dict, key: str) -> str:
    value = given.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be non-empty text")
    return value


@dataclass(frozen=True)
class _Tool:
    """One control tool: the parameters its decision needs, those it may also take, and how it makes the decision."""

    required: tuple[str, ...]
    optional: tuple[str, ...]
    decide: Callable[[dict], Decision]


TOOLS: Mapping[str, _Tool] = {
    "start_work": _Tool(
        ("delegations",),
        ("tasks",),
        lambda given: Start(_names(given, "delegations"), _names(given, "tasks", allow_empty=True)),
    ),
    "steer_work": _Tool(
        ("work_id", "delegations"),
        ("task_id",),
        lambda given: Steer(_name(given, "work_id"), _names(given, "delegations"), _optional_name(given, "task_id")),
    ),
    "cancel_work": _Tool(
        ("work_id",), ("task_id",), lambda given: Cancel(_name(given, "work_id"), _optional_name(given, "task_id"))
    ),
    "answer_status": _Tool(
        ("delegation_id", "text"), (), lambda given: Answer(_name(given, "delegation_id"), _text(given, "text"))
    ),
}


class ControlVocabulary:
    """A conductor's instructions and four control tools as the core's texts word them (``port.ControlTools``)."""

    def __init__(self, schemas: Mapping[str, Mapping[str, Any]], prompt: str, idle: str, failed: str):
        """Check ``schemas`` (``texts.control_tools``); keep ``prompt`` (``texts.conductor_prompt``) and two answers.

        ``idle`` is the outcome of a control call that comes before a ``decide()`` takes its calls: the refusal with
        ``texts.refusal_reasons.idle``. ``failed`` is the answer to a control call that raised (``texts.failed_output``,
        which holds ``{name}`` and ``{error}``).

        Raises:
            ValueError: A tool is missing or unknown, or its schema leaves out a parameter its decision needs or names
                one it does not take; or ``prompt`` is not text.
        """
        if not isinstance(prompt, str):
            raise ValueError("texts.conductor_prompt must be text")
        self.prompt, self.idle, self.failed = prompt, idle, failed
        if set(schemas) != set(TOOLS):
            raise ValueError(f"texts.control_tools must hold exactly {', '.join(TOOLS)}; it holds {sorted(schemas)}")
        for name, schema in schemas.items():
            properties = (schema.get("parameters") or {}).get("properties") or {}
            if not isinstance(schema.get("description"), str) or not schema["description"].strip():
                raise ValueError(f"texts.control_tools.{name} needs a description")
            missing = set(TOOLS[name].required) - set(properties)
            unknown = set(properties) - set(TOOLS[name].required) - set(TOOLS[name].optional)
            if missing or unknown:
                raise ValueError(
                    f"texts.control_tools.{name}: parameters missing {sorted(missing)}, not taken {sorted(unknown)}"
                )
        self.schemas = {name: deepcopy(dict(schemas[name])) for name in TOOLS}

    def instructions(self, max_running: int) -> str:
        """The conductor's instructions: ``texts.conductor_prompt`` with the tasks its backend runs at once."""
        return self.prompt.format(max_running=max_running)

    def conductor_tools(self) -> list[dict]:
        """The four control tools, as Responses function tools, in a fixed order."""
        return [
            {
                "type": "function",
                "name": name,
                "description": schema["description"],
                "parameters": deepcopy(schema["parameters"]),
            }
            for name, schema in self.schemas.items()
        ]

    def decision(self, name: str, arguments: str) -> Decision:
        """The decision one control call makes.

        Raises:
            ValueError: The tool is unknown, the arguments are not a JSON object, or a parameter is missing or of the
                wrong type. A conductor logs such a call and skips it.
        """
        tool = TOOLS.get(name)
        if tool is None:
            raise ValueError(f"Unknown control tool {name!r}")
        try:
            given = json.loads(arguments or "{}")
        except ValueError:
            raise ValueError(f"{name}: the arguments are not JSON") from None
        if not isinstance(given, dict):
            raise ValueError(f"{name}: the arguments must be a JSON object")
        missing = [key for key in tool.required if key not in given]
        if missing:
            raise ValueError(f"{name}: missing {', '.join(missing)}")
        return tool.decide(given)
