# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The conductor's control tools: their schemas and instructions from the core's texts, and each call's decision."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, ValidationError, WrapValidator

from examples.frontend_backend_live.delegation_core.backend.port import Answer, Cancel, Decision, Start, Steer
from examples.shared.strict_settings import Text


def _worded(problem: str) -> WrapValidator:
    """Return a validator that refuses what the type refuses, with ``problem`` as its error's words."""

    def check(value: Any, handler: Callable[[Any], Any]) -> Any:
        try:
            return handler(value)
        except ValidationError:
            raise ValueError(problem) from None

    return WrapValidator(check)


def _at_least_one(names: tuple[str, ...]) -> tuple[str, ...]:
    if not names:
        raise ValueError("must name at least one")
    return names


Name = Annotated[Text, _worded("must be a non-empty string")]
# A name a call may leave out; null or empty is none as well.
OptionalName = Annotated[Name | None, BeforeValidator(lambda value: None if value == "" else value)]
Names = Annotated[tuple[Text, ...], _worded("must be a list of non-empty strings")]
SomeNames = Annotated[Names, AfterValidator(_at_least_one)]


class Arguments(BaseModel):
    """Base of a control call's arguments, each named as its decision's field."""

    # A conductor may add a key its tool does not take: it is ignored, and the call still makes its decision.
    model_config = ConfigDict(extra="ignore")


class StartWork(Arguments):
    """``start_work``'s arguments."""

    delegations: SomeNames
    tasks: Names = ()


class SteerWork(Arguments):
    """``steer_work``'s arguments."""

    work_id: Name
    delegations: SomeNames
    task_id: OptionalName = None


class CancelWork(Arguments):
    """``cancel_work``'s arguments."""

    work_id: Name
    task_id: OptionalName = None


class AnswerStatus(Arguments):
    """``answer_status``'s arguments."""

    delegation_id: Name
    text: Annotated[Text, _worded("must be non-empty text")]


# Each tool's parameters are its decision's contract; this cancel_work is the conductor's, not the worker's.
TOOLS: Mapping[str, tuple[type[Arguments], Callable[..., Decision]]] = {
    "start_work": (StartWork, Start),
    "steer_work": (SteerWork, Steer),
    "cancel_work": (CancelWork, Cancel),
    "answer_status": (AnswerStatus, Answer),
}


def _problem(name: str, refused: ValidationError) -> str:
    """Return a refused call's error: the parameters it lacks, else its first bad one."""
    problems = refused.errors(include_url=False)
    missing = [str(problem["loc"][0]) for problem in problems if problem["type"] == "missing"]
    if missing:
        return f"{name}: missing {', '.join(missing)}"
    return f"{problems[0]['loc'][0]} {problems[0]['msg'].removeprefix('Value error, ')}"


class ControlVocabulary:
    """A conductor's instructions and four control tools as the core's texts word them (``port.ControlTools``)."""

    def __init__(self, schemas: Mapping[str, Mapping[str, Any]], prompt: str, idle: str, failed: str):
        """Check ``schemas`` (``texts.control_tools``) against the decisions; keep the prompt and the two answers."""
        self.prompt, self.idle, self.failed = prompt, idle, failed
        if set(schemas) != set(TOOLS):
            raise ValueError(f"texts.control_tools must hold exactly {', '.join(TOOLS)}; it holds {sorted(schemas)}")
        for name, schema in schemas.items():
            fields = TOOLS[name][0].model_fields
            properties = (schema.get("parameters") or {}).get("properties") or {}
            missing = {key for key, field in fields.items() if field.is_required()} - set(properties)
            unknown = set(properties) - set(fields)
            if missing or unknown:
                raise ValueError(
                    f"texts.control_tools.{name}: parameters missing {sorted(missing)}, not taken {sorted(unknown)}"
                )
        self.schemas = {name: deepcopy(dict(schemas[name])) for name in TOOLS}

    def instructions(self, max_running: int) -> str:
        """Return the conductor's instructions: ``texts.conductor_prompt`` with the tasks its backend runs at once."""
        return self.prompt.format(max_running=max_running)

    def conductor_tools(self) -> list[dict]:
        """Return the four control tools, as Responses function tools, in a fixed order."""
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
        """Return the decision one control call makes; ``ValueError`` for a malformed call, which a conductor skips."""
        tool = TOOLS.get(name)
        if tool is None:
            raise ValueError(f"Unknown control tool {name!r}")
        try:
            given = json.loads(arguments or "{}")
        except ValueError:
            raise ValueError(f"{name}: the arguments are not JSON") from None
        if not isinstance(given, dict):
            raise ValueError(f"{name}: the arguments must be a JSON object")
        model, decide = tool
        try:
            checked = model.model_validate(given)
        except ValidationError as refused:
            raise ValueError(_problem(name, refused)) from None
        return decide(**checked.model_dump())
