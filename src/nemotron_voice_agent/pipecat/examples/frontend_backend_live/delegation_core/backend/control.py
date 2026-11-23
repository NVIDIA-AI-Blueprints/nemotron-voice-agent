# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The conductor's complete action envelope and its prompt-catalog vocabulary."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StrictInt, ValidationError, model_validator

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    CallerLine,
    CallerMessage,
    Cancel,
    CompleteDecision,
    DecisionAction,
    DecisionRejected,
    Pause,
    Resume,
    Start,
    Steer,
    TaskSpec,
    TaskUpdate,
    WorkTarget,
)
from nemotron_voice_agent.pipecat.shared.strict_settings import Text


def _unique(names: tuple[str, ...]) -> tuple[str, ...]:
    if len(names) != len(set(names)):
        raise ValueError("identities must be unique")
    return names


Names = Annotated[tuple[Text, ...], AfterValidator(_unique)]
SomeNames = Annotated[Names, Field(min_length=1)]


class Arguments(BaseModel):
    """Strict arguments shared by the action envelope and its members."""

    model_config = ConfigDict(extra="forbid")


class CallerLineArguments(Arguments):
    """One selection from the presented caller lines."""

    source_message_id: Text
    sequence: StrictInt = Field(ge=0)
    excerpt: Text | None = None


class ScopedArguments(Arguments):
    """Caller evidence shared by task assignments and updates."""

    caller_lines: tuple[CallerLineArguments, ...] = ()

    def _selections(self) -> tuple[CallerLine, ...]:
        return tuple(CallerLine(line.source_message_id, line.sequence, line.excerpt) for line in self.caller_lines)


class TaskArguments(ScopedArguments):
    """One task assignment and its optional exact predecessor task."""

    brief: Text
    continues: Text | None = None
    verifies: Text | None = None

    @model_validator(mode="after")
    def one_thread_use(self):
        """Keep continuation and read-only verification mutually exclusive."""
        if self.continues is not None and self.verifies is not None:
            raise ValueError("Verification reads an exact prior task without resuming its goal")
        return self


class TargetArguments(Arguments):
    """One Work-wide target or an exact nonempty set of its tasks."""

    work_id: Text
    task_ids: SomeNames | None = None


class SourceArguments(Arguments):
    """The caller sources and routing questions selected by one action."""

    delegations: SomeNames
    clarifications: Names = ()


class StartWork(SourceArguments):
    """An independent task assignment."""

    action: Literal["start"]
    tasks: tuple[TaskArguments, ...] = Field(min_length=1)


class TargetedAction(SourceArguments):
    """An atomic action over exact Work/task target sets."""

    targets: tuple[TargetArguments, ...] = Field(min_length=1)


class UpdateArguments(ScopedArguments):
    """Scoped update for one exact target in a steer action."""

    task_id: Text
    text: Text


class SteerWork(TargetedAction):
    """An update to existing live or ended task threads."""

    action: Literal["steer"]
    updates: tuple[UpdateArguments, ...] = ()


class StopWork(SourceArguments):
    """An immediate stop request, without a business reversal."""

    action: Literal["stop"]
    targets: tuple[TargetArguments, ...] = Field(default=(), min_length=1)
    scope: Literal["all"] | None = None

    @model_validator(mode="after")
    def select_work_to_stop(self) -> StopWork:
        """Select exact targets or let Core resolve all current work."""
        if bool(self.targets) == (self.scope == "all"):
            raise ValueError("Stop requires exactly one of targets or scope: all")
        if self.scope == "all" and len(self.delegations) != 1:
            raise ValueError("For scope: all, use only the caller's stop request in delegations")
        return self


class PauseWork(TargetedAction):
    """Retain a task while fencing further business effects."""

    action: Literal["pause"]


class ResumeWork(TargetedAction):
    """Continue a retained task through ordinary inference admission."""

    action: Literal["resume"]


class PublishQuestion(SourceArguments):
    """Ask the caller which work to route or control."""

    action: Literal["caller_message"]
    kind: Literal["question"]
    message: Text


class PublishStatus(TargetedAction):
    """Report Core's operational state for selected jobs."""

    action: Literal["caller_message"]
    kind: Literal["status"]


PublishMessage = Annotated[PublishQuestion | PublishStatus, Field(discriminator="kind")]


ActionArguments = Annotated[
    StartWork | SteerWork | StopWork | PauseWork | ResumeWork | PublishMessage,
    Field(discriminator="action"),
]


class SubmitDecision(Arguments):
    """Every declared action for one placement decision."""

    actions: tuple[ActionArguments, ...] = Field(min_length=1)


def _action(
    given: StartWork | SteerWork | StopWork | PauseWork | ResumeWork | PublishQuestion | PublishStatus,
) -> DecisionAction:
    values = {"delegations": given.delegations, "clarifications": given.clarifications}
    if isinstance(given, StartWork):
        return Start(
            **values,
            tasks=tuple(
                TaskSpec(task.brief, task.continues, task.verifies, task._selections()) for task in given.tasks
            ),
        )
    if isinstance(given, PublishQuestion):
        return CallerMessage(**values, kind=given.kind, message=given.message)
    targets = tuple(WorkTarget(target.work_id, target.task_ids) for target in given.targets)
    if isinstance(given, PublishStatus):
        return CallerMessage(**values, kind=given.kind, targets=targets)
    if isinstance(given, StopWork):
        return Cancel(**values, targets=targets, scope=given.scope)
    if isinstance(given, SteerWork):
        return Steer(
            **values,
            targets=targets,
            updates=tuple(TaskUpdate(update.task_id, update.text, update._selections()) for update in given.updates),
        )
    action = {PauseWork: Pause, ResumeWork: Resume}[type(given)]
    return action(**values, targets=targets)


def _argument_problem(problem: Mapping[str, Any]) -> str:
    """Describe an invalid argument using its wire types and location."""
    path = ".".join(map(str, problem["loc"])) or "arguments"
    kind = problem["type"]
    expected_types = {
        "tuple_type": "a JSON array",
        "model_attributes_type": "a JSON object",
        "model_type": "a JSON object",
        "string_type": "a string",
        "int_type": "an integer",
    }
    if expected := expected_types.get(kind):
        wire_types = {
            list: "an array",
            dict: "an object",
            str: "a string",
            bool: "a boolean",
            int: "an integer",
            float: "a number",
            type(None): "null",
        }
        actual = wire_types.get(type(problem["input"]), "a different type")
        return f"{path} must be {expected}. Received {actual}."
    context = problem.get("ctx", {})
    if kind == "union_tag_invalid":
        return f"{path} requires {context['discriminator']} to be one of {context['expected_tags']}."
    if kind == "too_short":
        return f"{path} must contain {context['min_length']} or more valid entries."
    return f"{path}: {problem['msg'].removeprefix('Value error, ')}"


def parse_decision(arguments: dict) -> CompleteDecision:
    """Validate a complete decision from the selected backend conductor."""
    try:
        checked = SubmitDecision.model_validate(arguments)
    except ValidationError as exc:
        details = "\n".join(_argument_problem(problem) for problem in exc.errors(include_url=False))
        raise DecisionRejected(f"submit_decision: {details}") from None
    return CompleteDecision(tuple(_action(action) for action in checked.actions), model_authored=True)


class ControlVocabulary:
    """The complete-decision tool and instructions from the Core prompt catalog."""

    def __init__(self, schemas: Mapping[str, Mapping[str, Any]], prompt: str, idle: str, failed: str):
        """Validate the catalog's envelope fields and retain its model-facing wording."""
        self.prompt, self.idle, self.failed = prompt, idle, failed
        if set(schemas) != {"submit_decision"}:
            raise ValueError("texts.control_tools must hold exactly submit_decision")
        schema = schemas["submit_decision"]
        properties = (schema.get("parameters") or {}).get("properties") or {}
        if set(properties) != set(SubmitDecision.model_fields):
            raise ValueError("texts.control_tools.submit_decision must define the complete actions envelope")
        self.schema = deepcopy(dict(schema))

    def instructions(self, max_running: int) -> str:
        """Render the conductor's task-capacity instructions."""
        return self.prompt.format(max_running=max_running)

    def conductor_tools(self) -> list[dict]:
        """Return one complete-decision Responses function tool."""
        return [{"type": "function", "name": "submit_decision", **deepcopy(self.schema)}]

    def decision(self, name: str, arguments: str) -> CompleteDecision:
        """Validate every action before constructing an executable Core decision."""
        if name != "submit_decision":
            raise DecisionRejected(f"Unknown control tool {name!r}")
        try:
            given = json.loads(arguments or "{}")
        except ValueError:
            raise DecisionRejected(f"{name}: the arguments are not JSON") from None
        if not isinstance(given, dict):
            raise DecisionRejected(f"{name}: the arguments must be a JSON object")
        return parse_decision(given)
