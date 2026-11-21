# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The IM's work tools: the tools a task gets from the IM, not from its backend."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import Field, model_validator

from examples.frontend_backend_live.delegation_core.backend.port import ToolNotExecuted
from examples.shared.strict_settings import StrictSettings, Text

if TYPE_CHECKING:
    from examples.frontend_backend_live.delegation_core.config import Texts


@dataclass(frozen=True)
class Invocation:
    """One canonical work-tool invocation; never a mutable task-wide last-call variable."""

    call_id: str
    name: str
    arguments: dict


class CallerMessage(StrictSettings):
    """A worker publication; an existing ID explicitly reissues an unpublished question."""

    kind: Literal["question", "status"]
    message: Text
    question_id: Text | None = None

    @model_validator(mode="after")
    def question_only(self):
        """Keep reissue identity exclusive to question publications."""
        if self.kind != "question" and self.question_id is not None:
            raise ValueError("Only a question can name question_id")
        return self


class QuestionResolution(StrictSettings):
    """An explicit semantic disposition, distinct from publication or business permission."""

    question_id: Text
    disposition: Literal["answered", "no_longer_needed"]
    input_ref: Text | None = None

    @model_validator(mode="after")
    def evidence(self):
        """Require actual answer evidence and distinguish withdrawal from an answer."""
        if self.disposition == "answered" and self.input_ref is None:
            raise ValueError("An answered question requires input_ref")
        if self.disposition == "no_longer_needed" and self.input_ref is not None:
            raise ValueError("A withdrawal does not consume an answer input")
        return self


class QuestionResolutions(StrictSettings):
    """The entire selected batch is checked before any question changes."""

    resolutions: tuple[QuestionResolution, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique(self):
        """Reject duplicate question selections before any resolution changes."""
        ids = [row.question_id for row in self.resolutions]
        if len(ids) != len(set(ids)):
            raise ValueError("Each question may appear only once")
        return self


class WorkStateChange(StrictSettings):
    """A worker's semantic choice of its own retained execution state."""

    state: Literal["paused", "running"]


class WorkTools:
    """One task's idempotent Core-owned tools (``port.ToolSource``)."""

    def __init__(
        self,
        names: tuple[str, ...],
        texts: Texts,
        cancel: Callable[[], str],
        publish: Callable[[Invocation], str],
        carry_on: Callable[[str], str],
        label: str,
        *,
        resolve: Callable[[Invocation], str],
        set_state: Callable[[Invocation], str],
    ):
        """Bind this session's existing records, tools, and execution callbacks."""
        self.names, self.texts = names, texts
        self.cancel, self.publish, self.carry_on, self.resolve = cancel, publish, carry_on, resolve
        self.set_state = set_state
        self.label, self.schemas = label, texts.work_tools
        self._bindings: dict[str, str] = {}
        self._receipts: dict[str, str] = {}
        self._failures: dict[str, BaseException] = {}

    def tools(self) -> list[dict]:
        """Return the configured vocabulary as Responses function tools."""
        return [{"type": "function", "name": name, **self.schemas[name]} for name in self.names]

    async def execute(self, call_id: str, name: str, arguments: str) -> str:
        """Reserve immutable identity before validation; replay an exact result or its original failure."""
        parse_error = None
        try:
            given = json.loads(arguments or "{}")
        except (ValueError, TypeError):
            given = None
            parse_error = ToolNotExecuted(f"{name}: the arguments are not JSON")
        signature = json.dumps(
            [name, {"raw": arguments} if parse_error is not None else {"json": given}], sort_keys=True
        )
        previous = self._bindings.get(call_id)
        if previous is not None:
            if previous != signature:
                raise ValueError("A work-tool call ID cannot be reused with different arguments")
            if call_id in self._receipts:
                return self._receipts[call_id]
            if call_id in self._failures:
                raise self._failures[call_id]
            raise RuntimeError("The work-tool operation is already executing")
        self._bindings[call_id] = signature
        try:
            if parse_error is not None and name in self.names:
                raise parse_error
            result = self._execute_bound(Invocation(call_id, name, given))
        except BaseException as error:
            self._failures[call_id] = error
            raise
        self._receipts[call_id] = result
        return result

    def _execute_bound(self, invocation: Invocation) -> str:
        """Execute one reserved invocation without changing its identity or validation-failure semantics."""
        self._validate_bound(invocation)
        name, given = invocation.name, invocation.arguments
        if name == "caller_message":
            return self.publish(invocation)
        if name == "resolve_questions":
            return self.resolve(invocation)
        if name == "set_work_state":
            return self.set_state(invocation)
        if name == "cancel_work":
            return self.cancel()
        return given["display"] if name == "show" else self.carry_on(given["message"])

    def _validate_bound(self, invocation: Invocation) -> None:
        """Only this before-callback stage can certify that an operation was not executed."""
        name, given = invocation.name, invocation.arguments
        if name not in self.names:
            offered = ", ".join(self.names) or self.texts.none
            raise ToolNotExecuted(self.texts.unknown_tool_output.format(name=name, offered=offered))
        if not isinstance(given, dict):
            raise ToolNotExecuted(f"{name}: the arguments must be a JSON object")
        try:
            if name == "caller_message":
                CallerMessage.model_validate(given)
            elif name == "resolve_questions":
                QuestionResolutions.model_validate(given)
            elif name == "set_work_state":
                WorkStateChange.model_validate(given)
            elif name == "cancel_work":
                if given:
                    raise ValueError("cancel_work takes no arguments")
            elif name in ("show", "continue_after_call"):
                key = "display" if name == "show" else "message"
                value = given.get(key)
                if set(given) != {key} or not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name}: {key} must be non-empty text")
            else:
                raise ValueError(f"Unsupported work tool {name}")
        except (ValueError, TypeError) as error:
            raise ToolNotExecuted(str(error)) from error
