# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External tool specs, deterministic result serialization, and internal execution.

Tools are always the *backend's*: the frontend only ever sees the delegation
tool. A tool spec carries a callable only when this process executes it
(``execution: internal``); under ``execution: external`` the caller — tau2's
environment, for instance — runs it and hands results back.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from examples.frontend_backend_verdict.text.errors import ToolResultSerializationError
from examples.frontend_backend_verdict.text.messages import ToolCall, ToolResult, canonical_json

_JSON_SCALARS = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One tool offered to the backend agent.

    ``callable`` is required only for ``execution: internal``. When one agent
    serves concurrent sessions, injected callables must be reentrant — this
    package cannot enforce that and does not try.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    callable: Callable[..., Any] | None = None

    def to_openai_schema(self) -> dict[str, Any]:
        """Return the OpenAI function-tool schema for this spec."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


def serialize_result(value: Any, *, max_result_chars: int = 0) -> str:
    """Serialize a tool result deterministically.

    ``default=str`` is deliberately not used: an arbitrary ``__repr__`` can
    carry a memory address, so two identical runs would produce different
    transcripts. Unsupported types raise instead of being stringified.

    Truncation never cuts serialized JSON in place; the whole result is replaced
    by a valid envelope so the model is never handed malformed content.
    """
    text = _serialize(value)
    if max_result_chars and len(text) > max_result_chars:
        return canonical_json({"truncated": True, "original_chars": len(text), "content": text[:max_result_chars]})
    return text


def _serialize(value: Any) -> str:
    if isinstance(value, str):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, _JSON_SCALARS | dict | list):
        try:
            return canonical_json(value)
        except (TypeError, ValueError) as exc:
            raise ToolResultSerializationError(f"tool result is not JSON-serializable: {exc}") from exc
    raise ToolResultSerializationError(f"tool result type {type(value).__name__!r} is not JSON-serializable")


def error_payload(exc: BaseException) -> str:
    """Return the deterministic error envelope for a failed tool call."""
    return canonical_json({"error": {"type": type(exc).__name__, "message": str(exc)}})


class ToolRegistry:
    """Name-indexed tool specs plus in-process execution."""

    def __init__(self, tools: Sequence[ToolSpec] = (), *, max_result_chars: int = 0) -> None:
        """Index ``tools`` by name, rejecting duplicates."""
        self._tools: dict[str, ToolSpec] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool
        self._max_result_chars = max_result_chars

    def __bool__(self) -> bool:
        """Whether any tool is registered."""
        return bool(self._tools)

    def names(self) -> list[str]:
        """Registered tool names, in declaration order."""
        return list(self._tools)

    def get(self, name: str) -> ToolSpec | None:
        """Look up a spec by name."""
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI tool schemas for every registered tool."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    async def execute(self, call: ToolCall) -> ToolResult:
        """Run one tool call in-process, converting failures to error results."""
        spec = self._tools.get(call.name)
        if spec is None or spec.callable is None:
            return ToolResult(
                tool_call_id=call.id,
                content=error_payload(KeyError(f"unknown or non-executable tool: {call.name}")),
                is_error=True,
            )
        try:
            outcome = spec.callable(**call.arguments)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            return ToolResult(
                tool_call_id=call.id,
                content=serialize_result(outcome, max_result_chars=self._max_result_chars),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any tool failure becomes a model-visible result
            return ToolResult(tool_call_id=call.id, content=error_payload(exc), is_error=True)


class InternalToolDriver:
    """Executes a backend tool batch in this process.

    Sequential by default: injected tools are arbitrary caller code that may
    mutate shared state, and a generic scaffold cannot know which of them
    commute. ``parallel_execution`` opts into concurrency; result order matches
    emission order under both settings, so the model sees identical input.
    """

    def __init__(self, registry: ToolRegistry, *, parallel: bool = False) -> None:
        """Bind the driver to a registry and an execution policy."""
        self._registry = registry
        self._parallel = parallel

    async def run_batch(self, calls: Sequence[ToolCall]) -> tuple[ToolResult, ...]:
        """Execute ``calls`` and return results in emission order."""
        if not calls:
            return ()
        if self._parallel and len(calls) > 1:
            return tuple(await asyncio.gather(*(self._registry.execute(call) for call in calls)))
        results: list[ToolResult] = []
        for call in calls:
            results.append(await self._registry.execute(call))
        return tuple(results)


def decode_arguments(call: ToolCall) -> dict[str, Any]:
    """Return a tool call's arguments, tolerating a JSON-string payload."""
    arguments = call.arguments
    nested = arguments.get("original_args")
    if isinstance(nested, str) and len(arguments) == 1:
        try:
            decoded = json.loads(nested)
        except json.JSONDecodeError:
            return arguments
        if isinstance(decoded, dict):
            return decoded
    return arguments
