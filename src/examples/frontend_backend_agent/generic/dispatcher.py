# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Atomic validation and deterministic execution of generic-domain plans."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from loguru import logger

from examples.frontend_backend_agent.generic.result_formatters import (
    combine_tool_results,
    disabled_tool,
    format_tool_result,
    invalid_parameters,
    missing_parameters,
    unsupported_request,
)
from examples.frontend_backend_agent.src.tools import ToolContext, ToolSpec, validate_arguments

MAX_PARALLEL_TOOL_CALLS = 3


@dataclass(slots=True, frozen=True)
class ValidatedToolCall:
    """One allowlisted call whose structure is safe to execute."""

    name: str
    arguments: dict[str, Any]


class PlanValidationError(ValueError):
    """A rejected model plan; no tool may execute after this exception."""


def _raw_calls(plan: dict[str, Any]) -> list[dict[str, Any]]:
    raw = plan.get("tool_calls")
    if raw is None and plan.get("tool"):
        raw = [plan]
    if not isinstance(raw, list):
        return []
    if len(raw) > MAX_PARALLEL_TOOL_CALLS:
        raise PlanValidationError("too many tool calls")
    if any(not isinstance(item, dict) for item in raw):
        raise PlanValidationError("tool calls must be objects")
    return [dict(item) for item in raw]


def validate_plan(
    plan: dict[str, Any],
    tools: Mapping[str, ToolSpec],
    enabled_tools: frozenset[str],
) -> list[ValidatedToolCall]:
    """Validate the whole plan before allowing any external side effect."""
    calls: list[ValidatedToolCall] = []
    for raw in _raw_calls(plan):
        name = str(raw.get("tool") or "").strip()
        if name not in tools:
            raise PlanValidationError(f"unknown tool: {name}")
        if name not in enabled_tools:
            raise PlanValidationError(f"disabled tool: {name}")
        arguments = raw.get("params")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise PlanValidationError(f"invalid params for {name}")
        if set(arguments) - set(tools[name].params):
            raise PlanValidationError(f"unexpected params for {name}")
        calls.append(ValidatedToolCall(name=name, arguments=dict(arguments)))
    return calls


async def _execute(
    call: ValidatedToolCall,
    spec: ToolSpec,
    tool_context: ToolContext,
    on_tool_started: Callable[[str], Awaitable[None]] | None,
) -> dict[str, Any]:
    try:
        if on_tool_started:
            await on_tool_started(call.name)
        data = await asyncio.wait_for(spec.run(call.arguments, tool_context), timeout=spec.timeout_s)
        return format_tool_result(spec, call.arguments, data)
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        logger.warning(f"generic domain tool {call.name} timed out")
        return format_tool_result(
            spec,
            call.arguments,
            {"status": "unavailable", "assistant_should_say": "That check timed out. Would you like me to retry?"},
        )
    except (TypeError, ValueError):
        return invalid_parameters(call.name)
    except Exception as exc:  # noqa: BLE001 - fail closed at the tool boundary
        logger.warning(f"generic domain tool {call.name} failed: {type(exc).__name__}")
        return format_tool_result(
            spec,
            call.arguments,
            {"status": "unavailable", "assistant_should_say": "I couldn't complete that check right now."},
        )


def _response_hint(
    plan: dict[str, Any],
    tools: Mapping[str, ToolSpec],
    enabled_tools: tuple[str, ...],
) -> dict[str, Any]:
    """Convert only the closed response-hint vocabulary into deterministic speech."""
    enabled = frozenset(enabled_tools)
    reason = str(plan.get("reason") or "")
    context = str(plan.get("context") or "")
    if reason == "params_missing":
        requested = plan.get("params_needed")
        spec = tools.get(context)
        if spec is None or not isinstance(requested, list) or not requested:
            raise PlanValidationError("invalid missing-parameter hint")
        names = list(dict.fromkeys(str(item) for item in requested))
        required = {name for name, param in spec.params.items() if param.required}
        if len(names) > 4 or any(name not in required for name in names):
            raise PlanValidationError("invalid missing-parameter fields")
        return missing_parameters(spec, names)
    if reason == "tool_disabled":
        if context not in tools or context in enabled:
            raise PlanValidationError("invalid disabled-tool hint")
        return disabled_tool(context)
    if reason == "unsupported_request" and context in {"", "general"}:
        return unsupported_request(tuple(tools[name] for name in enabled_tools if name in tools))
    raise PlanValidationError("unknown response hint")


async def dispatch_plan(
    plan: dict[str, Any],
    tools: Mapping[str, ToolSpec],
    enabled_tools: tuple[str, ...],
    *,
    tool_context: ToolContext | None = None,
    on_tool_started: Callable[[str], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Validate atomically, serialize mutating tools, and preserve planner order."""
    enabled = frozenset(enabled_tools)
    enabled_specs = tuple(tools[name] for name in enabled_tools if name in tools)
    if plan.get("tool") == "response_hint" and not plan.get("tool_calls"):
        return _response_hint(plan, tools, enabled_tools)
    try:
        calls = validate_plan(plan, tools, enabled)
    except PlanValidationError as exc:
        message = str(exc)
        logger.warning(f"generic domain plan rejected: {message}")
        if message.startswith("disabled tool:"):
            return disabled_tool(message.split(":", 1)[1].strip())
        raise
    if not calls:
        return unsupported_request(enabled_specs)
    # Preflight every call before the first side effect. A malformed member of
    # a multi-tool plan prevents all other members from running.
    for call in calls:
        spec = tools[call.name]
        try:
            missing = validate_arguments(spec, call.arguments)
        except (TypeError, ValueError):
            return invalid_parameters(call.name)
        if missing:
            return missing_parameters(spec, missing)

    context = tool_context or ToolContext()
    payloads: list[dict[str, Any] | None] = [None] * len(calls)

    async def run_one(index: int, call: ValidatedToolCall) -> None:
        payloads[index] = await _execute(call, tools[call.name], context, on_tool_started)

    async def run_mutating_chain(items: list[tuple[int, ValidatedToolCall]]) -> None:
        for index, call in items:
            await run_one(index, call)

    mutating: list[tuple[int, ValidatedToolCall]] = []
    coroutines: list[Awaitable[None]] = []
    for index, call in enumerate(calls):
        if tools[call.name].mutates:
            mutating.append((index, call))
        else:
            coroutines.append(run_one(index, call))
    if mutating:
        coroutines.append(run_mutating_chain(mutating))
    await asyncio.gather(*coroutines)

    resolved = [payload for payload in payloads if payload is not None]
    return resolved[0] if len(resolved) == 1 else combine_tool_results(resolved)
