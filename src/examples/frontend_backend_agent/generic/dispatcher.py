# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Atomic validation and deterministic execution of generic-domain plans."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from examples.frontend_backend_agent.generic.client_tools import (
    ClientToolRoundExecutor,
    ClientToolSpec,
    client_call_fingerprint,
    format_client_result,
    validate_client_arguments,
)
from examples.frontend_backend_agent.generic.result_formatters import (
    combine_tool_results,
    disabled_tool,
    format_tool_result,
    invalid_parameters,
    missing_parameters,
    unsupported_request,
)
from examples.frontend_backend_agent.src.tools import ToolContext, ToolSpec, validate_arguments

if TYPE_CHECKING:
    from examples.frontend_backend_agent.src.stage_metrics import StageMetricsCoordinator

MAX_PARALLEL_TOOL_CALLS = 3
_WORD_RE = re.compile(r"[a-z0-9]+")
_CORPORATE_DECORATION_WORDS = frozenset(
    {"company", "corp", "corporation", "inc", "incorporated", "limited", "ltd", "plc", "the"}
)
_SOURCE_GROUNDED_PARAMS: dict[str, tuple[str, ...]] = {"get_stock_price": ("company_name",)}


@dataclass(slots=True, frozen=True)
class ValidatedToolCall:
    """One allowlisted call whose structure is safe to execute."""

    name: str
    arguments: dict[str, Any]


class PlanValidationError(ValueError):
    """A rejected model plan; no tool may execute after this exception."""


def _distinctive_words(value: object) -> set[str]:
    """Return literal subject words without optional corporate decorations."""
    return set(_WORD_RE.findall(str(value or "").casefold())) - _CORPORATE_DECORATION_WORDS


def _source_grounding_missing(call: ValidatedToolCall, source_query: str) -> list[str]:
    """Reject planner-authored subjects that are absent from the delegated request."""
    required = _SOURCE_GROUNDED_PARAMS.get(call.name, ())
    query_words = set(_WORD_RE.findall(source_query.casefold()))
    missing: list[str] = []
    for name in required:
        argument_words = _distinctive_words(call.arguments.get(name))
        if not argument_words or query_words.isdisjoint(argument_words):
            missing.append(name)
    return missing


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
    client_tools: Mapping[str, ClientToolSpec] | None = None,
) -> list[ValidatedToolCall]:
    """Validate the whole plan before allowing any external side effect."""
    calls: list[ValidatedToolCall] = []
    client_tools = client_tools or {}
    for raw in _raw_calls(plan):
        name = str(raw.get("tool") or "").strip()
        if name not in tools and name not in client_tools:
            raise PlanValidationError(f"unknown tool: {name}")
        if name not in enabled_tools:
            raise PlanValidationError(f"disabled tool: {name}")
        arguments = raw.get("params")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise PlanValidationError(f"invalid params for {name}")
        if name in tools and set(arguments) - set(tools[name].params):
            raise PlanValidationError(f"unexpected params for {name}")
        calls.append(ValidatedToolCall(name=name, arguments=dict(arguments)))
    return calls


async def _execute(
    call: ValidatedToolCall,
    spec: ToolSpec,
    tool_context: ToolContext,
    on_tool_started: Callable[[str], Awaitable[None]] | None,
    stage_metrics: StageMetricsCoordinator | None,
    backend_call_id: str,
    ordinal: int,
) -> dict[str, Any]:
    span = (
        await stage_metrics.start_tool(backend_call_id, tool_name=call.name, ordinal=ordinal)
        if stage_metrics is not None
        else None
    )
    outcome = "success"
    try:
        if on_tool_started and stage_metrics is None:
            await on_tool_started(call.name)
        data = await asyncio.wait_for(spec.run(call.arguments, tool_context), timeout=spec.timeout_s)
        if str(data.get("status") or "success") not in {"success", "not_found"}:
            outcome = "error"
        return format_tool_result(spec, call.arguments, data)
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except TimeoutError:
        outcome = "timeout"
        logger.warning(f"generic domain tool {call.name} timed out")
        return format_tool_result(
            spec,
            call.arguments,
            {"status": "unavailable", "assistant_should_say": "That check timed out. Would you like me to retry?"},
        )
    except (TypeError, ValueError):
        outcome = "error"
        return invalid_parameters(call.name)
    except Exception as exc:  # noqa: BLE001 - fail closed at the tool boundary
        outcome = "error"
        logger.warning(f"generic domain tool {call.name} failed: {type(exc).__name__}")
        return format_tool_result(
            spec,
            call.arguments,
            {"status": "unavailable", "assistant_should_say": "I couldn't complete that check right now."},
        )
    finally:
        if stage_metrics is not None and span is not None:
            await stage_metrics.finish_tool(span, outcome)


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
    source_query: str | None = None,
    tool_context: ToolContext | None = None,
    on_tool_started: Callable[[str], Awaitable[None]] | None = None,
    stage_metrics: StageMetricsCoordinator | None = None,
    backend_call_id: str = "unbound",
    accumulated_results: list[dict[str, Any]] | None = None,
    tool_ordinal_offset: int = 0,
    client_tools: Mapping[str, ClientToolSpec] | None = None,
    client_tool_executor: ClientToolRoundExecutor | None = None,
    client_tool_timeout_seconds: float = 25.0,
    seen_client_calls: set[str] | None = None,
) -> dict[str, Any]:
    """Validate atomically, serialize mutating tools, and preserve planner order."""
    enabled = frozenset(enabled_tools)
    client_tools = client_tools or {}
    enabled_specs = tuple(tools[name] for name in enabled_tools if name in tools)
    if plan.get("tool") == "response_hint" and not plan.get("tool_calls"):
        return _response_hint(plan, tools, enabled_tools)
    try:
        calls = validate_plan(plan, tools, enabled, client_tools)
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
    pending_client_fingerprints: list[str] = []
    for call in calls:
        if call.name in client_tools:
            validation_error = validate_client_arguments(client_tools[call.name], call.arguments)
            if validation_error is not None:
                logger.warning(f"client-owned tool arguments rejected: tool={call.name}")
                return invalid_parameters(call.name)
            fingerprint = client_call_fingerprint(call.name, call.arguments)
            if (seen_client_calls is not None and fingerprint in seen_client_calls) or (
                fingerprint in pending_client_fingerprints
            ):
                if seen_client_calls is not None:
                    seen_client_calls.update(pending_client_fingerprints)
                    seen_client_calls.add(fingerprint)
                logger.warning(f"duplicate client-owned tool call suppressed: tool={call.name}")
                return format_client_result(
                    call.name,
                    call.arguments,
                    {
                        "ok": False,
                        "error": {
                            "code": "duplicate_client_tool_call",
                            "message": "I stopped a repeated tool request that had not produced a successful result.",
                        },
                    },
                )
            pending_client_fingerprints.append(fingerprint)
            continue
        spec = tools[call.name]
        if source_query is not None:
            ungrounded = _source_grounding_missing(call, source_query)
            if ungrounded:
                logger.warning(
                    "generic domain rejected planner-authored subject absent from source query: "
                    f"tool={call.name} params={','.join(ungrounded)}"
                )
                return missing_parameters(spec, ungrounded)
        try:
            missing = validate_arguments(spec, call.arguments)
        except (TypeError, ValueError):
            return invalid_parameters(call.name)
        if missing:
            return missing_parameters(spec, missing)

    context = tool_context or ToolContext()
    payloads: list[dict[str, Any] | None] = [None] * len(calls)

    async def run_one(index: int, call: ValidatedToolCall) -> None:
        payloads[index] = await _execute(
            call,
            tools[call.name],
            context,
            on_tool_started,
            stage_metrics,
            backend_call_id,
            tool_ordinal_offset + index,
        )

    async def run_client_batch(items: list[tuple[int, ValidatedToolCall]]) -> None:
        spans: list[Any] = []
        try:
            for index, call in items:
                if on_tool_started is not None and stage_metrics is None:
                    await on_tool_started(call.name)
                span = (
                    await stage_metrics.start_tool(
                        backend_call_id,
                        tool_name=call.name,
                        ordinal=tool_ordinal_offset + index,
                    )
                    if stage_metrics is not None
                    else None
                )
                spans.append(span)
            if client_tool_executor is None:
                outputs: list[str | dict[str, Any]] = [
                    {
                        "ok": False,
                        "error": {
                            "code": "client_tool_runtime_unavailable",
                            "message": "That client capability is unavailable for this session.",
                        },
                    }
                    for _item in items
                ]
            else:
                outputs = await client_tool_executor(
                    tuple((call.name, call.arguments) for _index, call in items),
                    client_tool_timeout_seconds,
                )
                if len(outputs) != len(items):
                    raise RuntimeError("Client tool executor returned the wrong result cardinality")
            for (index, call), output in zip(items, outputs, strict=True):
                payloads[index] = format_client_result(call.name, call.arguments, output)
                fingerprint = client_call_fingerprint(call.name, call.arguments)
                if seen_client_calls is not None:
                    if str(payloads[index].get("status")) == "success":
                        seen_client_calls.discard(fingerprint)
                    else:
                        seen_client_calls.add(fingerprint)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"client-owned tool round failed: {type(exc).__name__}: {exc}")
            for index, call in items:
                payloads[index] = format_client_result(
                    call.name,
                    call.arguments,
                    {
                        "ok": False,
                        "error": {
                            "code": "client_tool_error",
                            "message": "I couldn't complete that client tool request right now.",
                        },
                    },
                )
                if seen_client_calls is not None:
                    seen_client_calls.add(client_call_fingerprint(call.name, call.arguments))
        finally:
            if stage_metrics is not None:
                for span, (index, _call) in zip(spans, items, strict=False):
                    if span is not None:
                        outcome = (
                            "success"
                            if payloads[index] is not None and payloads[index].get("status") == "success"
                            else "error"
                        )
                        await stage_metrics.finish_tool(span, outcome)

    mutating: list[tuple[int, ValidatedToolCall]] = []
    client_items: list[tuple[int, ValidatedToolCall]] = []
    coroutines: list[Awaitable[None]] = []
    for index, call in enumerate(calls):
        if call.name in client_tools:
            client_items.append((index, call))
        elif tools[call.name].mutates:
            mutating.append((index, call))
        else:
            coroutines.append(run_one(index, call))
    if mutating:

        async def run_mutating_chain() -> None:
            for index, call in mutating:
                await run_one(index, call)

        coroutines.append(run_mutating_chain())
    if client_items:
        if seen_client_calls is not None:
            seen_client_calls.update(pending_client_fingerprints)
        coroutines.append(run_client_batch(client_items))
    await asyncio.gather(*coroutines)

    resolved = [payload for payload in payloads if payload is not None]
    if accumulated_results is not None:
        accumulated_results.extend(resolved)
    return resolved[0] if len(resolved) == 1 else combine_tool_results(resolved)


def combine_accumulated_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Return one final payload for all completed planning rounds."""
    if not results:
        raise PlanValidationError("planner completed without tool results")
    return results[0] if len(results) == 1 else combine_tool_results(results)
