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
    client_parameter_labels,
    format_client_result,
    validate_client_arguments,
)
from examples.frontend_backend_agent.generic.result_formatters import (
    combine_tool_results,
    confirmation_request,
    disabled_tool,
    format_tool_result,
    invalid_parameters,
    missing_client_parameters,
    missing_parameters,
    unspecified_clarification,
    unsupported_request,
)
from examples.frontend_backend_agent.src.normalization import tool_kind
from examples.frontend_backend_agent.src.tools import ToolContext, ToolSpec, validate_arguments

if TYPE_CHECKING:
    from examples.frontend_backend_agent.generic.argument_screen import ArgumentScreen
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


def _enabled_client_parameter_labels(
    client_tools: Mapping[str, ClientToolSpec],
    enabled: frozenset[str],
) -> dict[str, str]:
    """Return every parameter this session's enabled client tools declare."""
    labels: dict[str, str] = {}
    for name, spec in client_tools.items():
        if name in enabled:
            labels.update(client_parameter_labels(spec))
    return labels


def _confirmation_arguments(
    plan: dict[str, Any],
    tools: Mapping[str, ToolSpec],
    client_tools: Mapping[str, ClientToolSpec],
    context: str,
) -> dict[str, Any]:
    """Validate a proposed action exactly as if it were about to execute.

    The confirmation names a real call, so it is held to the same argument
    contract as one. Nothing here is spoken; the caller-facing sentence is
    rendered from the tool's own name.
    """
    arguments = plan.get("params")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise PlanValidationError("invalid confirmation arguments")
    if context in client_tools:
        if validate_client_arguments(client_tools[context], arguments) is not None:
            raise PlanValidationError("invalid confirmation arguments")
        return dict(arguments)
    spec = tools[context]
    if set(arguments) - set(spec.params):
        raise PlanValidationError("invalid confirmation arguments")
    try:
        validate_arguments(spec, arguments)
    except (TypeError, ValueError) as exc:
        raise PlanValidationError("invalid confirmation arguments") from exc
    return dict(arguments)


def _response_hint(
    plan: dict[str, Any],
    tools: Mapping[str, ToolSpec],
    enabled_tools: tuple[str, ...],
    client_tools: Mapping[str, ClientToolSpec] | None = None,
) -> dict[str, Any]:
    """Convert only the closed response-hint vocabulary into deterministic speech.

    A client-owned tool is a first-class member of this vocabulary. The caller
    declares its schema, so its field names are as trustworthy a source of
    spoken words as our own registry. Resolving a hint against the server
    registry alone rejects every clarification a client tool could ever need,
    which silently costs the whole turn.
    """
    enabled = frozenset(enabled_tools)
    client_tools = client_tools or {}
    reason = str(plan.get("reason") or "")
    context = str(plan.get("context") or "")
    if reason == "params_missing":
        requested = plan.get("params_needed")
        if not isinstance(requested, list) or not requested:
            raise PlanValidationError("invalid missing-parameter hint")
        names = list(dict.fromkeys(str(item) for item in requested))
        if len(names) > 4:
            raise PlanValidationError("invalid missing-parameter fields")
        # Dispatch on the registry that owns the context, never as a fallback
        # chain: a server tool whose fields are wrong must keep reporting that,
        # not decay into the weaker "unknown context" message.
        spec = tools.get(context)
        if spec is not None:
            required = {name for name, param in spec.params.items() if param.required}
            if any(name not in required for name in names):
                raise PlanValidationError("invalid missing-parameter fields")
            return missing_parameters(spec, names)
        client_spec = client_tools.get(context)
        if client_spec is not None:
            # A caller's policy can require something before the intended tool
            # may run at all -- an identity to verify, a prior lookup. That
            # field belongs to a different declared tool, so resolving it
            # against the target's schema alone rejects the question and costs
            # the turn. Accept any field this session's enabled tools declare;
            # the spoken words still come from a schema, never from the plan.
            allowed = _enabled_client_parameter_labels(client_tools, enabled)
            allowed.update(client_parameter_labels(client_spec))
            if not allowed:
                return unspecified_clarification(context)
            if any(name not in allowed for name in names):
                raise PlanValidationError("invalid missing-parameter fields")
            return missing_client_parameters(context, [allowed[name] for name in names], names)
        raise PlanValidationError("invalid missing-parameter hint")
    if reason == "confirmation_needed":
        # A caller's policy can require explicit consent before its own tool
        # changes anything. The planner names the call it intends to make and
        # we ask about it; it never authors the question. Scoped to sessions
        # that declare their own tools, so the built-in domains keep the
        # narrower vocabulary they were written against.
        if not client_tools:
            raise PlanValidationError("unknown response hint")
        if context not in enabled or (context not in tools and context not in client_tools):
            raise PlanValidationError("invalid confirmation hint")
        arguments = _confirmation_arguments(plan, tools, client_tools, context)
        client_spec = client_tools.get(context)
        if client_spec is not None:
            return confirmation_request(
                context,
                arguments,
                description=client_spec.description,
                labels=client_parameter_labels(client_spec),
            )
        spec = tools[context]
        return confirmation_request(
            context,
            arguments,
            description=spec.capability or spec.contract,
            labels={name: param.label or name.replace("_", " ") for name, param in spec.params.items()},
        )
    if reason == "tool_disabled":
        if (context not in tools and context not in client_tools) or context in enabled:
            raise PlanValidationError("invalid disabled-tool hint")
        return disabled_tool(context)
    if reason == "unsupported_request" and context in {"", "general"}:
        return unsupported_request(
            tuple(tools[name] for name in enabled_tools if name in tools),
            suppress_capabilities=bool(client_tools),
        )
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
    argument_screen: ArgumentScreen | None = None,
    running_client_calls: dict[str, asyncio.Future[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Validate atomically, serialize mutating tools, and preserve planner order.

    ``argument_screen`` puts a read's dictated identifiers into their schema
    shape and stops a clearly unfinished one with a question for the caller.
    It never touches a write's arguments.

    ``seen_client_calls`` holds calls that already failed; a repeat is
    suppressed. ``running_client_calls`` holds calls another run of the same
    session is still executing; a repeat waits for that result instead of
    being sent twice or reported as a failure. Without the running-call registry
    a call is marked seen before it runs, which is the older behavior.
    """
    enabled = frozenset(enabled_tools)
    client_tools = client_tools or {}
    enabled_specs = tuple(tools[name] for name in enabled_tools if name in tools)
    if plan.get("tool") == "response_hint" and not plan.get("tool_calls"):
        try:
            return _response_hint(plan, tools, enabled_tools, client_tools)
        except PlanValidationError as exc:
            # Keep the dispatcher the single place that reports a rejected plan.
            # This rejection used to surface only as generic planner failure
            # speech, with nothing in the log to say which rule fired.
            logger.warning(f"generic domain response hint rejected: {exc}")
            raise
    try:
        calls = validate_plan(plan, tools, enabled, client_tools)
    except PlanValidationError as exc:
        message = str(exc)
        logger.warning(f"generic domain plan rejected: {message}")
        if message.startswith("disabled tool:"):
            return disabled_tool(message.split(":", 1)[1].strip())
        raise
    if not calls:
        return unsupported_request(enabled_specs, suppress_capabilities=bool(client_tools))
    # Preflight every call before the first side effect. A malformed member of
    # a multi-tool plan prevents all other members from running.
    pending_client_fingerprints: list[str] = []
    suppressed_payloads: dict[int, dict[str, Any]] = {}
    shared_results: dict[int, asyncio.Future[dict[str, Any]]] = {}
    dropped_indexes: set[int] = set()
    for index, call in enumerate(calls):
        if call.name in client_tools:
            if argument_screen is not None:
                screened, question = argument_screen.screen(call.name, call.arguments)
                if question is not None:
                    logger.bind(event="identifier_incomplete", tool=call.name).info(
                        f"unfinished identifier answered locally: tool={call.name}"
                    )
                    return question
                if screened != call.arguments:
                    changed = sorted(key for key in screened if screened.get(key) != call.arguments.get(key))
                    logger.bind(event="identifier_screened", tool=call.name, fields=changed).info(
                        f"spoken identifier put into schema shape: tool={call.name} fields={','.join(changed)}"
                    )
                    call = ValidatedToolCall(name=call.name, arguments=screened)
                    calls[index] = call
            validation_error = validate_client_arguments(client_tools[call.name], call.arguments)
            if validation_error is not None:
                logger.warning(f"client-owned tool arguments rejected: tool={call.name}")
                return invalid_parameters(call.name)
            fingerprint = client_call_fingerprint(call.name, call.arguments)
            if running_client_calls is not None:
                if fingerprint in pending_client_fingerprints:
                    # The same call twice in one plan runs once; the repeat
                    # never reached the client, so it is not a failure.
                    logger.info(f"repeated client-owned call in one plan dropped: tool={call.name}")
                    dropped_indexes.add(index)
                    continue
                running = running_client_calls.get(fingerprint)
                if running is not None and not running.done():
                    logger.info(f"client-owned call already running in this session; sharing it: tool={call.name}")
                    shared_results[index] = running
                    continue
                if seen_client_calls is not None and fingerprint in seen_client_calls:
                    suppressed_payloads[index] = _suppressed_duplicate(call, argument_screen)
                    continue
                pending_client_fingerprints.append(fingerprint)
                continue
            if (seen_client_calls is not None and fingerprint in seen_client_calls) or (
                fingerprint in pending_client_fingerprints
            ):
                if seen_client_calls is not None:
                    seen_client_calls.update(pending_client_fingerprints)
                    seen_client_calls.add(fingerprint)
                return _suppressed_duplicate(call, argument_screen)
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
        owned: dict[int, tuple[str, asyncio.Future[dict[str, Any]]]] = {}
        if running_client_calls is not None:
            loop = asyncio.get_running_loop()
            for index, call in items:
                fingerprint = client_call_fingerprint(call.name, call.arguments)
                future: asyncio.Future[dict[str, Any]] = loop.create_future()
                running_client_calls[fingerprint] = future
                owned[index] = (fingerprint, future)
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
            # Always settle and unregister, so a batch cut short by a deadline
            # or a newer turn never leaves its calls blocked for the session.
            for index, (fingerprint, future) in owned.items():
                if running_client_calls is not None and running_client_calls.get(fingerprint) is future:
                    del running_client_calls[fingerprint]
                if not future.done():
                    if payloads[index] is not None:
                        future.set_result(payloads[index])
                    else:
                        future.cancel()
                if payloads[index] is None and seen_client_calls is not None:
                    # Cut short with its outcome unknown. A read may simply be
                    # asked again; a possible write must not be sent twice.
                    spec = client_tools[calls[index].name]
                    if tool_kind(spec.name, spec.description) != "read":
                        seen_client_calls.add(fingerprint)
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
    for index, payload in suppressed_payloads.items():
        payloads[index] = payload
    for index, call in enumerate(calls):
        if index in dropped_indexes or index in suppressed_payloads:
            continue
        if index in shared_results:
            coroutines.append(_await_shared_result(index, calls[index], shared_results[index], payloads))
        elif call.name in client_tools:
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
        if seen_client_calls is not None and running_client_calls is None:
            seen_client_calls.update(pending_client_fingerprints)
        coroutines.append(run_client_batch(client_items))
    await asyncio.gather(*coroutines)

    resolved = [payload for payload in payloads if payload is not None]
    if accumulated_results is not None:
        accumulated_results.extend(resolved)
    return resolved[0] if len(resolved) == 1 else combine_tool_results(resolved)


def _suppressed_duplicate(call: ValidatedToolCall, argument_screen: ArgumentScreen | None) -> dict[str, Any]:
    """Report a repeat of a client call that already failed, without sending it again."""
    logger.bind(event="retry_guard", tool=call.name).warning(
        f"duplicate client-owned tool call suppressed: tool={call.name}"
    )
    suppressed = format_client_result(
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
    if argument_screen is not None and argument_screen.enabled:
        suppressed["thinker_hint"] = argument_screen.repeated_call_hint()
    return suppressed


async def _await_shared_result(
    index: int,
    call: ValidatedToolCall,
    running: asyncio.Future[dict[str, Any]],
    payloads: list[dict[str, Any] | None],
) -> None:
    """Take the result of the identical call another run is executing."""
    try:
        payloads[index] = dict(await asyncio.shield(running))
    except asyncio.CancelledError:
        if not running.cancelled():
            raise
        # The other run was cut short before its result arrived. Report it as a
        # transient failure; it is not marked seen, so a later round may retry.
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


def combine_accumulated_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Return one final payload for all completed planning rounds."""
    if not results:
        raise PlanValidationError("planner completed without tool results")
    return results[0] if len(results) == 1 else combine_tool_results(results)
