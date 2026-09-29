# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Pipecat function-tool ownership and reliable terminal-result handling."""

from __future__ import annotations

import asyncio
import copy
import math
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import replace
from typing import Any

from loguru import logger
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.frames.frames import FunctionCallResultProperties
from pipecat.services.llm_service import FunctionCallParams

from realtime.tool_schema import compile_tool_arguments_validator, tool_argument_validation_failure

ToolHandler = Callable[[FunctionCallParams], Awaitable[None]]
ToolResultType = type[dict[str, Any]]


def _tool_name(tool: Mapping[str, Any]) -> str | None:
    """Return a function name from canonical Realtime or OpenAI chat shape."""
    direct_name = tool.get("name")
    if isinstance(direct_name, str) and direct_name:
        return direct_name
    function = tool.get("function")
    if not isinstance(function, Mapping):
        return None
    nested_name = function.get("name")
    return nested_name if isinstance(nested_name, str) and nested_name else None


def tool_parameter_schema(tools: ToolsSchema | None, name: str) -> dict[str, Any]:
    """Return the one declared JSON Schema for a named Pipecat function tool."""
    if tools is None:
        raise ValueError(f"Function tool {name!r} has no declared parameters schema")

    matches: list[dict[str, Any]] = []
    for schema in tools.standard_tools:
        if schema.name == name:
            matches.append(
                {
                    "type": "object",
                    "properties": copy.deepcopy(schema.properties),
                    "required": copy.deepcopy(schema.required),
                }
            )

    for provider_tools in (tools.custom_tools or {}).values():
        for tool in provider_tools:
            if not isinstance(tool, Mapping) or _tool_name(tool) != name:
                continue
            function = tool.get("function")
            definition = function if isinstance(function, Mapping) else tool
            parameters = definition.get("parameters", {})
            if not isinstance(parameters, Mapping):
                raise ValueError(f"Function tool {name!r} parameters must be a JSON Schema object")
            matches.append(copy.deepcopy(dict(parameters)))

    if not matches:
        raise ValueError(f"Function tool {name!r} has no declared parameters schema")
    first = matches[0]
    if any(candidate != first for candidate in matches[1:]):
        raise ValueError(f"Function tool {name!r} has conflicting parameters schemas")
    return first


def select_trusted_tools(
    tools: ToolsSchema | None,
    active_names: Iterable[str],
) -> ToolsSchema | None:
    """Return a session-local schema containing exactly the active trusted tools."""
    names = {name for name in active_names if isinstance(name, str) and name}
    if not names:
        return None
    if tools is None:
        raise ValueError(f"Trusted pipeline tool {sorted(names)[0]!r} has no schema")

    standard_tools = [schema for schema in tools.standard_tools if schema.name in names]
    custom_tools: dict[AdapterType, list[dict[str, Any]]] = {}
    selected_names = {schema.name for schema in standard_tools}
    for adapter, provider_tools in (tools.custom_tools or {}).items():
        selected: list[dict[str, Any]] = []
        for tool in provider_tools:
            if not isinstance(tool, Mapping) or (name := _tool_name(tool)) not in names:
                continue
            selected.append(copy.deepcopy(tool))
            selected_names.add(name)
        if selected:
            custom_tools[adapter] = selected

    missing = names - selected_names
    if missing:
        raise ValueError(f"Trusted pipeline tool {sorted(missing)[0]!r} has no schema")
    return ToolsSchema(standard_tools=standard_tools, custom_tools=custom_tools)


def schema_validating_tool_handler(
    handler: ToolHandler,
    *,
    parameters: Mapping[str, Any],
    failure_result_type: ToolResultType = dict,
    failure_properties: FunctionCallResultProperties | None = None,
) -> ToolHandler:
    """Validate arguments immediately before invoking one registered handler."""
    validator = compile_tool_arguments_validator(parameters)

    async def _wrapped(params: FunctionCallParams) -> None:
        failure = tool_argument_validation_failure(validator, params.arguments)
        if failure is not None:
            logger.warning(
                f"Rejecting tool arguments name={params.function_name} call_id={params.tool_call_id} "
                f"code={failure.code}"
            )
            properties = replace(failure_properties) if failure_properties is not None else None
            await params.result_callback(
                failure_result_type(tool_failure(failure.code, failure.message)),
                properties=properties,
            )
            return
        await handler(params)

    return _wrapped


def tool_success(result: Any) -> Any:
    """Preserve successful tool contracts while keeping terminal results truthy."""
    if isinstance(result, Mapping) and isinstance(result.get("ok"), bool):
        return dict(result)
    if isinstance(result, Mapping) and "error" in result:
        raw_error = result.get("error")
        message = str(raw_error or "Tool execution failed")
        details = {key: value for key, value in result.items() if key != "error"}
        error: dict[str, Any] = {"code": "tool_error", "message": message}
        if details:
            error["details"] = details
        return {"ok": False, "error": error}
    if result:
        return dict(result) if isinstance(result, Mapping) else result
    return {"ok": True, "result": result}


def tool_failure(code: str, message: str) -> dict[str, Any]:
    """Return a truthy, structured terminal failure envelope."""
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
        },
    }


def terminal_tool_handler(
    handler: ToolHandler,
    *,
    parameters: Mapping[str, Any],
    timeout_secs: float,
) -> ToolHandler:
    """Wrap a Pipecat handler so every invocation produces one final result.

    Pipecat 1.7 emits an ``ErrorFrame`` when a handler raises and supplies
    ``None`` when its built-in timeout expires. Neither value reliably advances
    the universal assistant aggregator. This wrapper owns the deadline, catches
    failures, normalizes empty values, and makes the final callback idempotent.
    """
    if not math.isfinite(timeout_secs) or timeout_secs <= 0:
        raise ValueError("timeout_secs must be a positive finite number")
    validated_handler = schema_validating_tool_handler(handler, parameters=parameters)

    async def _wrapped(params: FunctionCallParams) -> None:
        final_delivered = False
        handler_deliveries_allowed = True
        delivery_lock = asyncio.Lock()

        async def _deliver(
            result: Any,
            *,
            properties: FunctionCallResultProperties | None = None,
        ) -> None:
            nonlocal final_delivered
            is_final = properties is None or properties.is_final
            async with delivery_lock:
                if final_delivered:
                    logger.warning(
                        f"Ignoring tool callback after terminal result name={params.function_name} "
                        f"call_id={params.tool_call_id}"
                    )
                    return
                normalized = tool_success(result) if is_final else result
                await params.result_callback(normalized, properties=properties)
                if is_final:
                    final_delivered = True

        async def _handler_deliver(
            result: Any,
            *,
            properties: FunctionCallResultProperties | None = None,
        ) -> None:
            if not handler_deliveries_allowed:
                logger.warning(
                    f"Ignoring tool result after wrapper termination name={params.function_name} "
                    f"call_id={params.tool_call_id}"
                )
                return
            await _deliver(result, properties=properties)

        def _consume_task_result(task: asyncio.Task) -> None:
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning(
                    f"Detached tool handler finished with an error name={params.function_name} "
                    f"call_id={params.tool_call_id}: {exc}"
                )

        async def _stop_handler_task(task: asyncio.Task) -> None:
            if not task.done():
                task.cancel()
            done, _pending = await asyncio.wait({task}, timeout=1.0)
            if task not in done:
                # A handler that suppresses cancellation must not extend the
                # protocol deadline or deliver a late result. Its callback is
                # already fenced by ``handler_deliveries_allowed``.
                logger.error(
                    f"Tool handler ignored cancellation name={params.function_name} call_id={params.tool_call_id}"
                )
                task.add_done_callback(_consume_task_result)
                return
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning(
                    f"Tool handler raised while stopping name={params.function_name} "
                    f"call_id={params.tool_call_id}: {exc}"
                )

        wrapped_params = replace(params, result_callback=_handler_deliver)
        handler_task = asyncio.create_task(validated_handler(wrapped_params))
        try:
            done, _pending = await asyncio.wait({handler_task}, timeout=timeout_secs)
            if handler_task not in done:
                handler_deliveries_allowed = False
                logger.warning(
                    f"Tool timed out name={params.function_name} call_id={params.tool_call_id} "
                    f"deadline_secs={timeout_secs:.3f}"
                )
                await _deliver(
                    tool_failure(
                        "tool_timeout",
                        f"Tool '{params.function_name}' exceeded its {timeout_secs:g} second deadline",
                    )
                )
                await _stop_handler_task(handler_task)
            else:
                await handler_task
        except asyncio.CancelledError:
            handler_deliveries_allowed = False
            logger.info(f"Tool cancelled name={params.function_name} call_id={params.tool_call_id}")
            try:
                await _stop_handler_task(handler_task)
                await asyncio.shield(
                    _deliver(
                        tool_failure(
                            "tool_cancelled",
                            f"Tool '{params.function_name}' was cancelled",
                        ),
                        properties=FunctionCallResultProperties(run_llm=False),
                    )
                )
            finally:
                raise
        except Exception as exc:
            logger.exception(f"Tool failed name={params.function_name} call_id={params.tool_call_id}: {exc}")
            await _deliver(
                tool_failure(
                    "tool_execution_error",
                    f"Tool '{params.function_name}' failed",
                )
            )

        if not final_delivered:
            await _deliver(
                tool_failure(
                    "tool_result_missing",
                    f"Tool '{params.function_name}' completed without a result",
                )
            )

    return _wrapped
