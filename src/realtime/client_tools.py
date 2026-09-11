# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Connection-scoped bridge between Realtime client tools and Pipecat."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from loguru import logger
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.frames.frames import FunctionCallResultProperties
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.utils.async_tool_cancellation import CANCEL_ASYNC_TOOL_NAME

from realtime.protocol import RealtimeProtocolError

_MAX_TRACKED_CALLS = 4096
_DEFAULT_OUTPUT_TIMEOUT_SECS = 120.0
_MAX_TERMINAL_CALLS = 4096
_MAX_CLIENT_TOOL_BINDINGS = 4096
_MAX_CLIENT_TOOL_BINDING_ALLOCATION_ATTEMPTS = 4096
# Pipecat bounds task cancellation to one second so a dependency that swallows
# CancelledError cannot freeze pipeline teardown. Give its result callback the
# same bounded grace period when a client-tool handler is being cancelled.
_RESULT_CALLBACK_CANCELLATION_GRACE_SECS = 1.0
_CLIENT_TOOL_CANCELLED = object()
_CLIENT_TOOL_TIMED_OUT = object()
_CANONICAL_CLIENT_TOOL_FIELDS = frozenset({"description", "name", "parameters", "type"})

_TerminalOutcome = Literal["completed", "cancelled", "failed", "timed_out"]
_ClientToolHandler = Callable[[FunctionCallParams], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class PreparedClientTools:
    """Provider-safe projection for one public Realtime tool snapshot."""

    pipeline_tools: list[dict[str, Any]]
    pipeline_tool_choice: str | dict[str, Any]
    bindings: dict[str, str]


@dataclass(frozen=True, slots=True)
class ClientToolProjectionSnapshot:
    """Detached provider-name registry state for one atomic preparation."""

    primary_pipeline_names: tuple[tuple[str, str], ...]
    pipeline_names: tuple[tuple[str, str], ...]


class ClientToolTimeoutResult(dict[str, Any]):
    """Internal typed envelope for a client tool that missed its deadline."""


class ClientToolCancelledResult(dict[str, Any]):
    """Internal typed envelope used to close a cancelled Pipecat tool call."""


@dataclass(slots=True)
class _ClientToolCall:
    """One client-owned call waiting for its wire-protocol output."""

    name: str
    output_future: asyncio.Future[str | object] | None = None
    deadline_at: float | None = None
    output: str | None = None
    output_received_at: float | None = None
    output_released: bool = False
    handler_registered: bool = False
    handler_task: asyncio.Task[Any] | None = None
    cancelled: bool = False
    timed_out: bool = False
    context_applied: asyncio.Event = field(default_factory=asyncio.Event)
    context_finalized: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass(frozen=True, slots=True)
class _TerminalClientToolCall:
    """Lightweight tombstone retained after a broker call leaves active state."""

    name: str
    outcome: _TerminalOutcome
    output_received: bool
    context_applied: bool


def _schema_tool_name(tool: Mapping[str, Any]) -> str | None:
    """Return a function name from canonical Realtime or provider schema shape."""
    name = tool.get("name")
    if isinstance(name, str) and name:
        return name
    function = tool.get("function")
    if not isinstance(function, Mapping):
        return None
    name = function.get("name")
    return name if isinstance(name, str) and name else None


def _trusted_schema_names(tools: ToolsSchema | None) -> set[str]:
    """Return function names already owned by trusted pipeline handlers."""
    if tools is None:
        return set()
    names = {schema.name for schema in tools.standard_tools}
    for provider_tools in (tools.custom_tools or {}).values():
        for tool in provider_tools:
            if isinstance(tool, Mapping) and (name := _schema_tool_name(tool)):
                names.add(name)
    return names


def _canonical_client_tools(raw_tools: object) -> list[dict[str, Any]]:
    """Defensively copy canonical Realtime function schemas from runtime data."""
    if raw_tools is None:
        return []
    if not isinstance(raw_tools, list):
        raise ValueError("client_tools must be a list of canonical Realtime function schemas")

    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw_tool in enumerate(raw_tools):
        label = f"client_tools[{index}]"
        if not isinstance(raw_tool, Mapping):
            raise ValueError(f"{label} must be an object")
        unknown = set(raw_tool) - _CANONICAL_CLIENT_TOOL_FIELDS
        if unknown:
            field = min(str(name) for name in unknown)
            raise ValueError(f"{label} contains unsupported field {field!r}")
        if raw_tool.get("type") != "function":
            raise ValueError(f"{label}.type must be 'function'")
        name = raw_tool.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label}.name must be a non-empty string")
        if name in names:
            raise ValueError(f"Duplicate client tool name {name!r}")
        names.add(name)
        description = raw_tool.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError(f"{label}.description must be a string")
        parameters = raw_tool.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError(f"{label}.parameters must be an object")
        normalized_tool: dict[str, Any] = {
            "type": "function",
            "name": name,
            "parameters": copy.deepcopy(dict(parameters)),
        }
        if description is not None:
            normalized_tool["description"] = description
        normalized.append(normalized_tool)
    return normalized


def _as_openai_chat_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one canonical Realtime function schema to Chat Completions shape."""
    function: dict[str, Any] = {
        "name": tool["name"],
        "parameters": copy.deepcopy(tool["parameters"]),
    }
    if "description" in tool:
        function["description"] = tool["description"]
    return {"type": "function", "function": function}


def _merge_client_tool_schemas(
    trusted_tools: ToolsSchema | None,
    client_tools: list[dict[str, Any]],
) -> ToolsSchema:
    """Build one connection-local schema without mutating trusted definitions."""
    client_openai_tools = [_as_openai_chat_tool(tool) for tool in client_tools]
    if trusted_tools is None:
        return ToolsSchema(
            standard_tools=[],
            custom_tools={AdapterType.OPENAI: client_openai_tools},
        )
    combined = copy.copy(trusted_tools)
    custom_tools = {
        adapter: list(provider_tools) for adapter, provider_tools in (trusted_tools.custom_tools or {}).items()
    }
    custom_tools.setdefault(AdapterType.OPENAI, []).extend(client_openai_tools)
    combined.custom_tools = custom_tools
    return combined


class RealtimeClientToolProjection:
    """Map public client functions to collision-safe provider identities."""

    def __init__(self, *, scope_id: str) -> None:
        """Create a mapping registry isolated to one Realtime connection."""
        if not isinstance(scope_id, str) or not scope_id:
            raise ValueError("Realtime client tool projection requires a non-empty scope ID")
        self._scope_id = scope_id
        self._llm: LLMService | None = None
        self._client_tool_handler: _ClientToolHandler | None = None
        self._primary_pipeline_names: dict[str, str] = {}
        self._pipeline_names: dict[str, str] = {}

    def bind_llm(self, llm: LLMService) -> None:
        """Bind the sole provider service used by this connection."""
        if self._llm is not None and self._llm is not llm:
            raise RuntimeError("Realtime client tool projection cannot be rebound to another LLM")
        self._llm = llm

    def snapshot(self) -> ClientToolProjectionSnapshot:
        """Capture the complete provider-name registry without sharing maps."""
        return ClientToolProjectionSnapshot(
            primary_pipeline_names=tuple(sorted(self._primary_pipeline_names.items())),
            pipeline_names=tuple(sorted(self._pipeline_names.items())),
        )

    def restore(
        self,
        snapshot: ClientToolProjectionSnapshot,
        *,
        expected: ClientToolProjectionSnapshot | None = None,
    ) -> None:
        """Restore a failed preparation without overwriting newer state."""
        if not isinstance(snapshot, ClientToolProjectionSnapshot):
            raise TypeError("Realtime client tool projection rollback requires a snapshot")
        if expected is not None and self.snapshot() != expected:
            raise RuntimeError("Realtime client tool projection changed before rollback")
        self._primary_pipeline_names = dict(snapshot.primary_pipeline_names)
        self._pipeline_names = dict(snapshot.pipeline_names)

    def configure(
        self,
        llm: LLMService,
        raw_client_tools: object,
        *,
        client_tool_handler: _ClientToolHandler,
        trusted_tools: ToolsSchema | None = None,
        trusted_tool_names: Iterable[str] = (),
    ) -> ToolsSchema | None:
        """Validate initial ownership and install one client-tool handler."""
        client_tools = _canonical_client_tools(raw_client_tools)
        trusted_names = _trusted_schema_names(trusted_tools)
        trusted_names.update(name for name in trusted_tool_names if isinstance(name, str) and name)
        client_names = {tool["name"] for tool in client_tools}
        collisions = trusted_names & client_names
        if collisions:
            raise ValueError(f"Client tool name {sorted(collisions)[0]!r} conflicts with a trusted pipeline tool")
        if self._client_tool_handler is not None:
            raise RuntimeError("Realtime client tools are already configured for this connection")
        functions = getattr(llm, "_functions", None)
        if isinstance(functions, Mapping) and None in functions:
            raise RuntimeError("Realtime client tools cannot replace an existing catch-all function handler")

        self.bind_llm(llm)
        previous_primary = self._primary_pipeline_names
        previous_pipeline = self._pipeline_names
        self._client_tool_handler = client_tool_handler
        try:
            projected = self.project_tools(
                client_tools,
                "auto",
                trusted_names=trusted_names,
            )
            llm.register_function(
                None,
                self.handle,
                cancel_on_interruption=False,
            )
        except BaseException:
            self._primary_pipeline_names = previous_primary
            self._pipeline_names = previous_pipeline
            self._client_tool_handler = None
            raise

        if not client_tools:
            logger.info("Registered the Realtime response-scoped client-tool handler")
            return trusted_tools
        logger.info(f"Registered {len(client_tools)} Realtime client-owned tool(s): {sorted(client_names)}")
        return _merge_client_tool_schemas(trusted_tools, projected.pipeline_tools)

    def project_tools(
        self,
        tools: list[dict[str, Any]],
        tool_choice: Any,
        *,
        trusted_names: Iterable[str] = (),
        occupied_names: Iterable[str] = (),
    ) -> PreparedClientTools:
        """Project one session/response snapshot into provider-safe names."""
        trusted = {name for name in trusted_names if isinstance(name, str) and name}
        public_names: set[str] = set()
        for tool in tools:
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("Realtime function tools require non-empty names")
            if name in public_names:
                raise ValueError(f"Duplicate Realtime function tool name {name!r}")
            public_names.add(name)

        client_names = public_names - trusted
        if client_names and (self._llm is None or self._client_tool_handler is None):
            raise RuntimeError("Realtime client tools were not configured before provider projection")

        hard_reserved = {name for name in occupied_names if isinstance(name, str) and name}
        hard_reserved.update(trusted)
        hard_reserved.update(self._provider_reserved_names())
        # Generated aliases must never shadow a public tool, including a tool
        # processed later in this snapshot. A tool's own public name is still
        # its preferred provider identity when no real collision exists.
        alias_unavailable = hard_reserved | public_names
        planned_primary = dict(self._primary_pipeline_names)
        planned_pipeline = dict(self._pipeline_names)
        active_bindings: dict[str, str] = {}
        projected_tools: list[dict[str, Any]] = []
        for tool in tools:
            projected = copy.deepcopy(tool)
            public_name = str(tool["name"])
            if public_name not in client_names:
                projected_tools.append(projected)
                continue
            pipeline_name = planned_primary.get(public_name)
            if pipeline_name is not None and (
                pipeline_name in hard_reserved
                or planned_pipeline.get(pipeline_name) not in {None, public_name}
                or (pipeline_name != public_name and pipeline_name in public_names)
            ):
                pipeline_name = None
            if pipeline_name is None:
                existing_owner = planned_pipeline.get(public_name)
                if public_name not in hard_reserved and existing_owner in {None, public_name}:
                    if existing_owner is None and len(planned_pipeline) >= _MAX_CLIENT_TOOL_BINDINGS:
                        raise RealtimeProtocolError(
                            message=(f"Could not allocate a provider Realtime binding for {public_name!r}"),
                            code="client_tool_binding_limit",
                            param="tools",
                        )
                    pipeline_name = public_name
                else:
                    pipeline_name = self._allocate_pipeline_name(
                        public_name,
                        unavailable=alias_unavailable,
                        pipeline_names=planned_pipeline,
                    )
                planned_primary[public_name] = pipeline_name
                planned_pipeline[pipeline_name] = public_name
            alias_unavailable.add(pipeline_name)
            active_bindings[pipeline_name] = public_name
            projected["name"] = pipeline_name
            projected_tools.append(projected)

        projected_choice = copy.deepcopy(tool_choice)
        if isinstance(projected_choice, dict) and projected_choice.get("type") == "function":
            selected_name = projected_choice.get("name")
            if isinstance(selected_name, str):
                public_to_pipeline = {public: pipeline for pipeline, public in active_bindings.items()}
                projected_choice["name"] = public_to_pipeline.get(selected_name, selected_name)

        self._primary_pipeline_names = planned_primary
        self._pipeline_names = planned_pipeline
        return PreparedClientTools(
            pipeline_tools=projected_tools,
            pipeline_tool_choice=projected_choice,
            bindings=active_bindings,
        )

    async def handle(self, params: FunctionCallParams) -> None:
        """Translate a provider call back to its exact public client identity."""
        public_name = self._pipeline_names.get(str(params.function_name or ""))
        if public_name is None:
            raise RuntimeError(f"Provider function {params.function_name!r} has no Realtime client binding")
        handler = self._client_tool_handler
        if handler is None:
            raise RuntimeError("Realtime client tool handler is not configured")
        await handler(replace(params, function_name=public_name))

    def _provider_reserved_names(self) -> set[str]:
        """Read names owned by Pipecat/provider internals at the projection edge."""
        reserved = {CANCEL_ASYNC_TOOL_NAME}
        llm = self._llm
        if llm is None:
            return reserved
        functions = getattr(llm, "_functions", None)
        if isinstance(functions, Mapping):
            reserved.update(name for name in functions if isinstance(name, str) and name)
        adapter = getattr(llm, "_adapter", None)
        builtin_tools = getattr(adapter, "builtin_tools", None)
        if isinstance(builtin_tools, Mapping):
            reserved.update(name for name in builtin_tools if isinstance(name, str) and name)
        return reserved

    def _allocate_pipeline_name(
        self,
        public_name: str,
        *,
        unavailable: set[str],
        pipeline_names: Mapping[str, str],
    ) -> str:
        """Allocate a deterministic private name using exact collision checks."""
        if len(pipeline_names) >= _MAX_CLIENT_TOOL_BINDINGS:
            raise RealtimeProtocolError(
                message=f"Realtime client tools support at most {_MAX_CLIENT_TOOL_BINDINGS} connection bindings",
                code="client_tool_binding_limit",
                param="tools",
            )
        for attempt in range(_MAX_CLIENT_TOOL_BINDING_ALLOCATION_ATTEMPTS):
            digest = hashlib.sha256(f"{self._scope_id}\0{public_name}\0{attempt}".encode()).hexdigest()[:24]
            candidate = f"realtime_client_{digest}"
            if candidate in unavailable:
                continue
            bound_public_name = pipeline_names.get(candidate)
            if bound_public_name is None or bound_public_name == public_name:
                return candidate
        raise RealtimeProtocolError(
            message=f"Could not allocate a private Realtime binding for {public_name!r}",
            code="client_tool_binding_unavailable",
            param="tools",
        )


class ClientToolBroker:
    """Resolve client tool outputs into Pipecat without executing them locally."""

    def __init__(self, *, output_timeout_secs: float = _DEFAULT_OUTPUT_TIMEOUT_SECS) -> None:
        """Create an empty broker for one WebSocket connection."""
        if not math.isfinite(output_timeout_secs) or output_timeout_secs <= 0:
            raise ValueError("Client tool output timeout must be a positive finite number")
        self._calls: dict[str, _ClientToolCall] = {}
        self._terminal_calls: OrderedDict[str, _TerminalClientToolCall] = OrderedDict()
        self._lock = asyncio.Lock()
        self._closed = False
        self._closed_event = asyncio.Event()
        self._output_timeout_secs = output_timeout_secs
        self._llm_context: Any | None = None

    def bind_context(self, context: Any) -> None:
        """Bind Pipecat's canonical shared context for exact output storage."""
        get_messages = getattr(context, "get_messages", None)
        if not callable(get_messages):
            raise TypeError("Realtime client tools require an LLMContext-compatible object")
        if self._llm_context is not None and self._llm_context is not context:
            raise RuntimeError("Realtime client tool broker is already bound to another context")
        self._llm_context = context

    async def handle(self, params: FunctionCallParams) -> None:
        """Wait for the client output, then close Pipecat's tool context state."""
        call_id = str(params.tool_call_id or "")
        name = str(params.function_name or "")
        if not call_id or not name:
            raise RuntimeError("Client tool calls require non-empty names and call IDs")
        if self._llm_context is None:
            raise RuntimeError("Realtime client tool broker has no canonical LLM context")

        async with self._lock:
            if self._closed:
                raise RuntimeError("Realtime client tool broker is closed")
            terminal = self._terminal_calls.get(call_id)
            if terminal is not None:
                raise RuntimeError(
                    f"Client tool call {call_id!r} cannot be reused after terminal outcome {terminal.outcome!r}"
                )
            call = self._calls.get(call_id)
            if call is None:
                if len(self._calls) >= _MAX_TRACKED_CALLS:
                    raise RuntimeError("Realtime client tool call limit exceeded")
                call = _ClientToolCall(name=name)
                self._calls[call_id] = call
            if call.name != name:
                raise RuntimeError(f"Client tool call {call_id!r} changed name")
            if call.handler_registered:
                raise RuntimeError(f"Client tool call {call_id!r} registered more than once")
            call.handler_registered = True
            call.handler_task = asyncio.current_task()
            loop = asyncio.get_running_loop()
            call.deadline_at = loop.time() + self._output_timeout_secs
            call.output_future = loop.create_future()
            if call.cancelled:
                call.output_future.set_result(_CLIENT_TOOL_CANCELLED)
            elif call.timed_out:
                call.output_future.set_result(_CLIENT_TOOL_TIMED_OUT)
            elif call.output_released and call.output is not None:
                call.output_future.set_result(call.output)
            output_future = call.output_future
            deadline_at = call.deadline_at

        callback_task: asyncio.Task[None] | None = None
        context_output: str | None = None

        async def _context_applied() -> None:
            if context_output is not None:
                _restore_exact_context_output(
                    self._llm_context,
                    call_id=call_id,
                    output=context_output,
                )
            async with self._lock:
                self._mark_context_applied_locked(call_id, call)

        async def _deliver_result(result: Any) -> None:
            await params.result_callback(
                result,
                properties=FunctionCallResultProperties(
                    run_llm=False,
                    on_context_updated=_context_applied,
                ),
            )

        try:
            timeout_result: ClientToolTimeoutResult | None = None
            try:
                output = await asyncio.wait_for(
                    asyncio.shield(output_future),
                    timeout=max(0.0, deadline_at - asyncio.get_running_loop().time()),
                )
            except TimeoutError:
                async with self._lock:
                    # Response A can retain an output past its call deadline,
                    # but only when the event reached the broker on time. Use
                    # monotonic timestamps rather than callback/lock ordering.
                    staged_before_deadline = (
                        call.output is not None
                        and call.output_received_at is not None
                        and call.output_received_at <= deadline_at
                    )
                    if not staged_before_deadline:
                        timeout_result = _timeout_result(name, self._output_timeout_secs)
                        self._mark_timed_out_locked(call)
                if staged_before_deadline:
                    output = await asyncio.shield(output_future)

            if timeout_result is not None:
                result: Any = timeout_result
            elif output is _CLIENT_TOOL_TIMED_OUT:
                result = _timeout_result(name, self._output_timeout_secs)
            elif output is _CLIENT_TOOL_CANCELLED:
                result = _cancelled_result(name)
            else:
                assert isinstance(output, str)
                context_output = output
                result = _context_result(output)

            callback_task = asyncio.create_task(
                _deliver_result(result),
                name=f"realtime-client-tool-result-{call_id}",
            )
            await asyncio.shield(callback_task)
        except asyncio.CancelledError:
            # A Pipecat function call remains in-progress until result_callback
            # runs. Give that callback a bounded chance to close Pipecat's
            # internal lifecycle, but retire connection-owned state first so
            # a stalled callback cannot keep this broker or call active.
            async with self._lock:
                call.cancelled = True
                if call.output_future is not None and not call.output_future.done():
                    call.output_future.cancel()
                accepted_output = call.output if call.output_released else None
                timed_out = call.timed_out
                self._retire_call_locked(call_id, call, outcome="cancelled")

            if callback_task is None:
                if timed_out:
                    cancellation_result: Any = _timeout_result(name, self._output_timeout_secs)
                elif accepted_output is None:
                    cancellation_result: Any = _cancelled_result(name)
                else:
                    cancellation_result = _context_result(accepted_output)
                    # Preserve the client's exact opaque value if the output
                    # crossed the release boundary just before cancellation.
                    context_output = accepted_output
                callback_task = asyncio.create_task(
                    _deliver_result(cancellation_result),
                    name=f"realtime-client-tool-cancel-result-{call_id}",
                )

            cleanup_error = await _drain_with_deadline(
                callback_task,
                timeout=_RESULT_CALLBACK_CANCELLATION_GRACE_SECS,
            )
            if cleanup_error is not None:
                async with self._lock:
                    call.context_finalized.set()
                logger.warning(
                    "Could not close cancelled Realtime client tool context within the cleanup deadline "
                    "call_id={} error={!r}",
                    call_id,
                    cleanup_error,
                )
            raise
        except Exception:
            # A provider/context callback failure is terminal for this call.
            # Keep a bounded tombstone for duplicate detection, but never
            # leave the failed callback registered as pending context work.
            async with self._lock:
                call.context_finalized.set()
                self._retire_call_locked(call_id, call, outcome="failed")
            raise

    async def stage_output(self, *, call_id: str, name: str, output: str) -> None:
        """Reserve one validated client output without waking its handler."""
        output_received_at = asyncio.get_running_loop().time()
        async with self._lock:
            if self._closed:
                raise RealtimeProtocolError(
                    message="The Realtime client tool broker is closed",
                    code="client_tool_broker_closed",
                    param="item.call_id",
                    error_type="server_error",
                )
            terminal = self._terminal_calls.get(call_id)
            if terminal is not None:
                if terminal.name != name:
                    raise RealtimeProtocolError(
                        message=f"Client tool call {call_id!r} changed name",
                        code="tool_name_mismatch",
                        param="item.call_id",
                    )
                raise _terminal_output_error(call_id, terminal.outcome)
            call = self._calls.get(call_id)
            if call is None:
                if len(self._calls) >= _MAX_TRACKED_CALLS:
                    raise RealtimeProtocolError(
                        message="Realtime client tool call limit exceeded",
                        code="client_tool_call_limit_exceeded",
                        param="item.call_id",
                        error_type="server_error",
                    )
                call = _ClientToolCall(name=name)
                self._calls[call_id] = call
            if call.name != name:
                raise RealtimeProtocolError(
                    message=f"Client tool call {call_id!r} changed name",
                    code="tool_name_mismatch",
                    param="item.call_id",
                )
            if call.cancelled:
                raise RealtimeProtocolError(
                    message=f"Client tool call {call_id!r} was cancelled",
                    code="client_tool_call_cancelled",
                    param="item.call_id",
                )
            if call.timed_out:
                raise RealtimeProtocolError(
                    message=f"Client tool call {call_id!r} exceeded its output deadline",
                    code="client_tool_timeout",
                    param="item.call_id",
                )
            if call.deadline_at is not None and output_received_at > call.deadline_at:
                self._mark_timed_out_locked(call)
                raise RealtimeProtocolError(
                    message=f"Client tool call {call_id!r} exceeded its output deadline",
                    code="client_tool_timeout",
                    param="item.call_id",
                )
            if call.output is not None:
                raise RealtimeProtocolError(
                    message=f"Client tool call {call_id!r} already has an output",
                    code="duplicate_tool_output",
                    param="item.call_id",
                )
            call.output = output
            call.output_received_at = output_received_at

    async def release_output(self, *, call_id: str, name: str) -> None:
        """Wake the handler after Response A and output acknowledgements publish."""
        async with self._lock:
            if self._closed:
                raise RealtimeProtocolError(
                    message="The Realtime client tool broker is closed",
                    code="client_tool_broker_closed",
                    param="item.call_id",
                    error_type="server_error",
                )
            terminal = self._terminal_calls.get(call_id)
            if terminal is not None:
                if terminal.name != name:
                    raise RealtimeProtocolError(
                        message=f"Client tool call {call_id!r} changed name",
                        code="tool_name_mismatch",
                        param="item.call_id",
                    )
                raise _terminal_output_error(call_id, terminal.outcome)
            call = self._calls.get(call_id)
            if call is None or call.output is None:
                raise RealtimeProtocolError(
                    message=f"Client tool call {call_id!r} has no staged output",
                    code="client_tool_output_missing",
                    param="item.call_id",
                    error_type="server_error",
                )
            if call.name != name:
                raise RealtimeProtocolError(
                    message=f"Client tool call {call_id!r} changed name",
                    code="tool_name_mismatch",
                    param="item.call_id",
                )
            if call.cancelled:
                raise RealtimeProtocolError(
                    message=f"Client tool call {call_id!r} was cancelled",
                    code="client_tool_call_cancelled",
                    param="item.call_id",
                )
            if call.timed_out:
                raise RealtimeProtocolError(
                    message=f"Client tool call {call_id!r} exceeded its output deadline",
                    code="client_tool_timeout",
                    param="item.call_id",
                )
            if call.output_released:
                raise RealtimeProtocolError(
                    message=f"Client tool call {call_id!r} already has an output",
                    code="duplicate_tool_output",
                    param="item.call_id",
                )
            if call.output_future is not None and call.output_future.done():
                raise RealtimeProtocolError(
                    message=f"Client tool call {call_id!r} cannot accept an output",
                    code="client_tool_handler_unavailable",
                    param="item.call_id",
                    error_type="server_error",
                )
            call.output_released = True
            if call.output_future is not None:
                call.output_future.set_result(call.output)

    async def discard_staged_output(self, *, call_id: str, name: str) -> None:
        """Cancel a staged output when its originating response did not complete."""
        async with self._lock:
            call = self._calls.get(call_id)
            if call is None:
                return
            if call.name != name:
                raise RealtimeProtocolError(
                    message=f"Client tool call {call_id!r} changed name",
                    code="tool_name_mismatch",
                    param="item.call_id",
                )
            if call.output_released:
                raise RealtimeProtocolError(
                    message=f"Client tool call {call_id!r} was already released",
                    code="client_tool_handler_unavailable",
                    param="item.call_id",
                    error_type="server_error",
                )
            call.cancelled = True
            if call.output_future is not None and not call.output_future.done():
                call.output_future.set_result(_CLIENT_TOOL_CANCELLED)

    async def outputs_staged(self, call_ids: tuple[str, ...]) -> bool:
        """Return whether every active call already has a client output.

        This is the admission check for an immediately following
        ``response.create``.  It intentionally runs under the broker lock so a
        deadline transition cannot be mistaken for a client-staged output.
        """
        if not call_ids:
            return False
        async with self._lock:
            for call_id in call_ids:
                call = self._calls.get(call_id)
                if call is None or call.cancelled or call.timed_out or call.output is None:
                    return False
            return True

    async def wait_context_applied(self, call_id: str, *, timeout: float) -> None:
        """Wait until Pipecat has replaced the call's in-progress context item."""
        async with self._lock:
            terminal = self._terminal_calls.get(call_id)
            if terminal is not None:
                if terminal.context_applied:
                    return
                raise _terminal_context_error(call_id, terminal.outcome)
            call = self._calls.get(call_id)
            if call is None:
                raise RealtimeProtocolError(
                    message=f"Client tool call {call_id!r} was not registered",
                    code="client_tool_handler_missing",
                    param="item.call_id",
                    error_type="server_error",
                )
            context_finalized = call.context_finalized
        context_task = asyncio.create_task(context_finalized.wait())
        closed_task = asyncio.create_task(self._closed_event.wait())
        try:
            done, _ = await asyncio.wait(
                {context_task, closed_task},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if context_task in done and context_finalized.is_set():
                async with self._lock:
                    terminal = self._terminal_calls.get(call_id)
                    if terminal is not None:
                        if terminal.context_applied:
                            return
                        raise _terminal_context_error(call_id, terminal.outcome)
                    if call.context_applied.is_set():
                        return
                    raise RealtimeProtocolError(
                        message=f"Client tool call {call_id!r} did not reach pipeline context",
                        code="client_tool_context_unavailable",
                        param="item.call_id",
                        error_type="server_error",
                    )
            if closed_task in done and self._closed_event.is_set():
                raise RealtimeProtocolError(
                    message="The Realtime client tool broker is closed",
                    code="client_tool_broker_closed",
                    param="item.call_id",
                    error_type="server_error",
                )
            raise TimeoutError
        except TimeoutError as exc:
            raise RealtimeProtocolError(
                message=f"Client tool call {call_id!r} was not applied to the pipeline context in time",
                code="client_tool_context_timeout",
                param="item.call_id",
                error_type="server_error",
            ) from exc
        finally:
            for task in (context_task, closed_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(context_task, closed_task, return_exceptions=True)

    def pending_context_call_ids(self) -> tuple[str, ...]:
        """Return accepted outputs that have not reached Pipecat context yet."""
        active = (
            call_id
            for call_id, call in self._calls.items()
            if call.output_released and call.output is not None and not call.context_applied.is_set()
        )
        terminal = (
            call_id
            for call_id, call in self._terminal_calls.items()
            if call.outcome in {"completed", "failed", "timed_out"}
            and call.output_received
            and not call.context_applied
        )
        return tuple((*active, *terminal))

    def shutdown(self) -> None:
        """Cancel every outstanding handler when the transport disconnects."""
        self._closed = True
        self._closed_event.set()
        for call_id, call in list(self._calls.items()):
            call.cancelled = True
            if call.output_future is not None and not call.output_future.done():
                call.output_future.cancel()
            self._retire_call_locked(call_id, call, outcome="cancelled")
            if call.handler_task is not None and not call.handler_task.done():
                call.handler_task.cancel()

    def _retire_call_locked(
        self,
        call_id: str,
        call: _ClientToolCall,
        *,
        outcome: _TerminalOutcome,
    ) -> None:
        """Replace one active record with a bounded duplicate-detection tombstone."""
        if self._calls.get(call_id) is not call:
            return
        self._calls.pop(call_id, None)
        self._terminal_calls[call_id] = _TerminalClientToolCall(
            name=call.name,
            outcome=outcome,
            output_received=call.output is not None,
            context_applied=call.context_applied.is_set(),
        )
        self._terminal_calls.move_to_end(call_id)
        self._prune_terminal_calls_locked()

    def _mark_context_applied_locked(self, call_id: str, call: _ClientToolCall) -> None:
        """Publish successful context application to active and terminal waiters."""
        call.context_applied.set()
        call.context_finalized.set()
        if self._calls.get(call_id) is call:
            outcome: _TerminalOutcome
            if call.cancelled:
                outcome = "cancelled"
            elif call.timed_out:
                outcome = "timed_out"
            else:
                outcome = "completed"
            self._retire_call_locked(call_id, call, outcome=outcome)
            return
        terminal = self._terminal_calls.get(call_id)
        if terminal is not None and terminal.name == call.name and not terminal.context_applied:
            self._terminal_calls[call_id] = replace(terminal, context_applied=True)

    @staticmethod
    def _mark_timed_out_locked(call: _ClientToolCall) -> None:
        """Make one missed absolute deadline visible to every broker waiter."""
        call.timed_out = True
        call.output_released = True
        if call.output_future is not None and not call.output_future.done():
            call.output_future.set_result(_CLIENT_TOOL_TIMED_OUT)

    def _prune_terminal_calls_locked(self) -> None:
        """Bound terminal history without dropping an unapplied accepted output."""
        while len(self._terminal_calls) > _MAX_TERMINAL_CALLS:
            removable = next(
                (
                    call_id
                    for call_id, call in self._terminal_calls.items()
                    if call.outcome in {"cancelled", "failed"} or call.context_applied or not call.output_received
                ),
                None,
            )
            if removable is None:
                return
            self._terminal_calls.pop(removable, None)


async def _drain_with_deadline(
    task: asyncio.Task[None],
    *,
    timeout: float,
) -> BaseException | None:
    """Give a callback bounded cleanup time without extending caller teardown."""
    if not task.done():
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout)
        except asyncio.CancelledError as exc:
            _cancel_detached_task(task)
            return exc
        if task not in done:
            _cancel_detached_task(task)
            return TimeoutError(f"result callback did not finish within {timeout:g} seconds")
    if task.cancelled():
        return asyncio.CancelledError()
    try:
        task.result()
    except Exception as exc:
        return exc
    return None


def _cancel_detached_task(task: asyncio.Task[Any]) -> None:
    """Cancel a timed-out callback and consume its eventual terminal state."""
    if task.done():
        return
    task.cancel()
    task.add_done_callback(_consume_detached_task)


def _consume_detached_task(task: asyncio.Task[Any]) -> None:
    """Avoid an un-retrieved exception if a cancelled callback exits later."""
    with suppress(asyncio.CancelledError):
        task.exception()


def _context_result(output: str) -> Any:
    """Build a truthy result for Pipecat before restoring the exact wire value."""
    try:
        result = json.loads(output)
    except json.JSONDecodeError:
        result = output
    # Pipecat maps falsey results to the literal COMPLETED marker. Preserve the
    # client's value inside an explicit JSON envelope instead.
    return result if result else {"result": result}


def _restore_exact_context_output(context: Any, *, call_id: str, output: str) -> None:
    """Replace Pipecat's JSON serialization with the client's opaque output."""
    get_messages = getattr(context, "get_messages", None)
    if not callable(get_messages):
        raise RuntimeError("Realtime client tool result has no LLM context")
    messages = get_messages()
    if not isinstance(messages, list):
        raise RuntimeError("Realtime client tool LLM context did not return a message list")
    matches = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "tool" and message.get("tool_call_id") == call_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Realtime client tool call {call_id!r} has {len(matches)} matching LLM context messages")
    matches[0]["content"] = output


def _timeout_result(name: str, timeout_secs: float) -> ClientToolTimeoutResult:
    """Build the internal result that closes a client tool after its deadline."""
    return ClientToolTimeoutResult(
        {
            "ok": False,
            "error": {
                "code": "client_tool_timeout",
                "message": f"Client tool {name!r} did not return an output within {timeout_secs:g} seconds",
            },
        }
    )


def _cancelled_result(name: str) -> ClientToolCancelledResult:
    """Build the internal result that closes a cancelled client tool context."""
    return ClientToolCancelledResult(
        {
            "ok": False,
            "error": {
                "code": "client_tool_cancelled",
                "message": f"Client tool {name!r} was cancelled before returning an output",
            },
        }
    )


def _terminal_output_error(call_id: str, outcome: _TerminalOutcome) -> RealtimeProtocolError:
    """Return the stable wire error for output submitted after terminal state."""
    if outcome == "completed":
        return RealtimeProtocolError(
            message=f"Client tool call {call_id!r} already has an output",
            code="duplicate_tool_output",
            param="item.call_id",
        )
    if outcome == "timed_out":
        return RealtimeProtocolError(
            message=f"Client tool call {call_id!r} exceeded its output deadline",
            code="client_tool_timeout",
            param="item.call_id",
        )
    if outcome == "failed":
        return RealtimeProtocolError(
            message=f"Client tool call {call_id!r} could not apply its output to pipeline context",
            code="client_tool_handler_unavailable",
            param="item.call_id",
            error_type="server_error",
        )
    return RealtimeProtocolError(
        message=f"Client tool call {call_id!r} was cancelled",
        code="client_tool_call_cancelled",
        param="item.call_id",
    )


def _terminal_context_error(call_id: str, outcome: _TerminalOutcome) -> RealtimeProtocolError:
    """Return the stable context-wait error for a terminal unapplied call."""
    if outcome == "cancelled":
        return RealtimeProtocolError(
            message=f"Client tool call {call_id!r} was cancelled before its output reached pipeline context",
            code="client_tool_call_cancelled",
            param="item.call_id",
            error_type="server_error",
        )
    return RealtimeProtocolError(
        message=f"Client tool call {call_id!r} did not reach pipeline context",
        code="client_tool_context_unavailable",
        param="item.call_id",
        error_type="server_error",
    )
