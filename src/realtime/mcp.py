# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Connection-scoped hosted MCP runtime for the OpenAI Realtime protocol.

The public Realtime surface remains native MCP.  Discovered MCP operations are
projected to collision-safe function names only at the local LLM boundary,
because the cascaded NVIDIA model API accepts function tools rather than MCP
server definitions.  Calls and results are translated back to MCP items and
events before they reach the Realtime client.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from loguru import logger
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, Tool
from pipecat.frames.frames import FunctionCallResultProperties
from pipecat.services.llm_service import FunctionCallParams, LLMService

from realtime.client_tools import ClientToolProjectionSnapshot, RealtimeClientToolProjection
from realtime.controller import RealtimeSessionController
from realtime.events import EmitBatchFn
from realtime.protocol import RealtimeProtocolError, build_server_event
from realtime.tool_schema import (
    compile_tool_arguments_validator,
    tool_argument_validation_failure,
    validate_tool_collection_bounds,
)
from utils import parse_env_float, parse_env_int

_ENDPOINT_FIELDS = ("server_url", "connector_id", "tunnel_id")
_REFERENCE_FIELDS = frozenset({"type", "server_label"})
_BLOCKED_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "proxy-authorization",
        "te",
        "transfer-encoding",
        "upgrade",
    }
)
_CANCELLED_CONTEXT_CALLBACK_HISTORY_LIMIT = 512


@dataclass(frozen=True, slots=True)
class MCPToolBinding:
    """One native MCP operation and its private local-model function name."""

    pipeline_name: str
    server_label: str
    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any] | None
    require_approval: bool


@dataclass(frozen=True, slots=True)
class MCPPreparedTools:
    """Tool projection frozen for one session default or response."""

    pipeline_tools: list[dict[str, Any]]
    pipeline_tool_choice: str | dict[str, Any]
    mcp_pipeline_names: frozenset[str]
    client_tool_bindings: dict[str, str]
    session_update_id: int | None = None
    response_preparation_id: int | None = None


class MCPExecutionResult(dict[str, Any]):
    """JSON-serializable MCP result retaining its exact native wire payload.

    Pipecat serializes function results into the shared LLM context with
    ``json.dumps``. A dictionary subclass keeps that boundary native while the
    private ``output`` and ``error`` attributes preserve the Realtime MCP item
    representation used after the context update is committed.
    """

    __slots__ = ("output", "error")

    def __init__(
        self,
        *,
        output: str | None,
        error: dict[str, Any] | None,
        model_payload: dict[str, Any] | None = None,
    ) -> None:
        """Create one result accepted by both Pipecat context and Realtime wire."""
        if (output is None) == (error is None):
            raise ValueError("MCP execution results require exactly one of output or error")
        if model_payload is not None:
            payload = _without_mcp_private_metadata(model_payload)
        elif output is not None:
            payload = json.loads(output)
            if not isinstance(payload, dict):
                payload = {"output": payload}
            payload = _without_mcp_private_metadata(payload)
        else:
            payload = {"error": copy.deepcopy(error)}
        super().__init__(payload)
        self.output = output
        self.error = copy.deepcopy(error)

    @property
    def failed(self) -> bool:
        """Return whether the call ended with a native MCP error."""
        return self.error is not None


@dataclass(frozen=True, slots=True)
class MCPApprovalDecision:
    """One client-authored response to an exact approval request item."""

    approve: bool
    reason: str | None


@dataclass(slots=True)
class _MCPApprovalResponseClaim:
    """One approval response reserved while its journal event reaches the wire."""

    request_id: str
    call_id: str
    decision: MCPApprovalDecision
    publication_finished: bool = False
    waiter_outcome: Literal["cancelled", "failed"] | None = None
    deferred_cancel_events: list[dict[str, Any]] | None = None


def _without_mcp_private_metadata(value: Any) -> Any:
    """Detach a model-facing MCP payload without app-only ``_meta`` values."""
    if isinstance(value, dict):
        return {key: _without_mcp_private_metadata(child) for key, child in value.items() if key != "_meta"}
    if isinstance(value, list):
        return [_without_mcp_private_metadata(child) for child in value]
    return copy.deepcopy(value)


@dataclass(slots=True)
class _MCPServerState:
    connection_definition: dict[str, Any]
    worker: _MCPServerWorker
    tools: list[Tool] | None = None


@dataclass(frozen=True, slots=True)
class _WorkerRequest:
    operation: Literal["list", "call", "close"]
    future: asyncio.Future[Any]
    name: str | None = None
    arguments: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class _BindingRegistration:
    """Rollback receipt retained until discovery publication succeeds."""

    bindings: tuple[MCPToolBinding, ...]
    previous_bindings: dict[str, MCPToolBinding]
    previous_binding_names: dict[str, str]
    previous_pipeline_identities: dict[str, str]
    previous_registered_handlers: set[str]
    controller_added: tuple[MCPToolBinding, ...]
    llm_added: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _MCPToolPreparation:
    """Exact mutable state retained until a tool projection is accepted."""

    registrations: tuple[_BindingRegistration, ...]
    new_servers: tuple[tuple[str, _MCPServerState], ...]
    previous_client_projection: ClientToolProjectionSnapshot
    prepared_client_projection: ClientToolProjectionSnapshot
    prepared_bindings: dict[str, MCPToolBinding]
    prepared_binding_names: dict[str, str]
    prepared_pipeline_identities: dict[str, str]
    prepared_registered_handlers: frozenset[str]
    prepared_servers: tuple[tuple[str, _MCPServerState], ...]


@dataclass(frozen=True, slots=True)
class _MCPSessionUpdate:
    """Prepared MCP mutations retained until the session commit boundary."""

    definitions: dict[str, dict[str, Any]]
    preparation: _MCPToolPreparation


class _MCPWorkerUnavailableError(RuntimeError):
    """Signal that a request could not complete because its worker died."""


class MCPToolPreparationPoisonedError(RuntimeError):
    """Signal that provisional tool state could not be restored exactly."""


def _configured_allowed_urls() -> frozenset[str]:
    raw = os.getenv("REALTIME_MCP_ALLOWED_SERVER_URLS", "").strip()
    if not raw:
        return frozenset()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("REALTIME_MCP_ALLOWED_SERVER_URLS must be a JSON array of exact URLs") from exc
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise RuntimeError("REALTIME_MCP_ALLOWED_SERVER_URLS must be a JSON array of exact URLs")
    return frozenset(value)


async def _validate_server_url(server_url: str) -> None:
    """Apply the deployment egress policy before opening an MCP connection."""
    parsed = urlsplit(server_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RealtimeProtocolError(
            message="MCP server_url must be an absolute HTTP or HTTPS URL",
            code="invalid_mcp_server_url",
            param="tools.server_url",
        )
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise RealtimeProtocolError(
            message="MCP server_url cannot contain credentials or a fragment",
            code="invalid_mcp_server_url",
            param="tools.server_url",
        )
    if server_url in _configured_allowed_urls():
        return
    # A resolve-then-connect hostname check is vulnerable to DNS rebinding
    # because the HTTP stack performs its own later resolution. Until the
    # transport can pin the validated address while preserving TLS SNI, only
    # operator-configured exact URLs are accepted.
    raise RealtimeProtocolError(
        message=(
            "This MCP server URL is not permitted by the gateway egress policy; "
            "add the exact URL to REALTIME_MCP_ALLOWED_SERVER_URLS"
        ),
        code="mcp_server_not_allowed",
        param="tools.server_url",
    )


def _http_headers(definition: Mapping[str, Any]) -> dict[str, str]:
    headers = copy.deepcopy(definition.get("headers") or {})
    for name in headers:
        if name.lower() in _BLOCKED_HEADERS:
            raise RealtimeProtocolError(
                message=f"MCP header {name!r} is controlled by the gateway",
                code="invalid_mcp_header",
                param="tools.headers",
            )
    authorization = definition.get("authorization")
    if isinstance(authorization, str) and authorization:
        headers["Authorization"] = f"Bearer {authorization}"
    return headers


class _MCPServerWorker:
    """Own one MCP transport in the task that entered its async contexts."""

    def __init__(self, definition: Mapping[str, Any]) -> None:
        self._definition = copy.deepcopy(dict(definition))
        self._queue: asyncio.Queue[_WorkerRequest] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._ready: asyncio.Future[None] | None = None
        self._closed = False
        self._failure: BaseException | None = None

    @property
    def dead(self) -> bool:
        """Return whether this worker can no longer accept a request."""
        return self._task is not None and self._task.done()

    def _terminal_error(self) -> BaseException:
        if self._failure is not None:
            return self._failure
        task = self._task
        if task is not None and task.cancelled():
            return _MCPWorkerUnavailableError("MCP worker was cancelled")
        if task is not None:
            error = task.exception()
            if error is not None:
                return error
        return _MCPWorkerUnavailableError("MCP worker stopped before accepting the request")

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("MCP worker is closed")
        if self.dead:
            raise self._terminal_error()
        if self._task is None:
            self._ready = asyncio.get_running_loop().create_future()
            self._task = asyncio.create_task(
                self._run(),
                name=f"realtime-mcp-{self._definition['server_label']}",
            )
        if self._ready is None:
            raise RuntimeError("MCP worker readiness state is missing")
        await asyncio.shield(self._ready)
        if self.dead:
            raise self._terminal_error()

    async def list_tools(self) -> list[Tool]:
        result = await self._request("list")
        return list(result)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        result = await self._request("call", name=name, arguments=arguments)
        if not isinstance(result, CallToolResult):
            raise RuntimeError("MCP server returned an invalid call result")
        return result

    async def close(self) -> None:
        if self._closed:
            return
        task = self._task
        if task is None:
            self._closed = True
            return
        try:
            if not task.done():
                await self._request("close")
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
            raise
        except Exception:
            if not task.done():
                task.cancel()
        finally:
            self._closed = True
        await asyncio.gather(task, return_exceptions=True)

    def shutdown(self) -> None:
        """Request task-local context teardown without blocking disconnect."""
        self._closed = True
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def _request(
        self,
        operation: Literal["list", "call", "close"],
        *,
        name: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        if self._closed and operation != "close":
            raise RuntimeError("MCP worker is closed")
        await self.start()
        if self.dead:
            raise self._terminal_error()
        future = asyncio.get_running_loop().create_future()
        # The queue is unbounded, so put_nowait plus the dead check above has
        # no scheduling gap in which a completed worker can strand a request.
        self._queue.put_nowait(_WorkerRequest(operation, future, name, copy.deepcopy(arguments)))
        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            raise

    async def _run(self) -> None:
        if self._ready is None:
            raise RuntimeError("MCP worker started without a readiness future")
        url = self._definition["server_url"]
        connect_timeout = parse_env_float("REALTIME_MCP_CONNECT_TIMEOUT_SECONDS", 10.0, min_value=1.0)
        read_timeout = parse_env_float("REALTIME_MCP_READ_TIMEOUT_SECONDS", 120.0, min_value=1.0)
        try:
            await _validate_server_url(url)
            timeout = httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=read_timeout,
                pool=connect_timeout,
            )
            async with AsyncExitStack() as stack:
                http_client = await stack.enter_async_context(
                    httpx.AsyncClient(
                        headers=_http_headers(self._definition),
                        timeout=timeout,
                        follow_redirects=False,
                        trust_env=False,
                    )
                )
                streams = await stack.enter_async_context(
                    streamable_http_client(url, http_client=http_client, terminate_on_close=True)
                )
                read_stream, write_stream, _session_id = streams
                session = await stack.enter_async_context(
                    ClientSession(
                        read_stream,
                        write_stream,
                        read_timeout_seconds=timedelta(seconds=read_timeout),
                    )
                )
                await asyncio.wait_for(session.initialize(), timeout=connect_timeout)
                if not self._ready.done():
                    self._ready.set_result(None)
                await self._serve(session)
        except asyncio.CancelledError:
            if not self._closed:
                self._failure = _MCPWorkerUnavailableError("MCP worker stopped unexpectedly")
            if not self._ready.done():
                self._ready.cancel()
            self._fail_queued(asyncio.CancelledError() if self._closed else self._failure)
            raise
        except Exception as exc:
            self._failure = exc
            if not self._ready.done():
                self._ready.set_exception(exc)
            self._fail_queued(exc)
            logger.warning(
                "Realtime MCP connection failed server_label={} error_type={}",
                self._definition.get("server_label"),
                type(exc).__name__,
            )

    async def _serve(self, session: ClientSession) -> None:
        active: set[asyncio.Task[None]] = set()
        try:
            while True:
                request = await self._queue.get()
                if request.future.done():
                    continue
                if request.operation == "close":
                    if active:
                        await asyncio.gather(*active, return_exceptions=True)
                    if not request.future.done():
                        request.future.set_result(None)
                    return
                task = asyncio.create_task(
                    self._execute_request(session, request),
                    name=f"realtime-mcp-request-{self._definition['server_label']}",
                )

                def _cancel_remote_request(
                    future: asyncio.Future[Any], *, active_task: asyncio.Task[None] = task
                ) -> None:
                    if future.cancelled() and not active_task.done():
                        active_task.cancel()

                request.future.add_done_callback(_cancel_remote_request)
                active.add(task)
                task.add_done_callback(active.discard)
        finally:
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)

    async def _execute_request(self, session: ClientSession, request: _WorkerRequest) -> None:
        """Execute one MCP request while the owning transport task stays alive."""
        try:
            if request.operation == "list":
                page = await session.list_tools()
                tools = list(page.tools)
                max_tools = parse_env_int("REALTIME_MCP_MAX_TOOLS_PER_SERVER", 128, min_value=1)
                if len(tools) > max_tools:
                    raise RuntimeError(f"MCP server advertised more than {max_tools} tools")
                while page.nextCursor:
                    page = await session.list_tools(cursor=page.nextCursor)
                    tools.extend(page.tools)
                    if len(tools) > max_tools:
                        raise RuntimeError(f"MCP server advertised more than {max_tools} tools")
                if not request.future.done():
                    request.future.set_result(tools)
            else:
                result = await session.call_tool(request.name or "", arguments=request.arguments or {})
                if not request.future.done():
                    request.future.set_result(result)
        except asyncio.CancelledError:
            if not request.future.done():
                if self._closed:
                    request.future.cancel()
                else:
                    request.future.set_exception(
                        _MCPWorkerUnavailableError("MCP worker stopped while the request was in flight")
                    )
            raise
        except Exception as exc:
            if not request.future.done():
                request.future.set_exception(exc)

    def _fail_queued(self, exc: BaseException) -> None:
        while not self._queue.empty():
            request = self._queue.get_nowait()
            if request.future.done():
                continue
            if isinstance(exc, asyncio.CancelledError):
                request.future.cancel()
            else:
                request.future.set_exception(exc)


def _pipeline_name(connection_id: str, identity: str, name: str, attempt: int) -> str:
    """Build a connection-local provider-safe name never exposed on the wire."""
    readable = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in name)
    readable = readable.strip("_-")[:30] or "tool"
    digest = hashlib.sha256(f"{connection_id}\0{identity}\0{attempt}".encode()).hexdigest()[:16]
    return f"mcp_{readable}_{digest}"


def _tool_annotations(tool: Tool) -> dict[str, Any] | None:
    if tool.annotations is None:
        return None
    return tool.annotations.model_dump(mode="json", by_alias=True, exclude_none=True)


def _matches_filter(tool: Tool, value: Mapping[str, Any]) -> bool:
    names = value.get("tool_names")
    if isinstance(names, list) and tool.name not in names:
        return False
    if "read_only" in value:
        annotations = _tool_annotations(tool) or {}
        if bool(annotations.get("readOnlyHint", False)) is not value["read_only"]:
            return False
    return True


def _select_allowed_tools(tools: list[Tool], policy: Any) -> list[Tool]:
    if policy is None:
        return list(tools)
    if isinstance(policy, list):
        requested = set(policy)
        available = {tool.name for tool in tools}
        missing = sorted(requested - available)
        if missing:
            raise RealtimeProtocolError(
                message=f"MCP allowed tool {missing[0]!r} was not advertised by the server",
                code="mcp_allowed_tool_not_found",
                param="tools.allowed_tools",
            )
        return [tool for tool in tools if tool.name in requested]
    return [tool for tool in tools if _matches_filter(tool, policy)]


def _validate_advertised_tool_names(tools: list[Tool], *, server_label: str) -> None:
    """Reject an ambiguous MCP catalog before selecting or registering it."""
    seen: set[str] = set()
    for tool in tools:
        if tool.name in seen:
            raise RealtimeProtocolError(
                message=f"MCP server {server_label!r} advertised duplicate tool name {tool.name!r}",
                code="mcp_duplicate_tool_name",
                param="tools",
            )
        seen.add(tool.name)


def _requires_approval(tool: Tool, policy: Any) -> bool:
    if policy == "always":
        return True
    if policy == "never":
        return False
    if not isinstance(policy, Mapping):
        # OpenAI Realtime defaults an omitted MCP approval policy to always.
        return True
    always = policy.get("always")
    never = policy.get("never")
    matches_always = isinstance(always, Mapping) and _matches_filter(tool, always)
    matches_never = isinstance(never, Mapping) and _matches_filter(tool, never)
    if matches_always and matches_never:
        raise RealtimeProtocolError(
            message=f"MCP tool {tool.name!r} matches both approval filters",
            code="invalid_mcp_approval_policy",
            param="tools.require_approval",
        )
    if matches_always:
        return True
    if matches_never:
        return False
    # A grouped filter explicitly selects its affected tools. Unmatched tools
    # do not inherit the omitted-policy default.
    return False


def _canonical_function_tool(binding: MCPToolBinding) -> dict[str, Any]:
    description = (f"MCP server {binding.server_label!r}, operation {binding.name!r}. {binding.description}").strip()
    return {
        "type": "function",
        "name": binding.pipeline_name,
        "description": description,
        "parameters": copy.deepcopy(binding.input_schema),
    }


def _binding_identity(
    *,
    server_label: str,
    name: str,
    description: str,
    input_schema: Mapping[str, Any],
    require_approval: bool,
) -> str:
    """Return the exact private behavior identity used for handler reuse."""
    return json.dumps(
        {
            "server_label": server_label,
            "name": name,
            "description": description,
            "input_schema": input_schema,
            "require_approval": require_approval,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _serialize_call_result(result: CallToolResult, *, max_bytes: int) -> str:
    payload = result.model_dump(mode="json", by_alias=True, exclude_none=True)
    output = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(output.encode("utf-8")) > max_bytes:
        raise ValueError(f"MCP result exceeds the {max_bytes}-byte output limit")
    return output


def _mcp_error(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, TimeoutError):
        return {
            "type": "tool_execution_error",
            "message": "The MCP tool call exceeded the configured deadline",
        }
    if isinstance(exc, httpx.HTTPStatusError):
        return {
            "type": "http_error",
            "code": exc.response.status_code,
            "message": "The MCP server returned an HTTP error",
        }
    if isinstance(exc, McpError):
        raw_error = getattr(exc, "error", None)
        code = getattr(raw_error, "code", -32603)
        message = getattr(raw_error, "message", None) or "The MCP server returned a protocol error"
        try:
            numeric_code = int(code)
        except (TypeError, ValueError, OverflowError):
            numeric_code = -32603
        return {"type": "protocol_error", "code": numeric_code, "message": str(message)}
    try:
        message = str(exc)
    except Exception:
        message = ""
    return {"type": "tool_execution_error", "message": message or "The MCP tool call failed"}


class RealtimeMCPRuntime:
    """Discover, approve, and execute native MCP tools for one WebSocket."""

    def __init__(self, *, controller: RealtimeSessionController, emit_batch: EmitBatchFn) -> None:
        """Create a connection-owned MCP registry and execution state machine."""
        self._controller = controller
        self._emit_batch = emit_batch
        self._llm: LLMService | None = None
        self.client_tool_projection = RealtimeClientToolProjection(scope_id=controller.id)
        self._servers: dict[str, _MCPServerState] = {}
        self._session_definitions: dict[str, dict[str, Any]] = {}
        self._pending_session_updates: dict[int, _MCPSessionUpdate] = {}
        self._pending_response_preparations: dict[int, _MCPToolPreparation] = {}
        self._next_session_update_id = 0
        self._next_response_preparation_id = 0
        self._bindings: dict[str, MCPToolBinding] = {}
        self._binding_names: dict[str, str] = {}
        self._pipeline_identities: dict[str, str] = {}
        self._registered_handlers: set[str] = set()
        self._discovery_views: set[str] = set()
        self._call_announced: dict[str, asyncio.Event] = {}
        self._approvals: dict[str, asyncio.Future[MCPApprovalDecision]] = {}
        self._approval_calls: dict[str, str] = {}
        self._approval_response_claims: dict[str, _MCPApprovalResponseClaim] = {}
        self._pending_results: dict[str, MCPExecutionResult] = {}
        self._cancelled_context_callbacks: OrderedDict[str, None] = OrderedDict()
        self._closed = False
        self._fatal_error_emitted = False
        self._prepare_lock = asyncio.Lock()

    def bind_llm(self, llm: LLMService) -> None:
        """Bind the one local LLM service used for private tool projection."""
        if self._llm is not None and self._llm is not llm:
            raise RuntimeError("Realtime MCP runtime cannot be rebound to another LLM")
        self._llm = llm
        self.client_tool_projection.bind_llm(llm)

    async def prepare_session(self, llm: LLMService) -> MCPPreparedTools:
        """Prepare the session's native tools before the pipeline starts."""
        self.bind_llm(llm)
        session = self._controller.session.public_view()
        prepared = await self.prepare_session_update(
            session.get("tools", []),
            session.get("tool_choice", "auto"),
        )
        try:
            self._controller.bind_session_tool_projection(
                client_tool_bindings=prepared.client_tool_bindings,
                mcp_pipeline_names=prepared.mcp_pipeline_names,
            )
        except BaseException:
            self.rollback_session_update(prepared)
            raise
        self.commit_session_update(prepared)
        return prepared

    async def prepare_session_update(
        self,
        tools: list[dict[str, Any]],
        tool_choice: Any,
    ) -> MCPPreparedTools:
        """Prepare a live session tool update without mutating session state.

        The serializer owns the atomic session/controller commit. This method
        only performs MCP discovery and builds the private provider projection
        so the caller can validate all external work before publishing
        ``session.updated``.
        """
        if self._llm is None:
            raise RuntimeError("Realtime MCP session updates require a bound LLM service")
        return await self._prepare_tools(tools, tool_choice, prepare_session_update=True)

    async def prepare_tools(self, tools: list[dict[str, Any]], tool_choice: Any) -> MCPPreparedTools:
        """Prepare response tools whose receipt must be committed or rolled back."""
        return await self._prepare_tools(tools, tool_choice, prepare_session_update=False)

    async def _prepare_tools(
        self,
        tools: list[dict[str, Any]],
        tool_choice: Any,
        *,
        prepare_session_update: bool,
    ) -> MCPPreparedTools:
        """Prepare one provider projection and optionally retain a session receipt."""
        async with self._prepare_lock:
            if self._closed:
                raise RuntimeError("Realtime MCP runtime is closed")
            prepared_definitions: dict[str, dict[str, Any]] = {}
            registrations: list[_BindingRegistration] = []
            new_servers: list[tuple[str, _MCPServerState]] = []
            previous_client_projection = self.client_tool_projection.snapshot()
            try:
                function_tools = [copy.deepcopy(tool) for tool in tools if tool.get("type") == "function"]
                client_projection = self.client_tool_projection.project_tools(
                    function_tools,
                    tool_choice,
                    trusted_names=self._controller.server_tools | self._controller.delegate_tools,
                    occupied_names=self._pipeline_identities,
                )
                pipeline_tools = client_projection.pipeline_tools
                client_tool_bindings = client_projection.bindings
                occupied_names: set[str] = set()
                for tool in pipeline_tools:
                    name = tool.get("name")
                    if not isinstance(name, str) or not name:
                        raise RealtimeProtocolError(
                            message="Function tools require a non-empty name",
                            code="invalid_tool_schema",
                            param="tools.name",
                        )
                    if name in occupied_names:
                        raise RealtimeProtocolError(
                            message=f"Duplicate function tool name {name!r}",
                            code="tool_name_conflict",
                            param="tools.name",
                        )
                    occupied_names.add(name)
                occupied_names.update(
                    pipeline_name
                    for pipeline_name, _public_name in self.client_tool_projection.snapshot().pipeline_names
                )
                active_names: set[str] = set()
                bindings_for_response: list[MCPToolBinding] = []
                for definition in tools:
                    if definition.get("type") != "mcp":
                        continue
                    resolved = await self._resolve_definition(
                        definition,
                        prepared_definitions=prepared_definitions,
                        new_servers=new_servers,
                    )
                    bindings, registration = await self._bindings_for_definition(
                        resolved,
                        occupied_names=occupied_names,
                        existing_pipeline_tools=pipeline_tools,
                    )
                    if registration is not None:
                        registrations.append(registration)
                    bindings_for_response.extend(bindings)
                    active_names.update(binding.pipeline_name for binding in bindings)
                    pipeline_tools.extend(_canonical_function_tool(binding) for binding in bindings)
                    occupied_names.update(binding.pipeline_name for binding in bindings)

                projected_choice: str | dict[str, Any]
                if isinstance(tool_choice, Mapping) and tool_choice.get("type") == "mcp":
                    server_label = tool_choice["server_label"]
                    selected = [binding for binding in bindings_for_response if binding.server_label == server_label]
                    requested_name = tool_choice.get("name")
                    if isinstance(requested_name, str):
                        selected = [binding for binding in selected if binding.name == requested_name]
                    if not selected:
                        raise RealtimeProtocolError(
                            message="The forced MCP tool is not loaded for this response",
                            code="mcp_tool_not_loaded",
                            param="response.tool_choice",
                        )
                    selected_names = {binding.pipeline_name for binding in selected}
                    pipeline_tools = [tool for tool in pipeline_tools if tool.get("name") in selected_names]
                    active_names.intersection_update(selected_names)
                    client_tool_bindings = {}
                    projected_choice = (
                        {
                            "type": "function",
                            "function": {"name": selected[0].pipeline_name},
                        }
                        if requested_name is not None
                        else "required"
                    )
                elif isinstance(tool_choice, Mapping):
                    projected_client_choice = client_projection.pipeline_tool_choice
                    if not isinstance(projected_client_choice, Mapping):
                        raise RuntimeError("A named Realtime tool choice lost its provider projection")
                    projected_choice = {
                        "type": "function",
                        "function": {"name": projected_client_choice["name"]},
                    }
                else:
                    projected_choice = str(client_projection.pipeline_tool_choice)

                if projected_choice == "required" and not pipeline_tools:
                    raise RealtimeProtocolError(
                        message="Required tool choice has no loaded eligible tools",
                        code="tool_not_loaded",
                        param="response.tool_choice",
                    )
                try:
                    validate_tool_collection_bounds(pipeline_tools, param="pipeline.tools")
                except ValueError as exc:
                    raise RealtimeProtocolError(
                        message=str(exc),
                        code="mcp_tool_limit",
                        param="tools",
                    ) from exc
            except BaseException:
                rollback_failures: list[BaseException] = []
                try:
                    self._rollback_session_preparation(
                        registrations=registrations,
                        new_servers=new_servers,
                    )
                except BaseException as exc:
                    rollback_failures.append(exc)
                try:
                    self.client_tool_projection.restore(previous_client_projection)
                except BaseException as exc:
                    rollback_failures.append(exc)
                if rollback_failures:
                    first_failure = rollback_failures[0]
                    raise MCPToolPreparationPoisonedError(
                        "Could not rollback a failed Realtime tool preparation"
                    ) from first_failure
                raise

            preparation = _MCPToolPreparation(
                registrations=tuple(registrations),
                new_servers=tuple(new_servers),
                previous_client_projection=previous_client_projection,
                prepared_client_projection=self.client_tool_projection.snapshot(),
                prepared_bindings=dict(self._bindings),
                prepared_binding_names=dict(self._binding_names),
                prepared_pipeline_identities=dict(self._pipeline_identities),
                prepared_registered_handlers=frozenset(self._registered_handlers),
                prepared_servers=tuple(sorted(self._servers.items())),
            )
            session_update_id: int | None = None
            response_preparation_id: int | None = None
            if prepare_session_update:
                self._next_session_update_id += 1
                session_update_id = self._next_session_update_id
                self._pending_session_updates[session_update_id] = _MCPSessionUpdate(
                    definitions=copy.deepcopy(prepared_definitions),
                    preparation=preparation,
                )
            else:
                self._next_response_preparation_id += 1
                response_preparation_id = self._next_response_preparation_id
                self._pending_response_preparations[response_preparation_id] = preparation
            return MCPPreparedTools(
                pipeline_tools=pipeline_tools,
                pipeline_tool_choice=projected_choice,
                mcp_pipeline_names=frozenset(active_names),
                client_tool_bindings=client_tool_bindings,
                session_update_id=session_update_id,
                response_preparation_id=response_preparation_id,
            )

    def commit_session_update(self, prepared: MCPPreparedTools) -> None:
        """Publish one prepared MCP definition set as the session default."""
        update_id = prepared.session_update_id
        if update_id is None:
            raise ValueError("MCP projection is not a prepared session update")
        update = self._pending_session_updates.get(update_id)
        if update is None:
            raise RuntimeError("MCP session update is no longer pending")
        try:
            self._validate_prepared_tool_state(update.preparation)
        except BaseException as exc:
            raise MCPToolPreparationPoisonedError("Realtime session tool preparation changed before commit") from exc
        definitions = copy.deepcopy(update.definitions)
        self._session_definitions = definitions
        del self._pending_session_updates[update_id]

    def rollback_session_update(self, prepared: MCPPreparedTools) -> None:
        """Discard one uncommitted MCP session update and its registrations."""
        update_id = prepared.session_update_id
        if update_id is None:
            raise ValueError("MCP projection is not a prepared session update")
        update = self._pending_session_updates.pop(update_id, None)
        if update is None:
            raise RuntimeError("MCP session update is no longer pending")
        failures: list[BaseException] = []
        try:
            self._rollback_tool_preparation(update.preparation)
        except BaseException as exc:
            failures.append(exc)
        if failures:
            raise MCPToolPreparationPoisonedError(
                "Could not rollback a Realtime session tool preparation"
            ) from failures[0]

    def commit_response_tools(self, prepared: MCPPreparedTools) -> None:
        """Retain response-scoped tool state after response ownership is accepted."""
        preparation_id = prepared.response_preparation_id
        if preparation_id is None:
            raise ValueError("MCP projection is not a prepared response")
        preparation = self._pending_response_preparations.get(preparation_id)
        if preparation is None:
            raise RuntimeError("MCP response tool preparation is no longer pending")
        try:
            self._validate_prepared_tool_state(preparation)
        except BaseException as exc:
            raise MCPToolPreparationPoisonedError("Realtime response tool preparation changed before commit") from exc
        del self._pending_response_preparations[preparation_id]

    def rollback_response_tools(self, prepared: MCPPreparedTools) -> None:
        """Discard response-scoped tool state before response ownership is accepted."""
        preparation_id = prepared.response_preparation_id
        if preparation_id is None:
            raise ValueError("MCP projection is not a prepared response")
        preparation = self._pending_response_preparations.pop(preparation_id, None)
        if preparation is None:
            raise RuntimeError("MCP response tool preparation is no longer pending")
        try:
            self._rollback_tool_preparation(preparation)
        except BaseException as exc:
            if isinstance(exc, MCPToolPreparationPoisonedError):
                raise
            raise MCPToolPreparationPoisonedError("Could not rollback a Realtime response tool preparation") from exc

    def _validate_prepared_tool_state(self, preparation: _MCPToolPreparation) -> None:
        """Prove no later owner changed state covered by a preparation receipt."""
        if self.client_tool_projection.snapshot() != preparation.prepared_client_projection:
            raise RuntimeError("Realtime client tool projection changed after preparation")
        if self._bindings != preparation.prepared_bindings:
            raise RuntimeError("Realtime MCP bindings changed after preparation")
        if self._binding_names != preparation.prepared_binding_names:
            raise RuntimeError("Realtime MCP binding names changed after preparation")
        if self._pipeline_identities != preparation.prepared_pipeline_identities:
            raise RuntimeError("Realtime MCP pipeline identities changed after preparation")
        if self._registered_handlers != set(preparation.prepared_registered_handlers):
            raise RuntimeError("Realtime MCP handler registry changed after preparation")
        if len(self._servers) != len(preparation.prepared_servers) or any(
            self._servers.get(label) is not state for label, state in preparation.prepared_servers
        ):
            raise RuntimeError("Realtime MCP server registry changed after preparation")

    def _rollback_tool_preparation(self, preparation: _MCPToolPreparation) -> None:
        """Restore every mutable owner covered by a validated preparation receipt."""
        try:
            self._validate_prepared_tool_state(preparation)
        except BaseException as exc:
            raise MCPToolPreparationPoisonedError("Realtime tool preparation changed before rollback") from exc
        failures: list[BaseException] = []
        try:
            self._rollback_session_preparation(
                registrations=list(preparation.registrations),
                new_servers=list(preparation.new_servers),
            )
        except BaseException as exc:
            failures.append(exc)
        try:
            self.client_tool_projection.restore(
                preparation.previous_client_projection,
                expected=preparation.prepared_client_projection,
            )
        except BaseException as exc:
            failures.append(exc)
        if failures:
            raise MCPToolPreparationPoisonedError("Could not restore a Realtime tool preparation") from failures[0]

    def _rollback_session_preparation(
        self,
        *,
        registrations: list[_BindingRegistration],
        new_servers: list[tuple[str, _MCPServerState]],
    ) -> None:
        """Restore all connection-owned state mutated during preparation."""
        failures: list[BaseException] = []
        for registration in reversed(registrations):
            try:
                self._rollback_binding_registration(registration)
            except Exception as exc:
                failures.append(exc)
        for label, state in reversed(new_servers):
            if self._servers.get(label) is not state:
                failures.append(RuntimeError(f"MCP server state {label!r} changed during rollback"))
                continue
            try:
                state.worker.shutdown()
            except Exception as exc:
                failures.append(exc)
            self._servers.pop(label, None)
        if failures:
            raise MCPToolPreparationPoisonedError("Could not rollback an MCP tool preparation") from failures[0]

    async def _resolve_definition(
        self,
        raw: Mapping[str, Any],
        *,
        prepared_definitions: dict[str, dict[str, Any]],
        new_servers: list[tuple[str, _MCPServerState]],
    ) -> dict[str, Any]:
        label = str(raw["server_label"])
        endpoint_fields = [field for field in _ENDPOINT_FIELDS if raw.get(field) is not None]
        if not endpoint_fields:
            if set(raw) != _REFERENCE_FIELDS:
                raise RealtimeProtocolError(
                    message="A reused MCP server_label must be referenced without partial overrides",
                    code="invalid_mcp_reference",
                    param="tools",
                )
            definition = prepared_definitions.get(label)
            if definition is None:
                definition = self._session_definitions.get(label)
            if definition is None:
                raise RealtimeProtocolError(
                    message=f"MCP server_label {label!r} has not been defined in this session",
                    code="mcp_server_not_defined",
                    param="tools.server_label",
                )
            prepared_definitions[label] = copy.deepcopy(definition)
            return copy.deepcopy(definition)

        if raw.get("connector_id") is not None:
            raise RealtimeProtocolError(
                message="OpenAI-managed MCP connectors are not available on this self-hosted gateway",
                code="unsupported_capability",
                param="tools.connector_id",
            )
        if raw.get("tunnel_id") is not None:
            raise RealtimeProtocolError(
                message="OpenAI Secure MCP tunnel IDs are not available on this self-hosted gateway",
                code="unsupported_capability",
                param="tools.tunnel_id",
            )
        if raw.get("defer_loading") is True:
            raise RealtimeProtocolError(
                message="Deferred MCP loading requires tool-search support that is not installed",
                code="unsupported_capability",
                param="tools.defer_loading",
            )
        callers = raw.get("allowed_callers")
        if isinstance(callers, list) and any(caller != "direct" for caller in callers):
            raise RealtimeProtocolError(
                message="Programmatic MCP callers are not available on this model backend",
                code="unsupported_capability",
                param="tools.allowed_callers",
            )
        await _validate_server_url(str(raw["server_url"]))
        definition = copy.deepcopy(dict(raw))
        state = self._servers.get(label)
        if state is not None:
            identity_fields = ("server_url", "connector_id", "tunnel_id", "authorization", "headers")
            if any(state.connection_definition.get(field) != definition.get(field) for field in identity_fields):
                raise RealtimeProtocolError(
                    message=f"MCP server_label {label!r} cannot be rebound to another endpoint or credential",
                    code="mcp_server_label_conflict",
                    param="tools.server_label",
                )
            prepared_definitions[label] = copy.deepcopy(definition)
            return definition
        if len(self._servers) >= parse_env_int("REALTIME_MCP_MAX_SERVERS", 8, min_value=1):
            raise RealtimeProtocolError(
                message="This Realtime connection has reached its MCP server limit",
                code="mcp_server_limit",
                param="tools",
            )
        state = _MCPServerState(
            connection_definition=copy.deepcopy(definition),
            worker=_MCPServerWorker(definition),
        )
        self._servers[label] = state
        new_servers.append((label, state))
        prepared_definitions[label] = copy.deepcopy(definition)
        return definition

    async def _bindings_for_definition(
        self,
        definition: Mapping[str, Any],
        *,
        occupied_names: set[str],
        existing_pipeline_tools: list[dict[str, Any]],
    ) -> tuple[list[MCPToolBinding], _BindingRegistration | None]:
        label = str(definition["server_label"])
        state = self._servers[label]
        discovery_key = json.dumps(
            {
                "server_label": label,
                "allowed_tools": definition.get("allowed_tools"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if state.tools is None or discovery_key not in self._discovery_views:
            added = self._controller.conversation.add_item(
                {
                    "type": "mcp_list_tools",
                    "status": "in_progress",
                    "server_label": label,
                    "tools": [],
                }
            )
            list_item_id = added["item"]["id"]
            await self._emit_batch(
                [
                    added,
                    build_server_event("mcp_list_tools.in_progress", item_id=list_item_id),
                ]
            )
            registration: _BindingRegistration | None = None
            try:
                discovered_tools = state.tools
                if state.tools is None:
                    worker = state.worker
                    try:
                        await worker.start()
                        discovered_tools = await asyncio.wait_for(
                            worker.list_tools(),
                            timeout=parse_env_float(
                                "REALTIME_MCP_DISCOVERY_TIMEOUT_SECONDS",
                                20.0,
                                min_value=1.0,
                            ),
                        )
                    except Exception:
                        self._replace_worker_if_dead(state, worker)
                        raise
                if discovered_tools is None:
                    raise RuntimeError("MCP discovery completed without a tool catalog")
                _validate_advertised_tool_names(discovered_tools, server_label=label)
                selected = _select_allowed_tools(discovered_tools, definition.get("allowed_tools"))
                registration = self._register_bindings_atomic(
                    definition,
                    selected,
                    occupied_names=occupied_names,
                    existing_pipeline_tools=existing_pipeline_tools,
                )
                public_tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": copy.deepcopy(tool.inputSchema),
                        **({"annotations": _tool_annotations(tool)} if _tool_annotations(tool) is not None else {}),
                    }
                    for tool in selected
                ]
                done = self._controller.conversation.complete_item(
                    list_item_id,
                    item_patch={"tools": public_tools},
                )
                await self._emit_batch(
                    [
                        done,
                        build_server_event("mcp_list_tools.completed", item_id=list_item_id),
                    ]
                )
                state.tools = list(discovered_tools)
                self._discovery_views.add(discovery_key)
            except asyncio.CancelledError:
                if registration is not None:
                    self._rollback_binding_registration(registration)
                raise
            except MCPToolPreparationPoisonedError:
                raise
            except Exception as exc:
                if registration is not None:
                    self._rollback_binding_registration(registration)
                try:
                    done = self._controller.conversation.complete_item(list_item_id, status="incomplete")
                    failed_events = [
                        done,
                        build_server_event("mcp_list_tools.failed", item_id=list_item_id),
                    ]
                except Exception:
                    # Completion may already have been journaled if publication
                    # itself failed. The terminal list event still tells a live
                    # client that this discovery view is unusable.
                    failed_events = [build_server_event("mcp_list_tools.failed", item_id=list_item_id)]
                await self._emit_batch(failed_events)
                logger.warning(
                    "Realtime MCP discovery failed server_label={} error_type={}",
                    label,
                    type(exc).__name__,
                )
                self._replace_worker_if_dead(state, state.worker)
                return [], None
            return (list(registration.bindings), registration) if registration is not None else ([], None)
        cached_tools = state.tools or []
        _validate_advertised_tool_names(cached_tools, server_label=label)
        selected = _select_allowed_tools(cached_tools, definition.get("allowed_tools"))
        registration = self._register_bindings_atomic(
            definition,
            selected,
            occupied_names=occupied_names,
            existing_pipeline_tools=existing_pipeline_tools,
        )
        return list(registration.bindings), registration

    @staticmethod
    def _replace_worker_if_dead(state: _MCPServerState, worker: _MCPServerWorker) -> None:
        """Install a fresh worker for future requests without retrying this one."""
        if state.worker is worker and worker.dead:
            worker.shutdown()
            state.worker = _MCPServerWorker(state.connection_definition)

    def _register_bindings_atomic(
        self,
        definition: Mapping[str, Any],
        selected: list[Tool],
        *,
        occupied_names: set[str],
        existing_pipeline_tools: list[dict[str, Any]],
    ) -> _BindingRegistration:
        """Preflight and register a complete discovery view as one transaction."""
        if self._llm is None:
            raise RuntimeError("Realtime MCP tools require a bound LLM service")
        label = str(definition["server_label"])
        _validate_advertised_tool_names(selected, server_label=label)
        prepared: list[tuple[Tool, dict[str, Any], str, dict[str, Any] | None, bool, str]] = []
        for tool in selected:
            schema = copy.deepcopy(tool.inputSchema)
            try:
                compile_tool_arguments_validator(schema)
            except ValueError as exc:
                raise RealtimeProtocolError(
                    message=f"MCP tool {label}.{tool.name} exposes an invalid input schema",
                    code="invalid_tool_schema",
                    param="tools",
                ) from exc
            description = " ".join(
                part
                for part in (
                    definition.get("server_description"),
                    tool.description,
                )
                if isinstance(part, str) and part
            )
            annotations = _tool_annotations(tool)
            require_approval = _requires_approval(tool, definition.get("require_approval"))
            identity = _binding_identity(
                server_label=label,
                name=tool.name,
                description=description,
                input_schema=schema,
                require_approval=require_approval,
            )
            prepared.append((tool, schema, description, annotations, require_approval, identity))

        planned_binding_names = dict(self._binding_names)
        planned_pipeline_identities = dict(self._pipeline_identities)
        planned_bindings = dict(self._bindings)
        planned_registered_handlers = set(self._registered_handlers)
        bindings: list[MCPToolBinding] = []
        allocated_names = set(occupied_names)
        for tool, schema, description, annotations, require_approval, identity in prepared:
            pipeline_name = self._allocate_pipeline_name(
                identity=identity,
                tool_name=tool.name,
                occupied_names=allocated_names,
                binding_names=planned_binding_names,
                pipeline_identities=planned_pipeline_identities,
            )
            allocated_names.add(pipeline_name)
            binding = MCPToolBinding(
                pipeline_name=pipeline_name,
                server_label=label,
                name=tool.name,
                description=description,
                input_schema=schema,
                annotations=annotations,
                require_approval=require_approval,
            )
            existing = planned_bindings.get(pipeline_name)
            if existing is not None and existing != binding:
                raise RealtimeProtocolError(
                    message=f"Private MCP tool name {pipeline_name!r} conflicts with an existing binding",
                    code="tool_name_conflict",
                    param="tools",
                )
            planned_bindings[pipeline_name] = binding
            bindings.append(binding)

        try:
            validate_tool_collection_bounds(
                [
                    *existing_pipeline_tools,
                    *(_canonical_function_tool(binding) for binding in bindings),
                ],
                param="pipeline.tools",
            )
        except ValueError as exc:
            raise RealtimeProtocolError(
                message=str(exc),
                code="mcp_tool_limit",
                param="tools",
            ) from exc

        function_registry = getattr(self._llm, "_functions", None)
        for binding in bindings:
            if binding.pipeline_name in planned_registered_handlers:
                continue
            if isinstance(function_registry, Mapping) and binding.pipeline_name in function_registry:
                raise RealtimeProtocolError(
                    message=f"Private MCP tool name {binding.pipeline_name!r} conflicts with an LLM handler",
                    code="tool_name_conflict",
                    param="tools",
                )
            planned_registered_handlers.add(binding.pipeline_name)

        receipt = _BindingRegistration(
            bindings=tuple(bindings),
            previous_bindings=self._bindings,
            previous_binding_names=self._binding_names,
            previous_pipeline_identities=self._pipeline_identities,
            previous_registered_handlers=self._registered_handlers,
            controller_added=(),
            llm_added=(),
        )
        controller_added: list[MCPToolBinding] = []
        llm_added: list[str] = []
        try:
            for binding in bindings:
                added = self._controller.register_mcp_tool_binding(
                    pipeline_name=binding.pipeline_name,
                    server_label=binding.server_label,
                    name=binding.name,
                )
                if added:
                    controller_added.append(binding)
            for binding in bindings:
                if binding.pipeline_name in self._registered_handlers:
                    continue
                try:
                    self._llm.register_function(
                        binding.pipeline_name,
                        self.handle_tool_call,
                        # Keep the completed result in the standard assistant
                        # tool-call / tool-result pair used by cascaded LLMs.
                        # ``run_llm=False`` leaves Response B under explicit
                        # Realtime client control. Barge-in cancels media, not
                        # this durable MCP conversation item.
                        cancel_on_interruption=False,
                    )
                except Exception:
                    current_registry = getattr(self._llm, "_functions", None)
                    if isinstance(current_registry, Mapping) and binding.pipeline_name in current_registry:
                        llm_added.append(binding.pipeline_name)
                    raise
                llm_added.append(binding.pipeline_name)
        except Exception:
            failed_receipt = _BindingRegistration(
                bindings=receipt.bindings,
                previous_bindings=receipt.previous_bindings,
                previous_binding_names=receipt.previous_binding_names,
                previous_pipeline_identities=receipt.previous_pipeline_identities,
                previous_registered_handlers=receipt.previous_registered_handlers,
                controller_added=tuple(controller_added),
                llm_added=tuple(llm_added),
            )
            self._rollback_binding_registration(failed_receipt)
            raise

        self._bindings = planned_bindings
        self._binding_names = planned_binding_names
        self._pipeline_identities = planned_pipeline_identities
        self._registered_handlers = planned_registered_handlers
        return _BindingRegistration(
            bindings=receipt.bindings,
            previous_bindings=receipt.previous_bindings,
            previous_binding_names=receipt.previous_binding_names,
            previous_pipeline_identities=receipt.previous_pipeline_identities,
            previous_registered_handlers=receipt.previous_registered_handlers,
            controller_added=tuple(controller_added),
            llm_added=tuple(llm_added),
        )

    def _rollback_binding_registration(self, receipt: _BindingRegistration) -> None:
        """Undo every external and local mutation from one registration."""
        failures: list[BaseException] = []
        if self._llm is not None:
            for pipeline_name in reversed(receipt.llm_added):
                try:
                    self._llm.unregister_function(pipeline_name)
                    suppressed = getattr(self._llm, "_explicitly_unregistered_function_names", None)
                    if isinstance(suppressed, set):
                        suppressed.discard(pipeline_name)
                except Exception as exc:
                    failures.append(exc)
        for binding in reversed(receipt.controller_added):
            try:
                self._controller.unregister_mcp_tool_binding(
                    pipeline_name=binding.pipeline_name,
                    server_label=binding.server_label,
                    name=binding.name,
                )
            except Exception as exc:
                failures.append(exc)
        self._bindings = receipt.previous_bindings
        self._binding_names = receipt.previous_binding_names
        self._pipeline_identities = receipt.previous_pipeline_identities
        self._registered_handlers = receipt.previous_registered_handlers
        if failures:
            raise MCPToolPreparationPoisonedError(
                "Could not rollback an incomplete MCP binding registration"
            ) from failures[0]

    def _allocate_pipeline_name(
        self,
        *,
        identity: str,
        tool_name: str,
        occupied_names: set[str],
        binding_names: dict[str, str],
        pipeline_identities: dict[str, str],
    ) -> str:
        cached = binding_names.get(identity)
        if cached is not None and cached not in occupied_names:
            return cached
        max_bindings = parse_env_int("REALTIME_MCP_MAX_BINDINGS", 512, min_value=1)
        for attempt in range(max_bindings + 1):
            candidate = _pipeline_name(self._controller.id, identity, tool_name, attempt)
            if candidate in occupied_names:
                continue
            existing_identity = pipeline_identities.get(candidate)
            if existing_identity is not None and existing_identity != identity:
                continue
            if existing_identity is None and len(pipeline_identities) >= max_bindings:
                raise RealtimeProtocolError(
                    message="This Realtime connection has reached its MCP binding limit",
                    code="mcp_tool_limit",
                    param="tools",
                )
            binding_names[identity] = candidate
            pipeline_identities[candidate] = identity
            return candidate
        raise RealtimeProtocolError(
            message=f"Could not allocate a private MCP binding for {tool_name!r}",
            code="tool_name_conflict",
            param="tools",
        )

    def notify_call_announced(self, call_id: str) -> None:
        """Release execution after the call's native wire item is published."""
        self._call_announced.setdefault(call_id, asyncio.Event()).set()

    async def handle_tool_call(self, params: FunctionCallParams) -> None:
        """Resolve every non-disconnect outcome through Pipecat's result path."""
        try:
            result = await self._execute_tool_call(params)
        except asyncio.CancelledError:
            self._call_announced.pop(params.tool_call_id, None)
            raise

        self._pending_results[params.tool_call_id] = result

        async def _after_context_update() -> None:
            try:
                await self._complete_call_after_context(params.tool_call_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._raise_fatal_lifecycle_error(
                    call_id=params.tool_call_id,
                    stage="context commit",
                    cause=exc,
                )

        try:
            await params.result_callback(
                result,
                properties=FunctionCallResultProperties(
                    run_llm=False,
                    on_context_updated=_after_context_update,
                ),
            )
        except asyncio.CancelledError:
            pending_result = self._pending_results.pop(params.tool_call_id, None)
            record = self._controller.tool_call(params.tool_call_id)
            if (
                pending_result is not None
                and record is not None
                and record.owner == "mcp"
                and not record.completed
                and not record.retired
            ):
                self._remember_cancelled_context_callback(params.tool_call_id)
            raise
        except Exception as exc:
            self._pending_results.pop(params.tool_call_id, None)
            await self._raise_fatal_lifecycle_error(
                call_id=params.tool_call_id,
                stage="result callback",
                cause=exc,
            )
        finally:
            self._call_announced.pop(params.tool_call_id, None)

    async def _execute_tool_call(self, params: FunctionCallParams) -> MCPExecutionResult:
        """Return exactly one typed result for the complete pre-callback lifecycle."""
        approval_request_id: str | None = None
        try:
            if self._closed:
                raise ConnectionError("Realtime MCP runtime is closed")
            binding = self._bindings.get(params.function_name)
            if binding is None:
                raise RuntimeError(f"Unknown MCP pipeline function {params.function_name!r}")
            announced = self._call_announced.setdefault(params.tool_call_id, asyncio.Event())
            await asyncio.wait_for(
                announced.wait(),
                timeout=parse_env_float(
                    "REALTIME_MCP_CALL_ANNOUNCE_TIMEOUT_SECONDS",
                    10.0,
                    min_value=1.0,
                ),
            )
            record = self._controller.tool_call(params.tool_call_id)
            if record is None or record.owner != "mcp":
                raise RuntimeError("MCP call was not correlated to a native Realtime item")

            validator = compile_tool_arguments_validator(binding.input_schema)
            validation_failure = tool_argument_validation_failure(validator, params.arguments)
            if validation_failure is not None:
                raise ValueError(validation_failure.message)
            if binding.require_approval:
                approval_request_id, events = self._controller.request_mcp_approval(params.tool_call_id)
                future: asyncio.Future[MCPApprovalDecision] = asyncio.get_running_loop().create_future()
                self._approvals[approval_request_id] = future
                self._approval_calls[approval_request_id] = params.tool_call_id
                await self._emit_batch(events)
                try:
                    try:
                        decision = await asyncio.wait_for(
                            asyncio.shield(future),
                            timeout=parse_env_float(
                                "REALTIME_MCP_APPROVAL_TIMEOUT_SECONDS",
                                300.0,
                                min_value=1.0,
                            ),
                        )
                    except TimeoutError:
                        # The timeout wins if no response was accepted. Once a
                        # response is claimed, wait for its wire publication;
                        # the connection/task cancellation still bounds this.
                        if approval_request_id not in self._approval_response_claims:
                            raise
                        decision = await asyncio.shield(future)
                except asyncio.CancelledError:
                    self._mark_approval_waiter_stopped(approval_request_id, cancelled=True)
                    raise
                except BaseException:
                    self._mark_approval_waiter_stopped(approval_request_id, cancelled=False)
                    raise
                finally:
                    self._approvals.pop(approval_request_id, None)
                    self._approval_calls.pop(approval_request_id, None)
                if not decision.approve:
                    reason = decision.reason or "The MCP tool call was rejected"
                    raise PermissionError(reason)

            await self._emit_batch(self._controller.mark_mcp_call_in_progress(params.tool_call_id))
            state = self._servers[binding.server_label]
            worker = state.worker
            try:
                raw_result = await asyncio.wait_for(
                    worker.call_tool(binding.name, copy.deepcopy(params.arguments)),
                    timeout=parse_env_float("REALTIME_MCP_CALL_TIMEOUT_SECONDS", 120.0, min_value=1.0),
                )
            except Exception:
                # Never replay this call: a transport failure after enqueue is
                # ambiguous. A dead worker is replaced only for a later call.
                self._replace_worker_if_dead(state, worker)
                raise
            max_bytes = parse_env_int("REALTIME_MCP_MAX_OUTPUT_BYTES", 262_144, min_value=1024)
            output = _serialize_call_result(raw_result, max_bytes=max_bytes)
            if raw_result.isError:
                # Realtime clients are the MCP host/application and retain the
                # exact native result. The model receives the same failure
                # content through MCP's separate, recursively redacted view.
                model_output = json.dumps(
                    _without_mcp_private_metadata(json.loads(output)),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                return MCPExecutionResult(
                    output=None,
                    error={"type": "tool_execution_error", "message": output},
                    model_payload={
                        "error": {
                            "type": "tool_execution_error",
                            "message": model_output,
                        }
                    },
                )
            return MCPExecutionResult(output=output, error=None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Realtime MCP call failed call_id={} error_type={}",
                params.tool_call_id,
                type(exc).__name__,
            )
            return MCPExecutionResult(output=None, error=_mcp_error(exc))
        finally:
            if approval_request_id is not None:
                future = self._approvals.pop(approval_request_id, None)
                self._approval_calls.pop(approval_request_id, None)
                if future is not None and not future.done():
                    future.cancel()

    async def _complete_call_after_context(self, call_id: str) -> None:
        """Publish MCP terminal events only after Pipecat committed the result."""
        result = self._pending_results.get(call_id)
        if result is None:
            if self._closed:
                return
            if self._consume_cancelled_context_callback(call_id):
                return
            raise RuntimeError("MCP context callback has no pending typed result")
        record = self._controller.tool_call(call_id)
        if record is None or record.owner != "mcp":
            raise RuntimeError("MCP context callback is not correlated to a native call")
        if record.completed or record.retired:
            if self._consume_cancelled_context_callback(call_id):
                self._pending_results.pop(call_id, None)
                return
            raise RuntimeError("MCP context callback targeted an already-terminal call")
        await self._emit_batch(self.complete_call_events(call_id, result))
        self._pending_results.pop(call_id, None)

    def _remember_cancelled_context_callback(self, call_id: str) -> None:
        """Remember one callback discarded by an explicit Pipecat cancellation."""
        self._cancelled_context_callbacks[call_id] = None
        self._cancelled_context_callbacks.move_to_end(call_id)
        while len(self._cancelled_context_callbacks) > _CANCELLED_CONTEXT_CALLBACK_HISTORY_LIMIT:
            self._cancelled_context_callbacks.popitem(last=False)

    def _consume_cancelled_context_callback(self, call_id: str) -> bool:
        """Consume exactly one explicitly cancelled late-callback allowance."""
        if call_id not in self._cancelled_context_callbacks:
            return False
        self._cancelled_context_callbacks.pop(call_id, None)
        return True

    def _mark_approval_waiter_stopped(self, request_id: str, *, cancelled: bool) -> None:
        """Record that Pipecat can no longer consume a staged approval decision."""
        claim = self._approval_response_claims.get(request_id)
        if claim is None:
            return
        if cancelled or claim.waiter_outcome is None:
            claim.waiter_outcome = "cancelled" if cancelled else "failed"

    async def _raise_fatal_lifecycle_error(
        self,
        *,
        call_id: str,
        stage: str,
        cause: BaseException,
    ) -> None:
        """Fail all live MCP items, disable the runtime, emit, then raise."""
        message = f"MCP {stage} failed; the Realtime connection cannot continue safely"
        events: list[dict[str, Any]] = []
        if not self._fatal_error_emitted:
            self._fatal_error_emitted = True
            terminal_error = {
                "type": "tool_execution_error",
                "message": "The MCP lifecycle failed before its result could be committed safely",
            }
            for pending_call_id in self._controller.pending_tool_call_ids():
                record = self._controller.tool_call(pending_call_id)
                if record is None or record.owner != "mcp":
                    continue
                try:
                    events.extend(
                        self._controller.complete_mcp_call(
                            pending_call_id,
                            output=None,
                            error=copy.deepcopy(terminal_error),
                        )
                    )
                except Exception:
                    logger.exception("Could not terminalize MCP call during fatal lifecycle failure")
            events.append(
                RealtimeProtocolError(
                    message=message,
                    code="mcp_lifecycle_error",
                    param="tools",
                    error_type="server_error",
                ).to_event()
            )
        self.shutdown()
        if events:
            try:
                await self._emit_batch(events)
            except Exception:
                logger.exception("Could not publish fatal MCP lifecycle error")
        raise RuntimeError(f"{message} (call_id={call_id})") from cause

    def cancel_call_events(self, call_id: str) -> list[dict[str, Any]]:
        """Cancel one active native call and release its approval/result state."""
        record = self._controller.tool_call(call_id)
        pending_result = self._pending_results.pop(call_id, None)
        if (
            pending_result is not None
            and record is not None
            and record.owner == "mcp"
            and not record.completed
            and not record.retired
        ):
            self._remember_cancelled_context_callback(call_id)
        self._call_announced.pop(call_id, None)
        claim = next(
            (
                pending_claim
                for pending_claim in self._approval_response_claims.values()
                if pending_claim.call_id == call_id
            ),
            None,
        )
        if claim is not None:
            claim.waiter_outcome = "cancelled"
        for request_id, approval_call_id in tuple(self._approval_calls.items()):
            if approval_call_id != call_id:
                continue
            future = self._approvals.pop(request_id, None)
            self._approval_calls.pop(request_id, None)
            if future is not None and not future.done():
                future.cancel()
        events: list[dict[str, Any]] = []
        if record is None or record.completed or record.retired:
            events = []
        else:
            events = self._controller.complete_mcp_call(
                call_id,
                output=None,
                error={
                    "type": "tool_execution_error",
                    "message": "The MCP tool call was cancelled before completion",
                },
            )
        if claim is None:
            return events
        if claim.publication_finished:
            self._approval_response_claims.pop(claim.request_id, None)
            return [*(claim.deferred_cancel_events or []), *events]
        if events and claim.deferred_cancel_events is None:
            claim.deferred_cancel_events = events
        return []

    def validate_approval_response(self, item: Mapping[str, Any]) -> None:
        """Validate that a client approval targets one pending request."""
        request_id = item.get("approval_request_id")
        if isinstance(request_id, str) and request_id in self._approval_response_claims:
            raise RealtimeProtocolError(
                message=f"MCP approval request {request_id!r} already has a response",
                code="duplicate_mcp_approval",
                param="item.approval_request_id",
            )
        future = self._approvals.get(request_id) if isinstance(request_id, str) else None
        if future is None:
            raise RealtimeProtocolError(
                message=f"MCP approval request {request_id!r} was not found or is no longer active",
                code="mcp_approval_not_found",
                param="item.approval_request_id",
            )
        if future.done():
            raise RealtimeProtocolError(
                message=f"MCP approval request {request_id!r} already has a response",
                code="duplicate_mcp_approval",
                param="item.approval_request_id",
            )

    def claim_approval_response(self, item: Mapping[str, Any]) -> str:
        """Reserve one response without waking execution before wire publication."""
        self.validate_approval_response(item)
        request_id = str(item["approval_request_id"])
        call_id = self._approval_calls.get(request_id)
        if call_id is None:
            raise RuntimeError("MCP approval response has no correlated call")
        self._approval_response_claims[request_id] = _MCPApprovalResponseClaim(
            request_id=request_id,
            call_id=call_id,
            decision=MCPApprovalDecision(
                approve=bool(item["approve"]),
                reason=item.get("reason") if isinstance(item.get("reason"), str) else None,
            ),
        )
        return request_id

    def resolve_approval_response(self, request_id: str) -> list[dict[str, Any]]:
        """Commit a claimed decision after its conversation item reached the wire."""
        claim = self._approval_response_claims.get(request_id)
        if claim is None:
            if self._closed:
                return []
            raise RuntimeError("MCP approval response claim is no longer active")
        claim.publication_finished = True
        deferred_events = claim.deferred_cancel_events or []
        if claim.waiter_outcome == "cancelled":
            if deferred_events:
                self._approval_response_claims.pop(request_id, None)
            return deferred_events
        if claim.waiter_outcome == "failed":
            self._approval_response_claims.pop(request_id, None)
            return deferred_events
        future = self._approvals.get(request_id)
        if future is None or future.done():
            if self._closed:
                self._approval_response_claims.pop(request_id, None)
                return deferred_events
            raise RuntimeError("MCP approval waiter disappeared before publication completed")
        future.set_result(claim.decision)
        self._approval_response_claims.pop(request_id, None)
        return deferred_events

    def abandon_approval_response(self, request_id: str, *, cancel_call: bool) -> list[dict[str, Any]]:
        """Release a failed approval transaction and optionally terminalize its call."""
        claim = self._approval_response_claims.pop(request_id, None)
        if claim is None:
            return []
        deferred_events = claim.deferred_cancel_events or []
        if not cancel_call:
            return deferred_events
        future = self._approvals.pop(request_id, None)
        self._approval_calls.pop(request_id, None)
        if future is not None and not future.done():
            future.cancel()
        return [*deferred_events, *self.cancel_call_events(claim.call_id)]

    def complete_call_events(self, call_id: str, result: MCPExecutionResult) -> list[dict[str, Any]]:
        """Translate a typed adapter result to native MCP terminal events."""
        return self._controller.complete_mcp_call(
            call_id,
            output=result.output,
            error=copy.deepcopy(result.error),
        )

    def shutdown(self) -> None:
        """Retire pending calls and close every connection-owned worker."""
        self._closed = True
        for call_id in self._controller.pending_tool_call_ids():
            record = self._controller.tool_call(call_id)
            if record is None or record.owner != "mcp":
                continue
            try:
                self.cancel_call_events(call_id)
            except Exception:
                logger.exception(f"Could not retire MCP call during shutdown call_id={call_id}")
        for future in self._approvals.values():
            if not future.done():
                future.cancel()
        self._approvals.clear()
        self._approval_calls.clear()
        self._approval_response_claims.clear()
        for state in self._servers.values():
            state.worker.shutdown()
        self._servers.clear()
        self._session_definitions.clear()
        self._pending_session_updates.clear()
        self._pending_response_preparations.clear()
        self._bindings.clear()
        self._binding_names.clear()
        self._pipeline_identities.clear()
        self._call_announced.clear()
        self._pending_results.clear()
        self._cancelled_context_callbacks.clear()
