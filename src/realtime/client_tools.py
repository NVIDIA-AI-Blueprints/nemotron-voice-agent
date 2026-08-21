# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Client-owned Realtime function schemas and deferred Pipecat execution."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    FunctionCallCancelFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    FunctionCallsStartedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import FunctionCallParams

from realtime.events import EmitFn, error_event
from utils import parse_env_int

MAX_CLIENT_TOOLS = 32
MAX_CLIENT_TOOLS_JSON_BYTES = 64 * 1024
MAX_TOOL_DESCRIPTION_CHARS = 4096
MAX_TOOL_OUTPUT_BYTES = 1024 * 1024
_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

QueueRunFn = Callable[[], Awaitable[None]]
ToolsUpdateFn = Callable[[list[dict[str, Any]], Any], Awaitable[None]]


def validate_client_tools(raw: Any) -> list[dict[str, Any]]:
    """Validate and return canonical OpenAI Realtime function schemas."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("session.tools must be an array")
    if len(raw) > MAX_CLIENT_TOOLS:
        raise ValueError(f"session.tools supports at most {MAX_CLIENT_TOOLS} functions")
    try:
        encoded = json.dumps(raw, separators=(",", ":"), ensure_ascii=False).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("session.tools must be JSON serializable") from exc
    if len(encoded) > MAX_CLIENT_TOOLS_JSON_BYTES:
        raise ValueError(f"session.tools exceeds {MAX_CLIENT_TOOLS_JSON_BYTES} bytes")

    canonical: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, entry in enumerate(raw):
        param = f"session.tools[{index}]"
        if not isinstance(entry, dict) or entry.get("type") != "function":
            raise ValueError(f"{param} must be a function tool")
        name = entry.get("name")
        if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
            raise ValueError(f"{param}.name must match {_TOOL_NAME.pattern}")
        if name in names:
            raise ValueError(f"session.tools contains duplicate function name '{name}'")
        names.add(name)

        description = entry.get("description", "")
        if not isinstance(description, str):
            raise ValueError(f"{param}.description must be a string")
        if len(description) > MAX_TOOL_DESCRIPTION_CHARS:
            raise ValueError(f"{param}.description is too long")

        parameters = entry.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(parameters, dict):
            raise ValueError(f"{param}.parameters must be an object")
        if parameters.get("type", "object") != "object":
            raise ValueError(f"{param}.parameters.type must be 'object'")
        required = parameters.get("required", [])
        if not isinstance(required, list) or not all(isinstance(value, str) for value in required):
            raise ValueError(f"{param}.parameters.required must be an array of strings")

        canonical.append(
            {
                "type": "function",
                "name": name,
                "description": description,
                "parameters": copy.deepcopy(parameters),
            }
        )
    return canonical


def validate_tool_ownership(
    client_tools: Sequence[dict[str, Any]],
    server_tools: Sequence[str],
    *,
    pipeline_mode: str,
) -> None:
    """Reject unsupported pipeline modes and ambiguous client/server names."""
    if client_tools and pipeline_mode != "generic-assistant":
        raise ValueError("session.tools is currently supported only by pipeline_mode 'generic-assistant'")
    client_names = {str(tool.get("name") or "") for tool in client_tools}
    conflicts = sorted(client_names & set(server_tools))
    if conflicts:
        raise ValueError(f"client tool names conflict with server tools: {', '.join(conflicts)}")


def validate_tool_choice_names(tool_choice: Any, available_names: set[str]) -> None:
    """Ensure a forced function choice names a currently available tool."""
    if not isinstance(tool_choice, dict):
        return
    function = tool_choice.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    if isinstance(name, str) and name not in available_names:
        raise ValueError(f"session.tool_choice references unavailable function '{name}'")


@dataclass
class PendingCall:
    """One model-issued function call retained across live schema updates."""

    call_id: str
    name: str
    arguments: Any
    owner: str
    batch_id: str
    future: asyncio.Future[str] | None = None
    output_received: bool = False
    result_applied: bool = False
    terminal: str | None = None


@dataclass
class ToolBatch:
    """A model function-call batch and its explicit continuation state."""

    batch_id: str
    call_ids: list[str] = field(default_factory=list)
    client_call_ids: list[str] = field(default_factory=list)
    continuation_requested: bool = False
    continued: bool = False


class ClientToolBroker:
    """Coordinate client-owned tools between Realtime and Pipecat."""

    def __init__(
        self,
        *,
        client_tools: Sequence[dict[str, Any]] | None = None,
        server_tools: Sequence[str] | None = None,
    ) -> None:
        """Create a broker for one Realtime session."""
        self._client_tools = {tool["name"]: copy.deepcopy(tool) for tool in (client_tools or [])}
        self._server_tools = set(server_tools or [])
        self._calls: dict[str, PendingCall] = {}
        self._batches: dict[str, ToolBatch] = {}
        self._batch_counter = 0
        self._emit: EmitFn | None = None
        self._queue_run: QueueRunFn | None = None
        self._on_tools_update: ToolsUpdateFn | None = None
        self._closed = False
        self.timeout_secs = parse_env_int("REALTIME_CLIENT_TOOL_TIMEOUT_SECS", 60, min_value=1)

    @property
    def client_tools(self) -> list[dict[str, Any]]:
        """Return client schemas in stable insertion order."""
        return [copy.deepcopy(tool) for tool in self._client_tools.values()]

    @property
    def server_tools(self) -> set[str]:
        """Return server-owned function names."""
        return set(self._server_tools)

    def set_emit(self, emit: EmitFn | None) -> None:
        """Bind the session's outbound event callback."""
        self._emit = emit

    def set_queue_run(self, callback: QueueRunFn | None) -> None:
        """Bind the callback that queues an explicit continuation."""
        self._queue_run = callback

    def set_tools_update(self, callback: ToolsUpdateFn | None) -> None:
        """Bind the callback that updates the live Pipecat tool context."""
        self._on_tools_update = callback

    def is_client_name(self, name: str) -> bool:
        """Return whether ``name`` is currently client-owned."""
        return name in self._client_tools

    def is_client_call(self, call_id: str) -> bool:
        """Return whether ``call_id`` was issued for a client tool."""
        call = self._calls.get(call_id)
        return call is not None and call.owner == "client"

    def batch_has_client(self, call_id: str) -> bool:
        """Return whether the call's model batch contains a client tool."""
        call = self._calls.get(call_id)
        if call is None:
            return False
        batch = self._batches.get(call.batch_id)
        return bool(batch and batch.client_call_ids)

    def register_batch(self, function_calls: Sequence[Any]) -> ToolBatch:
        """Record ownership before Pipecat starts function handlers."""
        existing_ids = [str(getattr(call, "tool_call_id", "") or "") for call in function_calls]
        for call_id in existing_ids:
            existing = self._calls.get(call_id)
            if existing:
                return self._batches[existing.batch_id]

        self._batch_counter += 1
        batch_id = f"tool_batch_{self._batch_counter}"
        batch = ToolBatch(batch_id=batch_id)
        loop = asyncio.get_running_loop()
        for call in function_calls:
            call_id = str(getattr(call, "tool_call_id", "") or "")
            name = str(getattr(call, "function_name", "") or "")
            if not call_id:
                continue
            owner = "client" if self.is_client_name(name) else "server"
            pending = PendingCall(
                call_id=call_id,
                name=name,
                arguments=getattr(call, "arguments", None),
                owner=owner,
                batch_id=batch_id,
                future=loop.create_future() if owner == "client" else None,
            )
            self._calls[call_id] = pending
            batch.call_ids.append(call_id)
            if owner == "client":
                batch.client_call_ids.append(call_id)
        self._batches[batch_id] = batch
        return batch

    async def client_handler(self, params: FunctionCallParams) -> None:
        """Wait for the matching ``function_call_output`` then update context."""
        call = self._calls.get(params.tool_call_id)
        if call is None or call.owner != "client" or call.future is None:
            await params.result_callback(
                json.dumps({"error": "unknown client tool call"}),
                properties=FunctionCallResultProperties(run_llm=False),
            )
            return
        try:
            output = await asyncio.wait_for(asyncio.shield(call.future), timeout=self.timeout_secs)
        except TimeoutError:
            call.terminal = "expired"
            await self._emit_error(
                f"Client tool '{call.name}' timed out waiting for function_call_output",
                code="client_tool_timeout",
                param="item.call_id",
            )
            output = json.dumps({"error": "client tool output timed out"})
        except asyncio.CancelledError:
            call.terminal = "cancelled"
            raise
        try:
            result: Any = json.loads(output)
        except (TypeError, json.JSONDecodeError):
            result = output
        await params.result_callback(
            result,
            properties=FunctionCallResultProperties(run_llm=False),
        )

    async def submit_output(self, call_id: str, output: str) -> PendingCall:
        """Resolve a client call exactly once or raise a stable validation error."""
        if len(output.encode()) > MAX_TOOL_OUTPUT_BYTES:
            raise ValueError("tool_output_too_large")
        call = self._calls.get(call_id)
        if call is None or call.owner != "client":
            raise ValueError("unknown_call_id")
        if call.terminal in {"expired", "cancelled"}:
            raise ValueError("stale_call_id")
        if call.output_received or call.terminal == "completed":
            raise ValueError("duplicate_call_output")
        if call.future is None or call.future.done():
            raise ValueError("stale_call_id")
        call.output_received = True
        call.future.set_result(output)
        return call

    async def prepare_continuation(self) -> str:
        """Return ``ready``, ``queued``, ``missing_output``, or ``none``."""
        batches = [batch for batch in self._batches.values() if batch.client_call_ids and not batch.continued]
        if not batches:
            return "none"
        batch = batches[-1]
        if any(self._calls[call_id].terminal in {"expired", "cancelled"} for call_id in batch.client_call_ids):
            return "terminal"
        if any(not self._calls[call_id].output_received for call_id in batch.client_call_ids):
            return "missing_output"
        if self._batch_results_applied(batch):
            batch.continued = True
            return "ready"
        batch.continuation_requested = True
        return "queued"

    async def mark_result_applied(self, call_id: str) -> None:
        """Mark one Pipecat result consumed and release a queued continuation."""
        call = self._calls.get(call_id)
        if call is None:
            return
        call.result_applied = True
        if call.owner == "client":
            call.terminal = call.terminal or "completed"
        batch = self._batches.get(call.batch_id)
        if (
            batch
            and batch.client_call_ids
            and batch.continuation_requested
            and not batch.continued
            and self._batch_results_applied(batch)
        ):
            batch.continued = True
            if self._queue_run is not None:
                await self._queue_run()

    async def update_tools(self, tools: Sequence[dict[str, Any]], tool_choice: Any) -> None:
        """Apply live client schemas while preserving existing call records."""
        validate_tool_ownership(tools, sorted(self._server_tools), pipeline_mode="generic-assistant")
        self._client_tools = {tool["name"]: copy.deepcopy(tool) for tool in tools}
        if self._on_tools_update is not None:
            await self._on_tools_update(self.client_tools, tool_choice)

    async def cancel_call(self, call_id: str) -> None:
        """Cancel a deferred call and reject any later output."""
        call = self._calls.get(call_id)
        if call is None:
            return
        call.terminal = "cancelled"
        if call.future is not None and not call.future.done():
            call.future.cancel()

    async def close(self) -> None:
        """Cancel every deferred handler on disconnect."""
        if self._closed:
            return
        self._closed = True
        for call in self._calls.values():
            if call.future is not None and not call.future.done():
                call.terminal = "cancelled"
                call.future.cancel()

    def _batch_results_applied(self, batch: ToolBatch) -> bool:
        return all(self._calls[call_id].result_applied for call_id in batch.call_ids)

    async def _emit_error(self, message: str, *, code: str, param: str) -> None:
        if self._emit is None:
            logger.warning(message)
            return
        await self._emit(error_event(message, code=code, param=param))


class RealtimeToolCoordinator(FrameProcessor):
    """Suppress Pipecat auto-continuation for batches containing client tools."""

    def __init__(self, broker: ClientToolBroker) -> None:
        """Create a coordinator for one session broker."""
        super().__init__()
        self._broker = broker

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Forward frames while overriding mixed-batch continuation."""
        await super().process_frame(frame, direction)
        if isinstance(frame, FunctionCallsStartedFrame):
            self._broker.register_batch(frame.function_calls or [])
        elif isinstance(frame, FunctionCallResultFrame) and self._broker.batch_has_client(frame.tool_call_id):
            properties = frame.properties or FunctionCallResultProperties()
            previous_context_updated = properties.on_context_updated

            async def _context_updated() -> None:
                if previous_context_updated is not None:
                    await previous_context_updated()
                await self._broker.mark_result_applied(frame.tool_call_id)

            properties.run_llm = False
            properties.on_context_updated = _context_updated
            frame.properties = properties
            frame.run_llm = False
        elif isinstance(frame, FunctionCallCancelFrame):
            await self._broker.cancel_call(frame.tool_call_id)

        await self.push_frame(frame, direction)
