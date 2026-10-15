# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-executed Realtime function tools.

Clients register function schemas through ``session.tools``. The LLM may call
them like catalog tools, but the server does not run them: it emits the call as
a Realtime ``function_call`` output item and waits for the client's
``function_call_output`` item. Once the response that made the call is done,
the exact output is written into the LLM context. As in the OpenAI Realtime
API, the client sends ``response.create`` to continue after the output.

A pending call survives barge-in. It ends with the client's output, after
``REALTIME_CLIENT_TOOL_TIMEOUT_SECONDS`` without one, or when the response that
made it is cancelled.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from loguru import logger
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    Frame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    FunctionCallsStartedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService

from realtime.client_tool_broker import ClientToolBroker, RealtimeClientToolProjection
from realtime.protocol import RealtimeProtocolError
from utils import parse_env_float

# Pipeline examples that run client tools declare this registry capability.
CLIENT_TOOLS_CAPABILITY = "realtime_client_tools"

MAX_CLIENT_TOOLS = 128
MAX_CLIENT_TOOLS_JSON_BYTES = 256 * 1024
MAX_CLIENT_TOOL_OUTPUT_BYTES = 1024 * 1024
DEFAULT_CLIENT_TOOL_TIMEOUT_SECS = 120.0
_MAX_TRACKED_CALLS = 4096
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def normalize_client_tools(raw: Any) -> list[dict[str, Any]]:
    """Validate ``session.tools`` and return flat Realtime function tools.

    Accepts the Realtime shape ``{"type": "function", "name", "description",
    "parameters"}`` and the Chat Completions shape ``{"type": "function",
    "function": {...}}``. Raises ``ValueError`` on invalid input.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("session.tools must be an array")
    if len(raw) > MAX_CLIENT_TOOLS:
        raise ValueError(f"session.tools supports at most {MAX_CLIENT_TOOLS} tools")
    try:
        encoded = json.dumps(raw, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("session.tools must contain only JSON values") from exc
    if len(encoded.encode("utf-8")) > MAX_CLIENT_TOOLS_JSON_BYTES:
        raise ValueError(f"session.tools exceeds {MAX_CLIENT_TOOLS_JSON_BYTES} bytes")

    tools: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        param = f"session.tools[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{param} must be an object")
        if entry.get("type") != "function":
            raise ValueError(f"{param}.type must be 'function'")
        spec = entry.get("function") if isinstance(entry.get("function"), dict) else entry
        name = spec.get("name")
        if not isinstance(name, str) or not _TOOL_NAME_RE.match(name):
            raise ValueError(f"{param}.name must match {_TOOL_NAME_RE.pattern}")
        if name in seen:
            raise ValueError(f"{param}.name {name!r} is registered more than once")
        seen.add(name)
        tool: dict[str, Any] = {"type": "function", "name": name}
        description = spec.get("description")
        if description is not None:
            if not isinstance(description, str):
                raise ValueError(f"{param}.description must be a string")
            tool["description"] = description
        parameters = spec.get("parameters")
        if parameters is None:
            parameters = {"type": "object", "properties": {}}
        if not isinstance(parameters, dict):
            raise ValueError(f"{param}.parameters must be a JSON Schema object")
        if parameters.get("type", "object") != "object":
            raise ValueError(f"{param}.parameters.type must be 'object'")
        tool["parameters"] = copy.deepcopy(parameters)
        tools.append(tool)
    return tools


def client_tool_timeout_secs() -> float:
    """Return the client tool output deadline from ``REALTIME_CLIENT_TOOL_TIMEOUT_SECONDS``."""
    value = parse_env_float("REALTIME_CLIENT_TOOL_TIMEOUT_SECONDS", DEFAULT_CLIENT_TOOL_TIMEOUT_SECS, min_value=1.0)
    if not math.isfinite(value):
        logger.warning(f"REALTIME_CLIENT_TOOL_TIMEOUT_SECONDS={value!r} is not finite; using the default")
        return DEFAULT_CLIENT_TOOL_TIMEOUT_SECS
    return value


@dataclass(slots=True)
class _EmittedCall:
    """One client call published as a ``function_call`` output item."""

    name: str
    response_id: str
    response_done: bool = False
    output_staged: bool = False
    retired: bool = False


class RealtimeClientTools:
    """One Realtime session's client tools.

    The projection maps client tool names to provider-safe function names and
    registers one handler for them. The broker holds each call until the
    client's output arrives, then writes that exact output into the LLM
    context. This class ties both to Realtime responses: a staged output is
    released only after the response that made the call is done, and
    ``response.create`` is refused while a call of a done response has no output.
    """

    def __init__(self, tools: list[dict[str, Any]], *, scope_id: str, output_timeout_secs: float) -> None:
        """Create the client tool state for one Realtime connection."""
        self.tools = [copy.deepcopy(tool) for tool in tools]
        self.names = frozenset(tool["name"] for tool in self.tools)
        self._projection = RealtimeClientToolProjection(scope_id=scope_id)
        self._broker = ClientToolBroker(output_timeout_secs=output_timeout_secs)
        self._calls: OrderedDict[str, _EmittedCall] = OrderedDict()

    def configure(self, llm: LLMService, trusted_tools: ToolsSchema | None, trusted_tool_names: Iterable[str]):
        """Offer the client tools to ``llm`` next to ``trusted_tools``; return the combined schema."""
        return self._projection.configure(
            llm,
            self.tools,
            client_tool_handler=self._broker.handle,
            trusted_tools=trusted_tools,
            trusted_tool_names=trusted_tool_names,
        )

    def bind_context(self, context: Any) -> None:
        """Bind the LLM context that receives client outputs."""
        self._broker.bind_context(context)

    def public_name(self, function_name: str) -> str | None:
        """Return the client tool name for a provider function name, or ``None`` for other tools."""
        return self._projection.public_name(function_name)

    def call_emitted(self, *, call_id: str, name: str, response_id: str) -> None:
        """Record a client call published in response ``response_id``."""
        self._calls[call_id] = _EmittedCall(name=name, response_id=response_id)
        while len(self._calls) > _MAX_TRACKED_CALLS:
            self._calls.popitem(last=False)

    def has_call(self, call_id: str) -> bool:
        """Return whether ``call_id`` was published to the client."""
        return call_id in self._calls

    async def stage_output(self, *, call_id: str, output: str) -> None:
        """Accept the client's output for ``call_id``; release it once its response is done.

        Raises ``RealtimeProtocolError`` for an unknown, settled, cancelled, or
        timed-out call and for a duplicate output.
        """
        call = self._calls.get(call_id)
        if call is None:
            raise RealtimeProtocolError(
                message=f"unknown call_id for function_call_output: {call_id}",
                code="invalid_item",
                param="item.call_id",
            )
        if call.retired:
            raise RealtimeProtocolError(
                message=f"Client tool call {call_id!r} was cancelled",
                code="client_tool_call_cancelled",
                param="item.call_id",
            )
        await self._broker.stage_output(call_id=call_id, name=call.name, output=output)
        call.output_staged = True
        if call.response_done:
            await self._broker.release_output(call_id=call_id, name=call.name)

    async def response_finished(self, *, response_id: str, status: str) -> None:
        """Release staged outputs of a completed response, or retire the calls of a cancelled one."""
        for call_id, call in list(self._calls.items()):
            if call.response_id != response_id or call.response_done or call.retired:
                continue
            if status != "completed":
                call.retired = True
                await self._broker.discard_staged_output(call_id=call_id, name=call.name)
                continue
            call.response_done = True
            if call.output_staged:
                try:
                    await self._broker.release_output(call_id=call_id, name=call.name)
                except RealtimeProtocolError as exc:
                    logger.warning(f"Realtime client tool output for call_id={call_id} not released: {exc.message}")

    def calls_missing_output(self) -> list[str]:
        """Return call ids of done responses that are still waiting for the client's output."""
        return [
            call_id
            for call_id, call in self._calls.items()
            if call.response_done
            and not call.retired
            and not call.output_staged
            and self._broker.is_awaiting_output(call_id)
        ]

    async def wait_outputs_applied(self, *, timeout: float) -> None:
        """Wait until every released output is in the LLM context.

        Raises ``RealtimeProtocolError`` (``client_tool_context_timeout`` and similar).
        """
        for call_id in self._broker.pending_context_call_ids():
            await self._broker.wait_context_applied(call_id, timeout=timeout)

    def shutdown(self) -> None:
        """Cancel every pending client call on session teardown."""
        self._broker.shutdown()


class RealtimeToolResultPolicy(FrameProcessor):
    """Keep a tool round that mixes server and client tools under the client's control.

    Pipecat runs the LLM again when the last call of a round settles. When one
    LLM turn calls both catalog (server) tools and client tools, the client
    continues the conversation with ``response.create``, so the server tools'
    results must not start a run of their own.
    """

    def __init__(self, client_tools: RealtimeClientTools) -> None:
        """Bind the policy to one Realtime connection's client tools."""
        super().__init__()
        self._client_tools = client_tools
        self._client_continued_calls: set[str] = set()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Turn off auto-continuation for server tools that share a round with client tools."""
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, FunctionCallsStartedFrame):
            calls = tuple(frame.function_calls or ())
            client = [self._client_tools.public_name(call.function_name or "") is not None for call in calls]
            if any(client):
                self._client_continued_calls.update(
                    call.tool_call_id for call, is_client in zip(calls, client, strict=True) if not is_client
                )
        elif (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, FunctionCallResultFrame)
            and frame.tool_call_id in self._client_continued_calls
        ):
            properties = frame.properties or FunctionCallResultProperties()
            frame.properties = replace(properties, run_llm=False)
            if frame.properties.is_final:
                self._client_continued_calls.discard(frame.tool_call_id)
        await self.push_frame(frame, direction)
