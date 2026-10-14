# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-executed Realtime function tools.

Clients register function schemas through ``session.tools``. The LLM may call
them like catalog tools, but the server does not run them: it emits the call as
a Realtime ``function_call`` output item, waits for the client's
``function_call_output`` item, writes that output into the LLM context, then
acknowledges it with ``conversation.item.created``. As in the OpenAI Realtime
API, the client sends ``response.create`` to continue after the output.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
from collections.abc import Iterable
from typing import Any

from loguru import logger
from pipecat.frames.frames import FunctionCallResultProperties
from pipecat.services.llm_service import FunctionCallParams

from realtime.events import SERVER_ITEM_CREATED, EmitFn, emit_with_aliases, server_event

# Pipeline examples that run client tools declare this registry capability.
CLIENT_TOOLS_CAPABILITY = "realtime_client_tools"

MAX_CLIENT_TOOLS = 128
MAX_CLIENT_TOOLS_JSON_BYTES = 256 * 1024
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


def to_openai_tools(tools: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return Chat Completions tool definitions for flat Realtime tools."""
    out: list[dict[str, Any]] = []
    for tool in tools:
        function: dict[str, Any] = {"name": tool["name"], "parameters": copy.deepcopy(tool["parameters"])}
        if "description" in tool:
            function["description"] = tool["description"]
        out.append({"type": "function", "function": function})
    return out


def _output_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output, default=str)


class ClientToolBridge:
    """Route client tool calls between the LLM service and the Realtime socket.

    The lifecycle observer calls :meth:`expect` when it emits a call. The LLM
    handler (:meth:`handle_call`) waits for the matching output, which the
    serializer delivers through :meth:`submit_output`.
    """

    def __init__(self, *, tools: list[dict[str, Any]], emit: EmitFn) -> None:
        """Create a bridge for one Realtime session's registered client tools."""
        self.tools = [copy.deepcopy(tool) for tool in tools]
        self.names = frozenset(tool["name"] for tool in self.tools)
        self._emit = emit
        self._outputs: dict[str, asyncio.Future[tuple[str, str]]] = {}
        self._finished: set[str] = set()

    def _future(self, call_id: str) -> asyncio.Future[tuple[str, str]]:
        future = self._outputs.get(call_id)
        if future is None:
            future = asyncio.get_running_loop().create_future()
            self._outputs[call_id] = future
        return future

    def expect(self, call_id: str) -> None:
        """Record a call id the client may answer."""
        if call_id and call_id not in self._finished:
            self._future(call_id)

    def is_known(self, call_id: str) -> bool:
        """Return whether ``call_id`` was emitted to the client in this session."""
        return call_id in self._outputs or call_id in self._finished

    async def submit_output(self, *, call_id: str, item_id: str, output: Any) -> None:
        """Deliver one client output; acknowledge immediately if no handler waits."""
        text = _output_text(output)
        future = self._outputs.get(call_id)
        if future is None or future.done():
            # The call already settled (for example, cancelled by a barge-in).
            # Acknowledge so the client does not time out, but the LLM context
            # cannot take a second result for the same call.
            logger.warning(f"Realtime client tool output for settled call_id={call_id}; not added to context")
            await self.acknowledge(call_id=call_id, item_id=item_id, output=text)
            return
        future.set_result((item_id, text))

    async def acknowledge(self, *, call_id: str, item_id: str, output: str) -> None:
        """Emit the ``conversation.item.created`` acknowledgement for an output."""
        await emit_with_aliases(
            self._emit,
            server_event(
                SERVER_ITEM_CREATED,
                previous_item_id=None,
                item={
                    "id": item_id,
                    "object": "realtime.item",
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                },
            ),
        )

    async def handle_call(self, params: FunctionCallParams) -> None:
        """Pipecat handler: wait for the client's output and return it to the LLM."""
        call_id = params.tool_call_id
        future = self._future(call_id)
        try:
            item_id, output = await future
        finally:
            self._outputs.pop(call_id, None)
            self._finished.add(call_id)

        async def _acknowledge() -> None:
            await self.acknowledge(call_id=call_id, item_id=item_id, output=output)

        # The client continues with response.create once acknowledged, so the
        # acknowledgement waits until the output is in the LLM context.
        await params.result_callback(
            output,
            properties=FunctionCallResultProperties(run_llm=False, on_context_updated=_acknowledge),
        )

    def shutdown(self) -> None:
        """Cancel outstanding waits on session teardown."""
        for future in self._outputs.values():
            if not future.done():
                future.cancel()
        self._outputs.clear()
