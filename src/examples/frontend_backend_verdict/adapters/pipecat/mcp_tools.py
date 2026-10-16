# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in MCP tools for OpenAI Realtime sessions, run by the server and hidden from the client.

Plan ``misc/frontend-backend-verdict/fba-removal-design-5.8-5.9.md`` (5.9, approved: calls hidden;
a barge-in or cancel stops a running call and gives the agent a ``cancelled`` result). Used only
when ``FBV_REALTIME_MCP=true``, ``REALTIME_MCP_ALLOWED_SERVER_URLS`` is set and the container is not
in eval mode; otherwise this module is never imported and ``type: "mcp"`` tools are refused as
before. Nothing in ``core/`` changes:

* **Inbound** (:meth:`McpTools.prepare_update`): the ``mcp`` entries of a ``session.update`` are
  validated with the shared gateway rules (``src/realtime/mcp.py``: exact-URL allowlist, headers,
  allowed tools, limits), their servers are listed, and each MCP tool becomes an ordinary function
  tool named ``mcp__<server_label>__<tool>``. The agent sees only function tools.
* **Echo** (:meth:`McpTools.public_tools`): ``session.updated`` shows the client's ``mcp`` entries,
  not the generated functions.
* **Outbound** (:meth:`McpTools.outbound`): the events of a call to a generated function never reach
  the client; the call runs here and its result goes back to the agent the way a client's
  ``function_call_output`` does. ``previous_item_id``, ``output_index`` and ``response.done`` are
  repaired so the client sees a consistent conversation. A response that held only MCP calls is
  resumed by the server (the client never saw a call to answer).

Because the calls are hidden, MCP tools need ``require_approval: "never"``: there is nothing for
the client to approve.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from examples.frontend_backend_verdict.core.errors import WireProtocolError

# The shared gateway's MCP rules and transport (its RealtimeMCPRuntime binding is not used).
from realtime.mcp import (
    _configured_allowed_urls,
    _http_headers,
    _mcp_error,
    _MCPServerWorker,
    _select_allowed_tools,
    _serialize_call_result,
    _validate_advertised_tool_names,
    _validate_server_url,
)
from realtime.protocol import RealtimeProtocolError
from utils import parse_env_float, parse_env_int

#: Prefix of the function names generated for MCP tools (reserved: a client function cannot use it).
PREFIX = "mcp__"
_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]")
_ENDPOINT_FIELDS = ("server_url", "connector_id", "tunnel_id")


def configured() -> bool:
    """Whether the deployment allows any MCP server (``REALTIME_MCP_ALLOWED_SERVER_URLS``)."""
    return bool(_configured_allowed_urls())


def _wire_error(exc: RealtimeProtocolError) -> WireProtocolError:
    return WireProtocolError(exc.message, code=exc.code, param=f"session.{exc.param}" if exc.param else "session")


@dataclass(slots=True)
class _Binding:
    server_label: str
    tool: str
    function: dict[str, Any]


@dataclass(slots=True)
class _Response:
    hidden_indices: list[int] = field(default_factory=list)
    hidden_calls: list[str] = field(default_factory=list)
    visible_calls: int = 0


class McpTools:
    """One Realtime session's MCP servers, generated functions and hidden calls."""

    def __init__(
        self,
        *,
        feed_output: Callable[[str, str], None],
        resume: Callable[[], None],
        log: Callable[..., None],
    ) -> None:
        """``feed_output(call_id, output)`` and ``resume()`` reach the turn manager; ``log`` the event log."""
        self._feed_output = feed_output
        self._resume = resume
        self._log = log
        self._definitions: dict[str, dict[str, Any]] = {}
        self._workers: dict[str, _MCPServerWorker] = {}
        self._bindings: dict[str, _Binding] = {}
        self._client_entries: list[dict[str, Any]] = []
        self._hidden_items: dict[str, str | None] = {}  # hidden item id -> visible previous item id
        self._hidden_call_items: dict[str, str] = {}  # call_id -> hidden item id
        self._responses: dict[str, _Response] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    # -- inbound ------------------------------------------------------------------------

    async def prepare_update(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Replace the ``mcp`` entries of ``patch["tools"]`` with generated function tools."""
        tools = patch.get("tools")
        if not isinstance(tools, list):
            return patch
        client_functions = [t for t in tools if not (isinstance(t, dict) and t.get("type") == "mcp")]
        entries = [t for t in tools if isinstance(t, dict) and t.get("type") == "mcp"]
        for tool in client_functions:
            name = tool.get("name") if isinstance(tool, dict) else None
            if isinstance(name, str) and name.startswith(PREFIX):
                raise WireProtocolError(
                    f"function names starting with {PREFIX!r} are reserved for MCP tools",
                    code="invalid_value",
                    param="session.tools",
                )
        bindings: dict[str, _Binding] = {}
        try:
            for index, entry in enumerate(entries):
                for binding in await self._bindings_for(entry, index):
                    if binding.function["name"] in bindings:
                        raise WireProtocolError(
                            f"MCP tool {binding.function['name']!r} is declared twice",
                            code="invalid_value",
                            param="session.tools",
                        )
                    bindings[binding.function["name"]] = binding
        except RealtimeProtocolError as exc:
            raise _wire_error(exc) from exc
        self._bindings = bindings
        self._client_entries = copy.deepcopy(entries)
        generated = [copy.deepcopy(binding.function) for binding in bindings.values()]
        return {**patch, "tools": [*client_functions, *generated]}

    async def _bindings_for(self, entry: dict[str, Any], index: int) -> list[_Binding]:
        param = f"tools[{index}]"
        label = entry.get("server_label")
        if not isinstance(label, str) or not label.strip():
            raise WireProtocolError("an MCP tool needs a server_label", code="invalid_value", param=f"session.{param}")
        if not any(entry.get(name) is not None for name in _ENDPOINT_FIELDS):
            definition = self._definitions.get(label)
            if definition is None:
                raise WireProtocolError(
                    f"MCP server_label {label!r} has not been defined in this session",
                    code="mcp_server_not_defined",
                    param=f"session.{param}.server_label",
                )
        else:
            for unsupported in ("connector_id", "tunnel_id"):
                if entry.get(unsupported) is not None:
                    raise WireProtocolError(
                        f"MCP {unsupported} is not available on this server",
                        code="unsupported_capability",
                        param=f"session.{param}.{unsupported}",
                    )
            definition = copy.deepcopy(entry)
        if definition.get("require_approval") != "never":
            raise WireProtocolError(
                'MCP tools on this model run without client approval: set require_approval to "never"',
                code="unsupported_capability",
                param=f"session.{param}.require_approval",
            )
        await _validate_server_url(str(definition["server_url"]))
        _http_headers(definition)
        known = self._definitions.get(label)
        if known is not None and any(
            known.get(name) != definition.get(name) for name in ("server_url", "authorization", "headers")
        ):
            raise WireProtocolError(
                f"MCP server_label {label!r} cannot be rebound to another endpoint or credential",
                code="mcp_server_label_conflict",
                param=f"session.{param}.server_label",
            )
        if known is None and len(self._definitions) >= parse_env_int("REALTIME_MCP_MAX_SERVERS", 8, min_value=1):
            raise WireProtocolError(
                "this session has reached its MCP server limit", code="mcp_server_limit", param="session.tools"
            )
        worker = self._workers.get(label)
        if worker is None or worker.dead:
            worker = _MCPServerWorker(definition)
            self._workers[label] = worker
        timeout = parse_env_float("REALTIME_MCP_DISCOVERY_TIMEOUT_SECONDS", 30.0, min_value=1.0)
        try:
            await asyncio.wait_for(worker.start(), timeout=timeout)
            listed = await asyncio.wait_for(worker.list_tools(), timeout=timeout)
        except Exception as exc:
            raise WireProtocolError(
                f"MCP server {label!r} could not be reached: {_mcp_error(exc)['message']}",
                code="mcp_server_unavailable",
                param=f"session.{param}",
            ) from exc
        _validate_advertised_tool_names(listed, server_label=label)
        selected = _select_allowed_tools(listed, definition.get("allowed_tools"))
        if len(selected) > parse_env_int("REALTIME_MCP_MAX_TOOLS_PER_SERVER", 128, min_value=1):
            raise WireProtocolError(
                f"MCP server {label!r} advertises too many tools", code="mcp_tool_limit", param=f"session.{param}"
            )
        self._definitions[label] = definition
        bindings = []
        for tool in selected:
            name = f"{PREFIX}{_NAME_CHARS.sub('_', label)}__{_NAME_CHARS.sub('_', tool.name)}"[:64]
            description = f"MCP server {label!r}, tool {tool.name!r}. {tool.description or ''}".strip()
            function = {
                "type": "function",
                "name": name,
                "description": description,
                "parameters": copy.deepcopy(tool.input_schema) or {"type": "object", "properties": {}},
            }
            bindings.append(_Binding(server_label=label, tool=tool.name, function=function))
        return bindings

    # -- echo ---------------------------------------------------------------------------

    def public_tools(self, tools: list[Any]) -> list[Any]:
        """The session's tools as the client declared them: generated functions out, ``mcp`` entries back."""
        visible = [t for t in tools if not (isinstance(t, dict) and str(t.get("name", "")).startswith(PREFIX))]
        return [*visible, *copy.deepcopy(self._client_entries)]

    # -- outbound -----------------------------------------------------------------------

    def outbound(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        """The events the client gets for ``event``: none for a hidden call, repaired otherwise."""
        kind = event.get("type")
        item = event.get("item") if isinstance(event.get("item"), dict) else None
        response_id = str(event.get("response_id") or "")
        if kind == "response.output_item.added" and item is not None and item.get("type") == "function_call":
            response = self._responses.setdefault(response_id, _Response())
            if item.get("name") in self._bindings:
                self._hidden_call_items[str(item.get("call_id"))] = str(item.get("id"))
                self._hidden_items.setdefault(str(item.get("id")), None)
                response.hidden_indices.append(int(event.get("output_index", 0)))
                response.hidden_calls.append(str(item.get("call_id")))
                return []
            response.visible_calls += 1
        item_id = str(event.get("item_id") or (item or {}).get("id") or "")
        if item_id in self._hidden_items:
            if kind in ("conversation.item.added", "conversation.item.done"):
                previous = event.get("previous_item_id")
                self._hidden_items[item_id] = self._visible(previous)
            if kind == "response.function_call_arguments.done":
                self._start_call(str(event.get("call_id")), str(event.get("name")), str(event.get("arguments") or ""))
            return []
        event = copy.deepcopy(event)
        if "previous_item_id" in event:
            event["previous_item_id"] = self._visible(event.get("previous_item_id"))
        response = self._responses.get(response_id)
        if response is not None and response.hidden_indices and isinstance(event.get("output_index"), int):
            event["output_index"] -= sum(1 for index in response.hidden_indices if index < event["output_index"])
        if kind == "response.done":
            body = event.get("response") or {}
            done_id = str(body.get("id") or "")
            output = body.get("output")
            if isinstance(output, list):
                body["output"] = [o for o in output if not (isinstance(o, dict) and o.get("id") in self._hidden_items)]
            done = self._responses.pop(done_id, None)
            if done is not None and done.hidden_calls and not done.visible_calls:
                # The client saw no call to answer, so it sends no response.create: resume here.
                self._resume()
        return [event]

    def _visible(self, item_id: Any) -> Any:
        while isinstance(item_id, str) and item_id in self._hidden_items:
            item_id = self._hidden_items[item_id]
        return item_id

    def _start_call(self, call_id: str, name: str, arguments: str) -> None:
        binding = self._bindings.get(name)
        if binding is None:
            return
        self._tasks[call_id] = asyncio.get_running_loop().create_task(
            self._run_call(call_id, binding, arguments), name=f"fbv-mcp-{call_id}"
        )

    async def _run_call(self, call_id: str, binding: _Binding, arguments: str) -> None:
        loop = asyncio.get_running_loop()
        started = loop.time()
        worker = self._workers.get(binding.server_label)
        try:
            args = json.loads(arguments) if arguments else {}
            if not isinstance(args, dict) or worker is None:
                raise ValueError("MCP tool arguments must be a JSON object")
            timeout = parse_env_float("REALTIME_MCP_CALL_TIMEOUT_SECONDS", 120.0, min_value=1.0)
            result = await asyncio.wait_for(worker.call_tool(binding.tool, args), timeout=timeout)
            max_bytes = parse_env_int("REALTIME_MCP_MAX_OUTPUT_BYTES", 262_144, min_value=1024)
            output = _serialize_call_result(result, max_bytes=max_bytes)
            outcome = "error" if result.is_error else "ok"
        except asyncio.CancelledError:
            self._tasks.pop(call_id, None)
            raise
        except Exception as exc:  # noqa: BLE001 - every failure becomes the tool's result
            output = json.dumps({"error": _mcp_error(exc)})
            outcome = "failed"
        self._tasks.pop(call_id, None)
        self._log(
            "mcp_call",
            call_id=call_id,
            server=binding.server_label,
            tool=binding.tool,
            outcome=outcome,
            latency_ms=int((loop.time() - started) * 1000),
        )
        self._deliver(call_id, output)

    def _deliver(self, call_id: str, output: str) -> None:
        try:
            self._feed_output(call_id, output)
        except WireProtocolError as exc:
            # The turn already moved on (its wait timed out or was resumed): nothing waits for this.
            logger.info(f"MCP result for {call_id} dropped: {exc}")

    def cancel_all(self, reason: str) -> None:
        """Barge-in or cancel: stop running MCP calls; the agent gets a ``cancelled`` result."""
        for call_id, task in list(self._tasks.items()):
            task.cancel()
            self._tasks.pop(call_id, None)
            self._log("mcp_call", call_id=call_id, outcome="cancelled", reason=reason)
            self._deliver(call_id, json.dumps({"error": {"type": "cancelled", "message": f"cancelled: {reason}"}}))

    @property
    def running(self) -> int:
        """MCP calls still running."""
        return len(self._tasks)

    def shutdown(self) -> None:
        """Session end: stop calls and close the MCP connections."""
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
        for worker in self._workers.values():
            worker.shutdown()
