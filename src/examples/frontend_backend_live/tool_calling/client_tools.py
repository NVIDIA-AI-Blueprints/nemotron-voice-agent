# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tools that the live-session client executes.

The backend's function calls are forwarded to the client as response events. The client runs every call of a
round, returns each result with ``response.item.create``, and asks for one continuation with ``response.create``.
This gateway holds that exchange: it registers calls as the stream emits them, validates the client's results,
and releases the worker when the batch is complete and the continuation was requested.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Awaitable, Callable
from copy import deepcopy

from live.protocol import ProtocolError

MAX_COMPLETED_CALLS = 512
MAX_QUEUED_ITEMS = 16


class ClientToolGateway:
    """Collects a round's function calls and the client's results, and gates the continuation."""

    def __init__(
        self,
        submit: Callable[[list[dict]], Awaitable[None]],
        busy: Callable[[], bool],
        tool_timeout_seconds: float,
        observe: Callable[..., None] = lambda kind, **fields: None,
    ):
        """Wire the gateway to the worker.

        Args:
            submit: Awaited with queued input items to start a new backend task (``response.create``).
            busy: Returns True while a backend task is running or waiting.
            tool_timeout_seconds: How long the client may take to return results and continue.
            observe: Optional ``observe(kind, **fields)`` hook.
        """
        self.submit, self.busy, self.timeout, self.observe = submit, busy, tool_timeout_seconds, observe
        self.calls: dict[str, dict] = {}
        self.results: dict[str, dict] = {}
        self.completed: dict[str, dict] = {}
        self.queued: deque[dict] = deque()
        self.continuation = asyncio.Event()
        self.phase = "idle"

    # ------------------------------------------------------------------ the worker's side
    def begin_round(self) -> None:
        """A backend round started: forget the previous round's calls."""
        self.calls, self.results, self.phase = {}, {}, "running"
        self.continuation.clear()

    def note_call(self, item: dict) -> None:
        """A function call completed in the stream; the client may now answer it."""
        self.calls[item["call_id"]] = item

    async def execute_batch(self, calls: dict[str, dict]) -> list[dict]:
        """Wait for every result and the continuation; return the next round's input items."""
        self.phase = "waiting_tools"
        self.observe("tools.wait.started", call_ids=list(calls))
        async with asyncio.timeout(self.timeout):
            await self.continuation.wait()
        incoming = [self.results[call_id] for call_id in calls]
        self.completed.update(self.results)
        # Typed corrections queued while waiting are supplied after the verified results.
        incoming.extend(self.queued)
        self.queued.clear()
        if len(self.completed) > MAX_COMPLETED_CALLS:
            self.completed = dict(list(self.completed.items())[-MAX_COMPLETED_CALLS // 2 :])
        self.calls, self.results, self.phase = {}, {}, "running"
        self.observe("tools.wait.completed", call_ids=list(calls))
        return incoming

    # ------------------------------------------------------------------ the client's side
    def item(self, item: dict) -> None:
        """Accept a function result or a typed user message from ``response.item.create``."""
        if item.get("type") == "function_call_output":
            self._result(item)
        elif item.get("type", "message") == "message":
            self._message(item)
        else:
            raise ProtocolError("Unsupported response item", param="item.type")

    def _result(self, item: dict) -> None:
        call_id = item.get("call_id")
        if call_id in self.completed:
            if self.completed[call_id] != item:
                raise ProtocolError("Conflicting duplicate tool result", param="item.call_id")
            self.observe("tool.result.duplicate", call_id=call_id)
            return
        if call_id not in self.calls:
            raise ProtocolError("Unknown pending function call", param="item.call_id")
        if not isinstance(item.get("output"), str):
            raise ProtocolError("Function output must be a string", param="item.output")
        if call_id in self.results and self.results[call_id] != item:
            raise ProtocolError("Conflicting duplicate tool result", param="item.call_id")
        self.results[call_id] = deepcopy(item)
        self.observe("tool.result.received", call_id=call_id, name=self.calls[call_id].get("name"))

    def _message(self, item: dict) -> None:
        if item.get("role") != "user" or not isinstance(item.get("content"), list):
            raise ProtocolError("Expected a user message", param="item")
        parts = item["content"]
        if not parts or any(
            not isinstance(p, dict) or p.get("type") not in {"input_text", "input_image"} for p in parts
        ):
            raise ProtocolError("Unsupported message content", param="item.content")
        if len(self.queued) >= MAX_QUEUED_ITEMS:
            raise ProtocolError("Too many queued response items", "rate_limit_exceeded")
        self.queued.append(deepcopy(item))
        text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "input_text")
        self.observe("input.typed", text=text, chars=len(json.dumps(item)))

    async def create(self) -> None:
        """Handle ``response.create``: continue a complete batch, or run queued typed input."""
        if self.calls:
            if self.phase != "waiting_tools":
                raise ProtocolError("Wait for the complete function-call batch", "response_in_progress")
            missing = set(self.calls) - set(self.results)
            if missing:
                raise ProtocolError(
                    "Return every pending function result before response.create", "missing_tool_results"
                )
            if self.continuation.is_set():
                raise ProtocolError("A continuation was already requested", "response_in_progress")
            self.continuation.set()
            return
        if self.busy():
            raise ProtocolError("A response is already running", "response_in_progress")
        if not self.queued:
            raise ProtocolError("No queued input or function results", "no_response_input")
        items = list(self.queued)
        self.queued.clear()
        await self.submit(items)
