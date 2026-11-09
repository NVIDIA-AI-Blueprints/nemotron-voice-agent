# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Responses on the wire: each request's stream, the writer of one response, and the book of a session's responses."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable, Collection
from copy import deepcopy
from typing import Any

from examples.frontend_backend_live.delegation_core.backend.port import Outcome, Phase
from examples.frontend_backend_live.delegation_core.ledger.records import WorkLedger
from examples.shared.ids import uid

TERMINALS = {"response.completed": "completed", "response.failed": "failed", "response.incomplete": "incomplete"}
MCP_ENDINGS = {
    "done": ("response.mcp_call.completed", "completed"),
    "failed": ("response.mcp_call.failed", "failed"),
    # An unclear call may have run: it did not complete, and its output says so.
    "unclear": ("response.mcp_call.failed", "incomplete"),
}


def output_message(item_id: str, text: str, phase: Phase, status: str = "completed") -> dict:
    """Return an assistant message item whose one ``output_text`` part holds ``text``."""
    part = {"type": "output_text", "text": text, "annotations": [], "logprobs": []}
    return {"id": item_id, "type": "message", "status": status, "role": "assistant", "phase": phase, "content": [part]}


class ResponseStream:
    """The events of one request's response, in order: an async iterator that ends after the response's terminal."""

    def __init__(self):
        """Start empty and open."""
        self._events: deque[dict] = deque()
        self._ready = asyncio.Event()
        self.ended = False

    def put(self, event: dict) -> None:
        """Add an event for the reader, unless the stream has ended."""
        if not self.ended:
            self._events.append(event)
            self._ready.set()

    def end(self) -> None:
        """End the stream: the reader gets what was written before, then the iteration stops. Safe to call twice."""
        self.ended = True
        self._ready.set()

    async def aclose(self) -> None:
        """Stop reading; the response goes on, as a conversation's work does not stop with its reader."""

    def __aiter__(self) -> ResponseStream:
        """Return the stream itself."""
        return self

    async def __anext__(self) -> dict:
        """Return the next event; ``StopAsyncIteration`` once the stream has ended and every event was read."""
        while not self._events:
            if self.ended:
                raise StopAsyncIteration
            self._ready.clear()
            await self._ready.wait()
        return self._events.popleft()


class ResponseWriter:
    """Writes one response's events into a :class:`ResponseStream`: numbered, whole items, exactly one terminal."""

    def __init__(self, response_id: str, model: str, stream: ResponseStream, instructions: str | None = None):
        """Write the response ``response_id`` of ``model``'s work, with its ``instructions``, into ``stream``."""
        self.id, self.model, self.stream, self.instructions = response_id, model, stream, instructions
        self.created_at = int(time.time())
        self.sequence = 0
        self.running = False
        self.ended = False
        self.output: list[dict] = []
        self.usage: dict | None = None
        self._next_index = 0
        self._open_calls: dict[str, dict] = {}
        self._end: Callable[[], None] | None = None
        self.earlier: ResponseWriter | None = None

    def snapshot(self, status: str, output: list[dict] | None = None, **fields: Any) -> dict:
        """Return the response object as a lifecycle event carries it."""
        return {
            "id": self.id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
            "instructions": self.instructions,
            "output": deepcopy(output or []),
            **deepcopy(fields),
        }

    def _emit(self, kind: str, **fields: Any) -> None:
        if self.ended:
            raise RuntimeError(f"Response {self.id} has ended; nothing more is written to it")
        self.sequence += 1
        self.stream.put({"type": kind, "sequence_number": self.sequence, **fields})

    def open_queued(self) -> None:
        """Write ``response.created`` with status ``queued``, then ``response.queued``."""
        self._emit("response.created", response=self.snapshot("queued"))
        self._emit("response.queued", response=self.snapshot("queued"))

    def open_running(self) -> None:
        """Write ``response.created``, then ``response.in_progress``, for a request that runs at once (one-shot)."""
        self._emit("response.created", response=self.snapshot("in_progress"))
        self.start()

    def start(self, previous_response_id: str | None = None) -> None:
        """Write ``response.in_progress``; ``previous_response_id`` names the response this one takes over.

        That id is this core's convention: the Live API names no taker of joined work.
        """
        taken = {"previous_response_id": previous_response_id} if previous_response_id else {}
        self._emit("response.in_progress", response=self.snapshot("in_progress", **taken))
        self.running = True

    def terminal(self, kind: str, *, status: str | None = None, **fields: Any) -> None:
        """Close what is still open, then write the terminal ``kind``; nothing is written after it."""
        for item in self._open_calls.values():
            self._done({**item, "status": "incomplete"})
        self._open_calls.clear()
        usage = {"usage": self.usage} if self.usage else {}
        self._emit(kind, response=self.snapshot(status or TERMINALS[kind], self.output, **usage, **fields))
        self.ended = True
        self._end = None
        self.stream.end()

    def _added(self, item: dict) -> int:
        index, self._next_index = self._next_index, self._next_index + 1
        self._emit("response.output_item.added", output_index=index, item=item)
        return index

    def _done(self, item: dict) -> None:
        index = item.pop("_index")
        self.output.append(item)
        self._emit("response.output_item.done", output_index=index, item=item)

    def message(self, text: str, phase: Phase) -> dict:
        """Write one whole assistant message, with its ``phase``, and return the item."""
        item_id = uid("msg")
        opened = {"id": item_id, "type": "message", "status": "in_progress", "role": "assistant", "phase": phase}
        index = self._added({**opened, "content": []})
        empty = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
        where = {"item_id": item_id, "output_index": index, "content_index": 0}
        self._emit("response.content_part.added", **where, part=empty)
        self._emit("response.output_text.delta", **where, delta=text, logprobs=[])
        self._emit("response.output_text.done", **where, text=text, logprobs=[])
        item = output_message(item_id, text, phase)
        self._emit("response.content_part.done", **where, part=item["content"][0])
        self._done({**item, "_index": index})
        return item

    def function_call(self, call: dict) -> dict:
        """Write one of the app's function calls whole, and return the item."""
        item = {
            "id": call.get("id") or uid("fc"),
            "type": "function_call",
            "status": "completed",
            "call_id": call.get("call_id", ""),
            "name": call.get("name", ""),
            "arguments": call.get("arguments") or "{}",
        }
        index = self._added({**item, "status": "in_progress", "arguments": ""})
        where = {"item_id": item["id"], "output_index": index}
        self._emit("response.function_call_arguments.delta", **where, delta=item["arguments"])
        self._emit("response.function_call_arguments.done", **where, name=item["name"], arguments=item["arguments"])
        self._done({**item, "_index": index})
        return item

    def mcp_started(self, call: dict, label: str) -> None:
        """Write a backend call's ``mcp_call`` item as it starts, ``in_progress``, labeled with its tool source."""
        item = {
            "id": uid("mcp"),
            "type": "mcp_call",
            "server_label": label,
            "name": call.get("name", ""),
            "arguments": call.get("arguments") or "{}",
            "output": None,
            "error": None,
            "status": "in_progress",
        }
        item["_index"] = self._added({key: value for key, value in item.items() if key != "_index"})
        self._open_calls[call.get("call_id", "")] = item
        self._emit("response.mcp_call.in_progress", item_id=item["id"], output_index=item["_index"])

    def mcp_ended(self, call_id: str, output: str, outcome: Outcome) -> None:
        """Write a backend call's end: its ``mcp_call`` event, then the item with its output or its error."""
        item = self._open_calls.pop(call_id, None)
        if item is None:
            return
        kind, status = MCP_ENDINGS[outcome]
        self._emit(kind, item_id=item["id"], output_index=item["_index"])
        error = {"type": "mcp_tool_execution_error", "content": output} if status != "completed" else None
        self._done({**item, "status": status, "output": output, "error": error})
        self.end_when_ready()

    def holds(self, call_id: str) -> bool:
        """Return whether the backend call ``call_id`` is open in this response."""
        return call_id in self._open_calls

    def end_after_calls(self, end: Callable[[], None]) -> None:
        """Run ``end``, which writes the last items and the terminal, once :meth:`end_when_ready` finds it ready."""
        self._end = end
        self.end_when_ready()

    def drop(self, call_ids: Collection[str]) -> None:
        """Close the open items of these calls ``incomplete``: they report no end here."""
        for call_id in [each for each in self._open_calls if each in call_ids]:
            self._done({**self._open_calls.pop(call_id), "status": "incomplete"})
        self.end_when_ready()

    def end_when_ready(self) -> None:
        """Run the waiting end once no backend call is open in it and the earlier response has ended."""
        if self._end is not None and not self._open_calls and (self.earlier is None or self.earlier.ended):
            end, self._end = self._end, None
            end()


class ResponseBook:
    """Every open response of one IM session, by delegation in acceptance order; only it writes a response."""

    def __init__(self, ledger: WorkLedger):
        """Keep the responses of the session whose records ``ledger`` holds."""
        self.ledger = ledger
        self._open: dict[str, ResponseWriter] = {}
        self._closing: list[ResponseWriter] = []
        self._dropped: set[str] = set()

    def open(
        self,
        delegation_id: str,
        stream: ResponseStream,
        model: str,
        instructions: str | None = None,
        *,
        running: bool = False,
    ) -> ResponseWriter:
        """Open the delegation's next response on ``stream``: queued at acceptance, or running at once (one-shot)."""
        writer = self._open[delegation_id] = ResponseWriter(uid("resp"), model, stream, instructions)
        self.ledger.new_response(writer.id, delegation_id)
        delegation = self.ledger.delegations.get(delegation_id)
        if delegation is not None:
            delegation.responses.append(writer.id)
        if running:
            writer.open_running()
            self.ledger.responses[writer.id].move("in_progress")
        else:
            writer.open_queued()
        return writer

    def late(self, delegation_id: str, stream: ResponseStream, model: str, instructions: str | None = None) -> None:
        """Answer a continuation whose Work has ended with an empty response that completes at once."""
        writer = ResponseWriter(uid("resp"), model, stream, instructions)
        writer.open_queued()
        writer.start()
        writer.terminal("response.completed")
        delegation = self.ledger.delegations.get(delegation_id)
        if delegation is not None:
            response = self.ledger.new_response(writer.id, delegation_id)
            response.move("in_progress")
            response.move("completed")
            delegation.responses.append(writer.id)

    def writer(self, delegation_id: str) -> ResponseWriter | None:
        """Return the delegation's open response, if it has one."""
        return self._open.get(delegation_id)

    def running(self, delegation_id: str) -> ResponseWriter | None:
        """Return the delegation's open response if it is in progress, so items may be written into it."""
        writer = self._open.get(delegation_id)
        return writer if writer is not None and writer.running else None

    def open_delegations(self) -> list[str]:
        """Return the delegations with an open response, in the order their requests were accepted."""
        return list(self._open)

    def start(self, delegation_id: str, previous: str | None = None) -> None:
        """Put the delegation's open response in progress (once), naming the response it takes over, if any."""
        writer = self._open.get(delegation_id)
        if writer is not None and not writer.running:
            writer.start(self.response_id(previous) if previous is not None else None)
            self.ledger.responses[writer.id].move("in_progress")

    def response_id(self, delegation_id: str) -> str | None:
        """Return the id of the delegation's open response, or else of its newest one."""
        writer = self._open.get(delegation_id)
        if writer is not None:
            return writer.id
        record = self.ledger.delegations.get(delegation_id)
        return record.responses[-1] if record is not None and record.responses else None

    def end(self, delegation_id: str, kind: str, **fields: Any) -> bool:
        """End the delegation's open response with ``kind``, and its record with it; ``False`` if none was open."""
        writer = self._open.pop(delegation_id, None)
        if writer is None:
            return False
        writer.terminal(kind, **fields)
        self.ledger.responses[writer.id].move(TERMINALS[kind])
        return True

    def end_after_calls(
        self,
        delegation_id: str,
        kind: str,
        *,
        last: Callable[[ResponseWriter], Any] | None = None,
        ended: Callable[[], Any] | None = None,
    ) -> None:
        """End the delegation's open response while its work goes on, once the backend calls open in it have ended.

        It takes no more items, so the next response may open, and it ends after the delegation's earlier ones: ``last``
        writes its last items, then its terminal ``kind``, then ``ended`` runs.
        """
        writer = self._open.pop(delegation_id, None)
        if writer is None:
            return
        earlier = [each for each in self._closing if self.ledger.responses[each.id].delegation == delegation_id]
        writer.earlier = earlier[-1] if earlier else None

        def end() -> None:
            if last is not None:
                last(writer)
            writer.terminal(kind)
            self.ledger.responses[writer.id].move(TERMINALS[kind])
            self._closing.remove(writer)
            if ended is not None:
                ended()
            for later in [each for each in self._closing if each.earlier is writer]:
                later.end_when_ready()

        self._closing.append(writer)
        writer.drop(self._dropped)
        writer.end_after_calls(end)

    def mcp_ended(self, call_id: str, output: str, outcome: Outcome) -> bool:
        """Write a backend call's end into the response whose end waits for it; ``False`` when none holds it."""
        writer = next((each for each in self._closing if each.holds(call_id)), None)
        if writer is None:
            return False
        writer.mcp_ended(call_id, output, outcome)
        return True

    def drop(self, call_ids: list[str]) -> None:
        """Let no response's end wait for these calls, which report no end; their items close ``incomplete``."""
        self._dropped.update(call_ids)
        for writer in list(self._closing):
            writer.drop(call_ids)

    def cancel_closing(self) -> None:
        """End each response whose end waits as a close ends an open one (:meth:`cancelled`), in acceptance order."""
        order = list(self.ledger.responses)
        closing, self._closing = sorted(self._closing, key=lambda writer: order.index(writer.id)), []
        for writer in closing:
            writer.terminal("response.incomplete", status="cancelled", incomplete_details=None)
            self.ledger.responses[writer.id].move(TERMINALS["response.incomplete"])

    def failed(self, delegation_id: str, code: str, message: str) -> bool:
        """End the delegation's open response ``response.failed`` with an official error code."""
        return self.end(delegation_id, "response.failed", error={"code": code, "message": message})

    def cancelled(self, delegation_id: str) -> bool:
        """End the delegation's open response at a close: incomplete, status ``cancelled``, no reason given."""
        return self.end(delegation_id, "response.incomplete", status="cancelled", incomplete_details=None)
