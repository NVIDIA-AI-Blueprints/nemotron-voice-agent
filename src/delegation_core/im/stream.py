# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Responses on the wire: each request's stream, the writer of one response, and the book of a session's responses.

A :class:`ResponseWriter` writes one response in the official order: ``response.created`` and ``response.queued`` at
acceptance (``in_progress`` at once for a one-shot request), ``response.in_progress`` when work runs, whole items with
their own ``output_index``, then exactly one terminal. A response that takes over an older delegation's work names that
delegation's open response in ``previous_response_id``, a convention of this core: GPT-Live names no taker of joined
work.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from copy import deepcopy
from typing import Any

from delegation_core.backend.port import Outcome, Phase
from delegation_core.ids import uid
from delegation_core.ledger.records import WorkLedger

# A response's terminal events, and the status each gives the response.
TERMINALS = {"response.completed": "completed", "response.failed": "failed", "response.incomplete": "incomplete"}
# How a backend-run call ended, as its ``mcp_call`` event and item status say it.
MCP_ENDINGS = {
    "done": ("response.mcp_call.completed", "completed"),
    "failed": ("response.mcp_call.failed", "failed"),
    # it may have run: the call did not complete, and its outcome is not known (its output says so)
    "unclear": ("response.mcp_call.failed", "incomplete"),
}


def message_text(item: dict) -> str:
    """The text of a message item: its ``output_text`` parts, joined (a plain string content is taken whole)."""
    content = item.get("content")
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") for part in content or [] if isinstance(part, dict) and part.get("type") == "output_text"
    )


def output_message(item_id: str, text: str, phase: Phase, status: str = "completed") -> dict:
    """An assistant message item whose one ``output_text`` part holds ``text``."""
    part = {"type": "output_text", "text": text, "annotations": [], "logprobs": []}
    return {"id": item_id, "type": "message", "status": status, "role": "assistant", "phase": phase, "content": [part]}


class ResponseStream:
    """The events of one request's response, in order, as the request reads them.

    An async iterator: it ends after the response's terminal, or once the IM session ended.
    """

    def __init__(self):
        """Start empty and open."""
        self._events: deque[dict] = deque()
        self._ready = asyncio.Event()
        self.ended = False

    @property
    def unread(self) -> int:
        """How many events were written and not read yet."""
        return len(self._events)

    def put(self, event: dict) -> None:
        """Add an event for the reader; nothing is added once the stream has ended."""
        if not self.ended:
            self._events.append(event)
            self._ready.set()

    def end(self) -> None:
        """End the stream: the reader gets what was written before, then the iteration stops. Safe to call twice."""
        self.ended = True
        self._ready.set()

    async def aclose(self) -> None:
        """The reader stops reading. The response goes on: a conversation's work does not stop with its reader."""

    def __aiter__(self) -> ResponseStream:
        """Return the stream itself."""
        return self

    async def __anext__(self) -> dict:
        """The next event; ``StopAsyncIteration`` once the stream has ended and every event was read."""
        while not self._events:
            if self.ended:
                raise StopAsyncIteration
            self._ready.clear()
            await self._ready.wait()
        return self._events.popleft()


class ResponseWriter:
    """Writes one response's events into a :class:`ResponseStream`: numbered, whole items, exactly one terminal."""

    def __init__(self, response_id: str, model: str, stream: ResponseStream, instructions: str | None = None):
        """Write the response ``response_id`` of ``model``'s work into ``stream``.

        ``instructions`` are what the response's request runs on, as the IM composed them: every snapshot
        reports them, as the official response object does.
        """
        self.id, self.model, self.stream, self.instructions = response_id, model, stream, instructions
        self.created_at = int(time.time())
        self.sequence = 0
        self.running = False  # response.in_progress was written
        self.ended = False
        self.output: list[dict] = []  # the items finished, in order: the terminal's snapshot output
        self.usage: dict | None = None  # the newest usage a round of its work reported
        self._next_index = 0  # the output index of the next item
        self._open_calls: dict[str, dict] = {}  # each backend-run call's open mcp_call item, by call_id

    # ------------------------------------------------------------------------------------- lifecycle
    def snapshot(self, status: str, output: list[dict] | None = None, **fields: Any) -> dict:
        """The response object as a lifecycle event carries it."""
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
        """Accepted: ``response.created`` with status ``queued``, then ``response.queued``."""
        self._emit("response.created", response=self.snapshot("queued"))
        self._emit("response.queued", response=self.snapshot("queued"))

    def open_running(self) -> None:
        """Accepted and running at once (a one-shot request): ``response.created``, then ``response.in_progress``."""
        self._emit("response.created", response=self.snapshot("in_progress"))
        self.start()

    def start(self, previous_response_id: str | None = None) -> None:
        """``response.in_progress``; ``previous_response_id`` names the older response this one takes over."""
        taken = {"previous_response_id": previous_response_id} if previous_response_id else {}
        self._emit("response.in_progress", response=self.snapshot("in_progress", **taken))
        self.running = True

    def terminal(self, kind: str, *, status: str | None = None, **fields: Any) -> None:
        """Close what is still open, then write the terminal ``kind``; nothing is written after it.

        ``status`` is the snapshot's status when it is not the one ``kind`` gives (``cancelled`` at a close). The
        response's newest round usage goes into the snapshot when one was reported.
        """
        for item in self._open_calls.values():
            self._done({**item, "status": "incomplete"})
        self._open_calls.clear()
        usage = {"usage": self.usage} if self.usage else {}
        self._emit(kind, response=self.snapshot(status or TERMINALS[kind], self.output, **usage, **fields))
        self.ended = True
        self.stream.end()

    # ------------------------------------------------------------------------------------- items
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
        """A backend-run call starts: its ``mcp_call`` item, ``in_progress``, labeled with its tool source."""
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
        """A backend-run call ended: its ``mcp_call`` event, then the item with its output or its error."""
        item = self._open_calls.pop(call_id, None)
        if item is None:
            return
        kind, status = MCP_ENDINGS[outcome]
        self._emit(kind, item_id=item["id"], output_index=item["_index"])
        error = {"type": "mcp_tool_execution_error", "content": output} if status != "completed" else None
        self._done({**item, "status": status, "output": output, "error": error})


class ResponseBook:
    """Every open response of one IM session, by delegation (in the order their requests were accepted).

    It is the only code that writes a response and moves its ``Response`` record: each response opens, goes
    ``in_progress`` once, and ends once. A delegation has at most one open response.
    """

    def __init__(self, ledger: WorkLedger):
        """Keep the responses of the session whose records ``ledger`` holds."""
        self.ledger = ledger
        self._open: dict[str, ResponseWriter] = {}

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
        """A continuation whose Work has ended: its response opens, goes in progress, ends ``completed``, empty.

        It is no open response of the delegation, which may have none left; its record is kept when the delegation is.
        """
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
        """The delegation's open response, if it has one."""
        return self._open.get(delegation_id)

    def running(self, delegation_id: str) -> ResponseWriter | None:
        """The delegation's open response if it is in progress, so items may be written into it."""
        writer = self._open.get(delegation_id)
        return writer if writer is not None and writer.running else None

    def open_delegations(self) -> list[str]:
        """The delegations with an open response, in the order their requests were accepted."""
        return list(self._open)

    def start(self, delegation_id: str, previous: str | None = None) -> None:
        """The delegation's open response goes ``in_progress`` (once), naming the response it takes over, if any."""
        writer = self._open.get(delegation_id)
        if writer is not None and not writer.running:
            writer.start(self.response_id(previous) if previous is not None else None)
            self.ledger.responses[writer.id].move("in_progress")

    def response_id(self, delegation_id: str) -> str | None:
        """The id of the delegation's open response, or else of its newest one (one that ended with the app's calls)."""
        writer = self._open.get(delegation_id)
        if writer is not None:
            return writer.id
        record = self.ledger.delegations.get(delegation_id)
        return record.responses[-1] if record is not None and record.responses else None

    def end(self, delegation_id: str, kind: str, **fields: Any) -> bool:
        """End the delegation's open response with ``kind``; its record moves with it. ``False`` if none was open."""
        writer = self._open.pop(delegation_id, None)
        if writer is None:
            return False
        writer.terminal(kind, **fields)
        self.ledger.responses[writer.id].move(TERMINALS[kind])
        return True

    def failed(self, delegation_id: str, code: str, message: str) -> bool:
        """End the delegation's open response ``response.failed`` with an official error code."""
        return self.end(delegation_id, "response.failed", error={"code": code, "message": message})

    def cancelled(self, delegation_id: str) -> bool:
        """End the delegation's open response at a close: incomplete, status ``cancelled``, no reason given."""
        return self.end(delegation_id, "response.incomplete", status="cancelled", incomplete_details=None)
