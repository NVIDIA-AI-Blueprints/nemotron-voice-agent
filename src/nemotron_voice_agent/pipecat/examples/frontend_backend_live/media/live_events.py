# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the repository UI bot shows the OpenAI-format events of its delegations: ``live-event`` RTVI messages.

The bot's server delegation talks to the delegation core through a Responses client: it sends requests and reads the
official streaming events (``response.created``, ``response.output_text.delta``, ``response.completed``, ...). A
:class:`TappedResponsesClient` around that client tells :class:`RtviLiveEvents` about every request and every event, and
the UI's Event log prints them. The tap only reads: the core and the server delegation get the same objects as without
it, and a failure of the log is dropped, never raised into a call.

The relay is off unless the example's registry entry lists :data:`RELAY_CAPABILITY` (:func:`relays_events`), so an
example that does not show the Event log sends nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

import nemotron_voice_agent.pipecat.runtime.examples_registry as examples_registry
from nemotron_voice_agent.pipecat.shared.delegation_contract import DELEGATION_ID

MESSAGE = "live-event"
VERSION = 1
# The registry capability that turns the relay on: the UI's Event log tab.
RELAY_CAPABILITY = "event_log"
# One string of an event, in characters (a tool's output fits); a longer one is cut and says how much was left out.
MAX_TEXT = 2000
# One list of an event, in items.
MAX_ITEMS = 20
# One event as sent, in characters of JSON; a larger one is replaced by a marker that says how large it was.
MAX_EVENT_CHARS = 12000
# A server tool call's ``arguments`` and ``output`` are what a developer reads the log for, so they are kept whole. Such
# an event may be this large in all; a longer field is cut, as little as the event needs to fit.
MAX_TOOL_EVENT_CHARS = 64000
TOOL_FIELDS = ("arguments", "output")
# The events held for a client that has not connected yet or reads slowly; the oldest go first, and ``seq`` shows it.
MAX_BACKLOG = 500
MAX_DEPTH = 8
# What a request and a response snapshot carry that the log shows as a size or a name, not in full: the backend's prompt
# and tool schemas, which the repository UI's Prompts and Tools tabs show already.
SIZED = ("instructions",)
NAMED = ("tools",)
# A response snapshot's ``input`` repeats the request, which has its own row.
REPEATED = ("input",)


def clip(value: Any, depth: int = 0) -> Any:
    """Return a copy of ``value`` with every string, list and depth bounded, so one event stays small."""
    if isinstance(value, str):
        if len(value) <= MAX_TEXT:
            return value
        return f"{value[:MAX_TEXT]}… (+{len(value) - MAX_TEXT} chars)"
    if depth >= MAX_DEPTH:
        return "…"
    if isinstance(value, dict):
        return {str(key): clip(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [clip(item, depth + 1) for item in value[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            items.append(f"… +{len(value) - MAX_ITEMS} more")
        return items
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return clip(str(value), depth)


def tool_names(tools: Any) -> list[str]:
    """Return the names of a request's tools (or their types, for a tool with no name)."""
    names = []
    for tool in tools if isinstance(tools, list) else []:
        if isinstance(tool, dict):
            names.append(str(tool.get("name") or (tool.get("function") or {}).get("name") or tool.get("type") or "?"))
    return names


def is_tool_call(item: Any) -> bool:
    """Whether ``item`` is a server tool call of the core, an ``mcp_call`` item."""
    return isinstance(item, dict) and item.get("type") == "mcp_call"


def _sized(value: Any) -> Any:
    return f"<{len(value):,} chars>" if isinstance(value, str) else value


def tool_item(item: dict[str, Any], *, whole: bool, limit: int = MAX_TOOL_EVENT_CHARS) -> dict[str, Any]:
    """Return a tool call item with its ``arguments`` and ``output`` whole (each cut at ``limit``) or as sizes."""
    kept = clip({key: value for key, value in item.items() if key not in TOOL_FIELDS})
    for field in TOOL_FIELDS:
        if field not in item:
            continue
        value = item[field]
        if not whole:
            kept[field] = _sized(value)
        elif isinstance(value, str) and len(value) > limit:
            kept[field] = f"{value[:limit]}… (+{len(value) - limit} chars)"
        else:
            kept[field] = value
    return kept


def shrink(body: dict[str, Any], *, snapshot: bool = False) -> dict[str, Any]:
    """Return ``body`` with its prompt as a size and its tools as names, then clipped.

    A response ``snapshot`` also leaves out its ``input``, which only repeats the request, and shows the ``arguments``
    and ``output`` of its tool calls as sizes, since the item's own event carried them whole; a request keeps its
    input.
    """
    shrunk: dict[str, Any] = {}
    for key, value in body.items():
        if snapshot and key in REPEATED:
            continue
        if key in SIZED and isinstance(value, str):
            shrunk[key] = f"<{len(value):,} chars>"
        elif key in NAMED:
            shrunk[key] = tool_names(value)
        elif snapshot and key == "output" and isinstance(value, list):
            shrunk[key] = [tool_item(i, whole=False) if is_tool_call(i) else clip(i) for i in value[:MAX_ITEMS]]
        else:
            shrunk[key] = clip(value)
    return shrunk


def compact_event(event: dict[str, Any]) -> dict[str, Any]:
    """Return what the log keeps of one streamed event: its response snapshot is shrunk, and the whole is bounded.

    A tool call item's ``arguments`` and ``output`` are kept whole, and such an event may be larger. When it is still
    too large, its fields are cut shorter until it fits, so the log keeps the tool's name and the start of each field.
    """
    event = dict(event)
    if isinstance(event.get("response"), dict):
        event["response"] = shrink(event["response"], snapshot=True)
    call = event.get("item") if is_tool_call(event.get("item")) else None
    if call is not None:
        event["item"] = {}
    compacted = clip(event)
    size = len(json.dumps(compacted, default=str))
    limit = MAX_TOOL_EVENT_CHARS
    while call is not None:
        compacted["item"] = tool_item(call, whole=True, limit=limit)
        size = len(json.dumps(compacted, default=str))
        if size <= MAX_TOOL_EVENT_CHARS or limit <= MAX_TEXT:
            break
        limit //= 2
    if size > (MAX_TOOL_EVENT_CHARS if call is not None else MAX_EVENT_CHARS):
        return {"type": str(event.get("type", "")), "truncated": True, "chars": size}
    return compacted


def relays_events(example_key: str) -> bool:
    """Whether the example's registry entry asks for the event relay: it lists ``RELAY_CAPABILITY``."""
    return RELAY_CAPABILITY in examples_registry.find(example_key, ignore_lock=True)["capabilities"]


@dataclass(frozen=True)
class StreamTag:
    """Which response an event belongs to: its label in this call (``r1``, ``r2``, ...) and its delegation."""

    label: str
    delegation: str | None


class RtviLiveEvents:
    """Sends the delegations' OpenAI-format events to the repository UI as ``live-event`` messages, oldest first.

    Events are kept (the newest :data:`MAX_BACKLOG`) until :meth:`start` gives the pipeline worker's ``queue_frame``,
    when the client has connected: the conversation the bot opened before that is then the first event the UI sees. One
    task sends them in order, so a slow client never blocks the delegation. After :meth:`close` an event is dropped.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        """Hold no event; ``t`` of every message counts seconds from now."""
        self.clock, self.started_at = clock, clock()
        self._seq = 0
        self._streams = 0
        self._closed = False
        self._backlog: deque[dict[str, Any]] = deque(maxlen=MAX_BACKLOG)
        self._queue_frame: Callable[[Frame], Awaitable[None]] | None = None
        self._wake: asyncio.Event | None = None
        self._sender: asyncio.Task | None = None

    # ------------------------------------------------------------------------------------------------- recording

    def api(self, kind: str, **payload: Any) -> None:
        """Record a call of the core's other endpoints, such as ``conversations.create``."""
        self._record("out", kind, None, None, shrink(payload))

    def request(self, params: dict[str, Any]) -> StreamTag:
        """Record a ``responses.create`` request and return the tag of the response it starts."""
        self._streams += 1
        metadata = params.get("metadata") if isinstance(params.get("metadata"), dict) else {}
        delegation = metadata.get(DELEGATION_ID)
        tag = StreamTag(f"r{self._streams}", str(delegation) if delegation else None)
        self._record("out", "responses.create", tag.label, tag.delegation, shrink(params))
        return tag

    def event(self, tag: StreamTag, event: Any) -> None:
        """Record one event of a response's stream."""
        if not isinstance(event, dict):
            return
        self._record("in", str(event.get("type") or "event"), tag.label, tag.delegation, compact_event(event))

    def failure(self, tag: StreamTag, error: BaseException) -> None:
        """Record that the core refused a request."""
        body = {"type": "error", "message": clip(str(error)), "error_type": type(error).__name__}
        self._record("in", "error", tag.label, tag.delegation, body)

    def _record(
        self, direction: str, kind: str, stream: str | None, delegation: str | None, body: dict[str, Any]
    ) -> None:
        """Number one event and queue its message; a failure is logged at debug level and never raised."""
        if self._closed:
            return
        try:
            self._seq += 1
            self._backlog.append(
                {
                    "type": MESSAGE,
                    "v": VERSION,
                    "seq": self._seq,
                    "t": round(self.clock() - self.started_at, 3),
                    "direction": direction,
                    "kind": kind,
                    "stream": stream,
                    "delegation": delegation,
                    "event": body,
                }
            )
            if self._wake is not None:
                self._wake.set()
        except Exception:
            logger.opt(exception=True).debug("The live event log could not record an event")

    # ------------------------------------------------------------------------------------------------- sending

    def start(self, queue_frame: Callable[[Frame], Awaitable[None]]) -> None:
        """Start sending with the pipeline worker's ``queue_frame``, beginning with the events kept so far."""
        if self._sender is not None:
            return
        self._queue_frame = queue_frame
        self._wake = asyncio.Event()
        if self._backlog:
            self._wake.set()
        self._sender = asyncio.get_running_loop().create_task(self._send_all(), name="live-event-sender")

    async def _send_all(self) -> None:
        wake, queue_frame = self._wake, self._queue_frame
        while wake is not None and queue_frame is not None:
            await wake.wait()
            wake.clear()
            while self._backlog and self._queue_frame is not None:
                message = self._backlog.popleft()
                try:
                    await queue_frame(RTVIServerMessageFrame(data=message))
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.opt(exception=True).debug("The live event log could not send an event")

    def close(self) -> None:
        """Stop sending; later events are dropped and the ones still waiting are forgotten."""
        self._closed = True
        self._queue_frame = None
        self._backlog.clear()
        if self._sender is not None:
            self._sender.cancel()
        self._sender = None
        self._wake = None


class TappedStream:
    """A response's event stream that also tells the log of every event, and yields each one unchanged."""

    def __init__(self, inner: Any, on_event: Callable[[Any], None]) -> None:
        """Read ``inner``; ``on_event`` is called with each event before it is yielded."""
        self._inner, self._on_event = inner, on_event

    def __aiter__(self):
        """Iterate the inner stream's events."""
        return self._events()

    async def _events(self):
        async for event in self._inner:
            self._tell(event)
            yield event

    def _tell(self, event: Any) -> None:
        try:
            self._on_event(event)
        except Exception:
            logger.opt(exception=True).debug("The live event log could not take an event")

    async def aclose(self) -> None:
        """Stop reading the inner stream."""
        await self._inner.aclose()


def tapped(client: Any, log: RtviLiveEvents | None) -> Any:
    """Return ``client`` tapped by ``log``, or ``client`` itself when there is no log (the relay is off)."""
    return client if log is None else TappedResponsesClient(client, log)


class TappedResponsesClient:
    """A Responses client that tells a :class:`RtviLiveEvents` of its requests and its events, and changes nothing else.

    It wraps the delegation core's client for the server delegation. The endpoints it does not log (and any attribute,
    such as the core's close budget) are the wrapped client's own.
    """

    def __init__(self, inner: Any, log: RtviLiveEvents) -> None:
        """Wrap ``inner``, telling ``log``."""
        self._inner, self._log = inner, log

    def __getattr__(self, name: str) -> Any:
        """Give the wrapped client's own attribute."""
        return getattr(self._inner, name)

    def _note(self, kind: str, **payload: Any) -> None:
        with contextlib.suppress(Exception):
            self._log.api(kind, **payload)

    async def create_conversation(self, metadata: dict[str, str]) -> dict:
        """Open a conversation, and log it."""
        conversation = await self._inner.create_conversation(metadata)
        self._note("conversations.create", id=conversation.get("id"), metadata=metadata)
        return conversation

    async def update_conversation(self, conversation_id: str, metadata: dict[str, str]) -> dict:
        """Attach to a kept conversation, and log it."""
        conversation = await self._inner.update_conversation(conversation_id, metadata)
        self._note("conversations.update", id=conversation_id, metadata=metadata)
        return conversation

    async def delete_conversation(self, conversation_id: str) -> dict:
        """Delete a conversation, and log it."""
        result = await self._inner.delete_conversation(conversation_id)
        self._note("conversations.delete", id=conversation_id, deleted=result.get("deleted"))
        return result

    async def create_response(self, params: dict) -> TappedStream:
        """Send a request, log it, and return its stream, whose events are logged as they are read."""
        try:
            tag = self._log.request(params)
        except Exception:
            tag = None
        try:
            stream = await self._inner.create_response(params)
        except Exception as refused:
            if tag is not None:
                with contextlib.suppress(Exception):
                    self._log.failure(tag, refused)
            raise
        if tag is None:
            return stream
        return TappedStream(stream, lambda event: self._log.event(tag, event))


__all__ = [
    "MESSAGE",
    "RtviLiveEvents",
    "StreamTag",
    "TappedResponsesClient",
    "TappedStream",
    "clip",
    "compact_event",
    "shrink",
    "tool_names",
]
