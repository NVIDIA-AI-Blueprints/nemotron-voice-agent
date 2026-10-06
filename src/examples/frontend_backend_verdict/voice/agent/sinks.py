# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One shared event sink for every session, and the JSONL event log.

The text prototype emits internal events (delegation, filler, backend tool
calls, usage) to an ``EventSink`` stamped with ``SessionState.session_id``.
:class:`SessionRoutingSink` is that sink: it writes each event to the event log
and hands it to the listener registered for the session. ``emit`` never blocks
or awaits, because the text prototype calls it synchronously from inside
``agent.send()``; listeners must only schedule work.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from examples.frontend_backend_verdict.text.events import InternalEvent

#: Keys dropped from event-log records when ``logging.redact_content`` is on.
_CONTENT_KEYS = frozenset(
    {
        "text",
        "query",
        "transcript",
        "result",
        "arguments",
        "output",
        "calls",
        "rejected_text",
        # barge-in frontend verdict (user speech and the running/probe queries)
        "utterance",
        "merged_text",
        "running_query",
        "probe_query",
        # normalization events (identifiers are user content)
        "raw",
        "spans",
        "before",
        "after",
        "value",
    }
)


class EventListener(Protocol):
    """Per-session receiver of the text prototype's internal events."""

    def on_internal_event(self, event: InternalEvent) -> None:
        """Handle one event without blocking."""


class EventLog:
    """Append-only JSONL log shared by all sessions (thread-safe); a no-op without a path."""

    def __init__(self, path: str = "", *, redact_content: bool = False) -> None:
        """Open lazily; the parent directory is created on first write."""
        self._path = Path(path) if path else None
        self._redact = redact_content
        self._lock = threading.Lock()

    @property
    def path(self) -> Path | None:
        """The JSONL file, if any."""
        return self._path

    def write(self, kind: str, session_id: str, data: dict[str, Any], *, timestamp: float | None = None) -> None:
        """Append one record."""
        if self._path is None:
            return
        payload = {k: v for k, v in data.items() if not (self._redact and k in _CONTENT_KEYS)}
        record = {"timestamp": timestamp if timestamp is not None else time.time(), "kind": kind}
        record["session_id"] = session_id
        record.update(payload)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(f"{line}\n")


class SessionRoutingSink:
    """``EventSink`` that routes the text prototype's events to per-session listeners."""

    def __init__(self, event_log: EventLog | None = None) -> None:
        """Share ``event_log`` with the voice layer."""
        self._event_log = event_log or EventLog()
        self._listeners: dict[str, EventListener] = {}
        self._lock = threading.Lock()

    @property
    def event_log(self) -> EventLog:
        """The shared event log."""
        return self._event_log

    def register(self, session_id: str, listener: EventListener) -> None:
        """Route events for ``session_id`` to ``listener``."""
        with self._lock:
            self._listeners[session_id] = listener

    def unregister(self, session_id: str) -> None:
        """Stop routing events for ``session_id``."""
        with self._lock:
            self._listeners.pop(session_id, None)

    def emit(self, event: InternalEvent) -> None:
        """Log and route one event; never raises into the agent."""
        try:
            self._event_log.write(event.kind, event.session_id, dict(event.data), timestamp=event.timestamp)
            with self._lock:
                listener = self._listeners.get(event.session_id)
            if listener is not None:
                listener.on_internal_event(event)
        except Exception as exc:  # noqa: BLE001 - diagnostics must never break a turn
            logger.warning(f"event routing failed for {event.kind}: {exc}")
