# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal events and the sinks they are written to.

Diagnostics never ride on :class:`~examples.frontend_backend_verdict.text.messages.AgentTurn`.
Frontend filler text, delegation queries, tool executions and contract
violations are *only* observable here, which is what makes "no intermediate
text reaches the caller" a property of the types rather than a promise.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from loguru import logger

#: Event kinds emitted by the agent.
DELEGATION = "delegation"
FILLER = "filler"
DIRECT_ANSWER = "direct_answer"
BACKEND_TOOL_CALLS = "backend_tool_calls"
TOOL_EXECUTED = "tool_executed"
BACKEND_FINAL = "backend_final"
FRONTEND_CONTRACT_VIOLATION = "frontend_contract_violation"
FRONTEND_REPAIR = "frontend_repair"
PENDING_DISCARDED = "pending_discarded"
ITERATION_CAP = "iteration_cap"
BACKEND_ERROR = "backend_error"
STEP_USAGE = "step_usage"
BACKEND_CONTEXT = "backend_context"


@dataclass(frozen=True, slots=True)
class InternalEvent:
    """One internal diagnostic event.

    ``timestamp`` is wall-clock emission time, not render time: it is stamped
    where the event happens, so a JSONL trace and a live REPL agree about when
    something occurred even if one of them is reading the other later.
    """

    kind: str
    session_id: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        return {
            "timestamp": self.timestamp,
            "time": time.strftime("%H:%M:%S", time.localtime(self.timestamp))
            + f".{int(self.timestamp % 1 * 1000):03d}",
            "kind": self.kind,
            "session_id": self.session_id,
            **self.data,
        }


@runtime_checkable
class EventSink(Protocol):
    """Destination for internal events."""

    def emit(self, event: InternalEvent) -> None:
        """Record one event."""


class NullSink:
    """Sink that discards every event."""

    def emit(self, event: InternalEvent) -> None:
        """Discard ``event``."""


class LoggingSink:
    """Sink that writes events to loguru. Safe to share across sessions."""

    def emit(self, event: InternalEvent) -> None:
        """Log ``event`` at INFO level."""
        logger.info(f"[fba:{event.kind}] {json.dumps(event.as_dict(), ensure_ascii=False, default=str)}")


class JsonlSink:
    """Append-only JSONL sink. Safe to share across sessions."""

    def __init__(self, path: str | Path) -> None:
        """Create the parent directory and prepare the append lock."""
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: InternalEvent) -> None:
        """Append ``event`` as one JSON line."""
        line = json.dumps(event.as_dict(), ensure_ascii=False, default=str)
        with self._lock, self._path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")


class CollectingSink:
    """In-memory sink for tests and the REPL.

    Single-session/test use only: it does not order events across concurrent
    sessions. Share a :class:`LoggingSink` or :class:`JsonlSink` instead.
    """

    def __init__(self) -> None:
        """Start with an empty event list."""
        self.events: list[InternalEvent] = []

    def emit(self, event: InternalEvent) -> None:
        """Store ``event`` in memory."""
        self.events.append(event)

    def kinds(self) -> list[str]:
        """Return the recorded event kinds, in order."""
        return [event.kind for event in self.events]

    def of_kind(self, kind: str) -> list[InternalEvent]:
        """Return every recorded event of ``kind``."""
        return [event for event in self.events if event.kind == kind]

    def drain(self) -> list[InternalEvent]:
        """Return and clear the recorded events."""
        events, self.events = self.events, []
        return events


def build_sink(kind: str, path: str = "") -> EventSink:
    """Build a sink from configuration (``logging``, ``jsonl``, or ``none``)."""
    if kind == "none":
        return NullSink()
    if kind == "jsonl":
        if not path:
            raise ValueError("event_sink: jsonl requires event_sink_path")
        return JsonlSink(path)
    return LoggingSink()
