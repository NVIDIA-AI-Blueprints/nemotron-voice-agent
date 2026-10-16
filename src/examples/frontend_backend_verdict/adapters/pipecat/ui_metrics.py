# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-turn latency and token metrics for the browser UI's metrics panel.

The turn manager already records each turn's latency breakdown (``turn_latency``) and each agent
step's LLM usage (``agent_turn_done``) in the event log. Browser UI runtimes wrap that log in
:class:`UiEventLog`, which still writes the file (if any) and also hands the records of a
registered UI session to :class:`TurnMetrics`. That session sends one RTVI ``metric-group``
message per answered turn, the shape the UI's metrics panel already shows. OpenAI Realtime
runtimes never use this module.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from loguru import logger

from examples.frontend_backend_verdict.core.engine.event_log import EventLog, SessionRoutingSink

Listener = Callable[[str, dict[str, Any]], None]

#: ``turn_latency`` fields shown in the panel: (field, label).
LATENCY_FIELDS = (
    ("user_stop_to_first_audio_ms", "Voice to voice"),
    ("user_stop_to_asr_final_ms", "ASR final"),
    ("asr_final_to_agent_start_ms", "Agent start"),
    ("agent_start_to_first_audio_ms", "Agent to first audio"),
)
#: The two LLM roles of an ``agent_turn_done`` record.
ROLES = (("frontend", "Frontend"), ("backend", "Backend"))
#: The ``source`` the panel shows for these groups.
SOURCE = "frontend-backend-verdict"


class UiEventLog(EventLog):
    """The shared event log, plus per-session listeners for the browser UI sessions."""

    def __init__(self, inner: EventLog) -> None:
        """Wrap ``inner``, which keeps writing its file (if it has one)."""
        super().__init__()
        self._inner = inner
        self._listeners: dict[str, Listener] = {}
        self._listeners_lock = threading.Lock()

    @property
    def path(self) -> Any:
        """The wrapped log's file, if any."""
        return self._inner.path

    def register(self, session_id: str, listener: Listener) -> None:
        """Hand ``session_id``'s records to ``listener``."""
        with self._listeners_lock:
            self._listeners[session_id] = listener

    def unregister(self, session_id: str) -> None:
        """Stop handing ``session_id``'s records out."""
        with self._listeners_lock:
            self._listeners.pop(session_id, None)

    def write(self, kind: str, session_id: str, data: dict[str, Any], *, timestamp: float | None = None) -> None:
        """Write the record, then give it to the session's listener; a listener never raises into the caller."""
        self._inner.write(kind, session_id, data, timestamp=timestamp)
        with self._listeners_lock:
            listener = self._listeners.get(session_id)
        if listener is None:
            return
        try:
            listener(kind, data)
        except Exception as exc:  # noqa: BLE001 - metrics must never break a turn
            logger.warning(f"[{session_id}] UI metrics listener failed: {exc}")


def install(runtime: Any) -> None:
    """Give a new browser UI runtime a :class:`UiEventLog` (call before any session is created)."""
    sink = runtime.state.routing_sink
    if isinstance(sink.event_log, UiEventLog):
        return
    runtime.state.routing_sink = SessionRoutingSink(UiEventLog(sink.event_log))


class TurnMetrics:
    """Builds one ``metric-group`` message per turn from that turn's event-log records."""

    def __init__(self, send: Callable[[dict[str, Any]], None]) -> None:
        """``send`` gets each finished message (called on the writer's thread)."""
        self._send = send
        self._usage: dict[Any, dict[str, float]] = {}
        self._turns = 0

    def __call__(self, kind: str, data: dict[str, Any]) -> None:
        """Take one event-log record."""
        if kind == "agent_turn_done":
            totals = self._usage.setdefault(data.get("turn_id"), {})
            for role, _ in ROLES:
                record = data.get(role)
                if not isinstance(record, dict):
                    continue
                for field in ("latency_ms", "prompt_tokens", "completion_tokens"):
                    value = record.get(field)
                    if isinstance(value, int | float):
                        totals[f"{role}_{field}"] = totals.get(f"{role}_{field}", 0) + value
        elif kind == "turn_latency":
            message = self._message(data, self._usage.pop(data.get("turn_id"), {}))
            if message is not None:
                self._send(message)

    def _message(self, latency: dict[str, Any], usage: dict[str, float]) -> dict[str, Any] | None:
        metrics: list[dict[str, Any]] = [
            {"key": field, "label": label, "value": latency[field], "unit": "ms"}
            for field, label in LATENCY_FIELDS
            if isinstance(latency.get(field), int | float)
        ]
        for role, label in ROLES:
            unit_labels = (
                ("latency_ms", f"{label} LLM", "ms"),
                ("prompt_tokens", f"{label} input tokens", "tokens"),
                ("completion_tokens", f"{label} output tokens", "tokens"),
            )
            for field, metric_label, unit in unit_labels:
                key = f"{role}_{field}"
                if key in usage:
                    metrics.append({"key": key, "label": metric_label, "value": round(usage[key], 1), "unit": unit})
        if not metrics:
            return None
        self._turns += 1
        turn_id = latency.get("turn_id")
        return {
            "type": "metric-group",
            "group_id": f"fbv-turn-{turn_id if turn_id is not None else self._turns}",
            "group_label": f"Turn {self._turns}",
            "category": "latency",
            "source": SOURCE,
            "metrics": metrics,
        }
