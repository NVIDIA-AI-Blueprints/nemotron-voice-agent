# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in bounded scalar wall-time evidence; never a conversation clock.

Only the trusted server environment enables persistence. No transcripts, audio,
request messages, provider bodies, credentials, or exception messages are kept.
Workers append under a short lock; persistence happens outside provider work.
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
import uuid
from contextlib import suppress
from contextvars import ContextVar
from pathlib import Path

_CURRENT = ContextVar("offline_server_diagnostics", default=None)
_LABEL_FIELDS = frozenset(
    {
        "source",
        "phase",
        "processor",
        "metric_type",
        "response_id",
        "error_type",
        "outcome",
        "source_sha256",
        "boundary",
        "operation",
        "method",
        "result",
    }
)
_LABEL = re.compile(r"[A-Za-z0-9_.:#-]{1,128}\Z")
_NAME = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")


def current_diagnostics():
    """Return only this offline session's optional observer."""
    return _CURRENT.get()


def record(diagnostics, name: str, **fields):
    """Keep optional observation failures out of provider/lifecycle results."""
    if diagnostics is not None:
        with suppress(Exception):
            diagnostics.record(name, **fields)


def duration(diagnostics, name: str, elapsed_ns: int):
    """Accumulate local execution waits without logging each audio window."""
    if diagnostics is not None:
        with suppress(Exception):
            diagnostics.observe_duration(name, elapsed_ns / 1_000_000)


class ServerDiagnostics:
    """Retain numeric observations without backpressuring an actor or socket."""

    def __init__(self, *, clock=time.monotonic_ns, max_events=4096):
        """Bind one request-local clock and a bounded observation capacity."""
        if type(max_events) is not int or max_events <= 0:
            raise ValueError("Diagnostic capacity must be a positive integer")
        self.session_id = uuid.uuid4().hex
        self._clock = clock
        self._started = clock()
        self._lock = threading.Lock()
        self._events = []
        self._durations = {}
        self._max_events = max_events
        self._total = 0
        self._rejected = 0

    def record(self, name: str, **fields):
        """Append a scalar observation, including from a provider worker thread."""
        if not isinstance(name, str) or _NAME.fullmatch(name) is None:
            return
        safe = {}
        for key, value in fields.items():
            if not isinstance(key, str) or _NAME.fullmatch(key) is None:
                continue
            numeric = value is None or type(value) in (bool, int) or (type(value) is float and math.isfinite(value))
            label = key in _LABEL_FIELDS and isinstance(value, str) and _LABEL.fullmatch(value)
            if numeric or label:
                safe[key] = value
            else:
                with self._lock:
                    self._rejected += 1
        with self._lock:
            now = self._clock()
            self._total += 1
            if len(self._events) < self._max_events:
                self._events.append(
                    {"sequence": self._total, "name": name, "wall_elapsed_ms": (now - self._started) / 1e6, **safe}
                )

    def observe_duration(self, name: str, duration_ms: float):
        """Measure caller work without interpreting it as isolated model compute."""
        if not isinstance(name, str) or _NAME.fullmatch(name) is None:
            return
        if type(duration_ms) not in (float, int) or not math.isfinite(duration_ms) or duration_ms < 0:
            return
        with self._lock:
            if name not in self._durations and len(self._durations) >= 64:
                return
            stats = self._durations.setdefault(name, {"count": 0, "sum_ms": 0.0, "max_ms": 0.0})
            stats["count"] += 1
            stats["sum_ms"] += duration_ms
            stats["max_ms"] = max(stats["max_ms"], duration_ms)

    def snapshot(self):
        """Freeze evidence; incomplete provider/cleanup phases remain absent."""
        with self._lock:
            return {
                "schema": "nva.offline.scalar-metrics.v1",
                "session_id": self.session_id,
                "pid": os.getpid(),
                "started_monotonic_ns": self._started,
                "scope": "server_monotonic_wall_observations_not_pure_model_compute",
                "events": [dict(event) for event in self._events],
                "duration_summaries": {name: dict(stats) for name, stats in self._durations.items()},
                "total_events": self._total,
                "truncated": self._total > len(self._events),
                "rejected_fields": self._rejected,
                "full_provider_eof_or_cleanup_attested": False,
            }


def create_diagnostics():
    """Enable only through a job-owned directory supplied by the server operator."""
    root = os.environ.get("NVA_OFFLINE_SERVER_METRICS_DIR", "").strip()
    return (ServerDiagnostics(), Path(root)) if root else (None, None)


def bind_diagnostics(diagnostics):
    """Bind observation to the request context, including asyncio.to_thread."""
    return _CURRENT.set(diagnostics)


def reset_diagnostics(token):
    """Restore the caller's observer when this independent session closes."""
    _CURRENT.reset(token)


def persist_diagnostics(root: Path, diagnostics: ServerDiagnostics):
    """Publish only bounded scalar data using the existing private writer."""
    import json

    from .recorder import write_private_text

    write_private_text(root / (diagnostics.session_id + ".json"), json.dumps(diagnostics.snapshot(), allow_nan=False))
