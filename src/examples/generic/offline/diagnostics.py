# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in bounded scalar wall-time evidence; never a conversation clock.

Only the trusted server environment enables persistence. No transcripts, audio,
request messages, provider bodies, credentials, or exception messages are kept.
Workers append under a short lock; persistence happens outside provider work.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import threading
import time
import uuid
from concurrent.futures import Executor
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from pathlib import Path

_CURRENT = ContextVar("offline_server_diagnostics", default=None)
_VAD_WINDOW = ContextVar("offline_vad_profile_window", default=None)
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
        "lease_sha256",
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


@contextmanager
def observe_vad_model():
    """Observe calling-thread CPU and wall time without interpreting model output."""
    window = _VAD_WINDOW.get()
    if window is None:
        yield
        return
    started, cpu = time.monotonic_ns(), time.thread_time_ns()
    try:
        yield
    finally:
        with suppress(Exception):
            window.model_finished(time.monotonic_ns() - started, time.thread_time_ns() - cpu)


class ProfiledExecutor(Executor):
    """Observe an existing executor's exact submission boundary; keep its policy."""

    def __init__(self, delegate):
        """Retain the native executor rather than creating another thread pool."""
        self._delegate = delegate

    def submit(self, fn, /, *args, **kwargs):
        """Preserve the native future, return value, error and cancellation."""
        window = _VAD_WINDOW.get()
        if window is None:
            return self._delegate.submit(fn, *args, **kwargs)
        with suppress(Exception):
            window.submitted()

        def invoke():
            token = _VAD_WINDOW.set(window)
            with suppress(Exception):
                window.executor_started()
            try:
                return fn(*args, **kwargs)
            finally:
                with suppress(Exception):
                    window.executor_finished()
                _VAD_WINDOW.reset(token)

        try:
            future = self._delegate.submit(invoke)
        except BaseException:
            with suppress(Exception):
                window.future_finished()
            raise

        def finished(value):
            if value.cancelled():
                with suppress(Exception):
                    window.future_finished()

        future.add_done_callback(finished)
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        """Forward the native shutdown contract, including wait=False cleanup."""
        return self._delegate.shutdown(wait=wait, cancel_futures=cancel_futures)


class VADWindowProfile:
    """One bounded window observer shared by its coroutine and executor thread."""

    def __init__(self, diagnostics, window_id, source_end_sample):
        """Keep only scalar source pointers and measured execution stages."""
        self.diagnostics = diagnostics
        self.fields = {
            "window_id": window_id,
            "source_end_sample": source_end_sample,
            "analyze_started_elapsed_ns": diagnostics.elapsed_ns(),
            "submitted_elapsed_ns": None,
            "executor_started_elapsed_ns": None,
            "executor_finished_elapsed_ns": None,
            "event_loop_resumed_elapsed_ns": None,
            "executor_wall_ns": None,
            "executor_thread_cpu_ns": None,
            "model_wall_ns": 0,
            "model_thread_cpu_ns": 0,
            "model_calls": 0,
            "awaiter_cancelled": False,
            "outcome": "pending",
        }
        self._cpu_started = None
        self._future_done = False

    def submitted(self):
        """Record immediately before delegating the native executor submission."""
        with self.diagnostics._lock:
            self.fields["submitted_elapsed_ns"] = self.diagnostics.elapsed_ns()

    def executor_started(self):
        """Record actual thread entry before the unchanged native callable."""
        self._cpu_started = time.thread_time_ns()
        entered = self.diagnostics.elapsed_ns()
        self.diagnostics.update_vad_window(self, executor_started_elapsed_ns=entered)
        submitted = self.fields["submitted_elapsed_ns"]
        if submitted is not None:
            duration(self.diagnostics, "vad_executor_queue", entered - submitted)

    def model_finished(self, wall_ns, cpu_ns):
        """Accumulate nested model-body spans, including exceptional calls."""
        with self.diagnostics._lock:
            self.fields["model_wall_ns"] += wall_ns
            self.fields["model_thread_cpu_ns"] += cpu_ns
            self.fields["model_calls"] += 1
        duration(self.diagnostics, "vad_model_profile_wall", wall_ns)
        duration(self.diagnostics, "vad_model_thread_cpu", cpu_ns)

    def executor_finished(self):
        """Retain unfinished-after-cancel work without waiting for it."""
        ended = self.diagnostics.elapsed_ns()
        cpu = time.thread_time_ns() - self._cpu_started if self._cpu_started is not None else None
        started = self.fields["executor_started_elapsed_ns"]
        self.diagnostics.update_vad_window(
            self,
            executor_finished_elapsed_ns=ended,
            executor_wall_ns=ended - started if started is not None else None,
            executor_thread_cpu_ns=cpu,
        )
        if cpu is not None:
            duration(self.diagnostics, "vad_executor_thread_cpu", cpu)
        if started is not None:
            duration(self.diagnostics, "vad_executor_profile_wall", ended - started)
        self.future_finished()

    def future_finished(self):
        """Mark body completion or cancellation before thread entry."""
        self.diagnostics.update_vad_window(self, future_done=True)

    def resumed(self, outcome):
        """Record owning-loop resumption; cancelled awaits can precede thread exit."""
        resumed = self.diagnostics.elapsed_ns()
        self.diagnostics.update_vad_window(
            self,
            event_loop_resumed_elapsed_ns=resumed,
            awaiter_cancelled=outcome == "cancelled",
            outcome=outcome,
        )
        ended = self.fields["executor_finished_elapsed_ns"]
        if outcome != "cancelled" and ended is not None and resumed >= ended:
            duration(self.diagnostics, "vad_executor_finish_to_loop_resume", resumed - ended)


@contextmanager
def profile_vad_window(diagnostics, source_end_sample):
    """Bind only this analysis call; never move its native scheduling boundary."""
    window = None
    if getattr(diagnostics, "profiling_enabled", False) is True:
        with suppress(Exception):
            window = diagnostics.begin_vad_window(source_end_sample)
    if window is None:
        yield
        return
    token = _VAD_WINDOW.set(window)
    outcome = "error"
    try:
        yield
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    else:
        outcome = "returned"
    finally:
        with suppress(Exception):
            window.resumed(outcome)
        _VAD_WINDOW.reset(token)


class ServerDiagnostics:
    """Retain numeric observations without backpressuring an actor or socket."""

    def __init__(self, *, clock=time.monotonic_ns, max_events=4096, profiling=False):
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
        self.profiling_enabled = profiling is True
        self._profile_cpu_started = time.process_time_ns() if self.profiling_enabled else None
        self._vad_windows = 0
        self._vad_inflight = {}
        self._vad_slow = []
        self._vad_dropped = 0
        self._loop_task = None

    def elapsed_ns(self):
        """Read this observer's monotonic origin; never a conversation clock."""
        return max(0, self._clock() - self._started)

    def begin_vad_window(self, source_end_sample):
        """Keep at most two inflight records and sixteen slow completed records."""
        with self._lock:
            self._vad_windows += 1
            if len(self._vad_inflight) >= 2:
                self._vad_dropped += 1
                return None
            window = VADWindowProfile(self, self._vad_windows, source_end_sample)
            self._vad_inflight[window.fields["window_id"]] = window
            return window

    def update_vad_window(self, window, *, future_done=False, **fields):
        """Update only internal trusted scalar fields, including late thread exit."""
        with self._lock:
            window.fields.update(fields)
            window._future_done = window._future_done or future_done
            if window._future_done and window.fields["event_loop_resumed_elapsed_ns"] is not None:
                self._vad_inflight.pop(window.fields["window_id"], None)
                self._vad_slow = [x for x in self._vad_slow if x["window_id"] != window.fields["window_id"]]
                self._vad_slow.append(dict(window.fields))
                self._vad_slow.sort(
                    key=lambda x: (
                        max(x["executor_finished_elapsed_ns"] or 0, x["event_loop_resumed_elapsed_ns"] or 0)
                        - x["analyze_started_elapsed_ns"]
                    ),
                    reverse=True,
                )
                del self._vad_slow[16:]

    def start_loop_observer(self, *, interval_s=0.1):
        """Sample the owning loop without catch-up callbacks or per-sample events."""
        if not self.profiling_enabled or self._loop_task is not None:
            return
        if type(interval_s) not in (float, int) or not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("Loop observation interval must be finite and positive")

        async def observe():
            loop = asyncio.get_running_loop()
            while True:
                target = loop.time() + interval_s
                await asyncio.sleep(interval_s)
                self.observe_duration("event_loop_lag", max(0, loop.time() - target) * 1000)

        self._loop_task = asyncio.create_task(observe())

    async def stop_loop_observer(self):
        """Stop only the owned optional observer without joining model threads."""
        task = self._loop_task
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def record(self, name: str, **fields):
        """Append a scalar observation, including from a provider worker thread."""
        if not isinstance(name, str) or _NAME.fullmatch(name) is None:
            return
        safe = {}
        for key, value in fields.items():
            if not isinstance(key, str) or _NAME.fullmatch(key) is None:
                continue
            numeric = value is None or type(value) in (bool, int) or (type(value) is float and math.isfinite(value))
            label = (
                key in _LABEL_FIELDS
                and isinstance(value, str)
                and _LABEL.fullmatch(value)
                and (key != "lease_sha256" or re.fullmatch(r"[0-9a-f]{64}", value))
            )
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
            result = {
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
            if self.profiling_enabled:
                result["profiling"] = {
                    "schema": "nva.offline.profile.v1",
                    "total_vad_windows": self._vad_windows,
                    "dropped_vad_records": self._vad_dropped,
                    "slow_vad_windows": [dict(x) for x in self._vad_slow],
                    "inflight_vad_windows": [dict(x.fields) for x in self._vad_inflight.values()],
                    "process_cpu_elapsed_ms": (time.process_time_ns() - self._profile_cpu_started) / 1e6,
                    "process_wall_elapsed_ms": self.elapsed_ns() / 1e6,
                    "loop_observer_stopped": self._loop_task is None or self._loop_task.done(),
                }
            return result


def create_diagnostics():
    """Enable only through a job-owned directory supplied by the server operator."""
    root = os.environ.get("NVA_OFFLINE_SERVER_METRICS_DIR", "").strip()
    if not root:
        return None, None
    return ServerDiagnostics(profiling=os.environ.get("NVA_OFFLINE_PROFILE") == "1"), Path(root)


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
