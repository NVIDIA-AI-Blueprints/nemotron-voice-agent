# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filler interception and the filler timing record (plan section 9).

The frontend's ``call_backend(query, filler_text)`` makes the text prototype emit
a ``filler`` event *before* the backend starts. :class:`FillerTap` receives it
through the routing sink and hands it to the engine's handler, which only
records the delegation and schedules the filler task; nothing awaits, so the
backend starts immediately. Calls from another thread are marshalled onto the
session's loop with ``call_soon_threadsafe``.

Every delegation produces exactly one :class:`FillerTimingRecord`, from the
session's first turn on and in both filler modes. Each record is anchored to the
session start and to the user's turn start and end, on the wall clock (ISO-8601
UTC, for lining up with other logs) and on the input-audio clock (the clock
tau2 uses). Durations come from the monotonic clock.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from examples.frontend_backend_verdict.text import events as text_events
from examples.frontend_backend_verdict.text.events import InternalEvent
from examples.frontend_backend_verdict.voice.agent.sinks import EventLog
from examples.frontend_backend_verdict.voice.clock import WallClock, iso_utc


@dataclass(frozen=True, slots=True)
class Stamp:
    """One moment on the wall and monotonic clocks (and optionally the audio clock)."""

    wall: float
    mono: float
    audio_ms: float | None = None

    @classmethod
    def now(cls, clock: WallClock, audio_ms: float | None = None) -> Stamp:
        """Stamp the current moment."""
        return cls(wall=clock.time(), mono=clock.monotonic(), audio_ms=audio_ms)


def _ms(later: Stamp | None, earlier: Stamp | None) -> int | None:
    if later is None or earlier is None:
        return None
    return int(round((later.mono - earlier.mono) * 1000.0))


@dataclass(slots=True)
class FillerTimingRecord:
    """Timing of one delegation's filler, filled in as the turn progresses."""

    session_id: str
    turn_id: int
    mode: str
    text: str
    speak_after_ms: int
    session_start: Stamp
    turn_start: Stamp | None
    turn_end: Stamp | None
    asr_final: Stamp | None
    agent_start: Stamp
    filler_ready: Stamp
    backend_done: Stamp | None = None
    first_answer_audio: Stamp | None = None
    filler_audio_start: Stamp | None = None
    would_have_spoken: bool | None = None
    spoken: bool = False
    outcome: str = "pending"
    emitted: bool = False

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready record (plan section 9.2)."""

        def point(stamp: Stamp | None) -> dict[str, Any] | None:
            if stamp is None:
                return None
            return {"wall": iso_utc(stamp.wall), "audio_ms": None if stamp.audio_ms is None else int(stamp.audio_ms)}

        record: dict[str, Any] = {
            "kind": "filler",
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "mode": self.mode,
            "text": self.text,
            "session_start_wall": iso_utc(self.session_start.wall),
            "turn_start": point(self.turn_start),
            "turn_end": point(self.turn_end),
            "asr_final": None
            if self.asr_final is None
            else {"wall": iso_utc(self.asr_final.wall), "since_turn_end_ms": _ms(self.asr_final, self.turn_end)},
            "filler_ready": {
                "wall": iso_utc(self.filler_ready.wall),
                "since_session_start_ms": _ms(self.filler_ready, self.session_start),
                "since_turn_start_ms": _ms(self.filler_ready, self.turn_start),
                "since_turn_end_ms": _ms(self.filler_ready, self.turn_end),
                "frontend_latency_ms": _ms(self.filler_ready, self.agent_start),
            },
            "backend_done": None
            if self.backend_done is None
            else {
                "wall": iso_utc(self.backend_done.wall),
                "since_filler_ms": _ms(self.backend_done, self.filler_ready),
                "since_turn_end_ms": _ms(self.backend_done, self.turn_end),
            },
            "first_answer_audio": None
            if self.first_answer_audio is None
            else {
                "wall": iso_utc(self.first_answer_audio.wall),
                "since_turn_end_ms": _ms(self.first_answer_audio, self.turn_end),
                "since_filler_ms": _ms(self.first_answer_audio, self.filler_ready),
            },
            "filler_audio_start": None
            if self.filler_audio_start is None
            else {
                "wall": iso_utc(self.filler_audio_start.wall),
                "since_filler_ms": _ms(self.filler_audio_start, self.filler_ready),
            },
            "speak_after_ms": self.speak_after_ms,
            "would_have_spoken": self.would_have_spoken,
            "spoken": self.spoken,
            "outcome": self.outcome,
        }
        return record


class FillerLog:
    """Writes filler records to loguru, the shared event log, and the optional ``filler.log_path``."""

    def __init__(self, path: str = "", *, event_log: EventLog | None = None) -> None:
        """Bind the outputs; the JSONL file's parent directory is created on first write."""
        self._path = Path(path) if path else None
        self._event_log = event_log
        self._lock = threading.Lock()
        self.records: list[dict[str, Any]] = []

    def emit(self, record: FillerTimingRecord) -> None:
        """Write one record (exactly once per delegation)."""
        if record.emitted:
            return
        record.emitted = True
        data = record.as_dict()
        self.records.append(data)
        line = json.dumps(data, ensure_ascii=False)
        logger.info(f"[filler] {line}")
        if self._event_log is not None:
            payload = {k: v for k, v in data.items() if k not in ("kind", "session_id")}
            self._event_log.write("filler_timing", record.session_id, payload, timestamp=record.filler_ready.wall)
        if self._path is not None:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{line}\n")


class FillerTap:
    """Session listener: forwards ``filler`` events to the engine without blocking the agent."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_filler: Callable[[str | None, Stamp], None],
        *,
        clock: WallClock,
        on_other: Callable[[InternalEvent], None] | None = None,
    ) -> None:
        """``on_filler(text, stamp)`` runs on ``loop``; the stamp is taken at emission time.

        ``text`` is ``None`` for the ``delegation`` event, which the text prototype emits
        just before ``filler``; it guarantees a timing record even when a frontend
        delegates without filler text.
        """
        self._loop = loop
        self._on_filler = on_filler
        self._clock = clock
        self._on_other = on_other

    def on_internal_event(self, event: InternalEvent) -> None:
        """Stamp, schedule the handler, and return immediately."""
        if event.kind in (text_events.DELEGATION, text_events.FILLER):
            text = str(event.data.get("text") or "") if event.kind == text_events.FILLER else None
            # The text prototype stamped the wall time at emission; add the monotonic time now.
            stamp = Stamp(wall=event.timestamp, mono=self._clock.monotonic())
            self._dispatch(self._on_filler, text, stamp)
        elif self._on_other is not None:
            self._dispatch(self._on_other, event)

    def _dispatch(self, handler: Callable[..., None], *args: object) -> None:
        # On the session's loop (the agent's own task) the handler runs inline: it only
        # records and schedules, never awaits, and a fast backend must not finish before
        # the delegation is recorded. From any other thread it is scheduled onto the loop.
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            handler(*args)
        else:
            self._loop.call_soon_threadsafe(handler, *args)
