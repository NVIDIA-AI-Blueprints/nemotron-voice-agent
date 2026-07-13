# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Ephemeral session-scoped webcam frame store."""

from __future__ import annotations

import base64
import contextlib
import itertools
import math
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass(frozen=True)
class WebcamFrame:
    """One compressed browser webcam snapshot."""

    id: str
    session_id: str
    sequence: int
    name: str
    content_type: str
    data: bytes
    created_at: str

    def metadata(self) -> dict[str, str | int]:
        """Return public frame metadata without raw bytes."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "kind": "image",
            "name": self.name,
            "content_type": self.content_type,
            "bytes": len(self.data),
            "created_at": self.created_at,
        }

    def data_url(self) -> str:
        """Return the frame payload as a model-friendly data URL."""
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.content_type};base64,{encoded}"


_lock = threading.Lock()
_sequence = itertools.count(1)
_frames_by_session: dict[str, list[WebcamFrame]] = {}
_listeners_by_session: dict[str, list[Callable[[], None]]] = {}
_SAMPLE_INTERVAL_SECONDS = 1.0
_FRAME_MAX_WIDTH = 640
_JPEG_QUALITY = 0.7
_INITIAL_UPLOAD_DELAY_MS = 700
_FRAME_MAX_BYTES = 5_000_000
_FRAME_RING_LIMIT = 64


def register_webcam_frame_listener(session_id: str, listener: Callable[[], None]) -> Callable[[], None]:
    """Register a callback invoked whenever a session stores a new webcam frame."""
    cleaned_session_id = session_id.strip()
    if not cleaned_session_id:
        return lambda: None
    with _lock:
        _listeners_by_session.setdefault(cleaned_session_id, []).append(listener)

    def unregister() -> None:
        with _lock:
            listeners = _listeners_by_session.get(cleaned_session_id)
            if not listeners:
                return
            with contextlib.suppress(ValueError):
                listeners.remove(listener)
            if not listeners:
                _listeners_by_session.pop(cleaned_session_id, None)

    return unregister


def webcam_client_config() -> dict[str, float | int | bool]:
    """Return browser-facing webcam capture defaults."""
    return {
        "sample_interval_seconds": _SAMPLE_INTERVAL_SECONDS,
        "frame_max_width": _FRAME_MAX_WIDTH,
        "jpeg_quality": _JPEG_QUALITY,
        "initial_upload_enabled": True,
        "initial_upload_delay_ms": _INITIAL_UPLOAD_DELAY_MS,
    }


def store_webcam_frame(
    *,
    session_id: str,
    name: str,
    content_type: str,
    data: bytes,
) -> WebcamFrame:
    """Store one ephemeral webcam snapshot for a live session."""
    cleaned_session_id = session_id.strip()
    cleaned_content_type = content_type.strip().lower() or "image/jpeg"
    if not cleaned_session_id:
        raise ValueError("session_id is required")
    if not cleaned_content_type.startswith("image/"):
        raise ValueError("webcam frame must be an image")
    if not data:
        raise ValueError("webcam frame is empty")

    if len(data) > _FRAME_MAX_BYTES:
        raise ValueError(f"webcam frame exceeds max size ({_FRAME_MAX_BYTES} bytes)")

    with _lock:
        frame = WebcamFrame(
            id=uuid.uuid4().hex,
            session_id=cleaned_session_id,
            sequence=next(_sequence),
            name=name.strip() or "webcam-frame.jpg",
            content_type=cleaned_content_type,
            data=data,
            created_at=datetime.now(UTC).isoformat(),
        )
        frames = _frames_by_session.setdefault(cleaned_session_id, [])
        frames.append(frame)
        del frames[:-_FRAME_RING_LIMIT]
        listeners = list(_listeners_by_session.get(cleaned_session_id, ()))
    for listener in listeners:
        listener()
    return frame


def latest_webcam_frame(session_id: str) -> WebcamFrame | None:
    """Return the latest webcam frame for a session."""
    with _lock:
        frames = list(_frames_by_session.get(session_id.strip(), ()))
    return frames[-1] if frames else None


def recent_webcam_frames(
    session_id: str,
    *,
    max_seconds: float | None = None,
    max_count: int | None = None,
) -> list[WebcamFrame]:
    """Return recent webcam frames (oldest to newest) for a session.

    ``max_seconds`` keeps only frames within that many seconds of the latest frame;
    ``max_count`` caps how many of the most recent frames are returned. Used to build
    a single continuous video window for native temporal understanding.
    """
    cleaned_session_id = session_id.strip()
    if not cleaned_session_id:
        return []
    with _lock:
        frames = list(_frames_by_session.get(cleaned_session_id, ()))
    if not frames:
        return []
    if max_seconds is not None and math.isfinite(max_seconds) and max_seconds > 0:
        try:
            cutoff = datetime.fromisoformat(frames[-1].created_at) - timedelta(seconds=max_seconds)
            frames = [f for f in frames if datetime.fromisoformat(f.created_at) >= cutoff]
        except (OverflowError, ValueError):
            pass
    if max_count is not None and max_count > 0:
        frames = frames[-max_count:]
    return frames


def clear_session_webcam_frames(session_id: str) -> None:
    """Drop all webcam frames for a session."""
    with _lock:
        _frames_by_session.pop(session_id.strip(), None)
        _listeners_by_session.pop(session_id.strip(), None)


def clear_session_webcam_frame_data(session_id: str) -> None:
    """Drop stored webcam frames but keep live-session listeners registered."""
    with _lock:
        _frames_by_session.pop(session_id.strip(), None)
