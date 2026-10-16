# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One session's JSONL event-log writer, with v1's record names and fields (S21).

Ported from FDH ``core/session_log.py`` (``a0ef50c``) and reduced to v1's ``_log``: every
record carries the session id and ``audio_ms`` (the input-audio clock when it was written).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from examples.frontend_backend_verdict.core.engine.event_log import EventLog


class SessionLog:
    """Writes one session's records to the shared event log."""

    def __init__(self, event_log: EventLog, session_id: str, audio_now: Callable[[], float]) -> None:
        """Bind the shared log, the session id and the input-audio clock."""
        self._event_log = event_log
        self._session_id = session_id
        self._audio_now = audio_now

    def write(self, kind: str, **data: Any) -> None:
        """Write one record (``audio_ms`` first, as v1 does)."""
        self._event_log.write(kind, self._session_id, {"audio_ms": int(self._audio_now()), **data})
