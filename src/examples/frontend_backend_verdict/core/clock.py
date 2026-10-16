# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two clocks a session runs on.

Turn taking runs on the **audio clock**: input samples received since the
session started. A client such as tau2 can pause its input stream for seconds
of wall-clock time while its user simulator thinks, so a wall-clock endpointer
would close turns at the wrong moments. Wall time is only *recorded* (latency
logs) and used for the two non-turn timeouts.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable


class AudioClock:
    """Cumulative input-audio time, advanced by decoded samples."""

    def __init__(self) -> None:
        """Start at audio time zero."""
        self._ms = 0.0

    @property
    def now_ms(self) -> float:
        """Audio milliseconds received since the session started."""
        return self._ms

    def advance(self, samples: int, rate: int) -> float:
        """Advance by ``samples`` at ``rate`` Hz and return the step in milliseconds."""
        if samples <= 0 or rate <= 0:
            return 0.0
        step = samples * 1000.0 / rate
        self._ms += step
        return step


@runtime_checkable
class WallClock(Protocol):
    """Wall and monotonic time, injectable so tests can fake it."""

    def time(self) -> float:
        """Epoch seconds."""

    def monotonic(self) -> float:
        """Monotonic seconds, for durations."""


class SystemClock:
    """The real clock."""

    def time(self) -> float:
        """Epoch seconds."""
        return time.time()

    def monotonic(self) -> float:
        """Monotonic seconds."""
        return time.monotonic()


def iso_utc(epoch_seconds: float) -> str:
    """Format epoch seconds as ISO-8601 UTC with millisecond precision."""
    stamp = datetime.fromtimestamp(epoch_seconds, tz=UTC)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.") + f"{stamp.microsecond // 1000:03d}Z"
