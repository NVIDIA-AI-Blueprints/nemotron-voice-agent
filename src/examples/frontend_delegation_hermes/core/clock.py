# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wall-clock time for every head (the τ³ head's audio clock lives in ``adapters/tau/audio_clock.py``).

Wall time is only *recorded* (latency logs) and used for the non-turn timeouts.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable


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
