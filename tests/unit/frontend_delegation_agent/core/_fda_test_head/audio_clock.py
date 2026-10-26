# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The former standalone Realtime head's audio clock.

Turn taking runs on the **audio clock**: input samples received since the
session started. A client such as the benchmark client can pause its input stream for seconds
of wall-clock time while its user simulator thinks, so a wall-clock endpointer
would close turns at the wrong moments.
"""

from __future__ import annotations


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
