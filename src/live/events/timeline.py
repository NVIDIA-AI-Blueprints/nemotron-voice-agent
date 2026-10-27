# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The session's media timeline: the caller's input clock.

The frontend adapter advances the clock as it consumes caller audio. Events place themselves on it: the delegation
announcement's ``offset_ms`` and the span of each append's acknowledgment. Appended context takes no audio time, so
its span is zero-length at the clock (the GPT-Live reference: "For appended context, this can equal start_ms"), and the
acknowledgment goes out once the frontend adapter has taken the context.
"""

from __future__ import annotations


class Timeline:
    """The input clock: milliseconds of caller audio seen so far."""

    def __init__(self) -> None:
        """Start at 0 ms."""
        self.input_ms = 0

    def advance(self, ms: int) -> None:
        """Move the clock to ``ms``; it never moves backwards."""
        self.input_ms = max(self.input_ms, ms)
