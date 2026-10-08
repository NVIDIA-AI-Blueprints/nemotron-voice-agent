# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The session's media timeline: the caller's input clock, and acknowledgments that wait for it.

Context appends are acknowledged with the span of the timeline they take effect on. The acknowledgment is held
until the caller's audio reaches the end of that span, so a client sees them in media order.
"""

from __future__ import annotations

MAX_PENDING_ACKS = 64
INJECTION_MS = 200


class Timeline:
    """Input clock and the acknowledgments waiting on it."""

    def __init__(self) -> None:
        """Start at 0 ms with nothing pending."""
        self.input_ms = 0
        self.pending: list[tuple[int, dict]] = []

    @property
    def full(self) -> bool:
        """True when no more acknowledgments can be queued."""
        return len(self.pending) >= MAX_PENDING_ACKS

    def advance(self, ms: int) -> list[dict]:
        """Move the clock to ``ms`` (never backwards) and return the acknowledgments it releases, in order."""
        self.input_ms = max(self.input_ms, ms)
        released = []
        while self.pending and self.pending[0][0] <= self.input_ms:
            released.append(self.pending.pop(0)[1])
        return released

    def hold(self, make_ack) -> None:
        """Queue an acknowledgment built by ``make_ack(start_ms, end_ms)`` for the span that starts now."""
        start_ms = self.input_ms
        end_ms = start_ms + INJECTION_MS
        self.pending.append((end_ms, make_ack(start_ms, end_ms)))

    def discard(self) -> list[dict]:
        """Return and forget every acknowledgment that never got released (the session closed first)."""
        remaining = [ack for _, ack in self.pending]
        self.pending.clear()
        return remaining
