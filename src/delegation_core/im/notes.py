# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Progress notes: the short ``commentary`` lines the IM writes on the carrying delegation's response while work runs.

Notes never join a backend history: the voice reads them to answer "how is it going?" from real progress.
"""

from __future__ import annotations

import asyncio
import time

from delegation_core.im.stream import ResponseBook


class ProgressNotes:
    """The progress notes of one IM session, rate-limited per delegation."""

    def __init__(self, book: ResponseBook, interval: float):
        """Write notes into ``book``'s responses, at most one per delegation every ``interval`` seconds (0: none)."""
        self.book, self.interval = book, interval
        self._last: dict[str, float] = {}  # when each delegation's last note was written
        self._waiting: dict[str, tuple[str, asyncio.TimerHandle]] = {}  # the newest note waiting its turn

    def note(self, delegation_id: str, text: str) -> None:
        """Write ``text`` as a note of the delegation now, or when its interval has passed."""
        if self.interval <= 0 or not text.strip() or self.book.running(delegation_id) is None:
            return
        waited = time.monotonic() - self._last.get(delegation_id, float("-inf"))
        self.drop(delegation_id)
        if waited >= self.interval:
            self._write(delegation_id, text)
            return
        timer = asyncio.get_running_loop().call_later(self.interval - waited, self._flush, delegation_id)
        self._waiting[delegation_id] = (text, timer)

    def drop(self, delegation_id: str) -> None:
        """Forget the delegation's waiting note."""
        waiting = self._waiting.pop(delegation_id, None)
        if waiting is not None:
            waiting[1].cancel()

    def close(self) -> None:
        """Forget every waiting note."""
        for delegation_id in list(self._waiting):
            self.drop(delegation_id)

    def _flush(self, delegation_id: str) -> None:
        waiting = self._waiting.pop(delegation_id, None)
        if waiting is not None:
            self._write(delegation_id, waiting[0])

    def _write(self, delegation_id: str, text: str) -> None:
        writer = self.book.running(delegation_id)
        if writer is not None:
            writer.message(text, "commentary")
            self._last[delegation_id] = time.monotonic()
