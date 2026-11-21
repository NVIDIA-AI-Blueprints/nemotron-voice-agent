# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Progress notes: the short ``commentary`` lines the IM writes on the carrying response while work runs."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from examples.frontend_backend_live.delegation_core.im.stream import ResponseBook
from examples.shared.work_provenance import Attribution


class ProgressNotes:
    """The progress notes of one IM session, rate-limited per delegation; they never join a backend history."""

    def __init__(self, book: ResponseBook, interval: float):
        """Write notes into ``book``'s responses, at most one per delegation every ``interval`` seconds (0: none)."""
        self.book, self.interval = book, interval
        self._last: dict[tuple[str, str], float] = {}
        self._waiting: dict[
            tuple[str, str], tuple[str, Attribution | None, bool, asyncio.TimerHandle | None, Callable[[], bool] | None]
        ] = {}

    def note(
        self,
        delegation_id: str,
        text: str,
        attribution: Attribution | None = None,
        *,
        hold: bool = False,
        eligible: Callable[[], bool] | None = None,
    ) -> None:
        """Write ``text`` as a note of the delegation now, or when its interval has passed."""
        if self.interval <= 0 or not text.strip() or (eligible is not None and not eligible()):
            return
        key = delegation_id, attribution.branch_ref if attribution else ""
        waited = time.monotonic() - self._last.get(key, float("-inf"))
        self.drop(delegation_id, branch_ref=key[1])
        if self.book.running(delegation_id) is None:
            if hold:
                self._waiting[key] = text, attribution, hold, None, eligible
            return
        if waited >= self.interval:
            self._write(key, text, attribution, eligible)
            return
        timer = asyncio.get_running_loop().call_later(self.interval - waited, self._flush, key)
        self._waiting[key] = text, attribution, hold, timer, eligible

    def drop(self, delegation_id: str, *, branch_ref: str | None = None) -> None:
        """Forget the delegation's waiting note."""
        for key in [
            key for key in self._waiting if key[0] == delegation_id and (branch_ref is None or key[1] == branch_ref)
        ]:
            waiting = self._waiting.pop(key)
            if waiting[3] is not None:
                waiting[3].cancel()

    def close(self) -> None:
        """Forget every waiting note."""
        for delegation_id, _ in list(self._waiting):
            self.drop(delegation_id)

    def resume(self, delegation_id: str) -> None:
        """Publish retained sibling news after a legitimate response continuation opens."""
        for key, (text, attribution, hold, timer, eligible) in list(self._waiting.items()):
            if key[0] == delegation_id and timer is None:
                self.note(delegation_id, text, attribution, hold=hold, eligible=eligible)

    def _flush(self, key: tuple[str, str]) -> None:
        waiting = self._waiting.pop(key, None)
        if waiting is not None:
            text, attribution, hold, _, eligible = waiting
            if hold and self.book.running(key[0]) is None:
                self._waiting[key] = text, attribution, hold, None, eligible
            else:
                self._write(key, text, attribution, eligible)

    def _write(
        self, key: tuple[str, str], text: str, attribution: Attribution | None, eligible: Callable[[], bool] | None
    ) -> None:
        if eligible is not None and not eligible():
            return
        writer = self.book.running(key[0])
        if writer is not None:
            with writer.attributed(attribution):
                writer.message(text, "commentary")
            self._last[key] = time.monotonic()
