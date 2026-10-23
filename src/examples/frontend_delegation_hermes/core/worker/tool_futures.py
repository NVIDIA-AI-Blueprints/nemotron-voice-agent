# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keyed result futures for tools executed outside the worker (plan section 8.1).

A Hermes tool handler thread calls :meth:`ToolFutures.call`: it allocates a
``call_id``, registers a future, hands a ``tool.call`` message to the (thread-safe)
emitter and then waits on the **result** future — never on the send. Results,
timeouts, interrupts, link loss, duplicates, late and unknown results are all
resolved or dropped explicitly.

Stdlib only.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import itertools
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

#: Error strings returned to the agent (Hermes swallows handler exceptions, so always a string).
TIMEOUT_ERROR = "Error: the tool did not return a result in time"
INTERRUPTED_ERROR = "Error: interrupted"
CLOSED_ERROR = "Error: session closed"
STALE_ERROR = "Error: stale run; tool not executed"

Emit = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class PendingCall:
    """One outstanding call."""

    call_id: str
    epoch: int
    name: str
    future: concurrent.futures.Future[str]
    deadline: float


def epoch_of(task_id: str | None) -> int | None:
    """``"<session_id>:<epoch>"`` → epoch (``None`` when absent or malformed)."""
    if not task_id or ":" not in task_id:
        return None
    try:
        return int(task_id.rsplit(":", 1)[1])
    except ValueError:
        return None


class ToolFutures:
    """Outstanding tool calls of one worker, keyed by ``call_id``."""

    def __init__(
        self,
        *,
        session_id: str,
        emit: Emit,
        timeout_s: float = 120.0,
        poll_s: float = 0.1,
        is_interrupted: Callable[[], bool] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """``emit`` must be thread-safe and raise when the link is closed."""
        self._session_id = session_id
        self._emit = emit
        self.timeout_s = timeout_s
        self._poll_s = poll_s
        self._is_interrupted = is_interrupted or (lambda: False)
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._pending: dict[str, PendingCall] = {}
        self._finished: dict[str, str] = {}
        self._counter = itertools.count(1)
        self.current_epoch: int | None = None
        self._interrupt = threading.Event()

    # -- handler side (Hermes threads) -------------------------------------------------

    def call(self, name: str, args: dict[str, Any], task_id: str | None) -> str:
        """Send one call and block until its result (or an error string)."""
        epoch = epoch_of(task_id)
        if epoch is None or epoch != self.current_epoch:
            return STALE_ERROR
        call_id = f"fdh_{self._session_id}_{epoch}_{next(self._counter)}"
        future: concurrent.futures.Future[str] = concurrent.futures.Future()
        pending = PendingCall(call_id, epoch, name, future, self._monotonic() + self.timeout_s)
        with self._lock:
            self._pending[call_id] = pending
        try:
            self._emit(
                {
                    "type": "tool.call",
                    "call_id": call_id,
                    "epoch": epoch,
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                }
            )
        except Exception:  # noqa: BLE001 - a closed link must not leave the handler waiting
            self._finish(call_id, CLOSED_ERROR, "closed")
        return self._wait(pending)

    def _wait(self, pending: PendingCall) -> str:
        while True:
            try:
                return pending.future.result(timeout=self._poll_s)
            except concurrent.futures.TimeoutError:
                pass
            if self._interrupt.is_set() or self._is_interrupted():
                self._finish(pending.call_id, INTERRUPTED_ERROR, "interrupted")
            elif self._monotonic() >= pending.deadline and self._finish(pending.call_id, TIMEOUT_ERROR, "timeout"):
                self._safe_emit({"type": "tool.cancel", "call_id": pending.call_id, "reason": "timeout"})

    # -- loop side ----------------------------------------------------------------------------

    def resolve(self, call_id: str, output: str, epoch: int | None = None) -> str:
        """Deliver a result: ``ok``, ``duplicate``, ``late``, ``unknown`` or ``stale_epoch``."""
        with self._lock:
            pending = self._pending.get(call_id)
            if pending is None:
                reason = self._finished.get(call_id)
                if reason == "resolved":
                    return "duplicate"
                return "late" if reason is not None else "unknown"
            if epoch is not None and epoch != pending.epoch:
                return "stale_epoch"
        self._finish(call_id, output, "resolved")
        return "ok"

    def interrupt_all(self, text: str = INTERRUPTED_ERROR) -> int:
        """Resolve every outstanding call with ``text``; later calls also fail until :meth:`clear_interrupt`."""
        self._interrupt.set()
        return self._resolve_all(text, "interrupted")

    def close_all(self) -> int:
        """Resolve everything with the session-closed error."""
        self._interrupt.set()
        return self._resolve_all(CLOSED_ERROR, "closed")

    def clear_interrupt(self) -> None:
        """Accept calls again (a new run starts)."""
        self._interrupt.clear()

    def outstanding(self) -> list[str]:
        """Names of the calls still waiting."""
        with self._lock:
            return [p.name for p in self._pending.values()]

    # -- internals --------------------------------------------------------------------------------

    def _resolve_all(self, text: str, reason: str) -> int:
        with self._lock:
            call_ids = list(self._pending)
        return sum(1 for call_id in call_ids if self._finish(call_id, text, reason))

    def _finish(self, call_id: str, output: str, reason: str) -> bool:
        with self._lock:
            pending = self._pending.pop(call_id, None)
            if pending is None:
                return False
            self._finished[call_id] = reason
        if not pending.future.done():
            pending.future.set_result(output)
        return True

    def _safe_emit(self, message: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):  # the link is gone; the caller already has its error
            self._emit(message)
