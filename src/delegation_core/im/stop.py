# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Why work stops: the reasons the stop door takes, and how each one ends the work.

The stop door (``WorkRun.stop`` in ``im/run.py``) is the only way the IM stops work: once a stop is asked, nothing more
of the task is written and its calls are refused, and a running task's stop that does not return within
``delegation.stop_timeout_seconds`` is ``unconfirmed``.
"""

from __future__ import annotations

from delegation_core.backend.port import StopOutcome
from delegation_core.config import Texts
from delegation_core.im.stream import ResponseBook


class StopReason:
    """Why a task stops, and how that ends it on the wire."""

    work_state = "cancelled"  # the state of a Work that this reason ended

    def answer(self, texts: Texts, outcome: StopOutcome) -> str:
        """The stopped task's answer, a ``final_answer`` on its Work's open response (empty: none)."""
        return ""

    def end(self, book: ResponseBook, delegation_id: str) -> None:
        """End the Work's open response once its last task has ended: ``completed``, after the stop's answers."""
        book.start(delegation_id)
        book.end(delegation_id, "response.completed")


class ForNewerRequest(StopReason):
    """A conductor's ``Cancel``: a newer request replaces the work."""

    def answer(self, texts: Texts, outcome: StopOutcome) -> str:
        """The line that says whether the stop was confirmed."""
        return {"stopped": texts.cancelled_line, "unconfirmed": texts.unconfirmed_line}[outcome]


class AtDeadline(StopReason):
    """The client-tool deadline passed while a task waited for the app's outputs."""

    work_state = "failed"

    def answer(self, texts: Texts, outcome: StopOutcome) -> str:
        """The outcome-unknown line: whether the stuck calls ran is not known (said only where a response is open)."""
        return texts.owed_output

    def end(self, book: ResponseBook, delegation_id: str) -> None:
        """Nothing is written: the Work's last response already ended with its calls."""


class AtBackstop(StopReason):
    """The Work ran past ``delegation.work_timeout_seconds``; its adapter's limits should have ended it first."""

    work_state = "failed"

    def end(self, book: ResponseBook, delegation_id: str) -> None:
        """``response.failed`` with ``server_error``."""
        book.start(delegation_id)
        book.failed(delegation_id, "server_error", "The work did not end within its time limit.")


class AtClose(StopReason):
    """The session closes."""

    def end(self, book: ResponseBook, delegation_id: str) -> None:
        """Nothing here: the close ends every open response itself, in order."""
