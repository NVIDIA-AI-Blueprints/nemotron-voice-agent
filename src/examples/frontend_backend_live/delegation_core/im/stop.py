# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Why work stops: the reasons the stop door takes, and how each ends the work; the door is the IM's one way to stop."""

from __future__ import annotations

from examples.frontend_backend_live.delegation_core.backend.port import StopOutcome
from examples.frontend_backend_live.delegation_core.config import Texts
from examples.frontend_backend_live.delegation_core.im.stream import ResponseBook


class StopReason:
    """Why a task stops, and how that ends it on the wire."""

    work_state = "cancelled"

    def answer(self, texts: Texts, outcome: StopOutcome) -> str:
        """Return the stopped task's answer on its Work's open response; empty for none."""
        return ""

    def end(self, book: ResponseBook, delegation_id: str) -> None:
        """End the Work's open response once its last task has ended: ``completed``, after the stop's answers."""
        book.start(delegation_id)
        book.end(delegation_id, "response.completed")


class ForNewerRequest(StopReason):
    """A conductor's ``Cancel``: a newer request replaces the work."""

    def answer(self, texts: Texts, outcome: StopOutcome) -> str:
        """Return the line that says whether the stop was confirmed."""
        return {"stopped": texts.cancelled_line, "unconfirmed": texts.unconfirmed_line}[outcome]


class AtDeadline(StopReason):
    """The client-tool deadline passed while a task waited for the app's outputs."""

    work_state = "failed"

    def answer(self, texts: Texts, outcome: StopOutcome) -> str:
        """Return the outcome-unknown line, written only where a response is open."""
        return texts.owed_output

    def end(self, book: ResponseBook, delegation_id: str) -> None:
        """Write nothing: the Work's last response already ended with its calls."""


class AtBackstop(StopReason):
    """The Work ran past ``delegation.work_timeout_seconds``; its adapter's limits should have ended it first."""

    work_state = "failed"

    def end(self, book: ResponseBook, delegation_id: str) -> None:
        """End the response ``response.failed`` with ``server_error``."""
        book.start(delegation_id)
        book.failed(delegation_id, "server_error", "The work did not end within its time limit.")


class AtClose(StopReason):
    """The session closes."""

    def end(self, book: ResponseBook, delegation_id: str) -> None:
        """Write nothing: the close ends every open response itself, in order."""
