# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Why work stops: the reasons the stop door takes, and how each ends the work; the door is the IM's one way to stop."""

from __future__ import annotations

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import StopOutcome
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.config import Texts
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.stream import ResponseBook


class StopReason:
    """Why a task stops, and how that ends it on the wire."""

    work_state = "cancelled"
    discard_pending_requests = False

    def answer(self, texts: Texts, outcome: StopOutcome) -> str:
        """Return the stopped task's answer on its Work's open response; empty for none."""
        return ""

    def end(self, book: ResponseBook, delegation_id: str) -> None:
        """End the Work's open response once its last task has ended: ``completed``, after the stop's answers."""
        book.start(delegation_id)
        book.end(delegation_id, "response.completed")


class ForNewerRequest(StopReason):
    """Stop work replaced by a newer request."""

    discard_pending_requests = True

    def __init__(self, *, report_on_work: bool = True) -> None:
        """Report the outcome on the work when it owns the caller's reply."""
        self.report_on_work = report_on_work

    def answer(self, texts: Texts, outcome: StopOutcome) -> str:
        """Return the line that says whether the stop was confirmed."""
        if not self.report_on_work:
            return ""
        return {"stopped": texts.cancelled_line, "unconfirmed": texts.unconfirmed_line}[outcome]

    def end(self, book: ResponseBook, delegation_id: str) -> None:
        """Mark the original response cancelled when the stop has its own reply."""
        if self.report_on_work:
            super().end(book, delegation_id)
        else:
            book.start(delegation_id)
            book.cancelled(delegation_id)


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


class BeforeStartFailure(StopReason):
    """A validated logical assignment could not finish setup; its start barrier admitted no backend invocation."""

    work_state = "failed"
    discard_pending_requests = True

    def __init__(self, message: str):
        """Retain the configured failure wording without calling this mechanical failure a caller cancellation."""
        self.message = message

    def end(self, book: ResponseBook, delegation_id: str) -> None:
        """Fail the owned response truthfully without retrying a never-applied reference batch."""
        book.start(delegation_id)
        book.failed(delegation_id, "server_error", self.message)


class AtAttachmentLoss(StopReason):
    """An unqualified native runtime cannot safely keep acting without its caller."""

    work_state = "failed"

    def answer(self, texts: Texts, outcome: StopOutcome) -> str:
        """Report uncertainty instead of claiming a remote effect was cancelled."""
        return texts.owed_output
