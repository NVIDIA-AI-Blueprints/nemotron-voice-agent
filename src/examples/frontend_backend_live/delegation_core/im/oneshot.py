# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-shot sessions: the official stateless use, one request that carries its whole context."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from examples.frontend_backend_live.delegation_core.backend.port import BackendAdapter
from examples.frontend_backend_live.delegation_core.config import DelegationSettings, Texts
from examples.frontend_backend_live.delegation_core.im.conversation import TaskHistory
from examples.frontend_backend_live.delegation_core.im.input import BackendTask
from examples.frontend_backend_live.delegation_core.im.results import ResultFormat
from examples.frontend_backend_live.delegation_core.im.run import TaskRun, WorkRun
from examples.frontend_backend_live.delegation_core.im.session import Admission, DelegationError, IMSession, Incoming
from examples.frontend_backend_live.delegation_core.im.stream import ResponseStream
from examples.frontend_backend_live.delegation_core.items import message_text
from examples.shared.ids import uid


def _noop(kind: str, **fields: Any) -> None:
    """Ignore observations."""


class OneShotSession(IMSession):
    """One stateless request's session: its Work at once; no placement, records, progress notes, work tool or trim."""

    work_tools = ()

    def __init__(
        self,
        backend: BackendAdapter,
        defaults: BackendTask,
        settings: DelegationSettings,
        texts: Texts,
        *,
        observe: Callable[..., None] = _noop,
    ):
        """Serve one request on ``backend`` (``OneShotSource.new_one_shot()``), with the IM's defaults and texts."""
        super().__init__(backend, defaults, settings, texts, result_format=ResultFormat(), observe=observe)

    def start(self) -> None:
        """Place nothing: the one request is its Work."""

    def admit(
        self, input_items: list[dict], *, delegation_id: str | None = None, settings: dict | None = None
    ) -> Admission:
        """Accept the session's one request, refusing any other: its response opens in progress and its Work starts."""
        if self.closed or self.ledger.delegations:
            raise DelegationError("A one-shot session serves exactly one request")
        settings = deepcopy(settings or {})
        delegation = self.ledger.new_delegation(
            delegation_id or uid("item"), input_items=deepcopy(input_items), settings=settings
        )
        stream = ResponseStream()
        self.book.open(delegation.id, stream, *self.response_settings(settings), running=True)
        self.incoming[delegation.id] = Incoming(None)
        self._start_work([delegation.id], ())
        return Admission(delegation.id, stream)

    def admit_client_request(
        self, input_items: list[dict], *, delegation_id: str | None = None, settings: dict | None = None
    ) -> Admission:
        """Refuse any request of the client's own: the client owns everything after its one response."""
        raise DelegationError("A one-shot session serves exactly one request")

    def work_input(self, work: WorkRun) -> list[dict]:
        """Return the request's input as sent, with no records block."""
        return deepcopy(work.requested)

    def history_for(self, work: WorkRun, task: TaskRun) -> TaskHistory:
        """Return the input as sent, never trimmed or cut."""
        return TaskHistory(work.input, None, None, self.texts.cut_output)

    def commentary(self, task: TaskRun, message: dict) -> None:
        """Write a round's message that is not its answer as it is: a one-shot session has no notes."""
        text = message_text(message)
        task.record.progress = text
        task.work.write(lambda writer: writer.message(text, "commentary"))

    def progress(self, task: TaskRun, text: str) -> None:
        """Write nothing: a one-shot session has no progress notes."""

    def waits_for_app(self, task: TaskRun) -> bool:
        """Write the app's calls, which end the response; nothing waits for their outputs."""
        calls = list(task.app_calls.values())
        task.work.write(lambda writer: [writer.function_call(call) for call in calls])
        return False

    def work_ended(self, work: WorkRun, leftovers: list[str]) -> None:
        """End the session with its one Work."""
        super().work_ended(work, [])
        if self._closing is None:
            self._closing = asyncio.create_task(self._close(), name="one-shot-close")
