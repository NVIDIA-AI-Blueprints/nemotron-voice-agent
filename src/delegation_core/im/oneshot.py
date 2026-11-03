# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-shot sessions: the official stateless use, one request that carries its whole context.

A one-shot session is an IM session whose one request is its Work at once, with no placement, records block, brief,
result-format note, work tool or trim. The app's calls end its response, and the session ends and releases its adapter
once that response has ended.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from typing import Any

from delegation_core.backend.port import BackendAdapter
from delegation_core.config import DelegationSettings, Texts
from delegation_core.ids import uid
from delegation_core.im.conversation import TaskHistory
from delegation_core.im.input import BackendTask
from delegation_core.im.results import ResultFormat
from delegation_core.im.run import TaskRun, WorkRun
from delegation_core.im.session import Admission, DelegationError, IMSession, Incoming
from delegation_core.im.stream import ResponseStream, message_text


def _noop(kind: str, **fields: Any) -> None:
    """Ignore observations."""


class OneShotSession(IMSession):
    """The session of one stateless request (see the module's docstring)."""

    def __init__(
        self,
        backend: BackendAdapter,
        defaults: BackendTask,
        settings: DelegationSettings,
        texts: Texts,
        *,
        observe: Callable[..., None] = _noop,
    ):
        """Serve one request on ``backend`` (``OneShotSource.new_one_shot()``), with the IM's defaults and texts.

        It offers no work tool, whatever ``delegation.work_tools`` lists: the tools are the request's, as sent.
        """
        own = replace(settings, work_tools=())
        super().__init__(backend, defaults, own, texts, result_format=ResultFormat(), observe=observe)

    def start(self) -> None:
        """Nothing to place: the one request is its Work."""

    def admit(
        self, input_items: list[dict], *, delegation_id: str | None = None, settings: dict | None = None
    ) -> Admission:
        """Accept the session's one request: its response opens in progress, and its Work starts at once.

        Raises:
            DelegationError: The session already served its request, or it is closing.
        """
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
        """A one-shot session takes no request of the client's own: the client owns everything after its response.

        Raises:
            DelegationError: Always.
        """
        raise DelegationError("A one-shot session serves exactly one request")

    def work_input(self, work: WorkRun) -> list[dict]:
        """The request's input as sent: no records block."""
        return deepcopy(work.requested)

    def history_for(self, work: WorkRun, task: TaskRun) -> TaskHistory:
        """The input as sent, never trimmed or cut."""
        return TaskHistory(work.input, None, None, self.texts.cut_output)

    def commentary(self, task: TaskRun, message: dict) -> None:
        """A round's message that is not its answer is written as it is (no notes in a one-shot session)."""
        text = message_text(message)
        task.record.progress = text
        task.work.write(lambda writer: writer.message(text, "commentary"))

    def progress(self, task: TaskRun, text: str) -> None:
        """No progress notes in a one-shot session."""

    def waits_for_app(self, task: TaskRun) -> bool:
        """The app's calls are written and end the response; nothing waits for their outputs."""
        calls = list(task.app_calls.values())
        task.work.write(lambda writer: [writer.function_call(call) for call in calls])
        return False

    def work_ended(self, work: WorkRun, leftovers: list[str]) -> None:
        """The one Work ended: the session ends with it."""
        super().work_ended(work, [])
        if self._closing is None:
            self._closing = asyncio.create_task(self._close(), name="one-shot-close")
