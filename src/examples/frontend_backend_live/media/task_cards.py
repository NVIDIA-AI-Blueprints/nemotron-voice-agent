# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the repository UI bot shows each delegation: one task card per delegation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from pipecat.frames.frames import Frame
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

from examples.frontend_backend_live.delegations import Ending, ReadyResult

CARD_MESSAGE = "agent-task-update"
AGENT = "backend"
ENDED_STATUS = {
    Ending.ANSWERED: "done",
    Ending.HANDED_OVER: "done",
    Ending.LATE: "done",
    Ending.FAILED: "failed",
    Ending.CANCELLED: "cancelled",
    Ending.LOST: "failed",
}


@dataclass
class TaskCard:
    """What the UI shows of one delegation."""

    status: str = "running"
    stage: str = ""
    query: str = ""
    detail: str = ""
    answers: list[str] = field(default_factory=list)


class RtviTaskCards:
    """Shows each delegation as an ``agent-task-update`` card in the repository UI, updated as it moves on."""

    def __init__(self, queue_frame: Callable[[Frame], Awaitable[None]]):
        """Send cards with ``queue_frame``, the pipeline worker's."""
        self.queue_frame = queue_frame
        self.cards: dict[str, TaskCard] = {}
        self._sending: set[asyncio.Task] = set()

    def started(self, delegation_id: str, request: str) -> None:
        """Show a running card with what the caller asked."""
        self._update(delegation_id, query=request)

    def progress(self, delegation_id: str, line: str) -> None:
        """Show the delegation's newest progress line as the card's stage."""
        self._update(delegation_id, stage=line)

    def shown(self, delegation_id: str, item: dict) -> None:
        """Show the display part, the ``show`` call's output, as the card's detail."""
        self._update(delegation_id, detail=str(item.get("output") or ""))

    def answered(self, result: ReadyResult) -> None:
        """Add a ready result to the card's response."""
        self._card(result.delegation_id).answers.append(result.speak)
        self._update(result.delegation_id)

    def ended(self, delegation_id: str, how: Ending) -> None:
        """Show how the delegation ended in the card's status and stage."""
        self._update(delegation_id, status=ENDED_STATUS[how], stage=how.value)

    def _card(self, delegation_id: str) -> TaskCard:
        return self.cards.setdefault(delegation_id, TaskCard())

    def _update(self, delegation_id: str, **changes: str) -> None:
        card = self._card(delegation_id)
        for name, value in changes.items():
            setattr(card, name, value)
        # The whole card each time, so the UI's merge of a card's fields never loses one.
        message = {
            "type": CARD_MESSAGE,
            "task_id": delegation_id,
            "agent": AGENT,
            "status": card.status,
            "stage": card.stage,
            "query": card.query,
            "detail": card.detail,
            "response": "\n\n".join(card.answers),
        }
        task = asyncio.get_running_loop().create_task(self.queue_frame(RTVIServerMessageFrame(data=message)))
        self._sending.add(task)
        task.add_done_callback(self._sending.discard)
