# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the display part of each delegation goes in the repository UI bot: a ``DisplaySink`` (``cascade/ports.py``).

:class:`RtviTaskCards` shows each delegation as one ``agent-task-update`` card in the NVA UI
(``client/src/components/content/ConversationPanel.tsx``), with the request, its newest progress line, the display
part, the answer and the words said. The bot's root attaches it to its pipeline worker. A live session needs no sink of
its own: its client gets every official event through the gateway's relay, display items included, and the cascade's
default sink shows nothing.

Plain calls that never wait: a card is queued on the worker in a task of its own, in call order.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from pipecat.frames.frames import Frame
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

from examples.frontend_backend_live.cascade.ports import Ending, ReadyResult

CARD_MESSAGE = "agent-task-update"  # the NVA UI's task card message
AGENT = "backend"  # who does the delegated work, as the card names it
# The card's status for each ending: the UI shows "done" and "failed" in their own colors.
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
    spoken: list[str] = field(default_factory=list)


class RtviTaskCards:
    """Shows each delegation as an ``agent-task-update`` card in the NVA UI, updated as the delegation moves on.

    Each call updates the delegation's card and sends all of it, so the UI's merge of a card's fields never loses one.
    The card says ``done`` (or ``failed``, ``cancelled``) when the delegation ends; the UI also shows the spoken words
    as the assistant's turn once a done card carries them, whichever of the two comes first.
    """

    def __init__(self, queue_frame: Callable[[Frame], Awaitable[None]], agent: str = AGENT):
        """Send cards with ``queue_frame``, the pipeline worker's; ``agent`` names who does the delegated work."""
        self.queue_frame, self.agent = queue_frame, agent
        self.cards: dict[str, TaskCard] = {}
        self._sending: set[asyncio.Task] = set()

    def started(self, delegation_id: str, request: str) -> None:
        """A running card, with what the caller asked."""
        self._update(delegation_id, query=request)

    def progress(self, delegation_id: str, line: str) -> None:
        """The delegation's newest progress line, as the card's stage."""
        self._update(delegation_id, stage=line)

    def shown(self, delegation_id: str, item: dict) -> None:
        """The display part: the ``show`` call's output, as the card's detail."""
        self._update(delegation_id, detail=str(item.get("output") or ""))

    def answered(self, result: ReadyResult) -> None:
        """A ready result, added to the card's response."""
        self._card(result.delegation_id).answers.append(result.speak)
        self._update(result.delegation_id)

    def spoken(self, delegation_ids: Sequence[str], words: str) -> None:
        """The line said for these delegations' results."""
        for delegation_id in delegation_ids:
            self._card(delegation_id).spoken.append(words)
            self._update(delegation_id)

    def ended(self, delegation_id: str, how: Ending) -> None:
        """The delegation ended: the card's status and stage say how."""
        self._update(delegation_id, status=ENDED_STATUS[how], stage=how.value)

    def _card(self, delegation_id: str) -> TaskCard:
        return self.cards.setdefault(delegation_id, TaskCard())

    def _update(self, delegation_id: str, **changes: str) -> None:
        card = self._card(delegation_id)
        for name, value in changes.items():
            setattr(card, name, value)
        message = {
            "type": CARD_MESSAGE,
            "task_id": delegation_id,
            "agent": self.agent,
            "status": card.status,
            "stage": card.stage,
            "query": card.query,
            "detail": card.detail,
            "response": "\n\n".join(card.answers),
            "spoken_response": " ".join(card.spoken),
        }
        task = asyncio.get_running_loop().create_task(self.queue_frame(RTVIServerMessageFrame(data=message)))
        self._sending.add(task)
        task.add_done_callback(self._sending.discard)
