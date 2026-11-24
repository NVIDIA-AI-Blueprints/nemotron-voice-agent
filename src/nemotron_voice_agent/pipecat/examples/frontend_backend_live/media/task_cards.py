# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the repository UI bot shows each delegation: one task card per delegation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pipecat.frames.frames import Frame
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegations import Ending, ReadyResult

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
TERMINAL = frozenset(ENDED_STATUS.values())
WORKER_PHASES = ("running", "stopping", "pausing", "waiting_client", "queued", "stop_unconfirmed", "paused")
STATUS_LABELS = {
    "applying": "Applying decision",
    "waiting_result": "Waiting for result",
    "waiting_client": "Waiting for tool result",
    "stopping": "Stopping",
    "stop_unconfirmed": "Stop unconfirmed",
    "answered": "Completed",
}


@dataclass
class TaskCard:
    """What the UI shows of one delegation."""

    status: str = "queued"
    stage: str = ""
    query: str | None = None
    detail: str = ""
    answers: list[str] = field(default_factory=list)
    workers: dict[str, dict[str, str]] = field(default_factory=dict)
    routing_failed: bool = False

    def phase(self) -> str:
        """Show the actual execution phase once the decision has assigned workers."""
        if self.status not in ("applying", "waiting_result"):
            return self.status
        states = {worker["status"] for worker in self.workers.values()}
        fallback = "waiting_result" if states else self.status
        return next((phase for phase in WORKER_PHASES if phase in states), fallback)


class RtviTaskCards:
    """Shows each delegation as an ``agent-task-update`` card in the repository UI, updated as it moves on."""

    def __init__(self, queue_frame: Callable[[Frame], Awaitable[None]]):
        """Send cards with ``queue_frame``, the pipeline worker's."""
        self.queue_frame = queue_frame
        self.cards: dict[str, TaskCard] = {}
        self._sending: set[asyncio.Task] = set()

    def started(self, delegation_id: str, request: str) -> None:
        """Show the current request phase with what the caller asked."""
        self._update(delegation_id, query=request)

    def observe(self, kind: str, **fields: Any) -> None:
        """Use execution events for task cards, independent of progress wording."""
        if kind == "im.routing.started":
            self._routing(fields["delegation_ids"], "planning")
        elif kind == "im.routing.applying":
            self._routing(fields["delegation_ids"], "applying")
        elif kind == "im.decision.accepted":
            self._routing(fields["sources"], "waiting_result")
        elif kind in ("im.routing.unresolved", "im.routing.capacity_unavailable"):
            for delegation_id in fields["delegation_ids"]:
                card = self._card(delegation_id)
                if not card.workers and not card.routing_failed:
                    card.routing_failed = True
                    self._update(delegation_id, status="failed", stage="Couldn't start")
        elif kind == "im.task.state":
            delegation_id, task_id = fields["source_delegation_id"], fields["task_id"]
            card = self._card(delegation_id)
            label = fields["brief"] or f"Worker {len(card.workers) + 1}"
            if task_id in card.workers:
                label = card.workers[task_id]["label"]
            status = fields["state"]
            card.workers[task_id] = {
                "id": task_id,
                "label": label,
                "status": status,
                "status_label": STATUS_LABELS.get(status, status),
            }
            if card.query is not None:
                self._update(delegation_id)

    def _routing(self, delegation_ids: list[str], status: str) -> None:
        for delegation_id in delegation_ids:
            card = self._card(delegation_id)
            if card.status not in TERMINAL and card.status != status:
                previous = card.phase()
                card.status = status
                if card.query is not None and card.phase() != previous:
                    self._update(delegation_id)

    def progress(self, delegation_id: str, line: str) -> None:
        """Show the delegation's newest progress line as the card's stage."""
        if not self._card(delegation_id).routing_failed:
            self._update(delegation_id, stage=line)

    def shown(self, delegation_id: str, item: dict) -> None:
        """Show the display part, the ``show`` call's output, as the card's detail."""
        self._update(delegation_id, detail=str(item.get("output") or ""))

    def answered(self, result: ReadyResult) -> None:
        """Add a ready result to the card's response."""
        self._card(result.delegation_id).answers.append(result.speak)
        self._update(result.delegation_id)

    def ended(self, delegation_id: str, how: Ending) -> None:
        """Show how the delegation ended, retaining a recorded routing failure."""
        if not self._card(delegation_id).routing_failed:
            self._update(delegation_id, status=ENDED_STATUS[how], stage=how.value)

    def _card(self, delegation_id: str) -> TaskCard:
        return self.cards.setdefault(delegation_id, TaskCard())

    def _update(self, delegation_id: str, **changes: str) -> None:
        card = self._card(delegation_id)
        for name, value in changes.items():
            setattr(card, name, value)
        status = card.phase()
        message = {
            "type": CARD_MESSAGE,
            "task_id": delegation_id,
            "agent": AGENT,
            "status": status,
            "status_label": STATUS_LABELS.get(status, status),
            "stage": card.stage,
            "query": card.query or "",
            "detail": card.detail,
            "response": "\n\n".join(card.answers),
        }
        if card.workers:
            message["workers"] = list(card.workers.values())
        task = asyncio.get_running_loop().create_task(self.queue_frame(RTVIServerMessageFrame(data=message)))
        self._sending.add(task)
        task.add_done_callback(self._sending.discard)
