# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the client is told about delegation: the backend's events (redacted) and each delegation as it starts."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from examples.frontend_backend_live.delegation.tasks import DelegatedTask
    from examples.frontend_backend_live.tool_calling.client_tools import ClientToolGateway
    from live.session import LiveProtocolSession


def sanitize_response_event(raw: dict) -> dict:
    """Return ``raw`` with the response snapshot redacted, as lifecycle events are.

    Lifecycle snapshots drop ``instructions``, ``input``, ``tools`` and ``output``. Individual item events keep
    their completed function calls, so a client can collect every call of a round.
    """
    result = deepcopy(raw)
    if isinstance(result.get("response"), dict):
        response = result["response"]
        response.update(instructions=None, tools=[], output=[])
        response.pop("input", None)
    return result


class BackendEventRelay:
    """Tells the client what the backend is doing: delegations as they start, and the backend's own events."""

    def __init__(self, session: LiveProtocolSession, tools: ClientToolGateway):
        """Relay on ``session``'s events; ``tools`` is told about each round and each function call."""
        self.session, self.tools = session, tools
        self.announced: set[str] = set()

    async def backend_event(self, task: DelegatedTask, raw: dict) -> None:
        """Forward one backend event to the client; register function calls so the client can answer them."""
        kind = raw.get("type", "")
        if kind in {"response.incomplete", "response.failed"}:
            # The worker recovers from a failed round (retry once, then apologize). Clients treat these terminal
            # events as fatal, so they are not forwarded.
            return
        if kind == "response.created":
            self.tools.begin_round()
            if task.id not in self.announced:
                self.announced.add(task.id)
                await self.session.emit(
                    "session.delegation.created",
                    offset_ms=self.session.input_ms,
                    delegation={
                        "id": task.id,
                        "type": "delegation",
                        "target": "responses",
                        "response_id": raw["response"]["id"],
                    },
                )
        elif kind == "response.output_item.done" and raw.get("item", {}).get("type") == "function_call":
            self.tools.note_call(raw["item"])
        await self.session.emit("response.event", delegation_id=task.id, event=sanitize_response_event(raw))

    async def client_delegation(self, delegation_id: str) -> None:
        """Announce a delegation that the client owns (``client`` mode)."""
        await self.session.emit(
            "session.delegation.created",
            offset_ms=self.session.input_ms,
            delegation={"id": delegation_id, "type": "delegation", "target": "client"},
        )
