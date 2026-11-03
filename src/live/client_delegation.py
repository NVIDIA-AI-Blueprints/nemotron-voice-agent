# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gateway's client delegation: Live client mode, in which the client app does the delegated work.

A session whose resolved ``delegation.type`` is ``client`` (an omitted, ``null`` or ``{}`` delegation included) gets a
:class:`ClientDelegation`, a ``live.ports.DelegationPart`` like the server delegation. :meth:`ClientDelegation.delegate`
mints the delegation's id and announces it with ``session.delegation.created`` (``target: "client"``, no
``response_id``); the input is dropped, since the client app builds its request from its own transcripts, and no
event follows. The Responses-only commands are refused. The only state it keeps is the ids it announced, so the
session accepts the client's appends that name one of them until it closes: nothing marks a client delegation
finished. It reaches no delegation core, keeps no timer and never decides when work is done.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from live.protocol import ProtocolError, uid

if TYPE_CHECKING:
    from live.ports import DelegationEvents, DelegationView
    from live.protocol import SessionConfig

# Both Responses-only commands require Responses delegation.
RESPONSES_ONLY = "This command requires Responses delegation"


class ClientDelegation:
    """Announces one client-mode session's delegations to its client app, and remembers their ids."""

    target: Literal["client"] = "client"  # where the work goes, as session.delegation.created names it
    reports_events = False  # no event follows an announcement (live.ports.DelegationPort)
    conversation = None  # no conversation with a delegation core: the client app runs its own backend
    close_seconds = 10.0  # its close only forgets the ids it announced: the gateway's default close limit

    def __init__(self):
        """Start with nothing announced; nothing is sent until :meth:`delegate`."""
        self.view: DelegationView | None = None  # the session's delegation view, from open
        self.announced: set[str] = set()  # the client delegations announced, for the appends that name them
        self._handing = True
        self._closed = False

    @property
    def acknowledgement_delay_ms(self) -> int:
        """The session's acknowledgement delay, read when asked; 0 (at once) with no session."""
        return self.view.acknowledgement_delay_ms if self.view is not None else 0

    async def open(self, view: DelegationView | None) -> None:
        """Keep the session's delegation view; it opens nothing else. Safe to call twice."""
        self.view = view

    def stop_handing(self) -> None:
        """Close step 2: announce nothing more to the client."""
        self._handing = False

    async def close(self) -> None:
        """Close step 4: forget the announced ids. Safe twice, and before, during or after :meth:`open`."""
        self._closed = True
        self.announced.clear()

    # ------------------------------------------------------------------------------ the frontend
    async def delegate(self, input: list[dict]) -> str:
        """Mint a delegation id, announce it to the client and return it; ``input`` is not sent anywhere.

        The id is remembered before the announcement goes out, so an append that answers it is always accepted.
        """
        if self._closed:
            raise RuntimeError("Session is closing")
        delegation_id = uid("item")
        if self._handing and self.view is not None:
            self.announced.add(delegation_id)
            await self.view.emit(
                "session.delegation.created",
                offset_ms=self.view.input_ms,
                delegation={"id": delegation_id, "type": "delegation", "target": "client"},
            )
        return delegation_id

    def listen(self, listener: DelegationEvents) -> None:
        """Nothing follows a client delegation's announcement, so ``listener`` is never handed an event."""

    # --------------------------------------------------------------------- the session's commands
    def knows(self, delegation_id: object) -> bool:
        """Whether ``delegation_id`` names a client delegation this session announced."""
        return isinstance(delegation_id, str) and delegation_id in self.announced

    async def update(self, config: SessionConfig, patch: dict, client_event_id: str | None) -> SessionConfig:
        """Apply a ``session.update`` patch by the client delegation's rule; no backend runs here to check."""
        return config.updated(patch)

    async def item_create(self, item: dict, client_event_id: str | None) -> None:
        """``response.item.create`` requires Responses delegation: refused."""
        raise ProtocolError(RESPONSES_ONLY)

    async def response_create(self, client_event_id: str | None) -> None:
        """``response.create`` requires Responses delegation: refused."""
        raise ProtocolError(RESPONSES_ONLY)

    async def history(self) -> dict | None:
        """No backend runs on the server for a client delegation, so it has no half of ``/history``."""
        return None
