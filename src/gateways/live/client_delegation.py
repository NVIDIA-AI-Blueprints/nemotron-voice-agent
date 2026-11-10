# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The client delegation: Live client mode, in which the client app does the delegated work."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from examples.shared.ids import uid
from gateways.live.protocol import ProtocolError

if TYPE_CHECKING:
    from gateways.live.ports import DelegationEvents, DelegationView
    from gateways.live.protocol import SessionConfig

RESPONSES_ONLY = "This command requires Responses delegation"


class ClientDelegation:
    """Announces one client-mode session's delegations to its client app, and remembers their ids."""

    target: Literal["client"] = "client"
    reports_events = False
    conversation = None
    close_seconds = 10.0

    def __init__(self):
        """Start with nothing announced; nothing is sent until :meth:`delegate`."""
        self.view: DelegationView | None = None
        self.announced: set[str] = set()
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

    async def delegate(self, input: list[dict]) -> str:
        """Mint a delegation id, announce it to the client and return it; ``input`` goes nowhere.

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
        """Ignore ``listener``: no event follows a client delegation's announcement."""

    def knows(self, delegation_id: object) -> bool:
        """Return whether ``delegation_id`` names a client delegation this session announced."""
        return isinstance(delegation_id, str) and delegation_id in self.announced

    async def update(self, config: SessionConfig, patch: dict, client_event_id: str | None) -> SessionConfig:
        """Apply a ``session.update`` patch by the client delegation's rule; no backend runs here to check."""
        return config.updated(patch)

    async def item_create(self, item: dict, client_event_id: str | None) -> None:
        """Refuse ``response.item.create``, which requires Responses delegation."""
        raise ProtocolError(RESPONSES_ONLY)

    async def response_create(self, client_event_id: str | None) -> None:
        """Refuse ``response.create``, which requires Responses delegation."""
        raise ProtocolError(RESPONSES_ONLY)

    async def history(self) -> dict | None:
        """Return ``None``: no backend runs on the server for a client delegation."""
        return None
