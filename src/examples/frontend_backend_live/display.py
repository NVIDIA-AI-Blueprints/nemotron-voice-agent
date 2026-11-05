# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the display part of each delegation goes: the sink's contract, and the sink of a host that shows nothing.

``NullDisplay`` serves a Live session, whose client draws from the gateway relay; the repository UI bot shows task cards
(``examples.frontend_backend_live.media.task_cards.RtviTaskCards``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from examples.frontend_backend_live.delegations import Ending, ReadyResult


class DisplaySink(Protocol):
    """Where the display part of each delegation goes: ``NullDisplay`` (Live) or ``RtviTaskCards`` (the UI bot).

    Plain calls that never wait. Nothing given here is ever spoken, and a display item never enters the talker's
    context.
    """

    def started(self, delegation_id: str, request: str) -> None:
        """A delegation started; ``request`` is what the caller asked."""

    def progress(self, delegation_id: str, line: str) -> None:
        """The delegation's newest progress line."""

    def shown(self, delegation_id: str, item: dict) -> None:
        """A display item (the ``show`` work tool's ``mcp_call``), at once."""

    def answered(self, result: ReadyResult) -> None:
        """A ready result, at once, before it is said."""

    def ended(self, delegation_id: str, how: Ending) -> None:
        """The delegation ended."""


class NullDisplay:
    """A ``DisplaySink`` that shows nothing."""

    def started(self, delegation_id: str, request: str) -> None:
        """Nothing to show."""

    def progress(self, delegation_id: str, line: str) -> None:
        """Nothing to show."""

    def shown(self, delegation_id: str, item: dict) -> None:
        """Nothing to show."""

    def answered(self, result: ReadyResult) -> None:
        """Nothing to show."""

    def ended(self, delegation_id: str, how: Ending) -> None:
        """Nothing to show."""
