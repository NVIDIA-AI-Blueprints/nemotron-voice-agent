# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``session.update``: changing the delegation settings of a running session."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateways.live.session.session import LiveProtocolSession


async def update(session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None) -> None:
    """Apply a ``session.update`` command through the delegation part, store it, and announce ``session.updated``."""
    session.config = await session.delegation.update(session.config, message.get("session"), client_event_id)
    await session.emit("session.updated", session=session.snapshot(), client_event_id=client_event_id)
