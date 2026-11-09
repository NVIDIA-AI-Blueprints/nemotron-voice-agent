# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context appends: ``session.instructions.append``, ``session.thinking.append`` and ``session.commentary.append``."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from examples.shared.ids import approx_tokens
from gateways.live.protocol import ProtocolError

if TYPE_CHECKING:
    from gateways.live.session.session import LiveProtocolSession

MAX_CONTENT_TOKENS = 500
APPEND_WINDOW_SECONDS = 60


async def instructions(session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None) -> None:
    """Steer the frontend's behavior (``session.instructions.append``)."""
    await _append(
        session, message, client_event_id, session.frontend.append_instructions, "session.instructions.appended"
    )


async def thinking(session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None) -> None:
    """Add silent context for later decisions; an empty one only cancels the pending acknowledgements."""
    await _append(session, message, client_event_id, session.frontend.append_thinking, "session.thinking.appended")


async def commentary(session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None) -> None:
    """Add facts the frontend phrases for the caller (``session.commentary.append``)."""
    await _append(session, message, client_event_id, session.frontend.append_commentary, "session.commentary.appended")


async def _append(
    session: LiveProtocolSession,
    message: dict,
    client_event_id: str | None,
    deliver: Callable[..., Awaitable[None]],
    acknowledgment: str,
) -> None:
    """Validate an append command, pass it to the frontend adapter, then acknowledge it."""
    if "delegation_id" not in message:
        raise ProtocolError("delegation_id is required (nullable)", param="delegation_id")
    delegation_id = message["delegation_id"]
    if delegation_id is not None and not session.delegation.knows(delegation_id):
        raise ProtocolError("Unknown or disallowed delegation_id", param="delegation_id")
    content = message.get("content")
    if not isinstance(content, str) or approx_tokens(content) > MAX_CONTENT_TOKENS:
        raise ProtocolError(f"content must be a string of at most {MAX_CONTENT_TOKENS} tokens", param="content")
    now, recent = time.monotonic(), session.recent_appends
    while recent and now - recent[0] > APPEND_WINDOW_SECONDS:
        recent.popleft()
    if len(recent) >= session.max_appends_per_minute:
        raise ProtocolError(
            f"Too many appends: at most {session.max_appends_per_minute} per minute", "rate_limit_exceeded"
        )
    recent.append(now)
    await deliver(content, client_event_id, delegation_id)
    # Appended context takes no audio time, so its span is zero-length at the input clock.
    at = session.input_ms
    await session.emit(acknowledgment, client_event_id=client_event_id, start_ms=at, end_ms=at)
