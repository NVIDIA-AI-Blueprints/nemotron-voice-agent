# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context appends: ``session.instructions.append``, ``session.thinking.append``, ``session.commentary.append``.

An append reaches the frontend adapter at once, with the ``delegation_id`` it names: ``None`` for session-wide context,
or a client delegation this session announced (only the client delegation knows any). Its acknowledgment
(``...appended``) follows once the adapter has taken the context, with the zero-length span at the input clock that
appended context takes.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from live.protocol import ProtocolError, approx_tokens

if TYPE_CHECKING:
    from live.session.session import LiveProtocolSession

MAX_CONTENT_TOKENS = 500


async def instructions(session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None) -> None:
    """``session.instructions.append``: steer the frontend's behavior."""
    await _append(
        session, message, client_event_id, session.frontend.append_instructions, "session.instructions.appended"
    )


async def thinking(session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None) -> None:
    """``session.thinking.append``: silent context for later decisions."""
    await _append(session, message, client_event_id, session.frontend.append_thinking, "session.thinking.appended")


async def commentary(session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None) -> None:
    """``session.commentary.append``: facts the frontend phrases for the caller."""
    await _append(session, message, client_event_id, session.frontend.append_commentary, "session.commentary.appended")


async def _append(
    session: LiveProtocolSession,
    message: dict,
    client_event_id: str | None,
    deliver: Callable[..., Awaitable[None]],
    acknowledgment: str,
) -> None:
    """Validate an append command, pass it to the frontend adapter, and queue its acknowledgment."""
    if "delegation_id" not in message:
        raise ProtocolError("delegation_id is required (nullable)", param="delegation_id")
    delegation_id = message["delegation_id"]
    # A non-null ID must name a client delegation that this session announced; a server delegation knows none.
    if delegation_id is not None and not session.delegation.knows(delegation_id):
        raise ProtocolError("Unknown or disallowed delegation_id", param="delegation_id")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip() or approx_tokens(content) > MAX_CONTENT_TOKENS:
        raise ProtocolError(
            f"content must be a nonempty string of at most {MAX_CONTENT_TOKENS} tokens", param="content"
        )
    await deliver(content, client_event_id, delegation_id)
    at = session.input_ms
    await session.emit(acknowledgment, client_event_id=client_event_id, start_ms=at, end_ms=at)
