# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context appends: ``session.instructions.append``, ``session.thinking.append``, ``session.commentary.append``.

An append reaches the engine at once. Its acknowledgment (``...appended``) is held on the timeline until the caller's
audio reaches the end of the span it takes effect on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from live.events import event
from live.protocol import ProtocolError, approx_tokens

if TYPE_CHECKING:
    from live.session.session import LiveProtocolSession

APPEND_KINDS = {"session.instructions.append", "session.thinking.append", "session.commentary.append"}
MAX_CONTENT_TOKENS = 500


async def handle_append(session: LiveProtocolSession, kind: str, message: dict, client_event_id: str | None) -> None:
    """Validate an append command, pass it to the engine, and queue its acknowledgment."""
    if "delegation_id" not in message:
        raise ProtocolError("delegation_id is required (nullable)", param="delegation_id")
    delegation_id = message["delegation_id"]
    if delegation_id is not None and (
        session.config.mode == "responses" or not session.engine.knows_delegation(delegation_id)
    ):
        raise ProtocolError("Unknown or disallowed delegation_id", param="delegation_id")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip() or approx_tokens(content) > MAX_CONTENT_TOKENS:
        raise ProtocolError("content must be a nonempty string of at most 500 tokens", param="content")
    if session.timeline.full:
        raise ProtocolError("Too many pending appends", "rate_limit_exceeded")
    if kind == "session.instructions.append":
        await session.engine.append_instructions(content)
    elif kind == "session.commentary.append":
        await session.engine.append_commentary(content, delegation_id)
    else:
        await session.engine.append_thinking(content)
    session.timeline.hold(
        lambda start_ms, end_ms: event(kind + "ed", client_event_id=client_event_id, start_ms=start_ms, end_ms=end_ms)
    )
