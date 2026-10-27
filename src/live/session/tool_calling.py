# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool calling from the client's side: ``response.item.create`` and ``response.create``.

In ``responses`` delegation the backend's calls stream to the client as ``response.event``. The client runs them,
returns each result with ``response.item.create``, and continues every response whose results are complete with
``response.create``. This module checks each command's shape; the session's delegation part takes it: the
server delegation owns the exchange, and the client delegation refuses both commands, which require Responses
delegation.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from live.ports import TypedReader
from live.protocol import ProtocolError, approx_tokens

if TYPE_CHECKING:
    from live.session.session import LiveProtocolSession

MAX_ITEM_TOKENS = 32000


async def item_create(session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None) -> None:
    """``response.item.create``: check the item's shape and size, then hand it to the delegation part.

    Once the part has taken a typed user message, a frontend adapter that reads typed questions (:class:`TypedReader`)
    is given its text: the backend runs the message, and the voice knows what was asked.
    """
    item = message.get("item")
    if not isinstance(item, dict):
        raise ProtocolError("item must be an object", param="item")
    if approx_tokens(json.dumps(item)) > MAX_ITEM_TOKENS:
        raise ProtocolError(f"Response item is too large: at most {MAX_ITEM_TOKENS} tokens", param="item")
    await session.delegation.item_create(item, client_event_id)
    text = typed_text(item)
    if text and isinstance(session.frontend, TypedReader):
        session.frontend.typed(text)


def typed_text(item: dict) -> str:
    """The text parts of a typed user message, one per line; ``""`` for a result or a message with no text."""
    if item.get("type", "message") != "message" or item.get("role") != "user":
        return ""
    parts = item.get("content") if isinstance(item.get("content"), list) else []
    texts = [p["text"] for p in parts if isinstance(p, dict) and p.get("type") == "input_text"]
    return "\n".join(text for text in texts if isinstance(text, str) and text.strip())


async def response_create(
    session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None
) -> None:
    """``response.create``: check its shape, then hand it to the delegation part."""
    if set(message) - {"type", "event_id"}:
        raise ProtocolError("response.create accepts only type and event_id")
    await session.delegation.response_create(client_event_id)
