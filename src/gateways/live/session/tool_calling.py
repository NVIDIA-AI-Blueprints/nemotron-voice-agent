# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool calling from the client's side: ``response.item.create`` and ``response.create``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from examples.shared.ids import approx_tokens
from gateways.live.ports import TypedReader
from gateways.live.protocol import ProtocolError

if TYPE_CHECKING:
    from gateways.live.session.session import LiveProtocolSession

MAX_ITEM_TOKENS = 32000


async def item_create(session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None) -> None:
    """Check a ``response.item.create`` item's shape and size, then hand it to the delegation part."""
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
    """Return the text parts of a typed user message, one per line; ``""`` for anything else."""
    if item.get("type", "message") != "message" or item.get("role") != "user":
        return ""
    parts = item.get("content") if isinstance(item.get("content"), list) else []
    texts = [p["text"] for p in parts if isinstance(p, dict) and p.get("type") == "input_text"]
    return "\n".join(text for text in texts if isinstance(text, str) and text.strip())


async def response_create(
    session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None
) -> None:
    """Check a ``response.create`` command's shape, then hand it to the delegation part."""
    if set(message) - {"type", "event_id"}:
        raise ProtocolError("response.create accepts only type and event_id")
    await session.delegation.response_create(client_event_id)
