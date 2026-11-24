# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool calling from the client's side: ``response.item.create`` and ``response.create``."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from nemotron_voice_agent.pipecat.runtime.gateways.live.ports import TypedReader
from nemotron_voice_agent.pipecat.runtime.gateways.live.protocol import ProtocolError
from nemotron_voice_agent.pipecat.shared.delegation_contract import CALLER_CONTEXT, CALLER_EXCERPT
from nemotron_voice_agent.pipecat.shared.ids import approx_tokens

if TYPE_CHECKING:
    from nemotron_voice_agent.pipecat.runtime.gateways.live.session.session import LiveProtocolSession

MAX_ITEM_TOKENS = 32000


def _checked_item(message: dict) -> dict:
    item = message.get("item")
    if not isinstance(item, dict):
        raise ProtocolError("item must be an object", param="item")
    for field in (CALLER_CONTEXT, CALLER_EXCERPT):
        if field in item:
            raise ProtocolError("Caller evidence is reserved for the frontend", param=f"item.{field}")
    if approx_tokens(json.dumps(item)) > MAX_ITEM_TOKENS:
        raise ProtocolError(f"Response item is too large: at most {MAX_ITEM_TOKENS} tokens", param="item")
    return item


def _record_typed(session: LiveProtocolSession, text: str) -> None:
    if text and isinstance(session.frontend, TypedReader):
        session.frontend.typed(text)


async def item_create(session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None) -> None:
    """Check a ``response.item.create`` item's shape and size, then hand it to the delegation part."""
    item = _checked_item(message)
    await session.delegation.item_create(item, client_event_id)
    _record_typed(session, typed_text(item))


def prepare_item_create(
    session: LiveProtocolSession, message: dict, client_event_id: str | None
) -> Callable[[], Awaitable[None]]:
    """Check retained input and prepare its exact synchronous admission before reserving its operation."""
    item = _checked_item(message)
    commit = session.delegation.prepare_item_create(item)
    text = typed_text(item)

    async def deliver() -> None:
        result = commit()
        if inspect.isawaitable(result):
            await result
        _record_typed(session, text)

    return deliver


def prepare_response_create(
    session: LiveProtocolSession, message: dict, client_event_id: str | None
) -> Callable[[], Awaitable[None]]:
    """Prepare one exact continuation batch without dispatch, effects or asynchronous validation."""
    _check_response_create(message)
    commit = session.delegation.prepare_response_create(client_event_id)

    async def deliver() -> None:
        commit()

    return deliver


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
    _check_response_create(message)
    await session.delegation.response_create(client_event_id)


def _check_response_create(message: dict) -> None:
    if set(message) - {"type", "event_id"}:
        raise ProtocolError("response.create accepts only type and event_id")
