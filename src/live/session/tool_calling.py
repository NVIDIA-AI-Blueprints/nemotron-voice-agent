# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool calling from the client's side: ``response.item.create`` and ``response.create``.

In ``responses`` delegation the backend's function calls stream to the client as ``response.event``. The client runs
them, returns each result with ``response.item.create``, and asks for one continuation with ``response.create``. The
engine owns the exchange; this module validates the two commands.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from live.protocol import ProtocolError, approx_tokens

if TYPE_CHECKING:
    from live.session.session import LiveProtocolSession

RESPONSE_COMMANDS = {"response.item.create", "response.create"}
MAX_ITEM_TOKENS = 32000


async def handle_response_command(session: LiveProtocolSession, kind: str, message: dict) -> None:
    """Validate a tool-calling command and pass it to the engine."""
    if session.config.mode != "responses":
        raise ProtocolError("This command requires Responses delegation")
    if kind == "response.item.create":
        item = message.get("item")
        if not isinstance(item, dict):
            raise ProtocolError("item must be an object", param="item")
        if approx_tokens(json.dumps(item)) > MAX_ITEM_TOKENS:
            raise ProtocolError("Response item is too large", param="item")
        session.engine.item_create(item)
    else:
        if set(message) - {"type", "event_id"}:
            raise ProtocolError("response.create accepts only type and event_id")
        await session.engine.response_create()
