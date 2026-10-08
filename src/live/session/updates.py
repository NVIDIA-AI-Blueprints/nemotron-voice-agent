# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``session.update``: changing the backend settings of a running session.

Only ``delegation.responses`` can change, and only in ``responses`` mode. A patch is sparse: nested settings merge,
so changing ``reasoning.effort`` keeps ``reasoning.summary``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from live.protocol import ProtocolError, SessionConfig

if TYPE_CHECKING:
    from live.session.session import LiveProtocolSession

RESPONSES_FIELDS = {
    "model",
    "instructions",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "reasoning",
    "text",
    "service_tier",
    "max_output_tokens",
}


def apply_update(config: SessionConfig, patch: object) -> SessionConfig:
    """Return ``config`` with the sparse ``patch`` merged in; raise ``ProtocolError`` if the patch is not allowed."""
    if config.mode != "responses":
        raise ProtocolError("session.update requires Responses delegation")
    if not isinstance(patch, dict) or set(patch) != {"delegation"}:
        raise ProtocolError("Only delegation.responses can change", "immutable_field_update")
    delegation = patch["delegation"]
    if (
        not isinstance(delegation, dict)
        or set(delegation) - {"type", "responses"}
        or delegation.get("type", "responses") != "responses"
        or not isinstance(delegation.get("responses"), dict)
    ):
        raise ProtocolError("The delegation type is immutable", "immutable_field_update")
    changes = delegation["responses"]
    if set(changes) - RESPONSES_FIELDS:
        raise ProtocolError("Unsupported backend setting")
    resolved = deepcopy(config.model_dump(exclude_none=True))
    # Sparse patches merge nested settings so reasoning.summary survives an effort change.
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(resolved["delegation"]["responses"].get(key), dict):
            resolved["delegation"]["responses"][key].update(value)
        else:
            resolved["delegation"]["responses"][key] = value
    return SessionConfig.model_validate(resolved)


async def handle_update(session: LiveProtocolSession, message: dict, client_event_id: str | None) -> None:
    """Apply a ``session.update`` command: validate it, tell the engine, and announce ``session.updated``."""
    updated = apply_update(session.config, message.get("session"))
    if updated.mode == "responses":
        try:
            session.engine.validate_backend_model(updated.delegation.responses.model)
        except ValueError as exc:
            raise ProtocolError(str(exc), param="session.delegation.responses.model") from exc
    session.config = updated
    session.engine.update_backend(updated)
    await session.emit("session.updated", session=session.snapshot(), client_event_id=client_event_id)
