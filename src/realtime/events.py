# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Canonical OpenAI Realtime event names and wire helpers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from loguru import logger

from realtime.protocol import build_server_event

EmitFn = Callable[[dict[str, Any]], Awaitable[None]]
EmitBatchFn = Callable[[list[dict[str, Any]]], Awaitable[None]]

SERVER_SESSION_CREATED = "session.created"
SERVER_SESSION_UPDATED = "session.updated"
SERVER_CONVERSATION_CREATED = "conversation.created"
SERVER_ERROR = "error"


def error_event(
    message: str,
    *,
    code: str | None = None,
    event_id: str | None = None,
    param: str | None = None,
    metadata: dict[str, Any] | None = None,
    error_type: Literal["invalid_request_error", "server_error"] = "invalid_request_error",
) -> dict[str, Any]:
    """Build a Realtime-shaped ``error`` event."""
    err: dict[str, Any] = {
        "type": error_type,
        "message": message,
    }
    if code:
        err["code"] = code
    if param:
        err["param"] = param
    if event_id:
        err["event_id"] = event_id
    if metadata:
        err["metadata"] = metadata
    logger.info(f"Realtime error event code={code or '-'} param={param or '-'} message={message}")
    return build_server_event(SERVER_ERROR, error=err)
