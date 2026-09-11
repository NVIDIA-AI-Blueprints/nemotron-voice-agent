# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Canonical response lifecycle emission shared by transport and observer."""

from __future__ import annotations

from typing import Literal

from realtime.controller import OutputKind, RealtimeSessionController
from realtime.events import EmitBatchFn


async def emit_events(emit_batch: EmitBatchFn, events: list[dict]) -> None:
    """Emit one controller transition as an indivisible ordered batch."""
    if events:
        await emit_batch(events)


async def announce_response(
    controller: RealtimeSessionController,
    emit_batch: EmitBatchFn,
    *,
    kind: OutputKind | None = None,
) -> tuple[str, bool]:
    """Ensure a response and its assistant message are announced once.

    Capture the response owner before publishing. A concurrent interruption may
    terminalize that response while the WebSocket batch is being written; the
    completed announcement still belongs to the captured response and must not
    be reported as an internal lifecycle failure.
    """
    was_active = controller.response_in_progress
    events = controller.ensure_assistant_message(kind)
    response_id = controller.active_response_id
    if response_id is None:
        raise RuntimeError("Realtime controller did not create a response")
    await emit_events(emit_batch, events)
    return response_id, not was_active


async def finish_response(
    controller: RealtimeSessionController,
    emit_batch: EmitBatchFn,
    *,
    status: Literal["completed", "cancelled", "failed", "incomplete"],
    reason: str | None = None,
) -> bool:
    """Emit exactly one terminal sequence for the active response."""
    events = controller.finish_response(status=status, reason=reason)
    await emit_events(emit_batch, events)
    return bool(events)
