# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private bounded frontend text for replacing an audio attachment, without replaying audio or actions."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.voice import Voice

MAX_ATTACHMENT_BYTES = 256 * 1024
MAX_ATTACHMENT_ITEMS = 512


def _json_bytes(value: dict) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode())


def _retained_messages(messages: list[dict], snapshot: dict) -> list[dict]:
    """Give pending input and its exact related tool rows priority within both snapshot limits."""
    # Charge the intervening `, ` separators with each row, then credit the first row's separator once.
    costs = [_json_bytes(row) + 2 for row in messages]
    room = MAX_ATTACHMENT_BYTES - _json_bytes(snapshot)
    if len(messages) <= MAX_ATTACHMENT_ITEMS and sum(costs) - (2 if costs else 0) <= room:
        return messages

    required = set()
    call_ids: list[set[str]] = []
    rows_by_call: dict[str, list[int]] = {}
    for index, row in enumerate(messages):
        delivery = row["delivery"] or {}
        if (row["kind"] == "caller" and not row["decided"]) or delivery.get("state") in {
            "never_forwarded",
            "unknown",
            "attempted",
        }:
            required.add(index)
        fields = row["call_fields"]
        linked = {call["id"] for call in fields.get("tool_calls", ())}
        linked.update(key for key in (fields.get("tool_call_id"), delivery.get("call_id")) if key is not None)
        call_ids.append(linked)
        for key in linked:
            rows_by_call.setdefault(key, []).append(index)

    def complete_group(initial: set[int]) -> set[int]:
        selected, pending = set(initial), list(initial)
        visited = set()
        while pending:
            for key in call_ids[pending.pop()] - visited:
                visited.add(key)
                for index in rows_by_call[key]:
                    if index not in selected:
                        selected.add(index)
                        pending.append(index)
        return selected

    selected = complete_group(required)
    if len(selected) > MAX_ATTACHMENT_ITEMS:
        raise ValueError("Pending frontend input and tool evidence exceeds the attachment item limit")
    # The snapshot already contains the empty array. Reserve the omission marker before adding any rows.
    remaining = MAX_ATTACHMENT_BYTES - _json_bytes({**snapshot, "messages_omitted": True})
    remaining += 2 - sum(costs[index] for index in selected)
    if remaining < 0:
        raise ValueError("Required frontend attachment state exceeds 256 KiB")
    considered = set(selected)
    for index in reversed(range(len(messages))):
        if index in considered:
            continue
        group = complete_group({index})
        considered.update(group)
        cost = sum(costs[member] for member in group)
        if cost <= remaining and len(selected) + len(group) <= MAX_ATTACHMENT_ITEMS:
            selected.update(group)
            remaining -= cost
    return [row for index, row in enumerate(messages) if index in selected]


def _retained_presentations(rows: Iterable[tuple[int, dict]], remaining: int) -> list[dict]:
    retained = []
    for position, row in rows:
        cost = _json_bytes(row) + (2 if retained else 0)
        if cost <= remaining:
            retained.append((position, row))
            remaining -= cost
    return [row for _, row in sorted(retained, key=lambda item: item[0])]


def export_attachment(voice: Voice) -> dict:
    """Export bounded private context and playback evidence for reconnect."""
    state, rows = voice.export_attachment(MAX_ATTACHMENT_ITEMS)
    messages = state.pop("messages")
    snapshot = {"version": 3, **state, "messages": [], "presentations": []}
    snapshot["messages"] = _retained_messages(messages, snapshot)
    if len(snapshot["messages"]) != len(messages):
        snapshot["messages_omitted"] = True
    snapshot["presentations"] = _retained_presentations(rows, MAX_ATTACHMENT_BYTES - _json_bytes(snapshot))
    encoded = json.dumps(snapshot, ensure_ascii=False)
    if len(encoded.encode()) > MAX_ATTACHMENT_BYTES:
        raise ValueError("Frontend attachment snapshot exceeds 256 KiB")
    return json.loads(encoded)


def restore_attachment(voice: Voice, snapshot: dict) -> None:
    """Validate the private snapshot before handing it to the replacement voice."""
    if snapshot.get("version") not in {2, 3} or _json_bytes(snapshot) > MAX_ATTACHMENT_BYTES:
        raise ValueError("Invalid or oversized frontend attachment snapshot")
    if any(len(snapshot.get(key, ())) > MAX_ATTACHMENT_ITEMS for key in ("messages", "heard", "uncertain")):
        raise ValueError("Frontend attachment snapshot exceeds its item limit")
    voice.restore_attachment(snapshot)


def resume_presentation(voice: Voice) -> None:
    """Reconcile evidence after the host restores canonical work ownership."""
    voice.resume_attachment()
