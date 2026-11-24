# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded canonical delivery evidence for one process-local reconnectable call."""

from __future__ import annotations

import json
from collections.abc import Collection
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Literal

from nemotron_voice_agent.pipecat.runtime.gateways.live.protocol import ProtocolError

if TYPE_CHECKING:
    from nemotron_voice_agent.pipecat.runtime.gateways.live.ports import DelegationRequestState

MAX_ITEMS = 512
MAX_BYTES = 2 * 1024 * 1024
DUPLICATE = "duplicate"
KEPT_EVENTS = frozenset(
    {
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.output_item.done",
        "response.completed",
        "response.failed",
        "response.incomplete",
    }
)


class MailboxFull(ProtocolError):
    """The call cannot retain a complete recovery journal or admit new publications."""

    def __init__(self):
        """Use the existing capacity outcome for both call closure and request refusal."""
        super().__init__("The reconnect result mailbox is full. Start a new call.", "rate_limit_exceeded")


@dataclass
class Entry:
    """A canonical event and whether any frontend already had a chance to present it."""

    delegation_id: str | None
    event: dict
    forwarded: bool = False
    lost: bool = False
    request_state: DelegationRequestState | None = None
    request_ref: str | None = None


@dataclass
class RecoveringCall:
    """A previously emitted app call whose original result can still be reconciled."""

    name: str
    state: Literal["unknown", "waiting", "recorded"] = "unknown"


class DeliveryMailbox:
    """Retain completion facts without treating a frontend callback as proof of hearing."""

    def __init__(self):
        """Start empty; this owner outlives each attachment and has exactly one backend reader."""
        self.entries: list[Entry] = []
        self.keys: set[tuple] = set()
        self.bytes = 0
        self.exhausted = False
        self.recovering_calls: dict[str, RecoveringCall] = {}
        self.emitted: set[str] = set()

    def record(
        self, delegation_id: str, raw: dict, *, response_id: str | None = None
    ) -> Entry | Literal["duplicate"] | None:
        """Record a structural event once, distinguishing a repeat from unjournaled stream data."""
        kind = raw.get("type")
        if kind not in KEPT_EVENTS:
            return None
        item = raw.get("item") or {}
        if kind == "response.output_item.added" and item.get("type") != "mcp_call":
            return None  # Partial messages and incomplete function arguments are never canonical replay data.
        event = dict(raw)
        if "response" in raw:
            # Context restoration never needs prompt text, credentials, input or a second copy of output items.
            event["response"] = {
                name: value
                for name, value in raw["response"].items()
                if name not in {"instructions", "input", "tools", "output"}
            }
        frame = raw.get("response_id") or (raw.get("response") or {}).get("id") or response_id
        identity = item.get("id") or item.get("call_id")
        if kind == "response.output_item.done" and isinstance(identity, str) and identity:
            key = delegation_id, kind, frame, identity
        else:
            semantic = {name: value for name, value in event.items() if name != "sequence_number"}
            digest = sha256(json.dumps(semantic, ensure_ascii=False, sort_keys=True).encode()).digest()
            key = delegation_id, kind, frame, digest
        if key in self.keys:
            return DUPLICATE
        return self._append(key, Entry(delegation_id, deepcopy(event)))

    def record_request(
        self, request_ref: str, state: DelegationRequestState, delegation_id: str | None = None
    ) -> Entry | None:
        """Retain a private transport fact in the same ordered, bounded journal as the response frames."""
        previous = self.entries[-1] if self.entries else None
        replaces_submission = bool(
            state in {"accepted", "failed", "cancelled"}
            and previous is not None
            and previous.request_state == "submitted"
            and previous.request_ref == request_ref
            and previous.delegation_id == delegation_id
        )
        return self._append(
            (request_ref, "request", state),
            Entry(delegation_id, {}, request_state=state, request_ref=request_ref),
            replace_last=replaces_submission,
        )

    @staticmethod
    def _size(entry: Entry) -> int:
        payload = (
            {
                "request_ref": entry.request_ref,
                "delegation_id": entry.delegation_id,
                "request_state": entry.request_state,
            }
            if entry.request_state is not None
            else entry.event
        )
        return len(json.dumps(payload, ensure_ascii=False).encode())

    def _append(self, key: tuple, entry: Entry, *, replace_last: bool = False) -> Entry | None:
        if key in self.keys:
            return None
        if self.exhausted:
            raise MailboxFull()
        retained_bytes = self.bytes + self._size(entry)
        if replace_last:
            retained_bytes -= self._size(self.entries[-1])
        if len(self.entries) + (not replace_last) > MAX_ITEMS or retained_bytes > MAX_BYTES:
            self.exhausted = True
            raise MailboxFull()
        # Keep the compacted submission's dedup key. Each replacement retires at most one such key,
        # and the resulting disposition remains a retained record subject to the same item limit.
        self.keys.add(key)
        self.bytes = retained_bytes
        if replace_last:
            self.entries[-1] = entry
        else:
            self.entries.append(entry)
        return entry

    def mark_emitted(self, call_id: str) -> None:
        """Record that the completed app call was accepted by the primary connection's queue."""
        if not self.exhausted:
            self.emitted.add(call_id)

    def disconnected(self, routes) -> None:
        """An emitted call without committed evidence is uncertain and must never be re-emitted."""
        self.revalidate_calls(())
        committed = {result.key for result in routes.results} | routes.answered
        for call_id, route in routes.routes.items():
            if call_id in self.emitted and call_id not in committed:
                self.recovering_calls.setdefault(call_id, RecoveringCall(route.name))

    def revalidate_calls(self, pending: Collection[str]) -> None:
        """Use the Core's current snapshot to distinguish pending frames from evidence-only calls."""
        for call_id, call in self.recovering_calls.items():
            if call.state != "recorded":
                call.state = "waiting" if call_id in pending else "unknown"

    def record_result(self, call_id: str) -> None:
        """Retain the Core's acceptance separately from possible external dispatch."""
        call = self.recovering_calls.get(call_id)
        if call is not None:
            call.state = "recorded"

    def replay(self):
        """Yield structural state plus only answers which no old frontend could have presented."""
        yield from self.entries

    def snapshot(self) -> dict:
        """Return bounded safe result text and uncertain action names, without arguments or credentials."""
        results = []
        for entry in self.entries:
            item = entry.event.get("item") or {}
            if item.get("type") == "message" and item.get("phase", "final_answer") != "commentary":
                text = "".join(part.get("text", "") for part in item.get("content", []) if isinstance(part, dict))
                results.append(
                    {
                        "id": item.get("id"),
                        "text": text,
                        "presentation": "uncertain" if entry.forwarded else "never_forwarded",
                    }
                )
        return {
            "results": results,
            "unclear_calls": [
                {"call_id": call_id, "name": call.name}
                for call_id, call in self.recovering_calls.items()
                if call.state != "recorded"
            ],
        }
