# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Standards-compatible out-of-band Realtime projections for the UI.

VoiceClaw application state is represented as an ordinary text response with
``conversation_id`` set to ``null``. Correlation and routing live in response
metadata, whose values remain strings as required by the Realtime contract.
The projection is consumable by an ordinary Realtime client without inventing
a second WebSocket protocol or polluting model conversation history.
"""

from __future__ import annotations

import copy
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from voiceclaw.domain.response_only import REALTIME_PROJECTION_SCHEMA

_MAX_TEXT_CHARACTERS = 128_000
_MAX_METADATA_ENTRIES = 16
_MAX_METADATA_KEY_CHARACTERS = 64
_MAX_METADATA_VALUE_CHARACTERS = 512
_RESERVED_CORRELATION_KEYS = frozenset({"schema", "session_id", "kind", "phase", "title", "request_summary"})
_MAX_CORRELATION_ENTRIES = _MAX_METADATA_ENTRIES - 5


class ProjectionStreamAbortStatus(StrEnum):
    """Standard non-success terminal states for a projection stream."""

    CANCELLED = "cancelled"
    FAILED = "failed"


def _required(value: str, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(normalized) > maximum or "\x00" in normalized:
        raise ValueError(f"{name} is invalid")
    return normalized


@dataclass(frozen=True, slots=True)
class Projection:
    """One deterministic UI update that does not enter model context."""

    session_id: str
    kind: str
    phase: str
    title: str
    text: str
    correlation: Mapping[str, str] = field(default_factory=dict)
    request_summary: str | None = None

    def __post_init__(self) -> None:
        """Validate public text and freeze correlation identity."""
        object.__setattr__(self, "session_id", _required(self.session_id, "session_id"))
        object.__setattr__(self, "kind", _required(self.kind, "kind", maximum=64))
        object.__setattr__(self, "phase", _required(self.phase, "phase", maximum=64))
        object.__setattr__(self, "title", _required(self.title, "title"))
        if self.request_summary is not None:
            object.__setattr__(
                self,
                "request_summary",
                _required(self.request_summary, "request_summary", maximum=_MAX_METADATA_VALUE_CHARACTERS),
            )
        if not isinstance(self.text, str) or len(self.text) > _MAX_TEXT_CHARACTERS or "\x00" in self.text:
            raise ValueError("text is invalid")
        normalized: dict[str, str] = {}
        for key, value in self.correlation.items():
            safe_key = _required(key, "correlation key", maximum=40)
            if safe_key in _RESERVED_CORRELATION_KEYS:
                raise ValueError(f"correlation key {safe_key!r} is reserved")
            safe_value = _required(value, f"correlation.{safe_key}")
            normalized[safe_key] = safe_value
        maximum_correlation_entries = _MAX_CORRELATION_ENTRIES - int(self.request_summary is not None)
        if len(normalized) > maximum_correlation_entries:
            raise ValueError("projection correlation exceeds the Realtime metadata entry limit")
        object.__setattr__(self, "correlation", MappingProxyType(normalized))

    def metadata(self) -> dict[str, str]:
        """Return bounded response metadata used by a generic Realtime UI."""
        metadata = {
            "voiceclaw_schema": REALTIME_PROJECTION_SCHEMA,
            "voiceclaw_session_id": self.session_id,
            "voiceclaw_kind": self.kind,
            "voiceclaw_phase": self.phase,
            "voiceclaw_title": self.title,
        }
        if self.request_summary is not None:
            metadata["voiceclaw_request_summary"] = self.request_summary
        for key, value in self.correlation.items():
            metadata[f"voiceclaw_{key}"] = value
        if len(metadata) > _MAX_METADATA_ENTRIES:
            raise ValueError("projection metadata exceeds the Realtime entry limit")
        if any(len(key) > _MAX_METADATA_KEY_CHARACTERS for key in metadata):
            raise ValueError("projection metadata key exceeds the Realtime limit")
        if any(len(value) > _MAX_METADATA_VALUE_CHARACTERS for value in metadata.values()):
            raise ValueError("projection metadata value exceeds the Realtime limit")
        return metadata


class ProjectionEventFactory:
    """Render a complete, isolated Realtime text-response lifecycle."""

    def __init__(self, *, id_factory: Callable[[str], str] | None = None) -> None:
        """Allow deterministic IDs in tests while using unguessable IDs at runtime."""
        self._id_factory = id_factory or self._new_id

    def render(self, projection: Projection) -> tuple[dict[str, object], ...]:
        """Build one completed out-of-band response with no conversation item events."""
        response_id = self._id_factory("resp_vc")
        item_id = self._id_factory("item_vc")
        metadata = projection.metadata()
        in_progress_item: dict[str, object] = {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        completed_item: dict[str, object] = {
            **in_progress_item,
            "status": "completed",
            "content": [{"type": "output_text", "text": projection.text}],
        }
        response: dict[str, object] = {
            "id": response_id,
            "object": "realtime.response",
            "status": "in_progress",
            "status_details": None,
            "output": [],
            "conversation_id": None,
            "output_modalities": ["text"],
            "usage": None,
            "metadata": metadata,
        }
        fields = {
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
        }
        events = (
            {"type": "response.created", "response": copy.deepcopy(response)},
            {
                "type": "response.output_item.added",
                "response_id": response_id,
                "output_index": 0,
                "item": copy.deepcopy(in_progress_item),
            },
            {
                "type": "response.content_part.added",
                **fields,
                "part": {"type": "text", "text": ""},
            },
            {"type": "response.output_text.delta", **fields, "delta": projection.text},
            {"type": "response.output_text.done", **fields, "text": projection.text},
            {
                "type": "response.content_part.done",
                **fields,
                "part": {"type": "text", "text": projection.text},
            },
            {
                "type": "response.output_item.done",
                "response_id": response_id,
                "output_index": 0,
                "item": copy.deepcopy(completed_item),
            },
            {
                "type": "response.done",
                "response": {
                    **response,
                    "status": "completed",
                    "output": [copy.deepcopy(completed_item)],
                    "usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
                },
            },
        )
        return tuple(self._with_event_id(event) for event in events)

    def stream(self, projection: Projection) -> ProjectionEventStream:
        """Open one out-of-band response whose text can arrive incrementally."""
        return ProjectionEventStream(
            projection=projection,
            id_factory=self._id_factory,
        )

    def _with_event_id(self, event: dict[str, object]) -> dict[str, object]:
        return {"event_id": self._id_factory("event_vc"), **event}

    @staticmethod
    def _new_id(prefix: str) -> str:
        return f"{prefix}_{secrets.token_urlsafe(12)}"


class ProjectionEventStream:
    """Render one standard Realtime text response across multiple deltas.

    The response metadata identifies application state. Text remains
    provisional until :meth:`finish` validates that the terminal projection is
    an extension of the deltas already emitted. This class does not interpret
    the text as Markdown, speech, or domain state.
    """

    def __init__(
        self,
        *,
        projection: Projection,
        id_factory: Callable[[str], str],
    ) -> None:
        """Allocate stable response and item identities for one stream."""
        self._initial = projection
        self._id_factory = id_factory
        self._response_id = id_factory("resp_vc")
        self._item_id = id_factory("item_vc")
        self._text = ""
        self._started = False
        self._finished = False

    @property
    def text(self) -> str:
        """Return the exact provisional text emitted so far."""
        return self._text

    def start(self) -> tuple[dict[str, object], ...]:
        """Render the opening events for the isolated text response."""
        if self._started or self._finished:
            raise RuntimeError("projection stream has already started")
        self._started = True
        response = self._response(status="in_progress", projection=self._initial, output=[])
        item = self._item(status="in_progress", text=None)
        fields = self._fields()
        return tuple(
            self._with_event_id(event)
            for event in (
                {"type": "response.created", "response": response},
                {
                    "type": "response.output_item.added",
                    "response_id": self._response_id,
                    "output_index": 0,
                    "item": item,
                },
                {
                    "type": "response.content_part.added",
                    **fields,
                    "part": {"type": "text", "text": ""},
                },
            )
        )

    def delta(self, text: str) -> dict[str, object]:
        """Render one uninterpreted text delta without trimming whitespace."""
        if not self._started or self._finished:
            raise RuntimeError("projection stream is not active")
        if not isinstance(text, str) or not text or "\x00" in text:
            raise ValueError("projection delta must be non-empty text without NUL")
        if len(self._text) + len(text) > _MAX_TEXT_CHARACTERS:
            raise ValueError("projection stream exceeds the text limit")
        self._text += text
        return self._with_event_id(
            {
                "type": "response.output_text.delta",
                **self._fields(),
                "delta": text,
            }
        )

    def finish(self, projection: Projection) -> tuple[dict[str, object], ...]:
        """Commit terminal text and metadata after the producer validates it."""
        if not self._started or self._finished:
            raise RuntimeError("projection stream is not active")
        if projection.session_id != self._initial.session_id or projection.kind != self._initial.kind:
            raise ValueError("terminal projection changed stream identity")
        if not projection.text.startswith(self._text):
            raise ValueError("terminal projection contradicts emitted text")

        events: list[dict[str, object]] = []
        remaining = projection.text[len(self._text) :]
        if remaining:
            events.append(self.delta(remaining))
        self._finished = True
        item = self._item(status="completed", text=projection.text)
        fields = self._fields()
        events.extend(
            self._with_event_id(event)
            for event in (
                {
                    "type": "response.output_text.done",
                    **fields,
                    "text": projection.text,
                },
                {
                    "type": "response.content_part.done",
                    **fields,
                    "part": {"type": "text", "text": projection.text},
                },
                {
                    "type": "response.output_item.done",
                    "response_id": self._response_id,
                    "output_index": 0,
                    "item": item,
                },
                {
                    "type": "response.done",
                    "response": self._response(
                        status="completed",
                        projection=projection,
                        output=[item],
                    ),
                },
            )
        )
        return tuple(events)

    def abort(
        self,
        projection: Projection,
        *,
        status: ProjectionStreamAbortStatus,
        reason: str,
    ) -> tuple[dict[str, object], ...]:
        """Close provisional output without committing it as a successful result."""
        if not self._started or self._finished:
            raise RuntimeError("projection stream is not active")
        if projection.session_id != self._initial.session_id or projection.kind != self._initial.kind:
            raise ValueError("terminal projection changed stream identity")
        if projection.text != self._text:
            raise ValueError("aborted projection must contain exactly the provisional text")
        try:
            terminal_status = ProjectionStreamAbortStatus(status)
        except ValueError as error:
            raise ValueError("projection abort status is invalid") from error
        safe_reason = _required(reason, "reason", maximum=128)
        status_details: dict[str, object]
        if terminal_status is ProjectionStreamAbortStatus.CANCELLED:
            status_details = {"type": "cancelled", "reason": safe_reason}
        else:
            status_details = {
                "type": "failed",
                "error": {"type": "server_error", "code": safe_reason},
            }

        self._finished = True
        item = self._item(status="incomplete", text=self._text)
        fields = self._fields()
        events = (
            {
                "type": "response.output_text.done",
                **fields,
                "text": self._text,
            },
            {
                "type": "response.content_part.done",
                **fields,
                "part": {"type": "text", "text": self._text},
            },
            {
                "type": "response.output_item.done",
                "response_id": self._response_id,
                "output_index": 0,
                "item": item,
            },
            {
                "type": "response.done",
                "response": self._response(
                    status=terminal_status,
                    projection=projection,
                    output=[item],
                    status_details=status_details,
                ),
            },
        )
        return tuple(self._with_event_id(event) for event in events)

    def _fields(self) -> dict[str, object]:
        return {
            "response_id": self._response_id,
            "item_id": self._item_id,
            "output_index": 0,
            "content_index": 0,
        }

    def _item(self, *, status: str, text: str | None) -> dict[str, object]:
        content: list[dict[str, object]] = [] if text is None else [{"type": "output_text", "text": text}]
        return {
            "id": self._item_id,
            "object": "realtime.item",
            "type": "message",
            "status": status,
            "role": "assistant",
            "content": content,
        }

    def _response(
        self,
        *,
        status: str,
        projection: Projection,
        output: list[dict[str, object]],
        status_details: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "id": self._response_id,
            "object": "realtime.response",
            "status": status,
            "status_details": status_details,
            "output": output,
            "conversation_id": None,
            "output_modalities": ["text"],
            "usage": (None if status == "in_progress" else {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}),
            "metadata": projection.metadata(),
        }

    def _with_event_id(self, event: dict[str, object]) -> dict[str, object]:
        return {"event_id": self._id_factory("event_vc"), **event}


__all__ = [
    "Projection",
    "ProjectionEventFactory",
    "ProjectionEventStream",
    "ProjectionStreamAbortStatus",
]
