# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Shim-side conversation / response IDs for the Realtime wire protocol."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


def new_item_id() -> str:
    """Return a Realtime-shaped conversation item id."""
    return f"item_{uuid.uuid4().hex[:12]}"


def new_response_id() -> str:
    """Return a Realtime-shaped response id."""
    return f"resp_{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class ResponseSnapshot:
    """Ids/transcript captured at complete time (safe across async finish)."""

    generation: int
    response_id: str
    item_id: str | None
    transcript: str
    status: str
    audio_done_emitted: bool
    transcript_done_emitted: bool
    output_text_emitted: bool


@dataclass
class ConversationState:
    """Track active user/assistant items and the in-flight response.

    Shared by :class:`RealtimeFrameSerializer` (audio path) and
    :class:`RealtimeLifecycleObserver` (transcripts / lifecycle events).
    """

    user_item_id: str | None = None
    assistant_item_id: str | None = None
    response_id: str | None = None
    response_status: str | None = None  # in_progress | completed | cancelled
    response_generation: int = 0
    assistant_transcript: str = ""
    output_item_announced: bool = False
    content_part_announced: bool = False
    transcript_done_emitted: bool = False
    audio_done_emitted: bool = False
    # True after ≥1 response.output_text.delta this turn.
    output_text_emitted: bool = False
    item_transcripts: dict[str, str] = field(default_factory=dict)
    pending_function_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    # False during welcome window; True after first assistant response.done
    # (or immediately when welcome is disabled).
    assistant_has_responded: bool = False
    # After response.done, trailing PCM may still arrive; stream it on these ids
    # without opening a new assistant item.
    closed_response_id: str | None = None
    closed_item_id: str | None = None

    def open_client_text(self) -> None:
        """Allow client user-text items."""
        self.assistant_has_responded = True

    def begin_user_item(self) -> str:
        """Allocate a user item id for the current turn (idempotent per turn)."""
        if not self.user_item_id:
            self.user_item_id = new_item_id()
        return self.user_item_id

    def clear_user_item(self) -> None:
        """Clear the active user item after a final transcript."""
        self.user_item_id = None

    def begin_response(self) -> tuple[str, bool]:
        """Ensure an in-progress response exists.

        Returns:
            ``(response_id, newly_created)``.

        A new response may start while a prior finish sequence is still
        emitting; finish uses :class:`ResponseSnapshot` + generation-guarded
        reset so the new turn is not wiped.
        """
        if self.response_id and self.response_status == "in_progress":
            return self.response_id, False
        self.response_generation += 1
        self.response_id = new_response_id()
        self.response_status = "in_progress"
        self.assistant_item_id = new_item_id()
        self.assistant_transcript = ""
        self.output_item_announced = False
        self.content_part_announced = False
        self.transcript_done_emitted = False
        self.audio_done_emitted = False
        self.output_text_emitted = False
        self.closed_response_id = None
        self.closed_item_id = None
        return self.response_id, True

    def append_assistant_transcript(self, text: str) -> None:
        """Accumulate bot speech transcript for the active response."""
        if not text:
            return
        self.begin_response()
        self.assistant_transcript += text
        if self.assistant_item_id:
            self.item_transcripts[self.assistant_item_id] = self.assistant_transcript

    def remember_function_call(
        self,
        call_id: str,
        *,
        name: str,
        arguments: Any = None,
    ) -> None:
        """Store in-flight function-call metadata for a later client output."""
        if not call_id:
            return
        self.pending_function_calls[call_id] = {
            "name": name or "",
            "arguments": arguments if arguments is not None else {},
        }

    def pop_function_call(self, call_id: str) -> dict[str, Any] | None:
        """Return and clear stored metadata for ``call_id`` (if any)."""
        if not call_id:
            return None
        return self.pending_function_calls.pop(call_id, None)

    def complete_response(self, status: str = "completed") -> ResponseSnapshot | None:
        """Mark the in-flight response finished and return a wire snapshot.

        Returns ``None`` when there was no in-progress response (caller should
        not emit ``response.done``).
        """
        if not self.response_id or self.response_status != "in_progress":
            return None
        if self.assistant_item_id and self.assistant_transcript:
            self.item_transcripts[self.assistant_item_id] = self.assistant_transcript
        snap = ResponseSnapshot(
            generation=self.response_generation,
            response_id=self.response_id,
            item_id=self.assistant_item_id,
            transcript=self.assistant_transcript,
            status=status,
            audio_done_emitted=self.audio_done_emitted,
            transcript_done_emitted=self.transcript_done_emitted,
            output_text_emitted=self.output_text_emitted,
        )
        self.closed_response_id = self.response_id
        self.closed_item_id = self.assistant_item_id
        self.response_status = status
        self.output_text_emitted = False
        self.open_client_text()
        return snap

    def reset_response_slot(self, *, generation: int | None = None) -> None:
        """Clear response slot so the next bot turn allocates fresh ids.

        When ``generation`` is set, skip the reset if a newer response already
        owns the slot (finish raced with ``begin_response``).
        """
        if generation is not None and generation != self.response_generation:
            return
        self.response_id = None
        self.response_status = None
        self.assistant_item_id = None
        self.assistant_transcript = ""
        self.output_item_announced = False
        self.content_part_announced = False
        self.transcript_done_emitted = False
        self.audio_done_emitted = False
        self.output_text_emitted = False
