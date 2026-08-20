# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Shim-side conversation / response IDs for the Realtime wire protocol."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


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


@dataclass
class ConversationState:
    """Track active user/assistant items and the in-flight response.

    Shared by :class:`RealtimeFrameSerializer` (audio path) and
    :class:`RealtimeLifecycleObserver` (transcripts / lifecycle events).
    """

    user_item_id: str | None = None
    input_audio_samples: int = 0
    input_sample_rate: int = 16000
    user_turn_start_sample: int | None = None
    user_turn_end_sample: int | None = None
    user_turn_stopped: bool = False
    user_item_announced: bool = False
    pending_user_transcript: str | None = None
    assistant_item_id: str | None = None
    response_id: str | None = None
    response_status: str | None = None  # in_progress | completed | cancelled | failed
    response_requested: bool = False
    response_generation: int = 0
    assistant_transcript: str = ""
    output_item_announced: bool = False
    content_part_announced: bool = False
    transcript_done_emitted: bool = False
    audio_done_emitted: bool = False
    item_transcripts: dict[str, str] = field(default_factory=dict)
    # False during welcome window; True after first assistant response.done
    # (or immediately when welcome is disabled).
    assistant_has_responded: bool = False

    def open_client_text(self) -> None:
        """Allow client user-text items."""
        self.assistant_has_responded = True

    def begin_user_item(self) -> str:
        """Allocate a user item id for the current turn (idempotent per turn)."""
        if not self.user_item_id:
            self.user_item_id = new_item_id()
        return self.user_item_id

    def add_input_audio(self, byte_count: int, sample_rate: int) -> None:
        """Advance the received PCM16 mono clock."""
        if byte_count <= 0:
            return
        self.input_sample_rate = max(int(sample_rate), 1)
        self.input_audio_samples += byte_count // 2

    def begin_user_turn(self) -> tuple[str, int]:
        """Start one VAD turn and return ``(item_id, audio_start_ms)``."""
        if self.user_turn_stopped:
            self.clear_user_item()
        item_id = self.begin_user_item()
        if self.user_turn_start_sample is None:
            self.user_turn_start_sample = self.input_audio_samples
            self.user_turn_end_sample = None
            self.user_turn_stopped = False
            self.user_item_announced = False
            self.pending_user_transcript = None
        return item_id, self._input_ms(self.user_turn_start_sample)

    def stop_user_turn(self) -> tuple[str, int]:
        """Stop the active VAD turn and return ``(item_id, audio_end_ms)``."""
        item_id = self.begin_user_item()
        self.user_turn_end_sample = self.input_audio_samples
        self.user_turn_stopped = True
        return item_id, self._input_ms(self.user_turn_end_sample)

    def set_user_transcript(self, transcript: str) -> str:
        """Buffer a final ASR transcript until the VAD stop events are emitted."""
        item_id = self.begin_user_item()
        self.pending_user_transcript = transcript
        return item_id

    def announce_user_item(self) -> bool:
        """Mark the current user item announced; return whether it was new."""
        if self.user_item_announced:
            return False
        self.user_item_announced = True
        return True

    def ready_user_transcript(self) -> tuple[str, str] | None:
        """Return the buffered final transcript after VAD stop, if ready."""
        if not self.user_turn_stopped or self.pending_user_transcript is None:
            return None
        return self.begin_user_item(), self.pending_user_transcript

    def _input_ms(self, samples: int) -> int:
        return int(samples * 1000 / max(self.input_sample_rate, 1))

    def clear_user_item(self) -> None:
        """Clear the active user item after a final transcript."""
        self.user_item_id = None
        self.user_turn_start_sample = None
        self.user_turn_end_sample = None
        self.user_turn_stopped = False
        self.user_item_announced = False
        self.pending_user_transcript = None

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
        self.response_requested = False
        self.response_generation += 1
        self.response_id = new_response_id()
        self.response_status = "in_progress"
        self.assistant_item_id = new_item_id()
        self.assistant_transcript = ""
        self.output_item_announced = False
        self.content_part_announced = False
        self.transcript_done_emitted = False
        self.audio_done_emitted = False
        return self.response_id, True

    def request_response(self) -> bool:
        """Reserve a client-triggered response before pipeline output arrives."""
        if self.response_requested or self.response_status == "in_progress":
            return False
        self.response_requested = True
        return True

    def append_assistant_transcript(self, text: str) -> None:
        """Accumulate bot speech transcript for the active response."""
        if not text:
            return
        self.begin_response()
        self.assistant_transcript += text
        if self.assistant_item_id:
            self.item_transcripts[self.assistant_item_id] = self.assistant_transcript

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
        )
        self.response_status = status
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
        self.response_requested = False
        self.assistant_item_id = None
        self.assistant_transcript = ""
        self.output_item_announced = False
        self.content_part_announced = False
        self.transcript_done_emitted = False
        self.audio_done_emitted = False
