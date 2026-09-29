# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Realtime-only ASR turn ordering and terminal-aware stop strategies."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Protocol

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMAssistantAggregatorParams,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)

from examples.shared.frames import USER_TRANSCRIPT_TURN_FRAME_ID_METADATA
from realtime.audio import MAX_PENDING_INPUT_BYTES
from realtime.frames import (
    RealtimeASRTurnEndedFrame,
    RealtimeASRTurnReleaseFrame,
    RealtimeInputTranscriptionErrorFrame,
    RealtimeManualUserStartedSpeakingFrame,
    RealtimeManualUserStoppedSpeakingFrame,
)
from realtime.vad import (
    RealtimeVADUserStartedSpeakingFrame,
    RealtimeVADUserStoppedSpeakingFrame,
)


class _ManualResponseGate(Protocol):
    def fail_commit(self) -> None: ...


def _transcript_owner(frame: TranscriptionFrame) -> int | None:
    value = frame.metadata.get(USER_TRANSCRIPT_TURN_FRAME_ID_METADATA)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


class _TaskReentrantLock:
    """Serialize independent tasks while allowing synchronous Pipecat callbacks."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task | None = None
        self._depth = 0

    async def __aenter__(self) -> _TaskReentrantLock:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("Realtime frame processing requires an asyncio task")
        if self._owner is task:
            self._depth += 1
            return self
        await self._lock.acquire()
        self._owner = task
        self._depth = 1
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        task = asyncio.current_task()
        if task is None or self._owner is not task or self._depth <= 0:
            raise RuntimeError("Realtime frame processing lock ownership was lost")
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
            self._lock.release()


class RealtimeLLMContextAggregatorPair:
    """Build an ordered user aggregator for cascaded Realtime input."""

    def __init__(
        self,
        context: LLMContext,
        *,
        user_params: LLMUserAggregatorParams | None = None,
        assistant_params: LLMAssistantAggregatorParams | None = None,
        add_tool_change_messages: bool | None = None,
        realtime_service_mode: bool | None = None,
    ) -> None:
        """Preserve ASR audio/boundary order at Pipecat's user-turn state."""
        user_params = user_params or LLMUserAggregatorParams()
        assistant_params = assistant_params or LLMAssistantAggregatorParams()
        if add_tool_change_messages is not None:
            user_params.add_tool_change_messages = add_tool_change_messages
            assistant_params.add_tool_change_messages = add_tool_change_messages
        self._user = _RealtimeOrderedLLMUserAggregator(
            context,
            params=user_params,
            _realtime_service_mode=realtime_service_mode,
            enable_direct_mode=True,
        )
        self._assistant = LLMAssistantAggregator(
            context,
            params=assistant_params,
            _realtime_service_mode=realtime_service_mode,
            _paired_user_aggregator=self._user,
        )

    def user(self) -> LLMUserAggregator:
        """Return the user context aggregator."""
        return self._user

    def assistant(self) -> LLMAssistantAggregator:
        """Return the assistant context aggregator."""
        return self._assistant

    def __iter__(self):
        """Yield the user and assistant aggregators."""
        return iter((self._user, self._assistant))


class _RealtimeOrderedLLMUserAggregator(LLMUserAggregator):
    """Serialize the direct Realtime ASR input and result writers."""

    def __init__(self, *args, **kwargs) -> None:
        """Create a direct processor with one ordered state transition lane."""
        super().__init__(*args, **kwargs)
        self._realtime_process_lock = _TaskReentrantLock()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Apply concurrent inputs serially while permitting callback re-entry."""
        async with self._realtime_process_lock:
            await super().process_frame(frame, direction)


class RealtimeASRInputSequencer(FrameProcessor):
    """Serialize native ASR turns without reordering their downstream output.

    A stopped native stream may still be draining its final result. Subsequent
    input is retained before the ASR service until that stream publishes its
    typed terminal upstream. This keeps Pipecat turn state and NVIDIA request
    state on the same immutable start/stop owner.
    """

    def __init__(self, *, max_buffered_frames: int = 4096, **kwargs) -> None:
        """Create a bounded per-connection input sequencer."""
        if isinstance(max_buffered_frames, bool) or not isinstance(max_buffered_frames, int):
            raise TypeError("max_buffered_frames must be an integer")
        if max_buffered_frames <= 0:
            raise ValueError("max_buffered_frames must be positive")
        # Input audio and typed boundaries are system frames, while ASR
        # terminal/release signals are control frames. Direct mode plus the
        # lock below preserves one cross-direction order instead of allowing
        # Pipecat's separate priority queues to move later audio ahead.
        kwargs["enable_direct_mode"] = True
        super().__init__(**kwargs)
        self._realtime_process_lock = _TaskReentrantLock()
        self._max_buffered_frames = max_buffered_frames
        self._active_start_frame_id: int | None = None
        self._active_boundary_kind: str | None = None
        self._pending_terminal: tuple[int, int] | None = None
        self._received_terminal: RealtimeASRTurnEndedFrame | None = None
        self._release_received = False
        self._buffer: deque[tuple[Frame, FrameDirection]] = deque()
        self._buffered_audio_bytes = 0
        self._failed = False
        self._audio_cursor = 0
        self._sample_rate = 16_000

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Order downstream input and consume upstream ASR terminals."""
        async with self._realtime_process_lock:
            await self._process_frame_ordered(frame, direction)

    async def _process_frame_ordered(self, frame: Frame, direction: FrameDirection) -> None:
        """Apply one sequencer transition while retaining callback re-entry."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._reset()
            self._sample_rate = frame.audio_in_sample_rate
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, (EndFrame, CancelFrame)):
            self._reset()
            await self.push_frame(frame, direction)
            return
        if direction == FrameDirection.UPSTREAM and isinstance(frame, RealtimeASRTurnEndedFrame):
            await self._accept_terminal(frame)
            return
        if direction == FrameDirection.UPSTREAM and isinstance(frame, RealtimeASRTurnReleaseFrame):
            await self._accept_release(frame)
            return
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, InterruptionFrame):
            await self.push_frame(frame, direction)
            return
        if self._failed:
            return
        if not await self._accept_audio_clock(frame):
            return
        if self._pending_terminal is not None:
            await self._buffer_frame(frame, direction)
            return
        await self._forward_input(frame, direction)

    async def _accept_audio_clock(self, frame: Frame) -> bool:
        """Validate each boundary against the lossless accepted-audio clock."""
        if isinstance(frame, InputAudioRawFrame):
            if frame.sample_rate != self._sample_rate or frame.num_channels != 1 or len(frame.audio) % 2:
                await self._fail(
                    "asr_audio_invalid",
                    "Realtime ASR input must be complete mono PCM16 at the pipeline sample rate",
                )
                return False
            self._audio_cursor += len(frame.audio) // 2
            return True
        if isinstance(frame, (RealtimeManualUserStartedSpeakingFrame, RealtimeManualUserStoppedSpeakingFrame)):
            valid = frame.sample_rate == self._sample_rate and frame.audio_sample_cursor == self._audio_cursor
        elif isinstance(frame, RealtimeVADUserStartedSpeakingFrame):
            valid = frame.sample_rate == self._sample_rate and frame.audio_start_sample <= self._audio_cursor
        elif isinstance(frame, RealtimeVADUserStoppedSpeakingFrame):
            valid = frame.sample_rate == self._sample_rate and frame.audio_stop_sample == self._audio_cursor
        else:
            return True
        if valid:
            return True
        await self._fail(
            "asr_turn_boundary_invalid",
            "Realtime audio boundary does not match the accepted pipeline audio clock",
        )
        return False

    async def _forward_input(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, (RealtimeManualUserStartedSpeakingFrame, RealtimeVADUserStartedSpeakingFrame)):
            kind = "manual" if isinstance(frame, RealtimeManualUserStartedSpeakingFrame) else "vad"
            if self._active_start_frame_id is not None:
                await self._fail(
                    "asr_turn_boundary_invalid",
                    "Realtime ASR received a new audio start before the active input segment stopped",
                )
                return
            self._active_start_frame_id = frame.id
            self._active_boundary_kind = kind
        elif isinstance(frame, (RealtimeManualUserStoppedSpeakingFrame, RealtimeVADUserStoppedSpeakingFrame)):
            kind = "manual" if isinstance(frame, RealtimeManualUserStoppedSpeakingFrame) else "vad"
            if self._active_start_frame_id is None or self._active_boundary_kind != kind:
                await self._fail(
                    "asr_turn_boundary_invalid",
                    "Realtime ASR received an audio stop without a matching input start",
                )
                return
            self._pending_terminal = (self._active_start_frame_id, frame.id)
            self._received_terminal = None
            self._release_received = False
            self._active_start_frame_id = None
            self._active_boundary_kind = None
        await self.push_frame(frame, direction)

    async def _buffer_frame(self, frame: Frame, direction: FrameDirection) -> None:
        audio_bytes = len(frame.audio) if isinstance(frame, InputAudioRawFrame) else 0
        if len(self._buffer) >= self._max_buffered_frames or (
            self._buffered_audio_bytes + audio_bytes > MAX_PENDING_INPUT_BYTES
        ):
            await self._fail(
                "input_buffer_overflow",
                "Realtime audio waiting for the preceding ASR turn exceeded the pipeline buffer limit",
            )
            return
        self._buffer.append((frame, direction))
        self._buffered_audio_bytes += audio_bytes

    async def _accept_terminal(self, terminal: RealtimeASRTurnEndedFrame) -> None:
        expected = self._pending_terminal
        actual = (terminal.turn_start_frame_id, terminal.turn_stop_frame_id)
        if expected is None or actual != expected or self._received_terminal is not None:
            await self._fail(
                "asr_turn_owner_mismatch",
                "NVIDIA ASR returned a terminal for a different Realtime audio turn",
            )
            return
        self._received_terminal = terminal
        await self._release_if_ready()

    async def _accept_release(self, release: RealtimeASRTurnReleaseFrame) -> None:
        expected = self._pending_terminal
        actual = (release.turn_start_frame_id, release.turn_stop_frame_id)
        if expected is None or actual != expected or self._release_received:
            await self._fail(
                "asr_turn_owner_mismatch",
                "Pipecat released a different Realtime ASR turn",
            )
            return
        self._release_received = True
        await self._release_if_ready()

    async def _release_if_ready(self) -> None:
        terminal = self._received_terminal
        if terminal is None or not self._release_received:
            return
        if terminal.status == "cancelled" or (terminal.status == "failed" and terminal.fatal):
            self._buffer.clear()
            self._buffered_audio_bytes = 0
            self._pending_terminal = None
            self._received_terminal = None
            self._release_received = False
            self._active_start_frame_id = None
            self._active_boundary_kind = None
            self._failed = True
            return
        pending = tuple(self._buffer)
        self._buffer.clear()
        self._buffered_audio_bytes = 0
        self._pending_terminal = None
        self._received_terminal = None
        self._release_received = False
        for index, (frame, direction) in enumerate(pending):
            if self._failed:
                return
            if self._pending_terminal is not None:
                for retained_frame, retained_direction in pending[index:]:
                    await self._buffer_frame(retained_frame, retained_direction)
                return
            await self._forward_input(frame, direction)

    async def _fail(self, code: str, message: str) -> None:
        if self._failed:
            return
        self._failed = True
        self._buffer.clear()
        self._buffered_audio_bytes = 0
        self._pending_terminal = None
        self._received_terminal = None
        self._release_received = False
        self._active_start_frame_id = None
        self._active_boundary_kind = None
        await self.push_frame(RealtimeInputTranscriptionErrorFrame(code=code, message=message))

    def _reset(self) -> None:
        self._active_start_frame_id = None
        self._active_boundary_kind = None
        self._pending_terminal = None
        self._received_terminal = None
        self._release_received = False
        self._buffer.clear()
        self._buffered_audio_bytes = 0
        self._failed = False
        self._audio_cursor = 0


class RealtimeServerVADTurnStopStrategy(BaseUserTurnStopStrategy):
    """End a server-VAD turn only after its exact ASR producer terminal."""

    def __init__(self, **kwargs) -> None:
        """Create empty immutable-owner state for one server-VAD turn."""
        super().__init__(**kwargs)
        self._start_frame_id: int | None = None
        self._stop_frame_id: int | None = None
        self._text_observed = False
        self._triggered = False

    async def handle_user_turn_started(self) -> None:
        """Reset terminal state when Pipecat opens a user turn."""
        self._clear()

    async def handle_user_turn_stopped(self) -> None:
        """Release terminal state after Pipecat closes the user turn."""
        self._clear()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Collect one typed start, stop, transcript, and native terminal."""
        if isinstance(frame, RealtimeVADUserStartedSpeakingFrame):
            self._start_frame_id = frame.id
            self._stop_frame_id = None
            self._text_observed = False
            self._triggered = False
        elif isinstance(frame, RealtimeVADUserStoppedSpeakingFrame):
            self._stop_frame_id = frame.id
        elif isinstance(frame, TranscriptionFrame) and _transcript_owner(frame) == self._start_frame_id:
            self._text_observed |= bool(frame.text and frame.text.strip())
        elif isinstance(frame, RealtimeASRTurnEndedFrame):
            await self._handle_terminal(frame)
        return ProcessFrameResult.CONTINUE

    async def _handle_terminal(self, frame: RealtimeASRTurnEndedFrame) -> None:
        if self._triggered or self._start_frame_id is None or self._stop_frame_id is None:
            return
        if (frame.turn_start_frame_id, frame.turn_stop_frame_id) != (
            self._start_frame_id,
            self._stop_frame_id,
        ):
            return
        self._triggered = True
        release = RealtimeASRTurnReleaseFrame(
            turn_start_frame_id=frame.turn_start_frame_id,
            turn_stop_frame_id=frame.turn_stop_frame_id,
        )
        if frame.status in {"completed", "empty"} and self._text_observed:
            await self.trigger_user_turn_stopped()
        else:
            await self.trigger_user_turn_finalized()
        await self.push_frame(release, FrameDirection.UPSTREAM)

    def _clear(self) -> None:
        self._start_frame_id = None
        self._stop_frame_id = None
        self._text_observed = False
        self._triggered = False


class RealtimeSemanticTurnStopStrategy(TurnAnalyzerUserTurnStopStrategy):
    """Gate Smart Turn completion on the matching native ASR terminal."""

    def __init__(self, **kwargs) -> None:
        """Create Smart Turn state plus one acoustic-segment owner."""
        super().__init__(**kwargs)
        self._segment_start_frame_id: int | None = None
        self._segment_stop_frame_id: int | None = None
        self._segment_terminal = False
        self._segment_released = False
        self._triggered = False

    async def handle_user_turn_started(self) -> None:
        """Arm Smart Turn and clear any prior acoustic-segment owner."""
        await super().handle_user_turn_started()
        self._clear_segment()

    async def handle_user_turn_stopped(self) -> None:
        """Clear Smart Turn and its terminal owner after finalization."""
        await super().handle_user_turn_stopped()
        self._clear_segment()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Run Smart Turn while correlating the current ASR segment terminal."""
        if isinstance(frame, RealtimeVADUserStartedSpeakingFrame):
            self._segment_start_frame_id = frame.id
            self._segment_stop_frame_id = None
            self._segment_terminal = False
            self._segment_released = False
        elif isinstance(frame, RealtimeVADUserStoppedSpeakingFrame):
            self._segment_stop_frame_id = frame.id
        result = await super().process_frame(frame)
        if isinstance(frame, RealtimeASRTurnEndedFrame):
            await self._handle_terminal(frame)
        return result

    async def _handle_terminal(self, frame: RealtimeASRTurnEndedFrame) -> None:
        if self._segment_released or (
            frame.turn_start_frame_id,
            frame.turn_stop_frame_id,
        ) != (self._segment_start_frame_id, self._segment_stop_frame_id):
            return
        release = RealtimeASRTurnReleaseFrame(
            turn_start_frame_id=frame.turn_start_frame_id,
            turn_stop_frame_id=frame.turn_stop_frame_id,
        )
        self._segment_released = True
        if frame.status in {"completed", "empty"}:
            self._segment_terminal = True
            await self._maybe_trigger_user_turn_stopped()
        else:
            self._triggered = True
            await self.trigger_user_turn_finalized()
        await self.push_frame(release, FrameDirection.UPSTREAM)

    async def _maybe_trigger_user_turn_stopped(self) -> None:
        if self._triggered or not self._turn_complete or not self._segment_terminal:
            return
        self._triggered = True
        if self._timeout_task:
            await self.task_manager.cancel_task(self._timeout_task)
            self._timeout_task = None
        if self._text:
            await self.trigger_user_turn_stopped()
        else:
            await self.trigger_user_turn_finalized()

    def _clear_segment(self) -> None:
        self._segment_start_frame_id = None
        self._segment_stop_frame_id = None
        self._segment_terminal = False
        self._segment_released = False
        self._triggered = False


class RealtimeManualTurnStopStrategy(BaseUserTurnStopStrategy):
    """Finalize one client-committed turn after its exact ASR terminal."""

    def __init__(self, *, response_gate: _ManualResponseGate) -> None:
        """Bind the strategy to its manual response coordinator."""
        super().__init__(enable_user_speaking_frames=False)
        self._response_gate = response_gate
        self._start_frame_id: int | None = None
        self._stop_frame_id: int | None = None
        self._finalized_turn_frame_id: int | None = None
        self._text_observed = False
        self._triggered = False

    @property
    def turn_frame_id(self) -> int | None:
        """Return the immutable owner of the turn-finalized callback."""
        return self._finalized_turn_frame_id

    async def handle_user_turn_started(self) -> None:
        """Reset terminal state for a newly committed manual turn."""
        self._clear()

    async def handle_user_turn_stopped(self) -> None:
        """Release active state while retaining the finalized callback owner."""
        self._clear_active()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Wait for the matching ASR terminal instead of a segment-final bit."""
        if isinstance(frame, RealtimeManualUserStartedSpeakingFrame):
            self._start_frame_id = frame.id
            self._stop_frame_id = None
            self._text_observed = False
            self._triggered = False
        elif isinstance(frame, RealtimeManualUserStoppedSpeakingFrame):
            self._stop_frame_id = frame.id
        elif isinstance(frame, TranscriptionFrame) and _transcript_owner(frame) == self._start_frame_id:
            self._text_observed |= bool(frame.text and frame.text.strip())
        elif isinstance(frame, RealtimeASRTurnEndedFrame):
            await self._handle_terminal(frame)
        return ProcessFrameResult.CONTINUE

    async def _handle_terminal(self, frame: RealtimeASRTurnEndedFrame) -> None:
        if self._triggered or self._start_frame_id is None or self._stop_frame_id is None:
            return
        if (frame.turn_start_frame_id, frame.turn_stop_frame_id) != (
            self._start_frame_id,
            self._stop_frame_id,
        ):
            return
        self._triggered = True
        self._finalized_turn_frame_id = frame.turn_stop_frame_id
        release = RealtimeASRTurnReleaseFrame(
            turn_start_frame_id=frame.turn_start_frame_id,
            turn_stop_frame_id=frame.turn_stop_frame_id,
        )
        if frame.status not in {"completed", "empty"} or not self._text_observed:
            self._response_gate.fail_commit()
            await self.trigger_user_turn_finalized()
        else:
            await self.trigger_user_turn_stopped()
        await self.push_frame(release, FrameDirection.UPSTREAM)

    def _clear(self) -> None:
        self._clear_active()
        self._finalized_turn_frame_id = None

    def _clear_active(self) -> None:
        self._start_frame_id = None
        self._stop_frame_id = None
        self._text_observed = False
        self._triggered = False
