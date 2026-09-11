# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Realtime-only ownership for NVIDIA streaming ASR transcript frames."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict, deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.stt import NvidiaSTTService

from examples.shared.frames import USER_TRANSCRIPT_TURN_FRAME_ID_METADATA
from realtime.frames import (
    RealtimeInputTranscriptionErrorFrame,
    RealtimeManualUserStartedSpeakingFrame,
    RealtimeManualUserStoppedSpeakingFrame,
)


class RealtimeASROwnershipError(ValueError):
    """Reject an ASR result that cannot be assigned to one audio span."""

    def __init__(self, *, code: str, message: str) -> None:
        """Store a stable public error code and message."""
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(slots=True)
class _AudioTurnSpan:
    token: int
    start_sample: int
    stop_sample: int | None = None


class RealtimeASRTurnOwnership:
    """Resolve cumulative provider audio coordinates to immutable turn tokens."""

    def __init__(self, *, max_turns: int = 4096) -> None:
        """Create bounded ownership state for one provider stream."""
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns <= 0:
            raise ValueError("max_turns must be a positive integer")
        self._max_turns = max_turns
        self._submitted_samples = 0
        self._active_token: int | None = None
        self._turns: OrderedDict[int, _AudioTurnSpan] = OrderedDict()
        self._last_processed_sample: int | None = None

    @property
    def submitted_samples(self) -> int:
        """Return the number of samples submitted on the current ASR stream."""
        return self._submitted_samples

    def submit_samples(self, sample_count: int) -> None:
        """Advance the exact current-stream input coordinate."""
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 0:
            raise ValueError("sample_count must be a non-negative integer")
        self._submitted_samples += sample_count

    def start_turn(self, token: int, *, start_sample: int) -> None:
        """Open a turn at an exact audio-stream sample boundary."""
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise RealtimeASROwnershipError(
                code="asr_transcription_owner_invalid",
                message="The ASR turn token must be a non-negative frame identifier",
            )
        if self._active_token is not None:
            raise RealtimeASROwnershipError(
                code="asr_transcription_boundary_invalid",
                message="The ASR stream started a new audio turn before stopping the previous turn",
            )
        if token in self._turns:
            raise RealtimeASROwnershipError(
                code="asr_transcription_owner_invalid",
                message="The ASR stream reused an audio turn token",
            )
        if len(self._turns) >= self._max_turns:
            raise RealtimeASROwnershipError(
                code="asr_transcription_ownership_overflow",
                message="The ASR stream exceeded its bounded turn ownership capacity",
            )
        if isinstance(start_sample, bool) or not isinstance(start_sample, int):
            raise RealtimeASROwnershipError(
                code="asr_transcription_boundary_invalid",
                message="The ASR stream reported a non-integer audio turn boundary",
            )
        prior_span = next(reversed(self._turns.values()), None)
        minimum_start = prior_span.stop_sample if prior_span is not None else 0
        if (
            start_sample < 0
            or start_sample > self._submitted_samples
            or minimum_start is None
            or start_sample < minimum_start
        ):
            raise RealtimeASROwnershipError(
                code="asr_transcription_boundary_invalid",
                message="The ASR stream reported an invalid audio turn start boundary",
            )
        self._turns[token] = _AudioTurnSpan(token=token, start_sample=start_sample)
        self._active_token = token

    def stop_turn(self) -> None:
        """Close the active turn after all audio preceding its raw stop boundary."""
        if self._active_token is None:
            raise RealtimeASROwnershipError(
                code="asr_transcription_boundary_invalid",
                message="The ASR stream stopped an audio turn without a matching start",
            )
        span = self._turns[self._active_token]
        span.stop_sample = self._submitted_samples
        self._active_token = None

    def processed_sample(self, audio_processed: object, *, sample_rate: int) -> int:
        """Validate the provider stream cursor without using it as transcript ownership."""
        if (
            isinstance(audio_processed, bool)
            or not isinstance(audio_processed, int | float)
            or not math.isfinite(float(audio_processed))
            or audio_processed < 0
        ):
            raise RealtimeASROwnershipError(
                code="asr_transcription_coordinate_invalid",
                message="NVIDIA ASR returned an invalid audio_processed coordinate",
            )
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise RealtimeASROwnershipError(
                code="asr_transcription_coordinate_invalid",
                message="NVIDIA ASR returned a transcript before its stream sample rate was established",
            )
        sample = round(float(audio_processed) * sample_rate)
        if sample > self._submitted_samples:
            raise RealtimeASROwnershipError(
                code="asr_transcription_coordinate_out_of_range",
                message="NVIDIA ASR returned audio_processed beyond submitted stream audio",
            )
        if self._last_processed_sample is not None and sample < self._last_processed_sample:
            raise RealtimeASROwnershipError(
                code="asr_transcription_coordinate_regressed",
                message="NVIDIA ASR returned a regressing audio_processed coordinate",
            )
        self._last_processed_sample = sample
        return sample

    def owner_for_range(self, start_sample: int, end_sample: int) -> int | None:
        """Return the sole turn intersecting a transcript's word interval."""
        owners: list[int] = []
        for span in self._turns.values():
            if end_sample > span.start_sample and (span.stop_sample is None or start_sample < span.stop_sample):
                owners.append(span.token)
        return owners[0] if len(owners) == 1 else None

    @property
    def active_start_sample(self) -> int | None:
        """Return the active turn's immutable stream boundary, if any."""
        if self._active_token is None:
            return None
        return self._turns[self._active_token].start_sample

    @property
    def prior_stop_sample(self) -> int:
        """Return the most recent closed turn boundary on this stream."""
        prior = next(reversed(self._turns.values()), None)
        return prior.stop_sample if prior is not None and prior.stop_sample is not None else 0

    def reset_stream(self) -> None:
        """Discard coordinates that are invalid after a provider stream restart."""
        self._submitted_samples = 0
        self._active_token = None
        self._turns.clear()
        self._last_processed_sample = None


@dataclass(slots=True)
class _PendingTranscript:
    frame: InterimTranscriptionFrame | TranscriptionFrame
    direction: FrameDirection
    start_sample: int
    end_sample: int


class RealtimeNvidiaSTTService(NvidiaSTTService):
    """NVIDIA STT with pre-pipeline Realtime transcript ownership."""

    def __init__(self, *args, max_pending_transcripts: int = 128, **kwargs) -> None:
        """Create a stock NVIDIA STT service with Realtime-only owner state."""
        if (
            isinstance(max_pending_transcripts, bool)
            or not isinstance(max_pending_transcripts, int)
            or max_pending_transcripts <= 0
        ):
            raise ValueError("max_pending_transcripts must be a positive integer")
        super().__init__(*args, **kwargs)
        self._realtime_ownership = RealtimeASRTurnOwnership()
        self._realtime_ownership_lock = asyncio.Lock()
        self._realtime_pending_transcripts: deque[_PendingTranscript] = deque()
        self._realtime_max_pending_transcripts = max_pending_transcripts
        self._realtime_boundary_frame_ids: deque[int] = deque(maxlen=4096)
        self._realtime_boundary_frame_id_set: set[int] = set()
        self._realtime_ownership_faulted = False
        self._realtime_ownership_error: RealtimeASROwnershipError | None = None
        self._realtime_manual_mode = False

    def _create_recognition_config(self):
        """Require provider word offsets used for exact Realtime ownership."""
        config = super()._create_recognition_config()
        config.config.enable_word_time_offsets = True
        return config

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Record audio in the same order in which the base service submits it."""
        emitted: list[Frame | None] = []
        async with self._realtime_ownership_lock:
            iterator = self._audio_iterator
            async for frame in super().run_stt(audio):
                emitted.append(frame)
            self._count_submitted_audio(audio, iterator=iterator)
        for frame in emitted:
            yield frame

    async def _send_keepalive(self, silence: bytes) -> None:
        """Keep provider and ownership coordinates aligned across idle silence."""
        async with self._realtime_ownership_lock:
            iterator = self._audio_iterator
            await super()._send_keepalive(silence)
            self._count_submitted_audio(silence, iterator=iterator)

    def _count_submitted_audio(self, audio: bytes, *, iterator: object) -> None:
        """Advance coordinates only when the base call used the captured stream."""
        if iterator is None or iterator is not self._audio_iterator or getattr(iterator, "closed", True):
            return
        channels = max(1, int(self._audio_channel_count))
        self._realtime_ownership.submit_samples(len(audio) // (2 * channels))

    async def _do_reconnect(self) -> None:
        """Fail closed because a restarted Riva stream reuses coordinates from zero."""
        if self._realtime_manual_mode:
            await super()._do_reconnect()
            return
        async with self._realtime_ownership_lock:
            await self._disable_ownership(
                RealtimeASROwnershipError(
                    code="asr_transcription_stream_restarted",
                    message="NVIDIA ASR restarted its stream, invalidating transcript ownership coordinates",
                )
            )
            self._realtime_ownership.reset_stream()
        await super()._do_reconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Capture raw Realtime audio boundaries before forwarding them."""
        if isinstance(frame, (RealtimeManualUserStartedSpeakingFrame, RealtimeManualUserStoppedSpeakingFrame)):
            self._realtime_manual_mode = True
            await super().process_frame(frame, direction)
            return
        starts_turn = isinstance(frame, VADUserStartedSpeakingFrame)
        stops_turn = isinstance(frame, VADUserStoppedSpeakingFrame)
        if not starts_turn and not stops_turn:
            await super().process_frame(frame, direction)
            return
        if not self._remember_boundary_frame(frame.id):
            await super().process_frame(frame, direction)
            return
        if starts_turn:
            self._realtime_manual_mode = False

        async with self._realtime_ownership_lock:
            if not self._realtime_ownership_faulted:
                try:
                    if starts_turn:
                        start_sample = self._realtime_ownership.submitted_samples
                        history_samples = self._vad_history_samples(frame)
                        prior_stop = self._realtime_ownership.prior_stop_sample
                        start_sample = max(prior_stop, start_sample - history_samples)
                        self._realtime_ownership.start_turn(
                            frame.id,
                            start_sample=start_sample,
                        )
                    else:
                        self._realtime_ownership.stop_turn()
                except RealtimeASROwnershipError as exc:
                    await self._disable_ownership(exc)
            # Forward the boundary while transcript callbacks are excluded so
            # the observer installs its token before any stamped result.
            await super().process_frame(frame, direction)
            await self._flush_pending_transcripts()

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        """Stamp every provider transcript before its first pipeline edge."""
        if not isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)):
            await super().push_frame(frame, direction)
            return
        if self._realtime_manual_mode:
            await super().push_frame(frame, direction)
            return
        async with self._realtime_ownership_lock:
            if self._realtime_ownership_faulted:
                if isinstance(frame, TranscriptionFrame) and self._realtime_ownership_error is not None:
                    await self._emit_ownership_error(self._realtime_ownership_error)
                await super().push_frame(frame, direction)
                return
            try:
                start_sample, end_sample = self._result_range(frame)
            except RealtimeASROwnershipError as exc:
                if isinstance(frame, TranscriptionFrame):
                    await self._emit_ownership_error(exc)
                await super().push_frame(frame, direction)
                return
            owner = self._realtime_ownership.owner_for_range(start_sample, end_sample)
            if owner is None:
                active_start = self._realtime_ownership.active_start_sample
                if active_start is not None and start_sample < active_start:
                    if isinstance(frame, TranscriptionFrame):
                        await self._emit_ownership_error(self._unowned_range_error())
                    await super().push_frame(frame, direction)
                    return
                if len(self._realtime_pending_transcripts) >= self._realtime_max_pending_transcripts:
                    await self._disable_ownership(
                        RealtimeASROwnershipError(
                            code="asr_transcription_ownership_overflow",
                            message="ASR transcripts arrived before a bounded Realtime audio owner was available",
                        )
                    )
                    await super().push_frame(frame, direction)
                    return
                self._realtime_pending_transcripts.append(
                    _PendingTranscript(
                        frame=frame,
                        direction=direction,
                        start_sample=start_sample,
                        end_sample=end_sample,
                    )
                )
                return
            frame.metadata[USER_TRANSCRIPT_TURN_FRAME_ID_METADATA] = owner
            await super().push_frame(frame, direction)

    def _result_range(self, frame: InterimTranscriptionFrame | TranscriptionFrame) -> tuple[int, int]:
        result = frame.result
        if result is None:
            raise RealtimeASROwnershipError(
                code="asr_transcription_coordinate_missing",
                message="NVIDIA ASR returned a transcript without its recognition result",
            )
        audio_processed = getattr(result, "audio_processed", None)
        processed_sample = self._realtime_ownership.processed_sample(
            audio_processed,
            sample_rate=self.sample_rate,
        )
        alternatives = getattr(result, "alternatives", None)
        if not alternatives:
            raise RealtimeASROwnershipError(
                code="asr_transcription_coordinate_missing",
                message="NVIDIA ASR returned a transcript without a top recognition alternative",
            )
        words = getattr(alternatives[0], "words", None)
        if not words:
            raise RealtimeASROwnershipError(
                code="asr_transcription_word_offsets_missing",
                message="NVIDIA ASR returned a transcript without required word time offsets",
            )

        first_start_ms: int | None = None
        last_start_ms: int | None = None
        last_end_ms: int | None = None
        for word in words:
            start_ms = getattr(word, "start_time", None)
            end_ms = getattr(word, "end_time", None)
            if (
                isinstance(start_ms, bool)
                or not isinstance(start_ms, int)
                or isinstance(end_ms, bool)
                or not isinstance(end_ms, int)
                or start_ms < 0
                or end_ms < start_ms
                or (last_start_ms is not None and start_ms < last_start_ms)
                or (last_end_ms is not None and end_ms < last_end_ms)
            ):
                raise RealtimeASROwnershipError(
                    code="asr_transcription_word_offsets_invalid",
                    message="NVIDIA ASR returned invalid word time offsets",
                )
            if first_start_ms is None:
                first_start_ms = start_ms
            last_start_ms = start_ms
            last_end_ms = end_ms

        assert first_start_ms is not None and last_end_ms is not None
        start_sample = round(first_start_ms * self.sample_rate / 1000)
        end_sample = round(last_end_ms * self.sample_rate / 1000)
        if end_sample > processed_sample:
            raise RealtimeASROwnershipError(
                code="asr_transcription_word_offsets_out_of_range",
                message="NVIDIA ASR returned word time offsets beyond processed stream audio",
            )
        return start_sample, end_sample

    def _vad_history_samples(self, frame: VADUserStartedSpeakingFrame) -> int:
        start_secs = frame.start_secs
        sample_rate = self.sample_rate
        if (
            isinstance(start_secs, bool)
            or not isinstance(start_secs, int | float)
            or not math.isfinite(float(start_secs))
            or start_secs < 0
            or isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate <= 0
        ):
            raise RealtimeASROwnershipError(
                code="asr_transcription_boundary_invalid",
                message="The ASR stream reported an invalid VAD start boundary",
            )
        return round(float(start_secs) * sample_rate)

    async def _flush_pending_transcripts(self) -> None:
        if self._realtime_ownership_faulted or not self._realtime_pending_transcripts:
            return
        remaining: deque[_PendingTranscript] = deque()
        while self._realtime_pending_transcripts:
            pending = self._realtime_pending_transcripts.popleft()
            owner = self._realtime_ownership.owner_for_range(
                pending.start_sample,
                pending.end_sample,
            )
            if owner is None:
                active_start = self._realtime_ownership.active_start_sample
                if active_start is None or pending.start_sample >= active_start:
                    remaining.append(pending)
                elif isinstance(pending.frame, TranscriptionFrame):
                    await self._emit_ownership_error(self._unowned_range_error())
                    await super().push_frame(pending.frame, pending.direction)
                else:
                    await super().push_frame(pending.frame, pending.direction)
                continue
            pending.frame.metadata[USER_TRANSCRIPT_TURN_FRAME_ID_METADATA] = owner
            await super().push_frame(pending.frame, pending.direction)
        self._realtime_pending_transcripts = remaining

    async def _disable_ownership(self, error: RealtimeASROwnershipError) -> None:
        if self._realtime_ownership_faulted:
            return
        self._realtime_ownership_faulted = True
        self._realtime_ownership_error = error
        pending = tuple(self._realtime_pending_transcripts)
        self._realtime_pending_transcripts.clear()
        await self._emit_ownership_error(error)
        for transcript in pending:
            await super().push_frame(transcript.frame, transcript.direction)

    async def _emit_ownership_error(self, error: RealtimeASROwnershipError) -> None:
        await super().push_frame(
            RealtimeInputTranscriptionErrorFrame(
                code=error.code,
                message=error.message,
            )
        )

    @staticmethod
    def _unowned_range_error() -> RealtimeASROwnershipError:
        return RealtimeASROwnershipError(
            code="asr_transcription_owner_missing",
            message="NVIDIA ASR returned word time offsets outside one Realtime audio turn",
        )

    def _remember_boundary_frame(self, frame_id: int) -> bool:
        if frame_id in self._realtime_boundary_frame_id_set:
            return False
        if len(self._realtime_boundary_frame_ids) == self._realtime_boundary_frame_ids.maxlen:
            expired = self._realtime_boundary_frame_ids.popleft()
            self._realtime_boundary_frame_id_set.discard(expired)
        self._realtime_boundary_frame_ids.append(frame_id)
        self._realtime_boundary_frame_id_set.add(frame_id)
        return True
