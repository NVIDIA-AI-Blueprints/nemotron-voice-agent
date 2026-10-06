# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client audio in: decode -> audio clock -> resample -> VAD frames -> segmenter.

Synchronous and cheap per append (a 20 ms tau2 append is one or two VAD frames),
so it runs inline in the session's reader. In manual mode (``turn_detection:
null``) there is no VAD: audio accumulates until ``input_audio_buffer.commit``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from examples.frontend_backend_verdict.voice.audio.formats import AudioFormat
from examples.frontend_backend_verdict.voice.audio.resample import StreamResampler
from examples.frontend_backend_verdict.voice.clock import AudioClock
from examples.frontend_backend_verdict.voice.engine.segmenter import (
    SegmenterParams,
    SegmentEvent,
    SpeechAudio,
    SpeechEnd,
    SpeechStart,
    UtteranceSegmenter,
)
from examples.frontend_backend_verdict.voice.speech.ports import VoiceActivityDetector
from examples.frontend_backend_verdict.voice.wire.session_view import TurnDetectionSettings


@dataclass(frozen=True, slots=True)
class FeedResult:
    """What one append produced: the audio-clock step and any utterance events."""

    step_ms: float
    events: list[SegmentEvent] = field(default_factory=list)


class InputPath:
    """Per-session input audio pipeline."""

    def __init__(
        self,
        *,
        engine_rate: int,
        vad: VoiceActivityDetector,
        input_format: AudioFormat,
        turn_detection: TurnDetectionSettings,
        min_speech_ms: int,
    ) -> None:
        """Build the pipeline for the session's initial settings."""
        self.clock = AudioClock()
        self._engine_rate = engine_rate
        self._vad = vad
        self._min_speech_ms = min_speech_ms
        self._format = input_format
        self._resampler = StreamResampler(input_format.rate, engine_rate)
        self._turn = turn_detection
        self._segmenter = UtteranceSegmenter(self._params(turn_detection))
        self._carry = b""
        self._manual_buffer = bytearray()

    def _params(self, turn: TurnDetectionSettings) -> SegmenterParams:
        return SegmenterParams(
            rate=self._engine_rate,
            frame_samples=self._vad.frame_samples,
            threshold=turn.threshold,
            prefix_padding_ms=turn.prefix_padding_ms,
            silence_duration_ms=turn.silence_duration_ms,
            min_speech_ms=self._min_speech_ms,
        )

    @property
    def manual(self) -> bool:
        """Whether turn detection is off (client commits turns)."""
        return not self._turn.vad_enabled

    @property
    def in_speech(self) -> bool:
        """Whether a confirmed utterance is open."""
        return self._segmenter.in_speech

    def configure(self, input_format: AudioFormat, turn: TurnDetectionSettings) -> None:
        """Apply new session settings; a format change replaces the resampler."""
        if input_format != self._format:
            self._format = input_format
            self._resampler = StreamResampler(input_format.rate, self._engine_rate)
            self._carry = b""
        self._turn = turn
        self._segmenter.reconfigure(self._params(turn))

    def feed(self, payload: bytes) -> FeedResult:
        """Consume one ``input_audio_buffer.append`` payload."""
        samples = self._format.samples_in(payload)
        step = self.clock.advance(samples, self._format.rate)
        pcm = self._resampler.process(self._format.decode(payload))
        if self.manual:
            self._manual_buffer.extend(pcm)
            return FeedResult(step)
        data = self._carry + pcm
        frame_bytes = self._vad.frame_samples * 2
        events: list[SegmentEvent] = []
        offset = 0
        while len(data) - offset >= frame_bytes:
            frame = data[offset : offset + frame_bytes]
            offset += frame_bytes
            events.extend(self._segmenter.process(frame, self._vad.speech_probability(frame)))
        self._carry = data[offset:]
        return FeedResult(step, _coalesce(events))

    def commit(self) -> bytes | list[SegmentEvent]:
        """Manual mode: return the buffered utterance audio. VAD mode: flush an open utterance."""
        if self.manual:
            audio = bytes(self._manual_buffer)
            self._manual_buffer.clear()
            return audio
        return self._segmenter.flush()

    def clear(self) -> None:
        """Drop unfinished input (``input_audio_buffer.clear``)."""
        self._manual_buffer.clear()
        self._carry = b""
        self._segmenter.clear()

    @property
    def buffered_ms(self) -> float:
        """Manual mode: audio waiting for a commit."""
        return len(self._manual_buffer) / 2 * 1000.0 / self._engine_rate


def _coalesce(events: list[SegmentEvent]) -> list[SegmentEvent]:
    """Merge consecutive :class:`SpeechAudio` events (fewer ASR pushes per append)."""
    merged: list[SegmentEvent] = []
    for item in events:
        if isinstance(item, SpeechAudio) and merged and isinstance(merged[-1], SpeechAudio):
            merged[-1] = SpeechAudio(merged[-1].audio + item.audio)
        else:
            merged.append(item)
    return merged


__all__ = ["FeedResult", "InputPath", "SpeechAudio", "SpeechEnd", "SpeechStart"]
