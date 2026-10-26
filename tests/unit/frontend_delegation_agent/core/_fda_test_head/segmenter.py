# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utterance segmentation over VAD frames, timed purely in samples.

A pure state machine: frames and their speech probabilities in, utterance
events out. Every time is derived from the number of samples processed, so a
stream with multi-second wall-clock gaps segments exactly like a gap-free one.

* **Onset** is the first speech frame of a run. It is only *confirmed* once the
  run reaches ``min_speech_ms`` of consecutive speech; shorter bursts are
  discarded as noise and produce no event at all.
* **Start** (:class:`SpeechStart`) carries ``prefix_padding_ms`` of audio before
  the onset plus the confirmed run, so the recognizer hears the whole word.
* **End** (:class:`SpeechEnd`) fires after ``silence_duration_ms`` of
  consecutive non-speech frames; its time is the onset of that silence.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SegmenterParams:
    """Endpointing parameters, all in milliseconds except ``threshold``."""

    rate: int
    frame_samples: int
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    min_speech_ms: int = 120

    @property
    def frame_ms(self) -> float:
        """Duration of one VAD frame."""
        return self.frame_samples * 1000.0 / self.rate


@dataclass(frozen=True, slots=True)
class SpeechStart:
    """Confirmed speech: ``start_ms`` is the onset; ``audio`` is prefix padding + the run so far."""

    start_ms: float
    audio: bytes


@dataclass(frozen=True, slots=True)
class SpeechAudio:
    """More utterance audio (speech and the trailing silence being measured)."""

    audio: bytes


@dataclass(frozen=True, slots=True)
class SpeechEnd:
    """End of turn: ``end_ms`` is the onset of the silence that ended it."""

    end_ms: float


SegmentEvent = SpeechStart | SpeechAudio | SpeechEnd

_SILENT, _PENDING, _SPEAKING = "silent", "pending", "speaking"


class UtteranceSegmenter:
    """Frame-driven endpointer on the audio clock."""

    def __init__(self, params: SegmenterParams) -> None:
        """Start silent at time zero."""
        self.params = params
        self._frames = 0
        self._state = _SILENT
        self._prefix: deque[bytes] = deque(maxlen=self._frames_for(params.prefix_padding_ms))
        self._pending: list[bytes] = []
        self._onset_frame = 0
        self._silence_frames = 0
        self._silence_start_frame = 0

    def _frames_for(self, ms: int) -> int:
        return int(round(ms / self.params.frame_ms)) if ms > 0 else 0

    def _time_ms(self, frame_index: int) -> float:
        return frame_index * self.params.frame_ms

    @property
    def now_ms(self) -> float:
        """Audio time after the last processed frame."""
        return self._time_ms(self._frames)

    @property
    def in_speech(self) -> bool:
        """Whether a confirmed utterance is open."""
        return self._state == _SPEAKING

    def reconfigure(self, params: SegmenterParams) -> None:
        """Adopt new thresholds (keeps the clock; an open utterance continues)."""
        if params.frame_samples != self.params.frame_samples or params.rate != self.params.rate:
            raise ValueError("frame size and rate cannot change on a live segmenter")
        self.params = params
        self._prefix = deque(self._prefix, maxlen=self._frames_for(params.prefix_padding_ms))

    def process(self, frame: bytes, probability: float) -> list[SegmentEvent]:
        """Consume one frame and its speech probability."""
        index = self._frames
        self._frames += 1
        speech = probability >= self.params.threshold
        min_frames = max(1, self._frames_for(self.params.min_speech_ms))

        if self._state == _SILENT:
            if not speech:
                self._push_prefix(frame)
                return []
            self._state = _PENDING
            self._onset_frame = index
            self._pending = [frame]
            return self._maybe_confirm(min_frames)

        if self._state == _PENDING:
            if speech:
                self._pending.append(frame)
                return self._maybe_confirm(min_frames)
            # Too short: noise. Nothing is emitted; the frames become prefix context.
            for pending in self._pending:
                self._push_prefix(pending)
            self._push_prefix(frame)
            self._pending = []
            self._state = _SILENT
            return []

        # _SPEAKING
        if speech:
            self._silence_frames = 0
            return [SpeechAudio(frame)]
        if self._silence_frames == 0:
            self._silence_start_frame = index
        self._silence_frames += 1
        events: list[SegmentEvent] = [SpeechAudio(frame)]
        if self._silence_frames * self.params.frame_ms >= self.params.silence_duration_ms:
            events.append(SpeechEnd(self._time_ms(self._silence_start_frame)))
            self._end()
        return events

    def flush(self) -> list[SegmentEvent]:
        """Force the end of an open utterance (manual commit); pending noise is dropped."""
        if self._state == _SPEAKING:
            end = self._silence_start_frame if self._silence_frames else self._frames
            self._end()
            return [SpeechEnd(self._time_ms(end))]
        self._pending = []
        self._state = _SILENT
        return []

    def clear(self) -> None:
        """Drop any open or pending utterance without an event."""
        self._end()

    def _maybe_confirm(self, min_frames: int) -> list[SegmentEvent]:
        if len(self._pending) < min_frames:
            return []
        audio = b"".join(self._prefix) + b"".join(self._pending)
        self._prefix.clear()
        self._pending = []
        self._state = _SPEAKING
        self._silence_frames = 0
        return [SpeechStart(self._time_ms(self._onset_frame), audio)]

    def _push_prefix(self, frame: bytes) -> None:
        if self._prefix.maxlen:
            self._prefix.append(frame)

    def _end(self) -> None:
        self._state = _SILENT
        self._pending = []
        self._silence_frames = 0
        self._prefix.clear()
