# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One audio stream of a session on the wall clock, such as the audio that ASR heard or that TTS produced."""

import time
import wave
from pathlib import Path
from typing import NamedTuple

GAP_SECS = 0.2


class Recording(NamedTuple):
    """PCM16 audio and its format."""

    pcm: bytes
    sample_rate: int
    num_channels: int


def write_wav(path: Path, recording: Recording) -> Path:
    """Write ``recording`` to a WAV file at ``path``."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(recording.num_channels)
        wf.setsampwidth(2)
        wf.setframerate(recording.sample_rate)
        wf.writeframes(recording.pcm)
    return path


class SessionAudio:
    """PCM16 audio of one stream for a whole session, with silence where the stream fell behind the wall clock.

    A frame goes at the end of the audio so far, after silence when that end is more than ``gap_secs`` behind the
    time the frame arrived. Streams started together share one timeline, so the ASR and TTS files of a session line
    up, even though TTS produces audio faster than real time. The first frame fixes the format, and frames in another
    format are skipped.
    """

    def __init__(self, gap_secs: float = GAP_SECS):
        """Hold no audio until :meth:`start` runs."""
        self._gap_secs = gap_secs
        self._pcm = bytearray()
        self._format: tuple[int, int] | None = None
        self._started_at: float | None = None

    def start(self) -> None:
        """Start the timeline now."""
        self._started_at = time.monotonic()

    def add(self, pcm: bytes, sample_rate: int, num_channels: int) -> None:
        """Append a frame that arrived now."""
        if self._started_at is None or not pcm:
            return
        if self._format is None:
            self._format = (sample_rate, num_channels)
        elif self._format != (sample_rate, num_channels):
            return
        self._pad(time.monotonic() - len(pcm) / self._bytes_per_sec())
        self._pcm.extend(pcm)

    def finish(self) -> Recording | None:
        """Pad to now, stop, and return the session's audio once; return None when nothing was recorded."""
        if self._started_at is None:
            return None
        self._pad(time.monotonic())
        self._started_at = None
        if self._format is None:
            return None
        recording = Recording(bytes(self._pcm), *self._format)
        self._pcm = bytearray()
        return recording

    def _bytes_per_sec(self) -> int:
        sample_rate, num_channels = self._format
        return sample_rate * num_channels * 2

    def _pad(self, until: float) -> None:
        if self._format is None:
            return
        gap = until - self._started_at - len(self._pcm) / self._bytes_per_sec()
        if gap > self._gap_secs:
            frame_bytes = self._format[1] * 2
            self._pcm.extend(bytes(round(gap * self._bytes_per_sec()) // frame_bytes * frame_bytes))
