# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream resampling for PCM16 mono, driven by samples rather than wall time.

This wraps ``soxr.ResampleStream`` (the library Pipecat's stream resampler uses)
directly instead of ``realtime.audio.AudioResampler``, for two reasons that
matter here:

* Pipecat's ``SOXRStreamAudioResampler`` clears its state after 0.2 s of
  *wall-clock* inactivity. tau2 pauses its input stream for seconds of wall time,
  and turn timing in this package must depend on the audio alone.
* Output synthesis needs a ``flush`` at the end of each sentence, so the filter
  tail is delivered with the sentence and does not leak into the next item.

The stream runs in float32 and rounds to int16 itself: libsoxr dithers integer
output with run-dependent state, which made identical input produce audio that
differed by one LSB from run to run.
"""

from __future__ import annotations

import numpy as np
import soxr


class StreamResampler:
    """One-direction PCM16 resampler with explicit flush."""

    def __init__(self, in_rate: int, out_rate: int, *, quality: str = "HQ") -> None:
        """Create a stream from ``in_rate`` to ``out_rate`` Hz."""
        self.in_rate = in_rate
        self.out_rate = out_rate
        self._quality = quality
        self._stream: soxr.ResampleStream | None = None
        self._reset()

    def _reset(self) -> None:
        if self.in_rate == self.out_rate:
            self._stream = None
            return
        self._stream = soxr.ResampleStream(
            in_rate=self.in_rate, out_rate=self.out_rate, num_channels=1, dtype="float32", quality=self._quality
        )

    def process(self, pcm: bytes, *, last: bool = False) -> bytes:
        """Resample one chunk; ``last=True`` flushes the filter tail and resets the stream."""
        if self._stream is None:
            return pcm
        samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float32) / 32768.0
        resampled = self._stream.resample_chunk(samples, last=last)
        # Round to int16 here (and copy before any reset: the returned array may view
        # the stream's internal buffer, which is freed when the stream is replaced).
        out = np.clip(np.rint(resampled * 32768.0), -32768, 32767).astype("<i2").tobytes()
        if last:
            self._reset()
        return out

    def flush(self) -> bytes:
        """Return the remaining filter tail and reset the stream."""
        return self.process(b"", last=True)
