# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire audio conversion for the primary WebSocket.

24 kHz mono PCM16 is the exchange format with the pipeline. The wire carries PCM at 16 or 24 kHz, or G.711
mu-law or A-law at 8 kHz. Conversion keeps fractional-frame state across chunks: resetting a resampler for
every network packet would lose samples.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from av import AudioFrame, AudioResampler

from live.protocol import AudioFormat

INTERNAL_RATE = 24000


class PCMResampler:
    """Streaming sample-rate conversion for mono PCM16."""

    def __init__(self, source_rate: int, target_rate: int):
        """Create a resampler from ``source_rate`` to ``target_rate`` Hz."""
        self.source_rate, self.target_rate = source_rate, target_rate
        self.resampler = AudioResampler(format="s16", layout="mono", rate=target_rate)
        self.samples = 0

    def convert(self, pcm: bytes) -> bytes:
        """Convert one chunk of little-endian PCM16."""
        if not pcm:
            return b""
        if len(pcm) % 2:
            raise ValueError("PCM requires complete 16-bit samples")
        frame = AudioFrame.from_ndarray(np.frombuffer(pcm, dtype="<i2").reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self.source_rate
        frame.pts, frame.time_base = self.samples, Fraction(1, self.source_rate)
        self.samples += frame.samples
        return b"".join(f.to_ndarray().astype("<i2").tobytes() for f in self.resampler.resample(frame))


# G.711 tables keep Python 3.13+ compatibility (audioop was removed).
def _decode_ulaw(value: int) -> int:
    value = ~value & 255
    sample = ((value & 15) << 3) + 132
    sample <<= (value & 112) >> 4
    return 132 - sample if value & 128 else sample - 132


def _decode_alaw(value: int) -> int:
    value ^= 0x55
    sample = (value & 15) << 4
    exponent = (value & 112) >> 4
    sample += 8 if exponent == 0 else 264
    if exponent > 1:
        sample <<= exponent - 1
    return sample if value & 128 else -sample


ULAW = np.array([_decode_ulaw(x) for x in range(256)], dtype="<i2")
ALAW = np.array([_decode_alaw(x) for x in range(256)], dtype="<i2")


class WireCodec:
    """Converts between a session's wire audio format and 24 kHz PCM16."""

    def __init__(self, audio_format: AudioFormat | None = None):
        """Create the codec; ``None`` means the default 24 kHz PCM."""
        self.format = audio_format or AudioFormat()
        self.decode_resampler = PCMResampler(self.format.rate, INTERNAL_RATE)
        self.encode_resampler = PCMResampler(INTERNAL_RATE, self.format.rate)
        table = ULAW if self.format.type == "audio/pcmu" else ALAW
        # Exact nearest-code lookup; the table is small and built once per connection.
        order = np.argsort(table)
        self.sorted_values, self.sorted_codes = table[order].astype(np.int32), order

    def decode(self, raw: bytes) -> bytes:
        """Return 24 kHz PCM16 for a chunk of wire audio."""
        if self.format.type != "audio/pcm":
            table = ULAW if self.format.type == "audio/pcmu" else ALAW
            raw = table[np.frombuffer(raw, dtype=np.uint8)].tobytes()
        return self.decode_resampler.convert(raw)

    def encode(self, pcm: bytes) -> bytes:
        """Return wire audio for a chunk of 24 kHz PCM16."""
        raw = self.encode_resampler.convert(pcm)
        if self.format.type == "audio/pcm":
            return raw
        samples = np.frombuffer(raw, dtype="<i2").astype(np.int32)
        upper = np.clip(np.searchsorted(self.sorted_values, samples), 1, 255)
        lower = upper - 1
        closest = np.where(
            abs(samples - self.sorted_values[lower]) <= abs(samples - self.sorted_values[upper]), lower, upper
        )
        return self.sorted_codes[closest].astype(np.uint8).tobytes()
