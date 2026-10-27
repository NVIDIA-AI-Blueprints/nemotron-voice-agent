# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire audio conversion for the primary WebSocket.

24 kHz mono PCM16 is the exchange format with the pipeline. The wire carries PCM at 16 or 24 kHz, or G.711
mu-law or A-law at 8 kHz. Each wire format has its own sample codec (:data:`SAMPLE_CODECS`), picked once when the
session's codec is built; G.711 follows the standard ITU-T G.711 segment algorithm. Conversion keeps
fractional-frame state across chunks: resetting a resampler for every network packet would lose samples.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from av import AudioFrame, AudioResampler

from live.ports import EXCHANGE_RATE  # the PCM16 the gateway exchanges with the frontend adapter, both ways
from live.protocol import AudioFormat


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
# The upper end of each G.711 segment: of the 14-bit magnitude (mu-law, after its bias) and the 13-bit one (A-law).
ULAW_SEGMENT_ENDS = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF])
ALAW_SEGMENT_ENDS = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF])
ULAW_BIAS, ULAW_CLIP = 33, 8159  # in the 14-bit domain


class LinearPcm:
    """``audio/pcm``: the wire carries PCM16 samples as they are."""

    def decode(self, raw: bytes) -> bytes:
        """Return PCM16 for wire bytes."""
        return raw

    def encode(self, pcm: bytes) -> bytes:
        """Return wire bytes for PCM16."""
        return pcm


class MuLaw:
    """``audio/pcmu``: G.711 mu-law, one byte per sample."""

    def decode(self, raw: bytes) -> bytes:
        """Return PCM16 for mu-law bytes."""
        return ULAW[np.frombuffer(raw, dtype=np.uint8)].tobytes()

    def encode(self, pcm: bytes) -> bytes:
        """Return mu-law bytes for PCM16, by the standard segment algorithm (G.711 on the 14-bit sample)."""
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.int32) >> 2
        mask = np.where(samples < 0, 0x7F, 0xFF)
        magnitude = np.minimum(np.abs(samples), ULAW_CLIP) + ULAW_BIAS
        segment = np.searchsorted(ULAW_SEGMENT_ENDS, magnitude)
        code = np.where(segment >= 8, 0x7F, (segment << 4) | ((magnitude >> (segment + 1)) & 0xF))
        return (code ^ mask).astype(np.uint8).tobytes()


class ALaw:
    """``audio/pcma``: G.711 A-law, one byte per sample."""

    def decode(self, raw: bytes) -> bytes:
        """Return PCM16 for A-law bytes."""
        return ALAW[np.frombuffer(raw, dtype=np.uint8)].tobytes()

    def encode(self, pcm: bytes) -> bytes:
        """Return A-law bytes for PCM16, by the standard segment algorithm (G.711 on the 13-bit sample)."""
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.int32) >> 3
        negative = samples < 0
        mask = np.where(negative, 0x55, 0xD5)
        magnitude = np.where(negative, -samples - 1, samples)
        segment = np.searchsorted(ALAW_SEGMENT_ENDS, magnitude)
        quantized = (magnitude >> np.maximum(segment, 1)) & 0xF
        code = np.where(segment >= 8, 0x7F, (segment << 4) | quantized)
        return (code ^ mask).astype(np.uint8).tobytes()


# How each wire format carries its samples: the one place the format's type picks a class.
SAMPLE_CODECS: dict[str, type[LinearPcm | MuLaw | ALaw]] = {
    "audio/pcm": LinearPcm,
    "audio/pcmu": MuLaw,
    "audio/pcma": ALaw,
}


class WireCodec:
    """Converts between a session's wire audio format and 24 kHz PCM16."""

    exchange_rate = EXCHANGE_RATE  # the rate of the PCM16 that decode() returns and encode() takes

    def __init__(self, audio_format: AudioFormat | None = None):
        """Create the codec; ``None`` means the default 24 kHz PCM."""
        self.format = audio_format or AudioFormat()
        self.samples = SAMPLE_CODECS[self.format.type]()
        self.decode_resampler = PCMResampler(self.format.rate, EXCHANGE_RATE)
        self.encode_resampler = PCMResampler(EXCHANGE_RATE, self.format.rate)

    def decode(self, raw: bytes) -> bytes:
        """Return 24 kHz PCM16 for a chunk of wire audio."""
        return self.decode_resampler.convert(self.samples.decode(raw))

    def encode(self, pcm: bytes) -> bytes:
        """Return wire audio for a chunk of 24 kHz PCM16."""
        return self.samples.encode(self.encode_resampler.convert(pcm))
