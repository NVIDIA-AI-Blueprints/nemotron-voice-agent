# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PCM16 helpers: base64 transport, silence, duration math, level.

``realtime.audio`` has equivalent base64 helpers, but it imports Pipecat at
module load and only accepts even-length (PCM16) payloads; G.711 payloads can
have any length, so the codec layer keeps its own small helpers.
"""

from __future__ import annotations

import base64
import binascii
import math

import numpy as np

BYTES_PER_PCM16_SAMPLE = 2


def b64decode_audio(payload: str) -> bytes:
    """Decode a base64 audio payload, raising ``ValueError`` on malformed input."""
    if not isinstance(payload, str):
        raise ValueError("audio must be a base64 string")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"invalid base64 audio: {exc}") from exc


def b64encode_audio(data: bytes) -> str:
    """Encode audio bytes for a ``*.delta`` event."""
    return base64.b64encode(data).decode("ascii")


def pcm16_samples(pcm: bytes) -> int:
    """Number of PCM16 samples in ``pcm``."""
    return len(pcm) // BYTES_PER_PCM16_SAMPLE


def pcm16_duration_ms(pcm: bytes, rate: int) -> float:
    """Duration of ``pcm`` in milliseconds at ``rate`` Hz."""
    return pcm16_samples(pcm) * 1000.0 / rate if rate > 0 else 0.0


def pcm16_bytes_for_ms(ms: float, rate: int) -> int:
    """Byte length of ``ms`` milliseconds of PCM16 at ``rate`` Hz (whole samples)."""
    return int(round(ms * rate / 1000.0)) * BYTES_PER_PCM16_SAMPLE


def silence(ms: float, rate: int) -> bytes:
    """``ms`` milliseconds of PCM16 digital silence."""
    return b"\x00" * pcm16_bytes_for_ms(ms, rate)


def rms_dbfs(pcm: bytes) -> float:
    """RMS level of ``pcm`` in dBFS; ``-inf`` for digital silence or empty input."""
    samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float64)
    if samples.size == 0:
        return -math.inf
    rms = math.sqrt(float(np.mean(samples * samples)))
    if rms <= 0.0:
        return -math.inf
    return 20.0 * math.log10(rms / 32768.0)


def tone(ms: float, rate: int, *, frequency: float = 440.0, amplitude: float = 0.3) -> bytes:
    """A sine tone, used by stand-in synthesizers and tests."""
    count = pcm16_bytes_for_ms(ms, rate) // BYTES_PER_PCM16_SAMPLE
    t = np.arange(count, dtype=np.float64) / rate
    wave = amplitude * 32767.0 * np.sin(2.0 * math.pi * frequency * t)
    return wave.astype("<i2").tobytes()
