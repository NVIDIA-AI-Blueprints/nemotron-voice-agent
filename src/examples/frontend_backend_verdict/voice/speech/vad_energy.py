# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic RMS-energy VAD: the test double and a no-model fallback.

The frame level in dBFS is mapped linearly onto [0, 1] between ``floor_dbfs``
and ``ceiling_dbfs``, so the session's ``threshold`` keeps its meaning (0.5 ~
-40 dBFS with the defaults). Digital silence, including tau2's continuous
μ-law silence, is always 0.
"""

from __future__ import annotations

from examples.frontend_backend_verdict.voice.audio.pcm import rms_dbfs


class EnergyVad:
    """RMS-energy speech probability."""

    def __init__(self, sample_rate: int, *, frame_ms: int = 20, floor_dbfs: float = -60.0, ceiling_dbfs: float = -20.0):
        """Configure 20 ms frames at ``sample_rate``."""
        self._frame_samples = max(1, sample_rate * frame_ms // 1000)
        self._floor = floor_dbfs
        self._ceiling = ceiling_dbfs

    @property
    def frame_samples(self) -> int:
        """Samples per analysis frame."""
        return self._frame_samples

    def speech_probability(self, frame: bytes) -> float:
        """Map the frame's RMS level onto [0, 1]."""
        level = rms_dbfs(frame)
        if level <= self._floor:
            return 0.0
        if level >= self._ceiling:
            return 1.0
        return (level - self._floor) / (self._ceiling - self._floor)

    def reset(self) -> None:
        """Stateless."""
