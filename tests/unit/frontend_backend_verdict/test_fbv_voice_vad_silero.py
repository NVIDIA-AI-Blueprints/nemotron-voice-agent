# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103

"""The Silero adapter against the real model bundled with Pipecat (offline)."""

from __future__ import annotations

import math
import unittest

from examples.frontend_backend_verdict.voice.speech.vad_silero import SileroVad


class SileroVadTests(unittest.TestCase):
    def test_speech_probability_is_a_python_float_per_frame(self) -> None:
        for rate in (8000, 16000):
            vad = SileroVad(rate)
            silence = b"\x00\x00" * vad.frame_samples
            tone = b"".join(
                int(8000 * math.sin(2 * math.pi * 220 * i / rate)).to_bytes(2, "little", signed=True)
                for i in range(vad.frame_samples)
            )
            for frame in (silence, tone):
                probability = vad.speech_probability(frame)
                self.assertIs(type(probability), float)
                self.assertGreaterEqual(probability, 0.0)
                self.assertLessEqual(probability, 1.0)
            vad.reset()
