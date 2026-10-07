# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Silero VAD as a frame-probability source.

Only Pipecat's model wrapper is used (``voice_confidence`` per frame). Pipecat's
own start/stop state machine is not: stop timing belongs to the segmenter, which
counts samples, so the endpointer never depends on wall-clock time.
"""

from __future__ import annotations

import numpy as np
from pipecat.audio.vad.silero import SileroVADAnalyzer


class SileroVad:
    """Per-session Silero model (8 kHz or 16 kHz)."""

    def __init__(self, sample_rate: int) -> None:
        """Load the ONNX model for ``sample_rate``."""
        self._analyzer = SileroVADAnalyzer(sample_rate=sample_rate)
        self._analyzer.set_sample_rate(sample_rate)

    @property
    def frame_samples(self) -> int:
        """512 samples at 16 kHz, 256 at 8 kHz."""
        return self._analyzer.num_frames_required()

    def speech_probability(self, frame: bytes) -> float:
        """Silero's speech confidence for one frame."""
        # Pipecat returns the model's first output row, a one-element array, and
        # numpy refuses float() on arrays with a dimension.
        return float(np.asarray(self._analyzer.voice_confidence(frame)).item())

    def reset(self) -> None:
        """Reset the recurrent model state."""
        model = getattr(self._analyzer, "_model", None)
        if model is not None:
            model.reset_states()
