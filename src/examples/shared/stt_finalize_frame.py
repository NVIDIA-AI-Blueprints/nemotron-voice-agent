# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Control frame that asks an upstream STT service to finalize the current utterance."""

from dataclasses import dataclass

from pipecat.frames.frames import ControlFrame


@dataclass
class STTFinalizeRequestFrame(ControlFrame):
    """Ask an upstream STT service to finalize the current utterance.

    Pushed upstream by Smart Turn when the audio turn analyzer reports
    COMPLETE. As a control frame, the request remains ordered with queued
    audio and transcript frames instead of overtaking them.
    """

    pass
