# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Control frame that asks an upstream STT service to finalize the current utterance."""

from dataclasses import dataclass

from pipecat.frames.frames import SystemFrame


@dataclass
class STTFinalizeFrame(SystemFrame):
    """Ask the STT service to flush the current utterance immediately.

    Pushed upstream by Smart Turn when the audio turn analyzer reports
    COMPLETE, so NVIDIA ASR can send ``force_eou`` and emit a final
    transcript without waiting for server-side silence endpointing.
    """

    pass
