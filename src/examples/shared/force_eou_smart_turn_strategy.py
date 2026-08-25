# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Smart Turn stop strategy that flushes NVIDIA ASR when the audio turn is complete."""

from pipecat.processors.frame_processor import FrameDirection
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)

from examples.shared.stt_finalize_frame import STTFinalizeFrame


class ForceEouSmartTurnStopStrategy(TurnAnalyzerUserTurnStopStrategy):
    """Stock Smart Turn stop strategy plus an upstream ASR flush on COMPLETE.

    Pipecat's ``TurnAnalyzerUserTurnStopStrategy`` waits for a
    ``TranscriptionFrame`` after the audio turn analyzer reports COMPLETE.
    NVIDIA ASR normally finalizes only after its own ``stop_history`` silence
    window, so that wait can dominate turn latency.

    When COMPLETE is reached and a finalized transcript is not already in
    hand, this strategy pushes :class:`STTFinalizeFrame` upstream once. The
    NVIDIA STT service turns that into a per-chunk ``force_eou`` flag, which
    flushes the final transcript. Parent transcript gating (including the 1s
    STT safety net) is unchanged.
    """

    def __init__(self, **kwargs):
        """Initialize the strategy with the same kwargs as the parent."""
        super().__init__(**kwargs)
        self._force_eou_requested = False

    async def _discard_pending_end_of_turn(self):
        """Drop COMPLETE state and allow force_eou again on the next pause."""
        self._force_eou_requested = False
        await super()._discard_pending_end_of_turn()

    async def _maybe_trigger_user_turn_stopped(self):
        """Flush ASR once on COMPLETE, then wait for the transcript as usual."""
        if self._turn_complete and not self._transcript_finalized and not self._force_eou_requested:
            self._force_eou_requested = True
            await self.push_frame(STTFinalizeFrame(), FrameDirection.UPSTREAM)
        await super()._maybe_trigger_user_turn_stopped()
