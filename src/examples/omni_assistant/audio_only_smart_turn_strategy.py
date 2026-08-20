# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Audio-only Smart Turn strategy for Omni-style ASR+LLM services."""

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.frames.frames import (
    InputAudioRawFrame,
    MetricsFrame,
    StartFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy


class AudioOnlySmartTurnStopStrategy(BaseUserTurnStopStrategy):
    """Smart Turn stop strategy for services that do ASR inside the LLM call.

    Pipecat's stock Smart Turn stop strategy waits for a ``TranscriptionFrame``
    after the turn analyzer says the turn is complete. Omni receives audio and
    produces the transcript in the same model call, so there is no upstream
    transcription frame. This strategy keeps the same audio turn analyzer but
    finalizes the turn directly when the analyzer returns ``COMPLETE``.
    """

    def __init__(self, *, turn_analyzer: BaseTurnAnalyzer, **kwargs):
        """Initialize the strategy with an audio turn analyzer."""
        super().__init__(**kwargs)
        self._turn_analyzer = turn_analyzer
        self._vad_user_speaking = False

    async def cleanup(self):
        """Release turn analyzer resources."""
        await self._turn_analyzer.cleanup()
        await super().cleanup()

    async def handle_user_turn_stopped(self):
        """Clear analyzer state after the current user turn ends."""
        self._turn_analyzer.clear()
        await super().handle_user_turn_stopped()

    async def process_frame(self, frame) -> ProcessFrameResult:
        """Process audio and VAD frames to detect end-of-turn."""
        await super().process_frame(frame)

        if isinstance(frame, StartFrame):
            self._turn_analyzer.set_sample_rate(frame.audio_in_sample_rate)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._turn_analyzer.update_vad_start_secs(frame.start_secs)
            self._vad_user_speaking = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_user_speaking = False
            state, prediction = await self._turn_analyzer.analyze_end_of_turn()
            if prediction:
                await self.push_frame(MetricsFrame(data=[prediction]))
            if state == EndOfTurnState.COMPLETE:
                await self.trigger_user_turn_stopped()
        elif isinstance(frame, InputAudioRawFrame):
            state = self._turn_analyzer.append_audio(frame.audio, self._vad_user_speaking)
            if state == EndOfTurnState.COMPLETE:
                _, prediction = await self._turn_analyzer.analyze_end_of_turn()
                if prediction:
                    await self.push_frame(MetricsFrame(data=[prediction]))
                await self.trigger_user_turn_stopped()

        return ProcessFrameResult.CONTINUE
