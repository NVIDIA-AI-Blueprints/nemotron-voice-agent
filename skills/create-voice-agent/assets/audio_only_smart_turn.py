# ruff: noqa: D101,D102,D103,D107

"""Audio-only Smart Turn stop strategy for omni pipelines without STT.

COPY to project root unchanged. Required — stock Smart Turn stop waits for
TranscriptionFrame from STT; omni has no STT.
"""

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    MetricsFrame,
    SpeechControlParamsFrame,
    StartFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.user_stop import BaseUserTurnStopStrategy


class AudioOnlySmartTurnStopStrategy(BaseUserTurnStopStrategy):
    """Stop a user turn from Smart Turn audio classification only (no STT transcript)."""

    def __init__(self, *, turn_analyzer: BaseTurnAnalyzer | None = None, **kwargs):
        super().__init__(**kwargs)
        self._turn_analyzer = turn_analyzer or LocalSmartTurnAnalyzerV3()
        self._vad_user_speaking = False

    async def reset(self):
        await super().reset()
        self._turn_analyzer.clear()
        self._vad_user_speaking = False

    async def cleanup(self):
        await super().cleanup()
        await self._turn_analyzer.cleanup()

    async def process_frame(self, frame: Frame):
        await super().process_frame(frame)

        if isinstance(frame, StartFrame):
            await self._start(frame)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._turn_analyzer.update_vad_start_secs(frame.start_secs)
            self._vad_user_speaking = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_user_speaking = False
            await self._analyze_end_of_turn()
        elif isinstance(frame, InputAudioRawFrame):
            self._turn_analyzer.append_audio(frame.audio, self._vad_user_speaking)

    async def _start(self, frame: StartFrame):
        self._turn_analyzer.set_sample_rate(frame.audio_in_sample_rate)
        await self.broadcast_frame(SpeechControlParamsFrame, turn_params=self._turn_analyzer.params)

    async def _analyze_end_of_turn(self):
        state, result = await self._turn_analyzer.analyze_end_of_turn()
        if result:
            await self.push_frame(MetricsFrame(data=[result]))

        if state is EndOfTurnState.COMPLETE and (
            result is None or getattr(result, "is_complete", True)
        ):
            await self.trigger_user_turn_stopped()
