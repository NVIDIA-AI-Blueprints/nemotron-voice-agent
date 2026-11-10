# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How the caller's turn starts and ends (``speech``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy, VADUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from examples.shared.pipeline_utils import build_user_mute_strategies

if TYPE_CHECKING:
    from examples.frontend_backend_live.config_manager.schema import SpeechConfig


class VADWhileBotSilentUserTurnStartStrategy(VADUserTurnStartStrategy):
    """Pipecat's VAD start, only while the bot is silent: then the caller's first sound starts their turn."""

    def __init__(self, **kwargs):
        """Start with the bot silent."""
        super().__init__(**kwargs)
        self._bot_speaking = False

    async def handle_user_turn_started(self):
        """Count the bot as silent for the next turn, as ``MinWordsUserTurnStartStrategy`` does, so the two agree."""
        self._bot_speaking = False

    async def process_frame(self, frame: Frame):
        """Note when the bot starts and stops speaking; while it is silent, start the turn at the caller's sound."""
        if isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
            self._bot_speaking = isinstance(frame, BotStartedSpeakingFrame)
        elif not self._bot_speaking:
            await super().process_frame(frame)


class SmartTurn:
    """The caller's turns: a turn starts at the caller's sound or on their words, and Smart Turn judges each pause."""

    def __init__(self, speech: SpeechConfig):
        """Read the ``speech`` settings, refusing a judged pause not below the turn silence or a VAD start above it."""
        if speech.vad_stop_seconds >= speech.turn_silence_seconds:
            raise ValueError(
                "speech.vad_stop_seconds must be below speech.turn_silence_seconds: Smart Turn judges a pause before "
                "the silence that ends the turn anyway"
            )
        if speech.vad_start_seconds > speech.vad_stop_seconds:
            raise ValueError(
                "speech.vad_start_seconds must not be above speech.vad_stop_seconds, the VAD stop it is paired with"
            )
        self.vad_start = speech.vad_start_seconds
        self.vad_stop = speech.vad_stop_seconds
        self.fallback = round(speech.turn_silence_seconds - speech.vad_stop_seconds, 3)
        self.min_words = speech.interrupt_min_words

    def user_params(self, welcome_enabled: bool) -> LLMUserAggregatorParams:
        """Return new user-aggregator settings: Silero VAD, the turn strategies, and the shared mute.

        Pipecat's stop strategy ends only a turn with words: the stop timeout ends one with none at the same silence.
        """
        analyzer = LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=self.fallback))
        return LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(start_secs=self.vad_start, stop_secs=self.vad_stop)),
            user_mute_strategies=build_user_mute_strategies(welcome_enabled),
            user_turn_strategies=UserTurnStrategies(
                start=[
                    VADWhileBotSilentUserTurnStartStrategy(),
                    MinWordsUserTurnStartStrategy(min_words=self.min_words),
                ],
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer)],
            ),
            user_turn_stop_timeout=self.fallback,
        )

    def describe(self) -> dict:
        """Return the settings it runs with, as plain JSON for ``GET /v1/live/info``."""
        return {
            "vad_start_seconds": self.vad_start,
            "vad_stop_seconds": self.vad_stop,
            "turn_silence_seconds": round(self.vad_stop + self.fallback, 3),
            "interrupt_min_words": self.min_words,
        }
