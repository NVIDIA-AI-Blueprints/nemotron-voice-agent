# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How the caller's turn starts and ends (``speech``).

NVIDIA's streaming speech-to-text finalizes a stretch of speech on its own after 320 ms of silence (``stop_history``).
:class:`SmartTurn` builds the user aggregator's settings, the VAD and the turn strategies:

* a turn starts at whichever of two start strategies reports first. While the bot is silent, the caller's first sound
  starts it (:class:`VADWhileBotSilentUserTurnStartStrategy`, Pipecat's VAD start, as in the repository's other
  examples). While the bot speaks, ``interrupt_min_words`` words start it (Pipecat's
  ``MinWordsUserTurnStartStrategy``), so a cough or a one-word backchannel does not cut it. The turn's start is what
  interrupts the bot;
* the VAD reports a pause after ``vad_stop_seconds``, and Smart Turn judges whether the turn is over. A turn judged over
  ends at the speech-to-text's final, and a pause of ``turn_silence_seconds`` ends any turn, with or without words,
  whatever Smart Turn judged.

Its own Silero VAD: ``vad_start_seconds`` of speech mark the caller as speaking.
"""

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
    """Pipecat's VAD start, only while the bot is silent: then the caller's first sound starts their turn.

    It reads whether the bot speaks from the bot's speaking frames, and readies itself for each turn, as Pipecat's
    ``MinWordsUserTurnStartStrategy`` does, so the two always agree on it.
    """

    def __init__(self, **kwargs):
        """Start with the bot silent."""
        super().__init__(**kwargs)
        self._bot_speaking = False

    async def handle_user_turn_started(self):
        """Ready the strategy for the next turn: the bot counts as silent, as for ``MinWordsUserTurnStartStrategy``."""
        self._bot_speaking = False

    async def process_frame(self, frame: Frame):
        """Note when the bot starts and stops speaking; while it is silent, start the turn at the caller's sound."""
        if isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
            self._bot_speaking = isinstance(frame, BotStartedSpeakingFrame)
        elif not self._bot_speaking:
            await super().process_frame(frame)


class SmartTurn:
    """The caller's turns: a turn starts at the caller's sound or on their words, and Smart Turn judges each pause.

    It holds only its settings: each session gets its own user-aggregator settings from it.
    """

    def __init__(self, speech: SpeechConfig):
        """Read the VAD's start, the pause Smart Turn judges, the silence that ends a turn, and the words that cut.

        Raises:
            ValueError: The judged pause is not shorter than that silence, or the VAD's start is longer than the pause.
        """
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
        self.fallback = round(speech.turn_silence_seconds - speech.vad_stop_seconds, 3)  # Smart Turn's, after the stop
        self.min_words = speech.interrupt_min_words

    def user_params(self, welcome_enabled: bool) -> LLMUserAggregatorParams:
        """New user-aggregator settings: Silero VAD stopping after the judged pause, and the turn strategies.

        The turn starts at the first start strategy to report, Pipecat's default: the VAD's start while the bot is
        silent, or, while it speaks, the words of a transcript, interim ones included. Pipecat's stop strategy ends
        only a turn with words, so the user aggregator's stop timeout ends a turn with none at the same silence. With
        ``welcome_enabled``, the shared mute also keeps the caller's audio out until the welcome has played.
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
        """The settings it runs with, as plain JSON for ``GET /v1/live/info``."""
        return {
            "vad_start_seconds": self.vad_start,
            "vad_stop_seconds": self.vad_stop,
            "turn_silence_seconds": round(self.vad_stop + self.fallback, 3),
            "interrupt_min_words": self.min_words,
        }
