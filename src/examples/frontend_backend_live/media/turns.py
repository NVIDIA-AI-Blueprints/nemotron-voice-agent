# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How the caller's turn starts and ends (``speech``).

With ``spelling_hold``, a turn whose words so far end in a spelled run (single letters or digits in a row,
``spoken.ends_spelled``) is not ended on Smart Turn's verdict: the ``turn_silence_seconds`` pause ends it, so a caller
who pauses mid-spelling keeps their turn (:class:`SpellingHoldUserTurnStopStrategy`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame, TranscriptionFrame
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy, VADUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from examples.frontend_backend_live.spoken import ends_spelled
from examples.shared.observe import log_observation
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


class SpellingHoldUserTurnStopStrategy(TurnAnalyzerUserTurnStopStrategy):
    """Pipecat's Smart Turn stop, except that it does not report a turn whose words so far end in a spelled run.

    The user turn controller's stop timeout then ends it, as it ends any turn Smart Turn holds open; speech before then
    continues the same turn. Only Pipecat's public methods are overridden, each calling Pipecat's first.
    """

    def __init__(
        self, *, observe: Callable[..., None] = log_observation, clock: Callable[[], float] = time.monotonic, **kwargs
    ):
        """Pipecat's strategy with ``kwargs``; ``observe`` gets one ``speech.spelling_held`` per held turn."""
        super().__init__(**kwargs)
        self.observe, self.clock = observe, clock
        self._words: list[str] = []  # the turn's final transcripts so far
        self._held_at: float | None = None  # when it first declined to report the turn

    async def process_frame(self, frame: Frame):
        """Keep each final transcript of the turn, then process the frame as Pipecat does."""
        if isinstance(frame, TranscriptionFrame) and frame.text:
            self._words.append(frame.text)
        return await super().process_frame(frame)

    async def handle_user_turn_started(self):
        """Pipecat's reset, then forget the words."""
        await super().handle_user_turn_started()
        self._forget()

    async def handle_user_turn_stopped(self):
        """Pipecat's reset, then note a held turn's extra wait and forget the words."""
        await super().handle_user_turn_stopped()
        if self._held_at is not None:
            self.observe("speech.spelling_held", held_seconds=round(self.clock() - self._held_at, 3))
        self._forget()

    async def report_user_turn_stopped(self, **kwargs):
        """Decline while the turn's words end in a spelled run; otherwise report as Pipecat does.

        A failure of the check reports the turn, as Pipecat would.
        """
        try:
            hold = ends_spelled(" ".join(self._words))
        except Exception:  # never keep a turn open on an error of the check
            hold = False
        if hold:
            if self._held_at is None:
                self._held_at = self.clock()
                logger.debug("A turn that ends mid-spelling waits for the stop timeout")
            return
        await super().report_user_turn_stopped(**kwargs)

    def _forget(self) -> None:
        self._words, self._held_at = [], None


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
        self.spelling_hold = speech.spelling_hold

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
                stop=[self._stop(analyzer)],
            ),
            user_turn_stop_timeout=self.fallback,
        )

    def _stop(self, analyzer: LocalSmartTurnAnalyzerV3) -> TurnAnalyzerUserTurnStopStrategy:
        """The stop strategy: the spelling hold with ``spelling_hold``, else Pipecat's, exactly as before."""
        if self.spelling_hold:
            return SpellingHoldUserTurnStopStrategy(turn_analyzer=analyzer)
        return TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer)

    def describe(self) -> dict:
        """Return the settings it runs with, as plain JSON for ``GET /v1/live/info``."""
        return {
            "vad_start_seconds": self.vad_start,
            "vad_stop_seconds": self.vad_stop,
            "turn_silence_seconds": round(self.vad_stop + self.fallback, 3),
            "interrupt_min_words": self.min_words,
            "spelling_hold": self.spelling_hold,
        }
