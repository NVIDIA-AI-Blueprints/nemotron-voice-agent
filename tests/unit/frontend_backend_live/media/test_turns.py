# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The turn ends: their strategies in Pipecat's own user aggregator, and the user settings each builds.

Local only: no speech-to-text connects. A turn's words are read from each context frame as it passes: Pipecat's context
is one object, which later turns change.
"""

import time
import unittest

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, BaseTurnParams, EndOfTurnState
from pipecat.audio.vad.vad_analyzer import VAD_START_SECS
from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_mute import MuteUntilFirstBotCompleteUserMuteStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from examples.frontend_backend_live.config_manager.schema import SpeechConfig
from examples.frontend_backend_live.media.turns import TURN_ENDS, Shared, Silence, SmartTurn, ttfs_after


class Stamps(FrameProcessor):
    """Notes when each frame passes, and a caller turn's words as its context frame passes."""

    def __init__(self):
        super().__init__()
        self.seen, self.words = [], []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.seen.append((time.monotonic(), frame))
        if isinstance(frame, LLMContextFrame) and (message := frame.context.get_messages()[-1])["role"] == "user":
            self.words.append(message["content"])
        await self.push_frame(frame, direction)

    def after(self, kind, moment=0.0):
        return [stamp - moment for stamp, frame in self.seen if isinstance(frame, kind)]


class Complete(BaseTurnAnalyzer):
    """Smart Turn's stand-in: every pause judged complete."""

    @property
    def speech_triggered(self):
        return True

    @property
    def params(self):
        return BaseTurnParams()

    def append_audio(self, buffer, is_speech):
        return EndOfTurnState.INCOMPLETE

    async def analyze_end_of_turn(self):
        return EndOfTurnState.COMPLETE, None

    def clear(self):
        pass


def final(text):
    return TranscriptionFrame(text=text, user_id="caller", timestamp="0", finalized=True)


def spoken(stop, *before_the_pause):
    """The caller speaks, with the final's budget that a VAD stop of ``stop`` sets, then pauses."""
    audio = [InputAudioRawFrame(audio=b"\x00" * 640, sample_rate=16000, num_channels=1) for _ in range(5)]  # 100 ms
    metadata = STTMetadataFrame(service_name="stt", ttfs_p99_latency=ttfs_after(stop))
    started, pause = VADUserStartedSpeakingFrame(start_secs=0.2), VADUserStoppedSpeakingFrame(stop_secs=stop)
    return [metadata, started, *audio, *before_the_pause, pause]


class TurnTimingTests(unittest.IsolatedAsyncioTestCase):
    async def turns(self, stop_strategy, frames, **params):
        """Each turn's end and each final after the pause, and the turns' words, through Pipecat's user aggregator."""
        heard, out = Stamps(), Stamps()
        params = LLMUserAggregatorParams(user_turn_strategies=UserTurnStrategies(stop=[stop_strategy]), **params)
        await run_test(
            Pipeline([heard, LLMContextAggregatorPair(LLMContext(), user_params=params).user(), out]),
            frames_to_send=frames,
        )
        pause = heard.after(VADUserStoppedSpeakingFrame)[0]
        return out.after(UserStoppedSpeakingFrame, pause), heard.after(TranscriptionFrame, pause), out.words

    def silence(self):
        silence = Silence(SpeechConfig(turn_end="silence", turn_silence_seconds=0.8))
        return silence.user_params(welcome_enabled=False).user_turn_strategies.stop[0]

    async def test_smart_turn_ends_a_complete_turn_with_no_words_at_the_turn_silence_not_at_five_seconds(self):
        timeout = SmartTurn(SpeechConfig()).user_params(welcome_enabled=False).user_turn_stop_timeout
        stop = TurnAnalyzerUserTurnStopStrategy(turn_analyzer=Complete())
        ends, _, words = await self.turns(stop, [*spoken(0.2), SleepFrame(1.4)], user_turn_stop_timeout=timeout)
        self.assertEqual((len(ends), words), (1, []))
        self.assertAlmostEqual(ends[0], 1.0, delta=0.15)  # 1.2 s of silence, after the 0.2 s VAD stop

    async def test_silence_ends_a_turn_on_a_final_after_the_pause_at_once(self):
        ends, finals, words = await self.turns(
            self.silence(), [*spoken(0.8), SleepFrame(0.1), final("yes"), SleepFrame(0.1)]
        )
        self.assertEqual((len(ends), words), (1, ["yes"]))
        self.assertTrue(finals[-1] < ends[0] < 0.5)

    async def test_silence_ends_a_turn_with_no_words_at_the_finals_budget_not_at_five_seconds(self):
        ends, _, words = await self.turns(self.silence(), [*spoken(0.8), SleepFrame(1.2)])
        self.assertEqual((len(ends), words), (1, []))
        self.assertTrue(0.7 < ends[0] < 1.1)  # the final's 0.8 s after the VAD stop, with the event loop's slack

    async def test_a_final_later_than_its_budget_starts_a_turn_of_its_own(self):
        interim = InterimTranscriptionFrame(text="yes", user_id="caller", timestamp="0")
        frames = [*spoken(0.8, interim), SleepFrame(1.0), final("yes"), SleepFrame(0.5)]
        ends, finals, words = await self.turns(self.silence(), frames)
        self.assertEqual((len(ends), words), (2, ["yes"]))
        self.assertTrue(ends[0] < finals[0] < ends[1])


class TurnEndTests(unittest.TestCase):
    def test_each_turn_end_starts_its_vad_after_the_configured_speech_and_mutes_the_caller_only_for_a_welcome(self):
        for name, stop in (("smart_turn", 0.2), ("silence", 0.8)):
            with self.subTest(name):
                end = TURN_ENDS[name](SpeechConfig(turn_end=name, vad_start_seconds=0.16, turn_silence_seconds=0.8))
                vad = end.user_params(welcome_enabled=False).vad_analyzer.params
                self.assertEqual((vad.start_secs, vad.stop_secs), (0.16, stop))
                self.assertEqual(end.user_params(welcome_enabled=False).user_mute_strategies, [])
                (mute,) = end.user_params(welcome_enabled=True).user_mute_strategies  # until a welcome has played
                self.assertIsInstance(mute, MuteUntilFirstBotCompleteUserMuteStrategy)
        shared = Shared(SpeechConfig(turn_end="shared", vad_start_seconds=0.16)).user_params(welcome_enabled=False)
        self.assertEqual(shared.vad_analyzer.params.start_secs, VAD_START_SECS)  # the shared pipeline's VAD, as built
