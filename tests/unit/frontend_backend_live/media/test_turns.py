# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The caller's turns: their strategies in Pipecat's own user aggregator, and the user settings they build.

Local only: no speech-to-text connects. A turn's words are read from each context frame as it passes: Pipecat's context
is one object, which later turns change.
"""

import time
import unittest

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, BaseTurnParams, EndOfTurnState
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.stt_latency import NVIDIA_TTFS_P99
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_mute import MuteUntilFirstBotCompleteUserMuteStrategy
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import StrategyMode, UserTurnStrategies

from examples.frontend_backend_live.config_manager.schema import SpeechConfig
from examples.frontend_backend_live.media.turns import SmartTurn, VADWhileBotSilentUserTurnStartStrategy


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


def interim(text):
    return InterimTranscriptionFrame(text=text, user_id="caller", timestamp="0")


def spoken(stop, *before_the_pause):
    """The caller speaks, with Pipecat's NVIDIA budget for the final, then pauses: a VAD stop of ``stop``."""
    audio = [InputAudioRawFrame(audio=b"\x00" * 640, sample_rate=16000, num_channels=1) for _ in range(5)]  # 100 ms
    metadata = STTMetadataFrame(service_name="stt", ttfs_p99_latency=NVIDIA_TTFS_P99)
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

    async def test_smart_turn_ends_a_complete_turn_with_no_words_at_the_turn_silence_not_at_five_seconds(self):
        timeout = SmartTurn(SpeechConfig()).user_params(welcome_enabled=False).user_turn_stop_timeout
        stop = TurnAnalyzerUserTurnStopStrategy(turn_analyzer=Complete())
        ends, _, words = await self.turns(stop, [*spoken(0.2), SleepFrame(1.4)], user_turn_stop_timeout=timeout)
        self.assertEqual((len(ends), words), (1, []))
        self.assertAlmostEqual(ends[0], 1.0, delta=0.15)  # 1.2 s of silence, after the 0.2 s VAD stop


def said(*transcripts):
    """The caller speaks: the VAD's start, each transcript in its turn, then the VAD's stop."""
    frames = [VADUserStartedSpeakingFrame(start_secs=0.12)]
    for transcript in transcripts:
        frames += [SleepFrame(0.05), transcript]
    return [*frames, SleepFrame(0.05), VADUserStoppedSpeakingFrame(stop_secs=0.2)]


class TurnStartTests(unittest.IsolatedAsyncioTestCase):
    """What starts the caller's turn, and so cuts the bot: the start strategies the caller's turns build."""

    async def turns(self, *frames, **speech):
        """Each turn the caller's speech started, and the words each wrote, through Pipecat's user aggregator."""
        start = SmartTurn(SpeechConfig(**speech)).user_params(welcome_enabled=False).user_turn_strategies.start
        stop = TurnAnalyzerUserTurnStopStrategy(turn_analyzer=Complete())
        params = LLMUserAggregatorParams(user_turn_strategies=UserTurnStrategies(start=start, stop=[stop]))
        out = Stamps()
        await run_test(
            Pipeline([LLMContextAggregatorPair(LLMContext(), user_params=params).user(), out]),
            frames_to_send=[*frames, SleepFrame(0.3)],
        )
        return len(out.after(UserStartedSpeakingFrame)), len(out.after(InterruptionFrame)), out.words

    async def test_while_the_bot_speaks_a_cough_and_a_one_word_backchannel_start_no_turn_and_cut_nothing(self):
        cases = {"a cough": said(), "a one-word backchannel": said(interim("Okay"), final("Okay."))}
        for name, caller in cases.items():
            with self.subTest(name):
                self.assertEqual(await self.turns(BotStartedSpeakingFrame(), *caller), (0, 0, []))

    async def test_while_the_bot_speaks_two_words_start_the_turn_cut_it_and_carry_the_whole_turn(self):
        caller = said(interim("Okay"), interim("Okay wait"), final("Okay, wait a second."))
        self.assertEqual(await self.turns(BotStartedSpeakingFrame(), *caller), (1, 1, ["Okay, wait a second."]))

    async def test_while_the_bot_speaks_the_configured_interrupt_min_words_start_the_turn(self):
        two = said(interim("Okay"), interim("Okay wait"), final("Okay wait."))
        three = said(interim("Okay wait"), interim("Okay wait now"), final("Okay, wait now."))
        self.assertEqual(await self.turns(BotStartedSpeakingFrame(), *two, interrupt_min_words=3), (0, 0, []))
        cut = await self.turns(BotStartedSpeakingFrame(), *three, interrupt_min_words=3)
        self.assertEqual(cut, (1, 1, ["Okay, wait now."]))

    async def test_while_the_bot_is_silent_a_cough_starts_the_turn(self):
        cases = {
            "before the bot has spoken": said(),
            "after the bot's line": (BotStartedSpeakingFrame(), BotStoppedSpeakingFrame(), *said()),
        }
        for name, frames in cases.items():
            with self.subTest(name):
                self.assertEqual(await self.turns(*frames), (1, 1, []))

    async def test_while_the_bot_is_silent_the_first_sound_starts_the_turn_and_a_word_joins_it(self):
        self.assertEqual(await self.turns(*said(interim("Okay"), final("Okay."))), (1, 1, ["Okay."]))


class TurnSettingsTests(unittest.TestCase):
    def test_the_turns_start_their_vad_after_the_configured_speech_and_mute_the_caller_only_for_a_welcome(self):
        turns = SmartTurn(SpeechConfig(vad_start_seconds=0.16, interrupt_min_words=3))
        params = turns.user_params(welcome_enabled=False)
        self.assertEqual((params.vad_analyzer.params.start_secs, params.vad_analyzer.params.stop_secs), (0.16, 0.2))
        sound, words = params.user_turn_strategies.start
        self.assertIsInstance(sound, VADWhileBotSilentUserTurnStartStrategy)
        self.assertIsInstance(words, MinWordsUserTurnStartStrategy)
        self.assertIs(params.user_turn_strategies_params.start_mode, StrategyMode.FIRST)  # the first to report decides
        self.assertEqual(turns.describe()["interrupt_min_words"], 3)
        self.assertEqual(params.user_mute_strategies, [])
        (mute,) = turns.user_params(welcome_enabled=True).user_mute_strategies  # until a welcome has played
        self.assertIsInstance(mute, MuteUntilFirstBotCompleteUserMuteStrategy)
