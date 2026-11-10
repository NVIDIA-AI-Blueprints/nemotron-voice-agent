# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The caller's turns: their strategies in Pipecat's own user aggregator, and the user settings they build."""

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
    """Processor noting each frame's time, and a turn's words as its context passes, since later turns change it."""

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


def said(*transcripts):
    """Return the caller's speech: the VAD's start, each transcript in its turn, then the VAD's stop."""
    frames = [VADUserStartedSpeakingFrame(start_secs=0.12)]
    for transcript in transcripts:
        frames += [SleepFrame(0.05), transcript]
    return [*frames, SleepFrame(0.05), VADUserStoppedSpeakingFrame(stop_secs=0.2)]


async def turns(*frames, **speech):
    """Return the turns started, the cuts and each turn's words for ``frames`` through Pipecat's user aggregator."""
    start = SmartTurn(SpeechConfig(**speech)).user_params(welcome_enabled=False).user_turn_strategies.start
    stop = TurnAnalyzerUserTurnStopStrategy(turn_analyzer=Complete())
    params = LLMUserAggregatorParams(user_turn_strategies=UserTurnStrategies(start=start, stop=[stop]))
    out = Stamps()
    user = LLMContextAggregatorPair(LLMContext(), user_params=params).user()
    await run_test(Pipeline([user, out]), frames_to_send=[*frames, SleepFrame(0.3)])
    return len(out.after(UserStartedSpeakingFrame)), len(out.after(InterruptionFrame)), out.words


class TurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_smart_turn_ends_a_wordless_turn_at_the_turn_silence(self):
        timeout = SmartTurn(SpeechConfig()).user_params(welcome_enabled=False).user_turn_stop_timeout
        stop = TurnAnalyzerUserTurnStopStrategy(turn_analyzer=Complete())
        strategies = UserTurnStrategies(stop=[stop])
        params = LLMUserAggregatorParams(user_turn_strategies=strategies, user_turn_stop_timeout=timeout)
        heard, out = Stamps(), Stamps()
        audio = [InputAudioRawFrame(audio=b"\x00" * 640, sample_rate=16000, num_channels=1) for _ in range(5)]
        metadata = STTMetadataFrame(service_name="stt", ttfs_p99_latency=NVIDIA_TTFS_P99)
        frames = [
            metadata,
            VADUserStartedSpeakingFrame(start_secs=0.2),
            *audio,
            VADUserStoppedSpeakingFrame(stop_secs=0.2),
        ]
        user = LLMContextAggregatorPair(LLMContext(), user_params=params).user()
        await run_test(Pipeline([heard, user, out]), frames_to_send=[*frames, SleepFrame(1.4)])
        ends = out.after(UserStoppedSpeakingFrame, heard.after(VADUserStoppedSpeakingFrame)[0])
        self.assertEqual((len(ends), out.words), (1, []))
        self.assertAlmostEqual(ends[0], 1.0, delta=0.15)  # 1.2 s of silence, after the 0.2 s VAD stop

    async def test_while_the_bot_speaks_only_enough_words_start_a_turn(self):
        okay = said(interim("Okay"), final("Okay."))
        two = said(interim("Okay"), interim("Okay wait"), final("Okay, wait a second."))
        short = said(interim("Okay"), interim("Okay wait"), final("Okay wait."))
        three = said(interim("Okay wait"), interim("Okay wait now"), final("Okay, wait now."))
        for name, caller, speech, started in (
            ("a cough", said(), {}, (0, 0, [])),
            ("a one-word backchannel", okay, {}, (0, 0, [])),
            ("two words: the whole turn", two, {}, (1, 1, ["Okay, wait a second."])),
            ("two of three words", short, {"interrupt_min_words": 3}, (0, 0, [])),
            ("three of three words", three, {"interrupt_min_words": 3}, (1, 1, ["Okay, wait now."])),
        ):
            with self.subTest(name):
                self.assertEqual(await turns(BotStartedSpeakingFrame(), *caller, **speech), started)

    async def test_while_the_bot_is_silent_the_first_sound_starts_a_turn(self):
        after_a_line = (BotStartedSpeakingFrame(), BotStoppedSpeakingFrame(), *said())
        for name, frames, started in (
            ("a cough before the bot has spoken", said(), (1, 1, [])),
            ("a cough after the bot's line", after_a_line, (1, 1, [])),
            ("a sound, then a word that joins it", said(interim("Okay"), final("Okay.")), (1, 1, ["Okay."])),
        ):
            with self.subTest(name):
                self.assertEqual(await turns(*frames), started)

    def test_the_turns_set_the_vad_and_mute_only_for_a_welcome(self):
        smart_turn = SmartTurn(SpeechConfig(vad_start_seconds=0.16, interrupt_min_words=3))
        params = smart_turn.user_params(welcome_enabled=False)
        self.assertEqual((params.vad_analyzer.params.start_secs, params.vad_analyzer.params.stop_secs), (0.16, 0.2))
        sound, words = params.user_turn_strategies.start
        self.assertIsInstance(sound, VADWhileBotSilentUserTurnStartStrategy)
        self.assertIsInstance(words, MinWordsUserTurnStartStrategy)
        self.assertIs(params.user_turn_strategies_params.start_mode, StrategyMode.FIRST)
        (stop,) = params.user_turn_strategies.stop
        self.assertEqual(stop._turn_analyzer.params.stop_secs, 1.0)
        self.assertEqual(params.user_turn_stop_timeout, 1.0)
        self.assertEqual(smart_turn.describe()["interrupt_min_words"], 3)
        self.assertEqual(params.user_mute_strategies, [])
        (mute,) = smart_turn.user_params(welcome_enabled=True).user_mute_strategies
        self.assertIsInstance(mute, MuteUntilFirstBotCompleteUserMuteStrategy)
