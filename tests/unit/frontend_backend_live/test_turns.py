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
from examples.frontend_backend_live.media.turns import (
    SmartTurn,
    SpellingHoldUserTurnStopStrategy,
    VADWhileBotSilentUserTurnStartStrategy,
)


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


def spoken(stop, *before_the_pause):
    """Return the caller's speech with Pipecat's NVIDIA budget for the final, then a pause: a VAD stop of ``stop``."""
    audio = [InputAudioRawFrame(audio=b"\x00" * 640, sample_rate=16000, num_channels=1) for _ in range(5)]  # 100 ms
    metadata = STTMetadataFrame(service_name="stt", ttfs_p99_latency=NVIDIA_TTFS_P99)
    started, pause = VADUserStartedSpeakingFrame(start_secs=0.2), VADUserStoppedSpeakingFrame(stop_secs=stop)
    return [metadata, started, *audio, *before_the_pause, pause]


class SpellingHoldTests(unittest.IsolatedAsyncioTestCase):
    """A turn whose words end in single letters or digits in a row is ended by the silence, not Smart Turn's verdict."""

    TIMEOUT = SmartTurn(SpeechConfig()).fallback  # the user turn controller's stop timeout, 1.0 s by default

    async def turns(self, stop_strategy, frames, **params):
        """Return each turn's end and each final after the pause, and the turns' words, through Pipecat's aggregator."""
        heard, out = Stamps(), Stamps()
        params = LLMUserAggregatorParams(user_turn_strategies=UserTurnStrategies(stop=[stop_strategy]), **params)
        await run_test(
            Pipeline([heard, LLMContextAggregatorPair(LLMContext(), user_params=params).user(), out]),
            frames_to_send=frames,
        )
        pause = heard.after(VADUserStoppedSpeakingFrame)[0]
        return out.after(UserStoppedSpeakingFrame, pause), heard.after(TranscriptionFrame, pause), out.words

    def hold(self, observed=None):
        observe = (
            (lambda kind, **fields: observed.append((kind, fields))) if observed is not None else (lambda *a, **k: None)
        )
        return SpellingHoldUserTurnStopStrategy(turn_analyzer=Complete(), observe=observe)

    async def ends(self, *before_the_pause, wait=1.4, after=()):
        frames = [*spoken(0.2, *before_the_pause), *after, SleepFrame(wait)]
        return await self.turns(self.hold(), frames, user_turn_stop_timeout=self.TIMEOUT)

    async def test_a_turn_ending_in_a_spelled_run_ends_at_the_stop_timeout(self):
        ends, _, words = await self.ends(final("my name is J O H N"))
        self.assertEqual(len(ends), 1)
        self.assertAlmostEqual(ends[0], self.TIMEOUT, delta=0.2)

    async def test_a_turn_ending_in_ordinary_words_ends_at_smart_turns_verdict(self):
        ends, _, _ = await self.ends(final("my name is John"))
        self.assertEqual(len(ends), 1)
        self.assertLess(ends[0], 0.3)

    async def test_a_french_turn_that_ends_in_a_word_is_not_held(self):
        ends, _, _ = await self.ends(final("il y a un problème"))
        self.assertLess(ends[0], 0.3)

    async def test_speech_before_the_timeout_continues_the_same_turn(self):
        more = [
            SleepFrame(0.4),
            VADUserStartedSpeakingFrame(start_secs=0.12),
            final("N"),
            VADUserStoppedSpeakingFrame(stop_secs=0.2),
        ]
        ends, _, words = await self.ends(final("J O H"), after=more, wait=1.6)
        self.assertEqual(len(ends), 1)  # one turn, with every letter
        self.assertEqual(" ".join(words), "J O H N")

    async def test_pipecat_reports_every_turn_end_through_report_user_turn_stopped(self):
        # The hold overrides Pipecat's public hook; the controller's stop timeout ends a turn the strategy holds open.
        from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
        from pipecat.turns.user_turn_controller import UserTurnController

        self.assertTrue(hasattr(BaseUserTurnStopStrategy, "report_user_turn_stopped"))
        self.assertTrue(hasattr(UserTurnController, "_trigger_user_turn_stop"))
        reported = []

        class Counting(TurnAnalyzerUserTurnStopStrategy):
            async def report_user_turn_stopped(self, **kwargs):
                reported.append(kwargs)
                await super().report_user_turn_stopped(**kwargs)

        ends, _, _ = await self.turns(
            Counting(turn_analyzer=Complete()), [*spoken(0.2, final("ok then")), SleepFrame(0.5)]
        )
        self.assertEqual((len(ends), len(reported)), (1, 1))

    async def test_the_overrides_keep_pipecats_turn_reset(self):
        hold = self.hold()
        hold._words, hold._text = ["J O"], "J O"
        await hold.handle_user_turn_started()
        self.assertEqual((hold._words, hold._text), ([], ""))
        hold._words, hold._text = ["J O"], "J O"
        await hold.handle_user_turn_stopped()
        self.assertEqual((hold._words, hold._text), ([], ""))

    def test_with_the_hold_off_the_strategy_is_pipecats(self):
        for on, kind in ((True, SpellingHoldUserTurnStopStrategy), (False, TurnAnalyzerUserTurnStopStrategy)):
            turns = SmartTurn(SpeechConfig(spelling_hold=on))
            (stop,) = turns.user_params(welcome_enabled=False).user_turn_strategies.stop
            self.assertIs(type(stop), kind)
            self.assertEqual(turns.describe()["spelling_hold"], on)
