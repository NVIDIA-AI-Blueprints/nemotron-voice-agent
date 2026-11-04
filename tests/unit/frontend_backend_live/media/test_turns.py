# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The turn ends: their strategies in Pipecat's own user aggregator, and the speech-to-text each builds.

Local only: no speech-to-text connects. A turn's words are read from each context frame as it passes: Pipecat's context
is one object, which later turns change.
"""

import time
import unittest
from unittest.mock import patch

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
from pipecat.services.nvidia.stt import NvidiaSTTService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_mute import MuteUntilFirstBotCompleteUserMuteStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy, TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from examples.frontend_backend_live.config_manager.schema import SpeechConfig
from examples.frontend_backend_live.media import turns
from examples.frontend_backend_live.media.services import build_stt
from examples.frontend_backend_live.media.turns import TURN_ENDS, Shared, Silence, SmartTurn, ttfs_after

SILENCE = b"\x00" * 640  # 20 ms at 16 kHz
BUDGET = (0.7, 1.1)  # the final's 0.8 s after the VAD stop, with the event loop's slack


class Stamps(FrameProcessor):
    """Notes when each frame passes, and the newest message of each context frame as that frame passes."""

    def __init__(self):
        super().__init__()
        self.seen = []
        self.messages = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.seen.append((time.monotonic(), frame))
        if isinstance(frame, LLMContextFrame):
            message = frame.context.get_messages()[-1]
            self.messages.append((message["role"], message["content"]))
        await self.push_frame(frame, direction)

    def times(self, cls):
        return [t for t, f in self.seen if isinstance(f, cls)]


class Run:
    """One run of the user aggregator: the frames that reached it (``heard``), and what it passed on (``out``)."""

    def __init__(self, heard, out, up):
        self.heard, self.out, self.up = heard, out, up

    @property
    def ends(self):
        return self.out.times(UserStoppedSpeakingFrame)

    @property
    def finals(self):
        return self.heard.times(TranscriptionFrame)

    @property
    def words(self):
        """Each caller turn's words, as its context frame carried them."""
        return [content for role, content in self.out.messages if role == "user"]

    def after_the_pause(self, moment):
        return moment - self.heard.times(VADUserStoppedSpeakingFrame)[0]


def user_aggregator(stop_strategy, **params):
    params = LLMUserAggregatorParams(user_turn_strategies=UserTurnStrategies(stop=[stop_strategy]), **params)
    return LLMContextAggregatorPair(LLMContext(), user_params=params).user()


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


def speech(frames=5):
    return [InputAudioRawFrame(audio=SILENCE, sample_rate=16000, num_channels=1) for _ in range(frames)]


def start(stop=0.2):
    """The speech-to-text's budget for a VAD stop of ``stop``, as its turn end sets it, then the caller's speech."""
    return [
        STTMetadataFrame(service_name="stt", ttfs_p99_latency=ttfs_after(stop)),
        VADUserStartedSpeakingFrame(start_secs=0.2),
    ]


def pause(stop=0.2):
    return VADUserStoppedSpeakingFrame(stop_secs=stop)


class TurnCase(unittest.IsolatedAsyncioTestCase):
    async def run_turns(self, stop_strategy, frames, **params):
        heard, out = Stamps(), Stamps()
        _, up = await run_test(Pipeline([heard, user_aggregator(stop_strategy, **params), out]), frames_to_send=frames)
        return Run(heard, out, up)

    def assert_one_turn_on_its_final(self, run, words):
        """Exactly one turn ended, after the speech-to-text's last final, and carried ``words``."""
        self.assertEqual(len(run.ends), 1)
        self.assertGreater(run.ends[0], run.finals[-1])
        self.assertEqual(run.words, [words])

    def assert_within_the_budget(self, seconds):
        self.assertGreater(seconds, BUDGET[0])
        self.assertLess(seconds, BUDGET[1])


class SmartTurnTurns(TurnCase):
    async def test_a_complete_turn_with_no_words_ends_at_the_turn_silence_not_at_five_seconds(self):
        params = SmartTurn(SpeechConfig()).user_params(welcome_enabled=False)
        frames = [*start(), *speech(), pause(), SleepFrame(1.4)]
        stop = TurnAnalyzerUserTurnStopStrategy(turn_analyzer=Complete())
        run = await self.run_turns(stop, frames, user_turn_stop_timeout=params.user_turn_stop_timeout)
        self.assertEqual((len(run.ends), run.words), (1, []))
        self.assertAlmostEqual(run.after_the_pause(run.ends[0]), 1.0, delta=0.15)  # 1.2 s of silence, after 0.2 s


class SilenceTurns(TurnCase):
    def strategy(self):
        silence = Silence(SpeechConfig(turn_end="silence", turn_silence_seconds=0.8))
        return silence.user_params(welcome_enabled=False).user_turn_strategies.stop[0]

    async def test_a_final_after_the_pause_ends_the_turn_at_once(self):
        frames = [*start(0.8), *speech(), pause(0.8), SleepFrame(0.1), final("yes"), SleepFrame(0.1)]
        run = await self.run_turns(self.strategy(), frames)
        self.assert_one_turn_on_its_final(run, "yes")
        self.assertLess(run.after_the_pause(run.ends[0]), 0.5)

    async def test_a_turn_with_no_words_ends_at_the_final_budget_not_at_five_seconds(self):
        frames = [*start(0.8), *speech(), pause(0.8), SleepFrame(1.2)]
        run = await self.run_turns(self.strategy(), frames)
        self.assertEqual((len(run.ends), run.words), (1, []))
        self.assert_within_the_budget(run.after_the_pause(run.ends[0]))

    async def test_a_final_later_than_its_budget_starts_a_turn_of_its_own(self):
        frames = [*start(0.8), *speech(), interim("yes"), pause(0.8), SleepFrame(1.0), final("yes"), SleepFrame(0.5)]
        run = await self.run_turns(self.strategy(), frames)
        self.assertEqual(len(run.ends), 2)
        self.assertTrue(run.ends[0] < run.finals[0] < run.ends[1])
        self.assertEqual(run.words, ["yes"])


class TurnEnds(unittest.TestCase):
    def assert_mutes_only_for_a_welcome(self, end):
        """The mute keeps the caller's audio out until a welcome has played, and there is none without one."""
        (mute,) = end.user_params(welcome_enabled=True).user_mute_strategies
        self.assertIsInstance(mute, MuteUntilFirstBotCompleteUserMuteStrategy)
        self.assertEqual(end.user_params(welcome_enabled=False).user_mute_strategies, [])

    def test_smart_turn_judges_the_configured_pause_with_pipecats_strategy_and_speech_to_text(self):
        end = SmartTurn(SpeechConfig())
        stt = end.stt(use_ssl=False)
        self.assertIs(type(stt), NvidiaSTTService)
        self.assertAlmostEqual(stt._ttfs_p99_latency, 1.0)  # the final's 0.8 s budget, after the 0.2 s stop
        params = end.user_params(welcome_enabled=False)
        self.assertEqual(params.vad_analyzer.params.stop_secs, 0.2)
        (stop,) = params.user_turn_strategies.stop
        self.assertIs(type(stop), TurnAnalyzerUserTurnStopStrategy)
        self.assertAlmostEqual(stop._turn_analyzer.params.stop_secs, 1.0)  # 1.2 s of silence ends any turn
        self.assertEqual(
            end.describe(), {"vad_start_seconds": 0.12, "vad_stop_seconds": 0.2, "turn_silence_seconds": 1.2}
        )
        self.assert_mutes_only_for_a_welcome(end)

    def test_build_stt_finalizes_after_320_ms_of_silence_under_every_turn_end(self):
        for name, end in TURN_ENDS.items():
            with self.subTest(name):
                stt = build_stt({}, {"server": "nemotron-asr:50051"}, end(SpeechConfig(turn_end=name)))
                self.assertEqual((type(stt), stt._stop_history), (NvidiaSTTService, 320))

    def test_the_entries_with_their_own_vad_start_it_after_the_configured_speech_and_shared_keeps_its_own(self):
        for name in ("smart_turn", "silence"):
            with self.subTest(name):
                end = TURN_ENDS[name](SpeechConfig(turn_end=name, vad_start_seconds=0.16))
                self.assertEqual(end.user_params(welcome_enabled=False).vad_analyzer.params.start_secs, 0.16)
                self.assertEqual(end.describe()["vad_start_seconds"], 0.16)
        shared = Shared(SpeechConfig(turn_end="shared", vad_start_seconds=0.16)).user_params(welcome_enabled=False)
        self.assertEqual(shared.vad_analyzer.params.start_secs, VAD_START_SECS)  # the shared pipeline's VAD, as built

    def test_the_vad_start_may_be_as_long_as_the_vad_stop_it_is_paired_with_but_no_longer(self):
        self.assertEqual(SmartTurn(SpeechConfig(vad_start_seconds=0.2)).vad_start, 0.2)  # Pipecat's own pairing
        with self.assertRaisesRegex(ValueError, "vad_start_seconds must not be above speech.vad_stop_seconds"):
            SmartTurn(SpeechConfig(vad_start_seconds=0.25))
        silence = {"turn_end": "silence", "turn_silence_seconds": 0.8}
        self.assertEqual(Silence(SpeechConfig(**silence, vad_start_seconds=0.8)).start, 0.8)
        with self.assertRaisesRegex(ValueError, "vad_start_seconds must not be above speech.turn_silence_seconds"):
            Silence(SpeechConfig(**silence, vad_start_seconds=0.9))

    def test_silence_ends_the_turn_at_its_pause_with_no_smart_turn(self):
        end = Silence(SpeechConfig(turn_end="silence", turn_silence_seconds=0.8))
        stt = end.stt(use_ssl=False)
        self.assertIs(type(stt), NvidiaSTTService)
        self.assertAlmostEqual(stt._ttfs_p99_latency, 1.6)
        params = end.user_params(welcome_enabled=False)
        self.assertEqual(params.vad_analyzer.params.stop_secs, 0.8)
        (stop,) = params.user_turn_strategies.stop
        self.assertIsInstance(stop, SpeechTimeoutUserTurnStopStrategy)
        self.assertFalse(stop.wait_for_transcript)
        self.assertEqual(
            end.describe(),
            {"vad_start_seconds": 0.12, "vad_stop_seconds": 0.8, "turn_silence_seconds": 0.8},  # its own VAD stop
        )
        self.assert_mutes_only_for_a_welcome(end)

    def test_shared_is_nvas_shared_pipeline(self):
        end = Shared(SpeechConfig(turn_end="shared"))
        self.assertIs(type(end.stt(use_ssl=False)), NvidiaSTTService)
        with patch.object(turns, "build_user_aggregator_params", return_value="shared params") as shared:
            self.assertEqual(end.user_params(welcome_enabled=True), "shared params")
        shared.assert_called_once_with(True)
        self.assertEqual(end.describe(), {})  # the shared pipeline's environment switches set its seconds


if __name__ == "__main__":
    unittest.main()
