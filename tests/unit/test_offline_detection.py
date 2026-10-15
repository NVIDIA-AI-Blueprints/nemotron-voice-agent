# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import asyncio
import os
import struct
import unittest
from unittest.mock import patch

import numpy as np
from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn, SmartTurnParams
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams

from examples.generic.offline.detection import (
    AudioOffsetDetector,
    SampleTimedSileroMixin,
    SampleTimedSmartTurnMixin,
    create_generic_detector,
)
from examples.generic.offline.models import RATE, AudioSource, SpeechInterval

WINDOW = 512


def pcm(samples: int, speech: bool) -> bytes:
    """Use recognizable constant-valued PCM; avoid testing an actual neural model."""
    return struct.pack("<h", 2000 if speech else 0) * samples


class MarkerVAD(VADAnalyzer):
    def __init__(self, *, start_windows=2, stop_windows=3):
        """Reuse Pipecat's state machine with an injected marker classifier."""
        super().__init__(
            sample_rate=RATE,
            params=VADParams(
                start_secs=start_windows * WINDOW / RATE,
                stop_secs=stop_windows * WINDOW / RATE,
                min_volume=0,
            ),
        )
        self.inference_lengths = []
        self.cleaned = False

    def num_frames_required(self):
        return WINDOW

    def voice_confidence(self, buffer):
        self.inference_lengths.append(len(buffer) // 2)
        return float(np.any(np.frombuffer(buffer, dtype=np.int16)))

    def _get_smoothed_volume(self, audio):
        return 1.0

    async def cleanup(self):
        self.cleaned = True
        self._executor.shutdown(wait=True, cancel_futures=True)


class ScriptedSmartTurn(SampleTimedSmartTurnMixin, BaseSmartTurn):
    def __init__(self, predictions=(1,), **kwargs):
        """Reuse buffering and model dispatch with scripted endpoint predictions."""
        self.predictions = iter(predictions)
        self.predicted_audio = []
        super().__init__(sample_rate=RATE, **kwargs)

    def _predict_endpoint(self, audio_array):
        self.predicted_audio.append(audio_array.copy())
        prediction = next(self.predictions)
        return {"prediction": prediction, "probability": float(prediction)}


class StubSileroModel:
    def __init__(self):
        """Count recurrent resets without loading any neural model."""
        self.reset_count = 0

    def __call__(self, audio, sample_rate):
        return np.array([[0.5]])

    def reset_states(self):
        self.reset_count += 1


class StubSileroBase:
    def __init__(self):
        """Supply only the model and rate accessed by the reset mixin."""
        self.sample_rate = RATE
        self._model = StubSileroModel()


class SourceSilero(SampleTimedSileroMixin, StubSileroBase):
    pass


def stable_decisions(trace):
    """Exclude measured computation while comparing source timeline invariance."""
    return (
        trace.speech,
        tuple((turn.turn_id, turn.start_sample, turn.end_sample, turn.endpoint_sample) for turn in trace.turns),
        tuple({key: value for key, value in event.items() if not key.endswith("compute_ns")} for event in trace.events),
    )


class OfflineDetectionTests(unittest.TestCase):
    def detect(self, source, *, start=2, stop=3, turn=None):
        vad = MarkerVAD(start_windows=start, stop_windows=stop)
        trace = asyncio.run(AudioOffsetDetector(vad_analyzer=vad, turn_analyzer=turn).detect(AudioSource(source)))
        self.assertTrue(vad.cleaned)
        self.assertEqual(set(vad.inference_lengths), {WINDOW} if source else set())
        return trace

    def test_source_bounds_include_confirmation_but_exclude_endpoint_silence(self):
        source = pcm(4 * WINDOW, False) + pcm(5 * WINDOW, True) + pcm(8 * WINDOW, False)
        trace = self.detect(source)
        self.assertEqual(trace.speech, (SpeechInterval(4 * WINDOW, 9 * WINDOW, 6 * WINDOW),))
        self.assertEqual(len(trace.turns), 1)
        turn = trace.turns[0]
        self.assertEqual(
            (turn.start_sample, turn.end_sample, turn.endpoint_sample), (4 * WINDOW, 9 * WINDOW, 12 * WINDOW)
        )
        self.assertGreater(turn.detection_compute_ns, 0)

    def test_configured_thresholds_change_confirmation_offsets(self):
        source = pcm(8 * WINDOW, True) + pcm(8 * WINDOW, False)
        for start, stop in ((1, 1), (3, 5), (6, 6), (0, 0)):
            with self.subTest(start=start, stop=stop):
                trace = self.detect(source, start=start, stop=stop)
                self.assertEqual(trace.speech[0].confirmed_start_sample, max(1, start) * WINDOW)
                self.assertEqual(trace.turns[0].endpoint_sample, (8 + max(1, stop)) * WINDOW)

    def test_utterance_shorter_than_start_threshold_does_not_start_turn(self):
        trace = self.detect(pcm(WINDOW, True) + pcm(10 * WINDOW, False), start=2)
        self.assertFalse(trace.speech)
        self.assertFalse(trace.turns)

    def test_source_eof_closes_active_speech_without_fake_silence(self):
        extent = 3 * WINDOW + 17
        trace = self.detect(pcm(extent, True))
        self.assertEqual(trace.speech, (SpeechInterval(0, extent, 2 * WINDOW),))
        self.assertEqual((trace.turns[0].end_sample, trace.turns[0].endpoint_sample), (extent, extent))
        self.assertEqual(trace.events[-2]["reason"], "source_eof")

    def test_partial_eof_padding_cannot_finish_silence_confirmation(self):
        speech_end = 3 * WINDOW
        extent = speech_end + WINDOW + 17
        trace = self.detect(pcm(speech_end, True) + pcm(WINDOW + 17, False), stop=2)
        self.assertEqual(trace.speech[0].end_sample, speech_end)
        self.assertEqual(trace.turns[0].endpoint_sample, extent)
        self.assertEqual(trace.events[-2]["reason"], "source_eof")

    def test_partial_eof_padding_cannot_confirm_too_short_speech(self):
        trace = self.detect(pcm(WINDOW + 17, True), start=2)
        self.assertFalse(trace.speech)
        self.assertFalse(trace.turns)

    def test_speech_resumes_during_stop_count_and_is_one_interval(self):
        source = pcm(3 * WINDOW, True) + pcm(2 * WINDOW, False) + pcm(3 * WINDOW, True) + pcm(3 * WINDOW, False)
        trace = self.detect(source)
        self.assertEqual(trace.speech, (SpeechInterval(0, 8 * WINDOW, 2 * WINDOW),))
        self.assertEqual(len(trace.turns), 1)

    def test_long_silence_and_multiple_turns_keep_source_positions(self):
        gap = 1000 * WINDOW
        source = pcm(gap, False) + pcm(3 * WINDOW, True) + pcm(gap, False) + pcm(3 * WINDOW, True)
        trace = self.detect(source)
        self.assertEqual([speech.start_sample for speech in trace.speech], [gap, gap * 2 + 3 * WINDOW])
        self.assertEqual([turn.turn_id for turn in trace.turns], ["turn-1", "turn-2"])
        self.assertEqual(trace.turns[-1].endpoint_sample, len(source) // 2)

    def test_upload_packetization_cannot_hide_multiple_speech_transitions(self):
        source = pcm(3 * WINDOW, True) + pcm(4 * WINDOW, False) + pcm(3 * WINDOW, True) + pcm(4 * WINDOW, False)
        baseline = stable_decisions(self.detect(source))
        for packet_bytes in (2, 62, 1024, 8192, len(source)):
            uploaded = b"".join(source[pos : pos + packet_bytes] for pos in range(0, len(source), packet_bytes))
            self.assertEqual(stable_decisions(self.detect(uploaded)), baseline)
        self.assertEqual(len(baseline[1]), 2)

    def test_system_clock_changes_have_no_effect_on_detection(self):
        source = pcm(4 * WINDOW, True) + pcm(4 * WINDOW, False)
        expected = stable_decisions(self.detect(source))
        with patch("time.time", side_effect=AssertionError("wall clock used")):
            # The asyncio event loop needs monotonic itself; only source-level
            # components must not depend on it for conversational decisions.
            self.assertEqual(stable_decisions(self.detect(source)), expected)

    def test_empty_and_silence_only_sources_do_not_propose_turns(self):
        for source in (b"", pcm(5000, False)):
            trace = self.detect(source)
            self.assertFalse(trace.speech)
            self.assertFalse(trace.turns)
            self.assertEqual(trace.events[-1]["sample"], len(source) // 2)

    def test_smart_turn_prediction_can_join_separate_speech_intervals(self):
        turn = ScriptedSmartTurn(predictions=(0, 1))
        source = pcm(3 * WINDOW, True) + pcm(4 * WINDOW, False) + pcm(3 * WINDOW, True) + pcm(4 * WINDOW, False)
        trace = self.detect(source, turn=turn)
        self.assertEqual(len(trace.speech), 2)
        self.assertEqual(len(trace.turns), 1)
        self.assertEqual((trace.turns[0].start_sample, trace.turns[0].end_sample), (0, 10 * WINDOW))
        self.assertEqual(trace.turns[0].endpoint_sample, 13 * WINDOW)
        self.assertEqual(len(turn.predicted_audio), 2)

    def test_smart_turn_max_silence_uses_source_samples(self):
        turn = ScriptedSmartTurn(predictions=(0,), params=SmartTurnParams(stop_secs=4 * WINDOW / RATE))
        source = pcm(3 * WINDOW, True) + pcm(20 * WINDOW, False)
        trace = self.detect(source, stop=2, turn=turn)
        # The first silence window for Smart Turn is VAD's confirmation
        # window: source window 5. Its four-window deadline is window 8.
        self.assertEqual(trace.turns[0].endpoint_sample, 8 * WINDOW)
        self.assertEqual(trace.events[-2]["reason"], "smart_turn_silence")

    def test_smart_turn_incomplete_at_eof_is_explicitly_flushed(self):
        turn = ScriptedSmartTurn(predictions=(0,))
        source = pcm(4 * WINDOW, True) + pcm(3 * WINDOW + 13, False)
        trace = self.detect(source, turn=turn)
        self.assertEqual(trace.turns[0].endpoint_sample, len(source) // 2)
        self.assertEqual(trace.events[-2]["reason"], "source_eof")

    def test_detection_failure_cleans_owned_analyzers(self):
        class BrokenVAD(MarkerVAD):
            async def analyze_audio(self, audio):
                raise RuntimeError("model failed")

        vad = BrokenVAD()
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            asyncio.run(AudioOffsetDetector(vad_analyzer=vad).detect(AudioSource(pcm(WINDOW, True))))
        self.assertTrue(vad.cleaned)


class SmartTurnSampleBufferTests(unittest.TestCase):
    def test_cleanup_handles_an_unused_lazy_executor(self):
        turn = ScriptedSmartTurn()
        self.assertIsNone(turn._executor)
        asyncio.run(turn.cleanup())
        self.assertIsNone(turn._executor)

    def test_cleanup_releases_a_used_lazy_executor(self):
        turn = ScriptedSmartTurn()
        turn.set_sample_rate(RATE)
        turn.append_audio(pcm(WINDOW, True), True)
        asyncio.run(turn.analyze_end_of_turn())
        self.assertIsNotNone(turn._executor)
        asyncio.run(turn.cleanup())
        self.assertIsNone(turn._executor)

    def test_prespeech_selector_clips_first_chunk_at_exact_sample(self):
        turn = ScriptedSmartTurn(params=SmartTurnParams(pre_speech_ms=10))
        turn.set_sample_rate(RATE)
        turn.append_audio(pcm(WINDOW, False), False)
        turn.append_audio(pcm(WINDOW, True), True)
        asyncio.run(turn.analyze_end_of_turn())
        self.assertEqual(len(turn.predicted_audio[0]), 160 + WINDOW)
        np.testing.assert_array_equal(turn.predicted_audio[0][:160], np.zeros(160))
        asyncio.run(turn.cleanup())

    def test_source_zero_is_a_valid_speech_start(self):
        turn = ScriptedSmartTurn(params=SmartTurnParams(pre_speech_ms=0))
        turn.set_sample_rate(RATE)
        turn.append_audio(pcm(WINDOW, True), True)
        turn.append_audio(pcm(WINDOW, True), True)
        self.assertEqual(turn._offline_speech_start_sample, 0)
        asyncio.run(turn.analyze_end_of_turn())
        self.assertEqual(len(turn.predicted_audio[0]), 2 * WINDOW)
        asyncio.run(turn.cleanup())

    def test_quiet_buffer_is_bounded_by_source_duration(self):
        turn = ScriptedSmartTurn(params=SmartTurnParams(pre_speech_ms=0, stop_secs=1, max_duration_secs=1))
        turn.set_sample_rate(RATE)
        with patch("time.monotonic", side_effect=AssertionError("wall clock used")):
            for _ in range(700):
                turn.append_audio(pcm(WINDOW, False), False)
        self.assertEqual(sum(len(audio) for _, audio in turn._audio_buffer), 2 * RATE)
        self.assertEqual(turn._audio_buffer[0][0], 700 * WINDOW - 2 * RATE)
        asyncio.run(turn.cleanup())

    def test_turn_clear_keeps_absolute_source_cursor(self):
        turn = ScriptedSmartTurn()
        turn.set_sample_rate(RATE)
        turn.append_audio(pcm(WINDOW, True), True)
        turn.clear()
        turn.append_audio(pcm(WINDOW, True), True)
        self.assertEqual(turn._offline_sample_cursor, 2 * WINDOW)
        self.assertEqual(turn._offline_speech_start_sample, WINDOW)
        asyncio.run(turn.cleanup())


class SileroSampleResetTests(unittest.TestCase):
    def test_model_resets_at_first_window_after_each_five_source_seconds(self):
        model = SourceSilero()
        with patch("time.time", side_effect=AssertionError("wall clock used")):
            for position in range(WINDOW, 12 * RATE, WINDOW):
                model.set_source_window_end(position)
                model.voice_confidence(pcm(WINDOW, False))
        self.assertEqual(model.model_reset_offsets, (80384, 160256))
        self.assertEqual(model._model.reset_count, 2)

    def test_inference_padding_does_not_advance_reset_schedule(self):
        model = SourceSilero()
        model.set_source_window_end(5 * RATE - 1)
        model.voice_confidence(pcm(WINDOW, False))
        self.assertFalse(model.model_reset_offsets)
        model.set_source_window_end(5 * RATE)
        model.voice_confidence(pcm(WINDOW, False))
        self.assertEqual(model.model_reset_offsets, (5 * RATE,))

    def test_backwards_source_positions_are_rejected(self):
        model = SourceSilero()
        model.set_source_window_end(1000)
        with self.assertRaisesRegex(ValueError, "monotonic"):
            model.set_source_window_end(999)


class DetectorFactoryTests(unittest.TestCase):
    def configured_detector(self, environment):
        """Exercise real analyzer configuration without loading neural weights."""
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("pipecat.audio.vad.silero.SileroVADAnalyzer.__init__", VADAnalyzer.__init__),
            patch(
                "pipecat.audio.turn.smart_turn.local_smart_turn_v3.LocalSmartTurnAnalyzerV3.__init__",
                BaseSmartTurn.__init__,
            ),
        ):
            detector = create_generic_detector()
        asyncio.run(detector.detect(AudioSource(b"")))
        return detector.configuration

    def test_smart_turn_uses_the_current_shared_fallback(self):
        from examples.shared.pipeline_utils import SMART_TURN_FALLBACK_SECS

        configuration = self.configured_detector({})
        self.assertEqual(configuration["policy"], "smart_turn")
        self.assertEqual(configuration["vad"]["stop_secs"], 0.2)
        self.assertEqual(configuration["smart_turn"]["stop_secs"], SMART_TURN_FALLBACK_SECS)
        self.assertEqual(configuration["smart_turn"]["stop_secs"], 1.0)

    def test_smart_turn_fallback_honors_valid_overrides(self):
        for value in ("0", "0.75", "2.5"):
            with self.subTest(value=value):
                configuration = self.configured_detector({"SMART_TURN_STOP_SECS": value})
                self.assertEqual(configuration["smart_turn"]["stop_secs"], float(value))

    def test_smart_turn_invalid_override_uses_shared_default(self):
        configuration = self.configured_detector({"SMART_TURN_STOP_SECS": "invalid"})
        self.assertEqual(configuration["smart_turn"]["stop_secs"], 1.0)

    def test_smart_turn_negative_override_matches_shared_clamping(self):
        configuration = self.configured_detector({"SMART_TURN_STOP_SECS": "-1"})
        self.assertEqual(configuration["smart_turn"]["stop_secs"], 0.0)

    def test_silero_only_retains_its_own_threshold(self):
        for overrides, expected in (({}, 0.5), ({"SILERO_VAD_STOP_SECS": "0.8"}, 0.8)):
            with self.subTest(overrides=overrides):
                configuration = self.configured_detector({"USE_SILERO_VAD_TURN_DETECTION": "true", **overrides})
                self.assertEqual(configuration["policy"], "silero")
                self.assertEqual(configuration["vad"]["stop_secs"], expected)
                self.assertIsNone(configuration["smart_turn"])
