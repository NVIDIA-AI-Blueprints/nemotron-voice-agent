# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Rules 3, 4, 11b: segmentation on the audio clock, and the playout cursor."""

from __future__ import annotations

import asyncio
import unittest

from _fbv_voice_fakes import PCMU, SessionHarness, append_event, pcmu_silence, pcmu_speech, tau2_session_update

from examples.frontend_backend_verdict.voice.agent.scripted import ScriptedAgentPort
from examples.frontend_backend_verdict.voice.audio.pcm import tone
from examples.frontend_backend_verdict.voice.audio.resample import StreamResampler
from examples.frontend_backend_verdict.voice.engine.input_path import InputPath
from examples.frontend_backend_verdict.voice.engine.playback import ResponseProgress, SentenceSpan, heard_prefix
from examples.frontend_backend_verdict.voice.engine.segmenter import (
    SegmenterParams,
    SpeechEnd,
    SpeechStart,
    UtteranceSegmenter,
)
from examples.frontend_backend_verdict.voice.speech.vad_energy import EnergyVad
from examples.frontend_backend_verdict.voice.wire.session_view import TurnDetectionSettings

TURN = TurnDetectionSettings(mode="server_vad", threshold=0.5, prefix_padding_ms=300, silence_duration_ms=500)


def _input_path() -> InputPath:
    return InputPath(engine_rate=16000, vad=EnergyVad(16000), input_format=PCMU, turn_detection=TURN, min_speech_ms=120)


def _stream(path: InputPath, audio: bytes) -> list:
    events = []
    for offset in range(0, len(audio), 160):
        events.extend(path.feed(audio[offset : offset + 160]).events)
    return events


class SegmenterTests(unittest.TestCase):
    def test_audio_start_ms_is_on_the_cumulative_input_clock(self) -> None:
        path = _input_path()
        audio = pcmu_silence(3000) + pcmu_speech(800) + pcmu_silence(1000) + pcmu_speech(600) + pcmu_silence(1000)
        events = _stream(path, audio)
        starts = [e.start_ms for e in events if isinstance(e, SpeechStart)]
        ends = [e.end_ms for e in events if isinstance(e, SpeechEnd)]
        self.assertEqual(len(starts), 2)
        self.assertAlmostEqual(starts[0], 3000, delta=25)
        self.assertAlmostEqual(ends[0], 3800, delta=25)
        self.assertAlmostEqual(starts[1], 4800, delta=25)
        self.assertAlmostEqual(path.clock.now_ms, 6400, delta=0.01)

    def test_prefix_padding_and_min_speech(self) -> None:
        params = SegmenterParams(
            rate=16000, frame_samples=320, prefix_padding_ms=100, silence_duration_ms=100, min_speech_ms=60
        )
        seg = UtteranceSegmenter(params)
        frame = b"\x01\x00" * 320
        for _ in range(10):
            self.assertEqual(seg.process(frame, 0.0), [])
        self.assertEqual(seg.process(frame, 1.0), [])  # 20 ms: pending
        self.assertEqual(seg.process(frame, 0.0), [])  # burst shorter than min_speech: noise, no event
        for _ in range(2):
            seg.process(frame, 1.0)
        start = seg.process(frame, 1.0)[0]
        self.assertIsInstance(start, SpeechStart)
        self.assertEqual(start.start_ms, 12 * 20)
        self.assertEqual(len(start.audio), (5 + 3) * len(frame))  # 100 ms prefix + the confirmed run

    def test_wall_clock_gaps_do_not_matter(self) -> None:
        audio = pcmu_silence(1000) + pcmu_speech(700) + pcmu_silence(900) + pcmu_speech(500) + pcmu_silence(900)

        async def run(gap_every: int, gap_s: float) -> list[tuple[str, int]]:
            harness = SessionHarness(agent_factory=lambda _: ScriptedAgentPort([{"say": "ok."}]))
            await harness.start(tau2_session_update())
            for index, offset in enumerate(range(0, len(audio), 160)):
                await harness.session.handle_message(append_event(audio[offset : offset + 160]))
                if gap_every and index % gap_every == 0:
                    await asyncio.sleep(gap_s)  # tau2's user simulator blocks its tick loop
            await harness.settle()
            marks = [
                (e["type"], e.get("audio_start_ms", e.get("audio_end_ms")))
                for e in harness.events
                if e["type"] in ("input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped")
            ]
            await harness.close()
            return marks

        async def both() -> tuple[list, list]:
            return await run(0, 0.0), await run(25, 0.15)

        smooth, gappy = asyncio.run(both())
        self.assertEqual(len(smooth), 4)
        self.assertEqual(smooth, gappy)


class PlaybackTests(unittest.TestCase):
    def test_cursor_capped_by_sent_audio(self) -> None:
        progress = ResponseProgress()
        progress.begin_item("item_a", kind="answer", full_text="One. Two.")
        progress.begin_sentence("One.")
        progress.add_audio(400)
        progress.end_sentence()
        progress.advance(300)
        self.assertEqual(progress.heard_ms, 300)
        progress.advance(1000)  # TTS stall: the client's buffer ran dry at 400 ms
        self.assertEqual(progress.heard_ms, 400)
        progress.begin_sentence("Two.")
        progress.add_audio(400)
        progress.advance(100)
        self.assertEqual(progress.heard_ms, 500)  # resumes from delivered audio, not from wall time
        self.assertTrue(progress.active)

    def test_active_while_generating_with_nothing_buffered(self) -> None:
        progress = ResponseProgress()
        progress.generating = True
        self.assertEqual(progress.unplayed_ms, 0)
        self.assertTrue(progress.active)

    def test_heard_prefix_is_proportional_and_word_snapped(self) -> None:
        spans = [SentenceSpan("Your flight is booked.", 0, 1000), SentenceSpan("Anything else today?", 1000, 2000)]
        self.assertEqual(heard_prefix(spans, 1000), "Your flight is booked.")
        self.assertEqual(heard_prefix(spans, 1500), "Your flight is booked. Anything")  # "e" of "else" only
        self.assertEqual(heard_prefix(spans, 1700), "Your flight is booked. Anything else")
        self.assertEqual(heard_prefix(spans, 0), "")
        self.assertEqual(heard_prefix([SentenceSpan("Unsent.", 500, 500)], 900), "")


class ResamplerTests(unittest.TestCase):
    def test_identical_input_gives_identical_output(self) -> None:
        # Regression: libsoxr's int16 path dithered with run-dependent state, and the
        # flushed tail could be read after its stream was replaced.
        for in_rate, out_rate in ((16000, 8000), (8000, 16000), (22050, 24000)):
            pcm = tone(250, in_rate)
            outputs = set()
            for _ in range(50):
                resampler = StreamResampler(in_rate, out_rate)
                outputs.add(resampler.process(pcm[:3000]) + resampler.process(pcm[3000:]) + resampler.flush())
            with self.subTest(in_rate=in_rate, out_rate=out_rate):
                self.assertEqual(len(outputs), 1)
                expected = len(pcm) // 2 * out_rate // in_rate
                self.assertAlmostEqual(len(next(iter(outputs))) // 2, expected, delta=2)
