# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Rules 3, 4, 11b: turns on the input-audio clock, and the playout cursor.

v1's segmenter cases went with v1's host; the input clock itself is tested in
``test_fba_pipecat_timing``. Here the whole Pipecat session runs on the clock.
"""

from __future__ import annotations

import asyncio
import unittest

from _fba_core_fakes import append_event, benchmark_session_update, pcmu_silence, pcmu_speech
from _fba_pipecat_fakes import PipecatHarness, PipecatTestCase

from examples.frontend_backend_agent.core.audio.pcm import tone
from examples.frontend_backend_agent.core.audio.resample import StreamResampler
from examples.frontend_backend_agent.core.engine.playback import ResponseProgress, SentenceSpan, heard_prefix
from examples.frontend_backend_agent.core.engine.scripted import ScriptedAgentPort


class InputClockTests(PipecatTestCase):
    async def test_wall_clock_gaps_do_not_matter(self) -> None:
        """The benchmark pauses its input between ticks; turns are measured in input audio only (eval settings)."""
        audio = pcmu_silence(1000) + pcmu_speech(700) + pcmu_silence(900) + pcmu_speech(500) + pcmu_silence(900)

        async def run(gap_every: int, gap_s: float) -> list[tuple[str, int]]:
            harness = PipecatHarness(agent_factory=lambda _: ScriptedAgentPort([{"say": "ok."}, {"say": "ok."}]))
            await harness.start(benchmark_session_update())
            for index, offset in enumerate(range(0, len(audio), 160)):
                harness.websocket.put_text(append_event(audio[offset : offset + 160]))
                await harness._drain_frame()
                if gap_every and index % gap_every == 0:
                    await asyncio.sleep(gap_s)  # the benchmark's user simulator blocks its tick loop
            await harness.settle()
            marks = [
                (e["type"], e.get("audio_start_ms", e.get("audio_end_ms")))
                for e in harness.events
                if e["type"] in ("input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped")
            ]
            await harness.close()
            return marks

        smooth = await run(0, 0.0)
        gappy = await run(25, 0.15)
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
