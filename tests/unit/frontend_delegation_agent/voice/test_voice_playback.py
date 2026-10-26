# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""The playout cursor, the heard prefix and the streaming resampler (core playback and audio)."""

from __future__ import annotations

import unittest

from examples.frontend_delegation_agent.core.audio.pcm import tone
from examples.frontend_delegation_agent.core.audio.resample import StreamResampler
from examples.frontend_delegation_agent.core.engine.playback import ResponseProgress, SentenceSpan, heard_prefix


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
