# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The Pipecat adapter's input-audio clock (``adapters/pipecat/timing.py``, S1, B1, B4).

Ported from v1's ``test_fba_voice_audio_clock`` segmenter cases: the clock counts input audio
only, so wall-clock gaps between appends (Tau pauses its input between ticks) change nothing.
"""

from __future__ import annotations

import unittest

from pipecat.frames.frames import InputAudioRawFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.tests.utils import SleepFrame, run_test

from examples.frontend_backend_agent.adapters.pipecat.timing import STEP_KEY, InputCursor


def _audio(ms: float, *, step: float | None = None, rate: int = 16000) -> InputAudioRawFrame:
    frame = InputAudioRawFrame(audio=b"\x00\x00" * int(rate * ms / 1000), sample_rate=rate, num_channels=1)
    if step is not None:
        frame.metadata[STEP_KEY] = step
    return frame


class InputCursorTests(unittest.IsolatedAsyncioTestCase):
    async def test_steps_follow_the_append_metadata(self) -> None:
        cursor = InputCursor()
        steps: list[float] = []
        cursor.on_advance = steps.append
        # A 20 ms mu-law append resampled to 16 kHz may carry one sample more or less; the
        # clock follows v1's step (client samples at the client rate), not the resampled length.
        await run_test(cursor, frames_to_send=[_audio(20.0625, step=20.0), _audio(19.9375, step=20.0)])
        self.assertEqual(steps, [20.0, 20.0])
        self.assertEqual(cursor.now_ms, 40.0)

    async def test_frames_without_a_step_count_their_samples(self) -> None:
        cursor = InputCursor()
        await run_test(cursor, frames_to_send=[_audio(30.0)])
        self.assertAlmostEqual(cursor.now_ms, 30.0)

    async def test_onset_and_offset_are_back_dated_by_the_vad_delays(self) -> None:
        cursor = InputCursor()
        frames = [_audio(20, step=20) for _ in range(30)]  # 600 ms
        frames.insert(15, VADUserStartedSpeakingFrame(start_secs=0.2))  # confirmed at 300 ms
        frames.append(VADUserStoppedSpeakingFrame(stop_secs=0.5))  # confirmed at 600 ms
        await run_test(cursor, frames_to_send=frames)
        self.assertEqual(cursor.take_onset(), (100.0, "vad"))
        self.assertEqual(cursor.take_offset(), 100.0)
        # Taken once: a turn started by a transcript has no VAD onset.
        self.assertEqual(cursor.take_onset(), (600.0, "transcription"))
        self.assertEqual(cursor.take_offset(), 600.0)

    async def test_wall_clock_gaps_do_not_matter(self) -> None:
        async def run(gap_s: float) -> tuple[float, float]:
            cursor = InputCursor()
            frames: list = []
            for index in range(20):
                frames.append(_audio(20, step=20))
                if index == 4 and gap_s:
                    frames.append(SleepFrame(sleep=gap_s))  # the client pauses its input
                if index == 9:
                    frames.append(VADUserStartedSpeakingFrame(start_secs=0.1))
            frames.append(VADUserStoppedSpeakingFrame(stop_secs=0.2))
            await run_test(cursor, frames_to_send=frames)
            onset, _ = cursor.take_onset()
            return onset, cursor.take_offset()

        self.assertEqual(await run(0.0), (100.0, 200.0))
        self.assertEqual(await run(0.3), (100.0, 200.0))


if __name__ == "__main__":
    unittest.main()
