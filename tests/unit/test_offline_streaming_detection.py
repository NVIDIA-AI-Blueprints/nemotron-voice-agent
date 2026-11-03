# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the real native VAD state machine while upload is still incomplete."""

import asyncio
import unittest

from test_offline_detection import MarkerVAD, pcm

from examples.generic.offline.detection import AudioOffsetDetector
from examples.generic.offline.models import AudioSource, SourceAbortedError, StreamingAudioSource


class StreamingDetectionTests(unittest.IsolatedAsyncioTestCase):
    """Prefix processing preserves native source thresholds and explicit EOF."""

    async def test_detector_processes_full_prefix_windows_before_seal(self):
        """Slow fragmented upload changes availability, not acoustic decisions."""
        audio = pcm(1024, True) + pcm(1536 + 17, False)
        source = StreamingAudioSource(len(audio) // 2)
        vad = MarkerVAD(start_windows=1, stop_windows=1)
        checkpoints = asyncio.Queue()

        async def checkpoint(events, ready_ns, chunk):
            if chunk is not None:
                checkpoints.put_nowait(chunk.end_sample)

        task = asyncio.create_task(AudioOffsetDetector(vad_analyzer=vad).detect(source, on_checkpoint=checkpoint))
        source.append(audio[:1234])
        self.assertEqual(await asyncio.wait_for(checkpoints.get(), 1), 512)
        source.append(audio[1234:4096])
        for expected in (1024, 1536, 2048):
            self.assertEqual(await asyncio.wait_for(checkpoints.get(), 1), expected)
        self.assertFalse(task.done())
        self.assertFalse(vad.cleaned)
        source.append(audio[4096:])
        self.assertEqual(await asyncio.wait_for(checkpoints.get(), 1), 2560)
        self.assertFalse(task.done())
        source.seal()
        trace = await asyncio.wait_for(task, 1)
        reference = await AudioOffsetDetector(vad_analyzer=MarkerVAD(start_windows=1, stop_windows=1)).detect(
            AudioSource(audio)
        )
        self.assertEqual(trace.speech, reference.speech)
        self.assertEqual(
            [(v.start_sample, v.end_sample, v.endpoint_sample) for v in trace.turns],
            [(v.start_sample, v.end_sample, v.endpoint_sample) for v in reference.turns],
        )
        self.assertEqual(set(vad.inference_lengths), {512})
        self.assertTrue(vad.cleaned)
        self.assertEqual(source.pcm, audio)

    async def test_aborted_prefix_cleans_native_executor_without_source_eof(self):
        """An interrupted upload cannot return a completed detection trace."""
        source = StreamingAudioSource(1024)
        vad = MarkerVAD()
        task = asyncio.create_task(AudioOffsetDetector(vad_analyzer=vad).detect(source))
        await asyncio.sleep(0)
        source.abort()
        with self.assertRaises(SourceAbortedError):
            await asyncio.wait_for(task, 1)
        self.assertTrue(vad.cleaned)
