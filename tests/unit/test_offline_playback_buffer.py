# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the live sender's partial-buffer lifecycle across provider stalls."""

import unittest

from pipecat.frames.frames import MetricsFrame, OutputAudioRawFrame, TTSAudioRawFrame, TTSStoppedFrame, TTSTextFrame

from examples.generic.offline.playback import pack_frames
from examples.generic.offline.tts_trace import CapturedTTSFrame, TTSResponseTrace


def pcm(count, value):
    """Give old and new synthesis PCM distinguishable sample values."""
    return value.to_bytes(2, "little", signed=True) * count


class PlaybackBufferTests(unittest.IsolatedAsyncioTestCase):
    """A completed audio write starts the sender's queue-wait deadline."""

    async def packed(self, frames, *, rate=16000):
        """Run native-rate buffering with actual provider frame types."""
        trace = TTSResponseTrace("", (), frames[-1].ready_offset_ns, frames=tuple(frames))
        return await pack_frames(trace, 1, sample_rate=rate)

    def audio(self, frames):
        """Read only audio emitted by the live MediaSender buffer."""
        return [item for item in frames if isinstance(item.frame, OutputAudioRawFrame)]

    async def test_starvation_discards_partial_audio_before_next_provider_chunk(self):
        """The previous 40 samples must not prefix a later speaking episode."""
        frames = await self.packed(
            [
                CapturedTTSFrame(TTSAudioRawFrame(pcm(200, 1), 16000, 1), 0),
                CapturedTTSFrame(TTSAudioRawFrame(pcm(120, 2), 16000, 1), 3_200_000_000),
                CapturedTTSFrame(TTSStoppedFrame(), 3_210_000_000),
            ]
        )
        audio = self.audio(frames)
        self.assertEqual([item.frame.audio for item in audio], [pcm(160, 1), pcm(120, 2) + pcm(40, 0)])
        self.assertEqual([item.ready_offset_ns for item in audio], [0, 3_210_000_000])

    async def test_initial_partial_audio_does_not_start_a_starvation_timer(self):
        """No playback has started, so an arbitrary initial pause drops nothing."""
        frames = await self.packed(
            [
                CapturedTTSFrame(TTSAudioRawFrame(pcm(40, 1), 16000, 1), 0),
                CapturedTTSFrame(TTSAudioRawFrame(pcm(120, 2), 16000, 1), 3_200_000_000),
                CapturedTTSFrame(TTSStoppedFrame(), 3_210_000_000),
            ]
        )
        self.assertEqual([item.frame.audio for item in self.audio(frames)], [pcm(40, 1) + pcm(120, 2)])

    async def test_tts_stop_flushes_and_pads_partial_chunk_before_queued_stop(self):
        """Explicit synthesis completion flushes the partial before its stop."""
        frames = await self.packed(
            [
                CapturedTTSFrame(TTSAudioRawFrame(pcm(200, 1), 16000, 1), 0),
                CapturedTTSFrame(TTSStoppedFrame(), 1_000_000),
            ]
        )
        self.assertEqual([item.frame.audio for item in self.audio(frames)], [pcm(160, 1), pcm(40, 1) + pcm(120, 0)])
        self.assertIsInstance(frames[-1].frame, TTSStoppedFrame)
        # Readiness remains provider enqueue time; the session serializes these
        # through cumulative output sample pointers rather than rounding each
        # native chunk's duration independently.
        self.assertEqual([item.ready_offset_ns for item in frames], [0, 1_000_000, 1_000_000])

    async def test_first_ws_write_returns_immediately_and_starts_queue_wait(self):
        """Native first-write sleep is zero: the 3 s timeout drops old partial."""
        frames = await self.packed(
            [
                CapturedTTSFrame(TTSAudioRawFrame(pcm(200, 1), 16000, 1), 0),
                CapturedTTSFrame(TTSAudioRawFrame(pcm(120, 2), 16000, 1), 3_005_000_000),
                CapturedTTSFrame(TTSStoppedFrame(), 3_006_000_000),
            ]
        )
        self.assertEqual([item.frame.audio for item in self.audio(frames)], [pcm(160, 1), pcm(120, 2) + pcm(40, 0)])

    async def test_text_queue_item_renews_wait_without_flushing_partial_audio(self):
        """The live FIFO's wait is for any frame, including synthesized text."""
        frames = await self.packed(
            [
                CapturedTTSFrame(TTSAudioRawFrame(pcm(200, 1), 16000, 1), 0),
                CapturedTTSFrame(TTSTextFrame("First", "sentence"), 2_000_000_000),
                CapturedTTSFrame(TTSAudioRawFrame(pcm(120, 2), 16000, 1), 3_500_000_000),
                CapturedTTSFrame(TTSStoppedFrame(), 3_510_000_000),
            ]
        )
        self.assertEqual([item.frame.audio for item in self.audio(frames)], [pcm(160, 1), pcm(40, 1) + pcm(120, 2)])

    async def test_starvation_reset_uses_native_output_chunk_size(self):
        """22050 Hz uses Pipecat's integer 220-sample 10 ms chunk size."""
        frames = await self.packed(
            [
                CapturedTTSFrame(TTSAudioRawFrame(pcm(320, 1), 22050, 1), 0),
                CapturedTTSFrame(TTSAudioRawFrame(pcm(120, 2), 22050, 1), 3_200_000_000),
                CapturedTTSFrame(TTSStoppedFrame(), 3_210_000_000),
            ],
            rate=22050,
        )
        self.assertEqual([item.frame.audio for item in self.audio(frames)], [pcm(220, 1), pcm(120, 2) + pcm(100, 0)])

    async def test_metrics_bypass_media_fifo_and_do_not_renew_starvation_wait(self):
        """System metrics do not postpone the live sender's partial reset."""
        frames = await self.packed(
            [
                CapturedTTSFrame(TTSAudioRawFrame(pcm(200, 1), 16000, 1), 0),
                CapturedTTSFrame(MetricsFrame(data=[]), 2_000_000_000),
                CapturedTTSFrame(TTSAudioRawFrame(pcm(120, 2), 16000, 1), 3_200_000_000),
                CapturedTTSFrame(TTSStoppedFrame(), 3_210_000_000),
            ]
        )
        self.assertEqual([item.frame.audio for item in self.audio(frames)], [pcm(160, 1), pcm(120, 2) + pcm(40, 0)])
        self.assertFalse(any(isinstance(item.frame, MetricsFrame) for item in frames))


if __name__ == "__main__":
    unittest.main()
