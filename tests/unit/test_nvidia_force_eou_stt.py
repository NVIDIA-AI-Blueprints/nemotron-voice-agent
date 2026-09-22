# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from pipecat.frames.frames import VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.stt import AudioChunkIterator
from riva.client.proto import riva_asr_pb2 as rasr

from examples.shared.nvidia_force_eou_stt import (
    FORCE_EOU_SILENCE_SECS,
    NvidiaForceEouSTTService,
)


class ForceEouStreamingRequestTests(unittest.TestCase):
    def test_attaches_force_eou_to_injected_silence_not_buffered_speech(self) -> None:
        stt = NvidiaForceEouSTTService(use_ssl=False)
        stt._sample_rate = 16000
        config = rasr.StreamingRecognitionConfig()
        speech = b"speech"
        silence = stt._force_eou_silence()
        later = b"after"
        stt._force_eou_silences.append(silence)

        requests = list(stt._iter_streaming_requests([speech, silence, later], config))

        self.assertEqual(len(requests), 4)
        self.assertTrue(requests[0].HasField("streaming_config"))
        self.assertEqual(requests[1].audio_content, speech)
        self.assertEqual(dict(requests[1].runtime_config), {})
        self.assertEqual(requests[2].audio_content, silence)
        self.assertEqual(dict(requests[2].runtime_config), {"force_eou": "true"})
        self.assertEqual(requests[3].audio_content, later)
        self.assertEqual(dict(requests[3].runtime_config), {})
        self.assertFalse(stt._force_eou_silences)

    def test_equal_keepalive_silence_does_not_steal_force_eou(self) -> None:
        stt = NvidiaForceEouSTTService(use_ssl=False)
        stt._sample_rate = 16000
        config = rasr.StreamingRecognitionConfig()
        tagged = stt._force_eou_silence()
        keepalive = stt._force_eou_silence()
        stt._force_eou_silences.append(tagged)

        requests = list(stt._iter_streaming_requests([keepalive, tagged], config))

        self.assertEqual(dict(requests[1].runtime_config), {})
        self.assertEqual(dict(requests[2].runtime_config), {"force_eou": "true"})

    def test_response_handler_discards_stale_force_eou_markers(self) -> None:
        stt = NvidiaForceEouSTTService(use_ssl=False)
        stt._force_eou_silences.append(b"stale")
        stt._asr_service = Mock()
        stt._asr_service.streaming_response_generator = Mock()
        original = stt._asr_service.streaming_response_generator

        with patch("examples.shared.nvidia_force_eou_stt.NvidiaSTTService._response_handler") as parent:
            stt._response_handler(Mock(spec=AudioChunkIterator))

        parent.assert_called_once()
        self.assertIs(stt._asr_service.streaming_response_generator, original)
        self.assertFalse(stt._force_eou_silences)


class NvidiaForceEouSTTServiceTests(unittest.IsolatedAsyncioTestCase):
    def _service(self) -> NvidiaForceEouSTTService:
        return NvidiaForceEouSTTService(use_ssl=False)

    async def test_request_force_eou_sets_pending_and_queues_silence(self) -> None:
        stt = self._service()
        stt._sample_rate = 16000
        iterator = AudioChunkIterator(asyncio.get_running_loop())
        stt._audio_iterator = iterator

        await stt.request_force_eou()

        silence = iterator._queue.get_nowait()
        self.assertEqual(silence, b"\x00" * (int(16000 * FORCE_EOU_SILENCE_SECS) * 2))
        self.assertEqual(len(stt._force_eou_silences), 1)
        self.assertIs(stt._force_eou_silences[0], silence)
        self.assertTrue(iterator._queue.empty())

    async def test_request_force_eou_tags_silence_after_buffered_speech(self) -> None:
        stt = self._service()
        stt._sample_rate = 16000
        iterator = AudioChunkIterator(asyncio.get_running_loop())
        stt._audio_iterator = iterator
        await iterator.put(b"speech")

        await stt.request_force_eou()

        speech = iterator._queue.get_nowait()
        silence = iterator._queue.get_nowait()
        requests = list(stt._iter_streaming_requests([speech, silence], rasr.StreamingRecognitionConfig()))
        self.assertEqual(dict(requests[1].runtime_config), {})
        self.assertEqual(dict(requests[2].runtime_config), {"force_eou": "true"})

    async def test_vad_stop_forwards_then_requests_force_eou(self) -> None:
        stt = self._service()
        events = []

        async def process_parent(frame, direction):
            events.append(("parent", frame, direction))

        async def request_force_eou():
            events.append(("force_eou",))

        frame = VADUserStoppedSpeakingFrame()
        parent = AsyncMock(side_effect=process_parent)
        stt.request_force_eou = AsyncMock(side_effect=request_force_eou)

        with patch("examples.shared.nvidia_force_eou_stt.NvidiaSTTService.process_frame", parent):
            await stt.process_frame(frame, FrameDirection.DOWNSTREAM)

        parent.assert_awaited_once_with(frame, FrameDirection.DOWNSTREAM)
        stt.request_force_eou.assert_awaited_once()
        self.assertEqual(events, [("parent", frame, FrameDirection.DOWNSTREAM), ("force_eou",)])

    async def test_non_vad_stop_frames_do_not_request_force_eou(self) -> None:
        stt = self._service()
        stt.request_force_eou = AsyncMock()
        parent = AsyncMock()

        with patch("examples.shared.nvidia_force_eou_stt.NvidiaSTTService.process_frame", parent):
            await stt.process_frame(object(), FrameDirection.DOWNSTREAM)

        stt.request_force_eou.assert_not_awaited()
        parent.assert_awaited_once()

    async def test_request_force_eou_is_a_no_op_without_an_active_stream(self) -> None:
        stt = self._service()
        stt._audio_iterator = None
        await stt.request_force_eou()
        self.assertFalse(stt._force_eou_silences)
