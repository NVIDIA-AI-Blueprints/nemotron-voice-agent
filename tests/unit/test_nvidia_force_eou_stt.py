# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.stt import AudioChunkIterator
from riva.client.proto import riva_asr_pb2 as rasr

from examples.shared.nvidia_force_eou_stt import (
    DEFAULT_STOP_HISTORY_MS,
    FORCE_EOU_SILENCE_SECS,
    NvidiaForceEouSTTService,
    build_nvidia_stt_service,
)
from examples.shared.stt_finalize_frame import STTFinalizeFrame


class StopHistoryTests(unittest.TestCase):
    def test_builder_always_uses_default_stop_history(self) -> None:
        nemotron = build_nvidia_stt_service(asr_kwargs={"use_ssl": False}, asr_model="nemotron-asr-streaming")
        parakeet = build_nvidia_stt_service(asr_kwargs={"use_ssl": False}, asr_model="parakeet-ctc")
        cache_aware = build_nvidia_stt_service(
            asr_kwargs={"use_ssl": False},
            asr_model="cache-aware-parakeet-rnnt-multi-asr-streaming-sortformer",
        )
        self.assertIsInstance(nemotron, NvidiaForceEouSTTService)
        self.assertEqual(nemotron._stop_history, DEFAULT_STOP_HISTORY_MS)
        self.assertEqual(parakeet._stop_history, DEFAULT_STOP_HISTORY_MS)
        self.assertEqual(cache_aware._stop_history, DEFAULT_STOP_HISTORY_MS)

    def test_silero_vad_turn_detection_still_uses_default_stop_history(self) -> None:
        with patch.dict(os.environ, {"USE_SILERO_VAD_TURN_DETECTION": "true"}):
            stt = build_nvidia_stt_service(asr_kwargs={"use_ssl": False}, asr_model="nemotron-asr-streaming")
        self.assertEqual(stt._stop_history, DEFAULT_STOP_HISTORY_MS)


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

    async def test_finalize_frame_requests_force_eou_and_does_not_forward(self) -> None:
        stt = self._service()
        stt.request_force_eou = AsyncMock()
        stt.push_frame = AsyncMock()

        await stt.process_frame(STTFinalizeFrame(), FrameDirection.UPSTREAM)

        stt.request_force_eou.assert_awaited_once()
        stt.push_frame.assert_not_awaited()

    async def test_non_finalize_frames_do_not_request_force_eou(self) -> None:
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
