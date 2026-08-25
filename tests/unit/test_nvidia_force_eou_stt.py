# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import asyncio
import unittest
from unittest.mock import AsyncMock

from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.stt import AudioChunkIterator
from riva.client.proto import riva_asr_pb2 as rasr

from examples.shared.nvidia_force_eou_stt import (
    DEFAULT_STOP_HISTORY_MS,
    FORCE_EOU_SILENCE_SECS,
    NEMOTRON_FORCE_EOU_STOP_HISTORY_MS,
    NvidiaForceEouSTTService,
    build_nvidia_stt_service,
    stop_history_for_asr_model,
)
from examples.shared.stt_finalize_frame import STTFinalizeFrame


class StopHistoryForAsrModelTests(unittest.TestCase):
    def test_nemotron_and_default_raise_stop_history(self) -> None:
        self.assertEqual(stop_history_for_asr_model(""), NEMOTRON_FORCE_EOU_STOP_HISTORY_MS)
        self.assertEqual(
            stop_history_for_asr_model("nemotron-asr-streaming"),
            NEMOTRON_FORCE_EOU_STOP_HISTORY_MS,
        )

    def test_parakeet_keeps_short_stop_history(self) -> None:
        self.assertEqual(stop_history_for_asr_model("parakeet-ctc"), DEFAULT_STOP_HISTORY_MS)
        self.assertEqual(
            stop_history_for_asr_model("parakeet-1.1b-rnnt-multilingual"),
            DEFAULT_STOP_HISTORY_MS,
        )

    def test_local_nemotron_nim_name_is_not_treated_as_parakeet(self) -> None:
        self.assertEqual(
            stop_history_for_asr_model("cache-aware-parakeet-rnnt-multi-asr-streaming-sortformer"),
            NEMOTRON_FORCE_EOU_STOP_HISTORY_MS,
        )

    def test_builder_applies_model_specific_stop_history(self) -> None:
        nemotron = build_nvidia_stt_service(asr_kwargs={"use_ssl": False}, asr_model="nemotron-asr-streaming")
        parakeet = build_nvidia_stt_service(asr_kwargs={"use_ssl": False}, asr_model="parakeet-ctc")
        self.assertIsInstance(nemotron, NvidiaForceEouSTTService)
        self.assertEqual(nemotron._stop_history, NEMOTRON_FORCE_EOU_STOP_HISTORY_MS)
        self.assertEqual(parakeet._stop_history, DEFAULT_STOP_HISTORY_MS)


class ForceEouStreamingRequestTests(unittest.TestCase):
    def test_attaches_runtime_config_only_on_the_pending_chunk(self) -> None:
        stt = NvidiaForceEouSTTService(use_ssl=False)
        stt._force_eou_pending = True
        config = rasr.StreamingRecognitionConfig()

        requests = list(stt._iter_streaming_requests([b"first", b"second"], config))

        self.assertEqual(len(requests), 3)
        self.assertTrue(requests[0].HasField("streaming_config"))
        self.assertEqual(requests[1].audio_content, b"first")
        self.assertEqual(dict(requests[1].runtime_config), {"force_eou": "true"})
        self.assertEqual(requests[2].audio_content, b"second")
        self.assertEqual(dict(requests[2].runtime_config), {})
        self.assertFalse(stt._force_eou_pending)


class NvidiaForceEouSTTServiceTests(unittest.IsolatedAsyncioTestCase):
    def _service(self) -> NvidiaForceEouSTTService:
        return NvidiaForceEouSTTService(use_ssl=False)

    async def test_request_force_eou_sets_pending_and_queues_silence(self) -> None:
        stt = self._service()
        stt._sample_rate = 16000
        iterator = AudioChunkIterator(asyncio.get_running_loop())
        stt._audio_iterator = iterator

        await stt.request_force_eou()

        self.assertTrue(stt._force_eou_pending)
        silence = iterator._queue.get_nowait()
        self.assertEqual(silence, b"\x00" * (int(16000 * FORCE_EOU_SILENCE_SECS) * 2))
        self.assertTrue(iterator._queue.empty())

    async def test_finalize_frame_requests_force_eou_and_does_not_forward(self) -> None:
        stt = self._service()
        stt.request_force_eou = AsyncMock()
        stt.push_frame = AsyncMock()

        await stt.process_frame(STTFinalizeFrame(), FrameDirection.UPSTREAM)

        stt.request_force_eou.assert_awaited_once()
        stt.push_frame.assert_not_awaited()

    async def test_request_force_eou_is_a_no_op_without_an_active_stream(self) -> None:
        stt = self._service()
        stt._audio_iterator = None
        await stt.request_force_eou()
        self.assertFalse(stt._force_eou_pending)
