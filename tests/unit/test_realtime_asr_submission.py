# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Deterministic stream-submission interleavings for Realtime NVIDIA ASR."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pipecat.frames.frames import TranscriptionFrame
from pipecat.services.nvidia.stt import NvidiaSTTService

from realtime.asr import RealtimeNvidiaSTTService
from realtime.frames import RealtimeInputTranscriptionErrorFrame


class _PausedAudioIterator:
    def __init__(self) -> None:
        self.closed = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.chunks: list[bytes] = []

    async def put(self, audio: bytes) -> None:
        """Expose the enqueue suspension point to the test."""
        self.entered.set()
        await self.release.wait()
        self.chunks.append(audio)


async def _consume_run_stt(service: RealtimeNvidiaSTTService, audio: bytes) -> list[object]:
    return [frame async for frame in service.run_stt(audio)]


class _SubmissionCase:
    def __init__(self, mode: str, payload: bytes) -> None:
        self.mode = mode
        self.payload = payload
        self.service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        self.service._sample_rate = 16_000
        self.service.start_processing_metrics = AsyncMock()
        self.iterator = _PausedAudioIterator()
        self.service._audio_iterator = self.iterator
        self.base_push = AsyncMock()
        self.base_reconnect = AsyncMock()
        self.samples_before_reset: list[int] = []
        self._reset = self.service._realtime_ownership.reset_stream

    def record_reset(self) -> None:
        self.samples_before_reset.append(self.service._realtime_ownership.submitted_samples)
        self._reset()

    async def submit(self) -> object:
        if self.mode == "audio":
            return await _consume_run_stt(self.service, self.payload)
        return await self.service._send_keepalive(self.payload)


class RealtimeASRSubmissionTransactionTests(unittest.IsolatedAsyncioTestCase):
    """Verify stream reset cannot bisect provider enqueue and coordinate advance."""

    async def test_submission_and_count_complete_before_reconnect_reset(self) -> None:
        """Serialize both ordinary audio and keepalive silence with stream reset."""
        for mode, payload, expected_samples in (
            ("audio", b"\x00\x00" * 160, 160),
            ("keepalive", b"\x00\x00" * 80, 80),
        ):
            with self.subTest(mode=mode):
                case = _SubmissionCase(mode, payload)
                service = case.service

                with (
                    patch.object(NvidiaSTTService, "push_frame", case.base_push),
                    patch.object(NvidiaSTTService, "_do_reconnect", case.base_reconnect),
                    patch.object(service._realtime_ownership, "reset_stream", side_effect=case.record_reset),
                ):
                    submission = asyncio.create_task(case.submit())
                    await case.iterator.entered.wait()
                    reconnect = asyncio.create_task(service._do_reconnect())
                    await asyncio.sleep(0)

                    self.assertEqual(service._realtime_ownership.submitted_samples, 0)
                    case.base_reconnect.assert_not_awaited()

                    case.iterator.release.set()
                    result = await submission
                    await reconnect

                    self.assertEqual(result, [None] if mode == "audio" else None)
                    self.assertEqual(case.iterator.chunks, [payload])
                    self.assertEqual(case.samples_before_reset, [expected_samples])
                    self.assertEqual(service._realtime_ownership.submitted_samples, 0)
                    self.assertTrue(service._realtime_ownership_faulted)
                    case.base_reconnect.assert_awaited_once()

                    if mode == "audio":
                        new_audio = b"\x00\x00" * 80
                        self.assertEqual(await _consume_run_stt(service, new_audio), [None])
                        self.assertEqual(service._realtime_ownership.submitted_samples, 80)
                        transcript = TranscriptionFrame(
                            text="new stream",
                            user_id="",
                            timestamp="",
                            result=SimpleNamespace(audio_processed=0.005, alternatives=[]),
                            finalized=True,
                        )
                        case.base_push.reset_mock()
                        await service.push_frame(transcript)

                        self.assertEqual(case.base_push.await_count, 2)
                        error = case.base_push.await_args_list[0].args[0]
                        self.assertIsInstance(error, RealtimeInputTranscriptionErrorFrame)
                        self.assertEqual(error.code, "asr_transcription_stream_restarted")
                        self.assertIs(case.base_push.await_args_list[1].args[0], transcript)
