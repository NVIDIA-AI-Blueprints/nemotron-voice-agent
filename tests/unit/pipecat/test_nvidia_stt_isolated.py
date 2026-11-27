# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103

"""Dedicated ASR stream executor, per-worker stream cap and PCM batching."""

from __future__ import annotations

import asyncio
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import nemotron_voice_agent.pipecat.runtime.examples_registry as examples_registry
import nemotron_voice_agent.pipecat.runtime.server as server
import nemotron_voice_agent.pipecat.shared.nvidia_stt_isolated as isolated


class _FakeIterator:
    def __init__(self) -> None:
        self.closed = False
        self.chunks: list[bytes] = []

    def put_nowait(self, audio: bytes) -> None:
        self.chunks.append(audio)


def _service(batch_ms: int, loop: asyncio.AbstractEventLoop) -> isolated.IsolatedNvidiaSTTService:
    """Build the service without the Riva client, which needs a server."""
    service = object.__new__(isolated.IsolatedNvidiaSTTService)
    service._batch_ms = batch_ms
    service._pending_pcm = bytearray()
    service._tail_timer = None
    service._sample_rate = 16000
    service._audio_iterator = _FakeIterator()
    service._config = object()
    service.get_event_loop = lambda: loop
    return service


class AsrStreamSlotsTests(unittest.TestCase):
    def test_slots_stop_at_capacity_and_free_on_release(self) -> None:
        slots = isolated.AsrStreamSlots(2)

        self.assertTrue(slots.try_acquire())
        self.assertTrue(slots.try_acquire())
        self.assertFalse(slots.try_acquire())
        slots.release()
        self.assertTrue(slots.try_acquire())

    def test_extra_release_does_not_create_capacity(self) -> None:
        slots = isolated.AsrStreamSlots(1)

        slots.release()

        self.assertTrue(slots.try_acquire())
        self.assertFalse(slots.try_acquire())

    def test_capacity_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            isolated.AsrStreamSlots(0)

    def test_settings_come_from_the_environment(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(isolated.ASR_MAX_STREAMS_ENV, None)
            os.environ.pop(isolated.ASR_INPUT_BATCH_MS_ENV, None)
            self.assertEqual(isolated.max_streams_per_worker(), 64)
            self.assertEqual(isolated.input_batch_ms(), 0)
        with patch.dict(os.environ, {isolated.ASR_MAX_STREAMS_ENV: "12", isolated.ASR_INPUT_BATCH_MS_ENV: "160"}):
            self.assertEqual(isolated.max_streams_per_worker(), 12)
            self.assertEqual(isolated.input_batch_ms(), 160)
        with patch.dict(os.environ, {isolated.ASR_MAX_STREAMS_ENV: "0"}):
            self.assertEqual(isolated.max_streams_per_worker(), 1)


class ExecutorIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_reader_runs_on_the_dedicated_pool(self) -> None:
        service = _service(0, asyncio.get_running_loop())
        seen: dict[str, str] = {}

        def reader(iterator, config) -> None:
            seen["thread"] = threading.current_thread().name

        service._response_handler = reader
        with patch.object(isolated, "_executor", None):
            await service._thread_task_handler()

        self.assertTrue(seen["thread"].startswith("nva-asr-stream"), seen)

    async def test_blocked_readers_do_not_delay_default_executor_work(self) -> None:
        loop = asyncio.get_running_loop()
        service = _service(0, loop)
        release = threading.Event()
        started = threading.Event()

        def reader(iterator, config) -> None:
            started.set()
            release.wait(5)

        service._response_handler = reader
        with patch.object(isolated, "_executor", None):
            stream = asyncio.create_task(service._thread_task_handler())
            await asyncio.to_thread(started.wait, 5)
            # A readiness probe uses the default executor and must not queue behind the reader.
            probe = await asyncio.wait_for(asyncio.to_thread(lambda: "ready"), timeout=1)
            release.set()
            await stream

        self.assertEqual(probe, "ready")

    async def test_no_thread_is_started_without_an_iterator(self) -> None:
        service = _service(0, asyncio.get_running_loop())
        service._audio_iterator = None
        service._response_handler = AsyncMock()

        await service._thread_task_handler()

        service._response_handler.assert_not_called()


class InputBatchingTests(unittest.IsolatedAsyncioTestCase):
    async def run_stt(self, service, audio: bytes) -> list:
        return [frame async for frame in service.run_stt(audio)]

    async def test_five_32ms_chunks_make_one_160ms_message(self) -> None:
        service = _service(160, asyncio.get_running_loop())
        chunk = b"\x01\x00" * 512  # 32 ms of 16 kHz 16-bit mono PCM

        for _ in range(5):
            frames = await self.run_stt(service, chunk)

        self.assertEqual(frames, [None])
        self.assertEqual([len(sent) for sent in service._audio_iterator.chunks], [5120])
        self.assertIsNone(service._tail_timer)

    async def test_remainder_is_sent_after_one_batch_period(self) -> None:
        service = _service(20, asyncio.get_running_loop())

        await self.run_stt(service, b"\x01\x00" * 100)
        self.assertEqual(service._audio_iterator.chunks, [])
        await asyncio.sleep(0.08)

        self.assertEqual([len(sent) for sent in service._audio_iterator.chunks], [200])
        self.assertFalse(service._pending_pcm)

    async def test_batching_keeps_every_byte_in_order(self) -> None:
        service = _service(160, asyncio.get_running_loop())
        payload = bytes(range(256)) * 40  # 10,240 bytes

        for offset in range(0, len(payload), 1000):
            await self.run_stt(service, payload[offset : offset + 1000])
        service._cancel_tail_flush()
        service._flush_tail()

        self.assertEqual(b"".join(service._audio_iterator.chunks), payload)
        self.assertTrue(all(len(sent) == 5120 for sent in service._audio_iterator.chunks[:-1]))

    async def test_stopping_drops_unsent_remainder(self) -> None:
        service = _service(160, asyncio.get_running_loop())
        await self.run_stt(service, b"\x01\x00" * 100)

        with patch.object(isolated.NvidiaSTTService, "_stop_tasks", new_callable=AsyncMock):
            await service._stop_tasks()

        self.assertFalse(service._pending_pcm)
        self.assertIsNone(service._tail_timer)

    async def test_disabled_batching_forwards_to_the_stock_service(self) -> None:
        service = _service(0, asyncio.get_running_loop())

        async def stock(self, audio):
            yield f"stock:{len(audio)}"

        with patch.object(isolated.NvidiaSTTService, "run_stt", stock):
            frames = await self.run_stt(service, b"\x00" * 64)

        self.assertEqual(frames, ["stock:64"])
        self.assertEqual(service._audio_iterator.chunks, [])


class WebSocketAdmissionTests(unittest.TestCase):
    """The cap is reserved before readiness and released when the session ends."""

    def refused_code(self, slots: isolated.AsrStreamSlots, services: dict, ready: AsyncMock) -> int:
        example = {"key": "test-example", "services": services}
        real_find = examples_registry.find

        def find(key: str = ""):
            return example if key == "test-example" else real_find(key)

        with (
            patch.object(server, "get_asr_stream_slots", return_value=slots),
            patch.object(examples_registry, "find", side_effect=find),
            patch.object(server, "_resolve_config", return_value={"pipeline_mode": "test-example"}),
            patch.object(server, "_ensure_services_ready_for_connection", ready),
            TestClient(server.create_app()) as client,
            self.assertRaises(WebSocketDisconnect) as refused,
            client.websocket_connect("/api/ws"),
        ):
            pass
        error = refused.exception
        # A denial response carries an HTTP status; an ordinary close carries a WebSocket code.
        return getattr(error, "status_code", None) or error.code

    def test_worker_at_capacity_refuses_before_readiness(self) -> None:
        slots = isolated.AsrStreamSlots(1)
        slots.try_acquire()
        ready = AsyncMock()

        code = self.refused_code(slots, {"asr": ["nemotron-asr-streaming-english"]}, ready)

        self.assertEqual(code, 503)
        ready.assert_not_awaited()
        self.assertEqual(slots.active, 1)

    def test_slot_is_released_when_readiness_fails(self) -> None:
        slots = isolated.AsrStreamSlots(1)
        ready = AsyncMock(side_effect=RuntimeError("LLM is still starting"))

        code = self.refused_code(slots, {"asr": ["nemotron-asr-streaming-english"]}, ready)

        self.assertEqual(code, 1011)
        ready.assert_awaited_once()
        self.assertEqual(slots.active, 0)

    def test_examples_without_asr_do_not_use_a_slot(self) -> None:
        slots = isolated.AsrStreamSlots(1)
        slots.try_acquire()
        ready = AsyncMock(side_effect=RuntimeError("stop here"))

        code = self.refused_code(slots, {"llm": ["nemotron-super"]}, ready)

        self.assertEqual(code, 1011)
        ready.assert_awaited_once()
        self.assertEqual(slots.active, 1)


class RejectAtCapacityTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_denial_is_used_when_the_server_supports_it(self) -> None:
        websocket = SimpleNamespace(
            scope={"extensions": {"websocket.http.response": {}}},
            send_denial_response=AsyncMock(),
            close=AsyncMock(),
        )

        await isolated.reject_at_capacity(websocket, isolated.AsrStreamSlots(1))

        response = websocket.send_denial_response.await_args.args[0]
        self.assertEqual(response.status_code, 503)
        websocket.close.assert_not_awaited()

    async def test_falls_back_to_close_code_1013(self) -> None:
        websocket = SimpleNamespace(scope={}, send_denial_response=AsyncMock(), close=AsyncMock())

        await isolated.reject_at_capacity(websocket, isolated.AsrStreamSlots(1))

        self.assertEqual(websocket.close.await_args.kwargs["code"], 1013)
        websocket.send_denial_response.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
