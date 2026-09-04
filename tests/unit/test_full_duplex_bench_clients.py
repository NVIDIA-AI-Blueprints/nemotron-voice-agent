# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Tests for the standalone Full-Duplex-Bench inference clients."""

# ruff: noqa: D102

import asyncio
import base64
import importlib.util
import json
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
from pipecat.frames.protobufs import frames_pb2

CLIENT_DIR = Path(__file__).resolve().parents[2] / "benchmarking_tools" / "Full-Duplex-Bench-Eval"
if importlib.util.find_spec("soundfile") is None:
    sys.modules["soundfile"] = types.SimpleNamespace()


def load_client_module(name: str, filename: str):
    """Load a standalone benchmark script as an importable module."""
    spec = importlib.util.spec_from_file_location(name, CLIENT_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


realtime = load_client_module("full_duplex_realtime", "inference_realtime.py")
rtvi = load_client_module("full_duplex_rtvi", "inference_rtvi.py")


class FullDuplexRealtimeTests(unittest.IsolatedAsyncioTestCase):
    """Validate Realtime transport and PCM edge cases."""

    def client(self, **overrides):
        options = {
            "api_key": "secret",
            "auth_scheme": "Bearer",
            "transport_sample_rate": 24_000,
            "output_sample_rate": 16_000,
            "connect_timeout": 1,
            "verify_tls": True,
            "input_tail_silence": 0,
            "post_input_response_timeout": 1,
            "post_audio_idle_timeout": 1,
            "max_post_send_duration": 1,
            "preserve_late_output": False,
        }
        options.update(overrides)
        return realtime.InferenceClient("wss://example.test/realtime", **options)

    async def test_rejects_credentials_over_plain_websocket(self) -> None:
        socket = realtime.RealtimeSocket(
            "ws://example.test/realtime",
            api_key="secret",
            auth_scheme="Bearer",
            connect_timeout=1,
            verify_tls=True,
        )
        with (
            patch.object(realtime.websockets, "connect", new_callable=AsyncMock) as connect,
            self.assertRaisesRegex(RuntimeError, "unencrypted WebSocket"),
        ):
            await socket.__aenter__()
        connect.assert_not_awaited()

    def test_carries_odd_pcm_byte_into_next_delta(self) -> None:
        client = self.client()
        pcm_remainder = bytearray()

        first = client.decode_audio(
            {"type": "response.output_audio.delta", "delta": base64.b64encode(b"\x01").decode("ascii")},
            pcm_remainder,
        )
        second = client.decode_audio(
            {
                "type": "response.output_audio.delta",
                "delta": base64.b64encode(b"\x02\x03\x04").decode("ascii"),
            },
            pcm_remainder,
        )

        self.assertIsNone(first)
        np.testing.assert_array_equal(second, np.array([513, 1027], dtype=np.int16))

    def test_concurrent_sessions_keep_independent_pcm_remainders(self) -> None:
        client = self.client()
        first_remainder = bytearray()
        second_remainder = bytearray()

        first_partial = client.decode_audio(
            {"type": "response.output_audio.delta", "delta": base64.b64encode(b"\x01").decode("ascii")},
            first_remainder,
        )
        second_partial = client.decode_audio(
            {"type": "response.output_audio.delta", "delta": base64.b64encode(b"\x11").decode("ascii")},
            second_remainder,
        )
        first = client.decode_audio(
            {"type": "response.output_audio.delta", "delta": base64.b64encode(b"\x02").decode("ascii")},
            first_remainder,
        )
        second = client.decode_audio(
            {"type": "response.output_audio.delta", "delta": base64.b64encode(b"\x12").decode("ascii")},
            second_remainder,
        )

        self.assertIsNone(first_partial)
        self.assertIsNone(second_partial)
        np.testing.assert_array_equal(first, np.array([513], dtype=np.int16))
        np.testing.assert_array_equal(second, np.array([4625], dtype=np.int16))

    def test_credentialed_connector_rejects_redirects(self) -> None:
        connector = realtime._CredentialSafeConnect("wss://example.test/realtime")

        with patch.object(
            realtime.WebSocketConnect,
            "process_redirect",
            return_value="wss://redirected.example/realtime",
        ):
            result = connector.process_redirect(Exception())

        self.assertIsInstance(result, realtime.SecurityError)

    def test_preprocesses_input_at_provider_transport_rate(self) -> None:
        client = self.client(transport_sample_rate=24_000)
        with (
            patch.object(realtime.sf, "read", return_value=(np.zeros(320, dtype=np.int16), 16_000), create=True),
            patch.object(realtime.resampy, "resample", return_value=np.zeros(480, dtype=np.int16)),
        ):
            audio, duration = client.preprocess_audio(Path("input.wav"))

        self.assertAlmostEqual(duration, 0.02, places=3)
        self.assertEqual(len(audio), 480)

    def test_resamples_provider_output_to_benchmark_rate(self) -> None:
        client = self.client(output_sample_rate=24_000)
        pcm = np.zeros(480, dtype=np.int16).tobytes()

        chunk = client.decode_audio(
            {"type": "response.output_audio.delta", "delta": base64.b64encode(pcm).decode("ascii")},
            bytearray(),
        )

        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.dtype, np.int16)
        self.assertEqual(len(chunk), 320)

    def test_assembles_contiguous_and_late_chunks(self) -> None:
        client = self.client()
        chunk = np.full(320, 1000, dtype=np.int16)

        contiguous = client.assemble_output([chunk, chunk], [0.0, 0.0201], input_duration=0.1)
        late = client.assemble_output([chunk], [0.08], input_duration=0.05)

        np.testing.assert_array_equal(contiguous[:640], np.full(640, 1000, dtype=np.int16))
        np.testing.assert_array_equal(late, np.zeros(800, dtype=np.int16))

    def test_preserve_late_output_extends_timeline(self) -> None:
        client = self.client(preserve_late_output=True)
        chunk = np.full(320, 1000, dtype=np.int16)

        output = client.assemble_output([chunk], [0.08], input_duration=0.05)

        self.assertEqual(len(output), 1600)
        np.testing.assert_array_equal(output[1280:], chunk)

    async def test_absolute_post_send_deadline_bounds_continuous_audio(self) -> None:
        client = self.client(max_post_send_duration=0)
        encoded = base64.b64encode(b"\x00\x00").decode("ascii")
        socket = AsyncMock()
        socket.receive_event.return_value = {"type": "response.output_audio.delta", "delta": encoded}
        send_task = asyncio.create_task(asyncio.sleep(0))
        await send_task

        chunks, _ = await client.receive_audio(socket, time.monotonic(), send_task, bytearray())

        self.assertEqual(len(chunks), 1)
        socket.receive_event.assert_awaited_once()


class FullDuplexRTVITests(unittest.IsolatedAsyncioTestCase):
    """Validate RTVI readiness and bounded receive behavior."""

    async def test_client_ready_uses_installed_protocol_version(self) -> None:
        client = rtvi.InferenceClient("https://example.test", "wss://example.test", None)
        websocket = AsyncMock()

        await client._send_client_ready_frame(websocket)

        frame = frames_pb2.Frame.FromString(websocket.send.await_args.args[0])
        payload = json.loads(frame.message.data)
        self.assertEqual(payload["data"]["version"], rtvi.RTVI.PROTOCOL_VERSION)

    async def test_intro_deadline_bounds_non_audio_frames(self) -> None:
        client = rtvi.InferenceClient(
            "https://example.test",
            "wss://example.test",
            None,
            bot_intro_max_duration=0,
        )
        websocket = AsyncMock()

        chunks, duration = await client._drain_bot_intro_audio(websocket)

        self.assertEqual((chunks, duration), (0, 0.0))
        websocket.recv.assert_not_awaited()

    async def test_post_send_deadline_bounds_non_audio_frames(self) -> None:
        client = rtvi.InferenceClient(
            "https://example.test",
            "wss://example.test",
            None,
            max_post_send_duration=0,
        )
        websocket = AsyncMock()
        websocket.recv.return_value = b"not-a-protobuf-frame"
        send_task = asyncio.create_task(asyncio.sleep(0))
        await send_task

        chunks, timestamps = await client.receive_audio_stream(websocket, time.monotonic(), send_task)

        self.assertEqual((chunks, timestamps), ([], []))
        websocket.recv.assert_awaited_once()

    async def test_discards_odd_length_pcm_frames(self) -> None:
        client = rtvi.InferenceClient(
            "https://example.test",
            "wss://example.test",
            None,
            max_post_send_duration=0,
        )
        frame = frames_pb2.Frame(audio=frames_pb2.AudioRawFrame(audio=b"\x00", sample_rate=16_000, num_channels=1))
        websocket = AsyncMock()
        websocket.recv.return_value = frame.SerializeToString()
        send_task = asyncio.create_task(asyncio.sleep(0))
        await send_task

        chunks, timestamps = await client.receive_audio_stream(websocket, time.monotonic(), send_task)

        self.assertEqual((chunks, timestamps), ([], []))
        websocket.recv.assert_awaited_once()

    async def test_sender_cleanup_propagates_outer_cancellation(self) -> None:
        async def cancel_during_cleanup() -> None:
            send_task = asyncio.create_task(asyncio.sleep(10))
            current_task = asyncio.current_task()
            assert current_task is not None
            current_task.cancel()
            await rtvi.InferenceClient._settle_send_task(send_task)

        task = asyncio.create_task(cancel_during_cleanup())
        with self.assertRaises(asyncio.CancelledError):
            await task
