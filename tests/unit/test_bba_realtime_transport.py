# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D105, D107

"""Big Bench Audio's ``speech-inference.py`` over the OpenAI Realtime-compatible API (``--transport realtime``).

The client must change only its transport: the send pacing, the silence tail, the exit rules, and the
``output.wav`` assembly are the ``ws`` path's. A fake server on ``127.0.0.1`` behaves like the Frontend
Delegation Agent: ``session.created`` (no ``conversation.created``), ``session.updated`` after the client's
``session.update``, and audio as base64 ``response.output_audio.delta`` events. The exit-rule constants are
patched to small values here only; no test reads the developer's ``.env`` or ``REALTIME_API_KEY``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
import wave
from http import HTTPStatus
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import soundfile as sf
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed, InvalidStatus

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarking_tools" / "AA-BigBenchAudio-Eval" / "speech-inference.py"
_spec = importlib.util.spec_from_file_location("bba_speech_inference", SCRIPT)
assert _spec is not None and _spec.loader is not None
bba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bba)

KEY_ENV = bba.REALTIME_API_KEY_ENV
FILLER, ANSWER = 1000, 2000  # sample values that identify each response in output.wav
CHUNK_BYTES = bba.SAMPLE_RATE * bba.CHUNK_MS // 1000 * 2


def _delta(value: int, samples: int = 1600) -> dict[str, Any]:
    pcm = np.full(samples, value, dtype=np.int16).tobytes()
    return {
        "type": "response.output_audio.delta",
        "response_id": "resp_1",
        "item_id": "item_1",
        "output_index": 0,
        "content_index": 0,
        "delta": base64.b64encode(pcm).decode("ascii"),
    }


DONE: dict[str, Any] = {"type": "response.done", "response": {"status": "completed"}}


def _write_input_wav(path: Path, seconds: float = 0.1) -> None:
    samples = (np.sin(np.arange(int(bba.SAMPLE_RATE * seconds)) / 8.0) * 8000).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(bba.SAMPLE_RATE)
        handle.writeframes(samples.tobytes())


class FakeRealtimeServer:
    """A one-session-at-a-time stand-in for FDA's ``/v1/realtime``.

    ``mode``: ``normal``, ``busy`` (``server_busy`` and close 1013 before ``session.created``),
    ``close`` (close 1011 after ``session.created``), ``stall`` (no ``session.updated``), or
    ``forbid`` (HTTP 403 at the opening handshake). ``script`` lists ``(delay_s, event)`` pairs sent
    after ``session.updated``.
    """

    def __init__(self, mode: str = "normal", script: list[tuple[float, dict[str, Any]]] | None = None, *, extra=False):
        self.mode = mode
        self.script = script or []
        self.extra = extra
        self.messages: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self.authorization: list[str | None] = []
        self.finished = asyncio.Event()

    async def __aenter__(self) -> FakeRealtimeServer:
        self._server = await serve(self._handler, "127.0.0.1", 0, process_request=self._process_request)
        port = self._server.sockets[0].getsockname()[1]
        self.http_base, self.ws_origin = f"http://127.0.0.1:{port}", f"ws://127.0.0.1:{port}"
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._server.close()
        await self._server.wait_closed()

    def _process_request(self, connection: Any, request: Any) -> Any:
        if self.mode == "forbid":
            return connection.respond(HTTPStatus.FORBIDDEN, "invalid Realtime API key\n")
        return None

    async def _send(self, websocket: Any, event: dict[str, Any]) -> None:
        await websocket.send(json.dumps(event))

    async def _play(self, websocket: Any) -> None:
        with contextlib.suppress(ConnectionClosed):
            for delay, event in self.script:
                await asyncio.sleep(delay)
                await self._send(websocket, event)

    async def _handler(self, websocket: Any) -> None:
        self.paths.append(websocket.request.path)
        self.authorization.append(websocket.request.headers.get("Authorization"))
        player = None
        try:
            if self.mode == "busy":
                busy = {"code": "server_busy", "message": "4 Realtime sessions are open", "type": "server_error"}
                await self._send(websocket, {"type": "error", "error": busy})
                await websocket.close(1013, "server busy")
                return
            await self._send(websocket, {"type": "session.created", "session": {"type": "realtime"}})
            if self.mode == "close":
                await websocket.close(1011, "pipeline failed")
                return
            if self.extra:
                await self._send(websocket, {"type": "conversation.created", "conversation": {}})
            self.messages.append(json.loads(await websocket.recv()))
            if self.mode == "stall":
                await websocket.wait_closed()
                return
            await self._send(websocket, {"type": "session.updated", "session": {"type": "realtime"}})
            player = asyncio.create_task(self._play(websocket))
            async for raw in websocket:
                self.messages.append(json.loads(raw))
        except ConnectionClosed:
            pass
        finally:
            if player is not None:
                player.cancel()
            self.finished.set()

    def client(self, api_key: str | None = None) -> Any:
        return bba.BenchmarkClient(
            self.http_base, self.ws_origin, None, transport=bba.TRANSPORT_REALTIME, api_key=api_key
        )


class _FastExitRules(unittest.IsolatedAsyncioTestCase):
    """Small exit-rule constants for these tests only (the shipped values stay 5 s, 2 s, 2 s, 30 s)."""

    def setUp(self) -> None:
        for name, value in (
            ("POST_SEND_INITIAL_WAIT_SEC", 0.5),
            ("POST_SEND_GAP_TOLERANCE_SEC", 0.5),
            ("RECV_TIMEOUT_SEC", 0.2),
            ("REALTIME_HANDSHAKE_TIMEOUT_SEC", 0.5),
        ):
            patcher = mock.patch.object(bba, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.input_wav = self.tmp / "input.wav"
        _write_input_wav(self.input_wav)

    async def _run(self, server: FakeRealtimeServer, api_key: str | None = None) -> np.ndarray:
        """One sample: handshake, stream, collect; return ``output.wav``'s samples."""
        client = server.client(api_key)
        websocket = await client._connect()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                await client.process_conversation(websocket, str(self.input_wav), str(self.tmp))
        finally:
            await websocket.close()
        await asyncio.wait_for(server.finished.wait(), timeout=5)
        audio, rate = sf.read(self.tmp / "output.wav", dtype="int16")
        self.assertEqual(rate, bba.SAMPLE_RATE)
        return audio


class ShippedConstantsTests(unittest.TestCase):
    def test_exit_rules_and_handshake_limit_are_unchanged(self) -> None:
        self.assertEqual((bba.SAMPLE_RATE, bba.CHUNK_MS), (16000, 32))
        self.assertEqual(bba.POST_SEND_INITIAL_WAIT_SEC, 5.0)
        self.assertEqual(bba.POST_SEND_GAP_TOLERANCE_SEC, 2.0)
        self.assertEqual((bba.RECV_TIMEOUT_SEC, bba.OPEN_TIMEOUT_SEC), (2.0, 5.0))
        self.assertEqual(bba.REALTIME_HANDSHAKE_TIMEOUT_SEC, 30.0)
        self.assertEqual(bba.DEFAULT_REALTIME_MODEL, "nvidia/nemotron-realtime-frontend-delegation")


class RealtimeSessionTests(_FastExitRules):
    async def test_session_update_selects_16_khz_audio_only(self) -> None:
        async with FakeRealtimeServer(script=[(0.3, _delta(ANSWER)), (0.0, DONE)]) as server:
            await self._run(server, api_key="test-key")
        self.assertEqual(server.paths, ["/v1/realtime?model=nvidia%2Fnemotron-realtime-frontend-delegation"])
        self.assertEqual(server.authorization, ["Bearer test-key"])
        update = server.messages[0]
        self.assertEqual(update, bba.REALTIME_SESSION_UPDATE)
        self.assertEqual(update["session"]["audio"]["input"]["format"], {"type": "audio/pcm", "rate": 16000})
        self.assertEqual(update["session"]["audio"]["output"]["format"], {"type": "audio/pcm", "rate": 16000})
        self.assertNotIn("instructions", update["session"])
        self.assertNotIn("tools", update["session"])
        self.assertNotIn("turn_detection", update["session"]["audio"]["input"])

    async def test_audio_goes_out_as_appends_only(self) -> None:
        async with FakeRealtimeServer(script=[(0.3, _delta(ANSWER)), (0.0, DONE)]) as server:
            await self._run(server)
        appends = server.messages[1:]
        self.assertGreater(len(appends), 0)
        self.assertEqual({message["type"] for message in appends}, {"input_audio_buffer.append"})
        self.assertEqual({len(base64.b64decode(message["audio"])) for message in appends}, {CHUNK_BYTES})
        self.assertIsNone(server.authorization[0])  # no key, no header

    async def test_conversation_created_is_ignored(self) -> None:
        async with FakeRealtimeServer(script=[(0.3, _delta(ANSWER)), (0.0, DONE)], extra=True) as server:
            audio = await self._run(server)
        self.assertIn(ANSWER, audio)

    async def test_filler_and_answer_are_one_response(self) -> None:
        script = [(0.3, _delta(FILLER)), (0.0, DONE), (0.3, _delta(ANSWER)), (0.0, DONE)]
        async with FakeRealtimeServer(script=script) as server:
            audio = await self._run(server)
        self.assertIn(FILLER, audio)
        self.assertIn(ANSWER, audio)

    async def test_gap_tolerance_ends_the_sample_as_on_ws(self) -> None:
        # The answer arrives well after the gap tolerance following the filler, after the input was sent.
        script = [(2.4, _delta(FILLER)), (0.0, DONE), (1.6, _delta(ANSWER)), (0.0, DONE)]
        async with FakeRealtimeServer(script=script) as server:
            audio = await self._run(server)
        self.assertIn(FILLER, audio)
        self.assertNotIn(ANSWER, audio)

    async def test_initial_wait_ends_a_silent_sample_as_on_ws(self) -> None:
        async with FakeRealtimeServer(script=[]) as server:
            audio = await self._run(server)
        self.assertEqual(len(audio), 0)

    async def test_error_event_after_the_handshake_is_logged(self) -> None:
        error = {"type": "error", "error": {"code": "invalid_request_error", "message": "bad event"}}
        client = bba.BenchmarkClient("http://h", "ws://h", None, transport=bba.TRANSPORT_REALTIME)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertIsNone(client._decode_audio(json.dumps(error), "sample-7"))
        self.assertIn("sample-7", out.getvalue())
        self.assertIn("invalid_request_error: bad event", out.getvalue())


class RealtimeHandshakeFailureTests(_FastExitRules):
    async def test_forbidden_key_fails_at_the_opening_handshake(self) -> None:
        async with FakeRealtimeServer(mode="forbid") as server:
            with self.assertRaises(InvalidStatus) as caught:
                await server.client("wrong-key")._connect()
        self.assertEqual(caught.exception.response.status_code, 403)

    async def test_server_busy_fails_the_handshake(self) -> None:
        async with FakeRealtimeServer(mode="busy") as server:
            with self.assertRaisesRegex(RuntimeError, "server_busy"):
                await server.client()._connect()

    async def test_close_before_session_updated_fails_the_handshake(self) -> None:
        async with FakeRealtimeServer(mode="close") as server:
            with self.assertRaises(ConnectionClosed):
                await server.client()._connect()

    async def test_a_stalled_handshake_times_out(self) -> None:
        async with FakeRealtimeServer(mode="stall") as server:
            with self.assertRaisesRegex(TimeoutError, "within 0.5s"):
                await server.client()._connect()

    async def test_a_failed_sample_is_logged_and_writes_no_output(self) -> None:
        sample_dir = self.tmp / "dataset" / "3"
        sample_dir.mkdir(parents=True)
        _write_input_wav(sample_dir / "input.wav")
        async with FakeRealtimeServer(mode="busy") as server:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                await server.client().process_directory(str(self.tmp / "dataset"))
        self.assertIn("unsuccessful idx: 3", out.getvalue())
        self.assertIn("server_busy", out.getvalue())
        self.assertFalse((sample_dir / "output.wav").exists())


class RealtimeApiKeyTests(unittest.TestCase):
    """``REALTIME_API_KEY``: the environment first, then ``DOTENV_PATH``; empty or whitespace is unset."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dotenv = Path(tmp.name) / ".env"
        patcher = mock.patch.object(bba, "DOTENV_PATH", self.dotenv)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(KEY_ENV, None)

    def test_environment_wins(self) -> None:
        self.dotenv.write_text(f"{KEY_ENV}=from-file\n", encoding="utf-8")
        os.environ[KEY_ENV] = " from-env "
        self.assertEqual(bba.load_realtime_api_key(), "from-env")

    def test_dotenv_is_the_fallback(self) -> None:
        self.dotenv.write_text(f"{KEY_ENV}=from-file\n", encoding="utf-8")
        for value in (None, "", "   "):
            with self.subTest(env=value):
                if value is None:
                    os.environ.pop(KEY_ENV, None)
                else:
                    os.environ[KEY_ENV] = value
                self.assertEqual(bba.load_realtime_api_key(), "from-file")

    def test_unset_empty_whitespace_or_bare_is_none(self) -> None:
        self.assertIsNone(bba.load_realtime_api_key())  # no .env file
        for line in (f"{KEY_ENV}=", f"{KEY_ENV}=   ", KEY_ENV, "OTHER=1"):
            with self.subTest(line=line):
                self.dotenv.write_text(line + "\n", encoding="utf-8")
                self.assertIsNone(bba.load_realtime_api_key())

    def test_only_the_key_is_read_from_dotenv(self) -> None:
        os.environ.pop("NVIDIA_API_KEY", None)
        self.dotenv.write_text(f"NVIDIA_API_KEY=secret\n{KEY_ENV}=from-file\n", encoding="utf-8")
        self.assertEqual(bba.load_realtime_api_key(), "from-file")
        self.assertNotIn("NVIDIA_API_KEY", os.environ)
        self.assertNotIn(KEY_ENV, os.environ)


class WsTransportTests(unittest.TestCase):
    """The default ``ws`` transport sends and reads the same protobuf frames as before."""

    def setUp(self) -> None:
        self.client = bba.BenchmarkClient("http://h:7860", "ws://h:7860", None)

    def test_default_transport_is_ws(self) -> None:
        self.assertEqual(self.client.transport, bba.TRANSPORT_WS)
        self.assertEqual(self.client._websocket_url("abc"), "ws://h:7860/api/ws?session_id=abc")

    def test_encode_matches_the_original_audio_frame(self) -> None:
        pcm = np.arange(512, dtype=np.int16).tobytes()
        original = bba.frames_pb2.Frame(
            audio=bba.frames_pb2.AudioRawFrame(audio=pcm, sample_rate=bba.SAMPLE_RATE, num_channels=1)
        ).SerializeToString()
        self.assertEqual(self.client._encode_audio(pcm), original)

    def test_decode_reads_audio_frames_only(self) -> None:
        pcm = np.arange(256, dtype=np.int16)
        frame = self.client._encode_audio(pcm.tobytes())
        np.testing.assert_array_equal(self.client._decode_audio(frame, "x"), pcm)
        self.assertIsNone(self.client._decode_audio(self.client._encode_audio(b""), "x"))
        text = bba.frames_pb2.Frame(text=bba.frames_pb2.TextFrame(text="hello")).SerializeToString()
        self.assertIsNone(self.client._decode_audio(text, "x"))

    def test_unknown_transport_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            bba.BenchmarkClient("http://h", "ws://h", None, transport="grpc")


class CommandLineTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.input_dir = tmp.name
        patcher = mock.patch.object(bba, "DOTENV_PATH", Path(tmp.name) / "missing.env")
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(KEY_ENV, None)

    def _captured_client(self, *extra: str) -> Any:
        captured: list[Any] = []

        async def capture(client: Any, *args: Any, **kwargs: Any) -> None:
            captured.append(client)

        argv = ["speech-inference.py", "--input_dir", self.input_dir, "--inference", "--server-url"]
        argv += ["http://127.0.0.1:7860", *extra]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(bba.BenchmarkClient, "process_directory", capture):
            asyncio.run(bba.main())
        return captured[0]

    def test_ws_is_the_default(self) -> None:
        client = self._captured_client()
        self.assertEqual(client.transport, bba.TRANSPORT_WS)
        self.assertIsNone(client._api_key)

    def test_realtime_uses_the_default_model_and_the_key(self) -> None:
        os.environ[KEY_ENV] = "test-key"
        client = self._captured_client("--transport", "realtime")
        self.assertEqual(client.transport, bba.TRANSPORT_REALTIME)
        self.assertEqual(client.model, bba.DEFAULT_REALTIME_MODEL)
        self.assertEqual(client._api_key, "test-key")
        self.assertEqual(
            client._realtime_url(), f"ws://127.0.0.1:7860/v1/realtime?model={client.model.replace('/', '%2F')}"
        )

    def test_realtime_model_can_be_chosen(self) -> None:
        client = self._captured_client("--transport", "realtime", "--model", "nvidia/nemotron-realtime-client-tools")
        self.assertEqual(client.model, "nvidia/nemotron-realtime-client-tools")

    def test_model_requires_the_realtime_transport(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            self._captured_client("--model", "nvidia/nemotron-realtime-client-tools")
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
