# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import asyncio
import hashlib
import json
import tempfile
import unittest
import uuid
import wave
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from examples.generic.offline.models import RATE, AudioSpan, SessionResult
from examples.generic.offline.protocol import (
    HEADER,
    PROTOCOL,
    InputUpload,
    OfflineProtocolError,
    SessionLimits,
    configuration_fingerprint,
    decode_packet,
    encode_packet,
    parse_control,
    validate_start,
)
from examples.generic.offline.websocket import serve_offline_session


def start_message(pcm, **extra):
    """Create a finite-source handshake for the fixture."""
    return {
        "type": "offline.session.start",
        "protocol": PROTOCOL,
        "sample_rate": RATE,
        "channels": 1,
        "encoding": "pcm_s16le",
        "input_samples": len(pcm) // 2,
        **extra,
    }


def completion_message(pcm, packets):
    """Seal exactly the fixture bytes with their packet count."""
    return {
        "type": "offline.input.complete",
        "input_samples": len(pcm) // 2,
        "input_packets": packets,
        "sha256": hashlib.sha256(pcm).hexdigest(),
    }


def pcm_from_wav(path):
    """Read canonical PCM from a server recording."""
    with wave.open(str(path), "rb") as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (RATE, 1, 2)
        return audio.readframes(audio.getnframes())


class FakeSocket:
    def __init__(self):
        """Create an ASGI-compatible socket with controllable delivery."""
        self.incoming = asyncio.Queue()
        self.sent = []
        self.accepted = False
        self.closed = None

    def feed(self, payload):
        if isinstance(payload, bytes):
            self.incoming.put_nowait({"type": "websocket.receive", "bytes": payload})
        else:
            self.incoming.put_nowait({"type": "websocket.receive", "text": json.dumps(payload)})

    def disconnect(self):
        self.incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})

    async def accept(self):
        self.accepted = True

    async def receive(self):
        return await self.incoming.get()

    async def send_json(self, payload):
        self.sent.append(payload)

    async def send_bytes(self, payload):
        self.sent.append(payload)

    async def close(self, code=1000):
        self.closed = code


class MarkerRuntime:
    def __init__(self):
        """Expose marker output and public configuration for protocol tests."""
        self.source = None
        self.configuration = {"model": "scripted-marker", "api_key": "private-provider-key"}

    async def run(self, source):
        self.source = source
        end = source.sample_count + 17
        return SessionResult(
            spans=(AudioSpan(0, b"\x11\x00" * 13, "first"), AudioSpan(end - 17, b"\x22\x00" * 17, "last")),
            rendered_samples=end,
            events=({"type": "source_boundary", "sample": 0},),
            messages=(),
            metrics={"provider_compute_ns": 12345, "credentials": {"secret": "not-public"}},
        )


class OfflineProtocolTests(unittest.TestCase):
    def test_chunkization_retains_every_source_sample_and_offset_zero(self):
        pcm = bytes(range(256)) * 20
        for chunk in (1, 511, 512, 1024, 65536):
            with self.subTest(chunk=chunk):
                upload = InputUpload(len(pcm) // 2, SessionLimits())
                sequence = 0
                for offset in range(0, len(pcm) // 2, chunk):
                    upload.append(encode_packet(b"NVAI", sequence, offset, pcm[offset * 2 : (offset + chunk) * 2]))
                    sequence += 1
                self.assertEqual(upload.seal(completion_message(pcm, sequence)).pcm, pcm)

    def test_packet_rejects_bad_header_count_magic_and_size(self):
        packets = (
            b"x",
            HEADER.pack(b"NVAO", 0, 0, 1) + b"\x00\x00",
            HEADER.pack(b"NVAI", 0, 0, 1) + b"\x00",
            HEADER.pack(b"NVAI", 0, 0, 0),
            encode_packet(b"NVAI", 0, 0, b"\x00\x00" * 9),
        )
        for packet in packets:
            with self.subTest(packet=packet), self.assertRaises(OfflineProtocolError):
                decode_packet(packet, b"NVAI", 8)

    def test_input_rejects_duplicates_gaps_reordering_and_extent_overrun(self):
        for sequence, offset, pcm in ((0, 0, b"aa"), (2, 1, b"aa"), (1, 2, b"aa"), (1, 1, b"aaaa")):
            upload = InputUpload(2, SessionLimits())
            upload.append(encode_packet(b"NVAI", 0, 0, b"aa"))
            with self.subTest(sequence=sequence, offset=offset), self.assertRaises(OfflineProtocolError):
                upload.append(encode_packet(b"NVAI", sequence, offset, pcm))

    def test_eof_checks_extent_packet_count_digest_and_duplicate_completion(self):
        pcm = b"aaaa"
        for changes in ({"input_samples": 3}, {"input_packets": 2}, {"sha256": "0" * 64}, {"type": "bye"}):
            upload = InputUpload(2, SessionLimits())
            upload.append(encode_packet(b"NVAI", 0, 0, pcm))
            with self.subTest(changes=changes), self.assertRaises(OfflineProtocolError):
                upload.seal({**completion_message(pcm, 1), **changes})
        upload = InputUpload(2, SessionLimits())
        upload.append(encode_packet(b"NVAI", 0, 0, pcm))
        upload.seal(completion_message(pcm, 1))
        with self.assertRaises(OfflineProtocolError):
            upload.seal(completion_message(pcm, 1))
        with self.assertRaises(OfflineProtocolError):
            upload.append(encode_packet(b"NVAI", 1, 2, pcm))

    def test_declared_extent_must_be_entirely_uploaded(self):
        upload = InputUpload(2, SessionLimits())
        upload.append(encode_packet(b"NVAI", 0, 0, b"aa"))
        with self.assertRaises(OfflineProtocolError):
            upload.seal(completion_message(b"aaaa", 1))

    def test_handshake_rejects_wrong_format_duration_and_bool_integer(self):
        for changes in (
            {"protocol": "other"},
            {"sample_rate": 48000},
            {"channels": 2},
            {"channels": True},
            {"encoding": "float32"},
            {"input_samples": -1},
            {"input_samples": True},
            {"input_samples": RATE * 3600 + 1},
            {"config": []},
        ):
            with self.subTest(changes=changes), self.assertRaises(OfflineProtocolError):
                validate_start(start_message(b"aa", **changes), SessionLimits())

    def test_control_rejects_malformed_nonfinite_and_oversized_json(self):
        for control in ("{", "[]", '{"x":NaN}', '{"x":Infinity}', '{"x":"' + "a" * 32768 + '"}'):
            with self.subTest(control=control[:20]), self.assertRaises(OfflineProtocolError):
                parse_control(control, SessionLimits())

    def test_fingerprint_retains_config_but_excludes_credentials(self):
        first = {"vad": 0.8, "asr_api_key": "first-key", "tts": {"access_token": "first-token"}}
        second = {"tts": {"access_token": "second-token"}, "asr_api_key": "second-key", "vad": 0.8}
        self.assertEqual(configuration_fingerprint(first), configuration_fingerprint(second))
        self.assertNotEqual(configuration_fingerprint(first), configuration_fingerprint({**first, "vad": 0.5}))

    def test_bounds_reject_nonfinite_watchdog_and_nonpositive_packet_size(self):
        for changes in (
            {"max_packet_samples": 0},
            {"upload_timeout_seconds": float("nan")},
            {"max_input_samples": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                SessionLimits(**changes)


class OfflineSocketTests(unittest.IsolatedAsyncioTestCase):
    def prepared_socket(self, pcm=b"aa" * 64, chunk=511):
        socket = FakeSocket()
        socket.feed(start_message(pcm))
        packets = 0
        for offset in range(0, len(pcm) // 2, chunk):
            socket.feed(encode_packet(b"NVAI", packets, offset, pcm[offset * 2 : (offset + chunk) * 2]))
            packets += 1
        socket.feed(completion_message(pcm, packets))
        return socket

    async def test_sparse_output_and_saved_recording_match_for_all_packet_sizes(self):
        pcm = bytes(range(256)) * 31
        expected = b"\x11\x00" * 13 + bytes(len(pcm) - 26) + b"\x22\x00" * 17
        for chunk in (1, 511, 512, 4096, 65536):
            with self.subTest(chunk=chunk), tempfile.TemporaryDirectory() as root:
                socket = self.prepared_socket(pcm, chunk)
                runtime = MarkerRuntime()
                await serve_offline_session(
                    socket,
                    lambda start, runtime=runtime: runtime,
                    root,
                    limits=replace(SessionLimits(), output_packet_samples=7),
                )
                self.assertEqual(runtime.source.pcm, pcm)
                self.assertEqual(socket.closed, 1000)
                packets = [item for item in socket.sent if isinstance(item, bytes)]
                self.assertEqual(len(packets), 5)
                sparse = bytearray(len(expected))
                for sequence, packet in enumerate(packets):
                    actual_sequence, offset, audio = decode_packet(packet, b"NVAO", 7)
                    self.assertEqual(actual_sequence, sequence)
                    sparse[offset * 2 : offset * 2 + len(audio)] = audio
                self.assertEqual(sparse, expected)
                complete = socket.sent[-1]
                self.assertEqual(complete["type"], "offline.session.complete")
                self.assertEqual(complete["rendered_samples"], len(expected) // 2)
                self.assertEqual(complete["output_packets"], 5)
                self.assertEqual(
                    complete["configuration_fingerprint"], configuration_fingerprint(runtime.configuration)
                )
                self.assertNotIn(
                    "private-provider-key", json.dumps([item for item in socket.sent if isinstance(item, dict)])
                )
                self.assertNotIn("not-public", json.dumps(complete))
                directories = list(Path(root).iterdir())
                self.assertEqual(len(directories), 1)
                self.assertEqual(str(uuid.UUID(directories[0].name)), directories[0].name)
                self.assertEqual(pcm_from_wav(directories[0] / "source.wav"), pcm)
                self.assertEqual(pcm_from_wav(directories[0] / "full_output.wav"), expected)

    async def test_wall_phase_scopes_and_persisted_metrics_match_without_changing_audio_offsets(self):
        class ManualClock:
            value = 0

            def __call__(self):
                return self.value

            def advance(self, amount):
                self.value += amount

        clock = ManualClock()

        class TimedSocket(FakeSocket):
            received = 0

            async def accept(self):
                await super().accept()
                clock.advance(11)

            async def receive(self):
                message = await super().receive()
                clock.advance((7, 17, 19)[self.received])
                self.received += 1
                return message

            async def send_json(self, payload):
                await super().send_json(payload)
                clock.advance(
                    {
                        "offline.session.accepted": 13,
                        "offline.event": 47,
                        "offline.session.complete": 53,
                    }[payload["type"]]
                )

            async def send_bytes(self, payload):
                await super().send_bytes(payload)
                clock.advance(43)

            async def close(self, code=1000):
                await super().close(code)
                clock.advance(59)

        class TimedRuntime(MarkerRuntime):
            async def run(self, source):
                result = await super().run(source)
                clock.advance(31)
                return result

        async def factory(start):
            clock.advance(29)
            return TimedRuntime()

        async def timed_disk_call(function, *args, **kwargs):
            value = function(*args, **kwargs)
            clock.advance({"write_source": 23, "write_session": 37, "write_text": 41}[function.__name__])
            return value

        pcm = b"aa" * 64
        socket = TimedSocket()
        socket.feed(start_message(pcm))
        socket.feed(encode_packet(b"NVAI", 0, 0, pcm))
        socket.feed(completion_message(pcm, 1))
        with tempfile.TemporaryDirectory() as root:
            with patch("examples.generic.offline.websocket.asyncio.to_thread", side_effect=timed_disk_call):
                await serve_offline_session(socket, factory, root, clock=clock)
            directory = next(Path(root).iterdir())
            manifest = json.loads((directory / "manifest.json").read_text())
            complete = socket.sent[-1]
            self.assertEqual(manifest["metrics"], complete["metrics"])
            self.assertNotIn("not-public", json.dumps(manifest["metrics"]))
            self.assertEqual(pcm_from_wav(directory / "source.wav"), pcm)
            self.assertEqual(pcm_from_wav(directory / "full_output.wav")[:26], b"\x11\x00" * 13)
        server = complete["metrics"]["server"]
        expected = {
            "socket_accept_to_source_sealed_wall_ns": 56,
            "input_start_to_source_sealed_wall_ns": 49,
            "upload_accepted_to_source_sealed_wall_ns": 36,
            "source_recording_wall_ns": 23,
            "runtime_build_wall_ns": 29,
            "runtime_run_wall_ns": 31,
            "source_sealed_to_generation_complete_wall_ns": 83,
            "output_recording_wall_ns": 78,
            "source_sealed_to_first_output_packet_sent_wall_ns": 204,
            "output_publication_wall_ns": 133,
            "socket_accept_to_precomplete_wall_ns": 350,
            "source_sealed_to_precomplete_wall_ns": 294,
        }
        self.assertEqual({name: server[name] for name in expected}, expected)
        self.assertEqual(server["clock"], "monotonic_ns")
        self.assertIn("final_metrics_manifest_rewrite", server["precomplete_excludes"])
        self.assertEqual(clock(), 514)  # Manifest rewrite, complete send and close are outside the scopes.
        packets = [decode_packet(item, b"NVAO", 1600) for item in socket.sent if isinstance(item, bytes)]
        self.assertEqual([packet[1] for packet in packets], [0, 64])

    async def test_bad_upload_does_not_create_provider_or_recording(self):
        socket = FakeSocket()
        socket.feed(start_message(b"aaaa"))
        socket.feed(encode_packet(b"NVAI", 0, 0, b"aaaa"))
        socket.feed({**completion_message(b"aaaa", 1), "sha256": "incorrect"})
        factory_calls = []
        with tempfile.TemporaryDirectory() as root:
            await serve_offline_session(socket, lambda start: factory_calls.append(start), root)
            self.assertEqual(factory_calls, [])
            self.assertEqual(list(Path(root).iterdir()), [])
        self.assertEqual(socket.sent[-1]["code"], "input_digest")
        self.assertEqual(socket.closed, 1008)

    async def test_disconnect_during_upload_is_not_eof(self):
        socket = FakeSocket()
        socket.feed(start_message(b"aaaa"))
        socket.feed(encode_packet(b"NVAI", 0, 0, b"aa"))
        socket.disconnect()
        calls = []
        with tempfile.TemporaryDirectory() as root:
            await serve_offline_session(socket, lambda start: calls.append(start), root)
        self.assertFalse(calls)
        self.assertEqual([item["type"] for item in socket.sent], ["offline.session.accepted"])

    async def test_disconnect_after_seal_cancels_provider_and_preserves_source(self):
        socket = self.prepared_socket()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        class BlockingRuntime:
            async def run(self, source):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        with tempfile.TemporaryDirectory() as root:
            task = asyncio.create_task(serve_offline_session(socket, lambda start: BlockingRuntime(), root))
            await asyncio.wait_for(started.wait(), timeout=1)
            socket.disconnect()
            await asyncio.wait_for(task, timeout=1)
            self.assertTrue(cancelled.is_set())
            directory = next(Path(root).iterdir())
            self.assertTrue((directory / "source.wav").exists())
            self.assertEqual(json.loads((directory / "failure.json").read_text())["code"], "client_disconnected")
        self.assertFalse(
            any(item.get("type") == "offline.session.complete" for item in socket.sent if isinstance(item, dict))
        )

    async def test_duplicate_eof_after_seal_is_rejected_and_not_completed(self):
        socket = self.prepared_socket()
        socket.feed(completion_message(b"aa" * 64, 1))
        with tempfile.TemporaryDirectory() as root:
            await serve_offline_session(socket, lambda start: MarkerRuntime(), root)
        self.assertEqual(socket.sent[-1]["code"], "input_sealed")
        self.assertFalse(
            any(item.get("type") == "offline.session.complete" for item in socket.sent if isinstance(item, dict))
        )

    async def test_late_duplicate_eof_during_output_cannot_produce_success(self):
        class LateInputSocket(FakeSocket):
            async def send_bytes(self, payload):
                await super().send_bytes(payload)
                self.feed(completion_message(b"aa" * 64, 1))

        socket = LateInputSocket()
        pcm = b"aa" * 64
        socket.feed(start_message(pcm))
        socket.feed(encode_packet(b"NVAI", 0, 0, pcm))
        socket.feed(completion_message(pcm, 1))
        with tempfile.TemporaryDirectory() as root:
            await serve_offline_session(socket, lambda start: MarkerRuntime(), root)
        self.assertEqual(socket.sent[-1]["code"], "input_sealed")
        self.assertFalse(
            any(item.get("type") == "offline.session.complete" for item in socket.sent if isinstance(item, dict))
        )

    async def test_300_second_silence_session_writes_exact_horizon_without_output_packets(self):
        pcm = bytes(300 * RATE * 2)
        socket = self.prepared_socket(pcm, 65536)

        class SilentRuntime:
            async def run(self, source):
                return SessionResult((), source.sample_count, (), ())

        with tempfile.TemporaryDirectory() as root:
            await serve_offline_session(socket, lambda start: SilentRuntime(), root)
            directory = next(Path(root).iterdir())
            self.assertEqual(pcm_from_wav(directory / "source.wav"), pcm)
            self.assertEqual(pcm_from_wav(directory / "full_output.wav"), pcm)
        self.assertFalse(any(isinstance(item, bytes) for item in socket.sent))
        self.assertEqual(socket.sent[-1]["rendered_samples"], 300 * RATE)
        self.assertEqual(socket.sent[-1]["output_packets"], 0)
        self.assertIsNone(socket.sent[-1]["metrics"]["server"]["source_sealed_to_first_output_packet_sent_wall_ns"])

    async def test_upload_watchdog_fails_without_manufacturing_a_turn(self):
        socket = FakeSocket()
        socket.feed(start_message(b"aaaa"))
        calls = []
        with tempfile.TemporaryDirectory() as root:
            await serve_offline_session(
                socket,
                lambda start: calls.append(start),
                root,
                limits=replace(SessionLimits(), upload_timeout_seconds=0.01),
            )
        self.assertFalse(calls)
        self.assertEqual(socket.sent[-1]["code"], "watchdog_timeout")

    async def test_runtime_watchdog_cancels_provider_without_output(self):
        socket = self.prepared_socket()
        cancelled = asyncio.Event()

        class BlockingRuntime:
            async def run(self, source):
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        with tempfile.TemporaryDirectory() as root:
            await serve_offline_session(
                socket,
                lambda start: BlockingRuntime(),
                root,
                limits=replace(SessionLimits(), execution_timeout_seconds=0.01),
            )
        self.assertTrue(cancelled.is_set())
        self.assertEqual(socket.sent[-1]["code"], "watchdog_timeout")
        self.assertFalse(any(isinstance(item, bytes) for item in socket.sent))

    async def test_provider_error_is_not_echoed_and_source_is_preserved(self):
        socket = self.prepared_socket()

        class FailingRuntime:
            async def run(self, source):
                raise RuntimeError("Authorization: secret-private-key")

        with tempfile.TemporaryDirectory() as root:
            await serve_offline_session(socket, lambda start: FailingRuntime(), root)
            directory = next(Path(root).iterdir())
            self.assertTrue((directory / "source.wav").exists())
            self.assertNotIn("secret-private-key", (directory / "failure.json").read_text())
        self.assertEqual(socket.sent[-1]["code"], "session_failed")
        self.assertNotIn("secret-private-key", json.dumps(socket.sent))

    async def test_async_factory_observes_validated_recording_before_runtime_start(self):
        socket = self.prepared_socket()
        with tempfile.TemporaryDirectory() as root:

            async def factory(start):
                directory = next(Path(root).iterdir())
                self.assertEqual(pcm_from_wav(directory / "source.wav"), b"aa" * 64)
                return MarkerRuntime()

            await serve_offline_session(socket, factory, root)
        self.assertEqual(socket.sent[-1]["status"], "ok")

    async def test_invalid_result_is_rejected_before_any_output_packet(self):
        socket = self.prepared_socket()

        class InvalidRuntime:
            async def run(self, source):
                return SessionResult((AudioSpan(0, b"aa" * 9, "a"), AudioSpan(4, b"aa", "b")), 64, (), ())

        with tempfile.TemporaryDirectory() as root:
            await serve_offline_session(socket, lambda start: InvalidRuntime(), root)
        self.assertEqual(socket.sent[-1]["code"], "session_failed")
        self.assertFalse(any(isinstance(item, bytes) for item in socket.sent))


class OfflineASGITests(unittest.TestCase):
    def test_actual_fastapi_websocket_upload_seal_and_sparse_response(self):
        pcm = b"aa" * 64
        with tempfile.TemporaryDirectory() as root:
            app = FastAPI()

            @app.websocket("/api/ws/offline")
            async def route(socket: WebSocket):
                await serve_offline_session(socket, lambda start: MarkerRuntime(), root)

            with TestClient(app) as client, client.websocket_connect("/api/ws/offline") as socket:
                socket.send_json(start_message(pcm))
                accepted = socket.receive_json()
                self.assertEqual(accepted["protocol"], PROTOCOL)
                self.assertEqual(accepted["output_sample_rate"], RATE)
                socket.send_bytes(encode_packet(b"NVAI", 0, 0, pcm[:2]))
                socket.send_bytes(encode_packet(b"NVAI", 1, 1, pcm[2:]))
                socket.send_json(completion_message(pcm, 2))
                first = decode_packet(socket.receive_bytes(), b"NVAO", 1600)
                last = decode_packet(socket.receive_bytes(), b"NVAO", 1600)
                self.assertEqual(first[1], 0)
                self.assertEqual(last[1], 64)
                event = socket.receive_json()
                self.assertEqual(event["event"]["sample"], 0)
                complete = socket.receive_json()
                self.assertEqual(complete["input_samples"], 64)
                self.assertEqual(complete["rendered_samples"], 81)
                self.assertEqual(complete["output_packets"], 2)


if __name__ == "__main__":
    unittest.main()
