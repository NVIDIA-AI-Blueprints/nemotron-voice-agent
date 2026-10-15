# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Exercise the actual benchmark client against the real server and scheduler.

The duplex adapter substitutes only network I/O. Detection and provider outputs
are fixed traces, so this verifies causal scheduling, wire contracts, recording,
and upload invariance rather than recognition accuracy or live service latency.
"""

import asyncio
import copy
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.generic.offline.engine import OfflineSessionEngine
from examples.generic.offline.models import (
    DetectionTrace,
    GeneratedAudio,
    RecognitionSegment,
    RecognitionTrace,
    ResponseTrace,
    SpeechInterval,
    SpokenWord,
    UserTurn,
    samples_to_ns,
)
from examples.generic.offline.websocket import serve_offline_session

_CLIENT_PATH = Path(__file__).resolve().parents[2] / "benchmarking_tools" / "offline-eval" / "client.py"
_CLIENT_SPEC = importlib.util.spec_from_file_location("nva_offline_integration_client", _CLIENT_PATH)
assert _CLIENT_SPEC is not None and _CLIENT_SPEC.loader is not None
client = importlib.util.module_from_spec(_CLIENT_SPEC)
sys.modules[_CLIENT_SPEC.name] = client
_CLIENT_SPEC.loader.exec_module(client)
_CLOSED = object()


def at(seconds: str) -> int:
    """Use decimal positions to keep every expected boundary sample-exact."""
    return client.seconds_to_samples(seconds)


def marker(count: int, value: int) -> bytes:
    """Create identifiable PCM for source ownership and output placement."""
    return struct.pack("<h", value) * count


class DuplexSocket:
    """Bounded bidirectional ASGI transport with no presentation clock."""

    def __init__(self):
        """Model backpressure in both directions without opening a TCP socket."""
        self.incoming = asyncio.Queue(maxsize=16)
        self.outgoing = asyncio.Queue(maxsize=16)
        self.accepted = asyncio.Event()

    async def accept(self):
        """Make the server's explicit acceptance visible to the client."""
        self.accepted.set()

    async def receive(self):
        """Expose uploads in the shape Starlette's WebSocket receives."""
        return await self.incoming.get()

    async def send_json(self, message):
        """Serialize controls as a real network transport would."""
        await self.outgoing.put(json.dumps(message))

    async def send_bytes(self, payload):
        """Preserve binary packet contents and backpressure."""
        await self.outgoing.put(payload)

    async def close(self, code=1000):
        """Close output only after preceding server messages are queued."""
        await self.outgoing.put(_CLOSED)


class ClientSocket:
    """The send/recv/iterator interface used by websockets' actual client."""

    def __init__(self, server_socket):
        """Attach the opposite ends of the duplex queues."""
        self.server_socket = server_socket

    async def send(self, payload):
        """Translate client bytes and text into ASGI input messages."""
        key = "bytes" if isinstance(payload, bytes) else "text"
        await self.server_socket.incoming.put({"type": "websocket.receive", key: payload})

    async def recv(self):
        """Return server packets with their original text/binary distinction."""
        payload = await self.server_socket.outgoing.get()
        if payload is _CLOSED:
            raise EOFError("Server closed the in-process connection")
        return payload

    def __aiter__(self):
        """Allow the sample client's unmodified receive loop."""
        return self

    async def __anext__(self):
        """End iteration only on explicit transport closure."""
        try:
            return await self.recv()
        except EOFError as error:
            raise StopAsyncIteration from error


class InProcessConnection:
    """Connect the unmodified benchmark coroutine to its actual server entrypoint."""

    def __init__(self, factory, record_root):
        """Create a fresh server transport per client run."""
        self.socket = DuplexSocket()
        self.factory = factory
        self.record_root = record_root
        self.task = None

    async def __aenter__(self):
        """Start real session validation and await server acceptance."""
        self.task = asyncio.create_task(serve_offline_session(self.socket, self.factory, self.record_root))
        await asyncio.wait_for(self.socket.accepted.wait(), timeout=2)
        return ClientSocket(self.socket)

    async def __aexit__(self, exception_type, exception, traceback):
        """Clean up on failure; drain normal server completion before returning."""
        if exception_type is not None and self.task is not None:
            self.task.cancel()
        if self.task is not None:
            await asyncio.wait_for(asyncio.gather(self.task, return_exceptions=True), timeout=2)


class ContinuousRecognition:
    """One recognition invocation covering all three source turns."""

    def __init__(self, turns):
        """Retain immutable ownership and a measured 50 ms readiness delay."""
        self.turns = turns
        self.calls = 0
        self.source_samples = None

    async def recognize(self, source):
        """Return final portions from one drained continuous stream."""
        self.calls += 1
        self.source_samples = source.sample_count
        return RecognitionTrace(
            tuple(
                RecognitionSegment(
                    f"question {index + 1}",
                    turn.start_sample,
                    turn.end_sample,
                    turn.endpoint_sample + at("0.05"),
                )
                for index, turn in enumerate(self.turns)
            ),
            {"continuous_stream": True, "validation_kind": "fixed_provider_trace"},
        )


class SourceDetector:
    """Source-only decisions with confirmation delay and endpoint silence."""

    def __init__(self, turns, source):
        """Bind decisions to source bytes independently of packetization."""
        self.turns = turns
        self.expected_source = source
        self.calls = 0

    async def detect(self, source):
        """Validate source preservation and return immutable source offsets."""
        if source.pcm != self.expected_source:
            raise ValueError("Source samples were lost or modified")
        self.calls += 1
        return DetectionTrace(
            tuple(
                SpeechInterval(turn.start_sample, turn.end_sample, turn.start_sample + at("0.1")) for turn in self.turns
            ),
            self.turns,
        )


class TraceResponder:
    """Provisional generated audio with deterministic measured production delays."""

    def __init__(self):
        """Store snapshots to test history and future-input isolation."""
        self.requests = []
        self.responses = (
            ResponseTrace(
                "first remainder",
                (GeneratedAudio(marker(at("4"), 1111), samples_to_ns(at("0.1"))),),
                samples_to_ns(at("0.2")),
                (SpokenWord("first", 0, at("0.5")), SpokenWord("remainder", at("0.5"), at("4"))),
            ),
            ResponseTrace(
                "second reply",
                (
                    GeneratedAudio(marker(at("0.4"), 2222), samples_to_ns(at("0.1"))),
                    GeneratedAudio(marker(at("0.5"), 2223), samples_to_ns(at("0.6"))),
                ),
                samples_to_ns(at("0.7")),
            ),
            ResponseTrace(
                "last reply",
                (GeneratedAudio(marker(at("0.75"), 3333), samples_to_ns(at("0.2"))),),
                samples_to_ns(at("0.3")),
            ),
        )

    async def respond(self, messages, turn_id):
        """Generate without changing authoritative conversation history."""
        index = len(self.requests)
        self.requests.append((copy.deepcopy(messages), turn_id))
        return self.responses[index]


def fixture():
    """Prepare 300 seconds containing three source speech intervals and silence."""
    turns = (
        UserTurn("one", at("1"), at("3"), at("3.8"), samples_to_ns(at("0.02"))),
        UserTurn("two", at("5"), at("6"), at("6.8"), samples_to_ns(at("0.02"))),
        UserTurn("three", at("298"), at("299.2"), at("300"), samples_to_ns(at("0.02"))),
    )
    source = bytearray(at("300") * 2)
    for index, turn in enumerate(turns):
        source[turn.start_sample * 2 : turn.end_sample * 2] = marker(turn.end_sample - turn.start_sample, 101 + index)
    return turns, bytes(source)


def expected_output() -> bytes:
    """Place only played samples, including a TTS availability gap and EOF tail."""
    output = bytearray(at("301") * 2)
    for start, end, value in (
        ("3.95", "5.1", 1111),
        ("6.95", "7.35", 2222),
        ("7.45", "7.95", 2223),
        ("300.25", "301", 3333),
    ):
        output[at(start) * 2 : at(end) * 2] = marker(at(end) - at(start), value)
    return bytes(output)


class OfflineActualClientIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Validate actual client/server/scheduler behavior with a fixed provider trace."""

    async def test_300_second_session_is_identical_across_bulk_and_paced_uploads(self):
        """Keep offsets, interruptions, history, and exact WAVs upload-independent."""
        turns, source = fixture()
        expected = expected_output()
        traces = []
        output_files = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server_records = root / "server"
            previous_records = set()
            for chunk_samples, speed in ((511, 0), (512, 0), (32768, 0), (512, 200)):
                with self.subTest(chunk_samples=chunk_samples, upload_speed=speed):
                    detector = SourceDetector(turns, source)
                    recognizer = ContinuousRecognition(turns)
                    responder = TraceResponder()
                    engine = OfflineSessionEngine(detector, recognizer, responder)
                    engine.configuration = {"validation_kind": "fixed_provider_trace", "continuous_asr": True}
                    connection = InProcessConnection(lambda start, engine=engine: engine, server_records)
                    output_dir = root / f"client_{chunk_samples}_{speed}"
                    with patch("websockets.asyncio.client.connect", return_value=connection):
                        result = await client.run_client(
                            "ws://in-process/api/ws/offline",
                            source,
                            output_dir,
                            chunk_samples=chunk_samples,
                            upload_speed=speed,
                            watchdog=30,
                        )
                    self.assertEqual(detector.calls, 1)
                    self.assertEqual(recognizer.calls, 1)
                    self.assertEqual(recognizer.source_samples, at("300"))
                    self.assertEqual([turn_id for _, turn_id in responder.requests], ["one", "two", "three"])
                    self.assertEqual(responder.requests[0][0], [{"role": "user", "content": "question 1"}])
                    self.assertEqual(responder.requests[1][0][1], {"role": "assistant", "content": "first"})
                    self.assertNotIn("remainder", str(responder.requests[1][0]))
                    self.assertNotIn("question 3", str(responder.requests[1][0]))
                    self.assertEqual(result["source_duration_s"], 300)
                    self.assertEqual(result["output_duration_s"], 301)
                    self.assertEqual(result["rendered_samples"], at("301"))
                    self.assertEqual(client.read_pcm(output_dir / "full_output.wav"), expected)
                    self.assertEqual(client.read_pcm(output_dir / "output.wav"), expected[: at("300") * 2])
                    new_records = set(server_records.iterdir()) - previous_records
                    self.assertEqual(len(new_records), 1)
                    recording = next(iter(new_records))
                    previous_records |= new_records
                    self.assertEqual(client.read_pcm(recording / "source.wav"), source)
                    server_output = (recording / "full_output.wav").read_bytes()
                    self.assertEqual(server_output, (output_dir / "full_output.wav").read_bytes())
                    output_files.append(server_output)
                    server_events = [json.loads(line) for line in (recording / "events.jsonl").read_text().splitlines()]
                    received_events = [
                        json.loads(line) for line in (output_dir / "events.jsonl").read_text().splitlines()
                    ]
                    wire_events = [event["event"] for event in received_events if event["type"] == "offline.event"]
                    self.assertEqual(wire_events, server_events)
                    traces.append(server_events)
                    interrupted = next(event for event in server_events if event["type"] == "response_interrupted")
                    self.assertEqual(interrupted["sample"], at("5.1"))
                    self.assertEqual(interrupted["generated_samples"], at("4"))
                    self.assertEqual(interrupted["committed_samples"], at("1.15"))
                    self.assertEqual(interrupted["history_policy"], "aligned_played_prefix")
                    self.assertEqual(
                        [event["sample"] for event in server_events if event["type"] == "bot_speech_start"],
                        [at("3.95"), at("6.95"), at("300.25")],
                    )
                    self.assertEqual(
                        [event["sample"] for event in server_events if event["type"] == "user_turn_ready"],
                        [at("3.85"), at("6.85"), at("300.05")],
                    )
                    manifest = json.loads((recording / "manifest.json").read_text())
                    self.assertEqual(manifest["metrics"]["response_compute_ns"], 1_200_000_000)
                    self.assertEqual(manifest["metrics"]["committed_samples"], at("2.8"))
                    self.assertEqual(manifest["metrics"]["generated_samples"], at("5.65"))
                    self.assertEqual(manifest["messages"][-1], {"role": "assistant", "content": "last reply"})
            self.assertTrue(all(events == traces[0] for events in traces))
            self.assertTrue(all(wav == output_files[0] for wav in output_files))


if __name__ == "__main__":
    unittest.main()
