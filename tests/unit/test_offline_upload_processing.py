# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D102, D107

"""Pre-EOF work is cancellable and never makes an unvalidated input successful."""

import asyncio
import tempfile
import unittest
from dataclasses import replace

from test_offline_websocket import FakeSocket, completion_message, start_message

from examples.generic.offline.models import SessionResult
from examples.generic.offline.protocol import SessionLimits, encode_packet
from examples.generic.offline.websocket import serve_offline_session


class UploadRuntime:
    """Consume real prefix windows while exposing startup/cleanup boundaries."""

    configuration = {"model": "fixture"}

    def __init__(self):
        self.started = asyncio.Event()
        self.first_chunk = asyncio.Event()
        self.closed = asyncio.Event()
        self.source = None
        self.chunks = []

    async def run(self, source, *, event_sink):
        self.source = source
        self.started.set()
        event_sink({"type": "response_requested", "sample": 0, "response_id": "r1"})
        event_sink(
            {"type": "provider_observation", "sample": 0, "frame": "LLMTextFrame", "source": "llm", "text": "Hello"}
        )
        async for chunk in source.async_chunks():
            self.chunks.append(chunk)
            self.first_chunk.set()
        return SessionResult((), source.sample_count, (), (), {})

    async def aclose(self):
        self.closed.set()


class UploadProcessingTests(unittest.IsolatedAsyncioTestCase):
    """Exercise the wire EOF/hash/cancellation contract with controlled ingress."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.socket = FakeSocket()
        self.runtime = UploadRuntime()
        self.task = None

    async def asyncTearDown(self):
        if self.task is not None:
            if not self.task.done():
                self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    def start(self, pcm, *, factory=None, **limits):
        self.socket.feed(start_message(pcm, completion="text", input_processing="during_upload"))
        self.task = asyncio.create_task(
            serve_offline_session(
                self.socket,
                factory or (lambda start: self.runtime),
                self.temp.name,
                limits=replace(SessionLimits(), **limits),
            )
        )

    async def test_analysis_starts_before_upload_finishes_but_text_waits_for_validated_eof(self):
        pcm = b"\x01\0" * 1025
        self.start(pcm)
        self.socket.feed(encode_packet(b"NVAI", 0, 0, pcm[:1024]))
        await asyncio.wait_for(self.runtime.first_chunk.wait(), 2)
        self.assertFalse(self.runtime.source.sealed)
        self.assertEqual([x["type"] for x in self.socket.sent], ["offline.session.accepted"])
        # Split the next canonical window across packet boundaries.
        self.socket.feed(encode_packet(b"NVAI", 1, 512, pcm[1024:1536]))
        self.socket.feed(encode_packet(b"NVAI", 2, 768, pcm[1536:]))
        self.socket.feed(completion_message(pcm, 3))
        await asyncio.wait_for(self.task, 2)
        self.assertEqual([len(x.pcm) // 2 for x in self.runtime.chunks], [512, 512, 1])
        self.assertEqual(b"".join(x.pcm for x in self.runtime.chunks), pcm)
        self.assertTrue(self.runtime.source.sealed)
        self.assertTrue(self.runtime.closed.is_set())
        self.assertEqual(self.socket.sent[1]["type"], "offline.runtime.configuration")
        self.assertEqual(self.socket.sent[-1]["type"], "offline.events.complete")

    async def test_digest_rejection_aborts_provisional_analysis_without_publishing_text(self):
        pcm = b"\0\0" * 1024
        self.start(pcm)
        self.socket.feed(encode_packet(b"NVAI", 0, 0, pcm))
        await asyncio.wait_for(self.runtime.first_chunk.wait(), 2)
        eof = completion_message(pcm, 1)
        eof["sha256"] = "0" * 64
        self.socket.feed(eof)
        await asyncio.wait_for(self.task, 2)
        self.assertTrue(self.runtime.closed.is_set())
        self.assertFalse(self.runtime.source.sealed)
        self.assertEqual(self.socket.sent[-1]["code"], "input_digest")
        self.assertFalse(any(x.get("type") == "offline.event" for x in self.socket.sent))

    async def test_disconnect_during_upload_wakes_native_reader_and_closes_runtime(self):
        pcm = b"\0\0" * 1024
        self.start(pcm)
        self.socket.feed(encode_packet(b"NVAI", 0, 0, pcm[:1024]))
        await asyncio.wait_for(self.runtime.first_chunk.wait(), 2)
        self.socket.disconnect()
        await asyncio.wait_for(self.task, 2)
        self.assertTrue(self.runtime.closed.is_set())
        self.assertFalse(self.runtime.source.sealed)
        self.assertFalse(any(x.get("type") == "offline.events.complete" for x in self.socket.sent))

    async def test_ack_only_probe_does_not_initialize_services(self):
        called = []
        self.start(b"\0\0" * 512, factory=lambda start: called.append(start))
        self.socket.disconnect()
        await asyncio.wait_for(self.task, 2)
        self.assertEqual(called, [])

    async def test_provider_setup_error_interrupts_a_stalled_upload(self):
        def fail(start):
            raise RuntimeError("private provider failure")

        self.start(b"\0\0" * 1024, factory=fail)
        self.socket.feed(encode_packet(b"NVAI", 0, 0, b"\0\0" * 512))
        await asyncio.wait_for(self.task, 2)
        self.assertEqual(self.socket.sent[-1]["code"], "session_failed")
        self.assertNotIn("private provider failure", str(self.socket.sent))

    async def test_upload_deadline_aborts_early_services_without_fabricated_eof(self):
        self.start(b"\0\0" * 1024, upload_timeout_seconds=0.05)
        self.socket.feed(encode_packet(b"NVAI", 0, 0, b"\0\0" * 512))
        await asyncio.wait_for(self.runtime.first_chunk.wait(), 2)
        await asyncio.wait_for(self.task, 2)
        self.assertTrue(self.runtime.closed.is_set())
        self.assertEqual(self.socket.sent[-1]["code"], "watchdog_timeout")

    async def test_pre_eof_buffer_overflow_is_explicit_and_not_blame_for_slow_client(self):
        self.start(b"\0\0" * 1024, max_pending_stream_events=1)
        self.socket.feed(encode_packet(b"NVAI", 0, 0, b"\0\0" * 512))
        await asyncio.wait_for(self.task, 2)
        self.assertEqual(self.socket.sent[-1]["code"], "provisional_event_overflow")
        self.assertTrue(self.runtime.closed.is_set())

    async def test_empty_stream_cannot_publish_completion_before_validated_eof(self):
        returned = asyncio.Event()

        class EmptyRuntime(UploadRuntime):
            async def run(inner, source, *, event_sink):
                returned.set()
                return SessionResult((), source.sample_count, (), (), {})

        self.runtime = EmptyRuntime()
        pcm = b"\0\0" * 512
        self.start(pcm)
        self.socket.feed(encode_packet(b"NVAI", 0, 0, pcm))
        await asyncio.wait_for(returned.wait(), 2)
        await asyncio.sleep(0)
        self.assertEqual([x["type"] for x in self.socket.sent], ["offline.session.accepted"])
        self.socket.feed(completion_message(pcm, 1))
        await asyncio.wait_for(self.task, 2)
        self.assertEqual(self.socket.sent[-1]["type"], "offline.events.complete")

    async def test_server_cancellation_preserves_real_provider_cleanup_failure(self):
        class FailedCleanup(UploadRuntime):
            async def aclose(inner):
                inner.closed.set()
                raise RuntimeError("private cleanup diagnostic")

        self.runtime = FailedCleanup()
        self.start(b"\0\0" * 1024)
        self.socket.feed(encode_packet(b"NVAI", 0, 0, b"\0\0" * 512))
        await asyncio.wait_for(self.runtime.first_chunk.wait(), 2)
        self.task.cancel()
        await asyncio.wait_for(self.task, 2)
        self.assertTrue(self.runtime.closed.is_set())
        self.assertEqual(self.socket.sent[-1]["code"], "session_failed")
        self.assertNotIn("private cleanup diagnostic", str(self.socket.sent))
