# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Plan section 13.5: channel auth, thread-confined RPCs, bounded pools, and cancellation (fake riva)."""

from __future__ import annotations

import asyncio
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from examples.frontend_backend_verdict.voice.speech import asr_riva, tts_riva
from examples.frontend_backend_verdict.voice.speech.catalog import SpeechEndpoint, build_endpoint


class FakeAuth:
    instances: list[FakeAuth] = []

    def __init__(self, ssl_root_cert, use_ssl, uri, metadata_args) -> None:
        self.use_ssl, self.uri, self.metadata = use_ssl, uri, metadata_args
        FakeAuth.instances.append(self)

    def get_auth_metadata(self):
        return self.metadata


class FakeASRService:
    def __init__(self, auth) -> None:
        self.auth = auth
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.threads: list[set[str]] = []

    def streaming_response_generator(self, audio_chunks, streaming_config):
        touched: set[str] = {threading.current_thread().name}
        self.threads.append(touched)

        def generate():
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                total = 0
                for chunk in audio_chunks:
                    touched.add(threading.current_thread().name)
                    total += len(chunk)
                time.sleep(0.05)
                interim = SimpleNamespace(alternatives=[SimpleNamespace(transcript="heard")], is_final=False)
                yield SimpleNamespace(results=[interim])
                final = SimpleNamespace(alternatives=[SimpleNamespace(transcript=f"heard {total}")], is_final=True)
                touched.add(threading.current_thread().name)
                yield SimpleNamespace(results=[final])
            finally:
                with self.lock:
                    self.active -= 1

        return generate()


class FakeCall:
    def __init__(self) -> None:
        self.cancelled = threading.Event()
        self.thread = ""

    def __iter__(self):
        """Stream fake audio until cancelled, recording the consuming thread."""
        self.thread = threading.current_thread().name
        for index in range(100):
            if self.cancelled.is_set():
                return
            yield SimpleNamespace(audio=bytes([index % 256]) * 320)
            time.sleep(0.01)

    def cancel(self) -> None:
        self.cancelled.set()


class FakeTTSService:
    def __init__(self, auth) -> None:
        self.auth = auth
        self.calls: list[FakeCall] = []
        self.stub = SimpleNamespace(SynthesizeOnline=self._synthesize)

    def _synthesize(self, requests, metadata):
        list(requests)
        call = FakeCall()
        self.calls.append(call)
        return call


ENDPOINT = SpeechEndpoint(kind="asr", server="nemo-speech:50051", language_code="en-US")


class AsrAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_streams_are_thread_confined_bounded_and_private(self) -> None:
        with (
            mock.patch.object(asr_riva.riva.client, "Auth", FakeAuth),
            mock.patch.object(asr_riva.riva.client, "ASRService", FakeASRService),
        ):
            recognizer = asr_riva.RivaStreamingRecognizer(ENDPOINT, max_streams=2)
            interims: list[str] = []
            streams = [recognizer.open(sample_rate=16000, on_interim=interims.append) for _ in range(4)]
            for index, stream in enumerate(streams):
                await stream.push(b"\x00" * (100 * (index + 1)))
            results = await asyncio.gather(*(stream.finish() for stream in streams))
            service = recognizer._service
            await recognizer.aclose()
        self.assertEqual(results, [f"heard {100 * (i + 1)}" for i in range(4)])
        self.assertLessEqual(service.max_active, 2)
        for touched in service.threads:
            self.assertEqual(len(touched), 1, f"one stream used by several threads: {touched}")
        self.assertEqual(interims.count("heard"), 4)

    async def test_cancel_releases_the_worker(self) -> None:
        with (
            mock.patch.object(asr_riva.riva.client, "Auth", FakeAuth),
            mock.patch.object(asr_riva.riva.client, "ASRService", FakeASRService),
        ):
            recognizer = asr_riva.RivaStreamingRecognizer(ENDPOINT, max_streams=1)
            first = recognizer.open(sample_rate=16000)
            await first.push(b"\x00" * 10)
            await first.cancel()
            second = recognizer.open(sample_rate=16000)  # would starve if the single thread were held
            await second.push(b"\x00" * 20)
            self.assertEqual(await asyncio.wait_for(second.finish(), timeout=5), "heard 20")
            await recognizer.aclose()


class TtsAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_closing_the_iterator_cancels_the_rpc_and_frees_the_thread(self) -> None:
        endpoint = SpeechEndpoint(kind="tts", server="nemo-speech:50051", voice_id="John")
        with (
            mock.patch.object(tts_riva.riva.client, "Auth", FakeAuth),
            mock.patch.object(tts_riva.riva.client, "SpeechSynthesisService", FakeTTSService),
        ):
            synth = tts_riva.RivaSynthesizer(endpoint, sample_rate=22050, max_streams=1)
            stream = synth.synthesize("Hello there.")
            first = await stream.__anext__()
            self.assertEqual(len(first), 320)
            await stream.aclose()
            call = synth._service.calls[0]
            self.assertTrue(call.cancelled.wait(timeout=2))
            chunks = [chunk async for chunk in synth.synthesize("Second sentence.")]  # the single thread is free again
            self.assertEqual(len(chunks), 100)
            self.assertEqual(len({c.thread for c in synth._service.calls}), 1)
            await synth.aclose()


class AuthTests(unittest.TestCase):
    def test_nvcf_endpoint_uses_ssl_function_id_and_bearer(self) -> None:
        endpoint = build_endpoint(
            "asr",
            {"server": "grpc.nvcf.nvidia.com:443", "function_id": "fn-123", "model": "m"},
            source="test",
            environ={"NVIDIA_API_KEY": "nvapi-secret"},
        )
        FakeAuth.instances.clear()
        with (
            mock.patch.object(asr_riva.riva.client, "Auth", FakeAuth),
            mock.patch.object(asr_riva.riva.client, "ASRService", FakeASRService),
        ):
            asr_riva.RivaStreamingRecognizer(endpoint, max_streams=1)
        auth = FakeAuth.instances[-1]
        self.assertTrue(auth.use_ssl)
        self.assertEqual(auth.metadata, [["function-id", "fn-123"], ["authorization", "Bearer nvapi-secret"]])
        self.assertNotIn("nvapi-secret", endpoint.describe())
        self.assertNotIn("nvapi-secret", repr(endpoint))

    def test_local_endpoint_is_insecure_without_function_id(self) -> None:
        endpoint = build_endpoint("tts", {"server": "nemo-speech:50051"}, source="test", environ={})
        self.assertFalse(endpoint.use_ssl)
        self.assertEqual(endpoint.metadata(), [])
