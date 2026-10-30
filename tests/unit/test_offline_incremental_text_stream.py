# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D102, D107

"""Text events must precede native TTS drain, with owned cancellation and bounds."""

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

from grpc import RpcError, StatusCode
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from test_offline_incremental_engine import ControlledASR, ControlledDetector, ControlledResponses, vad_start, vad_stop
from test_offline_native_response import ScriptedLLM, ScriptedTTS
from test_offline_websocket import FakeSocket, completion_message, start_message

from examples.generic.offline.factory import GenericOfflineRuntime
from examples.generic.offline.incremental import IncrementalSessionEngine
from examples.generic.offline.models import AudioSource, SessionResult
from examples.generic.offline.native_response import NativeResponseSession
from examples.generic.offline.protocol import (
    PROTOCOL,
    SessionLimits,
    configuration_fingerprint,
    encode_packet,
    validate_start,
)
from examples.generic.offline.websocket import serve_offline_session


class ObservedSocket(FakeSocket):
    """Allow assertions at the actual send boundary without polling sleeps."""

    def __init__(self):
        super().__init__()
        self.delivered = asyncio.Queue()

    async def send_json(self, payload):
        await super().send_json(payload)
        self.delivered.put_nowait(payload)


class IncrementalTextStreamTests(unittest.IsolatedAsyncioTestCase):
    """Exercise the live protocol with fake I/O and real native processor queues."""

    async def asyncSetUp(self):
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        errors = []
        loop.set_exception_handler(lambda loop, context: errors.append(context))

        async def assert_loop_clean():
            # Registered first, so task/resource cancellation cleanups run
            # before checking callbacks scheduled during their finalizers.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            loop.set_exception_handler(previous)
            self.assertFalse(errors, [type(error.get("exception")).__name__ for error in errors])

        self.addAsyncCleanup(assert_loop_clean)

    def socket(self):
        socket = ObservedSocket()
        pcm = b"\0\0" * 16000
        socket.feed(start_message(pcm, completion="text"))
        socket.feed(encode_packet(b"NVAI", 0, 0, pcm))
        socket.feed(completion_message(pcm, 1))
        return socket

    async def test_real_engine_streams_llm_end_while_native_tts_is_blocked_then_awaits_cleanup(self):
        """A client can finish at raw LLM EOF without releasing native synthesis."""
        instances = {}

        def llm_factory(capture):
            instances["llm"] = ScriptedLLM(capture)
            instances["llm"].release.set()
            return instances["llm"]

        def tts_factory(capture):
            instances["tts"] = ScriptedTTS(capture)
            return instances["tts"]

        responder = NativeResponseSession(llm_factory, tts_factory, sample_rate=22050, failure_timeout=2)
        engine = IncrementalSessionEngine(
            ControlledDetector([vad_start(512), vad_stop(4096)]),
            ControlledASR([(3072, 3584, "Question", True)]),
            responder,
            params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0)]
                ),
                empty_user_turn=None,
            ),
            output_sample_rate=22050,
        )
        recognizer_resource = Mock()
        runtime = GenericOfflineRuntime(
            engine, None, recognizer_resource, responder, {"model": "fixture", "api_key": "secret"}
        )
        socket = self.socket()
        with tempfile.TemporaryDirectory() as root:
            task = asyncio.create_task(serve_offline_session(socket, lambda start: runtime, root))
            self.addAsyncCleanup(self.cancel_task, task)
            delivered = []
            async with asyncio.timeout(2):
                while True:
                    item = await socket.delivered.get()
                    delivered.append(item)
                    if item.get("event", {}).get("frame") == "LLMFullResponseEndFrame":
                        break
            self.assertFalse(task.done())
            self.assertFalse(instances["tts"].release.is_set())
            self.assertEqual(delivered[0]["type"], "offline.session.accepted")
            self.assertIn("incremental_text_events_v1", delivered[0]["capabilities"])
            cfg = delivered[1]
            self.assertEqual(cfg["type"], "offline.runtime.configuration")
            self.assertEqual(cfg["configuration"], {"model": "fixture", "api_key": "[redacted]"})
            self.assertEqual(cfg["configuration_fingerprint"], configuration_fingerprint(runtime.configuration))
            causal = [item["event"] for item in delivered if item["type"] == "offline.event"]
            self.assertEqual([e["sample"] for e in causal], sorted(e["sample"] for e in causal))
            socket.disconnect()
            await asyncio.wait_for(task, 2)
            recognizer_resource.cancel.assert_called_once()
            self.assertGreaterEqual(instances["tts"].cancel_calls, 1)
            self.assertEqual(instances["tts"].cleanup_calls, 1)
            self.assertTrue(instances["llm"]._client.is_closed())
            self.assertFalse(instances["tts"].release.is_set())
            directory = next(Path(root).iterdir())
            self.assertEqual(json.loads((directory / "cancellation.json").read_text())["status"], "cancelled")
            self.assertFalse((directory / "full_output.wav").exists())
            self.assertFalse((directory / "failure.json").exists())
        self.assertFalse(any(x.get("type") == "offline.session.complete" for x in socket.sent if isinstance(x, dict)))

    async def cancel_task(self, task):
        """Release a failed test's providers without leaking native tasks."""
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def test_count_and_byte_overflow_fail_instead_of_dropping_events(self):
        """Nonblocking observer bounds reject an unsendable burst explicitly."""
        for limits in (
            replace(SessionLimits(), max_pending_stream_events=1),
            replace(SessionLimits(), max_pending_stream_bytes=16),
        ):
            with self.subTest(limits=limits), tempfile.TemporaryDirectory() as root:
                closed = asyncio.Event()

                class Burst:
                    configuration = {}

                    async def run(self, source, *, event_sink):
                        for sample in range(3):
                            event_sink({"type": "user_turn_started", "sample": sample})
                        await asyncio.Event().wait()

                    async def aclose(self, closed=closed):
                        closed.set()

                socket = self.socket()
                await asyncio.wait_for(serve_offline_session(socket, lambda start: Burst(), root, limits=limits), 2)
                self.assertTrue(closed.is_set())
                self.assertEqual(socket.sent[-1]["code"], "event_queue_overflow")
                self.assertEqual(socket.closed, 1008)

    async def test_provider_failure_before_eligible_end_is_public_error_without_success(self):
        """Never convert partial text followed by a provider failure into success."""

        class Failed:
            configuration = {}

            async def run(self, source, *, event_sink):
                event_sink({"type": "user_turn_started", "sample": 0})
                raise RuntimeError("private-provider-response")

        socket = self.socket()
        with tempfile.TemporaryDirectory() as root:
            await serve_offline_session(socket, lambda start: Failed(), root)
        self.assertEqual(socket.sent[-1]["type"], "offline.session.error")
        self.assertNotIn("private-provider-response", json.dumps(socket.sent))
        self.assertFalse(any(x.get("type") == "offline.events.complete" for x in socket.sent if isinstance(x, dict)))

    async def test_writer_failure_cancels_pending_provider_and_awaits_runtime_close(self):
        """A failed socket writer cannot leave synthesis running in the background."""
        cancelled, closed = asyncio.Event(), asyncio.Event()

        class FailedSocket(ObservedSocket):
            async def send_json(self, payload):
                if payload["type"] == "offline.event":
                    raise OSError("private-socket-details")
                await super().send_json(payload)

        class Pending:
            configuration = {}

            async def run(self, source, *, event_sink):
                try:
                    event_sink({"type": "user_turn_started", "sample": 0})
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

            async def aclose(self):
                closed.set()

        socket = FailedSocket()
        prepared = self.socket()
        socket.incoming = prepared.incoming
        with tempfile.TemporaryDirectory() as root:
            await asyncio.wait_for(serve_offline_session(socket, lambda start: Pending(), root), 2)
        self.assertTrue(cancelled.is_set())
        self.assertTrue(closed.is_set())
        self.assertEqual(socket.sent[-1]["type"], "offline.session.error")
        self.assertNotIn("private-socket-details", json.dumps(socket.sent))

    async def test_disconnect_cancels_an_async_factory_before_provider_execution(self):
        """The input monitor covers runtime construction as well as generation."""
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def factory(start):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        socket = self.socket()
        with tempfile.TemporaryDirectory() as root:
            task = asyncio.create_task(serve_offline_session(socket, factory, root))
            self.addAsyncCleanup(self.cancel_task, task)
            await asyncio.wait_for(entered.wait(), 2)
            socket.disconnect()
            await asyncio.wait_for(task, 2)
        self.assertTrue(cancelled.is_set())
        self.assertEqual([x["type"] for x in socket.sent], ["offline.session.accepted"])

    async def test_config_send_disconnect_awaits_cleanup_and_reports_cleanup_failure(self):
        """A cleanup error remains a real failure even after a text-scope disconnect."""
        config_entered, close_entered, close_release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class BlockedConfigSocket(ObservedSocket):
            async def send_json(self, payload):
                if payload["type"] == "offline.runtime.configuration":
                    config_entered.set()
                    await asyncio.Event().wait()
                await super().send_json(payload)

        class Runtime:
            configuration = {}

            async def run(self, source, *, event_sink):
                raise AssertionError("Providers must not start before configuration delivery")

            async def aclose(self):
                close_entered.set()
                await close_release.wait()
                raise RuntimeError("private-cleanup-details")

        socket = BlockedConfigSocket()
        socket.incoming = self.socket().incoming
        with tempfile.TemporaryDirectory() as root:
            task = asyncio.create_task(serve_offline_session(socket, lambda start: Runtime(), root))
            self.addAsyncCleanup(self.cancel_task, task)
            await asyncio.wait_for(config_entered.wait(), 2)
            socket.disconnect()
            await asyncio.wait_for(close_entered.wait(), 2)
            self.assertFalse(task.done())
            close_release.set()
            await asyncio.wait_for(task, 2)
            directory = next(Path(root).iterdir())
            self.assertTrue((directory / "failure.json").exists())
            self.assertFalse((directory / "cancellation.json").exists())
        self.assertEqual(socket.sent[-1]["type"], "offline.session.error")
        self.assertNotIn("private-cleanup-details", json.dumps(socket.sent))

    async def test_natural_event_eof_is_not_a_full_session_audio_completion(self):
        """No eligible candidate remains the client's explicit incomplete result."""

        class Empty:
            configuration = {}

            async def run(self, source, *, event_sink):
                event_sink({"type": "input_end", "sample": source.sample_count})
                return SessionResult((), source.sample_count, (), ())

        socket = self.socket()
        with tempfile.TemporaryDirectory() as root:
            await serve_offline_session(socket, lambda start: Empty(), root)
            self.assertFalse((next(Path(root).iterdir()) / "full_output.wav").exists())
        self.assertEqual(
            socket.sent[-1], {"type": "offline.events.complete", "protocol": PROTOCOL, "completion": "text"}
        )
        self.assertEqual(socket.closed, 1000)

    async def test_runner_cleanup_rpc_failure_is_not_swallowed_by_text_disconnect(self):
        """Retain actual typed provider status raised during cancellation cleanup."""
        entered = asyncio.Event()

        class CleanupFailure(RpcError):
            def code(self):
                return StatusCode.UNAVAILABLE

        class Runtime:
            configuration = {}

            async def run(self, source, *, event_sink):
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    raise CleanupFailure()

        socket = self.socket()
        with tempfile.TemporaryDirectory() as root:
            task = asyncio.create_task(serve_offline_session(socket, lambda start: Runtime(), root))
            self.addAsyncCleanup(self.cancel_task, task)
            await asyncio.wait_for(entered.wait(), 2)
            socket.disconnect()
            await asyncio.wait_for(task, 2)
            directory = next(Path(root).iterdir())
            self.assertEqual(json.loads((directory / "failure.json").read_text())["grpc_status"], "UNAVAILABLE")
            self.assertFalse((directory / "cancellation.json").exists())
        self.assertEqual(socket.sent[-1]["grpc_status"], "UNAVAILABLE")

    async def test_retrospective_audio_annotations_stay_on_full_tape_only(self):
        """A natural EOF cannot rewind the causal stream to earlier bot spans."""
        observed = []
        engine = IncrementalSessionEngine(
            ControlledDetector([vad_start(512), vad_stop(4096)]),
            ControlledASR([(3072, 3584, "Question", True)]),
            ControlledResponses(),
            params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0)]
                ),
                empty_user_turn=None,
            ),
        )
        result = await asyncio.wait_for(engine.run(AudioSource(bytes(32000)), event_sink=observed.append), 2)
        self.assertTrue(any(event["type"] == "bot_audio" for event in result.events))
        self.assertFalse(any(event["type"] == "bot_audio" for event in observed))
        self.assertEqual([event["sample"] for event in observed], sorted(event["sample"] for event in observed))

    async def test_unknown_completion_policy_fails_before_runtime_creation(self):
        """Only explicit supported response contracts are accepted."""
        for value in ("spoken", None, True, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_start(start_message(b"aa", completion=value), SessionLimits())


if __name__ == "__main__":
    unittest.main()
