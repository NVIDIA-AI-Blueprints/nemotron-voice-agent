# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test upload-frontier scheduling and real continuous-ASR reader cancellation."""

import asyncio
import threading
import unittest
from unittest.mock import patch

from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from test_offline_detection import MarkerVAD
from test_offline_incremental_engine import ControlledASR, ControlledDetector, ControlledResponses, vad_start, vad_stop
from test_offline_recognition import response, result

from examples.generic.offline import incremental
from examples.generic.offline.detection import AudioOffsetDetector
from examples.generic.offline.incremental import IncrementalSessionEngine
from examples.generic.offline.models import SourceAbortedError, StreamingAudioSource, samples_to_ns
from examples.generic.offline.recognition import ChunkProgressASR


class StreamingEngineTests(unittest.IsolatedAsyncioTestCase):
    """Declared duration cannot serve as received input or accepted source EOF."""

    def engine(self, detector, recognizer):
        """Use actual shared/native aggregation with synthetic provider outputs."""
        return IncrementalSessionEngine(
            detector,
            recognizer,
            ControlledResponses(),
            params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0)]
                ),
                empty_user_turn=None,
            ),
        )

    async def test_scheduler_waits_at_received_frontier_and_only_emits_eof_after_seal(self):
        """Even already available future callbacks cannot overtake missing PCM."""
        source = StreamingAudioSource(4096)
        source.append(bytes(1024))
        engine = self.engine(ControlledDetector([vad_start(1000), vad_stop(2000)]), ControlledASR())
        waits = asyncio.Queue()
        observed = asyncio.Queue()
        events = []
        original = incremental._Session.wait_for_progress

        async def wait_for_progress(session, target, bound):
            waits.put_nowait((target, bound))
            await original(session, target, bound)

        def observe(event):
            events.append((event, source.received_samples, source.sealed))
            observed.put_nowait(event)

        with patch.object(incremental._Session, "wait_for_progress", wait_for_progress):
            task = asyncio.create_task(engine.run(source, event_sink=observe))
            try:
                async with asyncio.timeout(1):
                    while True:
                        target, bound = await waits.get()
                        if target == 1000 and bound == samples_to_ns(512):
                            break
                self.assertFalse(any(e[0]["type"] == "user_turn_started" for e in events))
                source.append(bytes(1024))
                async with asyncio.timeout(1):
                    while (await observed.get())["type"] != "user_turn_started":
                        pass
                self.assertTrue(all(e[1] >= 1000 for e in events if e[0]["type"] == "user_turn_started"))
                self.assertFalse(any(e[0]["type"] == "input_end" for e in events))
                source.append(bytes(4096))
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                self.assertFalse(any(e[0]["type"] == "input_end" for e in events))
                source.append(bytes(2048))
                source.seal()
                result = await asyncio.wait_for(task, 1)
                self.assertEqual(result.rendered_samples, 4096)
                self.assertTrue(all(sealed for event, _, sealed in events if event["type"] == "input_end"))
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_cancellation_wakes_actual_ASR_thread_waiting_for_more_upload(self):
        """Thread cleanup cannot wait forever on an upload that will never finish."""
        source = StreamingAudioSource(1024)
        entered = threading.Event()
        left = threading.Event()

        def stream(audio):
            entered.set()
            try:
                list(audio)
            finally:
                left.set()
            yield  # No fabricated transcript or response EOF.

        recognizer = ChunkProgressASR(stream)
        engine = self.engine(AudioOffsetDetector(vad_analyzer=MarkerVAD()), recognizer)
        task = asyncio.create_task(engine.run(source))
        self.assertTrue(await asyncio.to_thread(entered.wait, 1))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        self.assertTrue(left.is_set())
        with self.assertRaises(SourceAbortedError):
            source.seal()

    async def test_continuous_ASR_emits_prefix_before_seal_and_half_closes_only_at_valid_eof(self):
        """The actual ASR actor consumes one stream with exact unpadded source PCM."""
        pcm = b"\x11\x22" * 1041
        source = StreamingAudioSource(1041)
        callback_published = threading.Event()
        second_window = threading.Event()
        consumed = []

        def stream(audio):
            iterator = iter(audio)
            consumed.append(next(iterator))
            yield response(result(512, "Prefix", final=False))
            callback_published.set()
            for chunk in iterator:
                consumed.append(chunk)
                if len(consumed) == 2:
                    second_window.set()
            yield response(result(1041, "Final", final=True))

        recognizer = ChunkProgressASR(stream)
        actor = recognizer.start(source, asyncio.Event())
        try:
            source.append(pcm[:1024])
            self.assertTrue(await asyncio.to_thread(callback_published.wait, 1))
            snapshot = actor.drain()
            self.assertFalse(snapshot.done)
            self.assertEqual([e.payload.text for e in snapshot.events], ["Prefix"])
            source.append(pcm[1024:])
            self.assertTrue(await asyncio.to_thread(second_window.wait, 1))
            self.assertEqual([len(chunk) // 2 for chunk in consumed], [512, 512])
            self.assertFalse(actor.drain().done)
            source.seal()
            trace = await asyncio.wait_for(actor.result(), 1)
            self.assertTrue(trace.metrics["asr_rpc_eof_drained"])
            self.assertEqual([e.text for e in trace.events], ["Prefix", "Final"])
            self.assertEqual([len(chunk) // 2 for chunk in consumed], [512, 512, 17])
            self.assertEqual(b"".join(consumed), pcm)
        finally:
            if not source.sealed:
                source.abort()
            await actor.aclose()
