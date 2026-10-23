# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise native processor queues without contacting model services."""

import asyncio
import copy
import unittest
from unittest.mock import AsyncMock, patch

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.tts import NvidiaTTSSettings

from examples.generic.offline.native_response import (
    CapturingNativeNvidiaLLMService,
    CapturingNativeNvidiaTTSService,
    NativeProviderFrame,
    NativeResponseSession,
)


class ScriptedLLM(CapturingNativeNvidiaLLMService):
    """Keep native dispatch/start/end/cancellation, replacing only network work."""

    def __init__(self, capture):
        """Prepare controllable token production without making HTTP requests."""
        super().__init__(capture=capture, api_key="local-test", base_url="http://127.0.0.1:1/v1")
        self.requests = []
        self.release = asyncio.Event()
        self.entered = asyncio.Event()
        self.error = None
        self.cancelled_requests = 0

    async def _process_context(self, context):
        self.requests.append(copy.deepcopy(context.get_messages()))
        self.entered.set()
        if self.error is not None:
            raise self.error
        await self.push_frame(LLMTextFrame(text="Hello world. More "))
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled_requests += 1
            raise
        await self.push_frame(LLMTextFrame(text="words."))


class ScriptedTTS(CapturingNativeNvidiaTTSService):
    """Keep real aggregation/audio queues, replacing only synthesis inference."""

    def __init__(self, capture):
        """Retain native aggregation and introduce a controllable synthesis wait."""
        super().__init__(capture=capture, use_ssl=False, settings=NvidiaTTSSettings(synthesis_mode="stitched"))
        self.release = asyncio.Event()
        self.generated = asyncio.Event()
        self.contexts = []
        self.cancel_calls = 0
        self.cleanup_calls = 0

    def _initialize_client(self):
        """The scripted provider has no gRPC channel or remote configuration."""

    def _create_synthesis_config(self):
        """Do not query a real service during native setup."""
        return None

    async def run_tts(self, text, context_id):
        """Yield native context PCM while later synthesis work remains pending."""
        if not self.audio_context_available(context_id):
            await self.create_audio_context(context_id)
            self.contexts.append(context_id)
            yield TTSStartedFrame(context_id=context_id)
        yield TTSAudioRawFrame(
            audio=b"\x01\x02" * 64, sample_rate=self.sample_rate, num_channels=1, context_id=context_id
        )
        self.generated.set()
        await self.release.wait()

    async def cancel(self, frame):
        """Count session cancellation separately from native barge-in."""
        self.cancel_calls += 1
        await super().cancel(frame)

    async def cleanup(self):
        """Record service cleanup while preserving the parent implementation."""
        self.cleanup_calls += 1
        await super().cleanup()


class NativeResponseTests(unittest.IsolatedAsyncioTestCase):
    """Verify incremental output, persistent services and cancellation fences."""

    async def asyncSetUp(self):
        """Create and start two services backed entirely by local scripts."""
        self.changed = asyncio.Event()
        self.session = NativeResponseSession(self.llm_factory, self.tts_factory, sample_rate=22050, failure_timeout=2)
        self.addAsyncCleanup(self.session.aclose)
        await self.session.start()

    def llm_factory(self, capture):
        """Keep the actual instance available for lifecycle assertions."""
        self.llm = ScriptedLLM(capture)
        return self.llm

    def tts_factory(self, capture):
        """Create one persistent service for all test generations."""
        self.tts = ScriptedTTS(capture)
        return self.tts

    async def wait_event(self, event):
        """Bound a scripted dependency without imposing conversation timers."""
        await asyncio.wait_for(event.wait(), 1)

    async def test_uncorrelated_idle_service_failure_blocks_the_next_dispatch(self):
        """A background failure after startup is not lost while no turn is active."""
        await self.session._from_tts(ErrorFrame(error="idle service failed"), FrameDirection.UPSTREAM)
        with self.assertRaisesRegex(RuntimeError, "idle service failed"):
            await self.session.begin([{"role": "user", "content": "next"}], 0, "first", self.changed)
        self.assertEqual(self.llm.requests, [])

    async def test_tagged_closed_generation_error_does_not_poison_its_successor(self):
        """A cancelled generation retains ownership of its delayed error frames."""
        first = await self.session.begin([{"role": "user", "content": "first"}], 0, "first", self.changed)
        self.llm.release.set()
        self.tts.release.set()
        await asyncio.wait_for(first.result(), 1)
        self.llm.release.clear()
        self.llm.entered.clear()
        second = await self.session.begin(
            [{"role": "user", "content": "second"}], 1_000_000_000, "second", self.changed
        )
        await self.wait_event(self.llm.entered)
        stale = ErrorFrame(error="old generation failed")
        stale.metadata["nva_offline_response_generation"] = "first"
        await self.session._from_tts(stale, FrameDirection.UPSTREAM)
        self.assertFalse(second.drain().done)
        self.llm.release.set()
        result = await asyncio.wait_for(second.result(), 1)
        self.assertFalse(result.interrupted)

    async def collect_until(self, actor, predicate):
        """Drain notifications until a particular native frame is available."""
        seen = []
        async with asyncio.timeout(1):
            while True:
                self.changed.clear()
                snapshot = actor.drain()
                seen.extend(snapshot.events)
                if predicate(seen, snapshot):
                    return seen
                await self.changed.wait()

    async def test_pcm_and_text_are_available_before_llm_and_tts_eof(self):
        """No full-response capture barrier may hide already generated PCM."""
        messages = [{"role": "user", "content": "A question"}]
        actor = await self.session.begin(messages, 1_000_000_000, "first", self.changed)
        messages[0]["content"] = "Changed after dispatch"
        await self.wait_event(self.llm.entered)
        events = await self.collect_until(
            actor,
            lambda events, snapshot: any(
                event.kind == "provider_frame" and isinstance(event.payload.frame, TTSAudioRawFrame) for event in events
            ),
        )
        self.assertFalse(actor.drain().done)
        self.assertFalse(self.llm.release.is_set())
        self.assertFalse(self.tts.release.is_set())
        self.assertTrue(
            any(
                event.kind == "provider_observation" and isinstance(event.payload.frame, LLMTextFrame)
                for event in events
            )
        )
        self.assertTrue(all(event.ready_ns >= 1_000_000_000 for event in events))
        self.assertEqual(self.llm.requests[0], [{"role": "user", "content": "A question"}])
        self.llm.release.set()
        self.tts.release.set()
        result = await asyncio.wait_for(actor.result(), 1)
        self.assertFalse(result.interrupted)
        self.assertTrue(result.metrics["tts_rpc_eof_drained"])
        self.assertIn("first_audio_ready_ns", result.metrics)
        self.assertEqual(result.request_messages, self.llm.requests[0])

    async def test_interruption_uses_native_queue_cancellation_and_reuses_tts(self):
        """Barge-in cancels the request while keeping the synthesis service alive."""
        actor = await self.session.begin([{"role": "user", "content": "first"}], 0, "first", self.changed)
        await self.wait_event(self.llm.entered)
        await self.wait_event(self.tts.generated)
        service = self.tts
        await self.session.interrupt("first", 500_000_000)
        result = await asyncio.wait_for(actor.result(), 1)
        self.assertTrue(result.interrupted)
        self.assertEqual(self.llm.cancelled_requests, 1)
        self.assertEqual(service.cancel_calls, 0)
        self.assertEqual(service.cleanup_calls, 0)
        self.llm.entered.clear()
        self.llm.release.set()
        self.tts.release.set()
        second = await self.session.begin([{"role": "user", "content": "second"}], 600_000_000, "second", self.changed)
        await asyncio.wait_for(second.result(), 1)
        self.assertIs(self.tts, service)
        self.assertEqual(len(self.llm.requests), 2)
        await self.session.aclose()
        self.assertEqual(service.cancel_calls, 1)
        self.assertEqual(service.cleanup_calls, 1)
        self.assertTrue(self.llm._client.is_closed())

    async def test_stale_context_pcm_is_not_assigned_to_the_next_generation(self):
        """An old context ID cannot acquire the newer active response identity."""
        first = await self.session.begin([{"role": "user", "content": "first"}], 0, "first", self.changed)
        await self.wait_event(self.tts.generated)
        old_context = self.tts.contexts[0]
        await self.session.interrupt("first", 500_000_000)
        await asyncio.wait_for(first.result(), 1)
        second = await self.session.begin([{"role": "user", "content": "second"}], 600_000_000, "second", self.changed)
        stale = TTSAudioRawFrame(audio=b"oldold", sample_rate=22050, num_channels=1, context_id=old_context)
        await self.tts.push_frame(stale)
        self.llm.release.set()
        self.tts.release.set()
        await asyncio.wait_for(second.result(), 1)
        events = second.drain().events
        self.assertFalse(
            any(isinstance(event.payload, NativeProviderFrame) and event.payload.frame is stale for event in events)
        )
        self.assertTrue(all(event.generation == "second" for event in events))

    async def test_stale_interruption_cannot_cancel_a_newer_native_request(self):
        """Delayed control from a completed generation cannot cancel its successor."""
        first = await self.session.begin([{"role": "user", "content": "first"}], 0, "first", self.changed)
        self.llm.release.set()
        self.tts.release.set()
        await asyncio.wait_for(first.result(), 1)
        self.llm.release.clear()
        self.llm.entered.clear()
        second = await self.session.begin(
            [{"role": "user", "content": "second"}], 1_000_000_000, "second", self.changed
        )
        await self.wait_event(self.llm.entered)
        await self.session.interrupt("first", 2_000_000_000)
        self.assertEqual(self.llm.cancelled_requests, 0)
        self.assertFalse(second.drain().done)
        self.llm.release.set()
        self.assertFalse((await asyncio.wait_for(second.result(), 1)).interrupted)

    async def test_native_service_error_is_propagated_without_waiting_for_eof(self):
        """Preserve the native provider exception and still clean up the session."""
        failure = RuntimeError("scripted provider failure")
        self.llm.error = failure
        actor = await self.session.begin([{"role": "user", "content": "question"}], 0, "error", self.changed)
        with self.assertRaises(RuntimeError) as caught:
            await asyncio.wait_for(actor.result(), 1)
        self.assertIs(caught.exception, failure)
        with self.assertRaises(RuntimeError):
            actor.drain()
        await self.session.aclose()
        self.assertEqual(self.tts.cleanup_calls, 1)

    async def test_native_background_error_without_context_still_fails_active_generation(self):
        """An uncorrelated native failure must not leave an actor pending forever."""
        actor = await self.session.begin([{"role": "user", "content": "question"}], 0, "error", self.changed)
        await self.wait_event(self.llm.entered)
        await self.session._from_tts(ErrorFrame(error="background failure"), FrameDirection.UPSTREAM)
        with self.assertRaisesRegex(RuntimeError, "background failure"):
            await asyncio.wait_for(actor.result(), 1)

    async def test_feedback_uses_native_upstream_speaking_state(self):
        """Sender feedback reaches the normal TTS bot-speaking state machine."""
        started = BotStartedSpeakingFrame()
        stopped = BotStoppedSpeakingFrame()
        for frame, expected in ((started, True), (stopped, False)):
            await self.session.feedback(frame)
            async with asyncio.timeout(1):
                while self.tts._bot_speaking != expected:
                    await asyncio.sleep(0)
        self.assertFalse(self.tts._bot_speaking)

    async def test_serialized_end_waits_for_provider_worker_drain(self):
        """Serialized EOF alone does not abandon a still-running provider worker."""
        actor = await self.session.begin([{"role": "user", "content": "question"}], 0, "first", self.changed)
        await self.wait_event(self.tts.generated)
        context = self.tts.contexts[0]
        worker_drained = asyncio.Event()
        self.tts._native_workers[context] = worker_drained
        self.llm.release.set()
        self.tts.release.set()
        await self.collect_until(
            actor,
            lambda events, snapshot: any(
                event.kind == "provider_frame" and isinstance(event.payload.frame, LLMFullResponseEndFrame)
                for event in events
            ),
        )
        self.assertFalse(actor.drain().done)
        worker_drained.set()
        await asyncio.wait_for(actor.result(), 1)
        self.assertTrue(actor.drain().done)

    async def test_actor_close_cancels_native_work_without_closing_session(self):
        """An actor owns cancellable work, while the session owns its services."""
        actor = await self.session.begin([{"role": "user", "content": "first"}], 0, "first", self.changed)
        await self.wait_event(self.llm.entered)
        await asyncio.wait_for(actor.aclose(), 1)
        self.assertTrue((await actor.result()).interrupted)
        self.assertEqual(self.tts.cleanup_calls, 0)

    async def test_cannot_overlap_context_generations_or_reuse_identifiers(self):
        """Disallow ambiguous identity and overlapping requests in one session."""
        actor = await self.session.begin([{"role": "user", "content": "first"}], 0, "first", self.changed)
        with self.assertRaises(RuntimeError):
            await self.session.begin([], 0, "other", self.changed)
        await self.session.interrupt("first", 1)
        await actor.result()
        with self.assertRaises(ValueError):
            await self.session.begin([], 0, "first", self.changed)

    async def test_startup_error_cleans_up_the_partially_initialized_services(self):
        """A failed TTS setup must release both partially initialized services."""
        await self.session.aclose()
        with patch.object(ScriptedTTS, "setup", AsyncMock(side_effect=RuntimeError("setup failure"))):
            session = NativeResponseSession(self.llm_factory, self.tts_factory, sample_rate=22050)
            with self.assertRaisesRegex(RuntimeError, "setup failure"):
                await session.start()
        self.assertEqual(self.tts.cleanup_calls, 1)

    def test_direct_mode_is_rejected_to_preserve_native_interruption_semantics(self):
        """Normal processor queues are required to cancel native HTTP generation."""
        with self.assertRaises(ValueError):
            CapturingNativeNvidiaLLMService(capture=AsyncMock(), enable_direct_mode=True)
        with self.assertRaises(ValueError):
            CapturingNativeNvidiaTTSService(capture=AsyncMock(), enable_direct_mode=True)
