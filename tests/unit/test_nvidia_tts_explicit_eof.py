# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D102, D105, D107

"""Compare native TTS completion against a controlled RPC, without model calls."""

import asyncio
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    CancelFrame,
    ErrorFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings
from pipecat.utils.asyncio.task_manager import TaskManager

from examples.shared.nvidia_tts_explicit_eof import (
    EXPLICIT_EOF_COMPLETION_POLICY,
    ExplicitEOFNvidiaTTSService,
    explicit_eof_tts_options,
)

_REFERENCE_IDLE_SECONDS = 0.02
_EOF = object()


class ControlledRpc:
    """Replace only network delivery; native NVIDIA code owns both stream loops."""

    def __init__(self, requests, loop):
        """Consume native requests as a gRPC transport would, independently of PCM."""
        self.responses = queue.Queue()
        self.request_text = []
        self.request_seen = asyncio.Event()
        self.request_eof = asyncio.Event()
        self.cancelled = False

        def consume():
            try:
                for request in requests:
                    self.request_text.append(request.text)
                    loop.call_soon_threadsafe(self.request_seen.set)
            finally:
                loop.call_soon_threadsafe(self.request_eof.set)

        self.request_worker = threading.Thread(target=consume, daemon=True)
        self.request_worker.start()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.responses.get()
        if item is _EOF:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(audio=item)

    def cancel(self):
        """Unblock the real native response iterator on native cancellation."""
        self.cancelled = True
        self.responses.put(_EOF)


class _ObservedNative:
    """Observe frames and replace only the Riva stub/config network boundary."""

    def __init__(self, **kwargs):
        self.frames = []
        self.calls = []
        self.outcomes = []
        self.contexts = []
        self.changed = asyncio.Event()
        self.channel = SimpleNamespace(close=Mock())
        super().__init__(
            settings=NvidiaTTSSettings(synthesis_mode="stitched"),
            use_ssl=False,
            sample_rate=22050,
            **kwargs,
        )

    def _initialize_client(self):
        self._service = SimpleNamespace(
            stub=SimpleNamespace(SynthesizeOnline=self._open_rpc),
            auth=SimpleNamespace(channel=self.channel, get_auth_metadata=lambda: []),
        )

    def _create_synthesis_config(self):
        return None

    def _open_rpc(self, requests, *, metadata):
        call = ControlledRpc(requests, self.get_event_loop())
        self.calls.append(call)
        self.get_event_loop().call_soon_threadsafe(self.changed.set)
        return call

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        await super().push_frame(frame, direction)
        self.frames.append(frame)
        self.changed.set()

    async def on_turn_context_created(self, context_id):
        self.contexts.append(context_id)
        await super().on_turn_context_created(context_id)

    async def _record_context_audio_outcome(self, context_id, received_audio):
        self.outcomes.append((context_id, received_audio))
        await super()._record_context_audio_outcome(context_id, received_audio)
        self.changed.set()


class ReferenceTTS(_ObservedNative, NvidiaTTSService):
    """Exercise the unchanged idle policy with a short test-only interval."""

    def __init__(self):
        super().__init__(stop_frame_timeout_s=_REFERENCE_IDLE_SECONDS)


class StockTTS(_ObservedNative, NvidiaTTSService):
    """Keep the real default idle policy for normal-completion comparisons."""


class CandidateTTS(_ObservedNative, ExplicitEOFNvidiaTTSService):
    """Exercise the candidate with the same native stub and frame observer."""


class ExplicitEOFContracts(unittest.IsolatedAsyncioTestCase):
    """Keep all service queues, synthesis and cleanup implementations native."""

    async def asyncSetUp(self):
        self.services = []

    async def asyncTearDown(self):
        for service, manager in reversed(self.services):
            await service.cancel(CancelFrame())
            await service.cleanup()
            for task in tuple(manager.current_tasks()):
                await manager.cancel_task(task)
            for call in service.calls:
                await asyncio.to_thread(call.request_worker.join, 1)
                self.assertFalse(call.request_worker.is_alive())

    async def make(self, kind=CandidateTTS):
        service = kind()
        manager = TaskManager()
        clock = SystemClock()
        clock.start()
        await service.setup(
            FrameProcessorSetup(
                clock=clock,
                task_manager=manager,
                pipeline_worker=None,
                audio_in_sample_rate=16000,
                audio_out_sample_rate=22050,
                enable_metrics=True,
            )
        )
        self.services.append((service, manager))
        await service.queue_frame(StartFrame())
        await self.until(service, lambda: any(isinstance(frame, StartFrame) for frame in service.frames))
        return service

    async def until(self, service, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                service.changed.clear()
                if predicate():
                    break
                await service.changed.wait()

    async def request(self, service, *, finish_llm=True):
        previous = len(service.calls)
        await service.queue_frame(LLMFullResponseStartFrame())
        await service.queue_frame(LLMTextFrame(text="Hello there. Another sentence."))
        if finish_llm:
            await service.queue_frame(LLMFullResponseEndFrame())
        await self.until(service, lambda: len(service.calls) > previous)
        call = service.calls[-1]
        await asyncio.wait_for(call.request_seen.wait(), 2)
        if finish_llm:
            await asyncio.wait_for(call.request_eof.wait(), 2)
        self.assertTrue(call.request_text)
        return call

    def audio(self, service):
        return b"".join(frame.audio for frame in service.frames if isinstance(frame, TTSAudioRawFrame))

    def errors(self, service):
        return [frame for frame in service.frames if isinstance(frame, ErrorFrame)]

    async def test_normal_completion_retains_native_request_audio_and_serialized_frame_order(self):
        reference, candidate = await self.make(StockTTS), await self.make()
        reference_rpc, candidate_rpc = await self.request(reference), await self.request(candidate)
        for call in (reference_rpc, candidate_rpc):
            call.responses.put(b"\x11\x12" * 64)
            call.responses.put(b"\x13\x14" * 64)
            call.responses.put(_EOF)
        for service in (reference, candidate):
            await self.until(
                service,
                lambda service=service: any(isinstance(f, LLMFullResponseEndFrame) for f in service.frames),
            )

        def serialized(service):
            kinds = (
                LLMFullResponseStartFrame,
                TTSStartedFrame,
                TTSTextFrame,
                TTSAudioRawFrame,
                TTSStoppedFrame,
                LLMFullResponseEndFrame,
            )
            return [
                (type(f).__name__, getattr(f, "text", None), getattr(f, "audio", None))
                for f in service.frames
                if isinstance(f, kinds)
            ]

        self.assertEqual(reference_rpc.request_text, candidate_rpc.request_text)
        self.assertEqual(serialized(reference), serialized(candidate))
        self.assertEqual(self.errors(reference), [])
        self.assertEqual(self.errors(candidate), [])

    async def test_delayed_first_pcm_survives_reference_idle_expiry_until_rpc_eof(self):
        reference, candidate = await self.make(ReferenceTTS), await self.make()
        candidate_rpc = await self.request(candidate)
        reference_rpc = await self.request(reference)
        await self.until(reference, lambda: bool(reference.outcomes))
        self.assertFalse(reference.outcomes[0][1])
        self.assertTrue(self.errors(reference))
        self.assertTrue(candidate.audio_context_available(candidate.contexts[0]))
        self.assertEqual(candidate.outcomes, [])
        self.assertEqual(self.errors(candidate), [])
        pcm = b"\x01\x02" * 64
        for call in (reference_rpc, candidate_rpc):
            call.responses.put(pcm)
            call.responses.put(_EOF)
        await self.until(candidate, lambda: bool(candidate.outcomes))
        self.assertEqual(self.audio(candidate), pcm)
        self.assertEqual(self.audio(reference), b"")
        self.assertTrue(candidate.outcomes[0][1])
        self.assertEqual(self.errors(candidate), [])
        self.assertTrue(any(isinstance(frame, TTSStoppedFrame) for frame in candidate.frames))

    async def test_long_pcm_gap_preserves_one_context_and_all_native_audio(self):
        reference, candidate = await self.make(ReferenceTTS), await self.make()
        candidate_rpc = await self.request(candidate)
        reference_rpc = await self.request(reference)
        first, second = b"\x01\x02" * 64, b"\x03\x04" * 64
        candidate_rpc.responses.put(first)
        await self.until(candidate, lambda: self.audio(candidate) == first)
        reference_rpc.responses.put(first)
        await self.until(reference, lambda: self.audio(reference) == first)
        await self.until(reference, lambda: bool(reference.outcomes))
        self.assertTrue(reference.outcomes[0][1])
        self.assertEqual(candidate.outcomes, [])
        self.assertFalse(any(isinstance(frame, TTSStoppedFrame) for frame in candidate.frames))
        for call in (reference_rpc, candidate_rpc):
            call.responses.put(second)
            call.responses.put(_EOF)
        await self.until(candidate, lambda: bool(candidate.outcomes))
        self.assertEqual(self.audio(candidate), first + second)
        self.assertEqual(self.audio(reference), first)
        self.assertEqual(len(candidate.contexts), 1)

    async def test_genuine_empty_rpc_eof_retains_native_zero_audio_error(self):
        service = await self.make()
        call = await self.request(service)
        call.responses.put(_EOF)
        await self.until(service, lambda: bool(service.outcomes))
        self.assertFalse(service.outcomes[0][1])
        self.assertTrue(self.errors(service))
        self.assertEqual(service._consecutive_zero_audio_contexts, 1)

    async def test_pending_llm_input_does_not_close_the_native_context(self):
        reference, candidate = await self.make(ReferenceTTS), await self.make()
        candidate_rpc = await self.request(candidate, finish_llm=False)
        reference_rpc = await self.request(reference, finish_llm=False)
        await self.until(reference, lambda: bool(reference.outcomes))
        self.assertTrue(self.errors(reference))
        self.assertFalse(reference_rpc.request_eof.is_set())
        self.assertFalse(candidate_rpc.request_eof.is_set())
        self.assertTrue(candidate.audio_context_available(candidate.contexts[0]))
        self.assertEqual(candidate.outcomes, [])
        pcm = b"\x07\x08" * 64
        candidate_rpc.responses.put(pcm)
        await self.until(candidate, lambda: self.audio(candidate) == pcm)
        self.assertEqual(candidate.outcomes, [])
        await candidate.queue_frame(LLMFullResponseEndFrame())
        await asyncio.wait_for(candidate_rpc.request_eof.wait(), 2)
        candidate_rpc.responses.put(_EOF)
        await self.until(candidate, lambda: bool(candidate.outcomes))
        self.assertTrue(candidate.outcomes[0][1])
        self.assertEqual(self.errors(candidate), [])

    async def test_provider_error_is_observed_and_native_cleanup_cancels_waiting_context(self):
        service = await self.make()
        call = await self.request(service)
        call.responses.put(RuntimeError("scripted provider failure"))
        await self.until(service, lambda: bool(self.errors(service)))
        self.assertEqual(service.outcomes, [])
        await service.cancel(CancelFrame())
        await service.cleanup()
        self.assertIsNone(service._audio_context_task)
        self.assertIsNone(service._service)
        self.assertTrue(service.channel.close.called)

    async def test_interruption_cancels_native_rpc_without_false_empty_completion(self):
        service = await self.make()
        call = await self.request(service)
        await service.queue_frame(InterruptionFrame())
        await self.until(service, lambda: any(isinstance(frame, InterruptionFrame) for frame in service.frames))
        self.assertTrue(call.cancelled)
        self.assertEqual(service.outcomes, [])
        self.assertEqual(self.errors(service), [])
        self.assertFalse(service.channel.close.called)
        fresh = await self.request(service)
        pcm = b"\x05\x06" * 64
        call.responses.put(b"old-response-must-not-play")
        fresh.responses.put(pcm)
        fresh.responses.put(_EOF)
        await self.until(service, lambda: bool(service.outcomes))
        self.assertEqual(self.audio(service), pcm)

    async def test_separate_compute_watchdog_aborts_without_declaring_silent_success(self):
        service = await self.make()
        call = await self.request(service)
        with self.assertRaises(TimeoutError):
            async with asyncio.timeout(_REFERENCE_IDLE_SECONDS * 2):
                await self.until(service, lambda: bool(service.outcomes))
        await service.cancel(CancelFrame())
        await service.cleanup()
        self.assertTrue(call.cancelled)
        self.assertEqual(service.outcomes, [])
        self.assertEqual(self.errors(service), [])
        self.assertIsNone(service._audio_context_task)
        self.assertIsNone(service._service)

    def test_candidate_refuses_other_versions_modes_and_idle_overrides(self):
        with (
            patch("examples.shared.nvidia_tts_explicit_eof.version", return_value="1.13.0"),
            self.assertRaisesRegex(RuntimeError, "exactly Pipecat 1.12.0"),
        ):
            CandidateTTS()
        with self.assertRaisesRegex(ValueError, "stitched synthesis"):
            ExplicitEOFNvidiaTTSService(settings=NvidiaTTSSettings(synthesis_mode="per_sentence"))
        with self.assertRaisesRegex(ValueError, "idle deadline"):
            CandidateTTS(stop_frame_timeout_s=20)

    def test_shared_options_validate_without_changing_service_inheritance_or_global_state(self):
        self.assertEqual(EXPLICIT_EOF_COMPLETION_POLICY, "explicit_provider_eof_pipecat_1_12_compat")
        options = explicit_eof_tts_options("stitched")
        self.assertEqual(options, {"stop_frame_timeout_s": None})
        options.clear()
        self.assertEqual(explicit_eof_tts_options("stitched"), {"stop_frame_timeout_s": None})
        with self.assertRaisesRegex(ValueError, "stitched synthesis"):
            explicit_eof_tts_options("per_sentence")
        with (
            patch("examples.shared.nvidia_tts_explicit_eof.version", return_value="1.12.1"),
            self.assertRaisesRegex(RuntimeError, "exactly Pipecat 1.12.0"),
        ):
            explicit_eof_tts_options("stitched")
