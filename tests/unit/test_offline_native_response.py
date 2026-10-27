# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise native processor queues without contacting model services."""

import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from grpc import Call, RpcError, StatusCode
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
from pipecat.utils.errors import ErrorCategory

from examples.generic.offline.native_response import (
    CapturingNativeNvidiaLLMService,
    CapturingNativeNvidiaTTSService,
    NativeProviderErrorDiagnostic,
    NativeProviderFrame,
    NativeResponseSession,
    get_native_provider_error_diagnostic,
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


class ScriptedRpcCall(Call):
    """Expose typed call state without a network, and forbid private metadata reads."""

    def __init__(self, *, active=True, status=StatusCode.UNAVAILABLE, activity_error=False, code_error=False):
        """Retain counters so observation tests can detect blocking or private access."""
        self.active = active
        self.status = status
        self.activity_error = activity_error
        self.code_error = code_error
        self.activity_reads = 0
        self.code_reads = 0

    def is_active(self):
        """Return only the scripted state or a contained observation failure."""
        self.activity_reads += 1
        if self.activity_error:
            raise RuntimeError("secret-activity-failure")
        return self.active

    def code(self):
        """Reject a potentially blocking read while termination is unverified."""
        self.code_reads += 1
        if self.active is not False:
            raise AssertionError("code() must not be read while a call may be active")
        if self.code_error:
            raise RuntimeError("secret-code-failure")
        return self.status

    def details(self):
        """Provider error bodies are outside the diagnostic contract."""
        raise AssertionError("Provider details must not be read")

    def initial_metadata(self):
        """Authentication metadata is outside the diagnostic contract."""
        raise AssertionError("Initial metadata must not be read")

    def trailing_metadata(self):
        """Provider headers are outside the diagnostic contract."""
        raise AssertionError("Trailing metadata must not be read")

    def time_remaining(self):
        """The observer does not inspect or change a call's deadline."""
        raise AssertionError("Deadline must not be read")

    def cancel(self):
        """The observer may not cancel the call."""
        raise AssertionError("Call cancellation must not be changed")

    def add_callback(self, callback):
        """The direct state observation registers no RPC callback."""
        raise AssertionError("RPC callbacks must not be changed")


class DelayedFirstAudioTTS(ScriptedTTS):
    """Let the real native context timer fire while synthetic inference waits."""

    def __init__(self, capture, *, forget_generation=False, rpc_call=None):
        """Shorten only this fixture's idle deadline; production keeps its default."""
        super().__init__(capture)
        self._stop_frame_timeout_s = 0.02
        self._forget_generation = forget_generation
        self._scripted_rpc_call = rpc_call

    async def on_turn_context_created(self, context_id):
        """Exercise known and missing observation ownership without tagging errors."""
        await super().on_turn_context_created(context_id)
        if self._forget_generation:
            self._native_context_generations.pop(context_id, None)

    async def run_tts(self, text, context_id):
        """Open the native audio queue, then wait before producing any PCM."""
        if not self.audio_context_available(context_id):
            await self.create_audio_context(context_id)
            self.contexts.append(context_id)
            yield TTSStartedFrame(context_id=context_id)
        state = SimpleNamespace(context_id=context_id, rpc_call=self._scripted_rpc_call)
        with self._retain_native_rpc_state(state):
            self.generated.set()
            await self.release.wait()


class NativeResponseTests(unittest.IsolatedAsyncioTestCase):
    """Verify incremental output, persistent services and cancellation fences."""

    async def test_rpc_status_requires_a_complete_observation(self):
        """Keep unfinished or invalid observations from claiming a final status."""
        for rpc_state in ("active", "unavailable", "unrecognized", "complete"):
            with self.subTest(rpc_state=rpc_state):
                failure = RuntimeError("secret-provider-body")
                failure._nva_native_provider_error_diagnostic = NativeProviderErrorDiagnostic(
                    "tts",
                    "tts",
                    ErrorCategory.UNKNOWN,
                    observed_provider_rpc_state=rpc_state,
                    observed_provider_rpc_status=StatusCode.UNAVAILABLE,
                )
                diagnostic = get_native_provider_error_diagnostic(failure)
                if rpc_state == "complete":
                    self.assertEqual(diagnostic["observed_provider_rpc_status"], "UNAVAILABLE")
                else:
                    self.assertNotIn("observed_provider_rpc_status", diagnostic)
                self.assertNotIn("secret-provider-body", json.dumps(diagnostic))

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

    async def restart_session(self, tts_factory=None):
        """Create an independent fixture after releasing the previous providers."""
        await self.session.aclose()
        self.session = NativeResponseSession(
            self.llm_factory, tts_factory or self.tts_factory, sample_rate=22050, failure_timeout=2
        )
        self.addAsyncCleanup(self.session.aclose)
        await self.session.start()

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
        self.assertEqual(
            get_native_provider_error_diagnostic(failure),
            {
                "provider_source": "llm",
                "observed_callback_boundary": "llm",
                "provider_error_category": "UNKNOWN",
            },
        )
        with self.assertRaises(RuntimeError):
            actor.drain()
        await self.session.aclose()
        self.assertEqual(self.tts.cleanup_calls, 1)

    async def test_error_origin_is_processor_identity_and_preserves_typed_rpc_exception(self):
        """An upstream callback edge must not relabel the actual owned source."""

        class PrivateRpcError(RpcError):
            def code(self):
                return StatusCode.UNAVAILABLE

            def details(self):
                raise AssertionError("Provider details must not be read")

        failure = PrivateRpcError("secret-provider-diagnostic")
        actor = await self.session.begin([{"role": "user", "content": "question"}], 0, "error", self.changed)
        await self.wait_event(self.llm.entered)
        frame = ErrorFrame(
            error="secret-provider-body",
            exception=failure,
            processor=self.tts,
            category=ErrorCategory.CONNECTIVITY,
        )
        await self.session._from_llm(frame, FrameDirection.UPSTREAM)
        # A forwarded report of the same failure must retain its first
        # diagnostic, just as the generation retains its first exception.
        await self.session._from_tts(
            ErrorFrame(
                error="secret-second-provider-body",
                exception=failure,
                processor=self.llm,
                category=ErrorCategory.AUTHENTICATION,
            ),
            FrameDirection.UPSTREAM,
        )
        with self.assertRaises(PrivateRpcError) as caught:
            await asyncio.wait_for(actor.result(), 1)
        self.assertIs(caught.exception, failure)
        self.assertIs(caught.exception.code(), StatusCode.UNAVAILABLE)
        diagnostic = get_native_provider_error_diagnostic(caught.exception)
        self.assertEqual(
            diagnostic,
            {
                "provider_source": "tts",
                "observed_callback_boundary": "llm",
                "provider_error_category": "CONNECTIVITY",
            },
        )
        self.assertNotIn("secret-provider", json.dumps(diagnostic))

    async def test_unknown_error_processor_retains_only_observed_boundary(self):
        """An unowned object or category string cannot claim a provider identity."""

        class UnownedProcessor:
            def __str__(self):
                raise AssertionError("Processor names must not establish ownership")

        actor = await self.session.begin([{"role": "user", "content": "question"}], 0, "error", self.changed)
        await self.wait_event(self.llm.entered)
        frame = ErrorFrame(
            error="secret-provider-body",
            processor=UnownedProcessor(),
            category="CONNECTIVITY secret-provider-category",
        )
        await self.session._from_tts(frame, FrameDirection.UPSTREAM)
        with self.assertRaises(RuntimeError) as caught:
            await asyncio.wait_for(actor.result(), 1)
        self.assertEqual(
            get_native_provider_error_diagnostic(caught.exception),
            {"observed_callback_boundary": "tts"},
        )

    async def test_sidecar_assignment_failure_preserves_the_original_exception(self):
        """Optional diagnostics cannot turn an immutable provider failure into another error."""

        class ClosedError(RuntimeError):
            def __setattr__(self, name, value):
                if name == "_nva_native_provider_error_diagnostic":
                    raise RuntimeError("secret-assignment-failure")
                return super().__setattr__(name, value)

        failure = ClosedError("secret-provider-body")
        actor = await self.session.begin([{"role": "user", "content": "question"}], 0, "error", self.changed)
        await self.wait_event(self.llm.entered)
        await self.session._from_tts(
            ErrorFrame(error="secret-provider-body", exception=failure, processor=self.tts), FrameDirection.UPSTREAM
        )
        with self.assertRaises(ClosedError) as caught:
            await asyncio.wait_for(actor.result(), 1)
        self.assertIs(caught.exception, failure)
        self.assertEqual(get_native_provider_error_diagnostic(failure), {})

    def test_diagnostic_accessor_rejects_untyped_and_nonallowlisted_values(self):
        """The public sidecar reader emits no arbitrary strings or provider objects."""
        failure = RuntimeError("secret-provider-body")
        for value in (
            {"provider_source": "secret-source"},
            NativeProviderErrorDiagnostic(
                "secret-source",
                "secret-boundary",
                "secret-category",
                "secret-operation",
                "secret-state",
                "secret-rpc-state",
                "secret-rpc-status",
            ),
        ):
            with self.subTest(kind=type(value).__name__):
                failure._nva_native_provider_error_diagnostic = value
                self.assertEqual(get_native_provider_error_diagnostic(failure), {})

    async def test_native_idle_deadline_reports_operation_without_changing_error_correlation(self):
        """The real native empty-context error is distinct from an RPC failure."""
        for forget_generation in (False, True):
            with self.subTest(forget_generation=forget_generation):

                def delayed_factory(capture, forget=forget_generation):
                    self.tts = DelayedFirstAudioTTS(capture, forget_generation=forget)
                    return self.tts

                await self.restart_session(delayed_factory)
                actor = await self.session.begin([{"role": "user", "content": "question"}], 0, "first", self.changed)
                await self.wait_event(self.tts.generated)
                with self.assertRaisesRegex(RuntimeError, "completed with no audio") as caught:
                    await asyncio.wait_for(actor.result(), 1)
                self.assertEqual(
                    get_native_provider_error_diagnostic(caught.exception),
                    {
                        "provider_source": "tts",
                        "observed_callback_boundary": "tts",
                        "provider_error_category": "UNKNOWN",
                        "observed_provider_operation": "tts_context_audio_outcome_no_audio",
                        "observed_generation_state": "missing" if forget_generation else "active",
                        "observed_provider_rpc_state": "unavailable",
                    },
                )
                self.assertFalse(self.tts.release.is_set())
                self.assertFalse(self.llm.release.is_set())
                # Operation observations must not tag the previously unscoped
                # frame: the existing session-failure policy remains intact.
                self.assertIs(self.session._failure, caught.exception)

    async def test_idle_context_observation_never_reads_status_of_an_active_rpc(self):
        """An early context outcome can observe a still-active exact native call."""
        call = ScriptedRpcCall(active=True)

        def delayed_factory(capture):
            self.tts = DelayedFirstAudioTTS(capture, rpc_call=call)
            return self.tts

        await self.restart_session(delayed_factory)
        actor = await self.session.begin([{"role": "user", "content": "question"}], 0, "first", self.changed)
        await self.wait_event(self.tts.generated)
        with self.assertRaises(RuntimeError) as caught:
            await asyncio.wait_for(actor.result(), 1)
        diagnostic = get_native_provider_error_diagnostic(caught.exception)
        self.assertEqual(diagnostic["observed_provider_operation"], "tts_context_audio_outcome_no_audio")
        self.assertEqual(diagnostic["observed_provider_rpc_state"], "active")
        self.assertNotIn("observed_provider_rpc_status", diagnostic)
        self.assertGreater(call.activity_reads, 0)
        self.assertEqual(call.code_reads, 0)
        self.assertFalse(self.llm.release.is_set())
        self.assertFalse(self.tts.release.is_set())
        await self.session.aclose()
        self.assertEqual(self.tts._native_rpc_states, {})

    async def test_late_context_diagnostic_does_not_change_unscoped_failure_policy(self):
        """Observe an old context without silently redirecting its error frame."""
        for prior_state in ("closed", "interrupted"):
            with self.subTest(prior_state=prior_state):
                await self.restart_session()
                first = await self.session.begin([{"role": "user", "content": "first"}], 0, "first", self.changed)
                await self.wait_event(self.tts.generated)
                old_context = self.tts.contexts[0]
                if prior_state == "interrupted":
                    await self.session.interrupt("first", 1)
                else:
                    self.llm.release.set()
                    self.tts.release.set()
                await asyncio.wait_for(first.result(), 1)
                self.llm.entered.clear()
                self.llm.release.clear()
                self.tts.release.clear()
                second = await self.session.begin([{"role": "user", "content": "second"}], 2, "second", self.changed)
                await self.wait_event(self.llm.entered)
                # Inject a late native outcome callback; this does not assert
                # that ordinary interruption itself emits a zero-audio error.
                await self.tts._record_context_audio_outcome(old_context, False)
                with self.assertRaises(RuntimeError) as caught:
                    await asyncio.wait_for(second.result(), 1)
                diagnostic = get_native_provider_error_diagnostic(caught.exception)
                self.assertEqual(diagnostic["observed_provider_operation"], "tts_context_audio_outcome_no_audio")
                self.assertEqual(diagnostic["observed_generation_state"], prior_state)
                self.assertIs(self.session._failure, caught.exception)
                with self.assertRaises(RuntimeError):
                    await self.session.begin([], 3, "third", self.changed)

    async def test_owned_tts_operation_does_not_relabel_an_llm_error(self):
        """An owned TTS scope cannot assign its operation to an LLM failure."""
        actor = await self.session.begin([{"role": "user", "content": "question"}], 0, "first", self.changed)
        await self.wait_event(self.llm.entered)
        with self.tts._observe_native_operation("unknown-context", "tts_response_stream"):
            await self.session._from_llm(
                ErrorFrame(error="secret-llm-body", processor=self.llm, category=ErrorCategory.SERVER),
                FrameDirection.UPSTREAM,
            )
        with self.assertRaises(RuntimeError) as caught:
            await asyncio.wait_for(actor.result(), 1)
        self.assertEqual(
            get_native_provider_error_diagnostic(caught.exception),
            {
                "provider_source": "llm",
                "observed_callback_boundary": "llm",
                "provider_error_category": "SERVER",
            },
        )
        self.assertNotIn("secret-llm", json.dumps(get_native_provider_error_diagnostic(caught.exception)))

    async def test_foreign_tts_scope_cannot_assign_an_operation_to_the_owned_service(self):
        """An operation from another service instance has no attribution authority."""
        actor = await self.session.begin([{"role": "user", "content": "question"}], 0, "first", self.changed)
        await self.wait_event(self.llm.entered)
        foreign_service = ScriptedTTS(AsyncMock())
        with foreign_service._observe_native_operation("context", "tts_audio_context"):
            await self.session._from_tts(
                ErrorFrame(error="secret-body", processor=self.tts, category=ErrorCategory.UNKNOWN),
                FrameDirection.UPSTREAM,
            )
        with self.assertRaises(RuntimeError) as caught:
            await asyncio.wait_for(actor.result(), 1)
        self.assertEqual(
            get_native_provider_error_diagnostic(caught.exception),
            {
                "provider_source": "tts",
                "observed_callback_boundary": "tts",
                "provider_error_category": "UNKNOWN",
            },
        )

    async def test_native_response_operation_delegates_original_queue_and_marks_unknown_generation(self):
        """A mapped but unrecognized generation must not resolve to the active turn."""
        actor = await self.session.begin([{"role": "user", "content": "question"}], 0, "first", self.changed)
        await self.wait_event(self.llm.entered)
        context_id = "observed-context"
        self.tts._native_context_generations[context_id] = "unrecognized-generation"
        queue = asyncio.Queue()
        state = SimpleNamespace(context_id=context_id, response_queue=queue, stop_event=asyncio.Event())
        await queue.put(RuntimeError("secret-stream-body"))
        self.tts._stream_state = state
        try:
            await self.tts._process_responses(state)
        finally:
            # The fixture did not create a real synthesis stream to tear down.
            self.tts._stream_state = None
        with self.assertRaises(RuntimeError) as caught:
            await asyncio.wait_for(actor.result(), 1)
        diagnostic = get_native_provider_error_diagnostic(caught.exception)
        self.assertEqual(diagnostic["observed_provider_operation"], "tts_response_stream")
        self.assertEqual(diagnostic["observed_generation_state"], "unknown")
        self.assertEqual(queue.qsize(), 0)
        later_failure = RuntimeError("secret-later-body")
        await self.session._from_tts(
            ErrorFrame(error="secret-later-body", exception=later_failure, processor=self.tts),
            FrameDirection.UPSTREAM,
        )
        self.assertNotIn("observed_provider_operation", get_native_provider_error_diagnostic(later_failure))

    async def test_native_rpc_observation_reads_only_typed_status_after_verified_completion(self):
        """Unavailable metadata must not mask a failure or cause a blocking status read."""
        scenarios = (
            (ScriptedRpcCall(active=True), "active", None, 0),
            (ScriptedRpcCall(active=False, status=StatusCode.RESOURCE_EXHAUSTED), "complete", "RESOURCE_EXHAUSTED", 1),
            (ScriptedRpcCall(active=False, status="UNAVAILABLE secret-status"), "complete", None, 1),
            (ScriptedRpcCall(active=False, code_error=True), "complete", None, 1),
            (ScriptedRpcCall(active="secret-state"), "unavailable", None, 0),
            (ScriptedRpcCall(activity_error=True), "unavailable", None, 0),
            (None, "unavailable", None, None),
            (object(), "unavailable", None, None),
        )
        for index, (call, expected_state, expected_status, code_reads) in enumerate(scenarios):
            with self.subTest(index=index):
                await self.restart_session()
                actor = await self.session.begin([{"role": "user", "content": "question"}], 0, "first", self.changed)
                await self.wait_event(self.llm.entered)
                context_id = "scripted-context"
                self.tts._native_context_generations[context_id] = "first"
                queue = asyncio.Queue()
                await queue.put(RuntimeError("secret-stream-body"))
                state = SimpleNamespace(
                    context_id=context_id, response_queue=queue, stop_event=asyncio.Event(), rpc_call=call
                )
                self.tts._stream_state = state
                try:
                    await self.tts._process_responses(state)
                finally:
                    self.tts._stream_state = None
                with self.assertRaises(RuntimeError) as caught:
                    await asyncio.wait_for(actor.result(), 1)
                diagnostic = get_native_provider_error_diagnostic(caught.exception)
                self.assertEqual(diagnostic["observed_provider_rpc_state"], expected_state)
                self.assertEqual(diagnostic.get("observed_provider_rpc_status"), expected_status)
                self.assertNotIn("secret", json.dumps(diagnostic))
                self.assertNotIsInstance(caught.exception, RpcError)
                self.assertEqual(self.tts._native_rpc_states, {})
                if code_reads is not None:
                    self.assertEqual(call.code_reads, code_reads)

    async def test_rpc_observation_does_not_choose_foreign_ambiguous_or_cleared_calls(self):
        """Context identity and live state ownership must survive observation gaps."""
        for scenario in ("foreign_context", "ambiguous_context", "cleared_call"):
            with self.subTest(scenario=scenario):
                await self.restart_session()
                actor = await self.session.begin([{"role": "user", "content": "question"}], 0, "first", self.changed)
                await self.wait_event(self.tts.generated)
                observed_context = self.tts.contexts[0]
                call = ScriptedRpcCall(active=False)
                other_call = ScriptedRpcCall(active=False)
                state = SimpleNamespace(
                    context_id="different-context" if scenario == "foreign_context" else observed_context,
                    rpc_call=call,
                )
                other = SimpleNamespace(context_id=observed_context, rpc_call=other_call)
                with self.tts._retain_native_rpc_state(state):
                    if scenario == "cleared_call":
                        state.rpc_call = None
                    if scenario == "ambiguous_context":
                        with self.tts._retain_native_rpc_state(other):
                            await self.tts._record_context_audio_outcome(observed_context, False)
                    else:
                        await self.tts._record_context_audio_outcome(observed_context, False)
                with self.assertRaises(RuntimeError) as caught:
                    await asyncio.wait_for(actor.result(), 1)
                diagnostic = get_native_provider_error_diagnostic(caught.exception)
                self.assertEqual(diagnostic["observed_provider_rpc_state"], "unavailable")
                self.assertNotIn("observed_provider_rpc_status", diagnostic)
                self.assertEqual(call.activity_reads, 0)
                self.assertEqual(other_call.activity_reads, 0)
                self.assertEqual(self.tts._native_rpc_states, {})

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
