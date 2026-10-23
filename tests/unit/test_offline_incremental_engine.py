# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controlled provider events test the stateful incremental conversation path."""

import asyncio
import copy
import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager

from examples.generic.offline.actors import ActorEvent, EventActor
from examples.generic.offline.aggregation import SampleClock
from examples.generic.offline.incremental import IncrementalSessionEngine
from examples.generic.offline.models import AudioSource, DetectionTrace, samples_to_ns
from examples.generic.offline.native_response import NativeProviderFrame, NativeResponseResult
from examples.generic.offline.recognition import ASRChunkEvent, ASREventTrace


def vad_start(sample):
    """A confirmed detector start is a decision, not a backdated raw onset."""
    return {"type": "speech_start", "sample": sample, "confirmed_sample": sample}


def vad_stop(sample):
    """Keep acoustic and confirmed-stop positions explicit in the fixture."""
    return {"type": "speech_end", "sample": sample, "decision_sample": sample, "reason": "vad_silence"}


class ControlledDetector:
    """Deliver source decisions with checkpoints and explicit cancellation."""

    configuration = {"policy": "silero", "vad": {"start_secs": 0.2, "stop_secs": 0.5}}

    def __init__(self, events=(), *, block=False):
        """Hold a detector open only for the failure-cleanup test."""
        self.events = tuple(events)
        self.block = block
        self.cancelled = False
        self.entered = asyncio.Event()

    async def detect(self, source, *, on_checkpoint):
        """Publish native-shaped VAD events without model weights."""
        self.entered.set()
        try:
            if self.block:
                await asyncio.Event().wait()
            for event in self.events:
                sample = event.get("confirmed_sample", event.get("decision_sample"))
                await on_checkpoint([event], samples_to_ns(sample), None)
            await on_checkpoint([], samples_to_ns(source.sample_count), None)
            return DetectionTrace((), (), self.events)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class ControlledASR:
    """Publish one continuous ordered stream with separately controlled EOF."""

    def __init__(self, events=(), *, hold_eof=False, burst=False, error=None):
        """Events are (source-prefix, availability, text, finalized)."""
        self.events = tuple(events)
        self.hold_eof = hold_eof
        self.burst = burst
        self.error = error
        self.eof = False
        self.cancelled = False
        self.release_eof = asyncio.Event()

    def start(self, source, changed):
        """Start a producer without requiring completion of the recording."""
        progress = [0]
        worker = []
        actor = EventActor(changed, lower_bound=lambda _: progress[0], cancel=lambda: worker[0].cancel())

        async def produce():
            try:
                values = []
                for index, (prefix, available, text, final) in enumerate(self.events):
                    value = ASRChunkEvent(
                        index, prefix, min(source.sample_count, ((prefix + 511) // 512) * 512), available, text, final
                    )
                    values.append(value)
                    if not self.burst:
                        actor.publish_batch([ActorEvent(samples_to_ns(available), "asr", value)])
                        await asyncio.sleep(0)
                if self.burst:
                    actor.publish_batch(
                        [ActorEvent(samples_to_ns(value.available_sample), "asr", value) for value in values]
                    )
                actor.publish_batch([], update=lambda: progress.__setitem__(0, samples_to_ns(source.sample_count + 1)))
                if self.hold_eof:
                    await self.release_eof.wait()
                if self.error is not None:
                    raise self.error
                self.eof = True
                actor.finish(ASREventTrace(tuple(values), {"asr_rpc_eof_drained": True, "asr_streams": 1}))
            except BaseException as error:
                self.cancelled = isinstance(error, asyncio.CancelledError)
                actor.finish(error=error)

        worker.append(asyncio.create_task(produce()))
        actor.bind_worker(worker[0])
        return actor


class ControlledResponses:
    """Inject fixed native frames/readiness so model randomness is absent."""

    def __init__(self, recognizer=None, *, delayed_end=False):
        """Retain contexts and record whether generation starts before ASR EOF."""
        self.recognizer = recognizer
        self.delayed_end = delayed_end
        self.contexts = []
        self.interruptions = []
        self.feedback_frames = []
        self.started_before_asr_eof = []
        self.actors = {}
        self.closed = False

    async def start(self):
        """No setup work or network connections are needed by fixed events."""

    async def begin(self, messages, logical_start_ns, generation_id, changed):
        """Emit provider frames, with an optional response left pending for barge-in."""
        self.contexts.append(copy.deepcopy(messages))
        if self.recognizer is not None:
            self.started_before_asr_eof.append(not self.recognizer.eof)
            self.recognizer.release_eof.set()
        actor = EventActor(changed, lower_bound=lambda _: float("inf"))
        self.actors[generation_id] = actor
        number = len(self.contexts)
        frames = [
            (0, LLMFullResponseStartFrame()),
            (50, TTSTextFrame(text=f"Answer {number}.", aggregated_by="sentence", context_id=generation_id)),
            (
                100,
                TTSAudioRawFrame(audio=b"\x01\x02" * 640, sample_rate=16000, num_channels=1, context_id=generation_id),
            ),
        ]
        if not self.delayed_end or number > 1:
            frames.extend([(200, TTSStoppedFrame(context_id=generation_id)), (220, LLMFullResponseEndFrame())])
        actor.publish_batch(
            [
                ActorEvent(
                    logical_start_ns + samples_to_ns(offset),
                    "provider_frame",
                    NativeProviderFrame(frame, FrameDirection.DOWNSTREAM, "tts"),
                    generation_id,
                )
                for offset, frame in frames
            ]
        )
        if not self.delayed_end or number > 1:
            actor.finish(
                NativeResponseResult(
                    f"Answer {number}.", copy.deepcopy(messages), {"response_observed_wall_ns": 1}, False
                )
            )
        return actor

    async def interrupt(self, generation_id, logical_ns):
        """Complete pending compute when native aggregation requests barge-in."""
        self.interruptions.append((generation_id, logical_ns))
        self.actors[generation_id].finish(NativeResponseResult("", [], {"response_observed_wall_ns": 1}, True))

    async def feedback(self, frame):
        """Retain sender feedback rather than inventing assistant context."""
        self.feedback_frames.append(type(frame).__name__)

    async def aclose(self):
        """Record final ownership cleanup."""
        self.closed = True


class ReferenceUserAggregator(LLMUserAggregator):
    """Couple unmodified native policies without the offline scheduling adapters."""

    def __init__(self, context, assistant):
        """Use stock zero-wait policy timers and share its actual context."""
        super().__init__(
            context,
            params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0)]
                ),
                empty_user_turn=None,
            ),
            enable_direct_mode=True,
        )
        self.assistant = assistant
        self.contexts = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        """Capture real requests and propagate native interruption to the assistant."""
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMContextFrame):
                self.contexts.append(copy.deepcopy(frame.context.get_messages()))
            elif isinstance(frame, InterruptionFrame):
                await self.assistant.queue_frame(copy.deepcopy(frame))

    async def settle(self):
        """Await only native policy zero timers created by the current input."""
        for strategy in self._user_turn_controller.user_turn_strategies.stop:
            tasks = [task for task in (strategy._user_speech_timeout_task, strategy._stt_timeout_task) if task]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)


class IncrementalEngineTests(unittest.IsolatedAsyncioTestCase):
    """Test complete conversations using native aggregators and fixed providers."""

    async def evaluate(
        self, *, vad=(), asr=(), source_samples=16000, hold_eof=False, burst=False, delayed_end=False, policy_wait=0
    ):
        """Run a finite recording without providers, sockets or realtime pacing."""
        recognizer = ControlledASR(asr, hold_eof=hold_eof, burst=burst)
        responder = ControlledResponses(recognizer, delayed_end=delayed_end)
        engine = IncrementalSessionEngine(
            ControlledDetector(vad),
            recognizer,
            responder,
            params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=policy_wait)]
                ),
                empty_user_turn=None,
            ),
            output_chunks_10ms=1,
            output_sample_rate=16000,
        )
        result = await asyncio.wait_for(engine.run(AudioSource(b"\0\0" * source_samples)), 2)
        return result, recognizer, responder

    async def test_response_starts_before_continuous_asr_stream_eof(self):
        """The next response cannot depend on a whole-file recognition barrier."""
        result, recognizer, responder = await self.evaluate(
            vad=[vad_start(512), vad_stop(4096)], asr=[(3072, 3584, "Question", True)], hold_eof=True
        )
        self.assertEqual(responder.started_before_asr_eof, [True])
        self.assertTrue(recognizer.eof)
        self.assertEqual(len(result.metrics["responses"]), 1)
        self.assertEqual(result.metrics["responses"][0]["request_sample"], 4096)
        self.assertEqual(result.rendered_samples, 16000)
        self.assertTrue(responder.closed)

    async def test_identical_provider_events_preserve_context_and_audio_under_burst_delivery(self):
        """Callback batching may change execution speed, but not conversation state."""
        args = {
            "vad": [vad_start(512), vad_stop(4096), vad_start(8192), vad_stop(12288)],
            "asr": [(3072, 3584, "First question", True), (11264, 11776, "Second question", True)],
        }
        paced, _, paced_responder = await self.evaluate(**args)
        burst, _, burst_responder = await self.evaluate(**args, burst=True)
        self.assertEqual(paced.messages, burst.messages)
        self.assertEqual(paced_responder.contexts, burst_responder.contexts)
        self.assertEqual(paced.spans, burst.spans)
        self.assertEqual([row["request_sample"] for row in paced.metrics["responses"]], [4096, 12288])
        self.assertEqual([row["request_sample"] for row in burst.metrics["responses"]], [4096, 12288])

    async def test_barge_in_clips_pcm_and_preserves_only_delivered_assistant_context(self):
        """A resumed user turn cancels pending compute at its detected offset."""
        result, _, responder = await self.evaluate(
            vad=[vad_start(512), vad_stop(4096), vad_start(4320), vad_stop(8192)],
            asr=[(3072, 3584, "First question", True), (7168, 7680, "Second question", True)],
            delayed_end=True,
        )
        self.assertEqual(responder.interruptions[0], ("response-1", samples_to_ns(4320)))
        first_pcm = [span for span in result.spans if span.response_id == "response-1"]
        self.assertTrue(first_pcm)
        self.assertTrue(all(span.end_sample <= 4320 for span in first_pcm))
        self.assertIn({"role": "assistant", "content": "Answer 1."}, responder.contexts[1])
        self.assertTrue(result.metrics["responses"][0]["interrupted"])

    async def test_equal_time_transcripts_are_delivered_before_due_turn_timer(self):
        """A same-offset final must reach native policy before its timeout fires."""
        result, _, responder = await self.evaluate(
            vad=[vad_start(512), vad_stop(4096)], asr=[(3072, 4096, "Question", True)], burst=True
        )
        self.assertEqual(len(responder.contexts), 1)
        self.assertEqual(responder.contexts[0], [{"role": "user", "content": "Question"}])
        self.assertEqual(result.metrics["responses"][0]["request_sample"], 4096)

    async def test_interrupted_request_contexts_match_coupled_stock_native_aggregators(self):
        """Fixed delivery order yields the same context in the stock online policies."""
        result, _, responder = await self.evaluate(
            vad=[vad_start(512), vad_stop(4096), vad_start(4320), vad_stop(8192)],
            asr=[(3072, 3584, "First question", True), (7168, 7680, "Second question", True)],
            delayed_end=True,
        )
        context = LLMContext()
        assistant = LLMAssistantAggregator(context, enable_direct_mode=True)
        user = ReferenceUserAggregator(context, assistant)
        setup = FrameProcessorSetup(clock=SampleClock(), task_manager=TaskManager(), pipeline_worker=None)
        for processor in (assistant, user):
            await processor.setup(setup)
            self.addAsyncCleanup(processor.cleanup)
            await processor.queue_frame(StartFrame())

        async def input_frame(frame):
            await user.queue_frame(frame)
            await user.settle()

        await input_frame(VADUserStartedSpeakingFrame(start_secs=0.2))
        await input_frame(TranscriptionFrame("First question", "", "", finalized=True))
        await input_frame(VADUserStoppedSpeakingFrame(stop_secs=0.5))
        await assistant.queue_frame(LLMFullResponseStartFrame())
        await assistant.queue_frame(TTSTextFrame("Answer 1.", aggregated_by="sentence"))
        await assistant.queue_frame(BotStartedSpeakingFrame())
        await input_frame(BotStartedSpeakingFrame())
        await input_frame(VADUserStartedSpeakingFrame(start_secs=0.2))
        await assistant.queue_frame(BotStoppedSpeakingFrame())
        await input_frame(BotStoppedSpeakingFrame())
        await input_frame(TranscriptionFrame("Second question", "", "", finalized=True))
        await input_frame(VADUserStoppedSpeakingFrame(stop_secs=0.5))
        await assistant.queue_frame(LLMFullResponseStartFrame())
        await assistant.queue_frame(TTSTextFrame("Answer 2.", aggregated_by="sentence"))
        await assistant.queue_frame(LLMFullResponseEndFrame())
        self.assertEqual(responder.contexts, user.contexts)
        self.assertEqual(list(result.messages), context.get_messages())

    async def test_source_eof_does_not_preempt_pending_user_stop_policy(self):
        """Retain the configured wait even when no more provider events remain."""
        result, _, _ = await self.evaluate(
            vad=[vad_start(512), vad_stop(1024)],
            asr=[(1024, 1024, "Question", True)],
            source_samples=2048,
            policy_wait=0.1,
        )
        self.assertEqual(result.metrics["responses"][0]["request_sample"], 2624)
        self.assertGreater(result.rendered_samples, 2048)

    async def test_source_without_speech_finishes_without_advancing_idle_watchdog_forever(self):
        """No output means exactly source coverage and no model request."""
        result, _, responder = await self.evaluate()
        self.assertEqual(result.rendered_samples, 16000)
        self.assertFalse(responder.contexts)
        self.assertFalse(result.spans)

    async def test_latency_retains_acoustic_end_separately_from_vad_confirmation(self):
        """User-end latency includes silence needed to confirm a VAD stop."""
        stop = {**vad_stop(4096), "sample": 2048}
        result, _, _ = await self.evaluate(vad=[vad_start(512), stop], asr=[(3072, 3584, "Question", True)])
        response = result.metrics["responses"][0]
        self.assertEqual(response["latest_vad_speech_end_sample"], 2048)
        self.assertEqual(response["latest_vad_stop_decision_sample"], 4096)
        self.assertEqual(response["request_sample"], 4096)

    async def test_complete_engine_with_native_queued_llm_and_tts(self):
        """Exercise the complete adapter chain, replacing only model I/O."""
        from test_offline_native_response import ScriptedLLM, ScriptedTTS

        from examples.generic.offline.native_response import NativeResponseSession

        instances = {}

        def llm_factory(capture):
            instances["llm"] = ScriptedLLM(capture)
            instances["llm"].release.set()
            return instances["llm"]

        def tts_factory(capture):
            instances["tts"] = ScriptedTTS(capture)
            instances["tts"].release.set()
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
            output_chunks_10ms=40,
            output_sample_rate=22050,
        )
        result = await asyncio.wait_for(engine.run(AudioSource(b"\0\0" * 16000)), 3)
        self.assertEqual(instances["llm"].requests, [[{"role": "user", "content": "Question"}]])
        self.assertTrue(result.spans)
        self.assertTrue(result.metrics["responses"][0]["provider"]["tts_rpc_eof_drained"])
        self.assertTrue(any(m["role"] == "assistant" for m in result.messages))
        self.assertTrue(instances["llm"]._client.is_closed())

    async def test_provider_failure_cancels_an_incomplete_detector(self):
        """Session cleanup must cancel a waiting source actor rather than await it forever."""
        detector = ControlledDetector(block=True)
        recognizer = ControlledASR(error=RuntimeError("ASR failure"))
        responder = ControlledResponses()
        engine = IncrementalSessionEngine(
            detector,
            recognizer,
            responder,
            params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0)]
                ),
                empty_user_turn=None,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "ASR failure"):
            await asyncio.wait_for(engine.run(AudioSource(b"\0\0" * 512)), 1)
        self.assertTrue(detector.cancelled)
        self.assertTrue(responder.closed)
