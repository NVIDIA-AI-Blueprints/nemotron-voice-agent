# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace native transcript consumption separately from publication and context.

The controlled backlog proves a queue mechanism, not the cause of a historical
benchmark difference. No VAD model, ASR, LLM, TTS, or network call is involved.
"""

import asyncio
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    ErrorFrame,
    InterruptionFrame,
    LLMContextFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameProcessorSetup
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager

from examples.generic.offline.aggregation import ChunkPositionedFrame, SharedUserAggregation
from examples.shared.evaluation_trace import EvaluationTrace

TOKENS = ("first", "second", "third")


def _params():
    """Use the same cascaded policy with explicit VAD events and no model."""
    return LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.0)])
    )


async def _until(predicate):
    """Yield to native workers until an observable acknowledgement arrives."""
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


class _Publisher(FrameProcessor):
    """Publish identical frames through the ordinary native push path."""

    def __init__(self):
        super().__init__(enable_direct_mode=True)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class _Sink(FrameProcessor):
    """Capture native context output without invoking an LLM."""

    def __init__(self):
        super().__init__(enable_direct_mode=True)
        self.contexts = []
        self.errors = []
        self.context_added = asyncio.Event()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            self.contexts.append(copy.deepcopy(frame.context.get_messages()))
            self.context_added.set()
        elif isinstance(frame, ErrorFrame):
            self.errors.append(type(frame.exception).__name__ if frame.exception else "ErrorFrame")


async def _native(*, trace=None, paused=False):
    """Preserve actual queues and use the native public pause/interruption APIs."""
    context = LLMContext()
    user = LLMUserAggregator(context, params=_params())
    publisher, sink = _Publisher(), _Sink()
    publisher.link(user)
    user.link(sink)
    manager, clock = TaskManager(), SystemClock()
    clock.start()
    setup = FrameProcessorSetup(clock=clock, task_manager=manager, pipeline_worker=None)
    returned, published = [], []
    if trace is not None:
        trace.bind_processor(publisher, role="asr")
        trace.bind_processor(user, role="user")
        # Duplicate wiring must not duplicate inline observations.
        trace.bind_processor(user, role="user")

    @user.event_handler("on_after_process_frame")
    def processed(_processor, frame):
        returned.append(frame)

    async def publish(frame, *, ack=False):
        published.append(frame)
        await publisher.queue_frame(frame)
        if ack:
            await _until(lambda: any(item is frame for item in returned))

    transcript_frames = [TranscriptionFrame(token, "synthetic", "", finalized=True) for token in TOKENS]
    try:
        for processor in (publisher, user, sink):
            await processor.setup(setup)
        await publish(StartFrame(), ack=True)
        await publish(VADUserStartedSpeakingFrame(start_secs=0.2), ack=True)
        await publish(transcript_frames[0], ack=True)
        if paused:
            await user.pause_processing_frames()
        for frame in transcript_frames[1:]:
            await publish(frame, ack=not paused)
        if paused:
            await _until(
                lambda: (
                    getattr(user, "_FrameProcessor__process_current_frame", None) is transcript_frames[1]
                    and user.has_queued_frame(TranscriptionFrame)
                )
            )
        await publish(VADUserStoppedSpeakingFrame(stop_secs=1.25), ack=True)
        await asyncio.wait_for(sink.context_added.wait(), 2)
        await publish(InterruptionFrame(), ack=True)
        await user.resume_processing_frames()
        if sink.errors:
            raise AssertionError(sink.errors)
        # This deliberately mutates the live context after publication. An
        # immutable trace must retain the earlier request's exact content.
        context.add_message({"role": "assistant", "content": "Later mutation"})
        result = {
            "published": [frame.text for frame in published if isinstance(frame, TranscriptionFrame)],
            "returned": [frame.text for frame in returned if isinstance(frame, TranscriptionFrame)],
            "contexts": sink.contexts,
            "frame_ids": [frame.id for frame in transcript_frames],
            "native_queue_enabled": not user._enable_direct_mode,
        }
    finally:
        for processor in (publisher, user, sink):
            await processor.cleanup()
        for task in tuple(manager.current_tasks()):
            await manager.cancel_task(task)
    return result


async def _offline(*, trace=None, paused=False):
    """Use the real shared adapter; direct delivery retains its existing semantics."""
    adapter = SharedUserAggregation(params=_params(), evaluation_trace=trace)
    publisher = SimpleNamespace(name="synthetic_asr")
    returned, frames = [], []

    @adapter._aggregator.event_handler("on_after_process_frame")
    def processed(_processor, frame):
        returned.append(frame)

    async def feed(sample, frame):
        frames.append(frame)
        if trace is not None and isinstance(frame, TranscriptionFrame):
            trace.processor_boundary("source_delivery", publisher, frame, role="asr", logical_sample=sample)
        await adapter.feed(ChunkPositionedFrame(sample, frame))

    try:
        await adapter.start()
        await feed(0, VADUserStartedSpeakingFrame(start_secs=0.2))
        await feed(512, TranscriptionFrame(TOKENS[0], "synthetic", "", finalized=True))
        if paused:
            await adapter._aggregator.pause_processing_frames()
        await feed(1024, TranscriptionFrame(TOKENS[1], "synthetic", "", finalized=True))
        await feed(1536, TranscriptionFrame(TOKENS[2], "synthetic", "", finalized=True))
        await feed(20000, VADUserStoppedSpeakingFrame(stop_secs=1.25))
        await feed(20001, InterruptionFrame())
        adapter.context.add_message({"role": "assistant", "content": "Later mutation"})
        return {
            "published": [frame.text for frame in frames if isinstance(frame, TranscriptionFrame)],
            "returned": [frame.text for frame in returned if isinstance(frame, TranscriptionFrame)],
            "contexts": [context for _sample, context in adapter.contexts],
            "frame_ids": [frame.id for frame in frames if isinstance(frame, TranscriptionFrame)],
            "native_queue_enabled": not adapter._aggregator._enable_direct_mode,
        }
    finally:
        await adapter.close()


class ContextCommitTraceTests(unittest.IsolatedAsyncioTestCase):
    """Use native behavior as the oracle, independently of trace observations."""

    def setUp(self):
        """Keep optional trace files outside repository state."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def _trace(self):
        return EvaluationTrace(Path(self.directory.name))

    def _saved(self, trace):
        trace.close()
        return json.loads(trace.path.read_text())

    def _assert_publication(self, document, frame_ids):
        publications = [
            event
            for event in document["events"]
            if event["event"] == "processor_boundary"
            and event["role"] == "asr"
            and event["phase"] in {"before_push", "source_delivery"}
            and event["frame_type"] == "TranscriptionFrame"
        ]
        self.assertEqual([event["frame_id"] for event in publications], frame_ids)
        self.assertEqual([event["text"] for event in publications], list(TOKENS))

    def _assert_context(self, document, content):
        contexts = [
            event
            for event in document["events"]
            if event["event"] == "processor_boundary"
            and event["role"] == "user"
            and event["phase"] == "before_push"
            and event["frame_type"] == "LLMContextFrame"
        ]
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0]["context_messages"], [{"role": "user", "content": content}])
        encoded = json.dumps(contexts[0]["context_messages"], sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(contexts[0]["context_messages_sha256"], hashlib.sha256(encoded).hexdigest())
        self.assertEqual(document["observation_errors"], 0)
        self.assertEqual(document["dropped_events"], 0)
        self.assertFalse(any("discard" in event["event"] for event in document["events"]))
        return contexts[0]

    async def test_queued_control_retains_complete_context_with_and_without_trace(self):
        """Normal native queues consume all published finals with identical outcomes."""
        plain = await _native()
        trace = self._trace()
        observed = await _native(trace=trace)
        for field in ("published", "returned", "contexts", "native_queue_enabled"):
            self.assertEqual(observed[field], plain[field])
        self.assertEqual(observed["returned"], list(TOKENS))
        self.assertTrue(observed["native_queue_enabled"])
        document = self._saved(trace)
        self._assert_publication(document, observed["frame_ids"])
        context = self._assert_context(document, "first second third")
        self.assertEqual(context["published_asr_without_processing_start"], [])
        self.assertEqual(context["published_asr_without_processing_return"], [])
        self.assertIsNone(context["logical_sample"])

    async def test_native_backlog_records_unprocessed_finals_without_claiming_discard(self):
        """Forced native pause plus interruption changes consumption, not publication."""
        plain = await _native(paused=True)
        trace = self._trace()
        observed = await _native(trace=trace, paused=True)
        for field in ("published", "returned", "contexts", "native_queue_enabled"):
            self.assertEqual(observed[field], plain[field])
        self.assertEqual(observed["published"], list(TOKENS))
        self.assertEqual(observed["returned"], ["first"])
        document = self._saved(trace)
        self._assert_publication(document, observed["frame_ids"])
        context = self._assert_context(document, "first")
        self.assertEqual(context["published_asr_without_processing_start"], observed["frame_ids"][1:])
        self.assertEqual(context["published_asr_without_processing_return"], observed["frame_ids"][1:])
        started = [
            event["frame_id"]
            for event in document["events"]
            if event["event"] == "processor_boundary"
            and event["role"] == "user"
            and event["phase"] == "before_process"
            and event["frame_type"] == "TranscriptionFrame"
        ]
        self.assertEqual(started, observed["frame_ids"][:1])
        self.assertIn("does not prove", document["timing_scopes"]["processor_boundary"])

    async def test_offline_direct_adapter_exposes_same_publications_and_its_actual_context(self):
        """Trace the real direct adapter without pretending it has native queue loss."""
        for paused in (False, True):
            with self.subTest(paused=paused):
                plain = await _offline(paused=paused)
                trace = self._trace()
                observed = await _offline(trace=trace, paused=paused)
                for field in ("published", "returned", "contexts", "native_queue_enabled"):
                    self.assertEqual(observed[field], plain[field])
                self.assertEqual(observed["returned"], list(TOKENS))
                self.assertFalse(observed["native_queue_enabled"])
                document = self._saved(trace)
                self._assert_publication(document, observed["frame_ids"])
                context = self._assert_context(document, "first second third")
                self.assertEqual(context["logical_sample"], 20000)
                self.assertEqual(context["published_asr_without_processing_start"], [])
                self.assertEqual(context["published_asr_without_processing_return"], [])
                starts = [
                    event["logical_sample"]
                    for event in document["events"]
                    if event["event"] == "processor_boundary"
                    and event["role"] == "user"
                    and event["phase"] == "before_process"
                    and event["frame_type"] == "TranscriptionFrame"
                ]
                self.assertEqual(starts, [512, 1024, 1536])
