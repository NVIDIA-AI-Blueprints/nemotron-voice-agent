# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare turn policy on equivalent delivered tapes, without model inference.

Native queue processing is acknowledged per input; no backlog or pause is forced.
Only already-ready policy timers are drained. These synthetic delivery positions
are not ASR word timestamps or a reconstruction of provider availability.
"""

import asyncio
import copy
import unittest

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    ErrorFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    StartFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator, LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager

from examples.generic.offline.aggregation import ChunkPositionedFrame, SharedUserAggregation

_TURN_EVENTS = {"user_turn_started", "user_turn_stopped", "user_turn_stop_timeout", "user_turn_message_added"}
_INITIAL = [{"role": "system", "content": "Synthetic test instructions."}]


def _params():
    """Use the existing zero-floor Silero policy and native five-second watchdog."""
    return LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.0)]),
        user_turn_stop_timeout=5.0,
    )


def _frame(kind, text):
    """Inject policy inputs directly; VAD inference and ASR are outside this test."""
    if kind == "start":
        return VADUserStartedSpeakingFrame(start_secs=0.2)
    if kind == "stop":
        # Existing conformance configuration: the VAD interval covers the
        # one-second STT safety net, including an outstanding interim result.
        return VADUserStoppedSpeakingFrame(stop_secs=1.25)
    if kind == "interim":
        return InterimTranscriptionFrame(text, "synthetic", "")
    return TranscriptionFrame(text, "synthetic", "", finalized=True)


class _QueuedOracle(LLMUserAggregator):
    """Observe stock queued processing while preserving its timers and pushes."""

    def __init__(self):
        super().__init__(LLMContext(messages=copy.deepcopy(_INITIAL)), params=_params())
        self.contexts, self.events, self.errors = [], [], []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, ErrorFrame):
            self.errors.append(type(frame.exception).__name__ if frame.exception else "ErrorFrame")
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMContextFrame):
                self.contexts.append(copy.deepcopy(frame.context.get_messages()))
                self.events.append("request")
            elif isinstance(frame, InterruptionFrame):
                self.events.append("interruption")
        await super().push_frame(frame, direction)

    async def _call_event_handler(self, name, *args, **kwargs):
        await super()._call_event_handler(name, *args, **kwargs)
        if name.removeprefix("on_") in _TURN_EVENTS:
            self.events.append(name.removeprefix("on_"))


async def _native(tape):
    """Acknowledge actual native queue processing and drain zero-floor tasks."""
    oracle = _QueuedOracle()
    manager, clock = TaskManager(), SystemClock()
    clock.start()
    acknowledgements = {}

    @oracle.event_handler("on_after_process_frame")
    def processed(_processor, frame):
        if frame.id in acknowledgements:
            acknowledgements[frame.id].set()

    async def feed(frame):
        acknowledgement = asyncio.Event()
        acknowledgements[frame.id] = acknowledgement
        await oracle.queue_frame(frame)
        await asyncio.wait_for(acknowledgement.wait(), timeout=1)

    try:
        await oracle.setup(FrameProcessorSetup(clock=clock, task_manager=manager, pipeline_worker=None))
        await feed(StartFrame())
        await feed(STTMetadataFrame("NvidiaSTTService", 1.0))
        for kind, text in tape:
            if kind == "settle":
                for strategy in oracle._user_turn_controller.user_turn_strategies.stop:
                    ready = [task for task in (strategy._user_speech_timeout_task, strategy._stt_timeout_task) if task]
                    if ready:
                        await asyncio.wait_for(asyncio.gather(*ready), timeout=1)
                await asyncio.sleep(0)
            else:
                await feed(_frame(kind, text))
        if oracle.errors:
            raise AssertionError(oracle.errors)
        # A later live-context mutation must not rewrite any captured request.
        oracle.context.get_messages()[-1]["content"] = "later mutation"
        return oracle.contexts, oracle.events, not oracle._enable_direct_mode
    finally:
        await oracle.cleanup()
        for task in tuple(manager.current_tasks()):
            await manager.cancel_task(task)


async def _shared(tape):
    """Deliver the same policy tape using the explicit sample clock."""
    contexts, events = [], []

    async def context(_sample, messages):
        contexts.append(copy.deepcopy(messages))
        events.append("request")

    async def event(value):
        if value["type"] in _TURN_EVENTS | {"interruption"}:
            events.append(value["type"])

    adapter = SharedUserAggregation(_INITIAL, params=_params(), on_context=context, on_event=event)
    try:
        await adapter.feed(ChunkPositionedFrame(0, STTMetadataFrame("NvidiaSTTService", 1.0)))
        for position, (kind, text) in enumerate(tape, start=1):
            if kind == "settle":
                await adapter.advance_to(adapter.clock.now_sample)
            else:
                await adapter.feed(ChunkPositionedFrame(position * 512, _frame(kind, text)))
        adapter.context.get_messages()[-1]["content"] = "later mutation"
        return contexts, events
    finally:
        await adapter.close()


class OfflineTurnPolicyConformanceTests(unittest.IsolatedAsyncioTestCase):
    """Distinguish policy conformance from changed source/provider availability."""

    async def _compare(self, tape, expected_groups):
        native_contexts, native_events, queued = await _native(tape)
        shared_contexts, shared_events = await _shared(tape)
        self.assertTrue(queued)
        self.assertEqual(native_contexts, shared_contexts)
        self.assertEqual(native_events, shared_events)
        groups = [
            [message["content"] for message in context if message["role"] == "user"] for context in native_contexts
        ]
        self.assertEqual(groups, expected_groups)
        self.assertEqual(
            native_events,
            ["interruption", "user_turn_started", "request", "user_turn_message_added", "user_turn_stopped"]
            * len(expected_groups),
        )
        self.assertNotIn("later mutation", str(native_contexts))
        self.assertNotIn("user_turn_stop_timeout", native_events)

    async def test_final_before_vad_stop_retains_one_user_message(self):
        """Both runtimes commit the same immutable message when the final is ready."""
        await self._compare(
            [
                ("start", None),
                ("final", "alpha"),
                ("interim", "be"),
                ("final", "beta"),
                ("stop", None),
                ("settle", None),
            ],
            [["alpha beta"]],
        )

    async def test_late_final_restarts_turn_without_silently_merging_messages(self):
        """Changing final/stop availability may legitimately change message grouping."""
        await self._compare(
            [
                ("start", None),
                ("final", "alpha"),
                ("interim", "be"),
                ("stop", None),
                ("settle", None),
                ("final", "beta"),
                ("settle", None),
            ],
            [["alpha"], ["alpha", "beta"]],
        )

    async def test_transcription_fallback_can_start_and_restart_without_vad(self):
        """Soft-speech fallback remains a new turn rather than an assumed revision."""
        await self._compare(
            [("final", "quiet"), ("settle", None), ("final", "restart"), ("settle", None)],
            [["quiet"], ["quiet", "restart"]],
        )
