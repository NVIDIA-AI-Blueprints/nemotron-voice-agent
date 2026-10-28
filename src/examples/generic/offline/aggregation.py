# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay chunk-positioned events through the actual cascaded user aggregator.

This adapter does not assign recognized words to acoustic intervals. A transcript
is an ordered event at the audio prefix the recognizer has processed. The caller
must establish that prefix using a backend progress contract, not the latest
uploaded or submitted chunk. The adapter deliberately cannot infer that contract.

Pipecat 1.12's speech-timeout strategy and turn watchdog contain private timer
coroutines. Only their waits are adapted below; start/stop decisions, transcript
concatenation, context writes, recovery, and interruption policy remain upstream.
Smart Turn is explicitly unsupported until its additional waits are adapted.
"""

import asyncio
import copy
import heapq
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from importlib.metadata import version
from typing import Any

from pipecat.clocks.base_clock import BaseClock
from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMServiceMetadataFrame,
    StartFrame,
    UserIdleTimeoutUpdateFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMAssistantAggregatorParams,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager

from .models import RATE, samples_to_ns

EventCallback = Callable[[dict[str, Any]], Awaitable[None]]
ContextCallback = Callable[[int, list[dict[str, Any]]], Awaitable[None]]


@dataclass(frozen=True)
class ChunkPositionedFrame:
    """A delivered pipeline event and its causal audio-prefix position."""

    sample: int
    frame: Frame
    direction: FrameDirection = FrameDirection.DOWNSTREAM


@dataclass(order=True)
class _Deadline:
    sample: int
    serial: int
    future: asyncio.Future = field(compare=False)
    task: asyncio.Task | None = field(compare=False)


async def _yield_ready_tasks() -> None:
    """Run already-ready callbacks without advancing or reading elapsed time."""
    future = asyncio.get_running_loop().create_future()

    def resume():
        # A disconnected text client can cancel the waiter before this ready
        # callback runs. Do not resolve the future that cancellation closed.
        if not future.done():
            future.set_result(None)

    asyncio.get_running_loop().call_soon(resume)
    await future


class SampleClock(BaseClock):
    """Integer sample clock with explicitly advanced, cancellable timer waits.

    Timer tasks are tracked until blocked or finished. Advancing executes each
    deadline at its own position, so a large source chunk cannot move a 600 ms
    policy timer to the end of that chunk. A zero timeout runs after the current
    input-event batch, matching an asynchronous zero-duration timer.
    """

    def __init__(self, sample_rate: int = RATE):
        """Start at source position zero with no timer tasks."""
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        self.sample_rate = sample_rate
        self.now_sample = 0
        self._deadlines: list[_Deadline] = []
        self._serial = 0
        self._tasks: dict[asyncio.Task, bool] = {}
        self._pending_tasks = 0
        self._revision = 0
        self._errors: list[BaseException] = []

    def start(self):
        """Keep the session's explicit source origin."""

    def get_time(self) -> int:
        """Return media position in Pipecat's nanosecond clock units."""
        return samples_to_ns(self.now_sample, self.sample_rate)

    @property
    def next_deadline_sample(self) -> int | None:
        """Return the next live timeout, omitting cancelled waits."""
        while self._deadlines and self._deadlines[0].future.done():
            heapq.heappop(self._deadlines)
        return self._deadlines[0].sample if self._deadlines else None

    def _set_blocked(self, task: asyncio.Task | None, blocked: bool) -> None:
        if task in self._tasks:
            self._tasks[task] = blocked
            self._revision += 1

    def _future_at(self, sample: int) -> asyncio.Future:
        future = asyncio.get_running_loop().create_future()
        self._serial += 1
        heapq.heappush(self._deadlines, _Deadline(sample, self._serial, future, asyncio.current_task()))
        return future

    async def sleep(self, seconds: float) -> None:
        """Suspend a policy timer until its media-time deadline."""
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Timer durations must be finite and nonnegative")
        future = self._future_at(self.now_sample + math.ceil(seconds * self.sample_rate))
        task = asyncio.current_task()
        self._set_blocked(task, True)
        try:
            await future
        finally:
            self._set_blocked(task, False)

    async def wait_event(self, event: asyncio.Event, seconds: float) -> None:
        """Clock-backed equivalent of wait_for(event.wait(), seconds)."""
        if event.is_set():
            return
        deadline = self._future_at(self.now_sample + math.ceil(seconds * self.sample_rate))
        activity = asyncio.create_task(event.wait())
        task = asyncio.current_task()
        activity.add_done_callback(lambda _: self._set_blocked(task, False))
        self._set_blocked(task, True)
        try:
            done, _ = await asyncio.wait((activity, deadline), return_when=asyncio.FIRST_COMPLETED)
            if activity not in done:
                raise TimeoutError
        finally:
            self._set_blocked(task, False)
            deadline.cancel()
            activity.cancel()
            await asyncio.gather(activity, return_exceptions=True)

    async def settle(self) -> None:
        """Drain ready timer work; never wait for media time to pass."""
        stable_rounds = 0
        while stable_rounds < 2:
            revision = self._revision
            await _yield_ready_tasks()
            if self._errors:
                raise self._errors.pop(0)
            idle = self._pending_tasks == 0 and all(self._tasks.values())
            stable_rounds = stable_rounds + 1 if idle and revision == self._revision else 0

    async def advance_to(self, sample: int, *, inclusive: bool = True) -> None:
        """Run deadlines through a position; callers may batch tied input first."""
        if sample < self.now_sample:
            raise ValueError("Audio progress cannot move backwards")
        await self.settle()
        while (deadline := self.next_deadline_sample) is not None and (
            deadline <= sample if inclusive else deadline < sample
        ):
            self.now_sample = deadline
            while self._deadlines and self._deadlines[0].sample == deadline:
                entry = heapq.heappop(self._deadlines)
                if not entry.future.done():
                    self._set_blocked(entry.task, False)
                    entry.future.set_result(None)
            await self.settle()
        self.now_sample = sample


class _SampleTaskManager(TaskManager):
    """Track the adapter's timer tasks until they reach a clock wait."""

    def __init__(self, clock: SampleClock):
        super().__init__()
        self._sample_clock = clock

    def create_task(self, coroutine, name, context=None):
        clock = self._sample_clock
        clock._pending_tasks += 1
        started = False

        async def tracked():
            nonlocal started
            started = True
            task = asyncio.current_task()
            clock._pending_tasks -= 1
            clock._tasks[task] = False
            clock._revision += 1
            try:
                return await coroutine
            except Exception as error:
                clock._errors.append(error)
                raise
            finally:
                clock._tasks.pop(task, None)
                clock._revision += 1

        tracked_coroutine = tracked()
        task = super().create_task(tracked_coroutine, name, context)

        def finished(_):
            if not started:
                clock._pending_tasks -= 1
                clock._revision += 1
                tracked_coroutine.close()
                coroutine.close()

        task.add_done_callback(finished)
        return task


class _SampleSpeechTimeout(SpeechTimeoutUserTurnStopStrategy):
    """Keep the upstream strategy and replace only the two timer waits."""

    def __init__(self, clock: SampleClock, original: SpeechTimeoutUserTurnStopStrategy):
        super().__init__(
            user_speech_timeout=original._user_speech_timeout,
            wait_for_transcript=original.wait_for_transcript,
        )
        self._enable_user_speaking_frames = original._enable_user_speaking_frames
        self._sample_clock = clock

    async def _user_speech_timeout_handler(self, timeout: float):
        try:
            await self._sample_clock.sleep(timeout)
        except asyncio.CancelledError:
            return
        finally:
            self._user_speech_timeout_task = None
        self._user_speech_wait_done = True
        await self._maybe_trigger_user_turn_stopped()

    async def _stt_timeout_handler(self, timeout: float):
        try:
            await self._sample_clock.sleep(timeout)
        except asyncio.CancelledError:
            return
        finally:
            self._stt_timeout_task = None
        self._stt_wait_done = True
        await self._maybe_trigger_user_turn_stopped()


class _SampleTurnController(UserTurnController):
    """Keep upstream watchdog state and triggers, replacing wait_for only."""

    def __init__(self, clock: SampleClock, **kwargs):
        super().__init__(**kwargs)
        self._sample_clock = clock

    async def _user_turn_stop_timeout_task_handler(self):
        from pipecat.turns.user_stop import UserTurnStoppedParams

        while True:
            try:
                await self._sample_clock.wait_event(self._user_turn_stop_timeout_event, self._user_turn_stop_timeout)
                self._user_turn_stop_timeout_event.clear()
            except TimeoutError:
                if self._user_turn and not self._user_speaking:
                    await self._call_event_handler("on_user_turn_stop_timeout")
                    await self._trigger_user_turn_stop(None, UserTurnStoppedParams(enable_user_speaking_frames=True))


class _ObservedUserAggregator(LLMUserAggregator):
    """Observe upstream context/turn outputs without replacing their policy."""

    def __init__(self, owner: "SharedUserAggregation", context: LLMContext, params: LLMUserAggregatorParams):
        self._owner = owner
        super().__init__(context, params=params, enable_direct_mode=True)
        previous = self._user_turn_controller
        controller = _SampleTurnController(
            owner.clock,
            user_turn_strategies=previous.user_turn_strategies,
            user_turn_stop_timeout=params.user_turn_stop_timeout,
        )
        # Constructor injection is not available in pinned Pipecat 1.12. These
        # are the same controller-to-aggregator bindings as its constructor.
        for name in (
            "on_push_frame",
            "on_broadcast_frame",
            "on_user_turn_speculation_cancelled",
            "on_user_turn_started",
            "on_user_turn_inference_triggered",
            "on_user_turn_stopped",
            "on_user_turn_stop_timeout",
            "on_reset_aggregation",
        ):
            controller.add_event_handler(name, getattr(self, f"_{name}"))
        self._user_turn_controller = controller

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        if isinstance(frame, ErrorFrame):
            self._owner.clock._errors.append(frame.exception or RuntimeError(frame.error))
            return
        if direction != FrameDirection.DOWNSTREAM:
            return
        if isinstance(frame, LLMContextFrame):
            messages = copy.deepcopy(frame.context.get_messages())
            self._owner.contexts.append((self._owner.clock.now_sample, messages))
            if self._owner.on_context:
                try:
                    await self._owner.on_context(self._owner.clock.now_sample, messages)
                except Exception as error:
                    # Pipecat event handlers log rather than propagate errors.
                    # A failed outer scheduler callback must still fail replay.
                    self._owner.clock._errors.append(error)
                    raise
        elif isinstance(frame, InterruptionFrame):
            await self._owner._event("interruption")

    async def _call_event_handler(self, event_name: str, *args, **kwargs):
        await super()._call_event_handler(event_name, *args, **kwargs)
        if event_name in {
            "on_user_turn_started",
            "on_user_turn_stopped",
            "on_user_turn_stop_timeout",
            "on_user_turn_message_added",
        }:
            details = {}
            if args and hasattr(args[-1], "content"):
                details["text"] = args[-1].content
            if args and event_name in {"on_user_turn_started", "on_user_turn_stopped"}:
                details["strategy"] = type(args[0]).__name__ if args[0] else None
            await self._owner._event(event_name.removeprefix("on_"), **details)


class SharedUserAggregation:
    """Session-owned live aggregation driven by source-ordered frame delivery.

    The caller supplies VAD events from the same source chunk analysis used by
    the online policy. This class therefore does not run a second VAD analyzer.
    Bot and function-call frames can be fed too, preserving mute, recovery and
    interruption state. Callbacks should enqueue model/output work for the
    outer scheduler, rather than finish that work while handling an input frame.
    """

    def __init__(
        self,
        messages: Sequence[dict[str, Any]] = (),
        *,
        params: LLMUserAggregatorParams | None = None,
        clock: SampleClock | None = None,
        on_context: ContextCallback | None = None,
        on_event: EventCallback | None = None,
    ):
        """Consume fresh shared parameters and adapt their timer waits only."""
        if version("pipecat-ai") != "1.12.0":
            raise RuntimeError("Offline aggregation timer adapter requires Pipecat 1.12.0")
        if params is None:
            from examples.shared.pipeline_utils import build_user_aggregator_params

            params = build_user_aggregator_params(False)
        strategies = params.user_turn_strategies or UserTurnStrategies()
        if not strategies.stop or any(type(s) is not SpeechTimeoutUserTurnStopStrategy for s in strategies.stop):
            raise ValueError(
                "Offline shared aggregation currently requires the configured Silero speech-timeout policy"
            )
        if params.user_idle_timeout or params.filter_incomplete_user_turns:
            raise ValueError("Offline shared aggregation does not yet adapt idle or speculative-turn timers")
        if not math.isfinite(params.user_turn_stop_timeout) or params.user_turn_stop_timeout <= 0:
            raise ValueError("The shared turn watchdog timeout must be finite and positive")
        self.clock = clock or SampleClock()
        self.on_context = on_context
        self.on_event = on_event
        self.events: list[dict[str, Any]] = []
        self.contexts: list[tuple[int, list[dict[str, Any]]]] = []
        self.context = LLMContext(messages=copy.deepcopy(list(messages)))
        self._unused_vad = params.vad_analyzer
        adapted = replace(
            params,
            vad_analyzer=None,
            user_turn_strategies=UserTurnStrategies(
                start=strategies.start,
                stop=[_SampleSpeechTimeout(self.clock, strategy) for strategy in strategies.stop],
            ),
        )
        self._aggregator = _ObservedUserAggregator(self, self.context, adapted)
        self._tasks: _SampleTaskManager | None = None
        self._started = False
        self._closed = False

    @property
    def next_deadline_sample(self) -> int | None:
        """Next policy deadline for the outer conversation scheduler."""
        return self.clock.next_deadline_sample

    async def _event(self, kind: str, **details):
        event = {"type": kind, "sample": self.clock.now_sample, **details}
        self.events.append(event)
        if self.on_event:
            try:
                await self.on_event(event)
            except Exception as error:
                self.clock._errors.append(error)
                raise

    async def start(self) -> None:
        """Initialize stock controllers once without pipeline worker queues."""
        if self._started:
            return
        self._started = True
        self._tasks = _SampleTaskManager(self.clock)
        await self._aggregator.setup(
            FrameProcessorSetup(clock=self.clock, task_manager=self._tasks, pipeline_worker=None)
        )
        await self._aggregator.queue_frame(StartFrame())
        await self.clock.settle()

    async def feed(self, event: ChunkPositionedFrame) -> None:
        """Deliver one event, then execute timers due at its audio position."""
        await self.feed_batch((event,))

    async def feed_batch(self, events: Sequence[ChunkPositionedFrame], *, settle_timers: bool = True) -> None:
        """Deliver a same-position batch before timers tied to that position."""
        if self._closed:
            raise RuntimeError("Cannot deliver frames after session close")
        if not events:
            return
        if any(event.sample != events[0].sample for event in events):
            raise ValueError("A delivered event batch must share one audio progress position")
        await self.start()
        await self.clock.advance_to(events[0].sample, inclusive=False)
        for event in events:
            _validate_cascaded_frame(event.frame)
            await self._aggregator.queue_frame(event.frame, event.direction)
        if settle_timers:
            await self.clock.advance_to(events[0].sample)
        else:
            await self.clock.settle()

    async def advance_to(self, sample: int) -> None:
        """Execute policy timers through source or playback progress."""
        await self.start()
        await self.clock.advance_to(sample)

    async def finish(self, sample: int) -> None:
        """Drain EOF via stock EndFrame semantics, then release all resources."""
        if self._closed:
            return
        try:
            await self.feed(ChunkPositionedFrame(sample, EndFrame()))
        finally:
            await self.close()

    async def close(self) -> None:
        """Cancel outstanding policy waits without fabricating a user turn."""
        if self._closed:
            return
        self._closed = True
        if self._started:
            await self._aggregator.cleanup()
        if self._unused_vad:
            await self._unused_vad.cleanup()


def _validate_cascaded_frame(frame: Frame) -> None:
    if isinstance(frame, LLMServiceMetadataFrame) and frame.is_realtime_service:
        raise ValueError("Offline shared aggregation supports cascaded services, not realtime service handoffs")
    if isinstance(frame, UserIdleTimeoutUpdateFrame) and frame.timeout > 0:
        raise ValueError("Offline shared aggregation does not yet adapt enabled idle timers")


class _ObservedAssistantAggregator(LLMAssistantAggregator):
    """Observe the stock assistant's commits and interrupted-turn messages."""

    def __init__(self, owner: "SharedAssistantAggregation", context: LLMContext, params: LLMAssistantAggregatorParams):
        self._owner = owner
        super().__init__(context, params=params, enable_direct_mode=True)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        if isinstance(frame, ErrorFrame):
            self._owner.clock._errors.append(frame.exception or RuntimeError(frame.error))
        elif isinstance(frame, LLMContextFrame):
            messages = copy.deepcopy(frame.context.get_messages())
            self._owner.contexts.append((self._owner.clock.now_sample, messages))
            if self._owner.on_context:
                try:
                    await self._owner.on_context(self._owner.clock.now_sample, messages)
                except Exception as error:
                    self._owner.clock._errors.append(error)
                    raise

    async def _call_event_handler(self, event_name: str, *args, **kwargs):
        await super()._call_event_handler(event_name, *args, **kwargs)
        if event_name in {"on_assistant_turn_started", "on_assistant_turn_stopped"}:
            details = {}
            if args and hasattr(args[-1], "content"):
                details["text"] = args[-1].content
                details["interrupted"] = args[-1].interrupted
            await self._owner._event(event_name.removeprefix("on_"), **details)


class SharedAssistantAggregation:
    """Feed output-ordered frames through the actual assistant aggregator.

    Use the user adapter's exact context and clock. This adapter never advances
    that shared clock: its caller must already be at each frame's scheduled
    presentation position. Only text released by the output sequencer belongs
    here. Native text delivery can precede client PCM playback; history follows
    those frame semantics, without estimating spoken words from the waveform.
    The caller must send interruption before discarding pending output frames.
    """

    def __init__(
        self,
        context: LLMContext,
        clock: SampleClock,
        *,
        params: LLMAssistantAggregatorParams | None = None,
        on_context: ContextCallback | None = None,
        on_event: EventCallback | None = None,
    ):
        """Use stock cascaded assistant context policy with summarization off."""
        if version("pipecat-ai") != "1.12.0":
            raise RuntimeError("Offline aggregation adapter requires Pipecat 1.12.0")
        params = params or LLMAssistantAggregatorParams()
        if params.enable_auto_context_summarization:
            raise ValueError("Offline shared assistant aggregation does not yet support automatic summarization")
        self.context = context
        self.clock = clock
        self.on_context = on_context
        self.on_event = on_event
        self.contexts: list[tuple[int, list[dict[str, Any]]]] = []
        self.events: list[dict[str, Any]] = []
        self._aggregator = _ObservedAssistantAggregator(self, context, params)
        self._started = False
        self._closed = False

    async def _event(self, kind: str, **details):
        event = {"type": kind, "sample": self.clock.now_sample, **details}
        self.events.append(event)
        if self.on_event:
            try:
                await self.on_event(event)
            except Exception as error:
                self.clock._errors.append(error)
                raise

    async def start(self) -> None:
        """Initialize the stock direct-mode processor at current media time."""
        if self._started:
            return
        self._started = True
        await self._aggregator.setup(
            FrameProcessorSetup(clock=self.clock, task_manager=_SampleTaskManager(self.clock), pipeline_worker=None)
        )
        await self._aggregator.queue_frame(StartFrame())
        await self.clock.settle()

    async def feed(self, event: ChunkPositionedFrame) -> None:
        """Deliver an output frame at the scheduler's current sample pointer."""
        if self._closed:
            raise RuntimeError("Cannot deliver assistant frames after session close")
        if event.sample != self.clock.now_sample:
            raise ValueError("Assistant frames must be delivered at the shared scheduler's current sample position")
        _validate_cascaded_frame(event.frame)
        await self.start()
        await self._aggregator.queue_frame(event.frame, event.direction)
        await self.clock.settle()

    async def finish(self, sample: int) -> None:
        """Flush any delivered text with the stock EndFrame policy, then close."""
        if self._closed:
            return
        try:
            await self.feed(ChunkPositionedFrame(sample, EndFrame()))
        finally:
            await self.close()

    async def close(self) -> None:
        """Release the assistant processor without altering shared media time."""
        if self._closed:
            return
        self._closed = True
        if self._started:
            await self._aggregator.cleanup()
