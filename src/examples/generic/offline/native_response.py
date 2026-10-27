# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent native LLM/TTS actors with incremental, generation-fenced output.

The NVIDIA services own HTTP streaming, thinking filtering, sentence aggregation,
text filtering, synthesis and their internal serialization queues. This bridge
only routes their actual frames and timestamps provider availability. Native
processor queues remain enabled so an InterruptionFrame cancels the real LLM
processing task and resets TTS without replacing its service or channel.
"""

from __future__ import annotations

import asyncio
import copy
import time
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    CancelFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    MetricsFrame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.metrics.metrics import ProcessingMetricsData, TTFBMetricsData
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.nvidia.llm import NvidiaLLMService
from pipecat.services.nvidia.tts import NvidiaTTSService
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.errors import ErrorCategory

from .actors import ActorEvent, EventActor
from .models import RATE

Capture = Callable[[Frame, FrameDirection], Awaitable[None]]
_GENERATION_KEY = "nva_offline_response_generation"
_GENERATION: ContextVar[str | None] = ContextVar("nva_native_response_generation", default=None)
_ERROR_DIAGNOSTIC_KEY = "_nva_native_provider_error_diagnostic"


@dataclass(frozen=True, slots=True)
class NativeProviderErrorDiagnostic:
    """Keep owned processor identity separate from an observed callback edge."""

    provider_source: Literal["llm", "tts"] | None
    observed_callback_boundary: Literal["llm", "tts"]
    provider_error_category: ErrorCategory | None


def get_native_provider_error_diagnostic(error: BaseException) -> dict[str, str]:
    """Read allowlisted sidecar fields without exposing or stringifying an error."""
    try:
        diagnostic = getattr(error, _ERROR_DIAGNOSTIC_KEY, None)
    except Exception:
        return {}
    if type(diagnostic) is not NativeProviderErrorDiagnostic:
        return {}
    public = {}
    for field_name in ("provider_source", "observed_callback_boundary"):
        value = getattr(diagnostic, field_name)
        if isinstance(value, str) and value in {"llm", "tts"}:
            public[field_name] = value
    category = diagnostic.provider_error_category
    if isinstance(category, ErrorCategory):
        public["provider_error_category"] = category.name
    return public


class CapturingNativeNvidiaLLMService(NvidiaLLMService):
    """Delegate the native service, retaining generation identity across pushes."""

    def __init__(self, *, capture: Capture, **kwargs):
        """Use ordinary processor queues, including their interruption handling."""
        if kwargs.get("enable_direct_mode"):
            raise ValueError("Native response actors require the native LLM processor queues")
        super().__init__(**kwargs)
        self._native_capture = capture

    async def process_frame(self, frame, direction):
        """Scope native completion callbacks to the context frame that caused them."""
        token = _GENERATION.set(frame.metadata.get(_GENERATION_KEY))
        try:
            return await super().process_frame(frame, direction)
        finally:
            _GENERATION.reset(token)

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        """Preserve native side effects before forwarding an actual output frame."""
        generation = _GENERATION.get()
        if generation is not None:
            frame.metadata.setdefault(_GENERATION_KEY, generation)
        await super().push_frame(frame, direction)
        await self._native_capture(frame, direction)


class CapturingNativeNvidiaTTSService(NvidiaTTSService):
    """Keep native synthesis state persistent and correlate delayed context output."""

    def __init__(self, *, capture: Capture, **kwargs):
        """Observe native serialized frames without changing synthesis settings."""
        if kwargs.get("enable_direct_mode"):
            raise ValueError("Native response actors require the native TTS processor queues")
        super().__init__(**kwargs)
        self._native_capture = capture
        self._native_context_generations: dict[str, str] = {}
        self._native_workers: dict[str, asyncio.Event] = {}

    async def process_frame(self, frame, direction):
        """Retain provenance while native processing creates synthesis contexts."""
        token = _GENERATION.set(frame.metadata.get(_GENERATION_KEY))
        try:
            return await super().process_frame(frame, direction)
        finally:
            _GENERATION.reset(token)

    async def on_turn_context_created(self, context_id):
        """Bind a native context before its asynchronous audio can be produced."""
        generation = _GENERATION.get()
        if generation is not None:
            self._native_context_generations[context_id] = generation
        await super().on_turn_context_created(context_id)

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        """Fence old context callbacks even when a newer response is active."""
        context_id = getattr(frame, "context_id", None)
        generation = frame.metadata.get(_GENERATION_KEY)
        if generation is None and context_id is not None:
            generation = self._native_context_generations.get(context_id)
        if generation is None:
            generation = _GENERATION.get()
        token = _GENERATION.set(generation)
        try:
            if generation is not None:
                frame.metadata.setdefault(_GENERATION_KEY, generation)
            await super().push_frame(frame, direction)
            await self._native_capture(frame, direction)
        finally:
            _GENERATION.reset(token)

    async def _synth_task_handler(self, state):
        """Expose worker drain independently of serialized response completion."""
        finished = self._native_workers.setdefault(state.context_id, asyncio.Event())
        try:
            return await super()._synth_task_handler(state)
        finally:
            finished.set()

    async def wait_for_generation_workers(self, generation):
        """Await workers already represented by this generation's serialized EOF."""
        for context_id, owner in tuple(self._native_context_generations.items()):
            if owner == generation and context_id in self._native_workers:
                await self._native_workers[context_id].wait()


@dataclass(frozen=True)
class NativeProviderFrame:
    """A native frame at a provider boundary, before output transport pacing."""

    frame: Frame
    direction: FrameDirection
    source: str


@dataclass(frozen=True)
class NativeResponseResult:
    """Diagnostic generation summary; context remains owned by the aggregators."""

    text: str
    request_messages: list[dict[str, Any]]
    metrics: dict[str, Any]
    interrupted: bool


@dataclass
class _GenerationState:
    identity: str
    logical_start_ns: int
    wall_start_ns: int
    actor: EventActor
    messages: list[dict[str, Any]]
    serialized_end: asyncio.Event = field(default_factory=asyncio.Event)
    text: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    closed: bool = False
    interrupted: bool = False
    error: BaseException | None = None


class NativeResponseSession:
    """One persistent native LLM/TTS pair whose generations yield incrementally."""

    def __init__(self, llm_factory, tts_factory, *, sample_rate: int, clock=time.monotonic_ns, failure_timeout=600):
        """Factories accept a capture callback and return the native subclasses."""
        if sample_rate <= 0:
            raise ValueError("Native TTS sample rate must be positive")
        self._llm_factory = llm_factory
        self._tts_factory = tts_factory
        self._sample_rate = sample_rate
        self._clock = clock
        self._failure_timeout = failure_timeout
        self._manager = None
        self._llm = None
        self._tts = None
        self._active: _GenerationState | None = None
        self._generations: dict[str, _GenerationState] = {}
        self._interrupt_acks: dict[int, asyncio.Event] = {}
        self._background: set[asyncio.Task] = set()
        self._ready = None
        self._closed = False
        self._failure: BaseException | None = None

    async def start(self):
        """Set up both services before any generation's processing measurement."""
        if self._closed:
            raise RuntimeError("Native response session is closed")
        if self._failure is not None:
            raise self._failure
        if self._llm is not None:
            return
        self._ready = asyncio.get_running_loop().create_future()
        self._manager = TaskManager()
        provider_clock = SystemClock()
        provider_clock.start()
        setup = FrameProcessorSetup(
            clock=provider_clock,
            task_manager=self._manager,
            pipeline_worker=None,
            audio_in_sample_rate=RATE,
            audio_out_sample_rate=self._sample_rate,
            enable_metrics=True,
        )
        try:
            self._llm = self._llm_factory(self._from_llm)
            self._tts = self._tts_factory(self._from_tts)
            await self._llm.setup(setup)
            await self._tts.setup(setup)
            await self._llm.queue_frame(StartFrame())
            async with asyncio.timeout(self._failure_timeout):
                await self._ready
        except BaseException as error:
            try:
                await self.aclose()
            except BaseException as cleanup_error:
                error.add_note(f"Native response setup cleanup also failed: {type(cleanup_error).__name__}")
            raise

    def _state_for(self, frame):
        identity = frame.metadata.get(_GENERATION_KEY)
        return self._generations.get(identity) if identity is not None else None

    def _publish(self, state, kind, payload):
        if state.closed:
            return
        state.actor.publish_batch(
            lambda now: (
                ActorEvent(state.logical_start_ns + max(0, now - state.wall_start_ns), kind, payload, state.identity),
            )
        )

    def _provider_error(self, state, frame, *, observed_callback_boundary):
        error = frame.exception if getattr(frame, "exception", None) is not None else RuntimeError(frame.error)
        source = None
        if self._llm is not None and frame.processor is self._llm:
            source = "llm"
        elif self._tts is not None and frame.processor is self._tts:
            source = "tts"
        category = frame.category if isinstance(frame.category, ErrorCategory) else None
        diagnostic = NativeProviderErrorDiagnostic(source, observed_callback_boundary, category)
        # Keep the original exception, including typed gRPC information. Some
        # exception implementations forbid extra attributes; diagnostics must
        # never replace their failure or change the actor's completion path.
        try:
            if type(getattr(error, _ERROR_DIAGNOSTIC_KEY, None)) is not NativeProviderErrorDiagnostic:
                setattr(error, _ERROR_DIAGNOSTIC_KEY, diagnostic)
        except Exception:
            pass
        if state is None:
            # A native background task may report a session-level failure
            # outside a synthesis context. Never turn that failure into an
            # apparently silent, indefinitely pending response.
            self._failure = self._failure or error
            state = self._active
        if state is not None and not state.closed:
            state.error = state.error or error
            state.serialized_end.set()
        elif self._ready is not None and not self._ready.done():
            self._ready.set_exception(error)

    async def _from_llm(self, frame, direction):
        if isinstance(frame, StartFrame):
            await self._tts.queue_frame(frame, direction)
            return
        if isinstance(frame, InterruptionFrame):
            # An interruption must reach the persistent TTS even after its
            # generation has been fenced off from conversational output.
            await self._tts.queue_frame(frame, direction)
            return
        state = self._state_for(frame)
        if isinstance(frame, ErrorFrame):
            self._provider_error(state, frame, observed_callback_boundary="llm")
            return
        if state is None or state.closed:
            return
        self._publish(state, "provider_observation", NativeProviderFrame(frame, direction, "llm"))
        elapsed = max(0, self._clock() - state.wall_start_ns)
        if isinstance(frame, LLMTextFrame) and frame.text:
            state.metrics.setdefault("llm_first_text_ns", elapsed)
            state.text.append(frame.text)
        if isinstance(frame, LLMFullResponseEndFrame):
            state.metrics["llm_complete_ns"] = elapsed
        if direction == FrameDirection.DOWNSTREAM:
            await self._tts.queue_frame(frame, direction)

    async def _from_tts(self, frame, direction):
        if isinstance(frame, StartFrame):
            if self._ready is not None and not self._ready.done():
                self._ready.set_result(None)
            return
        if isinstance(frame, InterruptionFrame):
            ack = self._interrupt_acks.get(frame.id)
            if ack is not None:
                ack.set()
            return
        state = self._state_for(frame)
        if isinstance(frame, ErrorFrame):
            self._provider_error(state, frame, observed_callback_boundary="tts")
            return
        if state is None or state.closed:
            return
        self._publish(state, "provider_frame", NativeProviderFrame(frame, direction, "tts"))
        elapsed = max(0, self._clock() - state.wall_start_ns)
        if isinstance(frame, TTSAudioRawFrame):
            state.metrics.setdefault("first_audio_ready_ns", elapsed)
            native_samples = state.metrics.get("tts_generated_native_samples", 0) + len(frame.audio) // (
                2 * max(frame.num_channels, 1)
            )
            state.metrics["tts_generated_native_samples"] = native_samples
            state.metrics["tts_native_sample_rate"] = frame.sample_rate
            state.metrics["tts_generated_samples"] = round(native_samples * RATE / frame.sample_rate)
        if isinstance(frame, MetricsFrame):
            values = state.metrics.setdefault("native_provider_metrics", [])
            for datum in frame.data:
                if isinstance(datum, (TTFBMetricsData, ProcessingMetricsData)):
                    values.append(
                        {
                            "kind": type(datum).__name__,
                            "processor": datum.processor,
                            "value_ns": round(datum.value * 1e9),
                        }
                    )
        if isinstance(frame, LLMFullResponseEndFrame) and direction == FrameDirection.DOWNSTREAM:
            state.serialized_end.set()

    async def begin(self, messages: Sequence[dict], logical_start_ns: int, generation_id: str, changed: asyncio.Event):
        """Dispatch a context snapshot and return before its first token or PCM."""
        await self.start()
        if self._active is not None and not self._active.closed:
            raise RuntimeError("Interrupt or drain the active generation before starting another")
        if generation_id in self._generations:
            raise ValueError("Native response generation IDs must be unique within a session")
        snapshot = copy.deepcopy(list(messages))
        began = self._clock()
        actor = EventActor(
            changed,
            clock=self._clock,
            lower_bound=lambda now: logical_start_ns + max(0, now - began),
            cancel=lambda: self._schedule_interrupt(generation_id, logical_start_ns),
        )
        state = _GenerationState(generation_id, logical_start_ns, began, actor, snapshot)
        self._generations[generation_id] = state
        self._active = state
        self._publish(state, "response_dispatched", {"messages": copy.deepcopy(snapshot)})
        request = LLMContextFrame(LLMContext(messages=copy.deepcopy(snapshot)))
        request.metadata[_GENERATION_KEY] = generation_id
        await self._llm.queue_frame(request)
        task = asyncio.create_task(self._finish_generation(state))
        actor.bind_worker(task)
        return actor

    async def _finish_generation(self, state):
        try:
            async with asyncio.timeout(self._failure_timeout):
                await state.serialized_end.wait()
                if state.error is not None:
                    raise state.error
                if not state.interrupted:
                    await self._tts.wait_for_generation_workers(state.identity)
            elapsed = max(0, self._clock() - state.wall_start_ns)
            state.metrics.update(
                response_observed_wall_ns=elapsed,
                llm_first_text_censored=state.interrupted and "llm_first_text_ns" not in state.metrics,
                tts_rpc_eof_drained=not state.interrupted,
                timing_basis="native_incremental_provider_elapsed_including_queueing",
            )
            state.closed = True
            state.actor.finish(
                NativeResponseResult(
                    "".join(state.text), copy.deepcopy(state.messages), dict(state.metrics), state.interrupted
                )
            )
        except BaseException as error:
            state.closed = True
            state.actor.finish(error=error)

    def _schedule_interrupt(self, identity, logical_ns):
        task = asyncio.create_task(self.interrupt(identity, logical_ns))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def interrupt(self, generation_id, logical_ns):
        """Fence output, then use the same native interruption as a live pipeline."""
        state = self._generations.get(generation_id)
        if state is None or state is not self._active or state.interrupted:
            return
        state.interrupted = True
        state.closed = True
        state.metrics["interrupted_at_logical_ns"] = logical_ns
        # Unconsumed events from this generation cannot cross an interruption.
        state.actor.drain()
        frame = InterruptionFrame()
        frame.metadata[_GENERATION_KEY] = generation_id
        ack = asyncio.Event()
        self._interrupt_acks[frame.id] = ack
        try:
            await self._llm.queue_frame(frame)
            async with asyncio.timeout(self._failure_timeout):
                await ack.wait()
        finally:
            self._interrupt_acks.pop(frame.id, None)
            state.serialized_end.set()

    async def feedback(self, frame):
        """Deliver native sender bot-speaking feedback without creating output."""
        if self._tts is not None and not self._closed:
            await self._tts.queue_frame(frame, FrameDirection.UPSTREAM)
        if self._llm is not None and not self._closed:
            await self._llm.queue_frame(frame, FrameDirection.UPSTREAM)

    async def aclose(self):
        """Cancel and clean up providers only when the owning session ends."""
        if self._closed:
            return
        self._closed = True
        errors = []

        async def release(operation):
            try:
                await operation()
            except BaseException as error:
                errors.append(error)

        if self._active is not None and not self._active.closed:
            await release(lambda: self.interrupt(self._active.identity, self._active.logical_start_ns))
        for service in (self._llm, self._tts):
            if service is not None:
                await release(lambda service=service: service.cancel(CancelFrame()))
                await release(service.cleanup)
        # Pipecat 1.12's NVIDIA/OpenAI cleanup does not close its AsyncOpenAI
        # client. This session created the native service and owns that client;
        # retain it across interruptions and release it only here.
        client = getattr(self._llm, "_client", None)
        if client is not None:
            await release(client.close)
        if self._manager is not None:
            try:
                tasks = tuple(self._manager.current_tasks())
            except BaseException as error:
                errors.append(error)
            else:
                for task in tasks:
                    await release(lambda task=task: self._manager.cancel_task(task))
        for state in self._generations.values():
            if not state.closed:
                state.closed = True
                state.interrupted = True
            state.serialized_end.set()
            await release(state.actor.aclose)
            # Dispatch can fail before its result worker is bound. Such an
            # actor must not leave a permanently pending completion future.
            state.actor.finish(error=RuntimeError("Native response session closed before dispatch completed"))
        if self._background:
            outcomes = await asyncio.gather(*tuple(self._background), return_exceptions=True)
            errors.extend(outcome for outcome in outcomes if isinstance(outcome, BaseException))
        if errors:
            raise errors[0]
