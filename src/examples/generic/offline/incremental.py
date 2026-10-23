# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schedule incremental provider callbacks and native policies on one media clock."""

import asyncio
import copy
import heapq
import math
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    STTMetadataFrame,
    SystemFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_latency import DEFAULT_TTFS_P99

from .actors import ActorEvent, EventActor
from .aggregation import ChunkPositionedFrame, SharedAssistantAggregation, SharedUserAggregation
from .models import RATE, SessionResult, ns_to_samples, samples_to_ns
from .output_stream import IncrementalOutput


@dataclass(order=True)
class _Scheduled:
    sample: int
    serial: int
    kind: str = field(compare=False)
    payload: object = field(compare=False, default=None)
    generation: str | None = field(compare=False, default=None)


class IncrementalSessionEngine:
    """Run source analysis and persistent native response generation concurrently.

    A producer's lower bound constrains clock advancement until every earlier
    callback has been published. These bounds describe observed bulk processing,
    not a promise that provider timing equals a separately paced live run.
    """

    def __init__(
        self,
        detector,
        recognizer,
        responder,
        messages=(),
        *,
        params=None,
        output_chunks_10ms=40,
        output_sample_rate=RATE,
        history_recent_turns=None,
        preserve_prompt_messages=0,
    ):
        """Inject independently progressing sources and the native response port."""
        self.detector, self.recognizer, self.responder = detector, recognizer, responder
        self.initial_messages = copy.deepcopy(list(messages))
        self.params = params
        self.output_chunks_10ms = output_chunks_10ms
        self.output_sample_rate = output_sample_rate
        self.history_recent_turns = history_recent_turns
        self.preserve_prompt_messages = preserve_prompt_messages

    async def run(self, source):
        """Own one finite session and cancel every producer on failure."""
        return await _Session(self, source).run()


class _Session:
    def __init__(self, engine, source):
        self.engine, self.source = engine, source
        self.changed = asyncio.Event()
        self.actors = {}
        self.bounds = {}
        self.completed = {}
        self.queue = []
        self.serial = 0
        self.events = []
        self.responses = []
        self.requests = deque()
        self.active = None
        self.response_count = self.turn_count = 0
        self.last_vad_end = None
        self.last_vad_decision = None
        self.wait_count = 0
        self.user_turn_active = False
        self.physical_origin = time.monotonic_ns()
        self.asr_physical_start_ns = None
        self.output = IncrementalOutput(sample_rate=engine.output_sample_rate, chunks_10ms=engine.output_chunks_10ms)
        self.user = SharedUserAggregation(
            engine.initial_messages, params=engine.params, on_context=self._context, on_event=self._policy_event
        )
        self.assistant = SharedAssistantAggregation(self.user.context, self.user.clock, on_event=self._assistant_event)

    def emit(self, kind, sample=None, **details):
        self.events.append(
            {"type": kind, "sample": self.user.clock.now_sample if sample is None else sample, **details}
        )

    def schedule(self, sample, kind, payload=None, generation=None):
        if sample < self.user.clock.now_sample:
            raise ValueError("Provider event arrived behind the committed conversation clock")
        self.serial += 1
        heapq.heappush(self.queue, _Scheduled(sample, self.serial, kind, payload, generation))

    async def _context(self, sample, messages):
        self.requests.append((sample, copy.deepcopy(messages), f"turn-{self.turn_count}"))

    async def _assistant_event(self, event):
        self.events.append(event)
        if event["type"] == "assistant_turn_stopped" and self.engine.history_recent_turns is not None:
            from examples.shared.pipeline_utils import find_recent_turn_start

            history = self.user.context.get_messages()[self.engine.preserve_prompt_messages :]
            recent = self.engine.history_recent_turns
            if recent >= 1 and find_recent_turn_start(history, recent) > 0:
                raise ValueError("Incremental offline scheduling does not yet support context summarization")

    async def _policy_event(self, event):
        if event["type"] == "user_turn_started":
            self.turn_count += 1
            self.user_turn_active = True
        elif event["type"] == "user_turn_stopped":
            self.user_turn_active = False
        self.events.append({**event, "turn_id": f"turn-{self.turn_count}"})
        if event["type"] == "interruption" and self.active is not None:
            await self.finish_response(interrupted=True)

    async def notifications(self, items):
        for frame, direction in items:
            if direction == FrameDirection.DOWNSTREAM:
                await self.assistant.feed(ChunkPositionedFrame(self.user.clock.now_sample, frame))
                if isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
                    self.emit("bot_speech_start" if isinstance(frame, BotStartedSpeakingFrame) else "bot_speech_stop")
            else:
                await self.engine.responder.feedback(frame)
                await self.user.feed_batch(
                    [ChunkPositionedFrame(self.user.clock.now_sample, frame, direction)], settle_timers=False
                )

    async def start_response(self, sample, messages, turn_id):
        if self.active is not None:
            await self.finish_response(interrupted=True)
        self.response_count += 1
        identity = f"response-{self.response_count}"
        actor = await self.engine.responder.begin(messages, samples_to_ns(sample), identity, self.changed)
        self.actors[identity] = actor
        self.active = {
            "id": identity,
            "turn_id": turn_id,
            "sample": sample,
            "request_messages": messages,
            "vad_end": self.last_vad_end,
            "vad_decision": self.last_vad_decision,
            "first_audio": None,
            "physical_dispatch_ns": time.monotonic_ns() - self.physical_origin,
            "end_scheduled": False,
        }
        self.emit("response_requested", response_id=identity, turn_ids=[turn_id], request_messages=messages)

    async def finish_response(self, *, interrupted):
        response = self.active
        if response is None:
            return
        self.active = None
        identity = response["id"]
        sample = self.user.clock.now_sample
        if interrupted:
            await self.engine.responder.interrupt(identity, samples_to_ns(sample))
            await self.assistant.feed(ChunkPositionedFrame(sample, InterruptionFrame()))
            await self.notifications(await self.output.interrupt(identity, samples_to_ns(sample)))
            self.queue = [event for event in self.queue if event.generation != identity]
            heapq.heapify(self.queue)
        result = await self.actors[identity].result()
        self.completed[identity] = result
        self.bounds.pop(identity, None)
        self.actors.pop(identity, None)
        self.responses.append(
            {
                "response_id": identity,
                "turn_ids": [response["turn_id"]],
                "request_sample": response["sample"],
                "endpoint_sample": response["sample"],
                "latest_vad_speech_end_sample": response["vad_end"],
                "latest_vad_stop_decision_sample": response["vad_decision"],
                "first_packed_audio_ready_ns": response["first_audio"],
                "end_sample": sample,
                "compute_ns": result.metrics.get("response_observed_wall_ns", 0),
                "generated_samples": result.metrics.get("tts_generated_samples", 0),
                "committed_samples": 0,
                "interrupted": interrupted,
                "provider": result.metrics,
                "request_messages": response["request_messages"],
                "physical_dispatch_ns": response["physical_dispatch_ns"],
            }
        )
        self.emit("response_interrupted" if interrupted else "response_complete", response_id=identity)

    def start_detector(self):
        progress = [0]
        worker = []
        actor = EventActor(
            self.changed,
            lower_bound=lambda _: progress[0],
            cancel=lambda: worker[0].cancel() if worker else None,
        )

        async def checkpoint(events, ready_ns, chunk):
            actor.publish_batch(
                [
                    ActorEvent(ready_ns, "vad", event)
                    for event in events
                    if event["type"] in {"speech_start", "speech_end"}
                ],
                update=lambda: progress.__setitem__(0, ready_ns),
            )
            await asyncio.sleep(0)

        async def detect():
            try:
                trace = await self.engine.detector.detect(self.source, on_checkpoint=checkpoint)
                actor.finish(trace)
            except BaseException as error:
                actor.finish(error=error)

        task = asyncio.create_task(detect())
        worker.append(task)
        actor.bind_worker(task)
        return actor

    def collect(self):
        self.changed.clear()
        for identity, actor in tuple(self.actors.items()):
            snapshot = actor.drain()
            self.bounds[identity] = snapshot.lower_bound_ns
            for event in snapshot.events:
                self.schedule(ns_to_samples(event.ready_ns), event.kind, event.payload, event.generation)
            if snapshot.done and identity in {"asr", "vad"}:
                self.completed[identity] = True

    async def deliver(self, event):
        if event.generation is not None and (self.active is None or self.active["id"] != event.generation):
            self.emit("stale_provider_frame_discarded", response_id=event.generation)
            return
        sample = event.sample
        if event.kind == "vad":
            details = event.payload
            vad = self.engine.detector.configuration["vad"]
            if details["type"] == "speech_start":
                frame = VADUserStartedSpeakingFrame(start_secs=vad["start_secs"], timestamp=sample / RATE)
            else:
                self.last_vad_end = details["sample"]
                self.last_vad_decision = details["decision_sample"]
                frame = VADUserStoppedSpeakingFrame(stop_secs=vad["stop_secs"], timestamp=sample / RATE)
            self.emit("vad_decision", source_event=details)
            await self.user.feed_batch([ChunkPositionedFrame(sample, frame)], settle_timers=False)
        elif event.kind == "asr":
            value = event.payload
            self.emit(
                "asr_transcript",
                sequence=value.sequence,
                text=value.text,
                final=value.final,
                processed_sample=value.processed_sample,
                dependency_chunk_end=value.dependency_chunk_end,
            )
            frame = (
                TranscriptionFrame(value.text, "", "", finalized=True)
                if value.final
                else InterimTranscriptionFrame(value.text, "", "")
            )
            await self.user.feed_batch([ChunkPositionedFrame(sample, frame)], settle_timers=False)
        elif event.kind == "provider_frame":
            value = event.payload
            if value.direction == FrameDirection.DOWNSTREAM:
                if isinstance(value.frame, SystemFrame):
                    await self.assistant.feed(ChunkPositionedFrame(sample, value.frame))
                else:
                    await self.output.receive(event.generation, samples_to_ns(sample), value.frame)
            # Provider upstream service metadata follows the native user path.
            else:
                await self.user.feed_batch(
                    [ChunkPositionedFrame(sample, value.frame, value.direction)], settle_timers=False
                )
        elif event.kind == "provider_observation":
            self.emit("provider_observation", response_id=event.generation, frame=type(event.payload.frame).__name__)
        elif event.kind == "output_forward":
            await self.assistant.feed(ChunkPositionedFrame(sample, event.payload))
        elif event.kind == "response_end":
            await self.finish_response(interrupted=False)
        elif event.kind == "input_end":
            self.emit("input_end")

    async def output_step(self):
        sample = self.user.clock.now_sample
        delivery = await self.output.pop(samples_to_ns(sample))
        await self.notifications(delivery.notifications)
        if delivery.frame is None:
            return
        if isinstance(delivery.frame, OutputAudioRawFrame):
            if self.active and self.active["first_audio"] is None:
                self.active["first_audio"] = samples_to_ns(sample - self.active["sample"])
            self.schedule(ns_to_samples(delivery.complete_ns), "output_forward", delivery.frame, delivery.generation)
        else:
            await self.assistant.feed(ChunkPositionedFrame(sample, delivery.frame))

    async def wait_for_progress(self, target, bound):
        self.wait_count += 1
        # Dynamic pending-compute bounds can cross a target without a callback.
        # Static pre-EOF bounds only wake on actual supply/callback progress.
        delay = min(0.05, max(0.0001, (samples_to_ns(target) - bound + 1) / 1_000_000_000))
        with suppress(TimeoutError):
            await asyncio.wait_for(self.changed.wait(), delay)

    async def run(self):
        flushed = False
        failure = None
        try:
            await self.engine.responder.start()
            self.actors["vad"] = self.start_detector()
            self.asr_physical_start_ns = time.monotonic_ns() - self.physical_origin
            self.actors["asr"] = self.engine.recognizer.start(self.source, self.changed)
            self.schedule(self.source.sample_count, "input_end")
            await self.user.feed(ChunkPositionedFrame(0, STTMetadataFrame("NvidiaSTTService", DEFAULT_TTFS_P99)))
            while True:
                self.collect()
                if self.requests:
                    await self.start_response(*self.requests.popleft())
                    continue
                if self.active and math.isinf(self.bounds.get(self.active["id"], 0)):
                    identity = self.active["id"]
                    pending = any(e.generation == identity for e in self.queue)
                    if not pending and not self.output.pending and not self.active["end_scheduled"]:
                        self.active["end_scheduled"] = True
                        self.schedule(
                            max(self.user.clock.now_sample, self.output.presentation_sample),
                            "response_end",
                            generation=identity,
                        )
                next_output = ns_to_samples(self.output.next_ns) if self.output.next_ns is not None else None
                candidates = [e for e in (self.queue[0].sample if self.queue else None, next_output) if e is not None]
                # Input EOF does not supersede the native stop policy. Drain
                # its pending turn timers before flushing, but do not advance
                # forever through an idle controller's periodic watchdog.
                deadline = self.user.next_deadline_sample if self.user_turn_active and not flushed else None
                if deadline is not None:
                    candidates.append(deadline)
                sources_done = all(key in self.completed for key in ("asr", "vad"))
                if not candidates and sources_done and self.active is None:
                    if flushed:
                        break
                    flushed = True
                    await self.user.feed(ChunkPositionedFrame(self.user.clock.now_sample, EndFrame()))
                    continue
                if not candidates:
                    await self.wait_for_progress(self.user.clock.now_sample, min(self.bounds.values(), default=0))
                    continue
                target = min(candidates)
                deadline = self.user.next_deadline_sample
                if deadline is not None:
                    target = min(target, deadline)
                bound = min(self.bounds.values(), default=math.inf)
                if not math.isinf(bound) and target >= ns_to_samples(bound):
                    await self.wait_for_progress(target, bound)
                    continue
                await self.user.start()
                await self.user.clock.advance_to(target, inclusive=False)
                while self.queue and self.queue[0].sample == target:
                    await self.deliver(heapq.heappop(self.queue))
                while self.output.next_ns is not None and ns_to_samples(self.output.next_ns) <= target:
                    await self.output_step()
                await self.user.clock.advance_to(target)
            await self.assistant.finish(self.user.clock.now_sample)
            spans = self.output.spans()
            for response in self.responses:
                owned = [span for span in spans if span.response_id == response["response_id"]]
                response["committed_samples"] = sum(len(span.pcm) // 2 for span in owned)
                for span in owned:
                    self.emit("bot_audio", span.start_sample, response_id=span.response_id, end_sample=span.end_sample)
            rendered = max(self.source.sample_count, max((s.end_sample for s in spans), default=0))
            recognition = await self.actors["asr"].result()
            self.emit("session_complete", rendered_samples=rendered)
            return SessionResult(
                spans,
                rendered,
                tuple(sorted(self.events, key=lambda e: e["sample"])),
                tuple(copy.deepcopy(self.user.context.get_messages())),
                {
                    "recognition": recognition.metrics,
                    "responses": self.responses,
                    "recognition_binding_policy": "live_aggregator_event_order_no_word_timestamps",
                    "timing_policy": "incremental_source_chunks_plus_observed_bulk_compute",
                    "input_samples": self.source.sample_count,
                    "committed_samples": sum(len(s.pcm) // 2 for s in spans),
                    "scheduler_bound_waits": self.wait_count,
                    "runtime": "incremental_native_provider_actors",
                    "asr_physical_start_ns": self.asr_physical_start_ns,
                },
            )
        except BaseException as error:
            failure = error
            raise
        finally:
            cleanup_errors = []

            async def release(operation):
                try:
                    await operation()
                except BaseException as error:
                    cleanup_errors.append(error)

            await release(self.engine.responder.aclose)
            for actor in tuple(self.actors.values()):
                await release(actor.aclose)
            await release(self.assistant.close)
            await release(self.user.close)
            try:
                self.output.close()
            except BaseException as error:
                cleanup_errors.append(error)
            if cleanup_errors:
                if failure is None:
                    raise cleanup_errors[0]
                failure.add_note(
                    "Incremental session cleanup also failed: "
                    + ", ".join(type(error).__name__ for error in cleanup_errors)
                )
