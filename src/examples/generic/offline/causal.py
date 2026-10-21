# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive live aggregation from drained ASR events and source chunk decisions."""

import asyncio
import copy
import heapq
from collections import deque
from dataclasses import dataclass, field

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    OutputAudioRawFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.stt_latency import DEFAULT_TTFS_P99
from pipecat.transports.base_output import BOT_VAD_STOP_FALLBACK_SECS

from .aggregation import ChunkPositionedFrame, SharedAssistantAggregation, SharedUserAggregation
from .models import RATE, AudioSpan, SessionResult, ns_to_samples, samples_to_ns
from .playback import WebsocketSinkClock, canonicalize, pack_frames


@dataclass(order=True)
class _Event:
    sample: int
    priority: int
    serial: int
    kind: str = field(compare=False)
    payload: object = field(compare=False)


class CausalSessionEngine:
    """Use the live aggregation policies; no transcript-to-VAD ownership exists.

    Source-only recognition is drained before replay, preserving one continuous
    request. Conversation/context advances only through delivered events. This
    first integration supports the configured Silero policy and tools disabled.
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
    ):
        """Inject source-only observation and response-generation ports."""
        self.detector = detector
        self.recognizer = recognizer
        self.responder = responder
        self.initial_messages = copy.deepcopy(list(messages))
        self.params = params
        self.output_chunks_10ms = output_chunks_10ms
        self.output_sample_rate = output_sample_rate

    async def run(self, source):
        """Drain source observations, then drive one causal conversation."""
        tasks = [
            asyncio.create_task(self.detector.detect(source)),
            asyncio.create_task(self.recognizer.recognize(source)),
        ]
        try:
            detection, recognition = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            self.recognizer.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        if not recognition.metrics.get("asr_rpc_eof_drained"):
            raise ValueError("Causal replay requires an explicitly drained ASR event trace")
        return await _Session(self, source, detection, recognition).run()


class _Session:
    def __init__(self, engine, source, detection, recognition):
        self.engine, self.source = engine, source
        self.detection, self.recognition = detection, recognition
        self.queue = []
        self.serial = 0
        self.events = []
        self.requests = deque()
        self.spans = []
        self.responses = []
        self.active = None
        self.turn_count = 0
        self.response_count = 0
        self.sink_clock = WebsocketSinkClock()
        self.aggregation = SharedUserAggregation(
            engine.initial_messages, params=engine.params, on_context=self._context, on_event=self._policy_event
        )
        self.assistant = SharedAssistantAggregation(context=self.aggregation.context, clock=self.aggregation.clock)

    def emit(self, kind, sample, **details):
        self.events.append({"type": kind, "sample": sample, **details})

    def schedule(self, sample, priority, kind, payload=None):
        self.serial += 1
        heapq.heappush(self.queue, _Event(sample, priority, self.serial, kind, payload))

    async def _context(self, sample, messages):
        self.requests.append((sample, copy.deepcopy(messages), f"turn-{self.turn_count}"))

    async def _policy_event(self, event):
        if event["type"] == "user_turn_started":
            self.turn_count += 1
        self.events.append({**event, "turn_id": f"turn-{self.turn_count}"})
        if event["type"] == "interruption":
            self.sink_clock.interrupt()
            if self.active:
                await self.finish_response(event["sample"], interrupted=True)

    async def assistant_frame(self, sample, frame):
        await self.assistant.feed(ChunkPositionedFrame(sample, frame))

    def source_events(self):
        vad = self.engine.detector.configuration["vad"]
        for event in self.detection.events:
            if event["type"] == "speech_start":
                self.schedule(
                    event["confirmed_sample"],
                    0,
                    "frame",
                    VADUserStartedSpeakingFrame(
                        start_secs=vad["start_secs"], timestamp=event["confirmed_sample"] / RATE
                    ),
                )
            elif event["type"] == "speech_end":
                self.schedule(
                    event["decision_sample"],
                    0,
                    "frame",
                    VADUserStoppedSpeakingFrame(stop_secs=vad["stop_secs"], timestamp=event["decision_sample"] / RATE),
                )
        for expected, event in enumerate(self.recognition.events):
            if event.sequence != expected or not 0 <= event.dependency_chunk_end <= self.source.sample_count:
                raise ValueError("ASR event sequence and source progress must be valid")
            if event.available_sample < event.dependency_chunk_end:
                raise ValueError("ASR event cannot precede its required source prefix")
            self.emit(
                "asr_transcript",
                event.available_sample,
                sequence=event.sequence,
                text=event.text,
                final=event.final,
                processed_sample=event.processed_sample,
                dependency_chunk_end=event.dependency_chunk_end,
            )
            frame = (
                TranscriptionFrame(event.text, "", "", finalized=True)
                if event.final
                else InterimTranscriptionFrame(event.text, "", "")
            )
            self.schedule(event.available_sample, 1, "frame", frame)
        self.schedule(self.source.sample_count, 2, "input_end")

    async def start_response(self, sample, messages, turn_id):
        if self.active:
            await self.finish_response(sample, interrupted=True)
        self.response_count += 1
        response_id = f"response-{self.response_count}"
        self.emit("response_requested", sample, response_id=response_id, turn_ids=[turn_id])
        await self.assistant_frame(sample, LLMFullResponseStartFrame())
        # After this committed turn, the next speech/transcription start is the
        # next possible interruption. Limit *actual compute* to that media gap;
        # do not run tools to completion and retrospectively cancel their audio.
        future = [
            e.sample
            for e in self.queue
            if e.sample > sample
            and e.kind == "frame"
            and isinstance(e.payload, (VADUserStartedSpeakingFrame, TranscriptionFrame, InterimTranscriptionFrame))
        ]
        budget = samples_to_ns(min(future) - sample) if future else None
        trace = await self.engine.responder.respond(messages, turn_id, compute_budget_ns=budget)
        packed = canonicalize(
            await pack_frames(
                trace,
                self.engine.output_chunks_10ms,
                sample_rate=self.engine.output_sample_rate,
                sink_clock=self.sink_clock,
                start_ns=samples_to_ns(sample),
            )
        )
        planned = []
        end = sample
        for item in packed:
            delivered = sample + ns_to_samples(item.delivery_offset_ns)
            completed = sample + ns_to_samples(item.write_complete_offset_ns)
            if isinstance(item.frame, OutputAudioRawFrame):
                # Socket send and native consumer completion are distinct from
                # continuous client PCM presentation. Preserve canonical sample
                # counts instead of independently rounding native chunk lengths.
                planned.append(AudioSpan(max(delivered, end), item.frame.audio, response_id))
                end = planned[-1].end_sample
            if not isinstance(item.frame, LLMFullResponseStartFrame):
                self.schedule(delivered, 3, "output_frame", (response_id, item.frame, completed))
                if isinstance(item.frame, OutputAudioRawFrame):
                    self.schedule(completed, 3, "output_forward", (response_id, item.frame))
        completion = max(end, sample + ns_to_samples(trace.compute_ns))
        self.active = {
            "id": response_id,
            "turn_id": turn_id,
            "sample": sample,
            "trace": trace,
            "request_messages": copy.deepcopy(messages),
            "spans": planned,
            "bot_speaking": False,
            "tts_audio_received": False,
            "idle_generation": 0,
            "legacy_audio": getattr(trace, "frames", None) is None,
        }
        self.schedule(completion, 4, "response_end", response_id)
        for event in trace.metrics.get("context_events", ()):
            if not 0 <= event["ready_offset_ns"] <= trace.compute_ns:
                raise ValueError("Context changes must have causal provider availability")
            self.schedule(sample + ns_to_samples(event["ready_offset_ns"]), 2, "context", (response_id, event))

    async def bot_started(self, response, sample):
        """Mirror MediaSender's idempotent TTS playback start notification."""
        if response["bot_speaking"]:
            return
        response["bot_speaking"] = True
        frame = BotStartedSpeakingFrame()
        await self.assistant_frame(sample, frame)
        await self.aggregation.feed(ChunkPositionedFrame(sample, frame))
        self.emit("bot_speech_start", sample, response_id=response["id"])

    async def bot_stopped(self, response, sample, *, reason):
        """Stop one playback episode without completing the LLM response."""
        if not response["bot_speaking"]:
            return
        response["bot_speaking"] = False
        response["tts_audio_received"] = False
        frame = BotStoppedSpeakingFrame()
        await self.assistant_frame(sample, frame)
        await self.aggregation.feed(ChunkPositionedFrame(sample, frame))
        self.emit("bot_speech_stop", sample, response_id=response["id"], reason=reason)

    async def output_frame(self, response, sample, frame, completed):
        """Consume packed FIFO items with the live sender's speaking semantics.

        WS sends before its pacing sleep, so the first write completes without
        consuming a chunk duration. A queue wait starts after write completion;
        text and control frames follow this consumer clock, not client playout.
        """
        if isinstance(frame, OutputAudioRawFrame):
            # Real providers retain the TTS subtype. Scripted legacy providers
            # expose only PCM, whose established compatibility meaning is TTS.
            if isinstance(frame, TTSAudioRawFrame) or response["legacy_audio"]:
                response["tts_audio_received"] = True
                await self.bot_started(response, sample)
        elif isinstance(frame, TTSStoppedFrame) and response["tts_audio_received"]:
            await self.bot_stopped(response, sample, reason="tts_stopped")
        if not isinstance(frame, OutputAudioRawFrame):
            await self.assistant_frame(sample, frame)
        response["idle_generation"] += 1
        if response["bot_speaking"]:
            self.schedule(
                completed + round(BOT_VAD_STOP_FALLBACK_SECS * RATE),
                5,
                "bot_idle",
                (response["id"], response["idle_generation"]),
            )

    async def finish_response(self, sample, *, interrupted):
        response = self.active
        if response is None:
            return
        self.active = None
        if interrupted:
            self.sink_clock.interrupt()
        played = 0
        for span in response["spans"]:
            count = min(len(span.pcm) // 2, max(0, sample - span.start_sample))
            if count:
                committed = AudioSpan(span.start_sample, span.pcm[: count * 2], span.response_id)
                self.spans.append(committed)
                played += count
                self.emit(
                    "bot_audio", committed.start_sample, response_id=span.response_id, end_sample=committed.end_sample
                )
        if interrupted:
            await self.assistant_frame(sample, InterruptionFrame())
        await self.bot_stopped(response, sample, reason="interruption" if interrupted else "response_complete")
        # Neither a cancelled output item nor a prior queue-wait timeout may
        # change a newer response, or advance the session beyond its real end.
        self.queue = [
            event
            for event in self.queue
            if not (
                event.kind == "response_end"
                and event.payload == response["id"]
                or event.kind in {"output_frame", "output_forward", "context", "bot_idle"}
                and event.payload[0] == response["id"]
            )
        ]
        heapq.heapify(self.queue)
        self.emit("response_interrupted" if interrupted else "response_complete", sample, response_id=response["id"])
        self.responses.append(
            {
                "response_id": response["id"],
                "turn_ids": [response["turn_id"]],
                "request_sample": response["sample"],
                "endpoint_sample": response["sample"],
                "first_packed_audio_ready_ns": (
                    samples_to_ns(response["spans"][0].start_sample - response["sample"]) if response["spans"] else None
                ),
                "latest_vad_speech_end_sample": next(
                    (
                        event["sample"]
                        for event in reversed(self.detection.events)
                        if event["type"] == "speech_end" and event["decision_sample"] <= response["sample"]
                    ),
                    None,
                ),
                "end_sample": sample,
                "compute_ns": response["trace"].compute_ns,
                "generated_samples": sum(len(a.pcm) // 2 for a in response["trace"].audio),
                "committed_samples": played,
                "interrupted": interrupted,
                "provider": response["trace"].metrics,
                "request_messages": response["request_messages"],
            }
        )

    async def run(self):
        self.source_events()
        await self.aggregation.feed(ChunkPositionedFrame(0, STTMetadataFrame("NvidiaSTTService", DEFAULT_TTFS_P99)))
        try:
            flushed = False
            while self.queue or self.requests or not flushed:
                if self.requests:
                    sample, messages, turn_id = self.requests.popleft()
                    await self.start_response(sample, messages, turn_id)
                    continue
                if not self.queue:
                    flushed = True
                    await self.aggregation.feed(ChunkPositionedFrame(self.aggregation.clock.now_sample, EndFrame()))
                    continue
                next_sample = self.queue[0].sample
                deadline = self.aggregation.next_deadline_sample
                if deadline is not None and deadline < next_sample:
                    await self.aggregation.advance_to(deadline)
                    continue
                batch = []
                other = []
                while self.queue and self.queue[0].sample == next_sample:
                    event = heapq.heappop(self.queue)
                    if event.kind == "frame":
                        batch.append(ChunkPositionedFrame(next_sample, event.payload))
                    else:
                        other.append(event)
                if batch:
                    await self.aggregation.feed_batch(batch)
                else:
                    await self.aggregation.advance_to(next_sample)
                for event in other:
                    if event.kind == "input_end":
                        self.emit("input_end", event.sample)
                    elif event.kind == "context":
                        response_id, change = event.payload
                        if self.active and self.active["id"] == response_id:
                            if "replace_messages" in change:
                                self.aggregation.context.set_messages(copy.deepcopy(change["replace_messages"]))
                            elif "message" in change:
                                self.aggregation.context.add_message(copy.deepcopy(change["message"]))
                    elif event.kind == "output_frame":
                        response_id, frame, completed = event.payload
                        if self.active and self.active["id"] == response_id:
                            await self.output_frame(self.active, event.sample, frame, completed)
                    elif event.kind == "output_forward":
                        response_id, frame = event.payload
                        if self.active and self.active["id"] == response_id:
                            await self.assistant_frame(event.sample, frame)
                    elif event.kind == "bot_idle":
                        response_id, generation = event.payload
                        if (
                            self.active
                            and self.active["id"] == response_id
                            and self.active["idle_generation"] == generation
                        ):
                            await self.bot_stopped(self.active, event.sample, reason="output_starvation")
                    elif self.active and self.active["id"] == event.payload:
                        await self.finish_response(event.sample, interrupted=False)
            await self.assistant.finish(self.aggregation.clock.now_sample)
            rendered = max(self.source.sample_count, max((s.end_sample for s in self.spans), default=0))
            self.emit("session_complete", self.aggregation.clock.now_sample, rendered_samples=rendered)
            return SessionResult(
                tuple(sorted(self.spans, key=lambda s: s.start_sample)),
                rendered,
                tuple(sorted(self.events, key=lambda e: e["sample"])),
                tuple(copy.deepcopy(self.aggregation.context.get_messages())),
                {
                    "recognition": self.recognition.metrics,
                    "responses": self.responses,
                    "recognition_binding_policy": "live_aggregator_event_order_no_word_timestamps",
                    "timing_policy": "source_chunks_plus_observed_bulk_compute",
                    "input_samples": self.source.sample_count,
                    "committed_samples": sum(len(s.pcm) // 2 for s in self.spans),
                },
            )
        finally:
            await self.assistant.close()
            await self.aggregation.close()
