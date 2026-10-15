# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Causal offline conversation execution on an integer sample timeline.

Source-only analysis may look ahead; conversation history may not. Provider
results are provisional traces replayed against the immutable source events.
There are deliberately no sleeps or system-clock reads in this module.
"""

import asyncio
import copy
import heapq
from dataclasses import dataclass, field
from typing import Any

from .models import (
    AudioSource,
    AudioSpan,
    DetectionTrace,
    Detector,
    RecognitionTrace,
    Recognizer,
    Responder,
    ResponseTrace,
    SessionResult,
    UserTurn,
    ns_to_samples,
)


@dataclass(order=True)
class _Event:
    sample: int
    priority: int
    serial: int
    kind: str = field(compare=False)
    payload: Any = field(compare=False)


@dataclass
class _Response:
    response_id: str
    turn_ids: tuple[str, ...]
    request_sample: int
    trace: ResponseTrace
    spans: tuple[AudioSpan, ...]
    completion_sample: int
    generated_samples: int
    committed_tool_call_ids: set[str] = field(default_factory=set)


class OfflineSessionEngine:
    """Coordinate source events, measured provider traces, and committed playback.

    Scheduling state belongs to a run. Provider ports own their lifecycle and
    concurrency policy; the generic factory allocates them per session.
    """

    def __init__(
        self,
        detector: Detector,
        recognizer: Recognizer,
        responder: Responder,
        messages: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    ):
        """Inject analysis/provider ports and the initial conversation context."""
        self._detector = detector
        self._recognizer = recognizer
        self._responder = responder
        self._messages = copy.deepcopy(list(messages))

    async def run(self, source: AudioSource) -> SessionResult:
        """Seal, analyze, then replay one finite source without playback pacing."""
        tasks = [
            asyncio.create_task(self._detector.detect(source)),
            asyncio.create_task(self._recognizer.recognize(source)),
        ]
        try:
            detection, recognition = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        replay = _Replay(source, detection, recognition, self._responder, self._messages)
        return await replay.run()


class _Replay:
    def __init__(self, source, detection, recognition, responder, messages):
        self.source: AudioSource = source
        self.detection: DetectionTrace = detection
        self.recognition: RecognitionTrace = recognition
        self.responder: Responder = responder
        self.messages = copy.deepcopy(messages)
        self.events: list[dict[str, Any]] = []
        self.queue: list[_Event] = []
        self.serial = 0
        self.spans: list[AudioSpan] = []
        self.active: _Response | None = None
        self.active_speech: set[int] = set()
        self.seen: set[str] = set()
        self.ready: set[str] = set()
        self.consumed: set[str] = set()
        self.response_count = 0
        self.response_metrics: list[dict[str, Any]] = []
        self.turns = self._validate_detection()
        self.texts, self.asr_ready = self._bind_recognition()
        self._source_events()

    def _validate_detection(self) -> dict[str, UserTurn]:
        turns: dict[str, UserTurn] = {}
        previous_end = -1
        for turn in self.detection.turns:
            if (
                turn.turn_id in turns
                or not 0 <= turn.start_sample < turn.end_sample <= turn.endpoint_sample <= self.source.sample_count
                or turn.start_sample < previous_end
                or turn.detection_compute_ns < 0
            ):
                raise ValueError("Turn proposals must be ordered, unique, and inside the source")
            turns[turn.turn_id] = turn
            previous_end = turn.end_sample
        previous_end = -1
        for speech in self.detection.speech:
            if (
                not 0 <= speech.start_sample < speech.end_sample
                or not speech.start_sample <= speech.confirmed_start_sample <= speech.end_sample
                or speech.end_sample > self.source.sample_count
                or speech.start_sample < previous_end
            ):
                raise ValueError("Speech intervals must be ordered and have valid confirmation offsets")
            owners = self._owners(speech.start_sample, speech.end_sample, turns)
            if len(owners) != 1:
                raise ValueError("Each speech interval must belong to exactly one turn")
            previous_end = speech.end_sample
        return turns

    @staticmethod
    def _owners(start, end, turns):
        return [key for key, turn in turns.items() if start < turn.end_sample and end > turn.start_sample]

    def _recognition_owners(self, start, end):
        """Bind transcript coverage to an open turn, including VAD stop silence.

        VAD speech end is an acoustic estimate, not the close of an utterance:
        quiet trailing words may follow it before the endpoint decision. Riva
        also emits valid zero-duration words. Treat those as bounded points on
        the half-open turn window, with explicit source EOF closing its final
        point. An ordinary endpoint never owns speech after that endpoint.
        """
        if start == end:
            return [
                key
                for key, turn in self.turns.items()
                if turn.start_sample <= start < turn.endpoint_sample
                or start == self.source.sample_count == turn.endpoint_sample
            ]
        return [key for key, turn in self.turns.items() if start < turn.endpoint_sample and end > turn.start_sample]

    def _bind_recognition(self) -> tuple[dict[str, str], dict[str, int]]:
        fragments: dict[str, list[tuple[int, int, str]]] = {key: [] for key in self.turns}
        readiness = {key: 0 for key in self.turns}
        for index, segment in enumerate(self.recognition.segments):
            if (
                not 0 <= segment.start_sample <= segment.end_sample <= self.source.sample_count
                or segment.available_sample < segment.end_sample
            ):
                raise ValueError("ASR coverage and availability offsets must be valid and causal")
            text = segment.text.strip()
            if not text:
                continue
            owners = self._recognition_owners(segment.start_sample, segment.end_sample)
            if not owners:
                raise ValueError("Final ASR text has no detected source turn")
            if len(owners) == 1:
                owner = owners[0]
                fragments[owner].append((segment.start_sample, index, text))
                readiness[owner] = max(readiness[owner], segment.available_sample)
                continue
            if not segment.words:
                raise ValueError("ASR segment spans multiple turns without word offsets")
            grouped: dict[str, list[tuple[int, str]]] = {}
            previous_end = segment.start_sample
            for word, start, end in segment.words:
                if not segment.start_sample <= start <= end <= segment.end_sample or start < previous_end:
                    raise ValueError("ASR word offsets must be ordered and inside their segment")
                word_owners = self._recognition_owners(start, end)
                if len(word_owners) != 1:
                    raise ValueError("ASR word has ambiguous turn ownership")
                grouped.setdefault(word_owners[0], []).append((start, word))
                previous_end = end
            for owner, words in grouped.items():
                fragments[owner].append((words[0][0], index, " ".join(word for _, word in words)))
                readiness[owner] = max(readiness[owner], segment.available_sample)
        texts = {key: " ".join(text for _, _, text in sorted(parts)) for key, parts in fragments.items()}
        return texts, readiness

    def _schedule(self, sample, priority, kind, payload):
        self.serial += 1
        heapq.heappush(self.queue, _Event(sample, priority, self.serial, kind, payload))

    def _emit(self, kind, sample, **details):
        self.events.append({"type": kind, "sample": sample, **details})

    def _source_events(self):
        for index, speech in enumerate(self.detection.speech):
            owner = self._owners(speech.start_sample, speech.end_sample, self.turns)[0]
            # A run exactly as long as the VAD start threshold is confirmed at
            # its acoustic end. Start must still precede end at that offset so
            # interruption occurs and active speech is cleared before replying.
            self._schedule(speech.confirmed_start_sample, 0, "speech_start", (index, owner, speech))
            self._schedule(speech.end_sample, 1, "speech_end", (index, owner))
        for turn in self.turns.values():
            self._schedule(turn.endpoint_sample, 2, "endpoint", turn)
            ready = max(
                turn.endpoint_sample + ns_to_samples(turn.detection_compute_ns),
                self.asr_ready[turn.turn_id],
            )
            self._schedule(ready, 3, "turn_ready", turn)
        for segment in self.recognition.segments:
            self._schedule(segment.available_sample, 3, "asr_final", segment)
        self._schedule(self.source.sample_count, 2, "input_end", None)
        for event in self.detection.events:
            if "sample" not in event or not 0 <= event["sample"] <= self.source.sample_count:
                raise ValueError("Detector events require source sample offsets")
            self._schedule(event["sample"], 2, "detector_event", event)

    async def run(self) -> SessionResult:
        while self.queue:
            now = self.queue[0].sample
            # All source decisions at an offset precede submitting a new request.
            # Speech start also wins ties against output and context completion.
            while self.queue and self.queue[0].sample == now:
                self._process(heapq.heappop(self.queue))
            await self._maybe_respond(now)
        if self.active or self.active_speech or self.seen - self.consumed:
            raise RuntimeError("Offline session reached EOF with unresolved conversation state")
        rendered = max(self.source.sample_count, max((span.end_sample for span in self.spans), default=0))
        logical_end = max(rendered, max((event["sample"] for event in self.events), default=0))
        self._emit("session_complete", logical_end, rendered_samples=rendered)
        # Finishing a response can add its committed spans retrospectively. Sort
        # the final manifest while retaining stable tie ordering.
        events = tuple(sorted(self.events, key=lambda event: event["sample"]))
        return SessionResult(
            spans=tuple(sorted(self.spans, key=lambda span: span.start_sample)),
            rendered_samples=rendered,
            events=events,
            messages=tuple(copy.deepcopy(self.messages)),
            metrics={
                "recognition": copy.deepcopy(self.recognition.metrics),
                "recognition_binding_policy": "turn_start_to_endpoint_half_open_with_explicit_source_eof_point",
                "responses": self.response_metrics,
                "response_compute_ns": sum(item["compute_ns"] for item in self.response_metrics),
                "input_samples": self.source.sample_count,
                "generated_samples": sum(item["generated_samples"] for item in self.response_metrics),
                "committed_samples": sum(len(span.pcm) // 2 for span in self.spans),
                "logical_end_sample": logical_end,
                "timing_policy": "source_offsets_plus_measured_provider_availability",
            },
        )

    def _process(self, event: _Event):
        sample = event.sample
        if event.kind == "speech_start":
            index, owner, speech = event.payload
            self.active_speech.add(index)
            self.seen.add(owner)
            self._emit("user_speech_start", sample, turn_id=owner, source_start_sample=speech.start_sample)
            if self.active:
                self._finish(sample, interrupted=True)
        elif event.kind == "speech_end":
            index, owner = event.payload
            self.active_speech.discard(index)
            self._emit("user_speech_end", sample, turn_id=owner)
        elif event.kind == "endpoint":
            self.seen.add(event.payload.turn_id)
            self._emit("user_turn_endpoint", sample, turn_id=event.payload.turn_id)
        elif event.kind == "turn_ready":
            self.ready.add(event.payload.turn_id)
            self._emit("user_turn_ready", sample, turn_id=event.payload.turn_id)
        elif event.kind == "asr_final":
            segment = event.payload
            self._emit(
                "asr_final",
                sample,
                source_start_sample=segment.start_sample,
                source_end_sample=segment.end_sample,
                source_timestamp_kind="point" if segment.start_sample == segment.end_sample else "interval",
            )
        elif event.kind == "input_end":
            self._emit("input_end", sample)
        elif event.kind == "detector_event":
            self.events.append(copy.deepcopy(event.payload))
        elif self.active and event.payload[0] == self.active.response_id:
            if event.kind == "context":
                context = event.payload[1]
                if "message" in context:
                    message = copy.deepcopy(context["message"])
                    self.messages.append(message)
                    if message.get("role") == "assistant":
                        for call in message.get("tool_calls", ()):
                            if isinstance(call.get("id"), str) and call["id"]:
                                self.active.committed_tool_call_ids.add(call["id"])
                    self._emit(
                        "context_committed", sample, response_id=self.active.response_id, role=message.get("role")
                    )
                else:
                    self.messages = copy.deepcopy(context["replace_messages"])
                    self._emit(
                        "context_replaced",
                        sample,
                        response_id=self.active.response_id,
                        message_count=len(self.messages),
                    )
            elif event.kind == "bot_start":
                self._emit("bot_speech_start", sample, response_id=self.active.response_id)
            elif event.kind == "response_complete":
                self._finish(sample, interrupted=False)

    async def _maybe_respond(self, now):
        pending = self.seen - self.consumed
        if self.active or self.active_speech or not pending or not pending <= self.ready:
            return
        ordered = tuple(turn.turn_id for turn in self.turns.values() if turn.turn_id in pending)
        self.consumed.update(ordered)
        text = "\n".join(self.texts[key] for key in ordered if self.texts[key])
        if not text:
            self._emit("empty_user_turn", now, turn_ids=list(ordered))
            return
        self.messages.append({"role": "user", "content": text})
        self.response_count += 1
        response_id = f"response-{self.response_count}"
        self._emit("response_requested", now, response_id=response_id, turn_ids=list(ordered))
        trace = await self.responder.respond(copy.deepcopy(self.messages), ordered[-1])
        if trace.compute_ns < 0:
            raise ValueError("Response compute duration cannot be negative")
        planned: list[AudioSpan] = []
        play_end = now
        generated = 0
        for chunk in trace.audio:
            if len(chunk.pcm) % 2 or not 0 <= chunk.ready_offset_ns <= trace.compute_ns:
                raise ValueError("Generated audio must be PCM16 and become ready during response computation")
            if not chunk.pcm:
                continue
            play_start = max(now + ns_to_samples(chunk.ready_offset_ns), play_end)
            span = AudioSpan(play_start, chunk.pcm, response_id)
            planned.append(span)
            play_end = span.end_sample
            generated += len(chunk.pcm) // 2
        previous_end = 0
        for word in trace.words:
            if not 0 <= word.start_sample < word.end_sample <= generated or word.start_sample < previous_end:
                raise ValueError("Spoken word offsets must be ordered and inside generated audio")
            previous_end = word.end_sample
        completion = max(play_end, now + ns_to_samples(trace.compute_ns))
        self.active = _Response(response_id, ordered, now, trace, tuple(planned), completion, generated)
        if planned:
            self._schedule(planned[0].start_sample, 4, "bot_start", (response_id,))
        for context in trace.metrics.get("context_events", ()):
            offset = context["ready_offset_ns"]
            message = context.get("message")
            replacement = context.get("replace_messages")
            valid_append = "message" in context and isinstance(message, dict)
            valid_replace = (
                "replace_messages" in context
                and isinstance(replacement, list)
                and all(isinstance(item, dict) for item in replacement)
            )
            if (
                not isinstance(offset, int)
                or isinstance(offset, bool)
                or not 0 <= offset <= trace.compute_ns
                or ("message" in context) == ("replace_messages" in context)
                or not (valid_append or valid_replace)
            ):
                raise ValueError("Context events require one message operation and a causal compute offset")
            self._schedule(now + ns_to_samples(offset), 4, "context", (response_id, copy.deepcopy(context)))
        self._schedule(completion, 5, "response_complete", (response_id,))

    def _finish(self, sample, interrupted):
        response = self.active
        assert response is not None
        omitted_tool_calls = self._discard_unanswered_tool_calls(response) if interrupted else []
        committed = []
        for span in response.spans:
            count = min(len(span.pcm) // 2, max(0, sample - span.start_sample))
            if count:
                played = AudioSpan(span.start_sample, span.pcm[: count * 2], span.response_id)
                committed.append(played)
                self._emit(
                    "bot_audio", played.start_sample, response_id=played.response_id, end_sample=played.end_sample
                )
        self.spans.extend(committed)
        played_samples = sum(len(span.pcm) // 2 for span in committed)
        complete_audio = played_samples == response.generated_samples and played_samples > 0
        if not response.generated_samples:
            content = ""
            history_policy = "unspoken_text_omitted"
        elif not interrupted or complete_audio:
            content = response.trace.text
            history_policy = "complete_text"
        elif response.trace.words:
            content = " ".join(word.text for word in response.trace.words if word.end_sample <= played_samples)
            history_policy = "aligned_played_prefix"
        else:
            content = ""
            history_policy = "unaligned_interrupted_text_omitted"
        if content:
            self.messages.append({"role": "assistant", "content": content})
        self._emit(
            "response_interrupted" if interrupted else "response_complete",
            sample,
            response_id=response.response_id,
            turn_ids=list(response.turn_ids),
            generated_samples=response.generated_samples,
            committed_samples=played_samples,
            history_policy=history_policy,
            omitted_tool_call_ids=omitted_tool_calls,
        )
        if committed:
            self._emit("bot_speech_end", committed[-1].end_sample, response_id=response.response_id)
        self.response_metrics.append(
            {
                "response_id": response.response_id,
                "turn_ids": list(response.turn_ids),
                "request_sample": response.request_sample,
                "end_sample": sample,
                "compute_ns": response.trace.compute_ns,
                "generated_samples": response.generated_samples,
                "committed_samples": played_samples,
                "interrupted": interrupted,
                "provider": copy.deepcopy(
                    {key: value for key, value in response.trace.metrics.items() if key != "context_events"}
                ),
            }
        )
        self.active = None

    def _discard_unanswered_tool_calls(self, response: _Response) -> list[str]:
        """Remove interrupted requests without inventing their future results.

        Pair actual chronological call/result transactions rather than deleting
        globally by ID: a provider can reuse an ID present in prior, completed
        history. Locations are recomputed after any summary replacement.
        """
        pending: dict[str, list[tuple[int, int]]] = {}
        for message_index, message in enumerate(self.messages):
            if message.get("role") == "assistant":
                for call_index, call in enumerate(message.get("tool_calls", ())):
                    call_id = call.get("id")
                    if isinstance(call_id, str):
                        pending.setdefault(call_id, []).append((message_index, call_index))
            elif message.get("role") == "tool":
                locations = pending.get(message.get("tool_call_id"))
                if locations:
                    locations.pop()
        removals: dict[int, set[int]] = {}
        omitted = set()
        for call_id in response.committed_tool_call_ids:
            for message_index, call_index in pending.get(call_id, ()):
                removals.setdefault(message_index, set()).add(call_index)
                omitted.add(call_id)
        retained = []
        for message_index, message in enumerate(self.messages):
            if message_index in removals:
                message = copy.deepcopy(message)
                message["tool_calls"] = [
                    call for index, call in enumerate(message["tool_calls"]) if index not in removals[message_index]
                ]
                if not message["tool_calls"]:
                    del message["tool_calls"]
                    if not message.get("content"):
                        continue
            retained.append(message)
        self.messages = retained
        return sorted(omitted)
