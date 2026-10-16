# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Continuous ASR and overlapping LLM/TTS adapters with explicit stream EOF.

Monotonic time is used only to measure observed provider execution. The ASR
projection records residual delay after the source dependency was supplied in
the bulk request; it includes transport and provider queueing, not pure GPU time.
"""

import asyncio
import bisect
import copy
import json
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterable
from types import SimpleNamespace
from typing import Any

from .models import (
    RATE,
    AudioSource,
    GeneratedAudio,
    RecognitionSegment,
    RecognitionTrace,
    ResponseTrace,
    SpokenWord,
    ns_to_samples,
)


class ContinuousASR:
    """Drain one finite continuous streaming call over the complete source."""

    def __init__(self, stream: Callable[[Iterable[bytes]], Iterable[Any]], *, clock=time.monotonic_ns, cancel=None):
        """Inject streaming transport and an execution-measurement clock."""
        self._stream = stream
        self._clock = clock
        self._cancel = cancel

    async def recognize(self, source: AudioSource) -> RecognitionTrace:
        """Finish every queued source chunk and consume all final responses."""
        try:
            return await asyncio.to_thread(self._recognize, source)
        except asyncio.CancelledError:
            self.cancel()
            raise

    def cancel(self):
        """Cancel an active provider call, independent of source completion."""
        if self._cancel:
            self._cancel()

    def _recognize(self, source: AudioSource) -> RecognitionTrace:
        started = self._clock()
        positions: list[int] = []
        supplied: list[int] = []
        supplied_lock = threading.Lock()
        segments = []
        final_metrics = []
        final_coverage = 0
        previous_available = 0
        first_response_ns = None
        first_transcript_ns = None
        first_final_transcript_ns = None
        input_exhausted_ns = None

        def audio():
            nonlocal input_exhausted_ns
            for start in range(0, source.sample_count, 512):
                end = min(start + 512, source.sample_count)
                with supplied_lock:
                    positions.append(end)
                    supplied.append(self._clock())
                yield source.pcm[start * 2 : end * 2]
            input_exhausted_ns = self._clock() - started

        for response in self._stream(audio()):
            arrival = self._clock()
            if first_response_ns is None:
                first_response_ns = arrival - started
            for result in response.results:
                alternatives = result.alternatives
                alternative = alternatives[0] if alternatives else None
                text = alternative.transcript.strip() if alternative else ""
                if text and first_transcript_ns is None:
                    first_transcript_ns = arrival - started
                if not result.is_final:
                    continue
                if text and first_final_transcript_ns is None:
                    first_final_transcript_ns = arrival - started
                coverage = round(float(result.audio_processed) * RATE)
                if coverage < final_coverage or coverage > source.sample_count + 512:
                    raise ValueError("ASR returned invalid or nonmonotonic source coverage")
                coverage = min(coverage, source.sample_count)
                words = tuple(
                    (word.word, round(word.start_time * RATE / 1000), round(word.end_time * RATE / 1000))
                    for word in (alternative.words if alternative else ())
                )
                for _, start, end in words:
                    if start < 0 or end < start or end > source.sample_count:
                        raise ValueError("ASR word offsets are outside the source")
                dependency = max(coverage, max((end for _, _, end in words), default=0))
                if text and dependency == 0:
                    raise ValueError("ASR final text lacks audio_processed or word offsets")
                with supplied_lock:
                    index = bisect.bisect_left(positions, dependency)
                    supplied_time = supplied[index] if index < len(supplied) else None
                    supplying_end = positions[index] if index < len(positions) else None
                final_metric = {
                    "final_index": len(final_metrics),
                    "segment_index": len(segments) if text else None,
                    "has_transcript": bool(text),
                    "coverage_sample": coverage,
                    "dependency_sample": dependency,
                    "supplied_chunk_end_sample": supplying_end,
                    "dependency_supplied_ns": supplied_time - started if supplied_time is not None else None,
                    "arrival_ns": arrival - started,
                    "observed_residual_ns": None,
                    "projected_available_sample": None,
                }
                final_metrics.append(final_metric)
                if supplied_time is None:
                    if text:
                        raise ValueError("ASR final depends on source audio not supplied to its stream")
                    final_coverage = coverage
                    continue
                residual_ns = max(0, arrival - supplied_time)
                available = max(previous_available, dependency + ns_to_samples(residual_ns))
                final_metric["observed_residual_ns"] = residual_ns
                final_metric["projected_available_sample"] = available
                if text:
                    start = min((start for _, start, _ in words), default=final_coverage)
                    end = max((end for _, _, end in words), default=coverage)
                    segments.append(RecognitionSegment(text, start, end, available, words))
                final_coverage = coverage
                previous_available = available
        if source.sample_count and (not positions or positions[-1] != source.sample_count):
            raise ValueError("ASR request terminated before consuming all source audio")
        stream_end_ns = self._clock() - started
        return RecognitionTrace(
            tuple(segments),
            {
                "asr_mode": "continuous_stream",
                "asr_streams": 1,
                "asr_source_samples": source.sample_count,
                "asr_final_coverage_sample": final_coverage,
                "asr_observed_wall_ns": stream_end_ns,
                "asr_first_response_ns": first_response_ns,
                "asr_first_transcript_ns": first_transcript_ns,
                "asr_first_final_transcript_ns": first_final_transcript_ns,
                "asr_first_audio_supplied_ns": supplied[0] - started if supplied else None,
                "asr_last_audio_supplied_ns": supplied[-1] - started if supplied else None,
                "asr_input_exhausted_ns": input_exhausted_ns,
                "asr_stream_end_ns": stream_end_ns,
                "asr_finals": final_metrics,
                "asr_availability_policy": "source_dependency_plus_observed_bulk_residual",
                "asr_timing_note": "Observed transport/queue/processing delay; not backend-only GPU or live latency.",
            },
        )


class StreamingResponder:
    """Feed sentence text to one TTS stream while the LLM is still producing it."""

    def __init__(
        self, text_stream, synthesis_stream, *, aggregator_factory=None, text_filter=None, clock=time.monotonic_ns
    ):
        """Inject text/audio production without binding it to playback."""
        self._text_stream = text_stream
        self._synthesis_stream = synthesis_stream
        self._aggregator_factory = aggregator_factory
        self._filter = text_filter
        self._clock = clock

    async def respond(self, messages: list[dict[str, Any]], turn_id: str) -> ResponseTrace:
        """Capture generation without committing text or waiting for playback."""
        if self._aggregator_factory is None:
            from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

            aggregator = SimpleTextAggregator()
        else:
            aggregator = self._aggregator_factory()
        text_queue: queue.Queue[str | None] = queue.Queue()
        cancelled = threading.Event()
        started = self._clock()
        chunks: list[GeneratedAudio] = []
        word_metadata: list[SpokenWord] = []
        texts: list[str] = []
        context_events: list[dict] = []
        llm_first_ns = None
        llm_end_ns = None
        submitted_text = 0
        consumed_text = 0
        input_drained = False
        tts_first_queued_ns = None
        tts_first_input_ns = None
        tts_stream_end_ns = None

        def requests():
            nonlocal consumed_text, input_drained, tts_first_input_ns
            while not cancelled.is_set():
                text = text_queue.get()
                if text is None or cancelled.is_set():
                    input_drained = text is None and not cancelled.is_set()
                    return
                consumed_text += 1
                if tts_first_input_ns is None:
                    tts_first_input_ns = self._clock() - started
                yield text

        def synthesize():
            nonlocal tts_stream_end_ns
            for response in self._synthesis_stream(requests()):
                if cancelled.is_set():
                    return
                pcm = response if isinstance(response, bytes) else response.audio
                metadata = getattr(response, "meta", None)
                for word in getattr(metadata, "words", ()):
                    word_metadata.append(
                        SpokenWord(word.word, round(word.start_time * RATE / 1000), round(word.end_time * RATE / 1000))
                    )
                if len(pcm) % 2:
                    raise ValueError("TTS emitted unaligned PCM16")
                if pcm:
                    chunks.append(GeneratedAudio(bytes(pcm), self._clock() - started))
            tts_stream_end_ns = self._clock() - started

        async def submit(text):
            nonlocal submitted_text, tts_first_queued_ns
            if self._filter:
                text = await self._filter.filter(text)
            if any(char.isalnum() for char in text):
                submitted_text += 1
                if tts_first_queued_ns is None:
                    tts_first_queued_ns = self._clock() - started
                text_queue.put(text)

        async def generate_text():
            nonlocal llm_first_ns, llm_end_ns
            try:
                async for event in self._text_stream.stream(copy.deepcopy(messages)):
                    if isinstance(event, dict):
                        context_events.append(
                            {
                                "ready_offset_ns": self._clock() - started,
                                **(event if "replace_messages" in event else {"message": event}),
                            }
                        )
                        continue
                    if llm_first_ns is None:
                        llm_first_ns = self._clock() - started
                    texts.append(event)
                    async for aggregation in aggregator.aggregate(event):
                        await submit(aggregation.text)
                tail = await aggregator.flush()
                if tail:
                    await submit(tail.text)
                llm_end_ns = self._clock() - started
            finally:
                text_queue.put(None)

        try:
            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(generate_text())
                tasks.create_task(asyncio.to_thread(synthesize))
        finally:
            cancelled.set()
            text_queue.put(None)
            if hasattr(self._synthesis_stream, "cancel"):
                self._synthesis_stream.cancel()
        elapsed = self._clock() - started
        if consumed_text != submitted_text or not input_drained:
            raise ValueError("TTS stream ended before all submitted text and EOF were consumed")
        generated_samples = sum(len(chunk.pcm) // 2 for chunk in chunks)
        # A provider may repeat cumulative metadata. Preserve first-seen order:
        # sorting backwards/reset offsets could attribute future words to audio
        # that has already played, concealing an invalid provider alignment.
        words = tuple(dict.fromkeys(word_metadata))
        previous_end = 0
        alignment = "provider_reported" if words else "unavailable"
        for word in words:
            if not previous_end <= word.start_sample < word.end_sample <= generated_samples:
                words = ()
                alignment = "invalid_provider_offsets_not_used"
                break
            previous_end = word.end_sample
        first_audio_ns = chunks[0].ready_offset_ns if chunks else None
        metrics = {
            "turn_id": turn_id,
            "llm_first_text_ns": llm_first_ns,
            "llm_complete_ns": llm_end_ns,
            "first_audio_ready_ns": first_audio_ns,
            "response_observed_wall_ns": elapsed,
            "tts_first_text_queued_ns": tts_first_queued_ns,
            "tts_first_input_ns": tts_first_input_ns,
            "tts_first_audio_latency_ns": (
                first_audio_ns - tts_first_input_ns
                if first_audio_ns is not None and tts_first_input_ns is not None
                else None
            ),
            "tts_last_audio_ready_ns": chunks[-1].ready_offset_ns if chunks else None,
            "tts_stream_end_ns": tts_stream_end_ns,
            "tts_stream_elapsed_ns": (
                tts_stream_end_ns - tts_first_input_ns
                if tts_stream_end_ns is not None and tts_first_input_ns is not None
                else None
            ),
            "context_events": context_events,
            "spoken_alignment": alignment,
            "timing_basis": "overlapping_provider_wall_trace",
            "tts_timing_note": (
                "Input starts when the synthesis adapter consumes its first filtered text. "
                "Latency includes RPC/transport/provider delay and excludes earlier LLM aggregation; not GPU-only time."
            ),
        }
        provider_metrics = getattr(self._text_stream, "last_run_metrics", None)
        if provider_metrics is not None:
            metrics["llm_provider_metrics"] = copy.deepcopy(provider_metrics)
        return ResponseTrace(
            "".join(texts),
            tuple(chunks),
            elapsed,
            words=words,
            metrics=metrics,
        )


class OpenAITextStream:
    """NIM-compatible streaming text and generic read-only tool handlers."""

    def __init__(
        self,
        client,
        *,
        model: str,
        extra: dict,
        tools: list[dict],
        handlers: dict,
        tool_choice: str = "auto",
        history_preparer=None,
        clock=time.monotonic_ns,
    ):
        """Use the selected NIM model, tool catalog, and optional history policy."""
        self._client = client
        self._model = model
        self._extra = extra
        self._tools = tools
        self._handlers = handlers
        self._tool_choice = tool_choice
        self._history_preparer = history_preparer
        self._clock = clock
        self.last_run_metrics = None

    async def stream(self, messages: list[dict]) -> AsyncIterator[str | dict]:
        """Continue tool rounds inside a private conversation snapshot."""
        started = self._clock()
        self.last_run_metrics = {
            "rounds": [],
            "llm_observed_wall_ns": None,
            "timing_note": (
                "Offsets start at text-stream invocation; RPC latency includes transport, queueing, "
                "provider processing and stream consumption, not GPU-only time."
            ),
        }
        try:
            async for event in self._stream(messages, started):
                yield event
        finally:
            self.last_run_metrics["llm_observed_wall_ns"] = self._clock() - started

    async def _stream(self, messages: list[dict], started: int) -> AsyncIterator[str | dict]:
        history = copy.deepcopy(messages)
        if self._history_preparer:
            prepared = await self._history_preparer(history)
            if prepared != history:
                history = prepared
                yield {"replace_messages": copy.deepcopy(history)}
        for round_index in range(8):
            arguments = {"model": self._model, "messages": history, "stream": True, **self._extra}
            if self._tools:
                arguments["tools"] = self._tools
                arguments["tool_choice"] = self._tool_choice
            dispatched = self._clock()
            round_metrics = {
                "round_index": round_index,
                "rpc_dispatched_ns": dispatched - started,
                "first_response_ns": None,
                "first_text_ns": None,
                "stream_end_ns": None,
                "first_response_latency_ns": None,
                "first_text_latency_ns": None,
                "stream_elapsed_ns": None,
            }
            self.last_run_metrics["rounds"].append(round_metrics)
            stream = await self._client.chat.completions.create(**arguments)
            text = []
            calls: dict[int, dict] = {}
            async with stream:
                async for chunk in stream:
                    arrival = self._clock()
                    if round_metrics["first_response_ns"] is None:
                        round_metrics["first_response_ns"] = arrival - started
                        round_metrics["first_response_latency_ns"] = arrival - dispatched
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        if round_metrics["first_text_ns"] is None:
                            round_metrics["first_text_ns"] = arrival - started
                            round_metrics["first_text_latency_ns"] = arrival - dispatched
                        text.append(delta.content)
                        yield delta.content
                    for part in delta.tool_calls or ():
                        call = calls.setdefault(
                            part.index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                        )
                        if part.id:
                            call["id"] = part.id
                        if part.function:
                            call["function"]["name"] += part.function.name or ""
                            call["function"]["arguments"] += part.function.arguments or ""
            stream_end = self._clock()
            round_metrics["stream_end_ns"] = stream_end - started
            round_metrics["stream_elapsed_ns"] = stream_end - dispatched
            if not calls:
                return
            assistant = {
                "role": "assistant",
                "content": "".join(text) or None,
                "tool_calls": [calls[index] for index in sorted(calls)],
            }
            history.append(assistant)
            # Spoken text is committed exclusively by scheduled playback.
            yield {**assistant, "content": None}
            for call in assistant["tool_calls"]:
                name = call["function"]["name"]
                if name not in self._handlers:
                    raise ValueError(f"LLM requested unavailable tool {name}")
                result_box = []

                async def result_callback(value, _results=result_box, **_):
                    _results.append(value)

                params = SimpleNamespace(
                    arguments=json.loads(call["function"]["arguments"]), result_callback=result_callback
                )
                await self._handlers[name](params)
                if len(result_box) != 1:
                    raise ValueError(f"Tool {name} must produce exactly one result")
                message = {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result_box[0])}
                history.append(message)
                yield message
        raise ValueError("Offline tool-call round limit exceeded")


def create_riva_recognizer(settings: dict) -> ContinuousASR:
    """Create one continuous Riva streaming call with generic ASR endpointing."""
    import riva.client
    from riva.client.proto import riva_asr_pb2

    auth = _riva_auth(settings, "asr")
    service = riva.client.ASRService(auth)
    active = []
    config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=RATE,
            language_code=settings.get("asr_language_code") or "en-US",
            model=settings.get("asr_model") or "",
            max_alternatives=1,
            profanity_filter=False,
            enable_automatic_punctuation=settings.get("asr_automatic_punctuation", True),
            verbatim_transcripts=True,
            audio_channel_count=1,
            enable_word_time_offsets=True,
        ),
        interim_results=True,
    )
    riva.client.add_endpoint_parameters_to_config(config, -1, -1.0, 400, -1, -1.0, -1.0)

    def stream(audio):
        def requests():
            yield riva_asr_pb2.StreamingRecognizeRequest(streaming_config=config)
            for pcm in audio:
                yield riva_asr_pb2.StreamingRecognizeRequest(audio_content=pcm)

        call = service.stub.StreamingRecognize(requests(), metadata=auth.get_auth_metadata(), timeout=600)
        active.append(call)
        try:
            yield from call
        finally:
            call.cancel()
            auth.channel.close()

    def cancel():
        for call in active:
            call.cancel()
        auth.channel.close()

    return ContinuousASR(stream, cancel=cancel)


def _canonical_synthesis_audio(responses, source_rate: int):
    """Convert native synthesis PCM without chunk-boundary drift or idle resets."""
    if source_rate == RATE:
        yield from responses
        return

    import numpy as np
    import soxr

    stream = soxr.ResampleStream(source_rate, RATE, 1, dtype="int16", quality="VHQ")
    for response in responses:
        if len(response.audio) % 2:
            raise ValueError("TTS emitted unaligned native PCM16")
        audio = stream.resample_chunk(np.frombuffer(response.audio, dtype=np.int16)).tobytes()
        # Word times remain milliseconds from the same response origin.
        yield SimpleNamespace(audio=audio, meta=getattr(response, "meta", None))
    tail = stream.resample_chunk(np.empty(0, dtype=np.int16), last=True).tobytes()
    if tail:
        yield SimpleNamespace(audio=tail, meta=None)


def create_riva_synthesis(settings: dict):
    """Create an explicitly drained Magpie SynthesizeOnline call per response."""
    import riva.client
    from riva.client.proto import riva_tts_pb2

    dictionary = settings.get("custom_dictionary") or {}
    word_options = (
        {"enable_word_time_offsets": True}
        if "enable_word_time_offsets" in riva_tts_pb2.SynthesizeSpeechRequest.DESCRIPTOR.fields_by_name
        else {}
    )
    active = []
    lock = threading.Lock()
    epoch = [0]

    def synthesize(texts):
        with lock:
            request_epoch = epoch[0]
        auth = _riva_auth(settings, "tts")
        service = riva.client.SpeechSynthesisService(auth)

        def requests():
            for text in texts:
                # Match the generic NVIDIA service's <=200 character submissions.
                from pipecat.services.nvidia.tts import NvidiaTTSService

                for part in NvidiaTTSService._split_text_into_chunks(text):
                    yield riva_tts_pb2.SynthesizeSpeechRequest(
                        text=part,
                        language_code=settings.get("tts_language_code") or "en-US",
                        sample_rate_hz=settings.get("tts_sample_rate", RATE),
                        encoding=riva.client.AudioEncoding.LINEAR_PCM,
                        voice_name=settings.get("tts_voice_id") or "Magpie-Multilingual.EN-US.Aria",
                        custom_dictionary=", ".join(f"{key}  {value}" for key, value in dictionary.items()),
                        **word_options,
                    )

        call = service.stub.SynthesizeOnline(requests(), metadata=auth.get_auth_metadata(), timeout=600)
        with lock:
            active.append(call)
            if request_epoch != epoch[0]:
                call.cancel()
        try:
            yield from _canonical_synthesis_audio(call, settings.get("tts_sample_rate", RATE))
        finally:
            call.cancel()
            auth.channel.close()
            with lock:
                active.remove(call)

    def cancel():
        with lock:
            epoch[0] += 1
            for call in active:
                call.cancel()

    synthesize.cancel = cancel
    return synthesize


def _riva_auth(settings, prefix):
    import riva.client

    server = settings[f"{prefix}_server"]
    metadata = []
    function_id = settings.get(f"{prefix}_function_id")
    if function_id:
        metadata.append(("function-id", function_id))
    if settings.get("api_key"):
        metadata.append(("authorization", f"Bearer {settings['api_key']}"))
    return riva.client.Auth(None, "nvcf.nvidia.com" in server, server, metadata)
