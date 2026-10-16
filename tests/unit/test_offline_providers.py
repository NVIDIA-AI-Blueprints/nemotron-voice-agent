# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from examples.generic.offline.models import RATE, AudioSource
from examples.generic.offline.providers import (
    ContinuousASR,
    OpenAITextStream,
    StreamingResponder,
    _canonical_synthesis_audio,
    create_riva_recognizer,
    create_riva_synthesis,
)


def final(text, processed, words=()):
    return SimpleNamespace(
        results=[
            SimpleNamespace(
                is_final=True,
                audio_processed=processed,
                alternatives=[
                    SimpleNamespace(
                        transcript=text,
                        words=[
                            SimpleNamespace(word=text, start_time=start, end_time=end) for text, start, end in words
                        ],
                    )
                ],
            )
        ]
    )


class ManualClock:
    def __init__(self, value=0):
        self.value = value
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.value

    def set(self, value):
        with self.lock:
            self.value = value


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_tts_conversion_preserves_duration_tail_and_packetization(self):
        import numpy as np

        native = (np.sin(np.arange(22050) * 2 * np.pi * 440 / 22050) * 10000).astype(np.int16).tobytes()
        marker = SimpleNamespace(words=[SimpleNamespace(word="hello", start_time=0, end_time=1000)])

        def convert(chunk_bytes):
            responses = [
                SimpleNamespace(audio=native[offset : offset + chunk_bytes], meta=None)
                for offset in range(0, len(native), chunk_bytes)
            ]
            responses.append(SimpleNamespace(audio=b"", meta=marker))
            return list(_canonical_synthesis_audio(responses, 22050))

        small, whole = convert(514), convert(len(native))
        small_pcm = b"".join(response.audio for response in small)
        self.assertEqual(len(small_pcm), RATE * 2)
        # SoXR dithers int16 output; packetization must preserve positions and
        # duration, while sample values can differ within quantization noise.
        np.testing.assert_allclose(
            np.frombuffer(small_pcm, dtype=np.int16).astype(np.int32),
            np.frombuffer(b"".join(response.audio for response in whole), dtype=np.int16).astype(np.int32),
            atol=2,
            rtol=0,
        )
        self.assertEqual(sum(response.meta is marker for response in small), 1)
        self.assertTrue(small[-1].audio, "EOF must release samples retained in the resampling filter")

    async def test_native_tts_conversion_rejects_partial_pcm_samples(self):
        with self.assertRaisesRegex(ValueError, "unaligned native"):
            list(_canonical_synthesis_audio([SimpleNamespace(audio=b"x", meta=None)], 22050))

    async def test_asr_observed_events_separate_protocol_interim_final_and_source_dependency(self):
        base = 1_000_000_000
        clock = ManualClock(base)
        ms = 1_000_000

        def at(offset):
            clock.set(base + offset * ms)

        def stream(audio):
            audio = iter(audio)
            at(10)
            next(audio)
            at(20)
            yield SimpleNamespace(results=[])
            at(30)
            interim = final("par", 0.032)
            interim.results[0].is_final = False
            yield interim
            at(40)
            yield final("part", 0.032, (("part", 0, 16),))
            at(50)
            next(audio)
            at(60)
            next(audio)
            at(70)
            self.assertIsNone(next(audio, None))
            at(80)
            yield final("second", 0.096)
            at(90)

        trace = await ContinuousASR(stream, clock=clock).recognize(AudioSource(bytes(1536 * 2)))
        metrics = trace.metrics
        expected = {
            "asr_first_response_ns": 20,
            "asr_first_transcript_ns": 30,
            "asr_first_final_transcript_ns": 40,
            "asr_first_audio_supplied_ns": 10,
            "asr_last_audio_supplied_ns": 60,
            "asr_input_exhausted_ns": 70,
            "asr_stream_end_ns": 90,
            "asr_observed_wall_ns": 90,
        }
        for name, offset in expected.items():
            self.assertEqual(metrics[name], offset * ms, name)
        self.assertEqual(
            metrics["asr_finals"],
            [
                {
                    "final_index": 0,
                    "segment_index": 0,
                    "has_transcript": True,
                    "coverage_sample": 512,
                    "dependency_sample": 512,
                    "supplied_chunk_end_sample": 512,
                    "dependency_supplied_ns": 10 * ms,
                    "arrival_ns": 40 * ms,
                    "observed_residual_ns": 30 * ms,
                    "projected_available_sample": 992,
                },
                {
                    "final_index": 1,
                    "segment_index": 1,
                    "has_transcript": True,
                    "coverage_sample": 1536,
                    "dependency_sample": 1536,
                    "supplied_chunk_end_sample": 1536,
                    "dependency_supplied_ns": 60 * ms,
                    "arrival_ns": 80 * ms,
                    "observed_residual_ns": 20 * ms,
                    "projected_available_sample": 1856,
                },
            ],
        )

    async def test_asr_empty_protocol_response_does_not_fabricate_transcript_latency(self):
        clock = ManualClock()

        def stream(audio):
            list(audio)
            clock.set(20)
            yield final("  ", 0)
            clock.set(30)

        trace = await ContinuousASR(stream, clock=clock).recognize(AudioSource(b""))
        self.assertEqual(trace.metrics["asr_first_response_ns"], 20)
        for key in (
            "asr_first_transcript_ns",
            "asr_first_final_transcript_ns",
            "asr_first_audio_supplied_ns",
            "asr_last_audio_supplied_ns",
        ):
            self.assertIsNone(trace.metrics[key], key)
        self.assertEqual(trace.metrics["asr_input_exhausted_ns"], 0)
        self.assertIsNone(trace.metrics["asr_finals"][0]["observed_residual_ns"])
        self.assertIsNone(trace.metrics["asr_finals"][0]["projected_available_sample"])

    async def test_asr_no_response_keeps_first_event_metrics_null(self):
        def stream(audio):
            list(audio)
            return iter(())

        trace = await ContinuousASR(stream).recognize(AudioSource(b""))
        self.assertIsNone(trace.metrics["asr_first_response_ns"])
        self.assertIsNone(trace.metrics["asr_first_transcript_ns"])
        self.assertEqual(trace.metrics["asr_finals"], [])

    async def test_tts_input_latency_excludes_sentence_aggregation_and_eof_wait(self):
        clock = ManualClock()
        audio_produced = threading.Event()

        class Text:
            async def stream(self, messages):
                clock.set(10)
                yield "First"
                clock.set(20)
                yield " sentence."
                while not audio_produced.is_set():
                    await asyncio.sleep(0)
                clock.set(60)

        class Aggregator:
            async def aggregate(self, text):
                if text.endswith("."):
                    yield SimpleNamespace(text="First sentence.")

            async def flush(self):
                return None

        class Filter:
            async def filter(self, text):
                return text.replace("First", "Filtered")

        def synthesis(texts):
            self.assertEqual(next(texts), "Filtered sentence.")
            clock.set(40)
            yield bytes(16)
            audio_produced.set()
            self.assertEqual(list(texts), [])
            clock.set(80)
            yield b""

        trace = await StreamingResponder(
            Text(), synthesis, aggregator_factory=Aggregator, text_filter=Filter(), clock=clock
        ).respond([], "timed")
        expected = {
            "llm_first_text_ns": 10,
            "tts_first_text_queued_ns": 20,
            "tts_first_input_ns": 20,
            "first_audio_ready_ns": 40,
            "tts_first_audio_latency_ns": 20,
            "tts_last_audio_ready_ns": 40,
            "tts_stream_end_ns": 80,
            "tts_stream_elapsed_ns": 60,
            "response_observed_wall_ns": 80,
        }
        for name, value in expected.items():
            self.assertEqual(trace.metrics[name], value, name)
        self.assertEqual(trace.audio[0].ready_offset_ns, 40)

    async def test_tts_no_usable_text_or_audio_keeps_latency_null(self):
        class Text:
            async def stream(self, messages):
                yield "..."

        class Aggregator:
            async def aggregate(self, text):
                yield SimpleNamespace(text=text)

            async def flush(self):
                return None

        def synthesis(texts):
            self.assertEqual(list(texts), [])
            yield b""

        trace = await StreamingResponder(Text(), synthesis, aggregator_factory=Aggregator).respond([], "empty")
        for key in (
            "tts_first_text_queued_ns",
            "tts_first_input_ns",
            "first_audio_ready_ns",
            "tts_first_audio_latency_ns",
            "tts_last_audio_ready_ns",
            "tts_stream_elapsed_ns",
        ):
            self.assertIsNone(trace.metrics[key], key)
        self.assertIsNotNone(trace.metrics["tts_stream_end_ns"])

    async def test_llm_rpc_metrics_separate_empty_response_and_first_nonempty_text(self):
        clock = ManualClock()

        class Stream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                clock.set(50)

            def __aiter__(self):
                async def values():
                    clock.set(20)
                    yield SimpleNamespace(choices=[])
                    clock.set(30)
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="", tool_calls=None))])
                    clock.set(40)
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="answer", tool_calls=None))]
                    )

                return values()

        async def create(**kwargs):
            return Stream()

        async def prepare(history):
            clock.set(10)
            return history

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        text_stream = OpenAITextStream(
            client, model="test", extra={}, tools=[], handlers={}, history_preparer=prepare, clock=clock
        )
        self.assertEqual([event async for event in text_stream.stream([])], ["answer"])
        self.assertEqual(text_stream.last_run_metrics["llm_observed_wall_ns"], 50)
        self.assertEqual(
            text_stream.last_run_metrics["rounds"],
            [
                {
                    "round_index": 0,
                    "rpc_dispatched_ns": 10,
                    "first_response_ns": 20,
                    "first_text_ns": 40,
                    "stream_end_ns": 50,
                    "first_response_latency_ns": 10,
                    "first_text_latency_ns": 30,
                    "stream_elapsed_ns": 40,
                }
            ],
        )

    async def test_llm_empty_completion_keeps_text_latency_null(self):
        requests_seen = []

        class Stream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def __aiter__(self):
                async def values():
                    if False:
                        yield

                return values()

        async def create(**kwargs):
            requests_seen.append(kwargs)
            return Stream()

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        text_stream = OpenAITextStream(
            client, model="test", extra={}, tools=[{"type": "function"}], handlers={}, tool_choice="none"
        )
        self.assertEqual([event async for event in text_stream.stream([])], [])
        self.assertEqual(requests_seen[0]["tool_choice"], "none")
        round_metrics = text_stream.last_run_metrics["rounds"][0]
        self.assertIsNone(round_metrics["first_response_ns"])
        self.assertIsNone(round_metrics["first_text_ns"])
        self.assertIsNone(round_metrics["first_text_latency_ns"])
        self.assertIsNotNone(round_metrics["stream_elapsed_ns"])

    async def test_responder_copies_optional_provider_metrics_without_context_changes(self):
        class Text:
            last_run_metrics = {"rounds": [{"first_text_latency_ns": 12}]}

            async def stream(self, messages):
                yield "hello"

        class Aggregator:
            async def aggregate(self, text):
                yield SimpleNamespace(text=text)

            async def flush(self):
                return None

        def synthesis(texts):
            for _ in texts:
                yield bytes(2)

        text_stream = Text()
        trace = await StreamingResponder(text_stream, synthesis, aggregator_factory=Aggregator).respond([], "metrics")
        self.assertEqual(trace.metrics["llm_provider_metrics"], text_stream.last_run_metrics)
        self.assertIsNot(trace.metrics["llm_provider_metrics"], text_stream.last_run_metrics)
        self.assertEqual(trace.metrics["context_events"], [])

    async def test_real_tts_request_construction_without_optional_dictionary(self):
        from riva.client.proto import riva_tts_pb2

        requests_seen = []

        class Call:
            def __iter__(self):
                yield riva_tts_pb2.SynthesizeSpeechResponse(audio=b"\x01\x00")

            def cancel(self):
                pass

        def online(requests, **kwargs):
            requests_seen.extend(requests)
            return Call()

        auth = SimpleNamespace(channel=SimpleNamespace(close=lambda: None), get_auth_metadata=lambda: [])
        service = SimpleNamespace(stub=SimpleNamespace(SynthesizeOnline=online))
        with (
            patch("examples.generic.offline.providers._riva_auth", return_value=auth),
            patch("riva.client.SpeechSynthesisService", return_value=service),
        ):
            synthesis = create_riva_synthesis({"custom_dictionary": None, "tts_language_code": "fr-FR"})
            responses = list(synthesis(["hello"]))
        self.assertEqual(len(responses), 1)
        self.assertEqual(requests_seen[0].text, "hello")
        self.assertEqual(requests_seen[0].custom_dictionary, "")
        self.assertEqual(requests_seen[0].sample_rate_hz, RATE)
        self.assertEqual(requests_seen[0].language_code, "fr-FR")

    async def test_real_asr_request_preserves_model_and_punctuation_without_synthetic_audio(self):
        requests_seen = []

        class Call:
            def __iter__(self):
                return iter(())

            def cancel(self):
                pass

        def streaming(requests, **kwargs):
            requests_seen.extend(requests)
            return Call()

        auth = SimpleNamespace(channel=SimpleNamespace(close=lambda: None), get_auth_metadata=lambda: [])
        service = SimpleNamespace(stub=SimpleNamespace(StreamingRecognize=streaming))
        with (
            patch("examples.generic.offline.providers._riva_auth", return_value=auth),
            patch("riva.client.ASRService", return_value=service),
        ):
            recognizer = create_riva_recognizer({"asr_model": "selected-model", "asr_automatic_punctuation": False})
            await recognizer.recognize(AudioSource(bytes(1026)))
        config = requests_seen[0].streaming_config.config
        self.assertEqual(config.model, "selected-model")
        self.assertFalse(config.enable_automatic_punctuation)
        from pipecat.services.nvidia.stt import NvidiaSTTService

        native = NvidiaSTTService(use_ssl=False)
        native._sample_rate = RATE
        native_config = native._create_recognition_config().config
        self.assertEqual(config.endpointing_config, native_config.endpointing_config)
        self.assertEqual(config.endpointing_config.stop_history, 320)
        self.assertEqual(b"".join(request.audio_content for request in requests_seen[1:]), bytes(1026))
        self.assertTrue(all(not request.runtime_config for request in requests_seen))

    async def test_tts_clean_early_eof_rejects_unsynthesized_text(self):
        class Text:
            async def stream(self, messages):
                yield "First."
                yield "Second."

        class Aggregator:
            async def aggregate(self, text):
                yield SimpleNamespace(text=text)

            async def flush(self):
                return None

        def synthesis(texts):
            next(iter(texts))
            yield bytes(200)

        with self.assertRaisesRegex(ValueError, "all submitted text"):
            await StreamingResponder(Text(), synthesis, aggregator_factory=Aggregator).respond([], "one")

    async def test_final_metadata_only_tts_chunk_provides_played_word_alignment(self):
        class Text:
            async def stream(self, messages):
                yield "hello world"

        class Aggregator:
            async def aggregate(self, text):
                yield SimpleNamespace(text=text)

            async def flush(self):
                return None

        def synthesis(texts):
            list(texts)
            yield SimpleNamespace(audio=bytes(16000), meta=SimpleNamespace(words=[]))
            yield SimpleNamespace(
                audio=b"",
                meta=SimpleNamespace(
                    words=[
                        SimpleNamespace(word="hello", start_time=0, end_time=200),
                        SimpleNamespace(word="world", start_time=200, end_time=400),
                    ]
                ),
            )

        trace = await StreamingResponder(Text(), synthesis, aggregator_factory=Aggregator).respond([], "one")
        self.assertEqual(len(trace.audio), 1)
        self.assertEqual([(word.text, word.end_sample) for word in trace.words], [("hello", 3200), ("world", 6400)])
        self.assertEqual(trace.metrics["spoken_alignment"], "provider_reported")

    async def test_tts_metadata_preserves_first_seen_order_and_rejects_reset_offsets(self):
        class Text:
            async def stream(self, messages):
                yield "first future"

        class Aggregator:
            async def aggregate(self, text):
                yield SimpleNamespace(text=text)

            async def flush(self):
                return None

        def word(text, start, end):
            return SimpleNamespace(word=text, start_time=start, end_time=end)

        first = word("first", 500, 1000)
        for second, expected in (
            (word("future", 0, 500), "invalid_provider_offsets_not_used"),
            (word("future", 1000, 1500), "provider_reported"),
        ):
            with self.subTest(policy=expected):

                def synthesis(texts, second=second):
                    list(texts)
                    yield SimpleNamespace(audio=bytes(RATE * 2), meta=SimpleNamespace(words=[first]))
                    yield SimpleNamespace(audio=bytes(RATE * 2), meta=SimpleNamespace(words=[first, second]))

                trace = await StreamingResponder(Text(), synthesis, aggregator_factory=Aggregator).respond([], "one")
                self.assertEqual(trace.metrics["spoken_alignment"], expected)
                if expected == "provider_reported":
                    self.assertEqual([item.text for item in trace.words], ["first", "future"])
                else:
                    self.assertEqual(trace.words, ())

    async def test_tool_context_does_not_commit_unplayed_spoken_content(self):
        class Stream:
            def __init__(self, deltas):
                self.deltas = deltas

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def __aiter__(self):
                async def values():
                    for delta in self.deltas:
                        yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

                return values()

        calls = []

        async def create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                tool = SimpleNamespace(
                    index=0, id="call-1", function=SimpleNamespace(name="lookup", arguments='{"id": 1}')
                )
                return Stream([SimpleNamespace(content="Checking. ", tool_calls=[tool])])
            return Stream([SimpleNamespace(content="Found it.", tool_calls=None)])

        async def lookup(params):
            await params.result_callback({"found": params.arguments["id"]})

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        history = [{"role": "user", "content": "find it"}]
        events = [
            event
            async for event in OpenAITextStream(
                client, model="test", extra={}, tools=[{"type": "function"}], handlers={"lookup": lookup}
            ).stream(history)
        ]
        self.assertEqual(len(history), 1)
        self.assertEqual(events[0], "Checking. ")
        self.assertIsNone(events[1]["content"])
        self.assertEqual(events[1]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(events[2]["role"], "tool")
        self.assertEqual(events[3], "Found it.")
        self.assertEqual(calls[1]["messages"][-1]["role"], "tool")

    async def test_summary_is_a_provisional_context_event(self):
        class Stream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def __aiter__(self):
                async def values():
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="answer", tool_calls=None))]
                    )

                return values()

        async def create(**kwargs):
            self.assertEqual(kwargs["messages"][0]["content"], "summary")
            return Stream()

        async def prepare(history):
            return [{"role": "system", "content": "summary"}, history[-1]]

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        history = [{"role": "system", "content": "original"}, {"role": "user", "content": "hello"}]
        events = [
            event
            async for event in OpenAITextStream(
                client, model="test", extra={}, tools=[], handlers={}, history_preparer=prepare
            ).stream(history)
        ]
        self.assertEqual(history[0]["content"], "original")
        self.assertEqual(events[0]["replace_messages"][0]["content"], "summary")
        self.assertEqual(events[1], "answer")

    async def test_one_continuous_stream_drains_all_audio_and_late_final(self):
        source = AudioSource(b"\x01\x00" * 16017)
        observed = []

        def stream(audio):
            observed.extend(audio)
            yield final("hello", 1.0, (("hello", 200, 700),))
            yield final("", source.sample_count / RATE)

        trace = await ContinuousASR(stream).recognize(source)
        self.assertEqual(b"".join(observed), source.pcm)
        self.assertEqual(len(observed[-1]), 2 * (16017 % 512))
        self.assertEqual(trace.metrics["asr_streams"], 1)
        self.assertEqual(trace.metrics["asr_final_coverage_sample"], source.sample_count)
        self.assertEqual(trace.segments[0].words, (("hello", 3200, 11200),))
        self.assertGreaterEqual(trace.segments[0].available_sample, 16000)

    async def test_stream_early_eof_is_failure_not_silent_success(self):
        def stream(audio):
            next(iter(audio))
            yield final("", 0.032)

        with self.assertRaisesRegex(ValueError, "before consuming"):
            await ContinuousASR(stream).recognize(AudioSource(bytes(32000)))

    async def test_missing_offsets_cannot_guess_turn_ownership(self):
        def stream(audio):
            list(audio)
            yield final("unplaced text", 0)

        with self.assertRaisesRegex(ValueError, "lacks"):
            await ContinuousASR(stream).recognize(AudioSource(bytes(32000)))

    async def test_nonmonotonic_asr_coverage_rejected(self):
        def stream(audio):
            list(audio)
            yield final("one", 0.8)
            yield final("two", 0.5)

        with self.assertRaisesRegex(ValueError, "nonmonotonic"):
            await ContinuousASR(stream).recognize(AudioSource(bytes(32000)))

    async def test_availability_is_dependency_plus_measured_residual(self):
        ticks = iter((0, 10_000_000, 20_000_000, 30_000_000, 40_000_000, 80_000_000, 90_000_000))

        def stream(audio):
            list(audio)
            yield final("hello", 0.064)

        trace = await ContinuousASR(stream, clock=lambda: next(ticks)).recognize(AudioSource(bytes(3 * 512 * 2)))
        self.assertEqual(trace.segments[0].end_sample, 1024)
        self.assertEqual(trace.segments[0].available_sample, 1024 + 960)

    async def test_llm_and_tts_overlap_and_explicit_completion_preserves_tail(self):
        audio_started = threading.Event()

        class Text:
            async def stream(self, messages):
                messages.append({"role": "user", "content": "must not leak"})
                yield "First."
                while not audio_started.is_set():
                    await asyncio.sleep(0)
                yield "Second."

        class Aggregator:
            async def aggregate(self, text):
                yield SimpleNamespace(text=text)

            async def flush(self):
                return None

        def synthesis(texts):
            for text in texts:
                audio_started.set()
                yield b"\x01\x00" * len(text)

        history = [{"role": "user", "content": "hello"}]
        trace = await StreamingResponder(Text(), synthesis, aggregator_factory=Aggregator).respond(history, "one")
        self.assertEqual(len(history), 1)
        self.assertEqual(trace.text, "First.Second.")
        self.assertEqual(sum(len(chunk.pcm) for chunk in trace.audio), 26)
        self.assertLess(trace.audio[0].ready_offset_ns, trace.metrics["llm_complete_ns"])
        self.assertGreaterEqual(trace.compute_ns, trace.audio[-1].ready_offset_ns)

    async def test_provider_failure_does_not_return_success_trace(self):
        class Text:
            async def stream(self, messages):
                yield "hello"
                raise RuntimeError("provider failed")

        class Aggregator:
            async def aggregate(self, text):
                yield SimpleNamespace(text=text)

            async def flush(self):
                return None

        def synthesis(texts):
            for _ in texts:
                yield b"\x01\x00"

        with self.assertRaises(ExceptionGroup):
            await StreamingResponder(Text(), synthesis, aggregator_factory=Aggregator).respond([], "one")


if __name__ == "__main__":
    unittest.main()
