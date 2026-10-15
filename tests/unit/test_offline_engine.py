# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Offline invariants against scripted services; no model or transport required."""

import copy
import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from examples.generic.offline.engine import OfflineSessionEngine
from examples.generic.offline.models import (
    AudioSource,
    AudioSpan,
    DetectionTrace,
    GeneratedAudio,
    RecognitionSegment,
    RecognitionTrace,
    ResponseTrace,
    SessionResult,
    SpeechInterval,
    SpokenWord,
    UserTurn,
    ns_to_samples,
    samples_to_ns,
)
from examples.generic.offline.recorder import render_pcm, write_session, write_source


def pcm(samples, value=111):
    """Create exact identifiable PCM for placement and clipping checks."""
    return struct.pack("<h", value) * samples


def response(text="reply", chunks=((100, 500),), compute=200, words=(), metrics=None):
    """Build a response trace with sample-sized compute and availability."""
    return ResponseTrace(
        text,
        tuple(
            GeneratedAudio(pcm(count, index + 1), samples_to_ns(ready)) for index, (ready, count) in enumerate(chunks)
        ),
        samples_to_ns(compute),
        words,
        metrics or {},
    )


def tool_call(call_id):
    """Create a standard NIM/OpenAI tool request with an identifiable ID."""
    return {"id": call_id, "type": "function", "function": {"name": "lookup", "arguments": "{}"}}


class ScriptedDetector:
    """Return immutable source-only decisions."""

    def __init__(self, turns, speech=None):
        """Use each proposed turn as one speech interval unless specified."""
        self.trace = DetectionTrace(
            tuple(speech)
            if speech is not None
            else tuple(SpeechInterval(turn.start_sample, turn.end_sample, turn.start_sample) for turn in turns),
            tuple(turns),
        )

    async def detect(self, source):
        """Return precomputed detection."""
        return self.trace


class ScriptedRecognizer:
    """Model an explicitly drained single continuous source request."""

    def __init__(self, segments):
        """Store final source-owned transcript portions."""
        self.segments = tuple(segments)
        self.calls = 0

    async def recognize(self, source):
        """Count continuous requests and return coverage/availability."""
        self.calls += 1
        return RecognitionTrace(self.segments, {"continuous_stream": True})


class ScriptedResponder:
    """Keep snapshots and return provisional traces in request order."""

    def __init__(self, responses, mutate=False):
        """Capture requests while optionally mutating the private snapshot."""
        self.responses = list(responses)
        self.requests = []
        self.mutate = mutate

    async def respond(self, messages, turn_id):
        """Never assume future input has already entered history."""
        self.requests.append((copy.deepcopy(messages), turn_id))
        if self.mutate:
            messages.append({"role": "assistant", "content": "provider mutation"})
        return self.responses.pop(0)


class EngineTest(unittest.IsolatedAsyncioTestCase):
    """Validate conversation causality and exact sample-level playback."""

    async def evaluate(self, turns, segments, traces, count=5000, speech=None, messages=()):
        """Run one dependency-free session and expose the test ports."""
        detector = ScriptedDetector(turns, speech)
        recognizer = ScriptedRecognizer(segments)
        responder = ScriptedResponder(traces)
        engine = OfflineSessionEngine(detector, recognizer, responder, messages)
        result = await engine.run(AudioSource(pcm(count)))
        return result, responder, recognizer

    async def test_response_anchors_to_readiness_and_preserves_silence(self):
        """Measured delay is included once and the source horizon stays intact."""
        turn = UserTurn("one", 100, 400, 1000, samples_to_ns(20))
        result, responder, recognizer = await self.evaluate(
            [turn], [RecognitionSegment("question", 100, 400, 1050)], [response()], count=3000
        )
        self.assertEqual(recognizer.calls, 1)
        self.assertEqual(responder.requests[0][0], [{"role": "user", "content": "question"}])
        self.assertEqual(result.spans[0].start_sample, 1150)
        self.assertEqual(result.spans[0].end_sample, 1650)
        rendered = render_pcm(result)
        self.assertEqual(rendered[: 1150 * 2], bytes(1150 * 2))
        self.assertEqual(rendered[1150 * 2 : 1650 * 2], pcm(500, 1))
        self.assertEqual(len(rendered), 6000)

    async def test_future_transcript_is_not_in_first_snapshot(self):
        """Source lookahead cannot leak another turn into the current prompt."""
        result, responder, _ = await self.evaluate(
            [UserTurn("one", 100, 400, 600), UserTurn("two", 2000, 2400, 2600)],
            [RecognitionSegment("first", 100, 400, 700), RecognitionSegment("future", 2000, 2400, 3000)],
            [response("answer one"), response("answer two")],
        )
        self.assertEqual(responder.requests[0][0], [{"role": "user", "content": "first"}])
        self.assertEqual(
            responder.requests[1][0],
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer one"},
                {"role": "user", "content": "future"},
            ],
        )
        self.assertEqual(result.metrics["committed_samples"], 1000)

    async def test_confirmation_at_speech_end_accepts_eof_and_following_silence(self):
        """Speech lasting exactly the start threshold remains a valid turn."""
        for endpoint in (100, 400):
            with self.subTest(endpoint=endpoint):
                result, responder, _ = await self.evaluate(
                    [UserTurn("one", 0, 100, endpoint)],
                    [RecognitionSegment("short", 0, 100, endpoint)],
                    [response()],
                    count=endpoint,
                    speech=[SpeechInterval(0, 100, 100)],
                )
                self.assertEqual(len(responder.requests), 1)
                self.assertEqual(result.spans[0].start_sample, endpoint + 100)
                speech_events = [
                    (event["type"], event["sample"])
                    for event in result.events
                    if event["type"] in ("user_speech_start", "user_speech_end")
                ]
                self.assertEqual(speech_events, [("user_speech_start", 100), ("user_speech_end", 100)])

    async def test_confirmation_at_speech_end_interrupts_and_releases_active_speech(self):
        """A threshold-length barge-in clips playback and permits its own reply."""
        result, responder, _ = await self.evaluate(
            [UserTurn("one", 100, 400, 600), UserTurn("two", 1200, 1300, 1800)],
            [RecognitionSegment("first", 100, 400, 650), RecognitionSegment("second", 1200, 1300, 1850)],
            [response("long reply", ((100, 2000),)), response("second reply")],
            speech=[SpeechInterval(100, 400, 150), SpeechInterval(1200, 1300, 1300)],
        )
        self.assertEqual(result.spans[0].end_sample, 1300)
        self.assertEqual(result.metrics["responses"][0]["committed_samples"], 550)
        self.assertEqual(result.spans[1].start_sample, 1950)
        self.assertEqual(len(responder.requests), 2)

    async def test_late_asr_while_new_speech_defers_and_coalesces(self):
        """Delayed text cannot trigger a reply over an ongoing later utterance."""
        result, responder, _ = await self.evaluate(
            [UserTurn("one", 100, 300, 500), UserTurn("two", 700, 1600, 1800)],
            [RecognitionSegment("first", 100, 300, 1000), RecognitionSegment("second", 700, 1600, 1900)],
            [response()],
        )
        self.assertEqual(len(responder.requests), 1)
        self.assertEqual(responder.requests[0], ([{"role": "user", "content": "first\nsecond"}], "two"))
        request = next(event for event in result.events if event["type"] == "response_requested")
        self.assertEqual(request["sample"], 1900)
        self.assertEqual(request["turn_ids"], ["one", "two"])

    async def test_earlier_asr_ready_after_later_endpoint_still_coalesces(self):
        """Out-of-order results retain source order and wait for all ownership."""
        result, responder, _ = await self.evaluate(
            [UserTurn("one", 100, 300, 500), UserTurn("two", 700, 1600, 1800)],
            [RecognitionSegment("second", 700, 1600, 1900), RecognitionSegment("first", 100, 300, 2200)],
            [response()],
        )
        self.assertEqual(responder.requests[0][0][0]["content"], "first\nsecond")
        self.assertEqual(result.spans[0].start_sample, 2300)

    async def test_interruption_clips_partial_chunk_at_confirmed_start(self):
        """Playback interruption uses confirmation, never backdated speech onset."""
        result, responder, _ = await self.evaluate(
            [UserTurn("one", 100, 400, 600), UserTurn("two", 1200, 1600, 1800)],
            [RecognitionSegment("first", 100, 400, 650), RecognitionSegment("second", 1200, 1600, 1850)],
            [
                response(
                    "one two three",
                    ((100, 2000),),
                    words=(SpokenWord("one", 0, 300), SpokenWord("two", 300, 900), SpokenWord("three", 900, 2000)),
                ),
                response("second reply"),
            ],
            speech=[SpeechInterval(100, 400, 150), SpeechInterval(1200, 1600, 1300)],
        )
        self.assertEqual(result.spans[0].start_sample, 750)
        self.assertEqual(result.spans[0].end_sample, 1300)
        self.assertEqual(result.spans[0].pcm, pcm(550, 1))
        self.assertEqual(responder.requests[1][0][1], {"role": "assistant", "content": "one"})
        self.assertEqual(result.metrics["responses"][0]["generated_samples"], 2000)
        self.assertEqual(result.metrics["responses"][0]["committed_samples"], 550)

    async def test_unaligned_interrupted_text_is_not_committed(self):
        """Unplayed future response text must never become context."""
        result, responder, _ = await self.evaluate(
            [UserTurn("one", 100, 400, 600), UserTurn("two", 1200, 1600, 1800)],
            [RecognitionSegment("first", 100, 400, 650), RecognitionSegment("second", 1200, 1600, 1850)],
            [response("full unaligned reply", ((100, 2000),)), response()],
        )
        self.assertEqual([message["role"] for message in responder.requests[1][0]], ["user", "user"])
        interrupted = next(event for event in result.events if event["type"] == "response_interrupted")
        self.assertEqual(interrupted["history_policy"], "unaligned_interrupted_text_omitted")

    async def test_interruption_before_audio_drops_future_output(self):
        """A response still computing must be superseded by new input."""
        result, responder, _ = await self.evaluate(
            [UserTurn("one", 100, 400, 600), UserTurn("two", 900, 1100, 1300)],
            [RecognitionSegment("first", 100, 400, 650), RecognitionSegment("second", 900, 1100, 1350)],
            [response("stale", ((1000, 500),), compute=1100), response("new")],
        )
        self.assertTrue(all(span.response_id == "response-2" for span in result.spans))
        self.assertFalse(
            any(event["type"] == "bot_speech_start" and event["response_id"] == "response-1" for event in result.events)
        )
        self.assertNotIn("stale", str(responder.requests[1][0]))

    async def test_same_sample_user_start_wins_over_audio_and_context(self):
        """Equal-time causal order suppresses a newly available stale response."""
        context = {"ready_offset_ns": samples_to_ns(250), "message": {"role": "tool", "content": "late"}}
        result, responder, _ = await self.evaluate(
            [UserTurn("one", 100, 400, 600), UserTurn("two", 900, 1100, 1300)],
            [RecognitionSegment("first", 100, 400, 650), RecognitionSegment("second", 900, 1100, 1350)],
            [response("stale", ((250, 500),), compute=300, metrics={"context_events": [context]}), response()],
        )
        self.assertTrue(all(span.response_id == "response-2" for span in result.spans))
        self.assertNotIn("late", str(responder.requests[1][0]))

    async def test_context_events_commit_only_before_interruption(self):
        """Tool results have independent causal availability on the timeline."""
        contexts = [
            {"ready_offset_ns": samples_to_ns(100), "message": {"role": "tool", "content": "early"}},
            {"ready_offset_ns": samples_to_ns(400), "message": {"role": "tool", "content": "future"}},
        ]
        _, responder, _ = await self.evaluate(
            [UserTurn("one", 100, 400, 600), UserTurn("two", 900, 1100, 1300)],
            [RecognitionSegment("first", 100, 400, 650), RecognitionSegment("second", 900, 1100, 1350)],
            [response("reply", ((200, 500),), compute=500, metrics={"context_events": contexts}), response()],
        )
        self.assertIn("early", str(responder.requests[1][0]))
        self.assertNotIn("future", str(responder.requests[1][0]))

    async def test_summary_replacement_commits_before_interruption(self):
        """An available private summary becomes authoritative for the next turn."""
        replacement = [{"role": "system", "content": "summarized"}, {"role": "user", "content": "first"}]
        _, responder, _ = await self.evaluate(
            [UserTurn("one", 100, 400, 600), UserTurn("two", 900, 1100, 1300)],
            [RecognitionSegment("first", 100, 400, 650), RecognitionSegment("second", 900, 1100, 1350)],
            [
                response(
                    "reply",
                    ((200, 500),),
                    compute=500,
                    metrics={
                        "context_events": [{"ready_offset_ns": samples_to_ns(100), "replace_messages": replacement}]
                    },
                ),
                response(),
            ],
            messages=[{"role": "system", "content": "original"}],
        )
        self.assertEqual(responder.requests[1][0][0], {"role": "system", "content": "summarized"})
        self.assertNotIn("original", str(responder.requests[1][0]))

    async def test_interrupted_unanswered_tool_call_does_not_leak_into_next_request(self):
        """Interrupted requests retain prior complete transactions even with ID reuse."""
        prior = [
            {"role": "assistant", "content": None, "tool_calls": [tool_call("shared")]},
            {"role": "tool", "tool_call_id": "shared", "content": "previous result"},
            {"role": "assistant", "content": "previous answer"},
        ]
        contexts = [
            {
                "ready_offset_ns": samples_to_ns(100),
                "message": {"role": "assistant", "content": None, "tool_calls": [tool_call("shared")]},
            },
            {
                "ready_offset_ns": samples_to_ns(500),
                "message": {"role": "tool", "tool_call_id": "shared", "content": "future result"},
            },
        ]
        result, responder, _ = await self.evaluate(
            [UserTurn("one", 100, 400, 600), UserTurn("two", 900, 1100, 1300)],
            [RecognitionSegment("first", 100, 400, 650), RecognitionSegment("second", 900, 1100, 1350)],
            [response("reply", ((200, 500),), compute=500, metrics={"context_events": contexts}), response()],
            messages=prior,
        )
        self.assertEqual(
            responder.requests[1][0],
            prior + [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}],
        )
        event = next(item for item in result.events if item["type"] == "response_interrupted")
        self.assertEqual(event["omitted_tool_call_ids"], ["shared"])

    async def test_interruption_preserves_completed_subset_of_multicall(self):
        """A partial tool transaction retains matched calls/results and removes only pending calls."""
        completed_message = {"role": "tool", "tool_call_id": "done", "content": "available"}
        contexts = [
            {
                "ready_offset_ns": samples_to_ns(100),
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("done"), tool_call("pending")],
                },
            },
            {"ready_offset_ns": samples_to_ns(200), "message": completed_message},
            {
                "ready_offset_ns": samples_to_ns(500),
                "message": {"role": "tool", "tool_call_id": "pending", "content": "future"},
            },
        ]
        result, responder, _ = await self.evaluate(
            [UserTurn("one", 100, 400, 600), UserTurn("two", 900, 1100, 1300)],
            [RecognitionSegment("first", 100, 400, 650), RecognitionSegment("second", 900, 1100, 1350)],
            [response("reply", ((200, 500),), compute=500, metrics={"context_events": contexts}), response()],
        )
        self.assertEqual(
            responder.requests[1][0],
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": None, "tool_calls": [tool_call("done")]},
                completed_message,
                {"role": "user", "content": "second"},
            ],
        )
        event = next(item for item in result.events if item["type"] == "response_interrupted")
        self.assertEqual(event["omitted_tool_call_ids"], ["pending"])

    async def test_summary_replacement_does_not_hide_unanswered_tool_call(self):
        """Pruning uses current history positions after private summaries replace it."""
        unfinished = {"role": "assistant", "content": None, "tool_calls": [tool_call("pending")]}
        contexts = [
            {"ready_offset_ns": samples_to_ns(100), "message": unfinished},
            {
                "ready_offset_ns": samples_to_ns(200),
                "replace_messages": [
                    {"role": "system", "content": "summary"},
                    {"role": "user", "content": "first"},
                    unfinished,
                ],
            },
        ]
        _, responder, _ = await self.evaluate(
            [UserTurn("one", 100, 400, 600), UserTurn("two", 900, 1100, 1300)],
            [RecognitionSegment("first", 100, 400, 650), RecognitionSegment("second", 900, 1100, 1350)],
            [response("reply", ((200, 500),), compute=500, metrics={"context_events": contexts}), response()],
        )
        self.assertEqual(
            responder.requests[1][0],
            [
                {"role": "system", "content": "summary"},
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
            ],
        )

    async def test_summary_at_interruption_is_suppressed(self):
        """A summary becoming available at user start cannot overwrite history."""
        _, responder, _ = await self.evaluate(
            [UserTurn("one", 100, 400, 600), UserTurn("two", 900, 1100, 1300)],
            [RecognitionSegment("first", 100, 400, 650), RecognitionSegment("second", 900, 1100, 1350)],
            [
                response(
                    "reply",
                    ((200, 500),),
                    compute=500,
                    metrics={
                        "context_events": [
                            {
                                "ready_offset_ns": samples_to_ns(250),
                                "replace_messages": [{"role": "system", "content": "future"}],
                            }
                        ]
                    },
                ),
                response(),
            ],
            messages=[{"role": "system", "content": "original"}],
        )
        self.assertEqual(responder.requests[1][0][0], {"role": "system", "content": "original"})
        self.assertNotIn("future", str(responder.requests[1][0]))

    async def test_malformed_context_operations_are_rejected(self):
        """No trace may schedule ambiguous context writes or backdated summaries."""
        invalid = (
            {"ready_offset_ns": -1, "message": {}},
            {"ready_offset_ns": True, "message": {}},
            {"ready_offset_ns": 0, "message": {}, "replace_messages": []},
            {"ready_offset_ns": 0, "replace_messages": ["invalid"]},
            {"ready_offset_ns": 0},
        )
        for context in invalid:
            with self.subTest(context=context), self.assertRaises(ValueError):
                await self.evaluate(
                    [UserTurn("one", 100, 400, 600)],
                    [RecognitionSegment("first", 100, 400, 650)],
                    [response(metrics={"context_events": [context]})],
                )

    async def test_tts_gap_and_empty_chunks_are_silence_not_completion(self):
        """Availability gaps do not shorten or terminate a provisional response."""
        result, _, _ = await self.evaluate(
            [UserTurn("one", 100, 400, 600)],
            [RecognitionSegment("first", 100, 400, 650)],
            [response(chunks=((100, 200), (400, 0), (700, 123)), compute=750)],
            count=1000,
        )
        self.assertEqual([(span.start_sample, span.end_sample) for span in result.spans], [(750, 950), (1350, 1473)])
        self.assertEqual(render_pcm(result)[950 * 2 : 1350 * 2], bytes(400 * 2))
        self.assertEqual(result.rendered_samples, 1473)

    async def test_end_of_input_drains_late_asr_and_final_response(self):
        """Finite ingress EOF cannot cancel recognition or truncate the tail."""
        result, _, _ = await self.evaluate(
            [UserTurn("one", 100, 700, 1000)],
            [RecognitionSegment("last", 100, 700, 1700)],
            [response(chunks=((100, 501),))],
            count=1000,
        )
        self.assertEqual(result.rendered_samples, 2301)
        self.assertEqual(len(render_pcm(result)), 4602)
        self.assertEqual(result.spans[0].pcm, pcm(501, 1))

    async def test_silent_source_produces_exact_silent_output(self):
        """Silence-only recordings preserve the complete session duration."""
        result, responder, recognizer = await self.evaluate([], [], [], count=4800000)
        self.assertFalse(responder.requests)
        self.assertEqual(recognizer.calls, 1)
        self.assertEqual(result.rendered_samples, 4800000)
        self.assertEqual(render_pcm(result), bytes(9600000))

    async def test_empty_final_transcript_does_not_trigger_reply(self):
        """No-transcription turns resolve explicitly rather than hanging."""
        result, responder, _ = await self.evaluate([UserTurn("one", 100, 400, 600)], [], [])
        self.assertFalse(responder.requests)
        self.assertIn("empty_user_turn", [event["type"] for event in result.events])

    async def test_cross_turn_segment_without_alignment_fails(self):
        """The engine must not guess transcript ownership from a final timestamp."""
        with self.assertRaisesRegex(ValueError, "multiple turns"):
            await self.evaluate(
                [UserTurn("one", 100, 400, 600), UserTurn("two", 1500, 1800, 2000)],
                [RecognitionSegment("both turns", 100, 1800, 2100)],
                [],
            )

    async def test_cross_turn_segment_uses_word_ownership(self):
        """Aligned finalized text portions retain source ordering."""
        result, responder, _ = await self.evaluate(
            [UserTurn("one", 100, 400, 600), UserTurn("two", 1500, 1800, 2000)],
            [RecognitionSegment("first second", 100, 1800, 2100, (("first", 100, 400), ("second", 1500, 1800)))],
            [response()],
        )
        self.assertEqual(responder.requests[0][0][0]["content"], "first\nsecond")
        self.assertEqual(result.spans[0].start_sample, 2200)

    async def test_live_riva_quiet_trailing_word_belongs_to_open_turn(self):
        """ASR may hear words after VAD speech end and before the turn endpoint."""
        result, responder, _ = await self.evaluate(
            [UserTurn("one", 3072, 48128, 98816)],
            [
                RecognitionSegment("I need help.", 5120, 26880, 30000),
                RecognitionSegment("Any", 43520, 43520, 45000),
                RecognitionSegment("advice", 49920, 53760, 55000),
            ],
            [response()],
            count=100000,
        )
        self.assertEqual(responder.requests[0][0], [{"role": "user", "content": "I need help. Any advice"}])
        self.assertEqual(result.spans[0].start_sample, 98916)
        point = next(event for event in result.events if event.get("source_start_sample") == 43520)
        self.assertEqual(point["source_timestamp_kind"], "point")
        self.assertEqual(point["source_end_sample"], 43520)

    async def test_cross_turn_segment_accepts_zero_duration_word_offsets(self):
        """Riva point words split by bounded ownership rather than being rejected."""
        _, responder, _ = await self.evaluate(
            [UserTurn("one", 100, 400, 600), UserTurn("two", 1500, 1800, 2000)],
            [
                RecognitionSegment(
                    "first second",
                    100,
                    1800,
                    2100,
                    (("first", 300, 300), ("second", 1700, 1700)),
                )
            ],
            [response()],
        )
        self.assertEqual(responder.requests[0][0], [{"role": "user", "content": "first\nsecond"}])

    async def test_endpoint_point_belongs_to_new_turn_opened_at_that_offset(self):
        """Half-open windows make a shared endpoint/start point unambiguous."""
        _, responder, _ = await self.evaluate(
            [UserTurn("one", 100, 400, 1000), UserTurn("two", 1000, 1200, 1500)],
            [RecognitionSegment("first", 100, 400, 1100), RecognitionSegment("next", 1000, 1000, 1200)],
            [response()],
        )
        self.assertEqual(responder.requests[0][0], [{"role": "user", "content": "first\nnext"}])

    async def test_point_at_explicit_source_eof_belongs_to_final_turn(self):
        """Source EOF is a bounded closing point even for a zero-duration final."""
        result, responder, _ = await self.evaluate(
            [UserTurn("one", 100, 400, 1000)],
            [RecognitionSegment("last", 1000, 1000, 1100)],
            [response()],
            count=1000,
        )
        self.assertEqual(responder.requests[0][0], [{"role": "user", "content": "last"}])
        self.assertEqual(result.spans[0].start_sample, 1200)

    async def test_no_transcript_is_assigned_after_a_closed_turn(self):
        """A quiet-word allowance cannot absorb arbitrary later audio in gaps."""
        for start, end in ((600, 600), (601, 601), (700, 800)):
            with self.subTest(start=start, end=end), self.assertRaisesRegex(ValueError, "no detected source turn"):
                await self.evaluate(
                    [UserTurn("one", 100, 400, 600)],
                    [RecognitionSegment("outside", start, end, 900)],
                    [],
                    count=1000,
                )

    async def test_cross_turn_word_interval_remains_ambiguous(self):
        """A positive-length word touching both turn windows cannot be guessed."""
        with self.assertRaisesRegex(ValueError, "ambiguous turn ownership"):
            await self.evaluate(
                [UserTurn("one", 100, 400, 1000), UserTurn("two", 1000, 1200, 1500)],
                [RecognitionSegment("unclear", 900, 1100, 1600, (("unclear", 900, 1100),))],
                [],
            )

    async def test_snapshots_cannot_mutate_engine_history(self):
        """Provider-owned copies preserve dependency inversion and isolation."""
        initial = [{"role": "system", "content": "original"}]
        responder = ScriptedResponder([response()], mutate=True)
        engine = OfflineSessionEngine(
            ScriptedDetector([UserTurn("one", 100, 400, 600)]),
            ScriptedRecognizer([RecognitionSegment("first", 100, 400, 650)]),
            responder,
            initial,
        )
        initial[0]["content"] = "external mutation"
        result = await engine.run(AudioSource(pcm(3000)))
        self.assertEqual(result.messages[0]["content"], "original")
        self.assertNotIn("provider mutation", str(result.messages))

    async def test_media_logic_never_sleeps(self):
        """Conversation advance does not pay the logical duration in wall time."""
        with patch("asyncio.sleep", side_effect=AssertionError("unexpected media pacing")):
            result, _, _ = await self.evaluate(
                [UserTurn("one", 100, 400, 600)],
                [RecognitionSegment("first", 100, 400, 650)],
                [response()],
            )
        self.assertEqual(result.spans[0].start_sample, 750)

    async def test_invalid_asr_readiness_fails_instead_of_backdating(self):
        """Recognition cannot be available before its source coverage ends."""
        with self.assertRaisesRegex(ValueError, "causal"):
            await self.evaluate(
                [UserTurn("one", 100, 400, 600)],
                [RecognitionSegment("first", 100, 400, 300)],
                [],
            )

    async def test_empty_audio_explicit_completion_omits_unspoken_text(self):
        """A silent completed response adds no spoken history or artificial tail."""
        result, _, _ = await self.evaluate(
            [UserTurn("one", 100, 400, 600)],
            [RecognitionSegment("first", 100, 400, 1200)],
            [response(chunks=(), compute=100)],
            count=1000,
        )
        self.assertEqual(result.rendered_samples, 1000)
        self.assertEqual(result.events[-1]["type"], "session_complete")
        self.assertEqual(result.events[-1]["sample"], 1300)
        self.assertEqual(result.metrics["logical_end_sample"], 1300)
        self.assertEqual(result.messages, ({"role": "user", "content": "first"},))

    async def test_invalid_tts_availability_fails(self):
        """A producer trace cannot expose a chunk after its declared completion."""
        with self.assertRaisesRegex(ValueError, "response computation"):
            await self.evaluate(
                [UserTurn("one", 100, 400, 600)],
                [RecognitionSegment("first", 100, 400, 650)],
                [response(chunks=((300, 50),), compute=200)],
            )


class RecorderTest(unittest.TestCase):
    """Check exact duration, signed PCM fidelity, and unambiguous placement."""

    def test_rejects_overlapping_spans(self):
        """No last-writer-wins mixing can silently hide scheduler errors."""
        result = SessionResult((AudioSpan(10, pcm(20), "one"), AudioSpan(29, pcm(3), "two")), 50, (), ())
        with self.assertRaisesRegex(ValueError, "nonoverlapping"):
            render_pcm(result)

    def test_writes_300_second_horizon_with_extended_final_tail(self):
        """Artifacts retain exact source length and every output sample."""
        source = AudioSource(pcm(4800000, 0))
        result = SessionResult(
            (AudioSpan(4799999, pcm(4, -32768), "tail"),),
            4800003,
            ({"type": "input_end", "sample": 4800000},),
            (),
            {"compute_ns": 42},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_source(root, source)
            manifest = write_session(root, source, result)
            with wave.open(str(root / "source.wav"), "rb") as audio:
                self.assertEqual(audio.getnframes(), 4800000)
            with wave.open(str(root / "full_output.wav"), "rb") as audio:
                self.assertEqual(audio.getnframes(), 4800003)
                audio.setpos(4799999)
                self.assertEqual(audio.readframes(4), pcm(4, -32768))
            self.assertEqual(manifest["rendered_samples"], 4800003)
            self.assertEqual(json.loads((root / "manifest.json").read_text())["metrics"], {"compute_ns": 42})
            self.assertEqual(json.loads((root / "events.jsonl").read_text())["sample"], 4800000)

    def test_rejects_negative_or_outside_spans(self):
        """Offsets and extent remain checked at the recording boundary."""
        for span in [AudioSpan(-1, pcm(1), "bad"), AudioSpan(9, pcm(2), "bad"), AudioSpan(0, b"x", "bad")]:
            with self.subTest(span=span), self.assertRaises(ValueError):
                render_pcm(SessionResult((span,), 10, (), ()))

    def test_cumulative_rate_mapping_has_no_packet_rounding_drift(self):
        """Time positions round forward once, independently of packetization."""
        for rate in (16000, 22050, 24000, 48000):
            for position in (0, 1, 239, 100001, 300 * rate):
                stamp = samples_to_ns(position, rate)
                self.assertEqual(ns_to_samples(stamp, rate), position)
        self.assertEqual(ns_to_samples(1), 1)
        self.assertEqual(ns_to_samples(samples_to_ns(4800000)), 4800000)


if __name__ == "__main__":
    unittest.main()
