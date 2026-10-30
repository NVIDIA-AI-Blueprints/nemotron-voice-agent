# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise shared aggregation and live output packing in the causal engine."""

import copy
import struct
import unittest

from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    OutputAudioRawFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from examples.generic.offline.causal import CausalSessionEngine
from examples.generic.offline.models import (
    RATE,
    AudioSource,
    DetectionTrace,
    GeneratedAudio,
    ResponseTrace,
    samples_to_ns,
)
from examples.generic.offline.recognition import ASRChunkEvent, ASREventTrace
from examples.generic.offline.recorder import render_pcm
from examples.generic.offline.tts_trace import CapturedTTSFrame, TTSResponseTrace


def pcm(samples, value=300):
    """Create identifiable mono PCM with exact sample counts."""
    return struct.pack("<h", value) * samples


def response(text="Answer.", *, audio_samples=640, ready=100, compute=200, words=(), metrics=None):
    """Build a provider result using sample-sized compute times."""
    return ResponseTrace(
        text,
        (GeneratedAudio(pcm(audio_samples), samples_to_ns(ready)),) if audio_samples else (),
        samples_to_ns(compute),
        words=tuple(words),
        metrics=metrics or {},
    )


def vad_start(confirmed, acoustic=None):
    """Use the real detector's confirmed pointer shape."""
    return {
        "type": "speech_start",
        "sample": confirmed if acoustic is None else acoustic,
        "confirmed_sample": confirmed,
    }


def vad_stop(decision, acoustic=None):
    """Preserve the distinction between acoustic end and stop confirmation."""
    return {
        "type": "speech_end",
        "sample": decision if acoustic is None else acoustic,
        "decision_sample": decision,
        "reason": "vad_silence",
    }


class FakeDetector:
    """Return chunk-level VAD events with the deployed Silero settings."""

    configuration = {"policy": "silero", "vad": {"start_secs": 0.2, "stop_secs": 0.5}}

    def __init__(self, events=()):
        """Keep source events independent from transcription."""
        self.events = tuple(events)

    async def detect(self, source):
        """The actual engine must consume events, without interval ownership."""
        return DetectionTrace((), (), self.events)


class FakeRecognizer:
    """Return one fully drained continuous ASR stream, with no word timestamps."""

    def __init__(self, events=(), *, drained=True):
        """Events are (processed pointer, delivered pointer, text, final)."""
        self.events = tuple(events)
        self.drained = drained
        self.calls = 0
        self.cancelled = False

    async def recognize(self, source):
        """Attach each response to its containing source chunk."""
        self.calls += 1
        return ASREventTrace(
            tuple(
                ASRChunkEvent(
                    index,
                    processed,
                    min(source.sample_count, ((processed + 511) // 512) * 512),
                    available,
                    text,
                    final,
                )
                for index, (processed, available, text, final) in enumerate(self.events)
            ),
            {"asr_rpc_eof_drained": self.drained, "asr_word_timestamps": False, "asr_streams": 1},
        )

    def cancel(self):
        """Record cancellation without creating worker resources."""
        self.cancelled = True


class FakeResponder:
    """Capture the actual contexts and cancellation deadlines sent to models."""

    def __init__(self, responses=()):
        """Return provisional results in request order."""
        self.responses = list(responses)
        self.requests = []

    async def respond(self, messages, turn_id, *, compute_budget_ns=None):
        """No answer may read a later source transcript from future history."""
        self.requests.append((copy.deepcopy(messages), turn_id, compute_budget_ns))
        if not self.responses:
            raise AssertionError("The engine requested an unexpected extra response")
        return self.responses.pop(0)


class CausalEngineTests(unittest.IsolatedAsyncioTestCase):
    """Test the integrated sample scheduler with actual Pipecat aggregation."""

    async def evaluate(
        self, *, vad=(), asr=(), responses=(), source_samples=8192, chunk_units=1, policy_wait=0, output_rate=RATE
    ):
        """Use no model weights, sleeps, transport sockets, or credentials."""
        recognizer = FakeRecognizer(asr)
        responder = FakeResponder(responses)
        params = LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=policy_wait)]
            ),
            empty_user_turn=None,
        )
        engine = CausalSessionEngine(
            FakeDetector(vad),
            recognizer,
            responder,
            params=params,
            output_chunks_10ms=chunk_units,
            output_sample_rate=output_rate,
        )
        result = await engine.run(AudioSource(bytes(source_samples * 2)))
        return result, responder, recognizer

    async def test_leading_interim_before_vad_start_uses_online_fallback_and_no_word_ownership(self):
        """A leading word in an acoustic gap must not abort or be reassigned."""
        result, responder, recognizer = await self.evaluate(
            vad=[vad_start(1536, acoustic=1500), vad_stop(4096, acoustic=3500)],
            asr=[(1024, 1440, "Third", False), (2048, 2048, "Third statement is true.", True)],
            responses=[response()],
        )
        self.assertEqual(recognizer.calls, 1)
        self.assertEqual(responder.requests[0][0], [{"role": "user", "content": "Third statement is true."}])
        self.assertEqual(len(responder.requests), 1)
        starts = [event for event in result.events if event["type"] == "user_turn_started"]
        self.assertEqual(starts[0]["sample"], 1440)
        request = next(event for event in result.events if event["type"] == "response_requested")
        self.assertEqual(request["sample"], 4096)
        self.assertEqual(result.metrics["recognition_binding_policy"], "live_aggregator_event_order_no_word_timestamps")

    async def test_several_asr_finals_during_one_vad_turn_stay_one_llm_request(self):
        """ASR finalization does not become conversational end of turn."""
        result, responder, _ = await self.evaluate(
            vad=[vad_start(512), vad_stop(4096, acoustic=3500)],
            asr=[(1024, 1024, "First statement.", True), (2048, 2048, "Third statement.", True)],
            responses=[response()],
        )
        self.assertEqual(len(responder.requests), 1)
        self.assertEqual(responder.requests[0][0][-1]["content"], "First statement. Third statement.")
        self.assertEqual(result.messages[-1], {"role": "assistant", "content": "Answer."})

    async def test_final_before_vad_start_keeps_stock_fallback_rather_than_reassigning_words(self):
        """A genuine finalized fragment may start a turn before VAD confirms."""
        result, responder, _ = await self.evaluate(
            vad=[vad_start(1536, acoustic=1500), vad_stop(4096)],
            asr=[(1024, 1440, "Third", True), (2048, 2048, "statement is true.", True)],
            responses=[response("Early answer"), response("Next answer")],
        )
        self.assertEqual(len(responder.requests), 2)
        self.assertEqual(responder.requests[0][0], [{"role": "user", "content": "Third"}])
        self.assertEqual(responder.requests[1][0][-1], {"role": "user", "content": "statement is true."})
        interrupted = [event for event in result.events if event["type"] == "response_interrupted"]
        self.assertEqual(interrupted[0]["sample"], 1536)

    async def test_400ms_output_uses_live_padded_tail_and_extends_past_input_eof(self):
        """The source horizon is retained and final playback is not clipped."""
        result, responder, _ = await self.evaluate(
            vad=[vad_start(512), vad_stop(2048)],
            asr=[(1024, 1024, "Question", True)],
            responses=[response(audio_samples=100, ready=100, compute=200)],
            source_samples=4096,
            chunk_units=40,
        )
        self.assertEqual(len(responder.requests), 1)
        self.assertEqual(len(result.spans), 1)
        span = result.spans[0]
        self.assertEqual(span.start_sample, 2248)  # partial chunk flush waits for compute completion
        self.assertEqual(span.end_sample - span.start_sample, 6400)
        self.assertEqual(span.pcm[:200], pcm(100))
        self.assertEqual(span.pcm[200:], bytes(2 * (6400 - 100)))
        self.assertEqual(result.rendered_samples, 8648)
        self.assertEqual(len(render_pcm(result)), 2 * 8648)

    async def test_300_seconds_of_digital_silence_stays_silent_and_requests_nothing(self):
        """Upload duration and silence do not fabricate a turn or response."""
        result, responder, recognizer = await self.evaluate(source_samples=300 * RATE)
        self.assertEqual(responder.requests, [])
        self.assertEqual(recognizer.calls, 1)
        self.assertEqual(result.spans, ())
        self.assertEqual(result.rendered_samples, 300 * RATE)
        self.assertEqual(result.messages, ())

    async def test_final_asr_after_vad_stop_waits_for_text_at_its_event_position(self):
        """A delayed final keeps its delivery time rather than an acoustic timestamp."""
        result, responder, _ = await self.evaluate(
            vad=[vad_start(512), vad_stop(2048, acoustic=1536)],
            asr=[(2048, 3072, "Delayed final", True)],
            responses=[response()],
        )
        self.assertEqual(len(responder.requests), 1)
        request = next(event for event in result.events if event["type"] == "response_requested")
        self.assertEqual(request["sample"], 3072)
        self.assertEqual(responder.requests[0][0][-1]["content"], "Delayed final")

    async def test_no_vad_quiet_final_uses_live_transcription_start(self):
        """Silero's failure to confirm soft speech must not lose the utterance."""
        result, responder, _ = await self.evaluate(asr=[(1024, 1536, "Soft speech", True)], responses=[response()])
        self.assertEqual(responder.requests[0][0], [{"role": "user", "content": "Soft speech"}])
        self.assertEqual(
            next(event["sample"] for event in result.events if event["type"] == "response_requested"), 1536
        )

    async def test_eof_flush_generates_pending_turn_once_and_allows_tail_playback(self):
        """EndFrame flushes pending user text before closing its adapter."""
        result, responder, _ = await self.evaluate(
            asr=[(1024, 1024, "At EOF", True)],
            responses=[response(audio_samples=640, ready=100, compute=200)],
            source_samples=2048,
            policy_wait=0.6,
        )
        self.assertEqual(len(responder.requests), 1)
        self.assertEqual(
            result.messages, ({"role": "user", "content": "At EOF"}, {"role": "assistant", "content": "Answer."})
        )
        self.assertTrue(result.spans)
        self.assertGreater(result.rendered_samples, 2048)

    async def test_interruption_truncates_audio_at_shared_transcription_fallback_start(self):
        """A partial quiet transcript interrupts before later VAD confirmation."""
        result, responder, _ = await self.evaluate(
            vad=[vad_start(512), vad_stop(1536), vad_start(2304), vad_stop(4096)],
            asr=[
                (1024, 1024, "First question", True),
                (1536, 2048, "Next", False),
                (2560, 2560, "Next question", True),
            ],
            responses=[response("Long answer", audio_samples=2000), response("Second answer")],
        )
        self.assertEqual(len(responder.requests), 2)
        interrupted = [event for event in result.events if event["type"] == "response_interrupted"]
        self.assertEqual(interrupted[0]["sample"], 2048)
        first_spans = [span for span in result.spans if span.response_id == "response-1"]
        self.assertTrue(first_spans)
        self.assertTrue(all(span.end_sample <= 2048 for span in first_spans))
        self.assertNotIn({"role": "user", "content": "Next question"}, responder.requests[0][0])

    async def test_completed_assistant_precedes_next_user_context(self):
        """A subsequent request sees the committed response, not future text."""
        result, responder, _ = await self.evaluate(
            vad=[vad_start(512), vad_stop(1536), vad_start(4096), vad_stop(6144)],
            asr=[(1024, 1024, "First question", True), (5120, 5120, "Second question", True)],
            responses=[response("First answer"), response("Second answer")],
        )
        self.assertEqual(
            responder.requests[1][0],
            [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second question"},
            ],
        )
        self.assertEqual(result.messages[-1], {"role": "assistant", "content": "Second answer"})

    async def test_interrupted_assistant_context_uses_captured_text_frame_queue_order(self):
        """Use actual queued text events without inferring words from PCM offsets."""
        trace = TTSResponseTrace(
            "Alpha Beta Gamma",
            (),
            samples_to_ns(200),
            frames=(
                CapturedTTSFrame(LLMFullResponseStartFrame(), 0),
                CapturedTTSFrame(OutputAudioRawFrame(pcm(160), RATE, 1), samples_to_ns(100)),
                CapturedTTSFrame(TTSTextFrame("Alpha", "sentence"), samples_to_ns(100)),
                CapturedTTSFrame(OutputAudioRawFrame(pcm(400), RATE, 1), samples_to_ns(200)),
                CapturedTTSFrame(TTSTextFrame("Beta", "sentence"), samples_to_ns(200)),
                CapturedTTSFrame(OutputAudioRawFrame(pcm(400), RATE, 1), samples_to_ns(200)),
                CapturedTTSFrame(TTSTextFrame("Gamma", "sentence"), samples_to_ns(200)),
                CapturedTTSFrame(TTSStoppedFrame(), samples_to_ns(200)),
                CapturedTTSFrame(LLMFullResponseEndFrame(), samples_to_ns(200)),
            ),
        )
        result, responder, _ = await self.evaluate(
            vad=[vad_start(512), vad_stop(1536), vad_start(2304), vad_stop(4096)],
            asr=[
                (1024, 1024, "First question", True),
                (1536, 1900, "Next", False),
                (2560, 2560, "Next question", True),
            ],
            responses=[
                trace,
                response("Second answer"),
            ],
        )
        self.assertEqual(
            responder.requests[1][0],
            [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "Alpha"},
                {"role": "user", "content": "Next question"},
            ],
        )
        self.assertNotIn({"role": "assistant", "content": "Alpha Beta Gamma"}, result.messages)

    async def test_native_22050_packing_preserves_cumulative_sample_count_without_rounding_drift(self):
        """Use online's 8800-sample chunks and resample through cumulative pointers."""
        trace = TTSResponseTrace(
            "Native answer",
            (),
            samples_to_ns(200),
            frames=(
                CapturedTTSFrame(LLMFullResponseStartFrame(), 0),
                CapturedTTSFrame(OutputAudioRawFrame(pcm(8800, 1), 22050, 1), samples_to_ns(100)),
                CapturedTTSFrame(OutputAudioRawFrame(pcm(8800, 2), 22050, 1), samples_to_ns(200)),
                CapturedTTSFrame(TTSTextFrame("Native answer", "sentence"), samples_to_ns(200)),
                CapturedTTSFrame(TTSStoppedFrame(), samples_to_ns(200)),
                CapturedTTSFrame(LLMFullResponseEndFrame(), samples_to_ns(200)),
            ),
        )
        result, _, _ = await self.evaluate(
            vad=[vad_start(512), vad_stop(1536)],
            asr=[(1024, 1024, "Question", True)],
            responses=[trace],
            chunk_units=40,
            output_rate=22050,
        )
        self.assertEqual([len(span.pcm) // 2 for span in result.spans], [6385, 6386])
        self.assertEqual(sum(len(span.pcm) // 2 for span in result.spans), round(17600 * RATE / 22050))
        self.assertEqual(result.spans[0].end_sample, result.spans[1].start_sample)
        self.assertEqual(result.messages[-1], {"role": "assistant", "content": "Native answer"})

    async def test_context_changes_from_cancelled_response_do_not_land_after_interruption(self):
        """Early provider changes commit; late callbacks from old responses cannot."""
        early = {"role": "developer", "content": "Early context change"}
        late = {"role": "developer", "content": "Late context change"}
        result, responder, _ = await self.evaluate(
            vad=[vad_start(512), vad_stop(1536), vad_start(2304), vad_stop(4096)],
            asr=[
                (1024, 1024, "First question", True),
                (1536, 1900, "Next", False),
                (2560, 2560, "Next question", True),
            ],
            responses=[
                response(
                    "Long answer",
                    audio_samples=2000,
                    compute=600,
                    metrics={
                        "context_events": [
                            {"ready_offset_ns": samples_to_ns(100), "message": early},
                            {"ready_offset_ns": samples_to_ns(500), "message": late},
                        ]
                    },
                ),
                response("Second answer"),
            ],
        )
        self.assertIn(early, responder.requests[1][0])
        self.assertNotIn(late, responder.requests[1][0])
        self.assertNotIn(late, result.messages)

    async def test_audio_output_starvation_is_silence_on_sample_timeline(self):
        """Separate provider arrivals preserve the gap between output chunks."""
        trace = ResponseTrace(
            "Answer",
            (GeneratedAudio(pcm(160, 1), samples_to_ns(100)), GeneratedAudio(pcm(160, 2), samples_to_ns(800))),
            samples_to_ns(1000),
        )
        result, _, _ = await self.evaluate(
            vad=[vad_start(512), vad_stop(1536)],
            asr=[(1024, 1024, "Question", True)],
            responses=[trace],
        )
        self.assertEqual([(span.start_sample, span.end_sample) for span in result.spans], [(1636, 1796), (2336, 2496)])
        rendered = render_pcm(result)
        self.assertEqual(rendered[1796 * 2 : 2336 * 2], bytes((2336 - 1796) * 2))

    async def test_future_user_audio_is_not_visible_in_current_provider_request(self):
        """Source-only ASR lookahead cannot leak transcripts into model history."""
        _, responder, _ = await self.evaluate(
            vad=[vad_start(512), vad_stop(1536), vad_start(4096), vad_stop(6144)],
            asr=[(1024, 1024, "First question", True), (5120, 5120, "Future secret", True)],
            responses=[response("First answer"), response("Second answer")],
        )
        self.assertEqual(responder.requests[0][0], [{"role": "user", "content": "First question"}])
        self.assertEqual(responder.requests[0][2], samples_to_ns(4096 - 1536))

    async def test_asr_drain_is_required_before_conversation_replay(self):
        """An incomplete stream must not silently turn into an accuracy result."""
        recognizer = FakeRecognizer(drained=False)
        responder = FakeResponder()
        engine = CausalSessionEngine(FakeDetector(), recognizer, responder)
        with self.assertRaisesRegex(ValueError, "drained"):
            await engine.run(AudioSource(bytes(2048)))
        self.assertEqual(responder.requests, [])


if __name__ == "__main__":
    unittest.main()
