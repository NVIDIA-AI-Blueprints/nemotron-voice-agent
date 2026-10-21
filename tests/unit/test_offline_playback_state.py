# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check live output speaking episodes independently from response completion."""

import unittest

from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    OutputAudioRawFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.transports.base_output import BOT_VAD_STOP_FALLBACK_SECS
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from test_offline_causal import FakeDetector, FakeRecognizer, FakeResponder, pcm, vad_start, vad_stop

from examples.generic.offline.causal import CausalSessionEngine
from examples.generic.offline.models import RATE, AudioSource, samples_to_ns
from examples.generic.offline.tts_trace import CapturedTTSFrame, TTSResponseTrace


def captured(frame, sample):
    """Use exact source-sample readiness for a real service frame."""
    return CapturedTTSFrame(frame, samples_to_ns(sample))


def trace(frames, compute, text="First Second"):
    """Provide the live TTS event shape without a socket or provider timer."""
    return TTSResponseTrace(text, (), samples_to_ns(compute), frames=tuple(frames))


class PlaybackStateTests(unittest.IsolatedAsyncioTestCase):
    """The transport can have several speaking episodes in one LLM response."""

    async def evaluate(self, responses, *, extra_vad=(), extra_asr=(), source_samples=8192):
        """Drive shared Pipecat aggregation and the actual output buffer."""
        responder = FakeResponder(responses)
        engine = CausalSessionEngine(
            FakeDetector([vad_start(512), vad_stop(1536), *extra_vad]),
            FakeRecognizer([(1024, 1024, "Question", True), *extra_asr]),
            responder,
            params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0)]
                ),
                empty_user_turn=None,
            ),
            output_chunks_10ms=1,
        )
        result = await engine.run(AudioSource(bytes(source_samples * 2)))
        return result, responder

    def episodes(self, result):
        """Expose only bot state transitions, keeping source and response IDs."""
        return [
            (event["type"], event["sample"], event["response_id"], event.get("reason"))
            for event in result.events
            if event["type"] in {"bot_speech_start", "bot_speech_stop"}
        ]

    async def test_tts_stop_then_later_audio_restarts_within_same_response(self):
        """An intermediate TTS stop must not complete the LLM generation."""
        result, responder = await self.evaluate(
            [
                trace(
                    [
                        captured(LLMFullResponseStartFrame(), 0),
                        captured(TTSAudioRawFrame(pcm(160), RATE, 1), 100),
                        captured(TTSTextFrame("First", "sentence"), 100),
                        captured(TTSStoppedFrame(), 200),
                        captured(TTSAudioRawFrame(pcm(160), RATE, 1), 1000),
                        captured(TTSTextFrame("Second", "sentence"), 1000),
                        captured(TTSStoppedFrame(), 1100),
                        captured(LLMFullResponseEndFrame(), 1100),
                    ],
                    1100,
                )
            ]
        )
        self.assertEqual(len(responder.requests), 1)
        self.assertEqual(
            self.episodes(result),
            [
                ("bot_speech_start", 1636, "response-1", None),
                ("bot_speech_stop", 1736, "response-1", "tts_stopped"),
                ("bot_speech_start", 2536, "response-1", None),
                ("bot_speech_stop", 2636, "response-1", "tts_stopped"),
            ],
        )
        self.assertEqual(result.metrics["responses"][0]["end_sample"], 2696)
        self.assertEqual(result.messages[-1], {"role": "assistant", "content": "First Second"})

    async def test_starvation_stops_after_last_output_item_then_later_audio_restarts(self):
        """Use the native WS write-return position, which can precede playout."""
        result, _ = await self.evaluate(
            [
                trace(
                    [
                        captured(TTSAudioRawFrame(pcm(160), RATE, 1), 100),
                        captured(TTSTextFrame("First", "sentence"), 100),
                        captured(TTSAudioRawFrame(pcm(160), RATE, 1), 50000),
                        captured(TTSTextFrame("Second", "sentence"), 50000),
                        captured(TTSStoppedFrame(), 50100),
                        captured(LLMFullResponseEndFrame(), 50200),
                    ],
                    50200,
                )
            ],
            source_samples=60000,
        )
        self.assertEqual(
            self.episodes(result),
            [
                ("bot_speech_start", 1636, "response-1", None),
                ("bot_speech_stop", 1636 + round(BOT_VAD_STOP_FALLBACK_SECS * RATE), "response-1", "output_starvation"),
                ("bot_speech_start", 51536, "response-1", None),
                ("bot_speech_stop", 51636, "response-1", "tts_stopped"),
            ],
        )
        self.assertEqual(len(result.metrics["responses"]), 1)

    async def test_text_control_frame_renews_starvation_wait(self):
        """The live sender waits for any queued frame, not only for PCM."""
        result, _ = await self.evaluate(
            [
                trace(
                    [
                        captured(TTSAudioRawFrame(pcm(160), RATE, 1), 100),
                        captured(TTSTextFrame("First", "sentence"), 20000),
                        captured(TTSAudioRawFrame(pcm(160), RATE, 1), 60000),
                        captured(TTSStoppedFrame(), 60200),
                        captured(LLMFullResponseEndFrame(), 60200),
                    ],
                    60200,
                )
            ],
            source_samples=65000,
        )
        self.assertEqual(
            self.episodes(result),
            [
                ("bot_speech_start", 1636, "response-1", None),
                ("bot_speech_stop", 61736, "response-1", "tts_stopped"),
            ],
        )

    async def test_no_audio_has_no_false_bot_start_or_stop(self):
        """Text-only or empty synthesis must not announce playback."""
        result, _ = await self.evaluate(
            [
                trace(
                    [
                        captured(TTSTextFrame("Text only", "sentence"), 100),
                        captured(TTSStoppedFrame(), 200),
                        captured(LLMFullResponseEndFrame(), 200),
                    ],
                    200,
                    text="Text only",
                )
            ]
        )
        self.assertEqual(self.episodes(result), [])
        self.assertEqual(result.spans, ())
        self.assertEqual(len(result.metrics["responses"]), 1)

    async def test_plain_output_audio_does_not_invent_tts_speaking_state(self):
        """Actual traces preserve the sender's distinction between audio types."""
        result, _ = await self.evaluate(
            [
                trace(
                    [
                        captured(OutputAudioRawFrame(pcm(160), RATE, 1), 100),
                        captured(TTSStoppedFrame(), 200),
                        captured(LLMFullResponseEndFrame(), 200),
                    ],
                    200,
                    text="",
                )
            ]
        )
        self.assertEqual(self.episodes(result), [])
        self.assertTrue(result.spans)

    async def test_interruption_discards_old_output_and_idle_events(self):
        """A stale generation cannot restart playback or stop its successor."""
        result, responder = await self.evaluate(
            [
                trace(
                    [
                        captured(TTSAudioRawFrame(pcm(160), RATE, 1), 100),
                        captured(TTSTextFrame("First", "sentence"), 100),
                        captured(TTSAudioRawFrame(pcm(160), RATE, 1), 50000),
                        captured(TTSStoppedFrame(), 50200),
                        captured(LLMFullResponseEndFrame(), 50200),
                    ],
                    50200,
                ),
                trace(
                    [
                        captured(TTSAudioRawFrame(pcm(50000), RATE, 1), 100),
                        captured(TTSTextFrame("Second", "sentence"), 100),
                        captured(TTSStoppedFrame(), 50200),
                        captured(LLMFullResponseEndFrame(), 50200),
                    ],
                    50200,
                ),
            ],
            extra_vad=[vad_start(4096), vad_stop(6144)],
            extra_asr=[(5120, 5120, "Next question", True)],
            source_samples=65000,
        )
        self.assertEqual(len(responder.requests), 2)
        self.assertEqual(
            self.episodes(result),
            [
                ("bot_speech_start", 1636, "response-1", None),
                ("bot_speech_stop", 4096, "response-1", "interruption"),
                ("bot_speech_start", 6244, "response-2", None),
                ("bot_speech_stop", 56344, "response-2", "tts_stopped"),
            ],
        )
        self.assertTrue(all(span.end_sample <= 4096 for span in result.spans if span.response_id == "response-1"))


if __name__ == "__main__":
    unittest.main()
