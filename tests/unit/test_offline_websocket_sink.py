# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare virtual WS pacing against the unmodified native queue consumer."""

import asyncio
import unittest
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch

from pipecat.frames.frames import (
    EndFrame,
    LLMFullResponseEndFrame,
    OutputAudioRawFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.websocket import fastapi as native_ws
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from test_offline_causal import FakeDetector, FakeRecognizer, FakeResponder, pcm, response, vad_start, vad_stop

from examples.generic.offline.causal import CausalSessionEngine
from examples.generic.offline.models import RATE, AudioSource, samples_to_ns
from examples.generic.offline.playback import WebsocketSinkClock, pack_frames
from examples.generic.offline.tts_trace import CapturedTTSFrame, TTSResponseTrace


class WebsocketSinkTests(unittest.IsolatedAsyncioTestCase):
    """Socket completion and client presentation have separate media positions."""

    async def native_consumer(self, frames):
        """Run native MediaSender and WS writes against a deterministic clock."""
        now = [100.0]
        events = []
        real_sleep = asyncio.sleep

        async def fake_sleep(delay):
            now[0] += delay
            await real_sleep(0)

        class Wire:
            sample_rate = RATE
            is_usable = True
            _client = SimpleNamespace(is_closing=False, is_connected=True)
            _params = SimpleNamespace(audio_out_channels=1, add_wav_header=False)
            _send_interval = 0.4
            _next_send_time = 0

            async def _write_frame(self, frame):
                events.append(("wire", type(frame).__name__, round((now[0] - 100) * 1_000_000_000)))
                return True

            async def _write_audio_sleep(self):
                await native_ws.FastAPIWebsocketOutputTransport._write_audio_sleep(self)

            async def write_audio_frame(self, frame):
                return await native_ws.FastAPIWebsocketOutputTransport.write_audio_frame(self, frame)

            async def push_frame(self, frame, *args):
                events.append(("forward", type(frame).__name__, round((now[0] - 100) * 1_000_000_000)))

            async def write_transport_frame(self, frame):
                pass

            async def broadcast_frame(self, *args):
                pass

        sender = BaseOutputTransport.MediaSender(
            Wire(),
            destination=None,
            sample_rate=RATE,
            audio_chunk_size=12800,
            params=TransportParams(audio_out_enabled=True, audio_out_end_silence_secs=0),
        )
        sender._audio_queue = asyncio.Queue()
        for frame in [*frames, EndFrame()]:
            await sender._audio_queue.put(frame)
        try:
            # Patch the WS module references, leaving asyncio's real timeout
            # and event-loop clock untouched.
            with (
                patch.object(native_ws, "time", SimpleNamespace(monotonic=lambda: now[0])),
                patch.object(native_ws, "asyncio", SimpleNamespace(sleep=fake_sleep)),
            ):
                await sender._audio_task_handler()
        finally:
            sender._executor.shutdown(wait=False)
        return events

    async def test_native_two_queued_chunks_then_text_and_stop(self):
        """Native WS writes two chunks immediately and forwards text at 400 ms."""
        frames = [
            TTSAudioRawFrame(pcm(6400), RATE, 1),
            TTSAudioRawFrame(pcm(6400), RATE, 1),
            TTSTextFrame("Answer", "sentence"),
            TTSStoppedFrame(),
            LLMFullResponseEndFrame(),
        ]
        events = await self.native_consumer(frames)
        trace = TTSResponseTrace("Answer", (), 0, frames=tuple(CapturedTTSFrame(f, 0) for f in frames))
        packed = await pack_frames(trace, 40)
        self.assertEqual([i.delivery_offset_ns for i in packed], [0, 0, 400_000_000, 400_000_000, 400_000_000])
        self.assertEqual(
            [i.write_complete_offset_ns for i in packed], [0, 400_000_000, 400_000_000, 400_000_000, 400_000_000]
        )
        self.assertEqual([at for kind, name, at in events if kind == "wire" and name == "OutputAudioRawFrame"], [0, 0])
        self.assertEqual(
            [
                (name, at)
                for kind, name, at in events
                if kind == "forward" and name in {"TTSAudioRawFrame", "TTSTextFrame", "TTSStoppedFrame"}
            ],
            [
                (type(i.frame).__name__, i.write_complete_offset_ns)
                for i in packed
                if isinstance(i.frame, (OutputAudioRawFrame, TTSTextFrame, TTSStoppedFrame))
            ],
        )

    async def test_control_after_first_chunk_does_not_wait_for_its_playout(self):
        """The native first zero-duration sleep must not postpone text."""
        frames = [TTSAudioRawFrame(pcm(6400), RATE, 1), TTSTextFrame("First", "sentence"), TTSStoppedFrame()]
        events = await self.native_consumer(frames)
        self.assertIn(("forward", "TTSTextFrame", 0), events)
        trace = TTSResponseTrace("First", (), 0, frames=tuple(CapturedTTSFrame(f, 0) for f in frames))
        packed = await pack_frames(trace, 40)
        self.assertEqual([i.delivery_offset_ns for i in packed], [0, 0, 0])

    def test_interruption_resets_pacing_but_normal_stop_preserves_phase(self):
        """The WS instance retains _next_send_time until interruption."""
        clock = WebsocketSinkClock()
        interval = Fraction(400_000_000)
        self.assertEqual(clock.write_complete(100, interval), 100)
        self.assertEqual(clock.write_complete(200, interval), 400_000_100)
        self.assertEqual(clock.write_complete(500_000_000, interval), 800_000_100)
        clock.interrupt()
        self.assertEqual(clock.write_complete(500_000_001, interval), 500_000_001)

    async def test_barge_in_between_control_delivery_and_buffered_pcm_end_keeps_delivered_text(self):
        """Assistant history follows native text delivery, while PCM is clipped."""
        frames = [
            TTSAudioRawFrame(pcm(12800), RATE, 1),
            TTSTextFrame("Complete first answer", "sentence"),
            TTSStoppedFrame(),
            LLMFullResponseEndFrame(),
        ]
        trace = TTSResponseTrace(
            "Complete first answer",
            (),
            samples_to_ns(100),
            frames=tuple(CapturedTTSFrame(f, samples_to_ns(100)) for f in frames),
        )
        responder = FakeResponder([trace, response("Next answer")])
        engine = CausalSessionEngine(
            FakeDetector([vad_start(512), vad_stop(1536), vad_start(11264), vad_stop(14336)]),
            FakeRecognizer([(1024, 1024, "First question", True), (12288, 12288, "Next question", True)]),
            responder,
            params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0)]
                ),
                empty_user_turn=None,
            ),
            output_chunks_10ms=40,
        )
        result = await engine.run(AudioSource(bytes(32768 * 2)))
        self.assertIn({"role": "assistant", "content": "Complete first answer"}, responder.requests[1][0])
        spans = [s for s in result.spans if s.response_id == "response-1"]
        self.assertEqual(spans[0].start_sample, 1636)
        self.assertEqual(spans[0].end_sample, spans[1].start_sample)
        self.assertEqual(spans[-1].end_sample, 11264)
        stops = [e for e in result.events if e["type"] == "bot_speech_stop" and e["response_id"] == "response-1"]
        self.assertEqual(stops[0]["sample"], 8036)


if __name__ == "__main__":
    unittest.main()
