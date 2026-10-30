# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the observed case 3 pause using captured processing checkpoints."""

import unittest

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from test_offline_recognition import ManualClock, response, result

from examples.generic.offline.aggregation import ChunkPositionedFrame, SharedUserAggregation
from examples.generic.offline.models import AudioSource
from examples.generic.offline.recognition import ChunkProgressASR

# Actual case 3 response arrivals / progress around its first VAD stop. The
# terminal prefix is sufficient to reproduce the bug; charging its first entire
# observed interval as startup is deliberately conservative. Text is shortened;
# no linguistic timestamps or model calls are used in this regression.
CHECKPOINTS = (
    (225280, 949834943, 239436, "First premise", False),
    (227840, 960247808, 242152, "First premise", True),
    (230400, 970966069, 244872, "Southam", False),
    (232960, 981568511, 247584, "Southampt", False),
    (235520, 992073984, 250300, "Southampton", False),
    (238080, 1002607107, 253018, "Southampton F", False),
    (240640, 1013207149, 255737, "Southampton FC", False),
    (243200, 1023698332, 258456, "Southampton FC", False),
    (245760, 1034140326, 261170, "Southampton FC", True),
)
VAD_STOP = 246272  # 15,392 ms
NEXT_VAD_START = 252928  # 15,808 ms


class RealPauseProcessingClockTests(unittest.IsolatedAsyncioTestCase):
    """The shared turn policy must not inherit cumulative bulk upload backlog."""

    async def contexts_before_next_speech(self, events):
        """Replay the recorded pause through the actual shared turn policy."""
        aggregator = SharedUserAggregation(
            params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0)]
                ),
                empty_user_turn=None,
            )
        )
        try:
            await aggregator.feed(ChunkPositionedFrame(0, STTMetadataFrame("NvidiaSTTService", 1.0)))
            await aggregator.feed(ChunkPositionedFrame(13824, VADUserStartedSpeakingFrame(start_secs=0.2)))
            frames = [
                ChunkPositionedFrame(
                    sample,
                    TranscriptionFrame(text, "", "", finalized=True)
                    if final
                    else InterimTranscriptionFrame(text, "", ""),
                )
                for sample, text, final in events
                if sample <= NEXT_VAD_START
            ]
            frames.extend(
                [
                    ChunkPositionedFrame(VAD_STOP, VADUserStoppedSpeakingFrame(stop_secs=0.5)),
                    ChunkPositionedFrame(NEXT_VAD_START, VADUserStartedSpeakingFrame(start_secs=0.2)),
                ]
            )
            for frame in sorted(frames, key=lambda item: item.sample):
                await aggregator.feed(frame)
            return list(aggregator.contexts)
        finally:
            await aggregator.close()

    async def test_observed_incremental_service_profile_finalizes_before_the_next_speech(self):
        """The 15,360 ms prefix final must precede the 15,392 ms VAD stop."""
        clock = ManualClock()

        def stream(audio):
            list(audio)
            for processed, arrival, _, text, final in CHECKPOINTS:
                clock.now = arrival
                yield response(result(processed, text, final=final))

        trace = await ChunkProgressASR(stream, clock=clock).recognize(AudioSource(bytes(270336 * 2)))
        final = trace.events[-1]
        self.assertEqual(final.processed_sample, 245760)
        self.assertEqual(final.processing_ns, 1034140326 - 1023698332)
        self.assertLess(final.available_sample, VAD_STOP)
        self.assertTrue(trace.metrics["asr_rpc_eof_drained"])
        contexts = await self.contexts_before_next_speech(
            [(event.available_sample, event.text, event.final) for event in trace.events]
        )
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0][0], VAD_STOP)
        self.assertEqual(contexts[0][1], [{"role": "user", "content": "First premise Southampton FC"}])

    async def test_prior_cumulative_residual_projection_misses_the_same_real_pause(self):
        """The saved projection postpones this final until 16,323.125 ms."""
        contexts = await self.contexts_before_next_speech(
            [(old_available, text, final) for _, _, old_available, text, final in CHECKPOINTS]
        )
        self.assertGreater(CHECKPOINTS[-1][2], NEXT_VAD_START)
        self.assertEqual(contexts, [])


if __name__ == "__main__":
    unittest.main()
