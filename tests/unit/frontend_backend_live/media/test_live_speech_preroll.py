# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103

"""The speech services, and a reply's speech on its way out: a stitched line's text hold, and the pre-roll."""

import unittest
from unittest.mock import patch

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    InterruptionFrame,
    TextType,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.stt import NvidiaSTTService
from pipecat.services.stt_latency import NVIDIA_TTFS_P99
from pipecat.tests.utils import SleepFrame, run_test

from examples.frontend_backend_live.media.preroll import SpeechPreroll
from examples.frontend_backend_live.media.services import LineTextHold, build_stt, build_tts
from examples.shared.nemotron_speech_text_filter import nemotron_speech_markdown_text_transform

MAGPIE = {"server": "magpie-tts:50051", "synthesis_mode": "per_sentence"}  # an entry as the registry's settings give it
NEMOTRON_ASR = {"server": "nemotron-asr:50052"}


def audio(ms: int) -> TTSAudioRawFrame:
    return TTSAudioRawFrame(audio=b"\0\0" * (24 * ms), sample_rate=24000, num_channels=1)


def word(text: str) -> TTSTextFrame:
    return TTSTextFrame(text=text, text_type=TextType.WORD)


def preroll(ms: int) -> tuple[SpeechPreroll, list]:
    """A pre-roll of ``ms`` and the frames it passes on."""
    processor, sent = SpeechPreroll(ms), []

    async def record(frame, direction=FrameDirection.DOWNSTREAM):
        sent.append(frame)

    processor.push_frame = record
    return processor, sent


class SpeechTests(unittest.IsolatedAsyncioTestCase):
    def test_the_speech_to_text_is_pipecats_nvidia_service_with_its_own_budget_for_the_final(self):
        stt = build_stt({}, NEMOTRON_ASR)  # as the other examples build it
        self.assertIsInstance(stt, NvidiaSTTService)
        self.assertEqual(stt.service_metadata_frame().ttfs_p99_latency, NVIDIA_TTFS_P99)

    def test_the_speech_to_text_finalizes_a_stretch_of_speech_after_320_ms_of_silence(self):
        with patch("examples.frontend_backend_live.media.services.NvidiaSTTService") as service:
            build_stt({}, NEMOTRON_ASR)
        self.assertEqual(service.call_args.kwargs["stop_history"], 320)  # as the other examples

    def test_the_speech_to_text_punctuates_as_the_session_asks_else_as_its_entry_says(self):
        def punctuates(body: dict, entry: dict) -> bool:
            with patch("examples.frontend_backend_live.media.services.NvidiaSTTService") as service:
                build_stt(body, entry)
            return service.call_args.kwargs["settings"].automatic_punctuation

        unpunctuated = {**NEMOTRON_ASR, "automatic_punctuation": False}
        self.assertTrue(punctuates({}, NEMOTRON_ASR))
        self.assertFalse(punctuates({}, unpunctuated))  # a Live session's body names no punctuation
        self.assertTrue(punctuates({"asr_automatic_punctuation": "True"}, unpunctuated))
        self.assertFalse(punctuates({"asr_automatic_punctuation": "False"}, NEMOTRON_ASR))

    def test_the_text_to_speech_speaks_the_sessions_language_else_its_entrys(self):
        self.assertEqual(build_tts({"tts_language_code": "DE-de"}, MAGPIE).settings.language, "de-DE")
        self.assertEqual(build_tts({}, {**MAGPIE, "language_code": "fr-FR"}).settings.language, "fr-FR")

    def test_the_text_to_speech_keeps_pipecats_limit_of_contexts_with_no_audio(self):
        with patch("examples.frontend_backend_live.media.services.NvidiaTTSService") as service:
            build_tts({}, MAGPIE)
        self.assertNotIn("max_consecutive_zero_audio_contexts", service.call_args.kwargs)  # as the other examples

    def test_the_text_to_speech_strips_markdown_with_the_shared_transform(self):
        with patch("examples.frontend_backend_live.media.services.NvidiaTTSService") as service:
            build_tts({}, MAGPIE)
        self.assertEqual(service.call_args.kwargs["text_transforms"], [("*", nemotron_speech_markdown_text_transform)])

    async def test_a_stitched_lines_text_follows_all_of_its_audio_and_a_cut_drops_it(self):
        one, two = (TTSTextFrame(text=text, text_type=TextType.SENTENCE) for text in ("One.", "Two."))
        line = [TTSStartedFrame(), one, audio(100), two, audio(100), TTSStoppedFrame()]
        down, _ = await run_test(LineTextHold(), frames_to_send=line)
        self.assertEqual(
            [type(frame) for frame in down],
            [TTSStartedFrame, *[TTSAudioRawFrame] * 2, *[TTSTextFrame] * 2, TTSStoppedFrame],
        )
        cut = [
            TTSStartedFrame(),
            one,
            audio(100),
            SleepFrame(0.05),
            InterruptionFrame(),
            TTSStartedFrame(),
            two,
            TTSStoppedFrame(),
        ]
        down, _ = await run_test(LineTextHold(), frames_to_send=cut)
        self.assertEqual([frame.text for frame in down if isinstance(frame, TTSTextFrame)], ["Two."])


class SpeechPrerollTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_reply_waits_until_enough_audio_is_ready_then_passes_straight_through(self):
        processor, sent = preroll(300)
        first = [TTSStartedFrame(), audio(100), word("Hello"), audio(100)]
        for frame in first:
            await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(sent, [])
        for frame in (third := audio(100), later := audio(40)):
            await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(sent, [*first, third, later])

    async def test_a_reply_shorter_than_the_preroll_plays_when_it_ends_and_zero_holds_nothing(self):
        for ms in (300, 0):
            with self.subTest(ms=ms):
                processor, sent = preroll(ms)
                frames = [TTSStartedFrame(), audio(120), word("Okay."), TTSStoppedFrame()]
                for frame in frames:
                    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
                self.assertEqual(sent, frames)

    async def test_system_and_upstream_frames_are_never_held(self):
        processor, sent = preroll(300)
        system, upstream = BotStartedSpeakingFrame(), word("up")
        for frame in (TTSStartedFrame(), audio(100), system):
            await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
        await processor.process_frame(upstream, FrameDirection.UPSTREAM)
        self.assertEqual(sent, [system, upstream])

    async def test_frames_keep_their_order_and_an_interruption_drops_the_held_audio(self):
        frames = [TTSStartedFrame(), audio(100), word("One"), audio(250), word("two"), audio(50), TTSStoppedFrame()]
        down, _ = await run_test(
            SpeechPreroll(300), frames_to_send=frames, expected_down_frames=[type(f) for f in frames]
        )
        self.assertEqual([frame.text for frame in down if isinstance(frame, TTSTextFrame)], ["One", "two"])
        # System frames overtake queued ones, so the held audio comes first; the pipeline's end releases the next.
        dropped, kept = audio(100), audio(60)
        frames = [TTSStartedFrame(), dropped, SleepFrame(0.05), InterruptionFrame(), TTSStartedFrame(), kept]
        expected = [InterruptionFrame, TTSStartedFrame, TTSAudioRawFrame]
        down, _ = await run_test(SpeechPreroll(300), frames_to_send=frames, expected_down_frames=expected)
        self.assertIs(down[-1], kept)
