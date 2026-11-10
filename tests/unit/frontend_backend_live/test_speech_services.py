# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The speech services, and a reply's speech on its way out: a stitched line's text hold, and the pre-roll."""

import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipecat.frames.frames import (
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextType,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings
from pipecat.tests.utils import SleepFrame, run_test

from examples.frontend_backend_live.media.preroll import SpeechPreroll
from examples.frontend_backend_live.media.services import LineTextHold, build_stt, build_tts
from examples.shared.nemotron_speech_text_filter import nemotron_speech_markdown_text_transform

MAGPIE = {"server": "magpie-tts:50051", "synthesis_mode": "per_sentence"}
NEMOTRON_ASR = {"server": "nemotron-asr:50052"}
SERVICES = "examples.frontend_backend_live.media.services"


def audio(ms: int) -> TTSAudioRawFrame:
    return TTSAudioRawFrame(audio=b"\0\0" * (24 * ms), sample_rate=24000, num_channels=1)


def word(text: str) -> TTSTextFrame:
    return TTSTextFrame(text=text, text_type=TextType.WORD)


def late_stitched_stream(requests, late: float):
    """Stub Magpie's stitched stream: 100 ms of audio for each text it is sent, the first ``late`` seconds late."""
    texts: queue.Queue = queue.Queue()

    def read() -> None:  # Pipecat's request generator waits for the reply's next text, so a daemon thread reads it
        for request in requests:
            texts.put(request.text)
        texts.put(None)

    threading.Thread(target=read, daemon=True).start()
    time.sleep(late)
    while texts.get(timeout=5) is not None:
        yield SimpleNamespace(audio=b"\0\0" * 2400)


def late_stitched_tts(late: float, stop_timeout: float) -> NvidiaTTSService:
    """Build Pipecat's stitched NVIDIA text-to-speech on a stub server whose first audio is ``late`` seconds late."""
    tts = NvidiaTTSService(
        server="magpie-tts:50051",
        use_ssl=False,
        settings=NvidiaTTSSettings(synthesis_mode="stitched"),
        stop_frame_timeout_s=stop_timeout,
    )
    stub = SimpleNamespace(
        GetRivaSynthesisConfig=lambda request: None,
        SynthesizeOnline=lambda requests, metadata: late_stitched_stream(requests, late),
    )
    tts._service = SimpleNamespace(stub=stub, auth=SimpleNamespace(get_auth_metadata=lambda: ()))
    return tts


async def preroll(ms: int, frames, upstream=()) -> list:
    """Return what a pre-roll of ``ms`` passes on of ``frames``, then of ``upstream`` going up, one by one."""
    processor, sent = SpeechPreroll(ms), []

    async def record(frame, direction=FrameDirection.DOWNSTREAM):
        sent.append(frame)

    processor.push_frame = record
    for frame in frames:
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    for frame in upstream:
        await processor.process_frame(frame, FrameDirection.UPSTREAM)
    return sent


class SpeechTests(unittest.IsolatedAsyncioTestCase):
    def test_the_speech_to_text_punctuates_as_the_session_asks(self):
        def built(body: dict, entry: dict) -> dict:
            with patch(f"{SERVICES}.NvidiaSTTService") as service:
                build_stt(body, entry)
            return service.call_args.kwargs

        self.assertEqual(built({}, NEMOTRON_ASR)["stop_history"], 320)
        unpunctuated = {**NEMOTRON_ASR, "automatic_punctuation": False}
        for body, entry, punctuates in (
            ({}, NEMOTRON_ASR, True),
            ({}, unpunctuated, False),
            ({"asr_automatic_punctuation": "True"}, unpunctuated, True),
            ({"asr_automatic_punctuation": "False"}, NEMOTRON_ASR, False),
        ):
            self.assertEqual(built(body, entry)["settings"].automatic_punctuation, punctuates, (body, entry))

    def test_the_text_to_speech_speaks_the_language_with_no_markdown(self):
        self.assertEqual(build_tts({"tts_language_code": "DE-de"}, MAGPIE).settings.language, "de-DE")
        self.assertEqual(build_tts({}, {**MAGPIE, "language_code": "fr-FR"}).settings.language, "fr-FR")
        with patch(f"{SERVICES}.NvidiaTTSService") as service:
            build_tts({}, MAGPIE)
        built = service.call_args.kwargs
        self.assertNotIn("max_consecutive_zero_audio_contexts", built)
        self.assertEqual(built["text_transforms"], [("*", nemotron_speech_markdown_text_transform)])

    async def test_a_stitched_lines_text_follows_its_audio(self):
        one, two = (TTSTextFrame(text=text, text_type=TextType.SENTENCE) for text in ("One.", "Two."))
        line = [TTSStartedFrame(), one, audio(100), two, audio(100), TTSStoppedFrame()]
        down, _ = await run_test(LineTextHold(), frames_to_send=line)
        expected = [TTSStartedFrame, *[TTSAudioRawFrame] * 2, *[TTSTextFrame] * 2, TTSStoppedFrame]
        self.assertEqual([type(frame) for frame in down], expected)
        cut = [TTSStartedFrame(), one, audio(100), SleepFrame(0.05), InterruptionFrame(), TTSStartedFrame(), two]
        down, _ = await run_test(LineTextHold(), frames_to_send=[*cut, TTSStoppedFrame()])
        self.assertEqual([frame.text for frame in down if isinstance(frame, TTSTextFrame)], ["Two."])

    async def test_a_reopened_stitched_line_lets_its_text_out_at_its_end(self):
        # The first audio comes at 0.6 s, after the 0.3 s stop timeout: Pipecat reopens the audio context with no
        # TTSStartedFrame, and the next sentence's text goes into it at 0.75 s, before its own stop timeout.
        reply = [LLMFullResponseStartFrame(), LLMTextFrame("One."), LLMTextFrame(" Two"), SleepFrame(0.75)]
        reply += [LLMTextFrame("."), LLMFullResponseEndFrame(), SleepFrame(0.2)]
        tts = late_stitched_tts(late=0.6, stop_timeout=0.3)
        down, _ = await run_test(Pipeline([tts, LineTextHold()]), frames_to_send=reply)
        audio = [i for i, frame in enumerate(down) if isinstance(frame, TTSAudioRawFrame)]
        stop = next(i for i, frame in enumerate(down) if isinstance(frame, TTSStoppedFrame))
        self.assertLess(stop, audio[0])
        self.assertEqual(sum(isinstance(frame, TTSStartedFrame) for frame in down), 1)
        end = next(i for i, frame in enumerate(down) if isinstance(frame, LLMFullResponseEndFrame))
        texts = [(i, frame.text) for i, frame in enumerate(down) if isinstance(frame, TTSTextFrame)]
        self.assertEqual([text for i, text in texts if i < end], ["One.", "Two."])
        self.assertGreater(texts[-1][0], audio[-1])


class SpeechPrerollTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_reply_waits_for_enough_audio_then_passes_through(self):
        first = [TTSStartedFrame(), audio(100), word("Hello"), audio(100)]
        self.assertEqual(await preroll(300, first), [])
        up = word("up")
        self.assertEqual(await preroll(300, first, upstream=[up]), [up])
        rest = [audio(100), audio(40)]
        self.assertEqual(await preroll(300, [*first, *rest]), [*first, *rest])
        for ms in (300, 0):
            short = [TTSStartedFrame(), audio(120), word("Okay."), TTSStoppedFrame()]
            self.assertEqual(await preroll(ms, short), short)

    async def test_frames_keep_their_order_and_a_cut_drops_held_audio(self):
        frames = [TTSStartedFrame(), audio(100), word("One"), audio(250), word("two"), audio(50), TTSStoppedFrame()]
        expected = [type(frame) for frame in frames]
        down, _ = await run_test(SpeechPreroll(300), frames_to_send=frames, expected_down_frames=expected)
        self.assertEqual([frame.text for frame in down if isinstance(frame, TTSTextFrame)], ["One", "two"])
        # System frames overtake queued ones, so the held audio comes first; the pipeline's end releases the next.
        dropped, kept = audio(100), audio(60)
        frames = [TTSStartedFrame(), dropped, SleepFrame(0.05), InterruptionFrame(), TTSStartedFrame(), kept]
        expected = [InterruptionFrame, TTSStartedFrame, TTSAudioRawFrame]
        down, _ = await run_test(SpeechPreroll(300), frames_to_send=frames, expected_down_frames=expected)
        self.assertIs(down[-1], kept)
