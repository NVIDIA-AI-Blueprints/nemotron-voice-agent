# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""What reads each line's playout after the output: a line's text goes on only once some of its audio played.

A sentence's text goes on only if audio came for it, which only the frames right after the text-to-speech tell.
"""

import unittest

from pipecat.frames.frames import (
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TextType,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.tests.utils import SleepFrame, run_test

from examples.frontend_backend_live.media.silent_lines import SILENT_KEY, SilentLines, SilentSentences
from examples.frontend_backend_live.spoken import Line, LineKind, SpokenLines
from examples.frontend_backend_live.talker import RESPONSE_KEY
from tests.unit.frontend_helpers import TransportOutput


class Settled:
    def __init__(self):
        self.events: list[str] = []

    def heard(self, line, cut):
        self.events.append("heard")

    def not_heard(self, line):
        self.events.append("not_heard")

    def say_again(self, line):
        self.events.append("again")


def response(frame):
    frame.metadata[RESPONSE_KEY] = 1
    return frame


def text(words):
    return TTSTextFrame(text=words, text_type=TextType.SENTENCE)


def audio():
    return TTSAudioRawFrame(audio=b"\x10\x10" * 2400, sample_rate=24000, num_channels=1)


class Output(TransportOutput):
    """Pipecat's own output transport, in 200 ms chunks, that writes each chunk at once."""

    async def write_audio_frame(self, frame) -> bool:
        return True


class SilentLinesTests(unittest.IsolatedAsyncioTestCase):
    async def play(self, *frames):
        """The frames that go on past the reader, and what the line did, for one line handed as response 1."""
        settled = Settled()
        spoken = SpokenLines(settled, observe=lambda *a, **k: None)
        spoken.handed(Line(LineKind.REPLY, "Hi. How can I help?", number=1))
        down, _ = await run_test(SilentLines(spoken), frames_to_send=frames)
        return [frame.text if isinstance(frame, TextFrame) else type(frame).__name__ for frame in down], settled.events

    async def speak(self, *frames):
        """The texts that go on past the reader, and what the line did, for response 1 as per-sentence speech gives it.

        Each sentence's audio comes before its text, as Pipecat's text-to-speech gives them. The frames pass what marks
        a silent sentence right after it, then Pipecat's own output, which sends audio in whole chunks of 200 ms, then
        the reader.
        """
        settled = Settled()
        spoken = SpokenLines(settled, observe=lambda *a, **k: None)
        spoken.handed(Line(LineKind.REPLY, "Hi. How can I help?", number=1))
        down, _ = await run_test(Pipeline([SilentSentences(), Output(), SilentLines(spoken)]), frames_to_send=frames)
        return [frame.text for frame in down if isinstance(frame, TTSTextFrame)], settled.events

    async def test_a_short_first_sentences_text_that_comes_out_just_before_its_audio_goes_on_right_after_it(self):
        frames = [
            response(LLMFullResponseStartFrame()),
            text("Hi."),  # the output's chunk was not full yet: its audio follows
            audio(),
            text("How can I help?"),
            response(LLMFullResponseEndFrame()),
        ]
        down, events = await self.play(*frames)
        played = ["LLMFullResponseStartFrame", "TTSAudioRawFrame", "Hi.", "How can I help?", "LLMFullResponseEndFrame"]
        self.assertEqual(down, played)
        self.assertEqual(events, ["heard"])

    async def test_a_text_no_audio_followed_goes_no_further_and_its_line_is_said_once_more(self):
        down, events = await self.play(
            response(LLMFullResponseStartFrame()), text("Hi."), response(LLMFullResponseEndFrame())
        )
        self.assertEqual(down, ["LLMFullResponseStartFrame", "LLMFullResponseEndFrame"])
        self.assertEqual(events, ["again"])

    async def test_a_cut_before_any_audio_drops_the_text_held_and_nobody_heard_the_line(self):
        frames = [response(LLMFullResponseStartFrame()), text("Hi."), SleepFrame(0.05), InterruptionFrame(), audio()]
        down, events = await self.play(*frames)
        self.assertEqual(down, ["LLMFullResponseStartFrame", "InterruptionFrame", "TTSAudioRawFrame"])
        self.assertEqual(events, ["not_heard"])

    async def test_a_sentence_that_got_no_audio_goes_no_further_while_the_one_after_it_is_heard(self):
        heard, events = await self.speak(
            response(LLMFullResponseStartFrame()),
            TTSStartedFrame(),
            text("Hi."),  # its request failed: no audio came for it
            audio(),
            audio(),
            audio(),
            text("How can I help?"),
            TTSStoppedFrame(),
            response(LLMFullResponseEndFrame()),
        )
        self.assertEqual(heard, ["How can I help?"])
        self.assertEqual(events, ["heard"])

    async def test_a_line_whose_sentences_all_got_no_audio_leaves_no_text_and_is_said_once_more(self):
        heard, events = await self.speak(
            response(LLMFullResponseStartFrame()),
            TTSStartedFrame(),
            text("Hi."),
            text("How can I help?"),
            TTSStoppedFrame(),
            response(LLMFullResponseEndFrame()),
        )
        self.assertEqual(heard, [])
        self.assertEqual(events, ["again"])

    async def test_a_short_first_sentence_is_heard_though_its_text_leaves_the_output_before_its_audio(self):
        heard, events = await self.speak(
            response(LLMFullResponseStartFrame()),
            TTSStartedFrame(),
            audio(),  # half of the output's chunk: the chunk goes out with the next sentence's audio, after this text
            text("Hi."),
            audio(),
            audio(),
            audio(),
            text("How can I help?"),
            TTSStoppedFrame(),
            response(LLMFullResponseEndFrame()),
        )
        self.assertEqual(heard, ["Hi.", "How can I help?"])
        self.assertEqual(events, ["heard"])

    async def test_a_line_that_played_in_part_is_heard_and_not_said_once_more(self):
        heard, events = await self.speak(
            response(LLMFullResponseStartFrame()),
            TTSStartedFrame(),
            audio(),
            audio(),
            text("Hi."),
            text("How can I help?"),  # its request failed: no audio came for it
            TTSStoppedFrame(),
            response(LLMFullResponseEndFrame()),
        )
        self.assertEqual(heard, ["Hi."])
        self.assertEqual(events, ["heard"])

    async def test_a_cut_lines_audio_does_not_count_for_the_next_lines_first_sentence(self):
        frames = [
            response(LLMFullResponseStartFrame()),
            TTSStartedFrame(),
            audio(),  # the line is cut while its first sentence is synthesized: its text never comes
            SleepFrame(0.05),
            InterruptionFrame(),
            response(LLMFullResponseStartFrame()),
            TTSStartedFrame(),
            text("Hi."),  # its request failed: no audio came for it
            audio(),
            text("How can I help?"),
        ]
        down, _ = await run_test(SilentSentences(), frames_to_send=frames)
        silent = [frame.text for frame in down if isinstance(frame, TTSTextFrame) and SILENT_KEY in frame.metadata]
        self.assertEqual(silent, ["Hi."])


if __name__ == "__main__":
    unittest.main()
