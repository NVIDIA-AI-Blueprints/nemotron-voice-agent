# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""What reads each line's playout after the output: a sentence's text goes on only if audio came for it."""

import unittest

from pipecat.frames.frames import (
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
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


class Output(TransportOutput):
    """Pipecat's own output transport, in 200 ms chunks, that writes each chunk at once."""

    async def write_audio_frame(self, frame) -> bool:
        return True


def response(frame):
    frame.metadata[RESPONSE_KEY] = 1
    return frame


def text(words):
    return TTSTextFrame(text=words, text_type=TextType.SENTENCE)


def audio():
    return TTSAudioRawFrame(audio=b"\x10\x10" * 2400, sample_rate=24000, num_channels=1)


async def speak(*sentences):
    """Return the texts that pass the reader and what line 1 did, each sentence's audio coming before its text."""
    settled = Settled()
    spoken = SpokenLines(settled, observe=lambda *a, **k: None)
    spoken.handed(Line(LineKind.REPLY, "Hi. How can I help?", number=1))
    line = [response(LLMFullResponseStartFrame()), TTSStartedFrame(), *sentences, TTSStoppedFrame()]
    pipeline = Pipeline([SilentSentences(), Output(), SilentLines(spoken)])
    down, _ = await run_test(pipeline, frames_to_send=[*line, response(LLMFullResponseEndFrame())])
    return [frame.text for frame in down if isinstance(frame, TTSTextFrame)], settled.events


class SilentLinesTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_a_sentence_that_got_audio_passes_its_text_on(self):
        for name, (sentences, outcome) in {
            "a sentence with no audio goes no further; the next is heard": (
                [text("Hi."), audio(), audio(), audio(), text("How can I help?")],
                (["How can I help?"], ["heard"]),
            ),
            "a line whose sentences all got no audio is said once more": (
                [text("Hi."), text("How can I help?")],
                ([], ["again"]),
            ),
            # The first audio is half of the output's chunk: it goes out with the next sentence's, after "Hi.".
            "a short first sentence is heard though its text leaves first": (
                [audio(), text("Hi."), audio(), audio(), audio(), text("How can I help?")],
                (["Hi.", "How can I help?"], ["heard"]),
            ),
            "a line that played in part is heard and not said once more": (
                [audio(), audio(), text("Hi."), text("How can I help?")],
                (["Hi."], ["heard"]),
            ),
        }.items():
            with self.subTest(name):
                self.assertEqual(await speak(*sentences), outcome)

    async def test_a_cut_lines_audio_never_counts_for_the_next_line(self):
        frames = [
            response(LLMFullResponseStartFrame()),
            TTSStartedFrame(),
            audio(),  # the line is cut while its first sentence is synthesized: its text never comes
            SleepFrame(0.05),
            InterruptionFrame(),
            response(LLMFullResponseStartFrame()),
            TTSStartedFrame(),
            text("Hi."),
            audio(),
            text("How can I help?"),
        ]
        down, _ = await run_test(SilentSentences(), frames_to_send=frames)
        silent = [frame.text for frame in down if isinstance(frame, TTSTextFrame) and SILENT_KEY in frame.metadata]
        self.assertEqual(silent, ["Hi."])
