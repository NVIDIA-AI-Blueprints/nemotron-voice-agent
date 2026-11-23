# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""What reads each line's playout after the output: a sentence's text goes on only if audio came for it."""

import asyncio
import unittest

from pipecat.frames.frames import (
    EndWorkerFrame,
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
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.tests.utils import SleepFrame, run_test

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.media.silent_lines import (
    SILENT_KEY,
    SentenceAudioPositions,
    SilentLines,
    SilentSentences,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.spoken import Line, LineKind, SpokenLines
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.talker import RESPONSE_KEY
from tests.unit.pipecat.frontend_helpers import TransportOutput


class Settled:
    def __init__(self):
        self.events: list[str | tuple[str, str]] = []

    def heard(self, line):
        self.events.append(("heard", line.heard))

    def not_heard(self, line, *, audio_exhausted=False):
        self.events.append("not_heard")

    def say_again(self, line):
        self.events.append("again")


class Output(TransportOutput):
    """Pipecat's own output transport, set to write each chunk at once."""

    async def write_audio_frame(self, frame) -> bool:
        return True


def response(frame):
    frame.metadata[RESPONSE_KEY] = 1
    return frame


def text(words):
    return TTSTextFrame(text=words, text_type=TextType.SENTENCE)


def audio(ms=100):
    return TTSAudioRawFrame(audio=b"\x10\x10" * (24 * ms), sample_rate=24000, num_channels=1)


async def speak(*sentences, chunk_ms=200, cut=False):
    """Return the playout receipt using Pipecat's actual recorded transcript."""
    settled = Settled()
    spoken = SpokenLines(settled, observe=lambda *a, **k: None)
    spoken.handed(Line(LineKind.REPLY, "Hi. How can I help?", number=1))
    line = [response(LLMFullResponseStartFrame()), TTSStartedFrame(), *sentences]
    assistant = LLMAssistantAggregator(LLMContext())
    interruption = InterruptionFrame()
    output, played = Output(chunk_ms=chunk_ms), SilentLines(spoken)

    @assistant.event_handler("on_assistant_turn_stopped")
    async def settle_playout(_, message):
        if message.interrupted:
            spoken.cut_played(interruption.id, message.content)
        else:
            spoken.end_played(1, message.content)

    @played.event_handler("on_after_process_frame")
    async def cut_after_marker(_, frame):
        if cut and frame is sentences[-1]:
            await output.queue_frame(interruption)
            await output.queue_frame(EndWorkerFrame())

    pipeline = Pipeline([SilentSentences(), SentenceAudioPositions(), output, played, assistant])
    if not cut:
        line.extend([TTSStoppedFrame(), response(LLMFullResponseEndFrame())])
    async with asyncio.timeout(5):
        await run_test(pipeline, frames_to_send=line, send_end_frame=not cut)
    return settled.events


class SilentLinesTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_a_sentence_that_got_audio_passes_its_text_on(self):
        for name, (sentences, outcome) in {
            "a sentence with no audio goes no further; the next is heard": (
                [text("Hi."), audio(), audio(), audio(), text("How can I help?")],
                [("heard", "How can I help?")],
            ),
            "a line whose sentences all got no audio is said once more": (
                [text("Hi."), text("How can I help?")],
                ["again"],
            ),
            # The first audio is half of the output's chunk: it goes out with the next sentence's, after "Hi.".
            "a short first sentence is heard though its text leaves first": (
                [audio(), text("Hi."), audio(), audio(), audio(), text("How can I help?")],
                [("heard", "Hi. How can I help?")],
            ),
            "a line that played in part is heard and not said once more": (
                [audio(), audio(), text("Hi."), text("How can I help?")],
                [("heard", "Hi.")],
            ),
            "audio alone supplies no words": (
                [audio(), audio()],
                [("heard", "")],
            ),
        }.items():
            with self.subTest(name):
                self.assertEqual(await speak(*sentences), outcome)

        for chunk_ms in (50, 100):
            for cut in (True, False):
                with self.subTest(chunk_ms=chunk_ms, cut=cut):
                    outcome = await speak(
                        audio(chunk_ms), text("Hi."), audio(30), text("How can I help?"), chunk_ms=chunk_ms, cut=cut
                    )
                    self.assertEqual(outcome, [("heard", "Hi." if cut else "Hi. How can I help?")])
            with self.subTest(chunk_ms=chunk_ms, earlier_padding=True):
                outcome = await speak(
                    audio(10),
                    text("Hi."),
                    TTSStoppedFrame(),
                    TTSStartedFrame(),
                    audio(30),
                    text("How can I help?"),
                    chunk_ms=chunk_ms,
                    cut=True,
                )
                self.assertEqual(outcome, [("heard", "Hi.")])

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
