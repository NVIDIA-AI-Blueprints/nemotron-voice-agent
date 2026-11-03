# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D107
"""The repository UI bot's playout: what the caller heard corrects the cascade's records, as in a live session.

The call runs through ``pipeline.bot()`` on the bot goldens' edges (``BotGoldenCase``), with a text-to-speech that
plays only the first sentence of each line, so the test cuts a line at a known sentence.
"""

import re
from dataclasses import replace
from unittest.mock import patch

from pipecat.frames.frames import (
    AggregatedTextFrame,
    InterruptionFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameProcessor

from tests.unit.goldens import test_goldens_bot
from tests.unit.goldens.harness import decision
from tests.unit.goldens.test_goldens_bot import BotGoldenCase, golden_root

LINE = "We open at seven. The kitchen closes at ten."
FIRST_SENTENCE = "We open at seven."
THREE_SENTENCES = "We open at seven. Call us. The kitchen closes at ten."
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
SENTENCE_AUDIO = b"\x10\x10" * 7200  # 300 ms of tone at 24 kHz
PART_AUDIO = b"\x10\x10" * 6000  # 250 ms


def announced(sentence: str) -> AggregatedTextFrame:
    """The sentence a text-to-speech takes, announced ahead of its audio as Pipecat's does."""
    frame = AggregatedTextFrame(text=sentence, aggregated_by="sentence")
    frame.will_be_spoken = True
    return frame


class FirstSentenceTTS(FrameProcessor):
    """A per-sentence text-to-speech whose synthesis of each line stops after the line's first sentence.

    It announces that sentence, yields its audio, then its text, as Pipecat's NVIDIA service does with one request per
    sentence. The rest of the line never comes, as a synthesis still running when the caller cuts in. ``utterances``
    and ``on_change`` are those of the goldens' ``RecordingTTS``.
    """

    def __init__(self, on_change=None, **kwargs):
        super().__init__(**kwargs)
        self.on_change = on_change or (lambda: None)
        self.utterances: list[str] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMTextFrame):
            await self.push_frame(frame, direction)
            return
        first = SENTENCE_END.split(frame.text.strip(), maxsplit=1)[0]
        await self.push_frame(announced(first))
        await self.push_frame(TTSStartedFrame())
        await self.push_frame(TTSAudioRawFrame(audio=SENTENCE_AUDIO, sample_rate=24000, num_channels=1))
        await self.push_frame(TTSTextFrame(text=first, aggregated_by="sentence"))
        self.utterances.append(frame.text)
        self.on_change()


class CutLineTests(BotGoldenCase):
    async def serve(self, **script):
        with patch.object(test_goldens_bot, "RecordingTTS", FirstSentenceTTS):
            await super().serve(**script)

    async def test_a_line_cut_after_its_first_sentence_keeps_only_that_sentence(self):
        await self.serve(frontend=[decision("speak", LINE)], welcome=False)
        await self.client_ready()
        await self.wait_quiet()
        await self.say("When do you open?")
        await self.wait_until(lambda: self.tts.utterances == [LINE], "the reply handed to speech")
        [worker], [cascade] = self.workers, self.sessions
        await worker.flush_pipeline()  # the first sentence has played; the second is still being synthesized
        await worker.queue_frame(InterruptionFrame())  # the caller cuts in
        await worker.flush_pipeline()
        prompts = cascade.agent.prompts  # the set whose words the cascade writes
        cut = FIRST_SENTENCE + prompts.cut_mark
        self.assertEqual(cascade.memory.messages()[-1], {"role": "assistant", "content": cut})
        words = prompts.transcript
        self.assertEqual(
            cascade.heard_log.transcript(), [f"{words.caller}: When do you open?", f"{words.agent}: {cut}"]
        )
        await self.close()


class NextSentenceTTS(FirstSentenceTTS):
    """As :class:`FirstSentenceTTS`, then the next sentence, announced, with 250 ms of its audio but never its text."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            await self.push_frame(announced(SENTENCE_END.split(frame.text.strip())[1]))
            await self.push_frame(TTSAudioRawFrame(audio=PART_AUDIO, sample_rate=24000, num_channels=1))


def estimating_root():
    """The goldens' bot root with ``speech.heard_words: estimated`` in the cascade's settings."""
    state = golden_root()
    speech = replace(state.cascade.speech, heard_words="estimated")
    return replace(state, cascade=replace(state.cascade, speech=speech))


class EstimatedCutTests(BotGoldenCase):
    async def serve(self, **script):
        with (
            patch.object(test_goldens_bot, "RecordingTTS", NextSentenceTTS),
            patch.object(test_goldens_bot, "golden_root", estimating_root),
        ):
            await super().serve(**script)

    async def test_the_memory_keeps_the_words_its_audio_likely_carried_within_the_sentence_in_progress(self):
        await self.serve(frontend=[decision("speak", THREE_SENTENCES)], welcome=False)
        await self.client_ready()
        await self.wait_quiet()
        await self.say("When do you open?")
        await self.wait_until(lambda: self.tts.utterances == [THREE_SENTENCES], "the reply handed to speech")
        [worker], [cascade] = self.workers, self.sessions
        await worker.flush_pipeline()  # the first sentence played (17 characters in 300 ms), then 250 ms of the next
        await worker.queue_frame(InterruptionFrame())
        await worker.flush_pipeline()
        # About 14 characters: all of "Call us.", whose audio it was, and none of the sentence after it.
        prompts = cascade.agent.prompts  # the set whose words the cascade writes
        said = f"{FIRST_SENTENCE} Call us.{prompts.cut_mark}"
        self.assertEqual(cascade.memory.messages()[-1], {"role": "assistant", "content": said})
        words = prompts.transcript
        cut = f"{words.agent}: {FIRST_SENTENCE}{prompts.cut_mark}"  # the heard log keeps whole sentences
        self.assertEqual(cascade.heard_log.transcript(), [f"{words.caller}: When do you open?", cut])
        await self.close()
