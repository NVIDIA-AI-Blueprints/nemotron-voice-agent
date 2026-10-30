# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""A golden for barge-in on a whole call: what the caller heard corrects the records.

The playout tracker after the output reports each line's played words and the cut an interruption makes
(``media/playout.py``), and the cascade corrects its records by them (``cascade/utterances.py``). A line cut after
some words keeps what played, then the cut mark, in the talker's memory and the heard log, which the next delegation
carries; the result's own entry in the memory stays whole. A line nobody heard leaves both, and a result line goes back
to the presenter's inbox, to be worded and said again. The caller's barge-in turns open and close with Pipecat's
proposed frames, as a real speech-to-text's do, so the user aggregator broadcasts the interruption, as
``tests/unit/frontend_backend_live/media/test_cascade_real_turns.py`` sends them. The text-to-speech is paced by the
test, so a line is cut at a chosen word, never at a moment of the clock.
"""

import asyncio
import json
import time
from unittest.mock import patch

from pipecat.frames.frames import (
    InterruptionFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameProcessor

from tests.unit.cascade_helpers import FakeSTT
from tests.unit.goldens import harness
from tests.unit.goldens.harness import GOLDEN_PROMPTS, GoldenCase, answer_round, decision

TEAS = "We have Earl Grey, Matcha and Sencha."
COFFEE = "We have espresso and filter coffee."
# One word of the tone FakeTTS speaks: 100 ms at 24 kHz.
WORD_AUDIO = b"\x10\x10" * 2400


class InterruptingSTT(FakeSTT):
    """``FakeSTT``, plus a caller turn that opens and closes with proposals, so the aggregator interrupts the reply."""

    async def start_turn(self):
        await self.push_frame(ProposedUserStartedSpeakingFrame())

    async def finish_turn(self, text):
        await self.push_frame(
            TranscriptionFrame(text=text, user_id="caller", timestamp=str(time.time()), finalized=True)
        )
        await self.push_frame(ProposedUserStoppedSpeakingFrame())


class PacedTTS(harness.RecordingTTS):
    """The harness's text-to-speech, except that a line the test paces plays only the words it lets through.

    :meth:`pace` lets the first ``words`` words of the next line with that text reach the output; the rest wait, as a
    slow synthesis's would, until an interruption drops them. With ``words=0`` none of it plays. A line is paced once:
    said again, it plays whole. Each word is ``FakeTTS``'s 100 ms of tone.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.paced: dict[str, int] = {}
        self.held: set[str] = set()  # the paced lines waiting for words that never come

    def pace(self, text, *, words):
        self.paced[text] = words

    async def process_frame(self, frame, direction):
        if isinstance(frame, InterruptionFrame):
            self.held.clear()
            self._started = False  # the cut ended the line in progress: the next line starts with its own frame
        if not (isinstance(frame, LLMTextFrame) and frame.text in self.paced):
            await super().process_frame(frame, direction)
            return
        await FrameProcessor.process_frame(self, frame, direction)
        words = frame.text.split()[: self.paced.pop(frame.text)]
        self.utterances.append(frame.text)
        if not self._started:
            self._started = True
            await self.push_frame(TTSStartedFrame())
        for word in words:
            self.spoken.append(word)
            await self.push_frame(TTSAudioRawFrame(audio=WORD_AUDIO, sample_rate=24000, num_channels=1))
            await self.push_frame(TTSTextFrame(text=word, aggregated_by="word"))
        self.held.add(frame.text)
        self.on_change()
        await asyncio.Event().wait()  # the rest never comes: the interruption cancels this


class BargeInGoldens(GoldenCase):
    """Barge-in: a line cut after some words, and a line cut before any."""

    async def serve(self, session=None, **script):
        """Start the call as ``GoldenCase.serve`` does, with the speech-to-text and text-to-speech of this file."""
        with patch.object(harness, "FakeSTT", InterruptingSTT), patch.object(harness, "RecordingTTS", PacedTTS):
            return await super().serve(session, **script)

    async def barge_in(self, line, outcome, text):
        """The caller talks over ``line``: its turn starts, the cut settles the line as ``outcome``, then the turn ends.

        In a call the cut reaches the records while the caller is still speaking, long before the turn ends and the
        talker decides it; the test waits for it in between instead of pushing the turn's frames back to back. Return
        the turn's input transcript event.
        """
        after = len(self.ws.from_server)
        self.turns_expected += 1
        await self.stt.start_turn()
        await self.wait_until(lambda: self.outcome(line) == outcome, f"the line {line!r} settled {outcome}")
        await self.stt.finish_turn(text)
        return await self.wait_for(
            "session.input_transcript.delta", after=after, where=lambda e: e["delta"].strip() == text.strip()
        )

    def outcome(self, text):
        """How the newest line with ``text`` settled by playout (the cascade's own record, read past the wire)."""
        lines = [line for line in self.engine.cascade.utterances.recent if line.text == text]
        return lines[-1].outcome.value if lines else None

    def played(self):
        """The output transcript so far: the words that played, one event each."""
        return [event["delta"] for event in self.ws.from_server if event["type"] == "session.output_transcript.delta"]

    async def test_x10_barge_ins_cut_a_result_line_after_some_words_and_before_any(self):
        """X10: barge-ins cut a result line after some words, and one before any.

        The talker delegates the tea question with a lead-in, and the writer words the backend's answer. Three of its
        words play, and the caller barges in: the line is cut, so the talker's memory keeps "We have Earl [cut off]"
        beside the whole result, and the talker answers the barge-in. The caller asks about coffee; the talker
        delegates with a lead-in, and the writer words the answer, but its speech has not started when the caller
        barges in again: nobody heard the line, so it leaves the memory and the heard log, and the answer goes back to
        the inbox, where the writer words it again at once, while the caller speaks. The talker answers the barge-in,
        which moves that line's stamp, so the writer words the answer once more and it plays whole. The coffee
        delegation's input carries the cut line, as the caller heard it.
        """
        await self.serve(
            frontend=[
                decision("delegate", "Let me check."),  # the talker, on the tea question
                decision("speak", TEAS),  # the writer
                decision("speak", "Sure, take your time."),  # the talker, on the first barge-in
                decision("delegate", "Let me look."),  # the talker, on the coffee question
                decision("speak", COFFEE),  # the writer
                decision("speak", COFFEE),  # the writer, at once, for the answer nobody heard
                decision("speak", "Yes?"),  # the talker, on the second barge-in
                decision("speak", COFFEE),  # the writer again: the reply moved the line's stamp
            ],
            backend=[answer_round("Earl Grey, Matcha and Sencha."), answer_round("Espresso and filter coffee.")],
        )
        self.tts.pace(TEAS, words=3)
        await self.say("What teas do you have?")
        await self.wait_until(lambda: self.played()[-3:] == ["We ", "have ", "Earl "], "three words of the tea answer")
        await self.barge_in(TEAS, "cut", "Hold on a second.")
        await self.wait_quiet()
        self.tts.pace(COFFEE, words=0)
        await self.say("And coffee?")
        await self.wait_until(lambda: COFFEE in self.tts.held, "the coffee answer handed to speech, none of it played")
        await self.barge_in(COFFEE, "not_heard", "Wait.")
        await self.wait_quiet()
        await self.close()
        self.assertEqual(
            self.tts.utterances,
            ["Let me check.", TEAS, "Sure, take your time.", "Let me look.", COFFEE, "Yes?", COFFEE],
        )
        self.assertEqual(" ".join(self.played()).split().count("espresso"), 1)  # the coffee answer played once, whole
        cut_turn = json.dumps(self.frontend.requests[2]["messages"])  # the talker, on the first barge-in
        self.assertIn("We have Earl" + GOLDEN_PROMPTS.cut_mark, cut_turn)
        self.assertIn("Earl Grey, Matcha and Sencha.", cut_turn)  # the result's own entry stays whole
        second_turn = json.dumps(self.frontend.requests[6]["messages"])  # the talker, on the second barge-in
        self.assertNotIn(COFFEE, second_turn)  # nobody heard it: it left the memory
        cut_line = f"{GOLDEN_PROMPTS.transcript.agent}: We have Earl{GOLDEN_PROMPTS.cut_mark}"
        self.assertIn(cut_line, json.dumps(self.backend.requests[1]))
        self.assert_golden("X10")
