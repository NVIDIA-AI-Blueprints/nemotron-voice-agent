# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""What a delegation carries of the voice's lines on two rare paths: a retry an interruption drops, a failed reply."""

import unittest

from pipecat.frames.frames import TTSAudioRawFrame

from tests.unit.frontend_helpers import PROMPTS, CallHarness, SentenceTTS, decision

WORDS = PROMPTS.cascade.transcript


class HeardLogEdgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_silent_line_waiting_for_its_retry_leaves_with_the_interruption(self):
        tts = SentenceTTS(silent={"Hello."})
        script = (decision("speak", "Hello."), decision("speak", "Sure thing."), decision("delegate", ""))
        async with CallHarness(*script, tts=tts) as call:
            call.output.playing.clear()  # the silent line's end waits at the output
            await call.say("Hi")
            await call.until(lambda: call.tts.spoken == ["Hello."], "the silent line on its way")
            await call.stt.say("And?")  # a turn its speech service decided, which interrupts nothing
            await call.until(lambda: call.tts.spoken == ["Hello.", "Sure thing."], "the reply on its way")

            def hold_after_first_audio(frame):  # the reply's audio plays, its end waits
                if isinstance(frame, TTSAudioRawFrame):
                    call.output.playing.clear()
                return False

            call.output.pause_at = hold_after_first_audio
            call.output.playing.set()  # the silent line ends with no audio: it waits to be said once more
            await call.until(lambda: call.voice.waiting_lines, "the silent line waiting for its retry")
            await call.until(lambda: any(line.heard for line in call.voice.spoken.lines.values()), "the reply plays")
            call.output.pause_at = lambda frame: False
            await call.say("Book it.")  # the caller cuts the reply in; the silent line was never heard
            call.output.playing.set()
            await call.until(lambda: call.port.inputs, "the delegation")
            self.assertNotIn(f"{WORDS.agent}: Hello.", call.port.text().splitlines())
            await call.quiet()

    async def test_a_streamed_reply_whose_output_fails_keeps_the_words_the_caller_heard(self):
        script = (
            decision("delegate", ""),
            decision("speak", "Sure, one moment.", finish="length"),  # the words stream, then the output is cut short
            decision("delegate", ""),
        )
        async with CallHarness(*script) as call:
            await call.say("Book a table.")
            await call.until(lambda: call.port.inputs, "the first delegation")
            await call.say("Are you there?")
            await call.until(lambda: call.said() == ["Sure, one moment."], "the streamed words heard in full")
            await call.say("Thanks.")
            await call.until(lambda: len(call.port.inputs) == 2, "the second delegation")
            self.assertIn(f"{WORDS.agent}: Sure, one moment.", call.port.text().splitlines())


if __name__ == "__main__":
    unittest.main()
