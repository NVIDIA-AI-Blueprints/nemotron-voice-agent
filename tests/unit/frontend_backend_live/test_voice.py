# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The voice on Pipecat's LLM service: each call's request, what each decision does, and what the backend sends."""

import asyncio
import time
import unittest
from dataclasses import replace

from pipecat.frames.frames import (
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)

from examples.frontend_backend_live.config_manager.schema import MemoryConfig
from examples.frontend_backend_live.spoken import Line, LineKind
from examples.frontend_backend_live.voice import APP_CONTEXT, RESULT, Writer, message_tokens
from tests.unit.core_helpers import _timeout, chunk
from tests.unit.frontend_helpers import CONFIG, PROMPTS, SEED, CallHarness, Port, SentenceTTS, decision

TEXTS, LINES = PROMPTS.cascade, PROMPTS.lines
JSON_RULES = TEXTS.json_rules


def system(request: dict) -> str:
    return request["messages"][0]["content"]


def last(request: dict) -> str:
    return request["messages"][-1]["content"]


def enum(request: dict) -> list[str]:
    return request["response_format"]["json_schema"]["schema"]["properties"]["action"]["enum"]


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_turn_sends_the_routing_layout_and_a_speak_streams_into_the_context(self):
        async with CallHarness(decision("speak", "Hello there. How can I help?")) as call:
            await call.say("Hi there")
            await call.until(lambda: call.said() == ["Hello there. How can I help?"], "the reply")
            request = call.request(0)
            self.assertEqual(enum(request), ["speak", "delegate", "wait"])
            self.assertTrue(request["response_format"]["json_schema"]["strict"])
            framed = TEXTS.frame_before + "You are Remy." + TEXTS.frame_after + "\n" + TEXTS.routing
            self.assertEqual(system(request), framed + "\n" + JSON_RULES)
            runtime = call.voice.layout.runtime_data(TEXTS.no_work_in_progress, "Hi there")
            self.assertEqual(request["messages"][1:], [{"role": "user", "content": "Hi there\n\n" + runtime}])
            self.assertEqual(call.tts.spoken, ["Hello there.", "How can I help?"])
            await call.quiet()
            self.assertEqual(len(call.chat.requests), 1)  # a decided turn is never decided again

    async def test_a_refused_delegation_says_the_refusal_and_never_its_lead_in(self):
        async with CallHarness(decision("delegate", "Let me check."), port=Port(refuse=True)) as call:
            await call.say("What teas do you have?")
            await call.until(lambda: call.said() == [LINES.refusal], "the refusal")
            self.assertNotIn("Let me check.", call.tts.spoken)
            self.assertEqual(call.voice.worknote.open(), [])

    async def test_a_caller_turn_right_after_reference_data_carries_the_caller_label(self):
        async with CallHarness(decision("speak", "Welcome back."), decision("speak", "Great.")) as call:
            await call.say("Hi")
            await call.until(lambda: call.said() == ["Welcome back."], "the first reply")
            call.voice.thinking("The caller is a regular.")
            await call.say("yes")
            await call.until(lambda: len(call.said()) == 2, "the reply")
            user = [m["content"] for m in call.request(1)["messages"] if m["role"] == "user"]
            self.assertTrue(
                user[-1].startswith(TEXTS.context_label + "The caller is a regular.\n" + TEXTS.caller_label + "yes")
            )


class ResultTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_line_that_gives_the_update_back_says_the_result_as_written(self):
        async with CallHarness(decision("delegate", ""), decision("speak", "**Teas**: Earl Grey")) as call:
            await call.say("What teas?")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.answer("del_1", "**Teas**: Earl Grey")
            await call.until(lambda: call.said() == ["Teas: Earl Grey."], "the result as written")

    async def test_results_ready_together_are_worded_in_one_relay(self):
        async with CallHarness(decision("delegate", ""), decision("speak", "Tea and cake.")) as call:
            await call.say("Tea and cake?")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.answer("del_1", "Tea: yes.", "Cake: yes.")
            await call.until(lambda: call.said() == ["Tea and cake."], "the one line")
            self.assertIn("Tea: yes.\n\nCake: yes.", last(call.request(1)))
            await call.quiet()
            self.assertEqual(len(call.chat.requests), 2)

    async def test_a_result_that_arrives_before_a_relay_lets_its_line_out_is_worded_with_it(self):
        class AtIdle(Writer):
            words_early = False  # the relay words at idle, in the pipeline

        held = asyncio.Event()
        relay = [chunk('{"action":"speak",'), held, chunk('"speech":"Tea."}'), chunk(finish="stop")]
        async with CallHarness(decision("delegate", ""), relay, decision("speak", "Tea and cake.")) as call:
            call.voice.wording = AtIdle()
            await call.say("Tea and cake?")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.started("del_1")
            call.port.result("del_1", "Tea: yes.", "msg_1")
            await call.until(lambda: len(call.chat.requests) == 2, "the relay under way")
            call.port.result("del_1", "Cake: yes.", "msg_2")
            held.set()
            await call.until(lambda: call.said() == ["Tea and cake."], "one line for both")
            self.assertIn("Tea: yes.\n\nCake: yes.", last(call.request(2)))
            await call.quiet()
            self.assertEqual(len(call.chat.requests), 3)

    async def test_commentary_that_arrives_before_a_relay_lets_its_line_out_is_worded_with_it(self):
        class AtIdle(Writer):
            words_early = False  # the relay words at idle, in the pipeline

        held = asyncio.Event()
        relay = [chunk('{"action":"speak",'), held, chunk('"speech":"Tea."}'), chunk(finish="stop")]
        async with CallHarness(decision("delegate", ""), relay, decision("speak", "Tea, and table 4.")) as call:
            call.voice.wording = AtIdle()
            await call.say("Tea?")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.started("del_1")
            call.port.result("del_1", "Tea: yes.", "msg_1")
            await call.until(lambda: len(call.chat.requests) == 2, "the relay under way")
            call.voice.commentary("Table 4 is ready.")  # a move: the relay under way is cut
            held.set()
            await call.until(lambda: call.said() == ["Tea, and table 4."], "one line for both")
            self.assertIn("Tea: yes.\n\nTable 4 is ready.", last(call.request(2)))
            await call.quiet()
            self.assertEqual(len(call.chat.requests), 3)

    async def test_a_handover_while_its_result_is_worded_keeps_what_the_caller_heard_in_the_update(self):
        class AtIdle(Writer):
            words_early = False  # the relay words at idle, in the pipeline

        held = asyncio.Event()
        relay = [chunk('{"action":"speak",'), held, chunk('"speech":"Tea, yes."}'), chunk(finish="stop")]
        async with CallHarness(decision("delegate", "One moment."), relay, decision("speak", "Tea, yes.")) as call:
            call.voice.wording = AtIdle()
            await call.say("Tea?")
            await call.until(lambda: call.said() == ["One moment."], "the lead-in heard")
            call.port.started("del_1")
            call.port.result("del_1", "Tea: yes.", "msg_1")
            await call.until(lambda: len(call.chat.requests) == 2, "the relay under way")
            call.voice.handed_over("del_1", "del_2")  # a newer delegation takes its work over
            held.set()
            await call.until(lambda: call.said() == ["One moment.", "Tea, yes."], "the result's line")
            told = TEXTS.writer_notes.told.format(told="One moment.")
            self.assertEqual(
                [last(request) for request in call.chat.requests[1:]], [TEXTS.update_label + "Tea: yes." + told] * 2
            )
            self.assertIsNone(call.voice.spoken.told_text("del_2"))  # saying the result used what was heard
            await call.quiet()
            result = {"role": "developer", "content": TEXTS.result_label + "Tea: yes."}  # written right before its line
            self.assertEqual(call.context.get_messages()[-2:], [result, {"role": "assistant", "content": "Tea, yes."}])

    async def test_a_result_goes_into_the_context_with_its_line_right_before_it(self):
        script = (decision("delegate", "One moment."), decision("speak", "Your latte is added."), decision("wait"))
        async with CallHarness(*script) as call:
            await call.say("Add a latte.")
            await call.until(lambda: call.said() == ["One moment."], "the lead-in")
            await call.quiet()
            await call.stt.push_frame(ProposedUserStartedSpeakingFrame())
            await call.until(lambda: call.voice.floor.caller_speaking, "the caller speaking")
            call.port.answer("del_1", "Latte added.")  # the result arrives while the caller speaks
            await asyncio.sleep(0.1)
            words = TranscriptionFrame(text="Hmm.", user_id="caller", timestamp=str(time.time()), finalized=True)
            await call.stt.push_frame(words)
            await call.stt.push_frame(ProposedUserStoppedSpeakingFrame())
            await call.until(lambda: "Your latte is added." in call.said(), "the result's line")
            await call.quiet()
            result = TEXTS.result_label + "Latte added."
            contents = [message["content"] for message in call.context.get_messages()]
            self.assertEqual(contents, ["Add a latte.", "One moment.", "Hmm.", result, "Your latte is added."])

    async def test_commentary_is_worded_thinking_and_typed_questions_stay_silent(self):
        async with CallHarness(decision("speak", "Your table is ready.")) as call:
            call.voice.thinking("The caller is a regular.")
            call.voice.typed("Is table 4 free?")
            call.voice.commentary("Table 4 is ready.")
            await call.until(lambda: call.said() == ["Your table is ready."], "the commentary")
            self.assertTrue(last(call.request(0)).endswith(TEXTS.update_label + "Table 4 is ready."))
            contents = [m["content"] for m in call.context.get_messages()]
            self.assertIn(TEXTS.context_label + "The caller is a regular.", contents)
            self.assertIn(TEXTS.context_label + "Is table 4 free?", contents)
            self.assertIn("The caller is a regular.", call.voice.heard_log.transcript()[0])


class EarlyWordingTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_result_queued_while_a_lead_in_plays_is_worded_before_the_line_ends(self):
        async with CallHarness(decision("delegate", "Let me check."), decision("speak", "We have Earl Grey.")) as call:
            call.output.playing.clear()  # the lead-in plays until the test lets it end
            await call.say("What teas?")
            await call.until(lambda: call.tts.spoken == ["Let me check."], "the lead-in playing")
            call.port.answer("del_1", "Teas: Earl Grey")
            await call.until(lambda: len(call.chat.requests) == 2, "the relay worded while the lead-in plays")
            relay = call.request(1)
            self.assertFalse(relay["stream"])  # out of band: the relay's own request and format
            self.assertEqual(enum(relay), ["speak"])
            self.assertIn({"role": "assistant", "content": "Let me check."}, relay["messages"])  # read as said
            told = TEXTS.writer_notes.told.format(told="Let me check.")
            self.assertTrue(last(relay).endswith(TEXTS.update_label + "Teas: Earl Grey" + told))
            self.assertEqual(call.said(), [])
            call.output.playing.set()
            await call.until(lambda: call.said() == ["Let me check.", "We have Earl Grey."], "both lines")
            await call.quiet()
            self.assertEqual(len(call.chat.requests), 2)  # its stamp held: said with no request

    async def test_a_line_worded_early_is_worded_again_after_a_reply_moved_what_it_read(self):
        script = (
            decision("delegate", ""),
            decision("speak", "Earl Grey is in stock."),
            decision("speak", "Let me see."),
            decision("speak", "And we have Earl Grey."),
        )
        async with CallHarness(*script) as call:
            await call.say("What teas?")
            await call.until(lambda: call.port.inputs, "the delegation")
            await call.stt.push_frame(ProposedUserStartedSpeakingFrame())
            await call.until(lambda: call.voice.floor.caller_speaking, "the caller speaking")
            call.port.answer("del_1", "Teas: Earl Grey")
            await call.until(lambda: len(call.chat.requests) == 2, "the result worded while the caller speaks")
            words = TranscriptionFrame(text="Any news?", user_id="caller", timestamp=str(time.time()), finalized=True)
            await call.stt.push_frame(words)
            await call.stt.push_frame(ProposedUserStoppedSpeakingFrame())
            said = ["Let me see.", "And we have Earl Grey."]
            await call.until(lambda: call.said() == said, "the reply, then the result")
            self.assertEqual(len(call.chat.requests), 4)  # the reply moved the first wording: worded again
            self.assertIn({"role": "assistant", "content": "Let me see."}, call.request(3)["messages"])


class FailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_failed_routing_call_delegates_the_turn_and_the_talker_stays_usable(self):
        async with CallHarness(_timeout(), decision("speak", "Hi.")) as call:
            await call.say("What teas?")
            await call.until(lambda: call.port.inputs, "the delegation the fallback makes")
            self.assertTrue(call.talker.is_usable)
            self.assertTrue(call.errors)  # Pipecat reported it
            await call.say("Hello")
            await call.until(lambda: call.said() == ["Hi."], "the next reply")

    async def test_a_failed_routing_call_holds_while_work_is_open(self):
        async with CallHarness(decision("delegate", ""), _timeout()) as call:
            await call.say("Book it.")
            await call.until(lambda: call.port.inputs, "the delegation")
            await call.say("Are you there?")
            await call.until(lambda: call.said() == [LINES.holding], "the holding line")
            self.assertEqual(len(call.port.inputs), 1)

    async def test_a_failed_relay_says_the_results_as_written(self):
        async with CallHarness(decision("delegate", ""), _timeout()) as call:
            await call.say("What teas?")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.answer("del_1", "- Earl Grey")
            await call.until(lambda: call.said() == ["Earl Grey."], "the result as written")


class InterruptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_an_instruction_is_pinned_cuts_the_line_and_is_decided(self):
        async with CallHarness(decision("speak", "Of course.")) as call:
            await call.voice.instruction("Speak French.")
            await call.until(lambda: call.said() == ["Of course."], "the reply")
            self.assertIn("You are Remy.\n\nSpeak French.", system(call.request(0)))
            self.assertNotIn("newest turn", last(call.request(0)))
            self.assertIn("App instruction: Speak French.", call.voice.heard_log.transcript()[0])


class SilentLineTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_line_with_no_audio_leaves_no_text_and_is_said_once_more(self):
        tts = SentenceTTS(silent={"Hello."})
        async with CallHarness(decision("speak", "Hello."), tts=tts) as call:
            await call.say("Hi")
            await call.until(lambda: len(call.tts.spoken) == 2, "the line said once more")
            await call.quiet()
            self.assertEqual(call.said(), [])  # never heard: never written
            self.assertEqual(len(call.chat.requests), 1)

    async def test_off_a_silent_line_is_not_said_again(self):
        config = replace(CONFIG, speech=replace(CONFIG.speech, retry_silent_lines=False))
        tts = SentenceTTS(silent={"Hello."})
        async with CallHarness(decision("speak", "Hello."), tts=tts, config=config) as call:
            await call.say("Hi")
            await call.until(lambda: call.tts.spoken, "the line")
            await call.quiet()
            self.assertEqual(call.tts.spoken, ["Hello."])


class ResumeTests(unittest.IsolatedAsyncioTestCase):
    async def cut_after_first_sentence(self, call: CallHarness) -> None:
        """The caller hears the line's first sentence; the second's audio has not played."""
        audio = []
        call.output.playing.clear()
        call.output.pause_at = lambda frame: (
            isinstance(frame, TTSAudioRawFrame) and len(audio.append(frame) or audio) > 1
        )
        await call.say("Tell me")
        await call.until(lambda: any(line.heard for line in call.voice.spoken.lines.values()), "the first sentence")

    async def test_a_noise_that_cuts_a_line_says_its_unheard_rest_word_for_word(self):
        async with CallHarness(decision("speak", "First part. Second part.")) as call:
            await self.cut_after_first_sentence(call)
            call.output.pause_at = lambda frame: False
            await call.noise()
            call.output.playing.set()
            await call.until(lambda: call.said() == ["First part." + TEXTS.cut_mark, "Second part."], "the rest")
            self.assertEqual(len(call.chat.requests), 1)  # said word for word, with no model call

    async def test_a_wait_on_the_words_that_cut_a_line_resumes_it(self):
        async with CallHarness(decision("speak", "First part. Second part."), decision("wait")) as call:
            await self.cut_after_first_sentence(call)
            call.output.pause_at = lambda frame: False
            await call.say("mm-hmm")
            call.output.playing.set()
            await call.until(lambda: call.said() == ["First part." + TEXTS.cut_mark, "Second part."], "the rest")

    async def test_off_a_noise_resumes_nothing(self):
        config = replace(CONFIG, speech=replace(CONFIG.speech, resume_after_noise=False))
        async with CallHarness(decision("speak", "First part. Second part."), config=config) as call:
            await self.cut_after_first_sentence(call)
            call.output.pause_at = lambda frame: False
            await call.noise()
            call.output.playing.set()
            await call.quiet(0.4)
            self.assertEqual(call.said(), ["First part." + TEXTS.cut_mark])


class OwedTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_result_nobody_heard_is_said_again(self):
        # Owed again, the result is worded again at once, while the caller still speaks, and said after their turn.
        async with CallHarness(
            decision("delegate", ""), decision("speak", "We have tea."), decision("speak", "Tea."), decision("wait")
        ) as call:
            await call.say("What teas?")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.output.playing.clear()
            call.output.pause_at = lambda frame: isinstance(frame, TTSAudioRawFrame)
            call.port.answer("del_1", "Earl Grey.")
            await call.until(lambda: call.tts.spoken == ["We have tea."], "the worded result on its way")
            await call.stt.push_frame(ProposedUserStartedSpeakingFrame())  # the caller talks over it before any played
            await call.until(lambda: len(call.chat.requests) == 3, "the result worded again while the caller speaks")
            words = TranscriptionFrame(text="sorry?", user_id="caller", timestamp=str(time.time()), finalized=True)
            await call.stt.push_frame(words)
            await call.stt.push_frame(ProposedUserStoppedSpeakingFrame())
            call.output.pause_at = lambda frame: False
            call.output.playing.set()
            await call.until(lambda: call.said() == ["Tea."], "the result said again")
            self.assertEqual(len(call.chat.requests), 4)
            self.assertIn(TEXTS.update_label + "Earl Grey.", last(call.request(2)))
            self.assertIn("sorry?", last(call.request(3)))  # the caller's turn, decided after the wording

    async def test_a_result_said_again_is_in_the_context_once_right_before_the_line_that_says_it(self):
        # The line nobody heard wrote the result in ahead of it; saying it again moves it to right before the new line.
        async with CallHarness(
            decision("delegate", ""), decision("speak", "We have tea."), decision("speak", "Tea."), decision("wait")
        ) as call:
            await call.say("What teas?")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.output.playing.clear()  # the worded result's line waits at the output: none of its audio plays
            call.output.pause_at = lambda frame: isinstance(frame, TTSAudioRawFrame)
            call.port.answer("del_1", "Earl Grey.")
            result = {"role": "developer", "content": TEXTS.result_label + "Earl Grey."}
            await call.until(lambda: result in call.context.get_messages(), "the result written ahead of its line")
            await call.stt.push_frame(ProposedUserStartedSpeakingFrame())  # the caller barges in before any audio
            await call.until(lambda: len(call.chat.requests) == 3, "the result worded again while the caller speaks")
            words = TranscriptionFrame(text="sorry?", user_id="caller", timestamp=str(time.time()), finalized=True)
            await call.stt.push_frame(words)
            await call.stt.push_frame(ProposedUserStoppedSpeakingFrame())
            call.output.pause_at = lambda frame: False
            call.output.playing.set()
            await call.until(lambda: call.said() == ["Tea."], "the result said again")
            await call.quiet()
            contents = [message["content"] for message in call.context.get_messages()]
            self.assertEqual(contents, ["What teas?", "sorry?", result["content"], "Tea."])


class HeardLogTests(unittest.IsolatedAsyncioTestCase):
    """What a delegation carries of the voice's lines: each one from its hand-off, corrected by its playout."""

    words = TEXTS.transcript
    cut_after_first_sentence = ResumeTests.cut_after_first_sentence

    async def test_a_delegation_made_while_a_line_plays_carries_that_line(self):
        async with CallHarness(decision("delegate", "Let me check."), decision("delegate", "")) as call:
            call.output.playing.clear()  # the lead-in plays until the test lets it end
            await call.say("What teas?")
            await call.until(lambda: call.tts.spoken == ["Let me check."], "the lead-in playing")
            await call.stt.say("And cakes?")  # a turn its speech service decided, which interrupts nothing
            await call.until(lambda: len(call.port.inputs) == 2, "the second delegation")
            words = self.words
            transcript = [words.header, f"{words.agent}: Let me check.", f"{words.caller}: And cakes?"]
            self.assertEqual(call.port.text().splitlines(), transcript)
            call.output.playing.set()

    async def test_a_delegation_made_while_a_streamed_reply_plays_carries_the_whole_reply(self):
        async with CallHarness(decision("speak", "We have Earl Grey and mint."), decision("delegate", "")) as call:
            call.output.playing.clear()  # the reply plays until the test lets it end
            await call.say("What teas?")
            await call.until(lambda: call.voice.call is None and call.voice.spoken.lines, "the reply decided")
            await call.stt.say("The mint, please.")  # a turn its speech service decided, which interrupts nothing
            await call.until(lambda: call.port.inputs, "the delegation")
            words = self.words
            transcript = [
                words.header,
                f"{words.caller}: What teas?",
                f"{words.agent}: We have Earl Grey and mint.",
                f"{words.caller}: The mint, please.",
            ]
            self.assertEqual(call.port.text().splitlines(), transcript)
            call.output.playing.set()

    async def test_a_line_nobody_heard_is_not_in_the_next_delegation(self):
        async with CallHarness(decision("delegate", "Let me check."), decision("delegate", "")) as call:
            call.output.playing.clear()  # the lead-in waits at the output: none of its audio plays
            call.output.pause_at = lambda frame: isinstance(frame, TTSAudioRawFrame)
            await call.say("What teas?")
            await call.until(lambda: call.tts.spoken == ["Let me check."], "the lead-in on its way")
            words = self.words
            self.assertEqual(call.voice.heard_log.transcript(), [f"{words.agent}: Let me check."])  # pending meanwhile
            await call.say("And cakes?")  # the caller talks over it before any of it played
            call.output.pause_at = lambda frame: False
            call.output.playing.set()
            await call.until(lambda: len(call.port.inputs) == 2, "the second delegation")
            self.assertEqual(call.port.text().splitlines(), [words.header, f"{words.caller}: And cakes?"])

    async def test_a_cut_line_reaches_the_next_delegation_as_cut_in_its_place(self):
        async with CallHarness(decision("speak", "First part. Second part."), decision("delegate", "")) as call:
            await self.cut_after_first_sentence(call)
            call.voice.thinking("The caller is a regular.")  # while the line plays: after it in the transcript
            call.output.pause_at = lambda frame: False
            await call.say("Book the earlier one.")  # cuts the line after its first sentence
            call.output.playing.set()
            await call.until(lambda: call.port.inputs, "the delegation")
            words = self.words
            transcript = [
                words.header,
                f"{words.caller}: Tell me",
                f"{words.agent}: First part.{TEXTS.cut_mark}",
                f"{words.context}: The caller is a regular.",
                f"{words.caller}: Book the earlier one.",
            ]
            self.assertEqual(call.port.text().splitlines(), transcript)

    async def test_a_line_said_once_more_keeps_its_one_entry(self):
        tts = SentenceTTS(silent={"Hello."})
        async with CallHarness(decision("speak", "Hello."), decision("delegate", ""), tts=tts) as call:
            say_again = call.voice.say_again

            def held(line):  # said once more, the line has audio and plays until the test lets it end
                tts.silent.clear()
                call.output.playing.clear()
                say_again(line)

            call.voice.say_again = held
            await call.say("Hi")
            await call.until(lambda: any(line.heard for line in call.voice.spoken.lines.values()), "the line again")
            words = self.words
            self.assertEqual(call.voice.heard_log.transcript(), [f"{words.caller}: Hi", f"{words.agent}: Hello."])
            await call.say("Book it.")  # cuts it
            call.output.playing.set()
            await call.until(lambda: call.port.inputs, "the delegation")
            cut = f"{words.agent}: Hello.{TEXTS.cut_mark}"
            transcript = [words.header, f"{words.caller}: Hi", cut, f"{words.caller}: Book it."]
            self.assertEqual(call.port.text().splitlines(), transcript)


class AtOnce(Port):
    """A delegation part whose backend answers the moment it accepts the work, before the lead-in is said."""

    async def delegate(self, input):
        delegation_id = await super().delegate(input)
        asyncio.get_running_loop().call_soon(self.answer, delegation_id, "Teas: Earl Grey")
        return delegation_id


class LeadInTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_lead_in_asked_for_is_said_even_when_its_result_comes_at_once(self):
        script = (decision("delegate", "Let me check."), decision("speak", "We have Earl Grey."))
        async with CallHarness(*script, port=AtOnce()) as call:
            await call.say("What teas?")
            await call.until(lambda: call.said() == ["Let me check.", "We have Earl Grey."], "both lines")
            await call.quiet()
            self.assertEqual(len(call.chat.requests), 2)  # one wording, made once the lead-in was in flight
            told = TEXTS.writer_notes.told.format(told="Let me check.")
            self.assertTrue(last(call.request(1)).endswith(TEXTS.update_label + "Teas: Earl Grey" + told))

    async def test_a_lead_in_waits_for_its_delay_and_goes_once_the_caller_speaks(self):
        port = Port(acknowledgement_delay_ms=300)
        async with CallHarness(decision("delegate", "Let me check."), port=port) as call:
            await call.say("Book it.")
            await call.until(lambda: call.voice.acknowledgements.pending, "the pending lead-in")
            await call.noise()
            await call.quiet(0.5)
            self.assertNotIn("Let me check.", call.tts.spoken)

    async def test_commentary_drops_a_pending_lead_in(self):
        port = Port(acknowledgement_delay_ms=300)
        async with CallHarness(decision("delegate", "Let me check."), decision("speak", "Noted."), port=port) as call:
            await call.say("Book it.")
            await call.until(lambda: call.voice.acknowledgements.pending, "the pending lead-in")
            call.voice.commentary("The caller is a member.")
            await call.until(lambda: call.said() == ["Noted."], "the commentary")
            await asyncio.sleep(0.4)
            self.assertNotIn("Let me check.", call.tts.spoken)


class WorkTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_quiet_spell_brings_a_progress_line_worded_from_the_work_note(self):
        config = replace(CONFIG, delegation=replace(CONFIG.delegation, progress_after_seconds=0.2))
        async with CallHarness(decision("delegate", ""), decision("speak", "Still on it."), config=config) as call:
            await call.say("Book it.")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.started("del_1")
            await call.until(lambda: call.said() == ["Still on it."], "the progress line")
            progress = call.request(1)
            self.assertIn(TEXTS.progress, system(progress))
            self.assertIn(TEXTS.update_label + TEXTS.work_in_progress.split("{")[0], last(progress))
            await asyncio.sleep(0.6)
            self.assertEqual(len(call.chat.requests), 2)  # a later one needs news

    async def test_commentary_that_arrives_while_a_progress_line_is_worded_is_said_instead(self):
        config = replace(CONFIG, delegation=replace(CONFIG.delegation, progress_after_seconds=0.2))
        held = asyncio.Event()
        progress = [chunk('{"action":"speak",'), held, chunk('"speech":"Still on it."}'), chunk(finish="stop")]
        script = (decision("delegate", ""), progress, decision("speak", "Table 4 is ready."))
        async with CallHarness(*script, config=config) as call:
            await call.say("Book it.")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.started("del_1")
            await call.until(lambda: len(call.chat.requests) == 2, "the progress line under way")
            call.voice.commentary("Table 4 is ready.")  # a move: the progress line under way is dropped
            held.set()
            await call.until(lambda: call.said() == ["Table 4 is ready."], "the commentary's line")
            await call.quiet()
            self.assertNotIn("Still on it.", call.tts.spoken)

    async def test_the_app_wait_line_is_said_once_when_the_app_has_not_answered(self):
        config = replace(CONFIG, delegation=replace(CONFIG.delegation, app_wait_seconds=0.1))
        async with CallHarness(decision("delegate", ""), config=config) as call:
            await call.say("What teas?")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.waits_for_app("del_1")
            await call.until(lambda: call.said() == [LINES.app_wait_line], "the app-wait line")

    async def test_the_greeting_is_worded_from_its_instructions_else_its_fixed_line(self):
        for script, said in (((decision("speak", "Hi, I'm Remy."),), "Hi, I'm Remy."), ((_timeout(),), LINES.greeting)):
            with self.subTest(said=said):
                async with CallHarness(*script) as call:
                    call.voice.greet(LINES.intro)
                    await call.until(lambda said=said: call.said() == [said], "the greeting")
                    self.assertIn(LINES.intro, call.voice.instructions)


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def budget(context_tokens: int, compact_at: float):
        return replace(CONFIG, memory=MemoryConfig(context_tokens=context_tokens, compact_at=compact_at))

    async def test_the_bound_drops_the_oldest_turn_group_only_past_compact_at_of_the_budget(self):
        messages = [
            {"role": "user", "content": "first " * 20},
            {"role": "assistant", "content": "Reply one."},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "Reply two."},
        ]
        tokens = message_tokens(SEED[0]) + sum(map(message_tokens, messages))  # the instructions count too
        call = CallHarness(config=self.budget(2 * tokens, 0.5))
        call.context.set_messages(list(messages))
        call.voice.bound()
        self.assertEqual(call.context.get_messages(), messages)  # at the threshold, nothing leaves
        call.context.add_message({"role": "user", "content": "third"})
        call.voice.bound()
        self.assertEqual([m["content"] for m in call.context.get_messages()], ["second", "Reply two.", "third"])

    async def test_the_bound_keeps_the_instructions_the_newest_group_and_an_item_not_said_yet(self):
        call = CallHarness(config=self.budget(40, 0.8))
        result = {"role": "developer", "content": TEXTS.result_label + "Order 42 is ready."}
        call.voice.kinds[id(result)] = (result, RESULT)
        call.voice.unsaid.add(id(result))
        call.context.set_messages(
            [
                {"role": "user", "content": "older " * 30},
                result,
                {"role": "assistant", "content": "Reply."},
                {"role": "user", "content": "newest " * 40},  # alone above the budget: the newest group stays
            ]
        )
        call.voice.bound()
        self.assertEqual([m["content"] for m in call.context.get_messages()], [result["content"], "newest " * 40])
        self.assertEqual(call.voice.instructions, ["You are Remy."])

    async def test_the_bound_counts_an_append_on_its_way_to_the_context(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "Reply one."},
            {"role": "user", "content": "second"},
        ]
        tokens = message_tokens(SEED[0]) + sum(map(message_tokens, messages))
        call = CallHarness(config=self.budget(2 * tokens, 0.5))
        call.context.set_messages(list(messages))
        thinking = {"role": "developer", "content": TEXTS.context_label + "The caller is a regular."}
        call.voice.kinds[id(thinking)] = (thinking, APP_CONTEXT)
        call.voice.sent.append(thinking)  # queued at the talker, not in the context yet
        call.voice.bound()
        self.assertEqual([m["content"] for m in call.context.get_messages()], ["second"])

    async def test_a_line_counts_whole_from_its_hand_off(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "Reply one."},
            {"role": "user", "content": "second"},
        ]
        tokens = message_tokens(SEED[0]) + sum(map(message_tokens, messages))
        call = CallHarness(config=self.budget(2 * tokens, 0.5))
        call.context.set_messages(list(messages))
        call.voice.bound()
        self.assertEqual(call.context.get_messages(), messages)  # at the threshold, nothing leaves
        call.voice.hand(Line(LineKind.REPLY, "A long reply, " * 10))  # in flight: not in the context yet
        self.assertEqual([m["content"] for m in call.context.get_messages()], ["second"])

    async def test_a_streamed_reply_counts_whole_once_it_is_decided(self):
        turns = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "One."}]
        tokens = message_tokens(SEED[0]) + sum(map(message_tokens, [*turns, {"role": "user", "content": "second"}]))
        script = (decision("speak", "One."), decision("speak", "A long reply about the trains. " * 6))
        async with CallHarness(*script, config=self.budget(tokens + 20, 1.0)) as call:
            await call.say("first")
            await call.until(lambda: call.said() == ["One."], "the first reply")
            call.output.playing.clear()  # the long reply plays until the test lets it end
            await call.say("second")
            await call.until(lambda: call.voice.call is None and call.voice.spoken.lines, "the long reply decided")
            self.assertEqual([m["content"] for m in call.context.get_messages()], ["second"])
            call.output.playing.set()

    async def test_a_turn_past_the_budget_reads_without_the_oldest_turn_and_calls_no_model_for_it(self):
        async with CallHarness(
            decision("speak", "One."), decision("speak", "Two."), config=self.budget(100, 0.8)
        ) as call:
            await call.say("alpha " * 30)
            await call.until(lambda: call.said() == ["One."], "the first reply")
            await call.say("bravo " * 30)
            await call.until(lambda: call.said() == ["Two."], "the second reply")
            await call.quiet()
            self.assertEqual(len(call.chat.requests), 2)  # one routing call per turn, nothing else
            routing = call.request(1)
            self.assertIn("You are Remy.", system(routing))
            self.assertNotIn("alpha", str(routing["messages"]))
            self.assertIn("bravo", str(routing["messages"]))
            self.assertEqual([m["content"] for m in call.context.get_messages()], [("bravo " * 30).strip(), "Two."])
            held = {id(message) for message in call.context.get_messages()}
            self.assertLessEqual(set(call.voice.decided) | set(call.voice.noted), held)  # the dropped turn's go too

    async def test_the_normalizer_rewrites_the_newest_turn_before_the_talker_reads_it(self):
        config = replace(CONFIG, normalizer=replace(CONFIG.normalizer, enabled=True))
        rewrite = '{"action": "speak", "speech": "My code is Q91_T."}'
        async with CallHarness(rewrite, decision("delegate", ""), config=config) as call:
            await call.say("My code is Q nine one underscore T.")
            await call.until(lambda: call.port.inputs, "the delegation")
            self.assertIn('"My code is Q91_T."', last(call.request(1)))
            self.assertIn("Caller: My code is Q91_T.", call.port.text())
            one_shot = call.request(0)
            self.assertFalse(one_shot["stream"])
            # The normalizer's own instructions and its strict decision schema, through run_inference's arguments.
            self.assertTrue(one_shot["response_format"]["json_schema"]["strict"])
            self.assertIn("action", one_shot["response_format"]["json_schema"]["schema"]["properties"])
            self.assertEqual(one_shot["messages"][0]["content"].split("\n")[-1], JSON_RULES.split("\n")[-1])


if __name__ == "__main__":
    unittest.main()
