# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The voice on Pipecat's LLM service: each call's request, what each decision does, and what the backend sends."""

import asyncio
import time
import unittest

from pipecat.frames.frames import (
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)

from examples.frontend_backend_live.config_manager.schema import MemoryConfig
from examples.frontend_backend_live.spoken import Line, LineKind, Outcome
from examples.frontend_backend_live.voice import APP_CONTEXT, RESULT, message_tokens
from tests.unit.core_helpers import _timeout, chunk
from tests.unit.frontend_helpers import (
    CONFIG,
    PROMPTS,
    SEED,
    CallHarness,
    Port,
    SentenceTTS,
    ToneTTS,
    TransportOutput,
    decision,
    word_turns,
)

TEXTS, LINES = PROMPTS.cascade, PROMPTS.lines
JSON_RULES = TEXTS.json_rules


def system(request: dict) -> str:
    return request["messages"][0]["content"]


def last(request: dict) -> str:
    return request["messages"][-1]["content"]


def enum(request: dict) -> list[str]:
    return request["response_format"]["json_schema"]["schema"]["properties"]["action"]["enum"]


def at_idle(call: CallHarness) -> None:
    """The relay words at idle, in the pipeline: nothing is worded early."""
    call.voice._word_early = lambda: None


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
            thinking = {"role": "developer", "content": TEXTS.context_label + "The caller is a regular."}
            await call.until(lambda: thinking in call.context.get_messages(), "the thinking in the context")
            await call.say("yes")
            await call.until(lambda: len(call.said()) == 2, "the reply")
            user = [m["content"] for m in call.request(1)["messages"] if m["role"] == "user"]
            self.assertTrue(
                user[-1].startswith(TEXTS.context_label + "The caller is a regular.\n" + TEXTS.caller_label + "yes")
            )

    async def test_a_result_in_the_seed_is_reference_data_and_the_caller_turn_after_it_carries_the_label(self):
        result = TEXTS.result_label + "Order 42 is ready."
        async with CallHarness(decision("speak", "Great."), seed=[*SEED, {"role": "user", "content": result}]) as call:
            await call.say("yes")
            await call.until(lambda: call.said() == ["Great."], "the reply")
            self.assertTrue(last(call.request(0)).startswith(f"{result}\n{TEXTS.caller_label}yes\n\n"))
            self.assertEqual(call.voice.conversation()[0], {"role": "user", "content": result, "kind": "result"})

    async def test_a_note_of_pipecats_that_no_run_took_up_leaves_the_context_at_the_next_run(self):
        async with CallHarness(decision("speak", "Sure.")) as call:
            note = {"role": "developer", "content": TEXTS.empty_turn_note}
            call.context.add_message(note)  # a turn with no words whose run the caller's next turn overtook
            await call.say("Tea, please.")
            await call.until(lambda: call.said() == ["Sure."], "the reply")
            self.assertNotIn(TEXTS.empty_turn_note, str(call.request(0)["messages"]))
            self.assertNotIn(note, call.context.get_messages())


class ResultTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_line_that_gives_the_update_back_says_the_result_as_written(self):
        async with CallHarness(decision("delegate", ""), decision("speak", "teas: earl grey.")) as call:
            await call.say("What teas?")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.answer("del_1", "**Teas**: Earl Grey")
            await call.until(lambda: call.said() == ["**Teas**: Earl Grey"], "the result as written")

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
        held = asyncio.Event()
        relay = [chunk('{"action":"speak",'), held, chunk('"speech":"Tea."}'), chunk(finish="stop")]
        async with CallHarness(decision("delegate", ""), relay, decision("speak", "Tea and cake.")) as call:
            at_idle(call)
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
        held = asyncio.Event()
        relay = [chunk('{"action":"speak",'), held, chunk('"speech":"Tea."}'), chunk(finish="stop")]
        async with CallHarness(decision("delegate", ""), relay, decision("speak", "Tea, and table 4.")) as call:
            at_idle(call)
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
        held = asyncio.Event()
        relay = [chunk('{"action":"speak",'), held, chunk('"speech":"Yes, tea."}'), chunk(finish="stop")]
        async with CallHarness(decision("delegate", "One moment."), relay, decision("speak", "Yes, tea.")) as call:
            at_idle(call)
            await call.say("Tea?")
            await call.until(lambda: call.said() == ["One moment."], "the lead-in heard")
            call.port.started("del_1")
            call.port.result("del_1", "Tea: yes.", "msg_1")
            await call.until(lambda: len(call.chat.requests) == 2, "the relay under way")
            call.voice.handed_over("del_1", "del_2")  # a newer delegation takes its work over
            held.set()
            await call.until(lambda: call.said() == ["One moment.", "Yes, tea."], "the result's line")
            told = TEXTS.writer_notes.told.format(told="One moment.")
            self.assertEqual(
                [last(request) for request in call.chat.requests[1:]], [TEXTS.update_label + "Tea: yes." + told] * 2
            )
            self.assertIsNone(call.voice.spoken.told_text("del_2"))  # saying the result used what was heard
            await call.quiet()
            result = {"role": "developer", "content": TEXTS.result_label + "Tea: yes."}  # written right before its line
            self.assertEqual(call.context.get_messages()[-2:], [result, {"role": "assistant", "content": "Yes, tea."}])

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

    async def test_a_lead_in_that_played_whole_keeps_the_wording_made_while_it_played(self):
        script = (decision("delegate", "One moment.\nLet me check."), decision("speak", "We have Earl Grey."))
        async with CallHarness(*script) as call:
            call.output.playing.clear()  # the lead-in plays until the test lets it end
            await call.say("What teas?")
            await call.until(lambda: call.tts.spoken == ["One moment.", "Let me check."], "the lead-in playing")
            call.port.answer("del_1", "Teas: Earl Grey")
            await call.until(lambda: len(call.chat.requests) == 2, "the relay worded while the lead-in plays")
            call.output.playing.set()
            await call.until(lambda: len(call.said()) == 2, "both lines")
            await call.quiet()
            self.assertEqual(len(call.chat.requests), 2)  # its stamp held: said with no request


class GivesUp(SentenceTTS):
    """Speaks as :class:`SentenceTTS` until ``Bye.``, where it writes itself off as Pipecat's text-to-speech does."""

    async def _say(self, sentence: str) -> None:
        if sentence == "Bye.":
            await self.push_error("3 consecutive TTS contexts completed with no audio", force_treat_as_permanent=True)
        else:
            await super()._say(sentence)


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
            call.port.answer("del_1", "Teas:", "- Earl Grey")
            await call.until(lambda: call.said() == ["Teas:\n- Earl Grey"], "the results as written, a line each")

    async def test_a_speech_service_that_gives_up_ends_the_call_once_what_it_said_has_played(self):
        line = "One two three four five six. Bye."  # its first sentence still plays when the service gives up
        async with CallHarness(decision("speak", line), tts=GivesUp(), output=TransportOutput()) as call:
            await call.say("Hi")
            await call.until(call.run.done, "the call's end")
            self.assertEqual(call.said(), ["One two three four five six."])


class InterruptionTests(unittest.IsolatedAsyncioTestCase):
    """What cuts the bot: the caller's turn, which their first sound starts while the bot is silent, two words else."""

    async def first_sentence_played(self, call: CallHarness) -> None:
        """The caller asks; the bot's line plays its first sentence, and its second waits at the output."""
        audio = []
        call.output.playing.clear()
        call.output.pause_at = lambda frame: (
            isinstance(frame, TTSAudioRawFrame) and len(audio.append(frame) or audio) > 1
        )
        await call.speak("Tell", "Tell me")
        await call.until(lambda: call.assistant.aggregation_string() == "First part.", "the first sentence")

    async def test_a_cough_and_a_one_word_backchannel_while_the_bot_speaks_do_not_cut_it(self):
        async with CallHarness(decision("speak", "First part. Second part."), user_params=word_turns()) as call:
            await self.first_sentence_played(call)
            passed = call.interruptions.passed  # the caller's turn that asked
            await call.speak()  # a cough: speech with no words
            await call.speak("Okay", "Okay.")
            await asyncio.sleep(0.3)
            self.assertEqual(call.interruptions.passed, passed)
            self.assertFalse(call.voice.floor.caller_speaking)
            call.output.pause_at = lambda frame: False
            call.output.playing.set()
            await call.until(lambda: call.said() == ["First part. Second part."], "the whole line")
            await call.quiet()
            self.assertEqual(len(call.chat.requests), 1)  # the backchannel is no turn: nothing to decide
            self.assertNotIn("Okay.", [message["content"] for message in call.context.get_messages()])

    async def test_two_words_while_the_bot_speaks_cut_it_after_what_played_and_nothing_is_said_again(self):
        script = (decision("speak", "First part. Second part."), decision("wait"))
        async with CallHarness(*script, user_params=word_turns()) as call:
            await self.first_sentence_played(call)
            await call.speak("Wait", "Wait what", "Wait, what?")
            await call.until(lambda: len(call.chat.requests) == 2, "the decision on the caller's words")
            call.output.pause_at = lambda frame: False
            call.output.playing.set()
            await call.quiet(0.4)
            cut = {"role": "assistant", "content": "First part." + TEXTS.cut_mark}
            self.assertIn(cut, call.request(1)["messages"])  # the decision read the line as far as it played
            self.assertIn(TEXTS.newest_turn.format(turn="Wait, what?"), last(call.request(1)))
            self.assertEqual(call.said(), [cut["content"]])
            self.assertEqual(call.tts.spoken, ["First part.", "Second part."])  # its rest is never said again

    async def test_a_result_that_arrives_after_the_callers_first_sound_waits_for_their_turn(self):
        script = (decision("delegate", ""), decision("speak", "Let me see."), decision("speak", "We have Earl Grey."))
        async with CallHarness(*script, user_params=word_turns()) as call:
            at_idle(call)
            await call.speak("What", "What teas?")
            await call.until(lambda: call.port.inputs, "the delegation")
            await call.stt.push_frame(VADUserStartedSpeakingFrame(start_secs=0.12))  # a sound, no word yet
            await call.until(lambda: call.voice.floor.caller_speaking, "the caller's turn, started by the sound")
            call.port.answer("del_1", "Teas: Earl Grey")
            await asyncio.sleep(0.2)
            self.assertEqual(call.tts.spoken, [])  # the result waits while the caller speaks
            words = TranscriptionFrame(text="Any news?", user_id="caller", timestamp=str(time.time()), finalized=True)
            await call.stt.push_frame(words)
            await call.stt.push_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
            await call.until(lambda: call.said() == ["Let me see.", "We have Earl Grey."], "the reply, then the result")
            self.assertIn(TEXTS.newest_turn.format(turn="Any news?"), last(call.request(1)))
            self.assertTrue(last(call.request(2)).endswith(TEXTS.update_label + "Teas: Earl Grey"))
            await call.quiet()
            self.assertEqual(call.tts.spoken, ["Let me see.", "We have Earl Grey."])

    async def test_a_cough_while_the_talker_decides_the_callers_words_brings_a_new_decision_on_them(self):
        held = asyncio.Event()  # never set: the decision under way waits for the rest of its stream
        deciding = [chunk('{"action":'), held, chunk('"speak","speech":"Hi."}'), chunk(finish="stop")]
        async with CallHarness(deciding, decision("speak", "We have Earl Grey."), user_params=word_turns()) as call:
            await call.speak("What", "What teas?")
            await call.until(lambda: len(call.chat.requests) == 1, "the talker deciding the caller's words")
            await call.speak()  # a cough while the bot is silent: its turn cuts the decision, and ends with no words
            await call.until(lambda: call.said() == ["We have Earl Grey."], "the new decision's line")
            runtime = last(call.request(1))
            self.assertIn(TEXTS.newest_turn.format(turn="What teas?"), runtime)  # the same words, decided again
            self.assertIn(TEXTS.empty_turn_note + "\n" + TEXTS.decision_cue, runtime)  # Pipecat's run on the cough
            self.assertNotIn(TEXTS.empty_turn_note, str(call.context.get_messages()))  # the run took it out
            await call.quiet()
            self.assertEqual((len(call.chat.requests), call.tts.spoken), (2, ["We have Earl Grey."]))

    async def test_an_instruction_is_pinned_cuts_the_line_and_is_decided(self):
        async with CallHarness(decision("speak", "Of course.")) as call:
            await call.voice.instruction("Speak French.")
            await call.until(lambda: call.said() == ["Of course."], "the reply")
            self.assertIn("You are Remy.\n\nSpeak French.", system(call.request(0)))
            self.assertNotIn("newest turn", last(call.request(0)))
            self.assertIn("App instruction: Speak French.", call.voice.heard_log.transcript()[0])

    async def test_an_empty_instruction_does_nothing_and_the_line_plays_on(self):
        async with CallHarness(decision("speak", "First part. Second part."), user_params=word_turns()) as call:
            await self.first_sentence_played(call)
            passed = call.interruptions.passed
            before = (list(call.voice.instructions), call.voice.heard_log.transcript(), call.voice.version)
            await call.voice.instruction(" ")
            await asyncio.sleep(0.2)
            # Nothing is pinned or cut, and nothing reaches the next delegation or moves what every wording reads.
            after = (call.voice.instructions, call.voice.heard_log.transcript(), call.voice.version)
            self.assertEqual((after, call.interruptions.passed), (before, passed))
            call.output.pause_at = lambda frame: False
            call.output.playing.set()
            await call.until(lambda: call.said() == ["First part. Second part."], "the whole line")
            await call.quiet()
            self.assertEqual(len(call.chat.requests), 1)  # nothing to decide


class SilentLineTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_line_with_no_audio_leaves_no_text_and_is_said_once_more(self):
        tts = SentenceTTS(silent={"Hello."})
        async with CallHarness(decision("speak", "Hello."), tts=tts) as call:
            await call.say("Hi")
            await call.until(lambda: len(call.tts.spoken) == 2, "the line said once more")
            await call.quiet()
            self.assertEqual(call.said(), [])  # never heard: never written
            self.assertEqual(len(call.chat.requests), 1)

    async def test_a_short_first_sentence_whose_text_goes_out_just_before_its_audio_is_heard(self):
        # "Hi." has 100 ms of audio, half of the output's chunk: its text goes out first, with the chunk the next
        # sentence's audio fills up right behind it.
        async with CallHarness(decision("speak", "Hi. How can I help?"), output=TransportOutput(chunk_ms=200)) as call:
            await call.say("Hello")
            await call.until(lambda: call.said() == ["Hi. How can I help?"], "the whole line")
            await call.quiet()
            self.assertEqual(call.tts.spoken, ["Hi.", "How can I help?"])  # heard: never said once more
            self.assertEqual(call.voice.heard_log.transcript()[-1], f"{TEXTS.transcript.agent}: Hi. How can I help?")

    async def test_a_sentence_with_no_audio_is_left_out_of_a_line_that_played(self):
        tts = ToneTTS(silent={"Welcome to Bluebird."})
        line = "Welcome to Bluebird. How can I help you today?"
        async with CallHarness(decision("speak", line), tts=tts, output=TransportOutput(chunk_ms=50)) as call:
            await call.say("Hi")
            await call.until(lambda: call.said() == ["How can I help you today?"], "the sentence that played")
            await call.quiet()
            self.assertEqual(len(tts.spoken), 2)  # heard: never said once more
            heard = f"{TEXTS.transcript.agent}: How can I help you today?"
            self.assertEqual(call.voice.heard_log.transcript()[-1], heard)

    async def test_a_lead_in_tells_the_caller_only_the_sentences_that_played(self):
        tts, output = ToneTTS(silent={"One moment."}), TransportOutput(chunk_ms=50)
        async with CallHarness(decision("delegate", "One moment. Let me check."), tts=tts, output=output) as call:
            await call.say("What teas do you have?")
            await call.until(lambda: call.said() == ["Let me check."], "the sentence that played")
            await call.quiet()
            self.assertEqual(call.voice.spoken.told_text("del_1"), "Let me check.")

    async def test_a_result_a_sentence_of_which_got_no_audio_tells_the_backend_what_played(self):
        tts, output = ToneTTS(silent={"Your total is five dollars."}), TransportOutput(chunk_ms=50)
        async with CallHarness(decision("delegate", ""), _timeout(), tts=tts, output=output) as call:
            await call.say("What do I owe?")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.answer("del_1", "Your total is five dollars.", "Anything else?")
            await call.until(lambda: call.said() == ["Anything else?"], "the result that played")
            await call.quiet()
            self.assertEqual(call.voice.heard_log.transcript(), [f"{TEXTS.transcript.agent}: Anything else?"])

    async def test_a_silent_line_is_said_once_more_when_pipecats_tts_ends_its_audio_context(self):
        tts = ToneTTS(silent={"Hello."})
        async with CallHarness(decision("speak", "Hello."), tts=tts) as call:
            await call.say("Hi")
            await call.until(lambda: len(tts.spoken) == 2, "the line said once more", timeout=6.0)
            (first, _), (again, _) = tts.spoken
            self.assertGreaterEqual(again - first, 2.9)  # its context had nothing for the stop timeout: Pipecat's 3 s
            self.assertLess(again - first, 4.0)
            self.assertEqual(len(call.chat.requests), 1)  # said again with no model call

    async def test_a_line_with_no_audio_then_one_with_audio_keep_the_call_and_start_the_count_over(self):
        # Pipecat's text-to-speech writes itself off at three audio contexts in a row with no audio.
        tts = ToneTTS(silent={"Hello."}, stop_frame_timeout_s=0.2)
        async with CallHarness(decision("speak", "Hello."), decision("speak", "Still there?"), tts=tts) as call:
            call.output.playing.clear()  # the line's end waits at the output until the test lets it go
            await call.say("Hi")
            await call.until(lambda: len(call.errors) == 1, "the line's audio context, ended with no audio")
            tts.silent = {"Still there?"}  # the line said once more gets its audio; the next line gets none
            call.output.playing.set()
            await call.until(lambda: call.said() == ["Hello."], "the line said once more, heard")
            await call.say("Hm")
            await call.until(lambda: len(call.errors) == 3, "the next line and its retry, with no audio")
            await call.quiet()
            spoken = [sentence for _, sentence in tts.spoken]
            self.assertEqual(spoken, ["Hello.", "Hello.", "Still there?", "Still there?"])
            self.assertTrue(tts.is_usable)  # three contexts with no audio, never three in a row
            self.assertFalse(call.run.done())


class CutLineTests(unittest.IsolatedAsyncioTestCase):
    """A line the caller cut: the context keeps it as far as it played, with the cut mark, and the talker decides."""

    async def cut_after_first_sentence(self, call: CallHarness, first: str = "First part.") -> None:
        """The caller hears the line's first sentence, which the assistant aggregator records; the second's has not."""
        audio = []
        call.output.playing.clear()
        call.output.pause_at = lambda frame: (
            isinstance(frame, TTSAudioRawFrame) and len(audio.append(frame) or audio) > 1
        )
        await call.say("Tell me")
        await call.until(lambda: call.assistant.aggregation_string() == first, "the first sentence")

    async def test_a_turn_with_no_words_that_cuts_a_line_runs_the_talker_once_with_the_note(self):
        script = (decision("speak", "First part. Second part."), decision("speak", "Sorry, could you say that again?"))
        async with CallHarness(*script) as call:
            await self.cut_after_first_sentence(call)
            call.output.pause_at = lambda frame: False
            await call.noise()
            call.output.playing.set()
            asked = "Sorry, could you say that again?"
            await call.until(lambda: call.said() == ["First part." + TEXTS.cut_mark, asked], "the talker's line")
            runtime = last(call.request(1))
            self.assertIn(TEXTS.empty_turn_note + "\n" + TEXTS.decision_cue, runtime)  # where the newest turn would be
            self.assertEqual(str(call.request(1)["messages"]).count(TEXTS.empty_turn_note), 1)
            self.assertNotIn(TEXTS.empty_turn_note, str(call.context.get_messages()))  # the run took it out
            self.assertNotIn(TEXTS.newest_turn.split("{")[0], runtime)
            self.assertIn({"role": "assistant", "content": "First part." + TEXTS.cut_mark}, call.request(1)["messages"])
            await call.quiet()
            self.assertEqual(call.tts.spoken, ["First part.", "Second part.", asked])  # its rest never said again
            self.assertEqual(call.port.inputs, [])

    async def test_only_one_turn_with_no_words_in_a_row_runs_the_talker_and_a_delegate_hands_nothing_over(self):
        script = (decision("speak", "First part. Second part."), decision("delegate", "Let me check."))
        async with CallHarness(*script) as call:
            await self.cut_after_first_sentence(call)
            call.output.pause_at = lambda frame: False
            await call.noise()
            call.output.playing.set()
            await call.until(lambda: len(call.chat.requests) == 2, "the talker's one run on the turn with no words")
            await call.quiet()
            await call.noise()  # the bot said nothing since: no second run in a row
            await call.quiet(0.4)
            self.assertEqual(len(call.chat.requests), 2)
            self.assertEqual((call.port.inputs, call.said()), ([], ["First part." + TEXTS.cut_mark]))
            self.assertNotIn("Let me check.", call.tts.spoken)

    async def test_a_turn_with_no_words_after_a_line_has_played_gets_no_answer(self):
        async with CallHarness(decision("speak", "Hello.")) as call:
            await call.say("Hi")
            await call.until(lambda: call.said() == ["Hello."], "the line, heard")
            await call.quiet()
            await call.noise()  # the line has played: the bot waits for the caller, and the turn cuts nothing
            await call.quiet(1.2)  # past the turn's end and the caller's idle spell, which runs whatever is still due
            self.assertEqual((len(call.chat.requests), call.said()), (1, ["Hello."]))

    async def test_a_wait_on_the_words_that_cut_a_line_says_nothing(self):
        async with CallHarness(decision("speak", "First part. Second part."), decision("wait")) as call:
            await self.cut_after_first_sentence(call)
            call.output.pause_at = lambda frame: False
            await call.say("mm-hmm")
            call.output.playing.set()
            await call.until(lambda: len(call.chat.requests) == 2, "the decision on the words")
            await call.quiet(0.4)
            self.assertEqual(call.said(), ["First part." + TEXTS.cut_mark])
            # The decision on the words read the cut: the aggregator's turn end had settled it with what played.
            self.assertIn({"role": "assistant", "content": "First part." + TEXTS.cut_mark}, call.request(1)["messages"])


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
            self.assertNotIn(result, [*call.context.get_messages(), *call.voice.sent])  # out until it is said
            words = TranscriptionFrame(text="sorry?", user_id="caller", timestamp=str(time.time()), finalized=True)
            await call.stt.push_frame(words)
            await call.stt.push_frame(ProposedUserStoppedSpeakingFrame())
            call.output.pause_at = lambda frame: False
            call.output.playing.set()
            await call.until(lambda: call.said() == ["Tea."], "the result said again")
            await call.quiet()
            contents = [message["content"] for message in call.context.get_messages()]
            self.assertEqual(contents, ["What teas?", "sorry?", result["content"], "Tea."])

    async def test_an_item_not_said_yet_whose_append_reaches_the_context_late_leaves_it_before_a_request(self):
        async with CallHarness(decision("speak", "Sure.")) as call:
            result = {"role": "developer", "content": TEXTS.result_label + "Earl Grey."}
            call.voice.kinds[id(result)] = (result, RESULT)
            call.voice.unsaid.add(id(result))
            call.context.add_message(result)  # its append landed after the line that said it went unheard
            await call.say("Hello?")
            await call.until(lambda: call.said() == ["Sure."], "the reply")
            self.assertNotIn("Earl Grey", str(call.request(0)["messages"]))
            self.assertNotIn(result, call.context.get_messages())


class HeardLogTests(unittest.IsolatedAsyncioTestCase):
    """What a delegation carries of the voice's lines: each one from its hand-off, corrected by its playout."""

    words = TEXTS.transcript
    cut_after_first_sentence = CutLineTests.cut_after_first_sentence

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
            await call.until(lambda: call.assistant.aggregation_string() == "Hello.", "the line again")
            words = self.words
            self.assertEqual(call.voice.heard_log.transcript(), [f"{words.caller}: Hi", f"{words.agent}: Hello."])
            await call.say("Book it.")  # cuts it
            call.output.playing.set()
            await call.until(lambda: call.port.inputs, "the delegation")
            cut = f"{words.agent}: Hello.{TEXTS.cut_mark}"
            transcript = [words.header, f"{words.caller}: Hi", cut, f"{words.caller}: Book it."]
            self.assertEqual(call.port.text().splitlines(), transcript)


class CutTests(unittest.IsolatedAsyncioTestCase):
    """What a cut line leaves in each record: the words the assistant aggregator recorded of it."""

    cut_after_first_sentence = CutLineTests.cut_after_first_sentence

    async def test_a_result_after_its_cut_lead_in_notes_only_the_words_that_played(self):
        script = (
            decision("delegate", "Let me check. One moment."),
            decision("wait"),
            decision("speak", "We have Earl Grey."),
        )
        async with CallHarness(*script) as call:
            await self.cut_after_first_sentence(call, "Let me check.")
            call.output.pause_at = lambda frame: False
            await call.noise()  # cuts the lead-in after its first sentence; the talker waits on the turn
            call.output.playing.set()
            await call.until(lambda: len(call.chat.requests) == 2, "the talker's run on the turn with no words")
            await call.quiet()
            self.assertEqual(call.said(), ["Let me check." + TEXTS.cut_mark])
            call.port.answer("del_1", "Teas: Earl Grey")
            await call.until(lambda: call.said()[-1:] == ["We have Earl Grey."], "the result's line")
            told = TEXTS.writer_notes.told.format(told="Let me check.")
            self.assertTrue(last(call.request(2)).endswith(TEXTS.update_label + "Teas: Earl Grey" + told))

    async def test_a_line_cut_in_its_first_sentence_is_not_heard_and_leaves_no_words_in_the_context(self):
        script = (decision("speak", "We open at seven. The kitchen closes at ten."), decision("delegate", ""))
        async with CallHarness(*script, tts=ToneTTS(word_ms=250)) as call:
            call.output.playing.clear()  # the first sentence's audio plays (1000 ms); its text waits
            call.output.pause_at = lambda frame: isinstance(frame, TTSTextFrame)
            await call.say("When do you open?")
            lines = call.voice.spoken.lines
            await call.until(lambda: any(line.outcome is Outcome.PLAYING for line in lines.values()), "its audio")
            call.output.pause_at = lambda frame: False
            await call.say("Book a table.")
            call.output.playing.set()
            await call.until(lambda: call.port.inputs, "the delegation")
            self.assertEqual(call.said(), [])
            words = TEXTS.transcript
            transcript = [words.header, f"{words.caller}: When do you open?", f"{words.caller}: Book a table."]
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
        async with CallHarness(decision("delegate", "Let me check."), decision("wait"), port=port) as call:
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

    async def test_empty_thinking_or_commentary_only_drops_a_pending_lead_in(self):
        for append in ("thinking", "commentary"):
            with self.subTest(append=append):
                port = Port(acknowledgement_delay_ms=300)
                async with CallHarness(decision("delegate", "Let me check."), port=port) as call:
                    await call.say("Book it.")
                    await call.until(lambda: call.voice.acknowledgements.pending, "the pending lead-in")
                    before = (list(call.context.get_messages()), call.voice.heard_log.transcript(), call.voice.version)
                    getattr(call.voice, append)("")
                    self.assertEqual(call.voice.acknowledgements.pending, {})
                    await call.quiet(0.4)
                    self.assertNotIn("Let me check.", call.tts.spoken)
                    # Nothing reaches the talker's context or the next delegation, and nothing a wording reads moved.
                    after = (call.context.get_messages(), call.voice.heard_log.transcript(), call.voice.version)
                    self.assertEqual(after, before)
                    self.assertEqual((len(call.chat.requests), list(call.voice.owed)), (1, []))  # nothing to say


class WorkTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_quiet_spell_brings_a_progress_line_worded_from_the_work_note(self):
        config = CONFIG.model_copy(
            update={"delegation": CONFIG.delegation.model_copy(update={"progress_after_seconds": 0.2})}
        )
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
        config = CONFIG.model_copy(
            update={"delegation": CONFIG.delegation.model_copy(update={"progress_after_seconds": 0.2})}
        )
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
        config = CONFIG.model_copy(
            update={"delegation": CONFIG.delegation.model_copy(update={"app_wait_seconds": 0.1})}
        )
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
        return CONFIG.model_copy(update={"memory": MemoryConfig(context_tokens=context_tokens, compact_at=compact_at)})

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

    async def test_the_view_of_the_context_changes_nothing(self):
        messages = [
            {"role": "user", "content": "first " * 20},
            {"role": "assistant", "content": "Reply one."},
            {"role": "user", "content": "second"},
        ]
        call = CallHarness(config=self.budget(40, 0.5))  # the oldest group is past the budget
        call.context.set_messages(list(messages))
        self.assertEqual(
            [m["content"] for m in call.voice.conversation(labeled=True)], [m["content"] for m in messages]
        )
        self.assertEqual(call.context.get_messages(), messages)
        call.voice.bound()
        self.assertEqual(call.context.get_messages(), messages[2:])

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


if __name__ == "__main__":
    unittest.main()
