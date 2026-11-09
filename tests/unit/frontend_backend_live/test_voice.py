# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The voice in whole calls, in either talker style: each decision, what the backend sends, and what was heard."""

import asyncio
import json
import time
import unittest

from pipecat.frames.frames import (
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
)
from pipecat.processors.aggregators import async_tool_messages
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator

from examples.frontend_backend_live.config_manager.schema import MemoryConfig
from examples.frontend_backend_live.spoken import Line, LineKind, Outcome
from examples.frontend_backend_live.voice import RESULT, message_tokens
from tests.unit.core_helpers import _timeout, chunk, tool_fragment
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
    tool_call,
)

TEXTS, LINES, WORDS = PROMPTS.cascade, PROMPTS.lines, PROMPTS.cascade.transcript
LEAD_IN = '{"lead_in": "Let me check."}'


def system(request: dict) -> str:
    return request["messages"][0]["content"]


def last(request: dict) -> str:
    return request["messages"][-1]["content"]


def enum(request: dict) -> list[str]:
    return request["response_format"]["json_schema"]["schema"]["properties"]["action"]["enum"]


def told(text: str) -> str:
    return TEXTS.writer_notes.told.format(told=text)


def carried(call: CallHarness) -> list[str]:
    return [line for _, line in call.voice.heard_log.lines()]


def settings(**changes):
    """Return the tests' voice settings with ``delegation`` changes and other settings replaced."""
    delegation = CONFIG.delegation.model_copy(update=changes.pop("delegation", {}))
    return CONFIG.model_copy(update={"delegation": delegation, **changes})


def at_idle(call: CallHarness) -> None:
    """Make the relay word at idle, in the pipeline, with nothing worded early."""
    call.voice._word_early = lambda: None


async def caller_speaks(call: CallHarness, text: str, during) -> None:
    """Speak ``text`` as the caller, and run ``during()`` while they speak."""
    await call.stt.push_frame(ProposedUserStartedSpeakingFrame())
    await call.until(lambda: call.voice.floor.caller_speaking, "the caller speaking")
    await during()
    words = TranscriptionFrame(text=text, user_id="caller", timestamp=str(time.time()), finalized=True)
    await call.stt.push_frame(words)
    await call.stt.push_frame(ProposedUserStoppedSpeakingFrame())


async def first_sentence_played(call: CallHarness, ask, first: str = "First part.") -> None:
    """Run ``ask()``, then play the bot's line to its first sentence; the second waits until ``playing`` is set."""
    audio = []
    call.output.playing.clear()
    call.output.pause_at = lambda frame: isinstance(frame, TTSAudioRawFrame) and len(audio.append(frame) or audio) > 1
    await ask()
    await call.until(lambda: call.assistant.aggregation_string() == first, "the first sentence")
    call.output.pause_at = lambda frame: False


def runs(call: CallHarness) -> list:
    """Return the call each later run of the talker makes, ``None`` for a run that makes none."""
    made, next_call = [], call.voice.next_call

    async def run(context):
        made.append(await next_call(context))
        return made[-1]

    call.voice.next_call = run
    return made


def observed(call: CallHarness) -> list[tuple[str, dict]]:
    """Return the voice's later observations, each as its kind and fields."""
    events, observe = [], call.voice.observe

    def record(kind, **fields):
        events.append((kind, fields))
        observe(kind, **fields)

    call.voice.observe = record
    return events


def decided(events) -> list[dict]:
    """Return the fields of each routing decision the voice carried out."""
    return [fields for kind, fields in events if kind == "talker.completed"]


def calls_in(messages) -> list[dict]:
    """Return the routing tools' calls and their results among ``messages``, as Pipecat wrote them."""
    return [message for message in messages if message.get("tool_calls") or message.get("role") == "tool"]


def results(call: CallHarness) -> list[dict]:
    """Return each routing tool call's result in the talker's context, settled in place or deferred as a note."""
    found = []
    for message in call.context.get_messages():
        payload = async_tool_messages.parse_message(message)
        if message.get("role") == "tool" and payload is None:
            found.append(json.loads(message["content"]))
        elif payload is not None and payload.kind == "final":
            found.append(json.loads(payload.result))
    return found


class TurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_turn_sends_the_layout_and_a_speak_streams_in(self):
        async with CallHarness(decision("speak", "Hello there. How can I help?")) as call:
            made = runs(call)
            await call.say("Hi there")
            await call.until(lambda: call.said() == ["Hello there. How can I help?"], "the reply")
            request = call.request(0)
            self.assertEqual(enum(request), ["speak", "delegate", "wait"])
            self.assertTrue(request["response_format"]["json_schema"]["strict"])
            self.assertNotIsInstance(request.get("tools"), list)  # no tools: Pipecat's own "no tools" is no list
            self.assertNotIsInstance(request.get("tool_choice"), str)
            framed = TEXTS.frame_before + "You are Remy." + TEXTS.frame_after + "\n" + TEXTS.routing
            self.assertEqual(system(request), framed + "\n" + TEXTS.json_rules)
            runtime = call.voice.layout.runtime_data(TEXTS.no_work_in_progress, "Hi there")
            self.assertEqual(request["messages"][1:], [{"role": "user", "content": "Hi there\n\n" + runtime}])
            self.assertEqual(call.tts.spoken, ["Hello there.", "How can I help?"])
            await call.quiet(0.3)
            self.assertEqual(len(call.chat.requests), 1)
            self.assertEqual(len(made), 1)

    async def test_a_seed_result_is_reference_data_before_the_caller(self):
        result = TEXTS.result_label + "Order 42 is ready."
        async with CallHarness(decision("speak", "Great."), seed=[*SEED, {"role": "user", "content": result}]) as call:
            await call.say("yes")
            await call.until(lambda: call.said() == ["Great."], "the reply")
            self.assertTrue(last(call.request(0)).startswith(f"{result}\n{TEXTS.caller_label}yes\n\n"))
            self.assertEqual(call.voice.conversation()[0], {"role": "user", "content": result, "kind": "result"})

    async def test_a_wait_says_nothing_and_the_next_turn_joins_it(self):
        async with CallHarness(decision("wait"), decision("speak", "Sure.")) as call:
            await call.say("Hold on")
            await call.until(lambda: call.chat.requests, "the first decision")
            await call.quiet()
            await call.say("okay go ahead")
            await call.until(lambda: call.said() == ["Sure."], "the reply")
            runtime = last(call.request(1))
            self.assertTrue(runtime.startswith("Hold on\nokay go ahead\n\n"))
            self.assertIn(TEXTS.newest_turn.format(turn="okay go ahead"), runtime)
            self.assertEqual(call.port.inputs, [])

    async def test_a_note_no_run_took_up_leaves_at_the_next_run(self):
        async with CallHarness(decision("speak", "Sure.")) as call:
            note = {"role": "developer", "content": TEXTS.empty_turn_note}
            call.context.add_message(note)
            await call.say("Tea, please.")
            await call.until(lambda: call.said() == ["Sure."], "the reply")
            self.assertNotIn(TEXTS.empty_turn_note, str(call.request(0)["messages"]))
            self.assertNotIn(note, call.context.get_messages())

    async def test_a_barge_in_mid_decision_decides_both_turns_together(self):
        held = asyncio.Event()
        first = [chunk('{"action":"speak","spe'), held, chunk('ech":"We have Earl Grey."}'), chunk(finish="stop")]
        async with CallHarness(first, decision("speak", "Coffee it is.")) as call:
            await call.say("Tell me about your teas.")
            await call.until(lambda: call.chat.requests, "the first decision under way")
            await call.say("Actually, a coffee.")
            await call.until(lambda: len(call.chat.requests) == 2, "the decision on both turns")
            held.set()
            await call.until(lambda: call.said() == ["Coffee it is."], "the reply")
            await call.quiet()
            self.assertEqual(call.tts.spoken, ["Coffee it is."])
            turns = TEXTS.newest_turn.format(turn="Tell me about your teas. Actually, a coffee.")
            self.assertIn(turns, last(call.request(1)))

    async def test_a_failed_routing_call_delegates_or_holds_the_turn(self):
        async with CallHarness(_timeout(), _timeout(), decision("speak", "Hi.")) as call:
            await call.say("What teas?")
            await call.until(lambda: call.port.inputs, "the delegation the fallback makes")
            self.assertTrue(call.talker.is_usable)
            self.assertTrue(call.errors)
            await call.say("Are you there?")  # its work is open, so this failure holds the turn
            await call.until(lambda: call.said() == [LINES.holding], "the holding line")
            await call.say("Hello")
            await call.until(lambda: call.said() == [LINES.holding, "Hi."], "the next reply")
            self.assertEqual(len(call.port.inputs), 1)

    async def test_an_instruction_is_pinned_and_a_blank_one_does_nothing(self):
        async with CallHarness(decision("speak", "Of course.")) as call:
            await call.voice.instruction(" ")
            self.assertEqual((call.voice.instructions, carried(call)), (["You are Remy."], []))
            await call.voice.instruction("Speak French.")
            await call.until(lambda: call.said() == ["Of course."], "the reply")
            self.assertIn("You are Remy.\n\nSpeak French.", system(call.request(0)))
            self.assertNotIn("newest turn", last(call.request(0)))
            self.assertIn("App instruction: Speak French.", carried(call)[0])


class DelegationTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_refused_delegation_says_the_refusal_only(self):
        for style, reply, settled in (
            ("json", decision("delegate", "Let me check."), []),
            ("tools", tool_call("delegate", LEAD_IN), [{"status": "refused"}]),
        ):
            with self.subTest(style=style):
                async with CallHarness(reply, style=style, port=Port(refuse=True)) as call:
                    await call.say("What teas do you have?")
                    await call.until(lambda: call.said() == [LINES.refusal], "the refusal")
                    await call.until(lambda settled=settled: results(call) == settled, "the call's result")
                    self.assertNotIn("Let me check.", call.tts.spoken)
                    self.assertEqual(call.voice.worknote.open(), [])

    async def test_each_delegation_carries_what_was_heard_since(self):
        lines = ["Let me check.", "We have Earl Grey.", "One moment.", "Your order is placed."]
        script = [decision(action, line) for action, line in zip(["delegate", "speak"] * 2, lines, strict=True)]
        async with CallHarness(*script) as call:
            await call.say("What teas do you have?")
            await call.until(lambda: call.said() == lines[:1], "the first lead-in")
            call.port.answer("del_1", "Teas: Earl Grey")
            await call.until(lambda: call.said() == lines[:2], "the first result")
            await call.say("Order one, please.")
            await call.until(lambda: call.said() == lines[:3], "the second lead-in")
            call.port.answer("del_2", "Order placed.")
            await call.until(lambda: call.said() == lines, "the second result")
            first, second = call.port.text(0), call.port.text(1)
            self.assertIn(f"{WORDS.caller}: What teas do you have?", first)
            # The lead-in was said after the first input was built; a result the caller heard in full stays out.
            self.assertIn(f"{WORDS.agent}: Let me check.", second)
            self.assertIn(f"{WORDS.caller}: Order one, please.", second)
            self.assertNotIn("What teas do you have?", second)
            self.assertNotIn("We have Earl Grey.", second)

    async def test_a_failed_delegation_says_its_line_with_no_call(self):
        async with CallHarness(decision("delegate", "")) as call:
            await call.say("Place my order.")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.fail("del_1")
            await call.until(lambda: call.said() == [LINES.apology], "the apology")
            await call.quiet()
            self.assertEqual(len(call.chat.requests), 1)

    async def test_a_line_giving_the_update_back_says_it_as_written(self):
        async with CallHarness(decision("delegate", ""), decision("speak", "teas: earl grey.")) as call:
            await call.say("What teas?")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.answer("del_1", "**Teas**: Earl Grey")
            await call.until(lambda: call.said() == ["**Teas**: Earl Grey"], "the result as written")

    async def test_news_before_a_relays_line_is_worded_with_it(self):
        for case, (asked, news, line) in {
            "a result": ("Tea and cake?", "Cake: yes.", "Tea and cake."),
            "commentary": ("Tea?", "Table 4 is ready.", "Tea, and table 4."),
        }.items():
            with self.subTest(case):
                held = asyncio.Event()
                relay = [chunk('{"action":"speak",'), held, chunk('"speech":"Tea."}'), chunk(finish="stop")]
                async with CallHarness(decision("delegate", ""), relay, decision("speak", line)) as call:
                    at_idle(call)
                    await call.say(asked)
                    await call.until(lambda: call.port.inputs, "the delegation")
                    call.port.started("del_1")
                    call.port.result("del_1", "Tea: yes.", "msg_1")
                    await call.until(lambda: len(call.chat.requests) == 2, "the relay under way")
                    if case == "a result":
                        call.port.result("del_1", news, "msg_2")
                    else:
                        call.voice.commentary(news)
                    held.set()
                    await call.until(lambda line=line: call.said() == [line], "one line for both")
                    self.assertIn("Tea: yes.\n\n" + news, last(call.request(2)))
                    await call.quiet()
                    self.assertEqual(len(call.chat.requests), 3)

    async def test_a_handover_while_worded_keeps_what_was_heard(self):
        held = asyncio.Event()
        relay = [chunk('{"action":"speak",'), held, chunk('"speech":"Yes, tea."}'), chunk(finish="stop")]
        async with CallHarness(decision("delegate", "One moment."), relay, decision("speak", "Yes, tea.")) as call:
            at_idle(call)
            await call.say("Tea?")
            await call.until(lambda: call.said() == ["One moment."], "the lead-in heard")
            call.port.started("del_1")
            call.port.result("del_1", "Tea: yes.", "msg_1")
            await call.until(lambda: len(call.chat.requests) == 2, "the relay under way")
            call.voice.handed_over("del_1", "del_2")
            held.set()
            await call.until(lambda: call.said() == ["One moment.", "Yes, tea."], "the result's line")
            update = TEXTS.update_label + "Tea: yes." + told("One moment.")
            self.assertEqual([last(request) for request in call.chat.requests[1:]], [update] * 2)
            self.assertIsNone(call.voice.spoken.told_text("del_2"))
            await call.quiet()
            result = {"role": "developer", "content": TEXTS.result_label + "Tea: yes."}
            self.assertEqual(call.context.get_messages()[-2:], [result, {"role": "assistant", "content": "Yes, tea."}])

    async def test_commentary_is_worded_and_thinking_stays_silent(self):
        async with CallHarness(decision("speak", "Your table is ready.")) as call:
            call.voice.thinking("The caller is a regular.")
            call.voice.typed("Is table 4 free?")
            call.voice.commentary("Table 4 is ready.")
            await call.until(lambda: call.said() == ["Your table is ready."], "the commentary")
            self.assertTrue(last(call.request(0)).endswith(TEXTS.update_label + "Table 4 is ready."))
            contents = [m["content"] for m in call.context.get_messages()]
            self.assertIn(TEXTS.context_label + "The caller is a regular.", contents)
            self.assertIn(TEXTS.context_label + "Is table 4 free?", contents)
            self.assertIn("The caller is a regular.", carried(call)[0])

    async def test_a_result_is_worded_while_its_lead_in_plays(self):
        async with CallHarness(decision("delegate", "Let me check."), decision("speak", "We have Earl Grey.")) as call:
            call.output.playing.clear()
            await call.say("What teas?")
            await call.until(lambda: call.tts.spoken == ["Let me check."], "the lead-in playing")
            call.port.answer("del_1", "Teas: Earl Grey")
            await call.until(lambda: len(call.chat.requests) == 2, "the relay worded while the lead-in plays")
            relay = call.request(1)
            self.assertFalse(relay["stream"])
            self.assertEqual(enum(relay), ["speak"])
            self.assertIn({"role": "assistant", "content": "Let me check."}, relay["messages"])
            self.assertTrue(last(relay).endswith(TEXTS.update_label + "Teas: Earl Grey" + told("Let me check.")))
            self.assertEqual(call.said(), [])
            call.output.playing.set()
            await call.until(lambda: call.said() == ["Let me check.", "We have Earl Grey."], "both lines")
            await call.quiet()
            self.assertEqual(len(call.chat.requests), 2)

    async def test_an_early_wording_is_redone_after_a_reply_moves_it(self):
        script = [decision("delegate", "")]
        script += [decision("speak", line) for line in ("Earl Grey is in stock.", "Let me see.", "And Earl Grey.")]
        async with CallHarness(*script) as call:
            await call.say("What teas?")
            await call.until(lambda: call.port.inputs, "the delegation")

            async def answered():
                call.port.answer("del_1", "Teas: Earl Grey")
                await call.until(lambda: len(call.chat.requests) == 2, "the result worded while the caller speaks")

            await caller_speaks(call, "Any news?", answered)
            await call.until(lambda: call.said() == ["Let me see.", "And Earl Grey."], "the reply, then the result")
            self.assertEqual(len(call.chat.requests), 4)
            self.assertIn({"role": "assistant", "content": "Let me see."}, call.request(3)["messages"])

    async def test_a_result_nobody_heard_is_said_again_once(self):
        # The unheard line wrote its result ahead of it; the retry moves the result to right before its new line.
        script = (decision("delegate", ""), decision("speak", "We have tea."), decision("speak", "Tea."))
        async with CallHarness(*script, decision("wait")) as call:
            await call.say("What teas?")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.output.playing.clear()
            call.output.pause_at = lambda frame: isinstance(frame, TTSAudioRawFrame)
            call.port.answer("del_1", "Earl Grey.")
            result = {"role": "developer", "content": TEXTS.result_label + "Earl Grey."}
            await call.until(lambda: call.tts.spoken == ["We have tea."], "the worded result on its way")
            await call.until(lambda: result in call.context.get_messages(), "the result written ahead of its line")

            async def worded_again():
                await call.until(lambda: len(call.chat.requests) == 3, "the result worded again")
                self.assertNotIn(result, [*call.context.get_messages(), *call.voice.sent])

            await caller_speaks(call, "sorry?", worded_again)
            call.output.pause_at = lambda frame: False
            call.output.playing.set()
            await call.until(lambda: call.said() == ["Tea."], "the result said again")
            self.assertEqual(len(call.chat.requests), 4)
            self.assertIn(TEXTS.update_label + "Earl Grey.", last(call.request(2)))
            self.assertIn("sorry?", last(call.request(3)))
            await call.quiet()
            contents = [message["content"] for message in call.context.get_messages()]
            self.assertEqual(contents, ["What teas?", "sorry?", result["content"], "Tea."])

    async def test_a_lead_in_is_said_though_its_result_comes_at_once(self):
        class AtOnce(Port):
            async def delegate(self, input):
                delegation_id = await super().delegate(input)
                asyncio.get_running_loop().call_soon(self.answer, delegation_id, "Teas: Earl Grey")
                return delegation_id

        script = (decision("delegate", "Let me check."), decision("speak", "We have Earl Grey."))
        async with CallHarness(*script, port=AtOnce()) as call:
            await call.say("What teas?")
            await call.until(lambda: call.said() == ["Let me check.", "We have Earl Grey."], "both lines")
            await call.quiet()
            self.assertEqual(len(call.chat.requests), 2)
            self.assertTrue(
                last(call.request(1)).endswith(TEXTS.update_label + "Teas: Earl Grey" + told("Let me check."))
            )

    async def test_empty_thinking_or_commentary_drops_a_lead_in(self):
        for append in ("thinking", "commentary"):
            with self.subTest(append=append):
                port = Port(acknowledgement_delay_ms=300)
                async with CallHarness(decision("delegate", "Let me check."), port=port) as call:
                    await call.say("Place my order.")
                    await call.until(lambda: call.voice.acknowledgements.pending, "the pending lead-in")
                    before = (list(call.context.get_messages()), carried(call), call.voice.version)
                    getattr(call.voice, append)("")
                    self.assertEqual(call.voice.acknowledgements.pending, {})
                    await call.quiet(0.4)
                    self.assertNotIn("Let me check.", call.tts.spoken)
                    after = (call.context.get_messages(), carried(call), call.voice.version)
                    self.assertEqual(after, before)
                    self.assertEqual((len(call.chat.requests), list(call.voice.owed)), (1, []))


class PlayoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_sentence_with_no_audio_is_left_out_of_its_line(self):
        tts = ToneTTS(silent={"Welcome to Bluebird."})
        line = "Welcome to Bluebird. How can I help you today?"
        async with CallHarness(decision("speak", line), tts=tts, output=TransportOutput(chunk_ms=50)) as call:
            await call.say("Hi")
            await call.until(lambda: call.said() == ["How can I help you today?"], "the sentence that played")
            await call.quiet()
            self.assertEqual(len(tts.spoken), 2)
            self.assertEqual(carried(call)[-1], f"{WORDS.agent}: How can I help you today?")

    async def test_a_result_tells_the_backend_only_what_played(self):
        tts, output = ToneTTS(silent={"Your total is five dollars."}), TransportOutput(chunk_ms=50)
        async with CallHarness(decision("delegate", ""), _timeout(), tts=tts, output=output) as call:
            await call.say("What do I owe?")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.answer("del_1", "Your total is five dollars.", "Anything else?")
            await call.until(lambda: call.said() == ["Anything else?"], "the result that played")
            await call.quiet()
            self.assertEqual(carried(call), [f"{WORDS.agent}: Anything else?"])

    async def test_silent_lines_not_in_a_row_keep_the_call(self):
        # Pipecat's text-to-speech writes itself off at three audio contexts in a row with no audio.
        tts = ToneTTS(silent={"Hello."}, stop_frame_timeout_s=0.2)
        async with CallHarness(decision("speak", "Hello."), decision("speak", "Still there?"), tts=tts) as call:
            call.output.playing.clear()
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
            self.assertTrue(tts.is_usable)
            self.assertFalse(call.run.done())

    async def test_a_wordless_turn_that_cuts_a_line_is_decided_once(self):
        asked = "Sorry, could you say that again?"
        async with CallHarness(decision("speak", "First part. Second part."), decision("speak", asked)) as call:
            await first_sentence_played(call, lambda: call.say("Tell me"))
            await call.noise()
            call.output.playing.set()
            await call.until(lambda: call.said() == ["First part." + TEXTS.cut_mark, asked], "the talker's line")
            runtime = last(call.request(1))
            self.assertIn(TEXTS.empty_turn_note + "\n" + TEXTS.decision_cue, runtime)
            self.assertEqual(str(call.request(1)["messages"]).count(TEXTS.empty_turn_note), 1)
            self.assertNotIn(TEXTS.empty_turn_note, str(call.context.get_messages()))
            self.assertNotIn(TEXTS.newest_turn.split("{")[0], runtime)
            self.assertIn({"role": "assistant", "content": "First part." + TEXTS.cut_mark}, call.request(1)["messages"])
            await call.quiet()
            self.assertEqual(call.tts.spoken, ["First part.", "Second part.", asked])
            self.assertEqual(call.port.inputs, [])

    async def test_only_one_wordless_turn_in_a_row_is_decided(self):
        script = (decision("speak", "First part. Second part."), decision("delegate", "Let me check."))
        async with CallHarness(*script) as call:
            await first_sentence_played(call, lambda: call.say("Tell me"))
            await call.noise()
            call.output.playing.set()
            await call.until(lambda: len(call.chat.requests) == 2, "the talker's one run on the turn with no words")
            await call.quiet()
            await call.noise()
            await call.quiet(0.4)
            self.assertEqual(len(call.chat.requests), 2)
            self.assertEqual((call.port.inputs, call.said()), ([], ["First part." + TEXTS.cut_mark]))
            self.assertNotIn("Let me check.", call.tts.spoken)

    async def test_a_cut_line_reaches_the_next_delegation_as_cut(self):
        async with CallHarness(decision("speak", "First part. Second part."), decision("delegate", "")) as call:
            await first_sentence_played(call, lambda: call.say("Tell me"))
            call.voice.thinking("The caller is a regular.")
            await call.say("I'll take the first one.")
            call.output.playing.set()
            await call.until(lambda: call.port.inputs, "the delegation")
            transcript = [
                WORDS.header,
                f"{WORDS.caller}: Tell me",
                f"{WORDS.agent}: First part.{TEXTS.cut_mark}",
                f"{WORDS.context}: The caller is a regular.",
                f"{WORDS.caller}: I'll take the first one.",
            ]
            self.assertEqual(call.port.text().splitlines(), transcript)

    async def test_a_silent_line_awaiting_its_retry_leaves_on_a_cut(self):
        tts = SentenceTTS(silent={"Hello."})
        script = (decision("speak", "Hello."), decision("speak", "Sure thing."), decision("delegate", ""))
        async with CallHarness(*script, tts=tts) as call:
            call.output.playing.clear()
            await call.say("Hi")
            await call.until(lambda: call.tts.spoken == ["Hello."], "the silent line on its way")
            await call.stt.say("And?")  # a turn its speech service decided, which interrupts nothing
            await call.until(lambda: call.tts.spoken == ["Hello.", "Sure thing."], "the reply on its way")

            def hold_after_first_audio(frame):
                if isinstance(frame, TTSAudioRawFrame):
                    call.output.playing.clear()
                return False

            call.output.pause_at = hold_after_first_audio
            call.output.playing.set()
            await call.until(lambda: call.voice.waiting_lines, "the silent line waiting for its retry")
            await call.until(lambda: call.assistant.aggregation_string() == "Sure thing.", "the reply plays")
            call.output.pause_at = lambda frame: False
            await call.say("Place my order.")
            call.output.playing.set()
            await call.until(lambda: call.port.inputs, "the delegation")
            self.assertNotIn(f"{WORDS.agent}: Hello.", call.port.text().splitlines())
            await call.quiet()

    async def test_a_cut_short_reply_keeps_the_words_that_were_heard(self):
        script = (
            decision("delegate", ""),
            decision("speak", "Sure, one moment.", finish="length"),
            decision("delegate", ""),
        )
        async with CallHarness(*script) as call:
            await call.say("Order a latte.")
            await call.until(lambda: call.port.inputs, "the first delegation")
            await call.say("Are you there?")
            await call.until(lambda: call.said() == ["Sure, one moment."], "the streamed words heard in full")
            await call.say("Thanks.")
            await call.until(lambda: len(call.port.inputs) == 2, "the second delegation")
            self.assertIn(f"{WORDS.agent}: Sure, one moment.", call.port.text().splitlines())

    async def test_a_line_cut_in_its_first_sentence_is_not_heard(self):
        script = (decision("speak", "We open at seven. The kitchen closes at ten."), decision("delegate", ""))
        async with CallHarness(*script, tts=ToneTTS(word_ms=250)) as call:
            call.output.playing.clear()
            call.output.pause_at = lambda frame: isinstance(frame, TTSTextFrame)
            await call.say("When do you open?")
            lines = call.voice.spoken.lines
            await call.until(lambda: any(line.outcome is Outcome.PLAYING for line in lines.values()), "its audio")
            call.output.pause_at = lambda frame: False
            await call.say("Order a latte.")
            call.output.playing.set()
            await call.until(lambda: call.port.inputs, "the delegation")
            self.assertEqual(call.said(), [])
            transcript = [WORDS.header, f"{WORDS.caller}: When do you open?", f"{WORDS.caller}: Order a latte."]
            self.assertEqual(call.port.text().splitlines(), transcript)

    async def test_a_cut_result_is_not_said_again_and_is_read_as_cut(self):
        script = [decision("delegate", "")]
        script += [
            decision("speak", line) for line in ("We have Earl Grey. And green tea.", "Earl Grey and green tea.")
        ]
        async with CallHarness(*script) as call:
            await call.say("What teas do you have?")
            await call.until(lambda: call.port.inputs, "the delegation")

            async def answer():
                call.port.answer("del_1", "Teas: Earl Grey, green")

            await first_sentence_played(call, answer, "We have Earl Grey.")
            await call.say("Sorry, which ones?")
            call.output.playing.set()
            cut = "We have Earl Grey." + TEXTS.cut_mark
            await call.until(lambda: call.said() == [cut, "Earl Grey and green tea."], "the reply after the cut")
            self.assertIn({"role": "assistant", "content": cut}, call.request(2)["messages"])
            await call.quiet()
            self.assertEqual(len(call.chat.requests), 3)


class WorkTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_quiet_spell_brings_a_progress_line(self):
        script = (decision("delegate", ""), decision("speak", "Still on it."), decision("speak", "Still on it."))
        async with CallHarness(*script, config=settings(delegation={"progress_after_seconds": 0.2})) as call:
            await call.say("Place my order.")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.started("del_1")
            await call.until(lambda: call.said() == ["Still on it."], "the progress line")
            progress = call.request(1)
            self.assertIn(TEXTS.progress, system(progress))
            self.assertIn(TEXTS.update_label + TEXTS.work_in_progress.split("{")[0], last(progress))
            await asyncio.sleep(0.6)
            self.assertEqual(len(call.chat.requests), 2)
            news = {"type": "message", "status": "completed", "phase": "commentary", "content": "Checking the date."}
            call.port.send("del_1", "response.output_item.done", item=news)
            await call.until(lambda: len(call.chat.requests) == 3, "a later progress line")
            await call.quiet()
            self.assertEqual(call.said(), ["Still on it."])

    async def test_commentary_replaces_a_progress_line_being_worded(self):
        held = asyncio.Event()
        progress = [chunk('{"action":"speak",'), held, chunk('"speech":"Still on it."}'), chunk(finish="stop")]
        script = (decision("delegate", ""), progress, decision("speak", "Table 4 is ready."))
        async with CallHarness(*script, config=settings(delegation={"progress_after_seconds": 0.2})) as call:
            await call.say("Place my order.")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.started("del_1")
            await call.until(lambda: len(call.chat.requests) == 2, "the progress line under way")
            call.voice.commentary("Table 4 is ready.")
            held.set()
            await call.until(lambda: call.said() == ["Table 4 is ready."], "the commentary's line")
            await call.quiet()
            self.assertNotIn("Still on it.", call.tts.spoken)

    async def test_the_greeting_is_worded_else_its_fixed_line(self):
        for script, said in (((decision("speak", "Hi, I'm Remy."),), "Hi, I'm Remy."), ((_timeout(),), LINES.greeting)):
            with self.subTest(said=said):
                async with CallHarness(*script) as call:
                    call.voice.greet(LINES.intro)
                    await call.until(lambda said=said: call.said() == [said], "the greeting")
                    self.assertIn(LINES.intro, call.voice.instructions)


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def budget(context_tokens: int, compact_at: float):
        return settings(memory=MemoryConfig(context_tokens=context_tokens, compact_at=compact_at))

    async def test_the_bound_keeps_the_newest_group_and_unsaid_items(self):
        call = CallHarness(config=self.budget(40, 0.8))
        result = {"role": "developer", "content": TEXTS.result_label + "Order 42 is ready."}
        call.voice.kinds[id(result)] = (result, RESULT)
        call.voice.unsaid.add(id(result))
        newest = "newest " * 40
        older = {"role": "user", "content": "older " * 30}
        call.context.set_messages(
            [older, result, {"role": "assistant", "content": "Reply."}, {"role": "user", "content": newest}]
        )
        call.voice.bound()
        self.assertEqual([m["content"] for m in call.context.get_messages()], [result["content"], newest])
        self.assertEqual(call.voice.instructions, ["You are Remy."])

    async def test_the_view_of_the_context_changes_nothing(self):
        messages = [
            {"role": "user", "content": "first " * 20},
            {"role": "assistant", "content": "Reply one."},
            {"role": "user", "content": "second"},
        ]
        call = CallHarness(config=self.budget(40, 0.5))
        call.context.set_messages(list(messages))
        viewed = [m["content"] for m in call.voice.conversation(labeled=True)]
        self.assertEqual((viewed, call.context.get_messages()), ([m["content"] for m in messages], messages))
        call.voice.bound()
        self.assertEqual(call.context.get_messages(), messages[2:])

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
        self.assertEqual(call.context.get_messages(), messages)
        call.voice.hand(Line(LineKind.REPLY, "A long reply, " * 10))
        self.assertEqual([m["content"] for m in call.context.get_messages()], ["second"])

    async def test_a_streamed_reply_counts_whole_once_decided(self):
        turns = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "One."}]
        tokens = message_tokens(SEED[0]) + sum(map(message_tokens, [*turns, {"role": "user", "content": "second"}]))
        script = (decision("speak", "One."), decision("speak", "A long reply about the scones. " * 6))
        async with CallHarness(*script, config=self.budget(tokens + 20, 1.0)) as call:
            await call.say("first")
            await call.until(lambda: call.said() == ["One."], "the first reply")
            call.output.playing.clear()
            await call.say("second")
            await call.until(lambda: call.voice.call is None and call.voice.spoken.lines, "the long reply decided")
            self.assertEqual([m["content"] for m in call.context.get_messages()], ["second"])
            call.output.playing.set()

    async def test_a_turn_past_the_budget_drops_the_oldest_turn(self):
        async with CallHarness(
            decision("speak", "One."), decision("speak", "Two."), config=self.budget(100, 0.8)
        ) as call:
            await call.say("alpha " * 30)
            await call.until(lambda: call.said() == ["One."], "the first reply")
            await call.say("bravo " * 30)
            await call.until(lambda: call.said() == ["Two."], "the second reply")
            await call.quiet()
            self.assertEqual(len(call.chat.requests), 2)
            routing = call.request(1)
            self.assertIn("You are Remy.", system(routing))
            self.assertNotIn("alpha", str(routing["messages"]))
            self.assertIn("bravo", str(routing["messages"]))
            self.assertEqual([m["content"] for m in call.context.get_messages()], [("bravo " * 30).strip(), "Two."])
            held = {id(message) for message in call.context.get_messages()}
            self.assertLessEqual(set(call.voice.decided) | set(call.voice.noted), held)

    async def test_the_bound_drops_a_call_with_its_result(self):
        script = (tool_call("wait"), tool_call("wait", call_id="call_1"))
        async with CallHarness(*script, style="tools", config=self.budget(1000, 0.8)) as call:
            await call.say("Long details. " * 300)
            await call.until(lambda: len(results(call)) == 1, "the first call's result")
            await call.quiet()
            self.assertEqual(len(calls_in(call.context.get_messages())), 2)
            await call.say("And one more thing.")
            await call.until(lambda: len(call.chat.requests) == 2, "the second routing call")
            await call.until(lambda: len(results(call)) == 1 and results(call)[0]["status"] == "waited", "the result")
            await call.quiet()
            messages = call.context.get_messages()
            self.assertEqual([m.get("content") for m in messages if m.get("role") == "user"], ["And one more thing."])
            [made, result] = calls_in(messages)
            self.assertEqual((made["tool_calls"][0]["id"], result["tool_call_id"]), ("call_1", "call_1"))
            self.assertEqual(calls_in(call.request(1)["messages"]), [])


class ToolStyleTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_delegate_call_hands_over_once_then_its_lead_in(self):
        port = Port(answer=asyncio.Event())
        async with CallHarness(tool_call("delegate", LEAD_IN), style="tools", port=port) as call:
            events = observed(call)
            await call.say("What teas do you have?")
            await call.until(lambda: port.asked == 1, "the hand-off")
            await asyncio.sleep(0.3)
            self.assertEqual((call.said(), call.tts.spoken, results(call)), ([], [], []))
            port.answer_event.set()
            await call.until(lambda: call.said() == ["Let me check."], "the lead-in")
            await call.quiet()
            self.assertEqual(len(port.inputs), 1)
            self.assertIn("What teas do you have?", port.text())
            self.assertEqual(results(call), [{"status": "delegated"}])
            [made] = decided(events)
            self.assertEqual((made["action"], made["tool_calls"]), ("delegate", ["delegate"]))
            self.assertEqual(len(call.chat.requests), 1)
            [made, result] = calls_in(call.context.get_messages())
            self.assertEqual(made["tool_calls"][0]["function"], {"name": "delegate", "arguments": LEAD_IN})
            self.assertEqual(result["tool_call_id"], made["tool_calls"][0]["id"])
            self.assertEqual(call.context.get_messages()[-1], {"role": "assistant", "content": "Let me check."})

    async def test_the_lead_in_waits_for_the_acknowledgement_delay(self):
        port = Port(acknowledgement_delay_ms=400)
        async with CallHarness(tool_call("delegate", LEAD_IN), style="tools", port=port) as call:
            await call.say("What teas do you have?")
            await call.until(lambda: port.inputs, "the accepted hand-off")
            accepted = asyncio.get_running_loop().time()
            await call.until(lambda: results(call), "the call's result")
            await asyncio.sleep(0.2)
            self.assertEqual(call.tts.spoken, [])
            await call.until(lambda: call.tts.spoken == ["Let me check."], "the lead-in")
            self.assertGreaterEqual(asyncio.get_running_loop().time() - accepted, 0.4)

    async def test_words_beside_a_delegate_call_replace_its_lead_in(self):
        words = "Sure, let me look into that."
        for case, reply in {
            "before the call": [chunk(words), *tool_call("delegate", LEAD_IN)],
            "after the call": [*tool_call("delegate", LEAD_IN, finish=None), chunk(words), chunk(finish="tool_calls")],
        }.items():
            with self.subTest(case):
                async with CallHarness(reply, style="tools") as call:
                    events = observed(call)
                    await call.say("What teas?")
                    await call.until(lambda: call.port.inputs, "the delegation")
                    await call.until(lambda: results(call), "the call's result")
                    await call.quiet()
                    self.assertEqual((call.said(), call.tts.spoken), ([words], [words]))
                    self.assertIn(f"{WORDS.agent}: {words}", call.port.text().splitlines())
                    self.assertEqual([fields["action"] for fields in decided(events)], ["delegate"])
                    self.assertEqual(results(call), [{"status": "delegated"}])

    async def test_a_delegate_call_pipecat_skips_is_handed_over_at_end(self):
        unnamed = tool_fragment(1, "call_1", arguments="{}")
        cases = {
            # Pipecat reads only the first call of a delta: it runs the wait alone.
            "packed into one delta after a wait": (
                [chunk(tool_calls=[tool_fragment(0, "call_0", "wait", "{}"), tool_fragment(1, "call_1", "delegate")])],
                [{"status": "waited"}],
            ),
            # Pipecat runs no call when the stream's last call has no name.
            "before a call with no name": (
                [chunk(tool_calls=[tool_fragment(0, "call_0", "delegate", "{}")]), chunk(tool_calls=[unnamed])],
                [],
            ),
        }
        for case, (reply, ran) in cases.items():
            with self.subTest(case):
                async with CallHarness([*reply, chunk(finish="tool_calls")], style="tools") as call:
                    await call.say("What teas do you have?")
                    await call.until(lambda: call.port.inputs, "the hand-off")
                    await call.quiet()
                    self.assertEqual(results(call), ran)
                    self.assertEqual([work.id for work in call.voice.worknote.open()], ["del_1"])

    async def test_several_calls_hand_over_once_and_each_settles(self):
        reply = [
            chunk(tool_calls=[tool_fragment(0, "call_0", "delegate", LEAD_IN)]),
            chunk(tool_calls=[tool_fragment(1, "call_1", "wait", "{}")]),
            chunk(tool_calls=[tool_fragment(2, "call_2", "delegate", "{}")]),
            chunk(finish="tool_calls"),
        ]
        async with CallHarness(reply, style="tools") as call:
            await call.say("What teas do you have?")
            await call.until(lambda: len(results(call)) == 3, "the three calls' results")
            await call.quiet()
            self.assertEqual(len(call.port.inputs), 1)
            statuses = [result["status"] for result in results(call)]
            self.assertEqual(statuses, ["delegated", "waited", "nothing_to_delegate"])

    async def test_a_reply_cut_before_its_end_hands_nothing_over(self):
        words = "Sure, let me look into that."
        for case, turn in {
            "its words had streamed": "And scones?",  # they answered the first turn
            "its call had streamed": "What teas do you have? And scones?",
        }.items():
            with self.subTest(case):
                held = asyncio.Event()
                if case == "its words had streamed":
                    first = [chunk(words + " "), held, *tool_call("delegate", LEAD_IN)]
                else:
                    first = [*tool_call("delegate", LEAD_IN, finish=None), held, chunk(finish="tool_calls")]
                second = tool_call("delegate", '{"lead_in": "One moment."}', call_id="call_1")
                async with CallHarness(first, second, style="tools") as call:
                    await call.say("What teas do you have?")
                    await call.until(lambda: len(call.chat.requests) == 1, "the first routing call")
                    if case == "its words had streamed":
                        await call.until(lambda: call.assistant.aggregation_string() == words, "the words played")
                    await call.say("And scones?")
                    await call.until(lambda: call.port.inputs, "the second routing call's hand-off")
                    await call.until(lambda: results(call), "the call's result")
                    await call.quiet()
                    self.assertIn(TEXTS.newest_turn.format(turn=turn), last(call.request(1)))
                    self.assertEqual((len(call.port.inputs), results(call)), (1, [{"status": "delegated"}]))
                    self.assertIn("What teas do you have?", call.port.text())

    async def test_a_reply_a_result_cuts_runs_no_call_and_is_redone(self):
        held = asyncio.Event()
        cut = [
            chunk(tool_calls=[tool_fragment(0, "call_A", "delegate", '{"lead_in": "One moment."}')]),
            held,
            chunk(tool_calls=[tool_fragment(1, "call_B", "wait", "{}")]),
            chunk(finish="tool_calls"),
        ]
        again = tool_call("delegate", '{"lead_in": "Checking the scones."}', call_id="call_C")
        script = (tool_call("delegate", LEAD_IN), cut, again, decision("speak", "We have Earl Grey."))
        async with CallHarness(*script, style="tools") as call:
            at_idle(call)
            events = observed(call)
            await call.say("What teas do you have?")
            await call.until(lambda: call.said() == ["Let me check."], "the first turn's lead-in")
            await call.quiet()
            await call.say("And scones?")
            await call.until(lambda: len(call.chat.requests) == 2, "the second turn's routing call")
            call.port.answer("del_1", "Teas: Earl Grey")  # the reply has said nothing, so the result cuts it
            held.set()
            await call.until(lambda: len(call.port.inputs) == 2, "the second turn's hand-off")
            await call.until(lambda: len(results(call)) == 2, "both calls' results")
            await call.quiet()
            self.assertIn(("talker.cut_off", {}), events)
            self.assertEqual([fields["action"] for fields in decided(events)], ["delegate", "delegate"])
            request = call.request(2)
            self.assertIn(TEXTS.newest_turn.format(turn="And scones?"), last(request))
            [made, result] = calls_in(request["messages"])
            self.assertEqual((made["tool_calls"][0]["id"], result["tool_call_id"]), ("call_0", "call_0"))
            self.assertEqual(results(call), [{"status": "delegated"}] * 2)
            self.assertEqual(len(call.port.inputs), 2)

    async def test_the_idle_timer_fires_while_delegated_work_runs(self):
        # The delegate call settles once the hand-off is accepted, and Pipecat arms no idle timer while a call is open.
        async with CallHarness(tool_call("delegate", LEAD_IN), style="tools") as call:
            idle = []
            user = next(processor for processor in call.processors if isinstance(processor, LLMUserAggregator))
            user.add_event_handler("on_user_turn_idle", lambda aggregator: idle.append(call.voice.worknote.open()))
            await call.say("What teas do you have?")
            await call.until(lambda: call.said() == ["Let me check."], "the lead-in")
            await call.until(lambda: idle, "the caller's idle spell")
            self.assertEqual(results(call), [{"status": "delegated"}])
            self.assertEqual([work.id for work in idle[0]], ["del_1"])

    async def test_an_unusable_reply_holds_while_work_is_open(self):
        script = (tool_call("delegate"), tool_call("lookup_order", "{}", call_id="call_1"))
        async with CallHarness(*script, style="tools") as call:
            await call.say("Place my order.")
            await call.until(lambda: call.port.inputs, "the delegation")
            await call.say("Are you there?")
            await call.until(lambda: call.said() == [LINES.holding], "the holding line")
            self.assertEqual(len(call.port.inputs), 1)

    async def test_words_while_a_hand_off_waits_never_cancel_it(self):
        port = Port(answer=asyncio.Event())
        script = (
            tool_call("delegate", LEAD_IN),
            tool_call("wait", call_id="call_1"),
            tool_call("wait", call_id="call_2"),
        )
        async with CallHarness(*script, style="tools", port=port) as call:
            events = observed(call)
            await call.say("What teas do you have?")
            await call.until(lambda: port.asked == 1, "the hand-off")
            await call.say("mm-hmm")
            await call.until(lambda: len(decided(events)) == 2, "the second turn's decision")
            port.answer_event.set()
            await call.until(lambda: len(results(call)) == 2, "both calls' results")
            await call.quiet()
            self.assertEqual(len(port.inputs), 1)
            self.assertEqual(call.voice.worknote.open()[0].id, "del_1")
            self.assertNotIn("Let me check.", call.tts.spoken)
            self.assertEqual(sorted(result["status"] for result in results(call)), ["delegated", "waited"])
            # Pipecat deferred the first result past the caller's words; the next run settles it in the call's place.
            await call.say("Okay.")
            await call.until(lambda: len(call.chat.requests) == 3, "the third routing call")
            result = {"role": "tool", "content": '{"status": "delegated"}', "tool_call_id": "call_0"}
            self.assertIn(result, call.request(2)["messages"])
            self.assertFalse([m for m in call.request(2)["messages"] if async_tool_messages.parse_message(m)])

    async def test_the_view_shows_calls_and_delegations_carry_none(self):
        script = (tool_call("delegate", LEAD_IN), tool_call("delegate", "{}", call_id="call_1"))
        async with CallHarness(*script, style="tools") as call:
            await call.say("What teas do you have?")
            await call.until(lambda: call.said() == ["Let me check."], "the lead-in")
            await call.quiet()
            kinds = [entry["kind"] for entry in call.voice.conversation()]
            self.assertEqual(kinds, ["caller", "call", "call_result", "spoken"])
            made, result = call.voice.conversation()[1:3]
            self.assertEqual(made["tool_calls"][0]["function"]["name"], "delegate")
            self.assertEqual(result["tool_call_id"], made["tool_calls"][0]["id"])
            await call.say("And a scone.")
            await call.until(lambda: len(call.port.inputs) == 2, "the second delegation")
            transcript = call.port.text()
            self.assertIn("And a scone.", transcript)
            self.assertNotIn("delegate", transcript)
            self.assertNotIn("status", transcript)
