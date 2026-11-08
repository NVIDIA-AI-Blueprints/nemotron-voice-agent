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

from examples.frontend_backend_live.config_manager.schema import AnswersConfig, LiveConfig, MemoryConfig
from examples.frontend_backend_live.delegations import ReadyResult
from examples.frontend_backend_live.heard_log import HEARD, Unheard, UnheardHow
from examples.frontend_backend_live.models.decision import TalkerDecision
from examples.frontend_backend_live.spoken import Line, LineKind, Outcome
from examples.frontend_backend_live.talker import TalkerCall
from examples.frontend_backend_live.voice import RESULT, Owed, Relay, Routing, Saying, message_tokens
from examples.shared.delegation_contract import REPLY_UNHEARD, ReplyUnheard
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


def heard_lines(voice) -> list[str]:
    """Return the texts of the lines in ``voice``'s heard log."""
    return [text for _, text in voice.heard_log.lines()]


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
        # The unheard line wrote its result ahead of it; the retry says it again with its words, with no writer call
        # (answers.requeued_as_said), and moves the result to right before its new line.
        script = (decision("delegate", ""), decision("speak", "We have tea."))
        async with CallHarness(*script, decision("wait")) as call:
            await call.say("What teas?")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.output.playing.clear()
            call.output.pause_at = lambda frame: isinstance(frame, TTSAudioRawFrame)
            call.port.answer("del_1", "Earl Grey.")
            result = {"role": "developer", "content": TEXTS.result_label + "Earl Grey."}
            await call.until(lambda: call.tts.spoken == ["We have tea."], "the worded result on its way")
            await call.until(lambda: result in call.context.get_messages(), "the result written ahead of its line")

            async def owed_again():
                await call.until(lambda: call.voice.owed, "the result owed again")
                self.assertNotIn(result, [*call.context.get_messages(), *call.voice.sent])

            await caller_speaks(call, "sorry?", owed_again)
            call.output.pause_at = lambda frame: False
            call.output.playing.set()
            await call.until(lambda: call.said() == ["We have tea."], "the result said again")
            self.assertEqual(len(call.chat.requests), 3)
            self.assertIn(TEXTS.update_label + "Earl Grey.", last(call.request(1)))
            self.assertIn("sorry?", last(call.request(2)))
            await call.quiet()
            contents = [message["content"] for message in call.context.get_messages()]
            self.assertEqual(contents, ["What teas?", "sorry?", result["content"], "We have tea."])

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
            # A line that lost a sentence to missing audio is marked as cut (answers.unheard_questions).
            self.assertEqual(carried(call)[-1], f"{WORDS.agent}: How can I help you today?{TEXTS.cut_mark}")

    async def test_a_result_tells_the_backend_only_what_played(self):
        tts, output = ToneTTS(silent={"Your total is five dollars."}), TransportOutput(chunk_ms=50)
        async with CallHarness(decision("delegate", ""), _timeout(), tts=tts, output=output) as call:
            await call.say("What do I owe?")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.answer("del_1", "Your total is five dollars.", "Anything else?")
            await call.until(lambda: call.said() == ["Anything else?"], "the result that played")
            await call.quiet()
            # Marked as cut (answers.unheard_questions); the sentence with no audio asks nothing: no question note.
            self.assertEqual(carried(call), [f"{WORDS.agent}: Anything else?{TEXTS.cut_mark}"])

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


class MetadataPort(Port):
    """A delegation part that also records what the frontend states about each delegation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metadata: list[dict] = []

    async def delegate(self, input, *, metadata=None):
        self.metadata.append(dict(metadata or {}))
        return await super().delegate(input)


READ_BACK = "The note is ready. Shall I save it?"


def owed_result(text: str, delegation_id: str = "del_1") -> Owed:
    return Owed(ReadyResult(delegation_id, f"item_{len(text)}", text), LineKind.RESULT, {"content": text}, answers=True)


def result_line(text: str, number: int = 7) -> Line:
    return Line(LineKind.RESULT, text, ("del_1",), results=True, number=number)


class HeardFactTests(unittest.IsolatedAsyncioTestCase):
    """What each delegation states the caller had not heard of the backend's answers, which the write gate reads."""

    def voice(self):
        return CallHarness().voice  # built, not run: the tests set the voice's state themselves

    async def test_a_yes_said_while_the_answer_waits_in_the_queue_states_the_answer_unheard(self):
        # The read-back waits in the voice's queue while the caller says "yes" to an older line: that delegation
        # states the answer unheard (it ends with a period), even though the read-back plays before it is handed over.
        read_back = "Please confirm you want to exchange only the desk lamp."
        port = MetadataPort()
        script = (decision("delegate", ""), decision("speak", read_back), decision("delegate", ""))
        async with CallHarness(*script, port=port) as call:
            await call.say("Exchange the lamp.")
            await call.until(lambda: len(port.metadata) == 1, "the first delegation")
            await call.stt.push_frame(ProposedUserStartedSpeakingFrame())
            await call.until(lambda: call.voice.floor.caller_speaking, "the caller speaking")
            call.port.answer("del_1", read_back)
            words = TranscriptionFrame(
                text="Yes, just the desk lamp.", user_id="caller", timestamp=str(time.time()), finalized=True
            )
            await call.stt.push_frame(words)
            await call.stt.push_frame(ProposedUserStoppedSpeakingFrame())
            await call.until(lambda: len(port.metadata) == 2, "the second delegation")
            self.assertEqual(port.metadata, [{REPLY_UNHEARD: "none"}, {REPLY_UNHEARD: "answer"}])

    async def test_every_delegation_states_the_fact_in_its_metadata(self):
        port = MetadataPort()
        async with CallHarness(decision("delegate", ""), port=port) as call:
            await call.say("What teas?")
            await call.until(lambda: port.inputs, "the delegation")
            self.assertEqual(port.metadata, [{REPLY_UNHEARD: "none"}])

    async def test_a_result_with_a_question_still_owed_makes_the_next_delegation_state_question(self):
        voice = self.voice()
        voice.owed.append(owed_result(READ_BACK))
        self.assertEqual(voice.reply_unheard(), Unheard(ReplyUnheard.QUESTION, UnheardHow.WAITING))
        voice.owed.clear()
        voice.owed.append(owed_result("The note is saved."))
        self.assertEqual(voice.reply_unheard(), Unheard(ReplyUnheard.ANSWER, UnheardHow.WAITING))

    async def test_a_result_being_worded_waiting_or_playing_counts_as_waiting(self):
        for where in ("worded", "waiting", "playing", "said"):
            with self.subTest(where=where):
                voice = self.voice()
                if where == "worded":
                    voice.call = TalkerCall(plan=Relay((owed_result(READ_BACK),), READ_BACK, []))
                elif where == "waiting":
                    voice.waiting_lines.append(result_line(READ_BACK))
                elif where == "playing":
                    voice.spoken.handed(result_line(READ_BACK))
                else:
                    voice.call = TalkerCall(plan=Saying(result_line(READ_BACK)))
                self.assertEqual(voice.reply_unheard(), Unheard(ReplyUnheard.QUESTION, UnheardHow.WAITING))

    async def test_a_result_requeued_by_three_cuts_keeps_the_fact_for_every_delegation_meanwhile(self):
        voice = self.voice()
        item = owed_result(READ_BACK)
        for number in (1, 2, 3):  # each line that carries it goes unheard, and it is owed again
            voice.spoken.handed(Line(LineKind.RESULT, READ_BACK, ("del_1",), (item,), results=True, number=number))
            self.assertEqual(voice.sent_unheard().reply, ReplyUnheard.QUESTION)
            voice.spoken.lines.clear()
            voice.owed.append(item)
            self.assertEqual(voice.sent_unheard().reply, ReplyUnheard.QUESTION)
            voice.owed.clear()

    async def test_a_line_waiting_for_its_words_after_a_silent_sentence_counts_as_waiting(self):
        voice = self.voice()
        voice.spoken.ended_line = result_line(READ_BACK)  # between SpokenLines.ended and end_played
        self.assertEqual(voice.reply_unheard(), Unheard(ReplyUnheard.QUESTION, UnheardHow.WAITING))

    async def test_a_question_sentence_with_no_audio_inside_a_line_that_played_states_question(self):
        voice = self.voice()
        line = result_line("Your note has two versions. Which one should I keep?")
        voice.spoken.handed(line)
        line.heard = "Your note has two versions."  # end_played reports only the first sentence
        voice._result_settled(line, UnheardHow.NO_AUDIO)
        self.assertEqual(voice.sent_unheard(), Unheard(ReplyUnheard.QUESTION, UnheardHow.NO_AUDIO))

    async def test_lead_ins_replies_and_commentary_never_count(self):
        voice = self.voice()
        voice.waiting_lines.append(Line(LineKind.LEAD_IN, "Shall I look?", ("del_1",)))
        voice.spoken.handed(Line(LineKind.REPLY, "Which one do you mean?", number=3))
        commentary = Owed(ReadyResult("", "c1", "Want a refill?"), LineKind.COMMENTARY, {"content": "Want a refill?"})
        voice.owed.append(commentary)
        self.assertEqual(voice.reply_unheard(), HEARD)

    async def test_an_answer_that_plays_whole_between_turn_start_and_hand_off_still_counts(self):
        voice = self.voice()
        voice.owed.append(owed_result(READ_BACK))
        voice.turn_pending = voice._pending()  # what caller_started keeps
        voice.owed.clear()  # it played whole before the hand-off
        sent = voice.sent_unheard()
        self.assertEqual((sent.reply, sent.from_turn_start), (ReplyUnheard.QUESTION, True))

    async def test_the_turn_start_snapshot_is_cleared_once_sent(self):
        voice = self.voice()
        voice.owed.append(owed_result(READ_BACK))
        voice.turn_pending = voice._pending()
        voice.owed.clear()
        voice.sent_unheard()
        self.assertIsNone(voice.turn_pending)
        self.assertEqual(voice.sent_unheard(), HEARD)

    async def test_a_yes_over_the_last_word_of_a_read_back_heard_in_full_states_none(self):
        voice = self.voice()
        line = result_line(READ_BACK)
        voice.spoken.handed(line)
        line.outcome = Outcome.PLAYING
        voice.turn_pending = voice._pending()  # the caller's "yes" starts while its last sentence plays
        voice.spoken.lines.clear()
        line.heard, line.outcome = READ_BACK, Outcome.CUT  # cut after its last word: every sentence played
        self.assertEqual(voice.sent_unheard(), HEARD)

    async def test_a_line_still_unsettled_at_hand_off_counts_whole(self):
        voice = self.voice()
        line = result_line("The note is ready. It goes in the blue folder.")
        voice.spoken.handed(line)
        line.outcome = Outcome.PLAYING
        voice.turn_pending = voice._pending()
        voice.spoken.lines.clear()  # left the record, still playing at the hand-off
        self.assertEqual(voice.sent_unheard(), Unheard(ReplyUnheard.ANSWER, UnheardHow.WAITING, from_turn_start=True))

    async def test_a_decision_that_does_not_delegate_clears_the_snapshot(self):
        voice = self.voice()
        voice.owed.append(owed_result(READ_BACK))
        voice.turn_pending = voice._pending()
        plan = Routing((), "", False)
        plan.finished(voice, TalkerCall(plan=plan), TalkerDecision("wait"))
        voice.owed.clear()  # the read-back then plays whole; an instruction's delegation follows
        self.assertIsNone(voice.turn_pending)
        self.assertEqual(voice.sent_unheard(), HEARD)


def answers_config(**answers) -> LiveConfig:
    """The test config with ``answers`` set in its ``delegation.answers`` block."""
    delegation = CONFIG.delegation.model_copy(update={"answers": AnswersConfig(**answers)})
    return CONFIG.model_copy(update={"delegation": delegation})


class StillVoice:
    """A built voice that is never run: runs it asks for and early wordings are noted, not made."""

    def __init__(self, config=CONFIG):
        self.observed: list[tuple[str, dict]] = []
        self.voice = voice = CallHarness(config=config).voice
        voice._queue = lambda frame: None
        voice._word_early = lambda: None
        voice.observe = lambda kind, **fields: self.observed.append((kind, fields))
        voice.heard_log.observe = voice.observe

    def delegation(self, request: str) -> str:
        """An open delegation in the work note; its id."""
        worknote = self.voice.worknote
        reservation = worknote.reserve(request)
        delegation_id = f"del_{len(worknote.entries)}"
        worknote.confirm(reservation, delegation_id)
        return delegation_id

    def answer(self, delegation_id: str, text: str) -> Owed:
        """A ready result of ``delegation_id``, owed."""
        self.voice.worknote.answered(delegation_id)
        self.voice.ready(ReadyResult(delegation_id, f"item_{len(text)}", text))
        return self.voice.owed[-1]

    def say(self, words: str | None = None) -> Line:
        """The voice's next take, said with ``words`` (a relay's) or its own (a group's); the line handed."""
        voice = self.voice
        batch = voice._take()
        if voice._resaying:
            call = voice._said_by(batch)
            return call.plan.line
        return voice.delivered(batch, voice.joined(batch, words or "worded"))

    def unheard(self, line: Line, *, silent: bool = False) -> None:
        """``line`` settles not heard; ``silent``: it had been said once more after it produced no audio."""
        line.retried = silent
        self.voice.not_heard(line)

    def kinds(self, kind: str) -> list[dict]:
        return [fields for each, fields in self.observed if each == kind]


class RequeueTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_line_not_heard_is_said_again_with_its_words_and_no_writer_call(self):
        still = StillVoice()
        still.answer(still.delegation("Opening hours?"), "The gym opens at six.")
        line = still.say("We open at six.")
        still.unheard(line)
        again = still.say()
        self.assertEqual((again.text, again.items), ("We open at six.", line.items))
        self.assertEqual(len(still.kinds("answer.resaid")), 1)

    async def test_a_newer_result_owed_with_a_requeued_line_gets_its_own_relay(self):
        still = StillVoice()
        delegation = still.delegation("Hours and price?")
        still.answer(delegation, "Six.")
        line = still.say("We open at six.")
        late = still.answer(delegation, "Ten a month.")  # a newer result of the same request: no newer request
        still.unheard(line)
        self.assertEqual(still.say().text, "We open at six.")  # said first, as it was
        self.assertEqual(still.voice._take(), [late])  # the newer result, worded alone

    async def test_a_requeued_line_is_not_worded_early(self):
        still = StillVoice()
        still.answer(still.delegation("Hours?"), "Six.")
        line = still.say("We open at six.")
        late = still.answer(still.delegation("Price?"), "Ten a month.")
        still.unheard(line)
        self.assertEqual(still.voice._fresh_owed(), (late,))

    async def test_a_line_with_a_dropped_answer_never_resays_it(self):
        still = StillVoice()
        a, b = still.answer(still.delegation("Hours?"), "Six."), still.answer(still.delegation("Price?"), "Ten.")
        a.tries = 2
        line = still.say("Six, and ten a month.")
        still.unheard(line)  # A reaches the cap and is dropped; B is owed again, fresh
        self.assertEqual((list(still.voice.owed), b.words), ([b], ""))
        self.assertEqual(still.voice._take(), [b])  # worded anew, alone
        self.assertEqual(len(still.kinds("answer.dropped_unheard")), 1)

    async def test_a_relay_cut_by_a_move_goes_back_behind_the_requeued_group(self):
        still = StillVoice()
        still.answer(still.delegation("Hours?"), "Six.")
        line = still.say("We open at six.")
        still.unheard(line)
        fresh = Owed(ReadyResult("", "n1", "Card is on file."), LineKind.NOTICE)
        Relay((fresh,), "", []).cancelled(still.voice, TalkerCall())
        self.assertEqual(list(still.voice.owed), [*line.items, fresh])

    async def test_a_requeued_line_is_worded_again_when_a_newer_request_opened(self):
        still = StillVoice()
        still.answer(still.delegation("Book the bike."), "Bike 4 for an hour. Shall I go ahead?")
        line = still.say("Bike 4 for an hour. Shall I go ahead?")
        still.unheard(line)
        still.delegation("Make it two hours.")  # the caller asks something new before it is said again
        self.assertEqual(still.voice._take(), list(line.items))  # made fresh: worded with what else is owed
        self.assertFalse(still.voice._resaying)
        self.assertEqual([fields["reason"] for fields in still.kinds("answer.resay_reworded")], ["newer"])

    async def test_a_request_opened_and_answered_before_the_resay_still_refuses_it(self):
        still = StillVoice()
        still.answer(still.delegation("Book the bike."), "Bike 4. Shall I go ahead?")
        line = still.say("Bike 4. Shall I go ahead?")
        still.unheard(line)
        newer = still.delegation("Is there a helmet?")
        still.voice.worknote.answered(newer)  # answered by the next take: its open requests read the same
        still.voice._take()
        self.assertEqual([fields["reason"] for fields in still.kinds("answer.resay_reworded")], ["newer"])

    async def test_a_line_that_came_out_silent_is_not_resaid_at_once(self):
        still = StillVoice()
        still.answer(still.delegation("Hours?"), "Six.")
        still.unheard(still.say("We open at six."), silent=True)
        line = still.say("We open at six, every day.")  # through the writer: no third empty audio context at once
        self.assertFalse(still.voice._resaying)
        self.assertEqual([fields["reason"] for fields in still.kinds("answer.resay_reworded")], ["silent"])
        still.unheard(line)
        still.voice.heard(Line(LineKind.REPLY, "Sure.", heard="Sure."), cut=False)  # audio played since
        self.assertEqual(still.say().text, "We open at six, every day.")

    async def test_an_answer_unheard_max_tries_times_is_dropped_and_the_backend_is_told(self):
        still = StillVoice()
        delegation = still.delegation("Hours?")
        still.answer(delegation, "The gym opens at six.")
        for _ in range(3):
            still.unheard(still.say("We open at six."))
        voice = still.voice
        self.assertEqual((list(voice.owed), voice.worknote.answer_ready()), ([], False))
        self.assertNotIn(TEXTS.answer_ready_label, voice.worknote.line())
        note = TEXTS.transcript.not_said.format(answer="The gym opens at six.")
        self.assertEqual(heard_lines(voice)[-1], note)
        self.assertEqual(voice.heard_log.newest.reply, ReplyUnheard.ANSWER)

    async def test_commentary_is_never_dropped_by_the_cap(self):
        still = StillVoice()
        still.voice.commentary("Your table is ready.", "del_9")
        for _ in range(4):
            still.unheard(still.say("Your table is ready."))
        self.assertEqual(len(still.voice.owed), 1)
        self.assertEqual(still.kinds("answer.dropped_unheard"), [])

    async def test_commentary_groups_cannot_starve_fresh_answers(self):
        still = StillVoice()
        still.voice.commentary("Your table is ready.", "del_9")
        for _ in range(3):
            line = still.say("Your table is ready.")
            still.unheard(line)
        answer = still.answer(still.delegation("Hours?"), "Six.")
        batch = still.voice._take()
        self.assertEqual((batch, still.voice._resaying), ([*line.items, answer], False))
        self.assertEqual([fields["reason"] for fields in still.kinds("answer.resay_reworded")], ["tries"])

    async def test_with_one_try_an_unheard_answer_is_never_requeued(self):
        still = StillVoice(answers_config(max_tries=1))
        still.answer(still.delegation("Hours?"), "Six.")
        still.unheard(still.say("We open at six."))
        self.assertEqual((list(still.voice.owed), len(still.kinds("answer.dropped_unheard"))), ([], 1))

    async def test_with_no_limit_and_requeued_as_said_off_the_voice_behaves_as_before(self):
        still = StillVoice(answers_config(max_tries=None, requeued_as_said=False))
        still.answer(still.delegation("Hours?"), "Six.")
        for _ in range(5):
            line = still.say("We open at six.")
            still.unheard(line)
            self.assertFalse(still.voice._resaying)
        self.assertEqual(list(still.voice.owed), list(line.items))  # still owed, worded again each time
        self.assertEqual(still.kinds("answer.resaid") + still.kinds("answer.dropped_unheard"), [])

    async def test_with_requeued_as_said_off_reowed_items_are_fresh(self):
        still = StillVoice(answers_config(requeued_as_said=False))
        still.answer(still.delegation("Hours?"), "Six.")
        line = still.say("We open at six.")
        still.unheard(line)
        self.assertEqual(still.voice._fresh_owed(), line.items)
        self.assertEqual(still.voice._requeued_head(), ())

    async def test_a_refused_group_is_not_a_group_again_until_a_line_carrying_it_goes_unheard(self):
        still = StillVoice()
        a = still.answer(still.delegation("Hours?"), "Six.")
        still.unheard(still.say("We open at six."))
        b = still.answer(still.delegation("Price?"), "Ten.")  # a newer request: the group is refused
        merged = still.say("Six, and ten a month.")
        self.assertEqual(merged.items, (a, b))
        self.assertEqual(still.voice._requeued_head(), ())
        still.unheard(merged)
        self.assertEqual((still.voice._requeued_head(), a.tries, b.tries), ((a, b), 2, 1))


class KeepQuestionTests(unittest.IsolatedAsyncioTestCase):
    """The writer's line keeps the backend's last question last, or the results are said as the backend wrote them."""

    READ_BACK = "Bike 4 for an hour, ten dollars. Shall I book it?"

    async def test_a_worded_line_that_loses_the_question_says_the_result_as_written(self):
        script = (decision("delegate", ""), decision("speak", "Bike 4 is ten dollars for an hour."))
        async with CallHarness(*script) as call:
            at_idle(call)
            await call.say("A bike, please.")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.answer("del_1", self.READ_BACK)
            await call.until(lambda: call.said() == [self.READ_BACK], "the result as written")

    async def test_a_line_worded_early_that_loses_the_question_is_said_as_written(self):
        script = (decision("delegate", "One moment."), decision("speak", "Bike 4 is ten dollars for an hour."))
        async with CallHarness(*script) as call:
            call.output.playing.clear()  # the lead-in plays while the result is worded early
            await call.say("A bike, please.")
            await call.until(lambda: call.tts.spoken == ["One moment."], "the lead-in playing")
            call.port.answer("del_1", self.READ_BACK)
            await call.until(lambda: call.voice.early is not None and call.voice.early.words, "the early wording")
            self.assertEqual(call.voice.early.words, self.READ_BACK)
            call.output.playing.set()
            await call.until(lambda: call.said() == ["One moment.", self.READ_BACK], "the result as written")

    async def test_a_batch_with_commentary_is_not_checked(self):
        still = StillVoice()
        still.voice.commentary("The caller asked 'Is it vegan?' Tell them yes.", "del_9")
        still.answer(still.delegation("Hours?"), "We open at six. Shall I book you in?")
        relay = still.voice._relay_call(still.voice._take(), []).plan
        self.assertEqual(relay.asked, "")  # commentary may be an instruction to the writer: never read aloud

    async def test_as_written_results_are_said_with_no_writer_call(self):
        still = StillVoice(as_written_config())
        still.answer(still.delegation("Hours?"), "The gym opens at six.")
        call = still.voice._said_by(still.voice._take())
        self.assertEqual((call.spec, call.text), (None, "The gym opens at six."))
        self.assertEqual(len(still.kinds("answer.as_written")), 1)

    async def test_as_written_still_words_commentary(self):
        still = StillVoice(as_written_config())
        still.voice.commentary("Tell them their table is ready.", "del_9")
        call = still.voice._said_by(still.voice._take())
        self.assertIsNotNone(call.spec)  # a relay: the writer words it

    async def test_a_silent_group_is_not_said_as_written_at_once(self):
        still = StillVoice(as_written_config())
        still.answer(still.delegation("Hours?"), "The gym opens at six.")
        still.unheard(still.say(), silent=True)  # said as written, and it came out silent
        call = still.voice._said_by(still.voice._take())
        self.assertIsNotNone(call.spec)  # through a writer call first, not straight to speech


def as_written_config() -> LiveConfig:
    return CONFIG.model_copy(
        update={"delegation": CONFIG.delegation.model_copy(update={"result_wording": "as_written"})}
    )


ANSWER = "Bike 4 is free for an hour. It costs ten dollars. Shall I book it?"
REST = "It costs ten dollars. Shall I book it?"


class RestTests(unittest.IsolatedAsyncioTestCase):
    """A cut answer's unheard sentences are said once, as written, by the rules of the first decision after the cut."""

    def cut(self, still: StillVoice, *, number: int = 4) -> Line:
        """A result line cut after its first sentence; the voice holds its rest."""
        voice = still.voice
        line = Line(LineKind.RESULT, ANSWER, ("del_1",), results=True, number=number)
        voice.heard_log.agent(number, ANSWER, result=True)
        line.heard = "Bike 4 is free for an hour."
        voice.heard(line, cut=True)
        return line

    def decide(self, still: StillVoice, action: str, *, words: bool = True, steered: bool = False) -> None:
        """A routing decision on a caller turn (with words or none) or on an instruction."""
        turn = {"role": "user", "content": "okay" if words else ""}
        plan = Routing((turn,) if words else (), "okay" if words else "", False, wordless=not words, steered=steered)
        speech = "Yes, I'm here." if action == "speak" else ""
        plan.finished(still.voice, TalkerCall(plan=plan), TalkerDecision(action, speech))

    def rest_lines(self, still: StillVoice) -> list[str]:
        return [line.text for line in still.voice.waiting_lines if line.rest_of is not None]

    def dropped(self, still: StillVoice) -> list[str]:
        return [fields["reason"] for fields in still.kinds("answer.rest_dropped")]

    async def test_a_cut_result_then_a_wait_says_the_unheard_rest_once(self):
        still = StillVoice()
        self.cut(still)
        self.decide(still, "wait")
        self.assertEqual(self.rest_lines(still), [REST])
        self.assertEqual(len(still.kinds("answer.rest_said")), 1)
        self.decide(still, "wait")
        self.assertEqual(self.rest_lines(still), [REST])  # once

    async def test_a_cut_result_then_a_delegate_says_no_rest_and_the_input_carries_the_cut(self):
        still = StillVoice()
        self.cut(still)
        self.decide(still, "delegate")
        self.assertEqual((self.rest_lines(still), self.dropped(still)), ([], ["delegate"]))
        self.assertIn("Bike 4 is free for an hour." + TEXTS.cut_mark, heard_lines(still.voice)[0])

    async def test_a_cut_result_then_a_speak_on_a_turn_with_no_words_says_no_rest(self):
        still = StillVoice()
        self.cut(still)
        self.decide(still, "speak", words=False)
        self.assertEqual((self.rest_lines(still), self.dropped(still)), ([], ["wordless_speak"]))

    async def test_a_cut_result_then_a_speak_with_words_says_the_rest_after_the_next_wait(self):
        still = StillVoice()
        self.cut(still)
        self.decide(still, "speak")  # "Are you there?" - "Yes, I'm here."
        still.voice.on_user_idle()
        self.assertEqual((self.rest_lines(still), len(still.kinds("answer.rest_deferred"))), ([], 1))
        self.decide(still, "wait")  # the caller's "okay"
        self.assertEqual(self.rest_lines(still), [REST])

    async def test_a_rest_after_a_reply_to_hold_on_is_not_said_while_the_caller_is_silent(self):
        still = StillVoice()
        self.cut(still)
        self.decide(still, "speak")  # "Can you wait a second?" - "Of course, take your time."
        for _ in range(3):
            still.voice.on_user_idle()
        self.assertEqual(self.rest_lines(still), [])
        self.decide(still, "delegate")
        self.assertEqual(self.dropped(still), ["delegate"])

    async def test_a_rest_after_a_reply_is_dropped_by_a_second_reply(self):
        still = StillVoice()
        self.cut(still)
        self.decide(still, "speak")
        self.decide(still, "speak")
        self.assertEqual((self.rest_lines(still), self.dropped(still)), ([], ["speak"]))

    async def test_a_rest_is_held_again_when_the_caller_speaks_before_it_plays(self):
        still = StillVoice()
        self.cut(still)
        self.decide(still, "wait")
        still.voice.caller_started()
        self.assertEqual(self.rest_lines(still), [])
        self.decide(still, "delegate")
        self.assertEqual(self.dropped(still), ["delegate"])
        self.assertIsNone(still.voice._rest)

    async def test_a_held_rest_is_dropped_after_one_more_caller_turn(self):
        still = StillVoice()
        self.cut(still)
        for _ in range(2):  # decided turns whose words went to speech before their call was cancelled
            plan = Routing(({"role": "user", "content": "uh"},), "uh", False)
            plan.cancelled(still.voice, TalkerCall(plan=plan, said="Mm."))
        self.assertEqual((still.voice._rest, self.dropped(still)), (None, ["stale"]))

    async def test_noise_starts_do_not_make_a_held_rest_stale(self):
        still = StillVoice()
        self.cut(still)
        self.decide(still, "speak")
        still.voice.caller_started()  # two starts that never become decided turns
        still.voice.caller_started()
        self.decide(still, "wait")
        self.assertEqual(self.rest_lines(still), [REST])

    async def test_a_rest_that_is_cut_gets_no_rest_of_its_own(self):
        still = StillVoice()
        line = self.cut(still)
        self.decide(still, "wait")
        rest = still.voice.waiting_lines[-1]
        rest.number, rest.heard = 5, "It costs ten dollars."
        still.voice.heard(rest, cut=True)
        self.assertIsNone(still.voice._rest)
        self.assertIs(rest.rest_of, line)

    async def test_a_rest_heard_in_full_restores_the_cut_entry(self):
        still = StillVoice()
        self.cut(still)
        self.decide(still, "wait")
        rest = still.voice.waiting_lines[-1]
        rest.number, rest.heard = 5, REST
        still.voice.heard(rest, cut=False)
        entry = still.voice.heard_log.entries[0]
        self.assertEqual((entry.text, entry.cut, entry.lost), (ANSWER, False, ReplyUnheard.NONE))
        self.assertEqual(heard_lines(still.voice), [])  # a result heard in full: left to the backend

    async def test_a_rest_heard_after_the_cut_entry_was_carried_adds_a_now_heard_note(self):
        still = StillVoice()
        self.cut(still)
        self.decide(still, "wait")
        still.voice.heard_log.entries.clear()  # a delegation carried the cut entry
        rest = still.voice.waiting_lines[-1]
        rest.number, rest.heard = 5, REST
        still.voice.heard(rest, cut=False)
        self.assertEqual(heard_lines(still.voice), [TEXTS.transcript.now_heard.format(rest=REST)])

    async def test_a_line_that_lost_a_sentence_to_missing_audio_gets_no_rest(self):
        still = StillVoice()
        line = Line(LineKind.RESULT, ANSWER, ("del_1",), results=True, number=4, heard="Bike 4 is free for an hour.")
        still.voice.heard(line, cut=False)  # settled heard; the aggregator then reports fewer words (partly)
        self.assertIsNone(still.voice._rest)

    async def test_a_newer_result_or_an_instruction_drops_the_rest(self):
        still = StillVoice()
        self.cut(still)
        still.voice.owed.append(owed_result("Bike 5 is free too."))
        self.decide(still, "wait")
        still.voice.owed.clear()
        self.cut(still)
        self.decide(still, "wait", steered=True)
        self.assertEqual(self.dropped(still), ["newer_result", "steered"])

    async def test_a_cut_result_names_its_unheard_question_for_the_backend(self):
        still = StillVoice()
        self.cut(still)
        note = TEXTS.transcript.cut_question.format(questions="Shall I book it?")
        expected = f"{TEXTS.transcript.agent}: Bike 4 is free for an hour.{TEXTS.cut_mark}{note}"
        self.assertEqual(heard_lines(still.voice), [expected])
        self.assertEqual(still.kinds("answer.question_unheard"), [{"line": 4, "questions": 1}])

    async def test_the_rest_once_off_holds_no_rest(self):
        still = StillVoice(answers_config(unheard_rest_once=False))
        self.cut(still)
        self.assertIsNone(still.voice._rest)

    async def test_a_cut_result_then_a_wait_says_the_rest_through_the_pipeline(self):
        script = (decision("delegate", ""), decision("speak", "Bike 4 is free. Shall I book it?"), decision("wait"))
        async with CallHarness(*script) as call:
            at_idle(call)
            await call.say("A bike, please.")
            await call.until(lambda: call.port.inputs, "the delegation")
            audio = []
            call.output.playing.clear()
            call.output.pause_at = lambda frame: (
                isinstance(frame, TTSAudioRawFrame) and len(audio.append(frame) or audio) > 1
            )
            call.port.answer("del_1", "Bike 4 is free. Shall I book it?")
            await call.until(lambda: call.assistant.aggregation_string() == "Bike 4 is free.", "the first sentence")
            call.output.pause_at = lambda frame: False
            await call.say("mm-hmm")
            call.output.playing.set()
            await call.until(lambda: call.said()[-1:] == ["Shall I book it?"], "the rest, said once")
            await call.quiet()
            self.assertEqual(call.said(), ["Bike 4 is free." + TEXTS.cut_mark, "Shall I book it?"])
            self.assertEqual(len(call.chat.requests), 3)  # no writer call for the rest


class LeadInSkipTests(unittest.IsolatedAsyncioTestCase):
    """No lead-in in front of an answer that is due, beside another lead-in, or repeating the last line."""

    def lead_in(self, still: StillVoice, text: str = "One moment.", delegation_id: str = "del_2") -> list[str]:
        still.voice._lead_in(delegation_id, text)
        return [fields["reason"] for fields in still.kinds("acknowledgement.skipped")]

    def lead_ins(self, still: StillVoice) -> list[str]:
        return [line.text for line in still.voice.waiting_lines if line.kind is LineKind.LEAD_IN]

    async def test_no_lead_in_while_a_result_waits(self):
        still = StillVoice()
        still.voice.owed.append(owed_result("The gym opens at six."))
        self.assertEqual((self.lead_in(still), self.lead_ins(still)), (["answer_due"], []))

    async def test_no_lead_in_while_a_result_line_plays(self):
        still = StillVoice()
        still.voice.spoken.handed(result_line("The gym opens at six."))
        self.assertEqual(self.lead_in(still), ["answer_due"])

    async def test_no_lead_in_while_commentary_answering_a_client_delegation_waits(self):
        still = StillVoice()
        still.voice.commentary("Your table is ready.", "del_9")
        self.assertEqual(self.lead_in(still), ["answer_due"])

    async def test_two_delegations_decided_while_the_bot_is_silent_queue_one_lead_in(self):
        still = StillVoice()
        self.lead_in(still, "Let me check.", "del_1")
        self.assertEqual(self.lead_in(still, "Looking at that too.", "del_2"), ["lead_in_waiting"])
        self.assertEqual(self.lead_ins(still), ["Let me check."])

    async def test_a_second_request_after_a_lead_in_played_gets_its_own_lead_in(self):
        still = StillVoice()
        still.voice.last_handed = Line(LineKind.LEAD_IN, "Let me check.", ("del_1",))  # it played and settled
        self.assertEqual(self.lead_in(still, "Let's look at that one next."), [])
        self.assertEqual(self.lead_ins(still), ["Let's look at that one next."])

    async def test_a_lead_in_that_repeats_the_last_line_is_skipped(self):
        still = StillVoice()
        still.voice.last_handed = Line(LineKind.REPLY, "One moment.")
        self.assertEqual(self.lead_in(still, "One moment!"), ["same_as_last"])

    async def test_the_lead_in_skip_off_says_every_lead_in(self):
        still = StillVoice(answers_config(lead_in_skip=False))
        still.voice.owed.append(owed_result("The gym opens at six."))
        self.lead_in(still, "Let me check.", "del_1")
        self.lead_in(still, "Let me check.", "del_2")
        self.assertEqual(self.lead_ins(still), ["Let me check.", "Let me check."])
