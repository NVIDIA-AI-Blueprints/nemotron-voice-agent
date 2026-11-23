# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The voice in whole calls, in either talker style: each decision, what the backend sends, and what was heard."""

import asyncio
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from loguru import logger
from pipecat.frames.frames import (
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSTextFrame,
)
from pipecat.processors.aggregators import async_tool_messages
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.attachment import export_attachment, restore_attachment
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.config_manager.schema import MemoryConfig
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.media.services import build_tts
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.spoken import Line, LineKind, Outcome
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.talker import RESPONSE_FORMATS
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.voice import RESULT, message_tokens
from nemotron_voice_agent.pipecat.shared.delegation_contract import CALLER_CONTEXT
from tests.unit.pipecat.core_helpers import _timeout, chunk, tool_fragment
from tests.unit.pipecat.frontend_helpers import (
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


def agent_evidence(line: str) -> dict:
    return json.loads(line.partition(": ")[2])


def assert_presentation(test, request: dict, text: str) -> None:
    note = request["messages"][-2]["content"]
    test.assertIn('"recorded_sentence_prefix": ' + json.dumps(text), note)
    test.assertIn('"complete_conveyance":', note)


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
            await call.hear("Hello there. How can I help?")
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
            await call.hear("Great.")
            self.assertTrue(last(call.request(0)).startswith(f"{result}\n{TEXTS.caller_label}yes\n\n"))
            self.assertEqual(call.voice.conversation()[0], {"role": "user", "content": result, "kind": "result"})

    async def test_a_wait_says_nothing_and_the_next_turn_joins_it(self):
        async with CallHarness(decision("wait"), decision("speak", "Sure.")) as call:
            await call.say("Hold on")
            await call.until(lambda: call.chat.requests, "the first decision")
            await call.quiet()
            self.assertIsNone(await call.voice.next_call(call.context))
            self.assertFalse(call.port.pending_turns)
            self.assertEqual(call.port.failed_turns, 0)
            await call.say("okay go ahead")
            await call.hear("Sure.")
            runtime = last(call.request(1))
            self.assertTrue(runtime.startswith("Hold on\nokay go ahead\n\n"))
            self.assertIn(TEXTS.newest_turn.format(turn="okay go ahead"), runtime)
            self.assertEqual(call.port.inputs, [])

    async def test_a_note_no_run_took_up_leaves_at_the_next_run(self):
        async with CallHarness(decision("speak", "Sure.")) as call:
            note = {"role": "developer", "content": TEXTS.empty_turn_note}
            call.context.add_message(note)
            await call.say("Tea, please.")
            await call.hear("Sure.")
            self.assertNotIn(TEXTS.empty_turn_note, str(call.request(0)["messages"]))
            self.assertNotIn(note, call.context.get_messages())

    async def test_a_barge_in_mid_decision_decides_both_turns_together(self):
        held = asyncio.Event()
        first = [chunk('{"action":"speak","spe'), held, chunk('ech":"We have Earl Grey."}'), chunk(finish="stop")]
        async with CallHarness(first, decision("speak", "Coffee it is.")) as call:
            await call.say("Tell me about your teas.")
            await call.until(lambda: call.chat.requests, "the first decision under way")
            self.assertEqual(len(call.port.pending_turns), 1)
            await call.say("Actually, a coffee.")
            await call.until(lambda: len(call.chat.requests) == 2, "the decision on both turns")
            held.set()
            await call.hear("Coffee it is.")
            await call.quiet()
            self.assertEqual(call.tts.spoken, ["Coffee it is."])
            self.assertFalse(call.port.pending_turns)
            turns = TEXTS.newest_turn.format(turn="Tell me about your teas. Actually, a coffee.")
            self.assertIn(turns, last(call.request(1)))

    async def test_a_failed_routing_call_delegates_or_holds_the_turn(self):
        for failure in (_timeout(), [chunk("Not a routing decision.", finish="stop")]):
            with self.subTest(failure=type(failure).__name__):
                async with CallHarness(_timeout(), failure, decision("speak", "Hi.")) as call:
                    await call.say("What teas?")
                    await call.until(lambda: call.port.inputs, "the delegation the fallback makes")
                    self.assertTrue(call.talker.is_usable)
                    self.assertTrue(call.errors)
                    self.assertEqual(call.port.failed_turns, 0)
                    await call.say("Stop that work.")
                    await call.hear(LINES.holding)
                    self.assertEqual(call.port.failed_turns, 1)
                    self.assertFalse(call.port.pending_turns)
                    await call.say("Hello")
                    await call.hear(LINES.holding, "Hi.")
                    self.assertEqual(len(call.port.inputs), 1)

    async def test_a_failed_routing_call_waits_with_open_work_and_no_holding_line(self):
        async with CallHarness(decision("delegate", ""), _timeout(), decision("speak", "Hi.")) as call:
            call.voice.lines = LINES.model_copy(update={"holding": ""})
            events = observed(call)
            await call.say("Find my order.")
            await call.until(lambda: call.port.inputs, "the original delegation")
            await call.say("I have another question.")
            await call.until(lambda: ("talker.waited", {}) in events, "the silent routing fallback")
            self.assertEqual(len(call.port.inputs), 1)
            self.assertEqual(call.port.failed_turns, 1)
            self.assertFalse(call.port.pending_turns)
            self.assertEqual(call.said(), [])
            await call.say("Hello")
            await call.hear("Hi.")

    async def test_an_instruction_is_pinned_and_a_blank_one_does_nothing(self):
        for style, reply in (("json", decision("speak", "Of course.")), ("tools", [chunk("Of course.")])):
            with self.subTest(style=style):
                async with CallHarness(reply, reply, style=style) as call:
                    await call.voice.instruction(" ")
                    self.assertEqual((call.voice.instructions, carried(call)), (["You are Remy."], []))
                    await call.voice.instruction("Speak French.")
                    await call.hear("Of course.")
                    self.assertIn("You are Remy.\n\nSpeak French.", system(call.request(0)))
                    self.assertNotIn(PROMPTS.cascade.newest_turn.partition("{turn}")[0], last(call.request(0)))
                    self.assertIn("App instruction: Speak French.", carried(call)[0])
                    await call.say("Hello")
                    await call.until(lambda: len(call.chat.requests) == 2, "the caller's routing call")
                    self.assertIn(PROMPTS.cascade.newest_turn.format(turn="Hello"), last(call.request(1)))
                    self.assertIn("You are Remy.\n\nSpeak French.", system(call.request(1)))


class DelegationTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_multichoice_update_keeps_every_fact_and_question_when_wording_fails(self):
        update = (
            "Thursday at 09:15 costs $20. Friday at 17:45 costs $25. "
            "Which appointment would you like? Should I use your original payment method?"
        )
        logged = []
        self.addCleanup(logger.remove, logger.add(logged.append, level="DEBUG", format="{message}"))
        for style in ("json", "tools"):
            first = decision("delegate") if style == "json" else tool_call("delegate", call_id="first")
            failed = decision("speak", "Thursday at 09:15 costs $20.", finish="length")
            with self.subTest(style=style):
                async with CallHarness(first, failed if style == "json" else _timeout(), style=style) as call:
                    await call.say("Check the available appointments.")
                    await call.until(lambda: call.port.inputs, "the delegation")
                    call.port.answer("del_1", update)
                    await call.hear(update)
                    writer = call.request(1)
                    if style == "json":
                        self.assertFalse(writer["stream"])
                        self.assertEqual(enum(writer), ["speak"])
                        self.assertTrue(writer["response_format"]["json_schema"]["strict"])
                    self.assertIn("You are Remy.", system(writer))
                    self.assertIn(update, last(writer))
                    self.assertEqual(len(call.port.inputs), 1)
                    self.assertEqual(len(call.chat.requests), 2)
        self.assertTrue(any("writer reply (one-shot, finish_reason=length," in line for line in logged))

    async def test_an_early_acknowledgement_carries_the_unheard_question_boundary(self):
        question = "Should I place the order?"
        for style in ("json", "tools"):
            for heard in (False, True):
                first = decision("delegate") if style == "json" else tool_call("delegate", call_id="first")
                next_turn = decision("delegate") if style == "json" else tool_call("delegate", call_id="next")
                script = (first, decision("speak", question), next_turn, decision("speak", question))
                with self.subTest(style=style, heard=heard):
                    async with CallHarness(*script, style=style) as call:
                        await call.delegate("Please check my order.")
                        if heard:
                            call.port.answer("del_1", question)
                            await call.hear(question)
                            await call.quiet()
                            await call.say("Okay")
                        else:

                            async def answer_while_caller_speaks():
                                call.port.answer("del_1", question)
                                await call.until(lambda: len(call.chat.requests) == 2, "the question worded early")
                                self.assertEqual(call.said(), [])

                            await caller_speaks(call, "Okay", answer_while_caller_speaks)
                        await call.until(lambda: len(call.port.inputs) == 2, "the acknowledgement handed over")
                        text = call.port.text()
                        self.assertIn("Caller: Okay", text)
                        self.assertEqual(TEXTS.transcript.unheard in text, not heard)
                        if not heard:
                            self.assertNotIn("Agent: " + question, text)

    async def test_a_refused_delegation_says_the_refusal_only(self):
        for style, reply, settled in (
            ("json", decision("delegate", "Let me check."), []),
            ("tools", tool_call("delegate", LEAD_IN), [{"status": "refused"}]),
        ):
            with self.subTest(style=style):
                async with CallHarness(reply, style=style, port=Port(refuse=True)) as call:
                    await call.say("What teas do you have?")
                    await call.hear(LINES.refusal)
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
            # Actual frontend wording is retained for both the lead-in and completed backend-result speech.
            evidence = [agent_evidence(line) for line in second.splitlines() if line.startswith(f"{WORDS.agent}: ")]
            self.assertEqual(
                [(row["kind"], row["recorded_sentence_prefix"]) for row in evidence],
                [("lead_in", "Let me check."), ("result", "We have Earl Grey.")],
            )
            self.assertTrue(all(row["complete_conveyance"] for row in evidence))
            self.assertIn(f"{WORDS.caller}: Order one, please.", second)
            self.assertNotIn("What teas do you have?", second)

    async def test_a_failed_delegation_tries_one_task_bound_wording_then_uses_its_fixed_fallback(self):
        async with CallHarness(decision("delegate", "")) as call:
            await call.say("Place my order.")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.fail("del_1")
            await call.hear(LINES.apology)
            await call.quiet()
            self.assertEqual(len(call.chat.requests), 2)
            self.assertIn('Regarding the request "Place my order."', last(call.request(1)))
            self.assertNotIn("del_1", call.voice.worknote.entries)

    async def test_valid_writer_wording_does_not_require_paraphrasing_or_new_words(self):
        for update, words in (
            ("**Teas**: Earl Grey", "teas: earl grey."),
            ("Earl Grey is available.", "Earl Grey is available."),
            ("We still have Earl Grey.", "Hi there."),
        ):
            with self.subTest(words=words):
                seed = [*SEED, {"role": "assistant", "content": "Hi there."}]
                async with CallHarness(decision("delegate", ""), decision("speak", words), seed=seed) as call:
                    await call.say("What teas?")
                    await call.until(lambda: call.port.inputs, "the delegation")
                    call.port.answer("del_1", update)
                    await call.until(lambda words=words: call.tts.spoken == [words], "the Writer's exact wording")
                    await call.until(lambda: not call.voice.worknote.answer_ready(), "the actual Writer result settled")

    async def test_valid_writer_silence_journals_its_items_and_preserves_a_fixed_notice(self):
        async with CallHarness(decision("delegate"), decision("speak", "")) as call:
            await call.say("You may omit these already-known results.")
            await call.until(lambda: call.port.inputs, "the source")
            call.port.answer("del_1", "Old address reference.", "Independent already-known balance.")
            fixed = "The delivery change could not be confirmed."
            call.voice.notice(None, fixed)
            await call.hear(fixed)
            await call.until(lambda: not call.voice.worknote.answer_ready(), "presentation disposition")
            await call.quiet()
            self.assertEqual((call.said(), call.tts.spoken, len(call.chat.requests)), ([fixed], [fixed], 2))
            self.assertNotIn(fixed, last(call.request(1)))
            [entry] = [entry for entry in call.voice.heard_log.entries if entry.presentation == "not_conveyed"]
            self.assertIsNone(entry.line)
            disposition = json.loads(entry.text)["frontend_presentation"]
            self.assertEqual((disposition["state"], disposition["reason"]), ("not_conveyed", "writer_omitted"))
            self.assertEqual(
                disposition["items"],
                [
                    {"delegation_id": "del_1", "item_id": f"msg_del_1_{n}", "branch_ref": None, "kind": "result"}
                    for n in range(2)
                ],
            )
            self.assertIn("Old address reference.", str(call.context.get_messages()))
            self.assertIn("Independent already-known balance.", str(call.context.get_messages()))

    async def test_failed_mixed_wording_withholds_old_source_without_losing_current_independent_result(self):
        cases = ((_timeout(), False), ([chunk("not a decision", finish="stop")], False), (_timeout(), True))
        for failure, late_app_input in cases:
            with self.subTest(failure=type(failure).__name__, late_app_input=late_app_input):
                async with CallHarness(decision("delegate"), decision("delegate"), failure) as call:
                    at_idle(call)
                    original_hand, injected = call.voice.hand, False

                    def hand(line, *, late_app_input=late_app_input, original_hand=original_hand):
                        nonlocal injected
                        if late_app_input and not injected and line.text == "Your balance is $25.":
                            injected = True
                            call.voice.thinking("The pending balance is no longer current; keep it silent.")
                        return original_hand(line)

                    call.voice.hand = hand
                    await call.delegate("Check the old address.")
                    await call.delegate("Correct the address, and check my independent balance.", count=2)
                    call.port.answer("del_1", "The old address is 134 Elm Street.")
                    call.port.answer("del_2", "Your balance is $25.")
                    await call.until(lambda: not call.voice.worknote.answer_ready(), "actual presentation settlement")
                    await call.quiet()
                    self.assertEqual(call.said(), [] if late_app_input else ["Your balance is $25."])
                    self.assertFalse(any("134" in text for text in call.tts.spoken))
                    entries = [e for e in call.voice.heard_log.entries if e.presentation == "not_conveyed"]
                    dispositions = [json.loads(entry.text)["frontend_presentation"] for entry in entries]
                    self.assertTrue(all(row["reason"] == "writer_failed" for row in dispositions))
                    self.assertEqual(
                        [item["delegation_id"] for row in dispositions for item in row["items"]],
                        ["del_1", "del_2"] if late_app_input else ["del_1"],
                    )
                    self.assertEqual(len(call.chat.requests), 3)

    async def test_an_obsolete_address_cannot_replace_the_writers_repeated_correction_lead_in(self):
        old, lead_in = "Your order will deliver to 134 Elm Street.", "Let's fix that address now."
        corrected = "The address is now 1234 Elm Street."
        script = (
            decision("delegate"),
            decision("speak", old),  # Prepared while the caller speaks; invalidated before speech by the correction.
            decision("delegate", lead_in),
            decision("speak", lead_in),
            decision("speak", lead_in),  # Re-worded after the accepted source/lead-in changes the early stamp.
            decision("speak", corrected),
            decision("delegate"),
        )
        async with CallHarness(*script) as call:
            await call.delegate("Confirm my delivery address.")

            async def old_result_arrives():
                call.port.answer("del_1", old)
                await call.until(lambda: len(call.chat.requests) == 2, "old wording prepared while caller speaks")

            call.output.playing.clear()
            await caller_speaks(call, "Sorry, it should be 1234, not 134. Please fix that.", old_result_arrives)
            await call.until(lambda: len(call.chat.requests) == 5, "revised wording while its lead-in is pending")
            writer = call.request(4)
            self.assertIn("1234, not 134", writer["messages"][-2]["content"])
            self.assertNotIn({"role": "assistant", "content": lead_in}, writer["messages"])
            self.assertIn(old, last(writer))
            call.output.playing.set()
            await call.hear(lead_in, lead_in)
            self.assertFalse(any("134 Elm Street" in text for text in call.tts.spoken))
            call.port.answer("del_2", corrected)
            await call.until(lambda: call.said()[-1:] == [corrected], "corrected backend result")
            await call.until(
                lambda: any(line.text == corrected for line in call.voice.spoken.presentations),
                "the corrected result's playback receipt",
            )
            result_lines = [line for line in call.voice.spoken.presentations if line.items]
            self.assertEqual(
                [(line.text, line.delegation_ids, [item.said.item_id for item in line.items]) for line in result_lines],
                [(lead_in, ("del_1",), ["msg_del_1_0"]), (corrected, ("del_2",), ["msg_del_2_0"])],
            )
            await call.delegate("What address was confirmed?", count=3)
            receipts = [
                agent_evidence(line) for line in call.port.text().splitlines() if line.startswith(WORDS.agent + ": ")
            ]
            self.assertEqual(
                [row["recorded_sentence_prefix"] for row in receipts if row["kind"] == "result"], [lead_in, corrected]
            )
            self.assertTrue(all(row["complete_conveyance"] for row in receipts))

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
            await call.hear("One moment.")
            call.port.started("del_1")
            call.port.result("del_1", "Tea: yes.", "msg_1")
            await call.until(lambda: len(call.chat.requests) == 2, "the relay under way")
            call.voice.handed_over("del_1", "del_2")
            held.set()
            await call.hear("One moment.", "Yes, tea.")
            for request in call.chat.requests[1:]:
                self.assertIn(TEXTS.update_label + "Tea: yes.", last(request))
                assert_presentation(self, request, "One moment.")
            self.assertIsNone(call.voice.spoken.told_text("del_2"))
            await call.quiet()
            result = {"role": "developer", "content": TEXTS.result_label + "Tea: yes."}
            self.assertEqual(call.context.get_messages()[-2:], [result, {"role": "assistant", "content": "Yes, tea."}])

    async def test_commentary_is_worded_and_thinking_stays_silent(self):
        async with CallHarness(decision("speak", "Your table is ready.")) as call:
            call.voice.thinking("The caller is a regular.")
            call.voice.typed("Is table 4 free?")
            call.voice.commentary("Table 4 is ready.")
            await call.hear("Your table is ready.")
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
            self.assertNotIn({"role": "assistant", "content": "Let me check."}, relay["messages"])
            newest = TEXTS.newest_turn.format(turn="What teas?")
            self.assertEqual(last(relay), f"{TEXTS.answer_cue}\n\n{newest}\n\n{TEXTS.update_label}Teas: Earl Grey")
            assert_presentation(self, relay, "")
            self.assertNotIn("Let me check.", relay["messages"][-2]["content"])
            self.assertIn('"complete_conveyance": false', relay["messages"][-2]["content"])
            self.assertEqual(call.said(), [])
            call.output.playing.set()
            await call.hear("Let me check.", "We have Earl Grey.")
            await call.quiet()
            self.assertEqual(len(call.chat.requests), 2)

    async def test_an_early_wording_is_redone_after_a_reply_moves_it(self):
        held = asyncio.Event()
        original = decision("speak", "Earl Grey is in stock.")
        original.insert(1, held)
        script = [
            decision("delegate", ""),
            original,
            decision("speak", "Let me see."),
            decision("speak", "And Earl Grey."),
        ]
        async with CallHarness(*script) as call:
            await call.say("What teas?")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.answer("del_1", "Teas: Earl Grey")
            await call.until(lambda: len(call.chat.requests) == 2, "the first Writer request")

            async def interrupted():
                call.voice._move()
                await asyncio.sleep(0)
                self.assertEqual(len(call.chat.requests), 2)

            call.output.playing.clear()
            await caller_speaks(call, "Any news?", interrupted)
            await call.until(lambda: len(call.chat.requests) == 4, "the result worded again before reply playback ends")
            held.set()
            writer = call.request(3)
            self.assertNotIn({"role": "assistant", "content": "Let me see."}, writer["messages"])
            self.assertIn("Any news?", "\n".join(message["content"] for message in writer["messages"][1:-1]))
            call.output.playing.set()
            await call.hear("Let me see.", "And Earl Grey.")
            self.assertEqual(len(call.chat.requests), 4)

    async def test_a_result_nobody_heard_is_said_again_once(self):
        # The unheard line wrote its result ahead of it; the retry moves the result to right before its new line.
        script = (decision("delegate", ""), decision("speak", "We have tea."), decision("speak", "Tea."))
        async with CallHarness(*script, decision("wait"), decision("speak", "Tea.")) as call:
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
            await call.hear("Tea.")
            self.assertEqual(len(call.chat.requests), 5)
            self.assertIn(TEXTS.update_label + "Earl Grey.", last(call.request(2)))
            self.assertIn("sorry?", last(call.request(3)))
            self.assertIn(
                "sorry?",
                "\n".join(message["content"] for message in call.request(4)["messages"][:-1]),
                "re-wording retains the actually arrived caller turn before its separate update",
            )
            await call.quiet()
            contents = [message["content"] for message in call.context.get_messages()]
            self.assertEqual(contents, ["What teas?", "sorry?", result["content"], "Tea."])
            [request], _ = call.voice.heard_log.delegation_input()
            context = request[CALLER_CONTEXT]
            [unheard, replayed] = context["playback"]
            [(caller_sequence, _)] = context["caller_turns"]
            self.assertEqual(
                [
                    (row["playback_state"], row["audio_started"], row["recorded_sentence_prefix"])
                    for row in (unheard, replayed)
                ],
                [("not_heard", False, ""), ("complete", True, "Tea.")],
            )
            self.assertNotEqual(unheard["line_id"], replayed["line_id"])
            self.assertEqual(unheard["results"], replayed["results"])
            self.assertEqual(unheard["results"][0]["item_id"], "msg_del_1_0")
            self.assertLess(unheard["sequence"], caller_sequence)
            self.assertLess(caller_sequence, replayed["sequence"])
            self.assertNotIn("We have tea.", request["content"][0]["text"])

    async def test_a_lead_in_is_said_though_its_result_comes_at_once(self):
        class AtOnce(Port):
            async def delegate(self, input):
                delegation_id = await super().delegate(input)
                asyncio.get_running_loop().call_soon(self.answer, delegation_id, "Teas: Earl Grey")
                return delegation_id

        script = (decision("delegate", "Let me check."), decision("speak", "We have Earl Grey."))
        async with CallHarness(*script, port=AtOnce()) as call:
            await call.say("What teas?")
            await call.hear("Let me check.", "We have Earl Grey.")
            await call.quiet()
            self.assertEqual(len(call.chat.requests), 2)
            self.assertIn(TEXTS.update_label + "Teas: Earl Grey", last(call.request(1)))
            assert_presentation(self, call.request(1), "")
            self.assertIn('"kind": "lead_in"', call.request(1)["messages"][-2]["content"])

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
    async def test_a_result_with_no_audio_after_retry_settles_without_more_wording(self):
        result = "Your order is ready."
        tts = ToneTTS(silent={result}, stop_frame_timeout_s=0.2)
        script = (decision("delegate"), decision("speak", result), decision("speak", result))
        async with CallHarness(*script, tts=tts) as call:
            await call.say("Check my order.")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.answer("del_1", result)
            await call.until(lambda: len(call.errors) >= 2, "both no-audio attempts")
            await call.quiet()
            self.assertEqual([text for _, text in tts.spoken], [result, result])
            self.assertEqual((len(call.port.inputs), len(call.chat.requests), call.said()), (1, 2, []))
            self.assertTrue(tts.is_usable)
            self.assertFalse(call.voice.worknote.answer_ready())
            [entry] = [entry for entry in call.voice.heard_log.entries if entry.presentation == "not_conveyed"]
            receipt = json.loads(entry.text)["frontend_presentation"]
            self.assertEqual(receipt["reason"], "no_audio")
            [item] = receipt["items"]
            self.assertEqual((item["delegation_id"], item["item_id"]), ("del_1", "msg_del_1_0"))
            retained = [
                message
                for message in call.context.get_messages()
                if message.get("content") == TEXTS.result_label + result
            ]
            self.assertEqual(len(retained), 1)

    async def test_a_sentence_with_no_audio_is_left_out_of_its_line(self):
        tts = ToneTTS(silent={"Welcome to Bluebird."})
        line = "Welcome to Bluebird. How can I help you today?"
        async with CallHarness(decision("speak", line), tts=tts, output=TransportOutput(chunk_ms=50)) as call:
            await call.say("Hi")
            await call.hear("How can I help you today?")
            await call.quiet()
            self.assertEqual(len(tts.spoken), 2)
            evidence = agent_evidence(carried(call)[-1])
            self.assertEqual(evidence["recorded_sentence_prefix"], "How can I help you today?")
            self.assertEqual(evidence["playback_state"], "partial")

    async def test_a_result_tells_the_backend_only_what_played(self):
        tts, output = ToneTTS(silent={"Your total is five dollars."}), TransportOutput(chunk_ms=50)
        async with CallHarness(decision("delegate", ""), _timeout(), tts=tts, output=output) as call:
            initial_evidence = []
            telemetry = []
            call.voice.observe = lambda kind, **fields: telemetry.append((kind, fields))
            heard = call.voice.heard

            def record(line):
                heard(line)
                initial_evidence.append(agent_evidence(carried(call)[-1]))

            call.voice.heard = record
            await call.say("What do I owe?")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.answer("del_1", "Your total is five dollars.", "Anything else?")
            await call.hear("Anything else?")
            await call.quiet()
            [evidence] = [agent_evidence(line) for line in carried(call)]
            self.assertEqual(evidence["recorded_sentence_prefix"], "Anything else?")
            self.assertIn("Your total is five dollars.", evidence["generated_text"])
            self.assertEqual(evidence["playback_state"], "partial")
            self.assertEqual(len(initial_evidence), 1)
            self.assertFalse(initial_evidence[0]["complete_conveyance"])
            self.assertEqual(initial_evidence[0]["recorded_sentence_prefix"], "Anything else?")
            self.assertEqual([fields["turn"] for kind, fields in telemetry if kind == "caller.turn.ended"], [1])
            [audio] = [fields for kind, fields in telemetry if kind == "line.audio_started"]
            assert (audio["after_turn"], audio["line_kind"], audio["result_delegations"]) == (1, "result", ["del_1"])

        failed = "Your total is five dollars."
        earlier, later = "Here is your update.", "Anything else?"
        result = f"{earlier} {failed} {later}"
        for mode, recorded in (("per_sentence", f"{earlier} {later}"), ("stitched", "")):
            with self.subTest(mode=mode):
                tts = build_tts({}, {"server": "magpie-tts:50051", "voice_id": "John", "synthesis_mode": mode})
                tts._service = SimpleNamespace()
                tts._create_synthesis_config = AsyncMock()

                async def synthesize(text, context_id, tts=tts):
                    if not tts.audio_context_available(context_id):
                        await tts.create_audio_context(context_id)
                        yield TTSStartedFrame(context_id=context_id)
                    interrupted = text.strip() == failed
                    samples = 1200 if interrupted else 2400
                    yield TTSAudioRawFrame(b"\x10\x10" * samples, 24000, 1, context_id=context_id)
                    if interrupted:
                        raise RuntimeError("synthesis failed after 50 ms of audio")

                def finish_stream(state):
                    state.response_queue.put_nowait(SimpleNamespace(audio=b"\x10\x10" * 1200))
                    state.response_queue.put_nowait(RuntimeError("synthesis failed after 50 ms of audio"))

                tts._run_tts_per_sentence = synthesize
                tts._synth_task_handler = AsyncMock()
                tts._signal_synthesis_close = finish_stream
                script = (decision("delegate"), decision("speak", result), decision("delegate"))
                async with CallHarness(*script, tts=tts, output=TransportOutput(chunk_ms=50)) as call:
                    await call.delegate("What do I owe?")
                    call.port.answer("del_1", result)
                    await call.until(
                        lambda: call.errors and call.voice.spoken.presentations, "synthesis error and playout"
                    )
                    self.assertEqual(call.said(), [recorded] if recorded else [])
                    self.assertEqual(len(call.errors), 1)
                    outcome = Outcome.SYNTHESIS_FAILED if mode == "stitched" else Outcome.HEARD
                    self.assertEqual(call.voice.spoken.presentations[-1].outcome, outcome)
                    await call.delegate("Can you confirm?", count=2)
                    playback = call.port.inputs[1][0][CALLER_CONTEXT]["playback"]
                    evidence, *unspoken = playback
                    self.assertEqual(evidence["playback_state"], "partial")
                    self.assertTrue(evidence["audio_started"])
                    self.assertFalse(evidence["complete_conveyance"])
                    self.assertEqual(evidence["recorded_sentence_prefix"], recorded)
                    self.assertEqual([item["item_id"] for item in evidence["results"]], ["msg_del_1_0"])
                    if mode == "stitched":
                        [receipt] = unspoken
                        self.assertEqual(receipt["playback_state"], "not_conveyed")
                        self.assertEqual([item["item_id"] for item in receipt["results"]], ["msg_del_1_0"])
                        self.assertEqual(tts._synth_task_handler.call_count, 1)
                    else:
                        self.assertFalse(unspoken)

    async def test_silent_lines_not_in_a_row_keep_the_call(self):
        # Pipecat's text-to-speech writes itself off at three audio contexts in a row with no audio.
        tts = ToneTTS(silent={"Hello."}, stop_frame_timeout_s=0.2)
        async with CallHarness(decision("speak", "Hello."), decision("speak", "Still there?"), tts=tts) as call:
            call.output.playing.clear()
            await call.say("Hi")
            await call.until(lambda: len(call.errors) == 1, "the line's audio context, ended with no audio")
            tts.silent = {"Still there?"}  # the line said once more gets its audio; the next line gets none
            call.output.playing.set()
            await call.hear("Hello.")
            await call.say("Hm")
            await call.until(lambda: len(call.errors) == 3, "the next line and its retry, with no audio")
            await call.quiet()
            spoken = [sentence for _, sentence in tts.spoken]
            self.assertEqual(spoken, ["Hello.", "Hello.", "Still there?", "Still there?"])
            self.assertTrue(tts.is_usable)
            self.assertFalse(call.run.done())
        text = "Your order is ready."
        tts = ToneTTS(silent={text}, stop_frame_timeout_s=0.2)
        async with CallHarness(decision("delegate"), decision("speak", text), tts=tts) as call:
            await call.say("Check my order.")
            await call.until(lambda: call.port.inputs, "the original backend request")
            call.output.playing.clear()
            call.port.answer("del_1", text)
            await call.until(lambda: len(call.errors) == 1, "the first no-audio result attempt")
            tts.silent.clear()
            call.output.playing.set()
            await call.hear(text)
            await call.quiet()
            self.assertEqual((len(call.chat.requests), len(call.port.inputs)), (2, 1))
            [played] = [line for line in call.voice.spoken.presentations if line.items]
            self.assertTrue(played.retried)
            self.assertEqual([item.said.item_id for item in played.items], ["msg_del_1_0"])
            self.assertFalse(call.voice.worknote.answer_ready())

    async def test_a_wordless_turn_that_cuts_a_line_is_decided_once(self):
        asked = "Sorry, could you say that again?"
        async with CallHarness(decision("speak", "First part. Second part."), decision("speak", asked)) as call:
            await first_sentence_played(call, lambda: call.say("Tell me"))
            await call.noise()
            call.output.playing.set()
            await call.hear("First part." + TEXTS.cut_mark, asked)
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
            actual = call.port.text().splitlines()
            [agent] = [line for line in actual if line.startswith(f"{WORDS.agent}: ")]
            evidence = agent_evidence(agent)
            self.assertEqual(evidence["recorded_sentence_prefix"], "First part.")
            self.assertEqual(evidence["playback_state"], "interrupted")
            self.assertEqual(
                actual,
                [
                    WORDS.header,
                    f"{WORDS.caller}: Tell me",
                    f"{WORDS.context}: The caller is a regular.",
                    agent,
                    f"{WORDS.caller}: I'll take the first one.",
                ],
            )

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
        words = "Sure, one moment."
        cases = (
            ("length", decision("speak", words, finish="length")),
            ("timeout", [*decision("speak", words)[:-1], asyncio.Event()]),
        )
        reliability = CONFIG.reliability.model_copy(update={"talker_attempt_timeout_seconds": 0.2})
        for reason, partial in cases:
            with self.subTest(reason=reason):
                script = (decision("delegate", ""), partial, decision("delegate", ""))
                async with CallHarness(*script, config=settings(reliability=reliability)) as call:
                    await call.say("Order a latte.")
                    await call.until(lambda: call.port.inputs, "the first delegation")
                    await call.say("Stop that work.")
                    await call.hear(words)
                    await call.until(lambda: call.port.failed_turns == 1, "the incomplete routing settles")
                    self.assertFalse(call.port.pending_turns)
                    await call.delegate("Thanks.", count=2)
                    [evidence] = [
                        agent_evidence(line)
                        for line in call.port.text().splitlines()
                        if line.startswith(f"{WORDS.agent}: ")
                    ]
                    self.assertEqual(
                        (evidence["recorded_sentence_prefix"], evidence["complete_conveyance"]), (words, True)
                    )

    async def test_a_cut_short_spoken_reply_does_not_fall_back_to_delegation(self):
        script = (decision("speak", "Sure, one moment.", finish="length"), decision("delegate", ""))
        async with CallHarness(*script) as call:
            await call.say("What can you do?")
            await call.hear("Sure, one moment.")
            await call.quiet()
            self.assertEqual(call.port.inputs, [])
            await call.delegate("Find my order.")
            self.assertIn("Caller: Find my order.", call.port.text())

    async def test_audio_before_a_completed_sentence_keeps_hearing_uncertain(self):
        for mode in ("per_sentence", "stitched"):
            with self.subTest(mode=mode):
                await self.check_audio_before_recorded_text(mode)

    async def check_audio_before_recorded_text(self, mode):
        proposal = "The charge is four dollars. Do you approve?"
        script = (decision("speak", proposal), decision("delegate", ""))
        tts = ToneTTS(word_ms=250)
        tts.settings.synthesis_mode = mode
        async with CallHarness(*script, tts=tts) as call:
            call.output.playing.clear()
            call.output.pause_at = lambda frame: isinstance(frame, TTSTextFrame)
            await call.say("How much will it cost?")
            lines = call.voice.spoken.lines
            await call.until(lambda: any(line.outcome is Outcome.PLAYING for line in lines.values()), "its audio")
            call.output.pause_at = lambda frame: False
            await call.say("Yes.")
            call.output.playing.set()
            await call.until(lambda: call.port.inputs, "the delegation")
            self.assertEqual(call.said(), [])
            transcript = [WORDS.header, f"{WORDS.caller}: How much will it cost?", f"{WORDS.caller}: Yes."]
            actual = call.port.text().splitlines()
            evidence = agent_evidence(actual[2])
            self.assertNotIn("generated_text", evidence)
            self.assertNotIn(proposal, call.port.text())
            self.assertEqual(evidence["recorded_sentence_prefix"], "")
            self.assertTrue(evidence["audio_started"])
            self.assertFalse(evidence["complete_conveyance"])
            self.assertEqual(evidence["playback_state"], "interrupted")
            self.assertEqual(actual[:2] + actual[3:], transcript)

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
            await call.hear(cut, "Earl Grey and green tea.")
            self.assertIn({"role": "assistant", "content": cut}, call.request(2)["messages"])
            await call.quiet()
            self.assertEqual(len(call.chat.requests), 3)


class WorkTests(unittest.IsolatedAsyncioTestCase):
    async def test_progress_for_two_jobs_uses_only_their_current_updates(self):
        async with CallHarness(
            decision("speak", "Checking the fare and coverage."),
            config=settings(delegation={"progress_after_seconds": 0.1}),
        ) as call:
            work = call.voice.worknote
            for key, request, update in (
                ("D1", "Change my secret booking.", "Checking the reservation fare."),
                ("D2", "Cancel my private policy.", "Reading the coverage terms."),
            ):
                work.confirm(work.reserve(request), key)
                work.running(key)
                work.progress(key, update)
            await call.hear("Checking the fare and coverage.")
            await call.until(lambda: set(call.voice.spoken.told) == {"D1", "D2"}, "both jobs' spoken progress")
            [sent] = call.chat.requests
            payload = sent["messages"][-1]["content"]
            assert "reservation fare" in payload and "coverage terms" in payload
            assert "secret booking" not in payload and "private policy" not in payload
            assert [(entry.news, entry.progress_offers) for entry in work.open()] == [(False, 1), (False, 1)]

    async def test_a_quiet_spell_brings_a_progress_line(self):
        handoff, status = "Let me check.", "Still on it."
        script = (decision("delegate", handoff), *(decision("speak", line) for line in (handoff, status, status)))
        async with CallHarness(*script, config=settings(delegation={"progress_after_seconds": 0.2})) as call:
            await call.say("Place my order.")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.started("del_1")
            await call.hear(handoff)
            await asyncio.sleep(0.35)
            self.assertEqual(len(call.chat.requests), 1)  # A request/start is not backend progress evidence.
            news = {"type": "message", "status": "completed", "phase": "commentary", "content": "Checking the date."}
            call.port.send("del_1", "response.output_item.done", item=news)
            await call.until(lambda: len(call.chat.requests) == 2 and call.voice.call is None, "progress wording")
            self.assertEqual(call.said(), [handoff])
            progress = call.request(1)
            self.assertIn(TEXTS.progress, system(progress))
            self.assertIn(TEXTS.update_label + TEXTS.work_in_progress.split("{")[0], last(progress))
            self.assertNotIn(TEXTS.answer_cue, last(progress))
            await asyncio.sleep(0.6)
            self.assertEqual(len(call.chat.requests), 2)
            for count, update in enumerate(("Checking the collection slot.", "Checking the delivery fee."), 1):
                call.port.send("del_1", "response.output_item.done", item={**news, "content": update})
                await call.hear(handoff, *([status] * count))

    async def test_commentary_replaces_a_progress_line_being_worded(self):
        for commentary, spoken in ((False, "Checking the newly available date."), (True, "Table 4 is ready.")):
            with self.subTest(commentary=commentary):
                await self.check_progress_replacement(commentary, spoken)

    async def check_progress_replacement(self, commentary, spoken):
        held = asyncio.Event()
        progress = [chunk('{"action":"speak",'), held, chunk('"speech":"Still on it."}'), chunk(finish="stop")]
        script = (decision("delegate"), progress, decision("speak", spoken))
        async with CallHarness(*script, config=settings(delegation={"progress_after_seconds": 0.2})) as call:
            await call.say("Place my order.")
            await call.until(lambda: call.port.inputs, "the delegation")
            call.port.started("del_1")
            work = call.voice.worknote
            work.progress("del_1", "Checking the date.")
            await call.until(lambda: len(call.chat.requests) == 2, "the progress line under way")
            if commentary:
                call.voice.commentary(spoken)
            else:
                work.progress("del_1", "A date was found; checking its price.")
            held.set()
            await call.hear(spoken)
            await call.quiet()
            self.assertNotIn("Still on it.", call.tts.spoken)
            if not commentary:
                self.assertIn("A date was found", last(call.request(2)))
                self.assertFalse(work.entries["del_1"].news)

    async def test_the_greeting_is_worded_or_uses_its_fallback_and_retains_only_played_words(self):
        for script, said in (((decision("speak", "Hi, I'm Remy."),), "Hi, I'm Remy."), ((_timeout(),), LINES.greeting)):
            with self.subTest(said=said):
                async with CallHarness(*script, decision("wait")) as call:
                    call.talker.response_format = RESPONSE_FORMATS["none"]
                    call.voice.greet(LINES.intro)
                    await call.until(lambda said=said: call.said() == [said], "the greeting")
                    greeting = call.request(0)
                    self.assertIsNone(greeting.get("response_format"))
                    self.assertIn('{"action":"speak","speech":""}', system(greeting))
                    self.assertIn(LINES.intro, last(greeting))
                    await call.say("What tea do you have?")
                    await call.until(lambda: len(call.chat.requests) == 2, "routing after the greeting")
                    history = call.request(1)["messages"]
                    self.assertFalse(any(LINES.intro in message["content"] for message in history))
                    self.assertIn({"role": "assistant", "content": said}, history)


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def budget(context_tokens: int, compact_at: float):
        return settings(memory=MemoryConfig(context_tokens=context_tokens, compact_at=compact_at))

    def fit_first_turn(self, call: CallHarness, turn: str, compact_at: float) -> int:
        request = call.voice.layout.routing(SEED[0]["content"], call.voice.worknote.line(), turn)
        messages = request.messages([{"role": "user", "content": turn}])
        limit = sum(map(message_tokens, messages))
        call.voice.config = self.budget(limit, compact_at)
        return limit

    async def test_the_bound_keeps_the_newest_group_and_unsaid_items(self):
        call = CallHarness(config=self.budget(1000, 0.8))
        result = {"role": "developer", "content": TEXTS.result_label + "Order 42 is ready."}
        call.voice.kinds[id(result)] = (result, RESULT)
        call.voice.unsaid.add(id(result))
        newest = "newest " * 700
        older = {"role": "user", "content": "Earlier request."}
        call.context.set_messages([older])
        self.assertIsNotNone(call.voice._routing([older]).request)
        call.context.set_messages(
            [older, result, {"role": "assistant", "content": "Reply."}, {"role": "user", "content": newest}]
        )
        self.assertFalse(call.voice.bound())
        self.assertIn(result, call.context.get_messages())
        self.assertIn(older, call.context.get_messages())
        self.assertEqual(call.voice.instructions, ["You are Remy."])
        call.voice.input_delivery[id(older)] = {"state": "never_forwarded"}
        turns = call.voice._undecided()
        call.voice.hold_pending_caller_turns()
        reply = await call.voice.next_call(call.context)
        self.assertEqual(reply.text, LINES.apology)
        self.assertFalse(call.port.pending_turns)
        self.assertEqual(call.port.failed_turns, len(turns))
        self.assertFalse(call.voice._undecided())
        self.assertIn(newest, [message["content"] for message in call.context.get_messages()])
        self.assertIn(id(result), call.voice.unsaid)
        self.assertEqual((call.chat.requests, call.port.inputs), ([], []))
        carried, _ = call.voice.heard_log.delegation_input()
        self.assertEqual(carried[0][CALLER_CONTEXT]["caller_turns"], [])
        restored = CallHarness(config=self.budget(40, 0.8))
        restore_attachment(restored.voice, export_attachment(call.voice))
        self.assertFalse(restored.voice._undecided())

        async with CallHarness(decision("delegate", ""), config=self.budget(2000, 0.8)) as call:
            await call.say(newest.strip())
            await call.hear(LINES.apology)
            self.assertEqual(call.port.failed_turns, 1)
            self.assertFalse(call.port.pending_turns)
            self.assertFalse(call.chat.requests)
            await call.delegate("Check it again.")
            self.assertNotIn(newest.strip(), call.port.text())
            self.assertNotIn(newest.strip(), [message["content"] for message in call.context.get_messages()])
            self.assertIn('{"frontend_routing": {"state": "not_routed", "reason": "context_budget"}}', call.port.text())
            self.assertEqual(
                [words for _, words in call.port.inputs[0][0][CALLER_CONTEXT]["caller_turns"]], ["Check it again."]
            )

    async def test_the_view_of_the_context_changes_nothing(self):
        messages = [
            {"role": "user", "content": "first " * 20},
            {"role": "assistant", "content": "Reply one."},
            {"role": "user", "content": "second"},
        ]
        call = CallHarness(config=self.budget(40, 0.5))
        call.context.set_messages(list(messages))
        call.voice.decided[id(messages[0])] = messages[0]
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
        call = CallHarness(config=self.budget(4 * tokens, 0.25))
        call.context.set_messages(list(messages))
        call.voice.decided[id(messages[0])] = messages[0]
        call.voice.bound()
        self.assertEqual(call.context.get_messages(), messages)
        call.voice.hand(Line(LineKind.REPLY, "A long reply, " * 10))
        self.assertEqual([m["content"] for m in call.context.get_messages()], ["second"])

    async def test_a_streamed_reply_counts_whole_once_decided(self):
        turns = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "One."}]
        tokens = message_tokens(SEED[0]) + sum(map(message_tokens, [*turns, {"role": "user", "content": "second"}]))
        script = (decision("speak", "One."), decision("speak", "A long reply about the scones. " * 6))
        async with CallHarness(*script, config=self.budget(1000, (tokens + 100) / 4000)) as call:
            await call.say("first")
            await call.hear("One.")
            call.output.playing.clear()
            await call.say("second")
            await call.until(lambda: call.voice.call is None and call.voice.spoken.lines, "the long reply decided")
            self.assertEqual([m["content"] for m in call.context.get_messages()], ["second"])
            call.output.playing.set()

    async def test_a_turn_past_the_budget_drops_the_oldest_turn(self):
        async with CallHarness(decision("speak", "One."), decision("speak", "Two.")) as call:
            first, second = ("alpha " * 60).strip(), ("bravo " * 60).strip()
            limit = self.fit_first_turn(call, first, 1.0)
            await call.say(first)
            await call.hear("One.")
            await call.say(second)
            await call.hear("Two.")
            await call.quiet()
            self.assertEqual(len(call.chat.requests), 2)
            routing = call.request(1)
            self.assertIn("You are Remy.", system(routing))
            self.assertNotIn("alpha", str(routing["messages"]))
            self.assertIn("bravo", str(routing["messages"]))
            self.assertLessEqual(sum(map(message_tokens, routing["messages"])), limit)
            self.assertEqual([m["content"] for m in call.context.get_messages()], [second, "Two."])
            held = {id(message) for message in call.context.get_messages()}
            self.assertLessEqual(set(call.voice.decided) | set(call.voice.noted), held)

    async def test_the_bound_drops_a_call_with_its_result(self):
        script = (tool_call("wait"), tool_call("wait", call_id="call_1"))
        async with CallHarness(*script, style="tools") as call:
            first = ("Long details. " * 150).strip()
            self.fit_first_turn(call, first, 0.15)
            await call.say(first)
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
            await call.hear("Let me check.")
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
                    evidence = agent_evidence(
                        next(line for line in call.port.text().splitlines() if line.startswith(WORDS.agent + ": "))
                    )
                    self.assertNotIn("generated_text", evidence)
                    self.assertIn(evidence["recorded_sentence_prefix"], ("", words))
                    self.assertIn(evidence["playback_state"], ("queued", "playing", "complete"))
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
            await call.hear("Let me check.")
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
            await call.hear("Let me check.")
            await call.until(lambda: idle, "the caller's idle spell")
            self.assertEqual(results(call), [{"status": "delegated"}])
            self.assertEqual([work.id for work in idle[0]], ["del_1"])

    async def test_an_unusable_reply_holds_while_work_is_open(self):
        script = (tool_call("delegate"), tool_call("lookup_order", "{}", call_id="call_1"))
        async with CallHarness(*script, style="tools") as call:
            await call.say("Place my order.")
            await call.until(lambda: call.port.inputs, "the delegation")
            await call.say("Are you there?")
            await call.hear(LINES.holding)
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
            await call.hear("Let me check.")
            await call.quiet()
            kinds = [entry["kind"] for entry in call.voice.conversation()]
            self.assertEqual(kinds, ["caller", "call", "call_result", "spoken"])
            made, result = call.voice.conversation()[1:3]
            self.assertEqual(made["tool_calls"][0]["function"]["name"], "delegate")
            self.assertEqual(result["tool_call_id"], made["tool_calls"][0]["id"])
            await call.delegate("And a scone.", count=2)
            transcript = call.port.text()
            self.assertIn("And a scone.", transcript)
            self.assertNotIn("delegate", transcript)
            self.assertNotIn("status", transcript)
