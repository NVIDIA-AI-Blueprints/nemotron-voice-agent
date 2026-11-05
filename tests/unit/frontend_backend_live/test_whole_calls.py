# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""Whole calls through the frontend's own pipeline, played by Pipecat's ``run_test``.

Each test plays one call as a script of frames: the caller's turns, pauses, and the steps of the backend or the app.
Its checks read what the talker's context recorded as played, what each delegation carried, and what the talker was
asked.
"""

import asyncio
import itertools
import unittest
from dataclasses import replace

from pipecat.frames.frames import ProposedUserStartedSpeakingFrame, ProposedUserStoppedSpeakingFrame, TTSAudioRawFrame
from pipecat.tests.utils import SleepFrame

from gateways.live.client_delegation import ClientDelegation
from tests.unit.core_helpers import chunk
from tests.unit.frontend_helpers import CONFIG, PROMPTS, Act, CallHarness, caller, decision

TEXTS, LINES = PROMPTS.cascade, PROMPTS.lines
CALLER, AGENT = TEXTS.transcript.caller, TEXTS.transcript.agent


def last(request: dict) -> str:
    return request["messages"][-1]["content"]


class ClientView:
    """The session's delegation view in client mode: what the gateway sends the app."""

    input_ms = 0
    acknowledgement_delay_ms = 0

    def __init__(self):
        self.sent: list[tuple[str, dict]] = []

    async def emit(self, kind, **fields):
        self.sent.append((kind, fields))


class DelegationTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_delegations_each_carry_what_the_caller_heard_since_the_one_before(self):
        call = CallHarness(
            decision("delegate", "Let me check."),
            decision("speak", "We have Earl Grey."),
            decision("delegate", "One moment."),
            decision("speak", "Your order is placed."),
        )
        await call.play(
            *caller("What teas do you have?"),
            SleepFrame(0.3),
            Act(lambda: call.port.answer("del_1", "Teas: Earl Grey")),
            SleepFrame(0.4),
            *caller("Order one, please."),
            SleepFrame(0.3),
            Act(lambda: call.port.answer("del_2", "Order placed.")),
            SleepFrame(0.5),
        )
        self.assertEqual(call.said(), ["Let me check.", "We have Earl Grey.", "One moment.", "Your order is placed."])
        first, second = call.port.text(0), call.port.text(1)
        self.assertIn(f"{CALLER}: What teas do you have?", first)
        # The lead-in was said after the first input was built; a result the caller heard in full stays out.
        self.assertIn(f"{AGENT}: Let me check.", second)
        self.assertIn(f"{CALLER}: Order one, please.", second)
        self.assertNotIn("What teas do you have?", second)
        self.assertNotIn("We have Earl Grey.", second)

    async def test_in_client_mode_the_app_does_the_work_and_the_talker_words_its_commentary(self):
        view, port = ClientView(), ClientDelegation()
        await port.open(view)
        call = CallHarness(decision("delegate", "One moment."), decision("speak", "It is sunny today."), port=port)

        def app_answers():
            [(_, announced)] = view.sent
            call.voice.commentary("Weather: sunny, 72 degrees.", announced["delegation"]["id"])

        await call.play(*caller("What is the weather?"), SleepFrame(0.3), Act(app_answers), SleepFrame(0.5))
        [(kind, announced)] = view.sent
        self.assertEqual((kind, announced["delegation"]["target"]), ("session.delegation.created", "client"))
        self.assertEqual(call.said(), ["One moment.", "It is sunny today."])
        told = TEXTS.writer_notes.told.format(told="One moment.")
        self.assertTrue(last(call.request(1)).endswith(TEXTS.update_label + "Weather: sunny, 72 degrees." + told))
        self.assertEqual(len(call.chat.requests), 2)  # the routing call and the relay: no backend runs here

    async def test_a_failed_delegation_says_its_fixed_line_with_no_model_call(self):
        call = CallHarness(decision("delegate", ""))
        await call.play(*caller("Book it."), SleepFrame(0.3), Act(lambda: call.port.fail("del_1")), SleepFrame(0.4))
        self.assertEqual(call.said(), [LINES.apology])
        self.assertEqual(len(call.chat.requests), 1)


class TurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_wait_says_nothing_and_the_next_turn_is_decided_with_the_one_it_waited_on(self):
        call = CallHarness(decision("wait"), decision("speak", "Sure."))
        await call.play(*caller("Hold on"), SleepFrame(0.3), *caller("okay go ahead"), SleepFrame(0.4))
        self.assertEqual(call.said(), ["Sure."])
        runtime = last(call.request(1))
        self.assertTrue(runtime.startswith("Hold on\nokay go ahead\n\n"))
        self.assertIn(TEXTS.newest_turn.format(turn="okay go ahead"), runtime)
        self.assertEqual(call.port.inputs, [])

    async def test_a_barge_in_in_the_middle_of_the_decision_cuts_it_and_both_turns_are_decided_together(self):
        held = asyncio.Event()  # the first decision stops in the middle of its JSON
        first = [chunk('{"action":"speak","spe'), held, chunk('ech":"We have Earl Grey."}'), chunk(finish="stop")]
        call = CallHarness(first, decision("speak", "Coffee it is."))
        await call.play(
            *caller("Tell me about your teas."),
            SleepFrame(0.2),
            *caller("Actually, a coffee."),
            SleepFrame(0.3),
            Act(held.set),
            SleepFrame(0.2),
        )
        self.assertEqual(call.tts.spoken, ["Coffee it is."])  # nothing of the cut decision was said
        turns = TEXTS.newest_turn.format(turn="Tell me about your teas. Actually, a coffee.")
        self.assertIn(turns, last(call.request(1)))

    async def test_a_barge_in_cuts_the_result_after_what_played_and_the_next_request_carries_the_cut_mark(self):
        call = CallHarness(
            decision("delegate", ""),
            decision("speak", "We have Earl Grey. And green tea."),
            decision("speak", "Earl Grey and green tea."),
        )
        audio = itertools.count()
        call.output.playing.clear()  # the result's second sentence waits at the output until the caller cuts it
        call.output.pause_at = lambda frame: isinstance(frame, TTSAudioRawFrame) and next(audio) == 1
        await call.play(
            *caller("What teas do you have?"),
            SleepFrame(0.3),
            Act(lambda: call.port.answer("del_1", "Teas: Earl Grey, green")),
            SleepFrame(0.3),
            *caller("Sorry, which ones?", seconds=0.3),  # the cut reaches the assistant aggregator as the caller speaks
            Act(call.output.playing.set),
            SleepFrame(0.4),
        )
        cut = "We have Earl Grey." + TEXTS.cut_mark
        self.assertEqual(call.said(), [cut, "Earl Grey and green tea."])
        self.assertIn({"role": "assistant", "content": cut}, call.request(2)["messages"])
        self.assertEqual(len(call.chat.requests), 3)  # a result heard in part is not said again

    async def test_a_result_that_arrives_while_the_caller_speaks_is_said_after_their_turn_with_no_words(self):
        # With the resume on, the turn with no words runs the talker; with it off, the caller's idle spell asks again.
        for resume in (True, False):
            with self.subTest(resume_after_noise=resume):
                config = replace(CONFIG, speech=replace(CONFIG.speech, resume_after_noise=resume))
                call = CallHarness(decision("delegate", ""), decision("speak", "Tea is ready."), config=config)
                while_speaking = []
                await call.play(
                    *caller("Tea please"),
                    SleepFrame(0.3),
                    ProposedUserStartedSpeakingFrame(),
                    SleepFrame(0.05),
                    Act(lambda call=call: call.port.answer("del_1", "Tea is ready.")),
                    SleepFrame(0.1),
                    Act(lambda call=call, seen=while_speaking: seen.append((len(call.chat.requests), call.said()))),
                    ProposedUserStoppedSpeakingFrame(),
                    SleepFrame(1.2),
                )
                # Worded while the caller speaks, said only after their turn, with no request of its own.
                self.assertEqual(while_speaking, [(2, [])])
                self.assertFalse(call.request(1)["stream"])
                self.assertEqual(call.said(), ["Tea is ready."])
                self.assertEqual(len(call.chat.requests), 2)
