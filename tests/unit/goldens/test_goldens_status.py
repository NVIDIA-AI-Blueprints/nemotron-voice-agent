# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Goldens for status while work runs and for the floor: who may speak, and when.

The voice has no canned status lines: the talker answers a status question from its work line (each open
delegation's state and newest progress), or delegates it. A result waits for the floor: the caller
speaking, a turn being decided or a line in flight holds it. Every result ready when the floor frees is
worded together and said in one line.
"""

import asyncio
import json
from unittest.mock import patch

from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame

from examples.frontend_backend_live.cascade.floor import Floor
from examples.frontend_backend_live.cascade.presenter import Presenter
from tests.unit.goldens.harness import (
    WORK_RUNNING,
    GoldenCase,
    HeldRound,
    HoldingChat,
    answer_round,
    decision,
    result_item,
    tool_call_round,
)
from tests.unit.goldens.test_goldens_client import response_create, typed_item

TEAS = "We have Earl Grey and Matcha."
TEA_MENU = '{"teas": ["Earl Grey", "Matcha"]}'
# X9's two backend answers, and the one line the writer words for both.
TEA_ANSWER, PASTRY_ANSWER = "Earl Grey and Matcha.", "We have fresh croissants."
BOTH = "We have Earl Grey and Matcha, and fresh croissants."


class StatusGoldens(GoldenCase):
    """Status while work runs, and the order in which results and replies are spoken."""

    def instrument(self, patches):
        """The harness's counters, plus the floor's: whether the caller speaks, and the lines held for the caller.

        ``Floor.set_caller_speaking`` records whether the cascade knows the caller is speaking, and a worded line's
        wait for the floor (``Presenter._held``) that begins while the caller speaks counts as a line held for the
        caller. Each wrapper calls the original unchanged; they reach past the wire like the harness's own
        ``instrument``: re-point them when a change moves those reads.
        """
        super().instrument(patches)
        case = self
        set_caller_speaking, held = Floor.set_caller_speaking, Presenter._held
        self.caller_speaking = False
        self.holds = self.holds_open = 0

        def set_caller_speaking_seen(floor, speaking):
            set_caller_speaking(floor, speaking)
            case.caller_speaking = speaking
            case._changed()

        async def held_counted(presenter, batch, stamp, until=None):
            hold = until is None and presenter.floor.caller_speaking  # a wait for idle, not for a wording
            if hold:
                case.holds += 1
                case.holds_open += 1
                case._changed()
            try:
                return await held(presenter, batch, stamp, until)
            finally:
                if hold:
                    case.holds_open -= 1
                    case._changed()

        patches.enter_context(patch.object(Floor, "set_caller_speaking", set_caller_speaking_seen))
        patches.enter_context(patch.object(Presenter, "_held", held_counted))

    def hold_reply(self, client, number):
        """Hold the scripted reply to request ``number`` of ``client`` until the returned event is set.

        The request is recorded and its step is taken when it is made, as usual; only the reply waits, as a slow
        model's would. Release it well within the role's timeouts (a talker or writer attempt has 10 s).
        """
        release = asyncio.Event()
        create = client.create

        async def create_held(**kwargs):
            stream = await create(**kwargs)
            if len(client.requests) == number:
                await release.wait()
            return stream

        client.create = create_held  # a provider calls client.chat.completions.create, which is client.create
        return release

    def in_context(self, text):
        """Whether ``text`` was spoken; a line enters the talker's context when it is handed to speech."""
        return text in self.tts.utterances

    async def caller_starts_speaking(self):
        """The caller starts a sentence and keeps talking: the floor holds results until they stop."""
        await self.stt.push_frame(UserStartedSpeakingFrame())
        await self.wait_until(lambda: self.caller_speaking, "the cascade to see the caller speaking")

    async def caller_stops_speaking(self):
        """The caller stops without a word the speech-to-text kept: no turn follows, and held results go out."""
        await self.stt.push_frame(UserStoppedSpeakingFrame())
        await self.wait_until(lambda: not self.caller_speaking, "the cascade to see the caller stop")

    # ------------------------------------------------------------------------------------- the goldens
    async def test_g8_a_status_question_while_work_runs(self):
        """G8: a status question while delegated work runs, a request that joins it, then a later request.

        The talker delegates the tea question with a lead-in; the backend calls the client's get_menu tool, and the
        work waits for the client. "Any update?" goes to the talker, whose prompt carries the work line, and the
        talker answers it itself. "Can I also get a muffin?" is delegated and joins the waiting Work (the steer
        conductor). The client returns the menu: the Work's next round answers both, in the newer request's response,
        and the writer speaks it, told of both lead-ins. A later request is a Work of its own, delegated with a
        lead-in; its backend input carries the conversation since the joined request. Its backend reply is held until
        the lead-in is in the talker's context, so the writer's request is the same on every run.
        """
        await self.serve(
            frontend=[
                decision("delegate", "Let me check."),
                decision("speak", "I'm still checking the teas for you."),
                decision("delegate", "I'll check the muffin too."),
                decision("speak", "We have Earl Grey and Matcha, and a blueberry muffin."),
                decision("delegate", "Let me check the coffee."),
                decision("speak", "We have espresso and filter coffee."),
            ],
            backend=[
                tool_call_round(),
                answer_round("Earl Grey and Matcha; a blueberry muffin is available."),
                answer_round("We have espresso and filter coffee."),
            ],
        )
        await self.say("What teas do you have?")
        call = await self.wait_for_call("get_menu")
        # The work waits for the client's tool output: it stays open while the caller asks.
        await self.wait_quiet()
        await self.say("Any update?")
        await self.wait_quiet()
        await self.say("Can I also get a muffin?")
        await self.wait_quiet()
        self.send(result_item(call["call_id"], TEA_MENU))
        self.send({"type": "response.create", "event_id": "continue"})
        await self.wait_quiet()
        release = self.hold_reply(self.backend, 3)
        await self.say("And do you have coffee?")
        await self.wait_until(
            lambda: len(self.backend.requests) == 3 and self.in_context("Let me check the coffee."),
            "the coffee request at the backend and its lead-in in the talker's context",
        )
        release.set()
        await self.wait_quiet()
        await self.close()
        self.assert_golden("G8")

    async def test_x9_two_results_ready_together_are_said_in_one_line(self):
        """X9: two results ready together are worded together and said in one line (the presenter's inbox).

        The talker delegates the tea question with a lead-in, and the backend round is held. The caller starts to
        speak and keeps talking, so the floor is held, and the round answers: the tea answer waits in the inbox. The
        client types a question about pastries and runs it; no Work is open, so it starts one, and its answer waits
        beside the first. The writer words the tea answer at once, while the caller speaks, and the pastry answer that
        comes next makes the presenter take both at once: the writer words them in one request, and the line waits
        again. When the caller stops (with no words the speech-to-text kept), one line says both, and each answer
        enters the talker's memory with it. The caller then thanks the talker, whose request shows both answers and the
        one line.
        """
        held = HeldRound(answer_round(TEA_ANSWER))
        with patch("tests.unit.goldens.harness.ScriptedChat", HoldingChat):
            await self.serve(
                frontend=[
                    decision("delegate", "Let me check."),  # the talker, on the tea question
                    decision("speak", TEAS),  # the writer, at once: the tea result alone
                    decision("speak", BOTH),  # the writer: both results in one request
                    decision("speak", "You're welcome."),  # the talker, on the thanks
                ],
                backend=[held, answer_round(PASTRY_ANSWER)],
            )
        await self.say("What teas do you have?")
        await self.wait_until(
            lambda: held.waiting.is_set() and self.pending() == [WORK_RUNNING],
            "the lead-in played while the tea round is held",
        )
        await self.caller_starts_speaking()
        held.release()
        await self.wait_until(lambda: self.holds_open == 1, "the tea answer held for the caller")
        self.send(typed_item("And do you have pastries?", "typed_question"))
        self.send(response_create("typed_run"))
        await self.wait_until(lambda: self.completed_responses() == 2, "the typed run's answer beside the tea answer")
        await self.wait_until(
            lambda: (self.holds_open, len(self.frontend.requests)) == (1, 3), "both answers worded together, held"
        )
        await self.caller_stops_speaking()
        await self.wait_quiet()
        await self.say("Thanks.")
        await self.wait_quiet()
        await self.close()
        self.assertEqual(self.holds, 2)  # the tea answer, then both answers worded again
        self.assertEqual(self.tts.utterances, ["Let me check.", BOTH, "You're welcome."])
        writer = json.dumps(self.frontend.requests[2])
        self.assertTrue(TEA_ANSWER in writer and PASTRY_ANSWER in writer, "one writer request words both results")
        talker = json.dumps(self.frontend.requests[3])
        self.assertTrue(all(text in talker for text in (TEA_ANSWER, PASTRY_ANSWER, BOTH)), "both enter the memory")
        self.assert_golden("X9")

    def completed_responses(self):
        """How many delegations' responses completed, as relayed to the client."""
        return sum(
            event["type"] == "response.event" and event["event"]["type"] == "response.completed"
            for event in self.ws.from_server
        )

    async def test_x4_a_reply_spoken_while_a_result_is_held(self):
        """X4: a reply decided while the caller speaks again waits for their silence; the result is held meanwhile.

        The talker delegates the tea question with a lead-in; the backend calls the client's get_menu tool, and the
        work waits for the client. The caller asks about oat milk, and while the talker decides, starts to speak
        again. The client returns the menu: the writer words the tea answer at once, and it waits for the floor,
        because the caller is speaking. The talker's reply about oat milk comes while the caller speaks, so it is
        dropped and the oat-milk turn waits. The caller stops with no words the speech-to-text kept: the talker decides
        the waiting turn again and replies, and the reply moves the held line's stamp, so the writer words the answer
        again; then it is spoken.
        """
        await self.serve(
            frontend=[
                decision("delegate", "Let me check."),
                decision("speak", "Yes, we have oat milk."),  # the talker: completes while the caller speaks
                decision("speak", TEAS),  # the writer, at once, while the caller speaks
                decision("speak", "Yes, we have oat milk."),  # the talker again, once the caller is silent
                decision("speak", TEAS),  # the writer again: the reply moved the line's stamp
            ],
            backend=[tool_call_round(), answer_round(TEAS)],
        )
        await self.say("What teas do you have?")
        call = await self.wait_for_call("get_menu")
        await self.wait_quiet()
        # The talker's reply to the oat-milk turn is held until the tea answer waits for the caller.
        release = self.hold_reply(self.frontend, 2)
        await self.say("Do you have oat milk?")
        await self.wait_until(lambda: len(self.frontend.requests) == 2, "the talker's request for the second turn")
        await self.caller_starts_speaking()
        self.send(result_item(call["call_id"], TEA_MENU))
        self.send({"type": "response.create", "event_id": "continue"})
        await self.wait_until(lambda: self.holds_open == 1, "the tea answer held for the caller")
        release.set()
        turns = self.engine.cascade.turns
        await self.wait_until(lambda: not turns._deciding and turns.waiting, "the reply dropped, its turn waiting")
        self.assertNotIn("Yes, we have oat milk.", self.tts.utterances)
        await self.caller_stops_speaking()
        await self.wait_quiet()
        self.assertEqual(self.tts.utterances, ["Let me check.", "Yes, we have oat milk.", TEAS])
        self.assertEqual(len(self.frontend.requests), 5)
        await self.close()
        self.assert_golden("X4")
