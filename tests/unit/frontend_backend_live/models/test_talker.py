# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The talker and the writer as the prompt set shapes them, and the deterministic guards."""

import asyncio
import unittest

from examples.frontend_backend_live.cascade.memory import BACKEND_RESULT_PREFIX, CLIENT_CONTEXT_PREFIX
from examples.frontend_backend_live.cascade.talker.router import TurnRouter
from examples.frontend_backend_live.cascade.talker.talker import Talker
from examples.frontend_backend_live.cascade.writer.guards import (
    SAFE_LEAD_IN,
    asserts_completion,
    plain_speech,
    safe_lead_in,
)
from examples.frontend_backend_live.cascade.writer.writer import CommentaryWriter
from examples.frontend_backend_live.models.decision import TalkerDecision, decision_schema
from examples.frontend_backend_live.prompts import get_prompt_set, v3
from tests.unit.cascade_helpers import ScriptedFrontend

V1, V2, V3 = (get_prompt_set(name) for name in ("v1", "v2", "v3"))


def talker(prompts, script=()):
    return Talker(ScriptedFrontend(list(script)), prompts)


class FloorTests(unittest.TestCase):
    def test_a_reply_and_a_delegation_always_stand(self):
        for prompts in (V1, V2, V3):
            for decision in (TalkerDecision("speak", "Hello."), TalkerDecision("delegate", "")):
                with self.subTest(prompt_set=prompts.name, action=decision.action):
                    self.assertIs(talker(prompts).floored(decision), decision)

    def test_without_a_declared_wait_nothing_to_say_becomes_a_delegation(self):
        for prompts in (V1, V2):
            for decision, reason in (
                (TalkerDecision("listen", ""), "listen_removed"),
                (TalkerDecision("speak", "  "), "empty_speak"),
                (TalkerDecision("wait", "Okay."), "wait_removed"),
            ):
                with self.subTest(prompt_set=prompts.name, action=decision.action):
                    floored = talker(prompts).floored(decision)
                    self.assertEqual(
                        (floored.action, floored.speech, floored.diagnostics["recovery"]), ("delegate", "", reason)
                    )

    def test_a_set_that_declares_wait_keeps_it_and_waits_when_there_is_nothing_to_say(self):
        kept = talker(V3).floored(TalkerDecision("wait", "Hm.", {"response_id": "r1"}))
        self.assertEqual((kept.action, kept.speech, kept.diagnostics), ("wait", "", {"response_id": "r1"}))
        for decision, reason in (
            (TalkerDecision("listen", ""), "listen_removed"),
            (TalkerDecision("speak"), "empty_speak"),
        ):
            with self.subTest(action=decision.action):
                floored = talker(V3).floored(decision)
                self.assertEqual((floored.action, floored.diagnostics["recovery"]), ("wait", reason))

    def test_an_unusable_decision_still_delegates_the_turn(self):
        unusable = TalkerDecision("delegate", "", {"recovery": "invalid_json"})
        for prompts in (V1, V2, V3):
            with self.subTest(prompt_set=prompts.name):
                self.assertEqual(talker(prompts).floored(unusable).action, "delegate")

    def test_decide_returns_the_floored_decision(self):
        decision = asyncio.run(talker(V2, [TalkerDecision("listen", "")]).decide("Be Remy.", []))
        self.assertEqual((decision.action, decision.diagnostics["recovery"]), ("delegate", "listen_removed"))


class SetShapeTests(unittest.TestCase):
    def test_v3_declares_wait_in_its_actions_its_schema_and_its_hint(self):
        self.assertEqual(V3.actions, ("speak", "delegate", "wait"))
        schema = decision_schema(talker(V3).spec)
        self.assertEqual(schema["properties"]["action"]["enum"], ["speak", "delegate", "wait"])
        self.assertIn("For wait: empty.", schema["properties"]["speech"]["description"])
        self.assertIn('"wait"', V3.routing)

    def test_v3_reads_spawnthinking_as_a_delegation_and_never_names_the_backend_to_the_caller(self):
        self.assertIn("<|SpawnThinking|>", V3.frame_after)
        self.assertIn("never mention a backend", V3.frame_before)
        self.assertIn("never mention a backend", V3.commentary)

    def test_v3s_rule_on_details_no_longer_delegates_a_bare_repeat(self):
        self.assertIn("corrects, adds to, or repeats details given earlier", V2.routing)
        self.assertIn("corrects or adds to details given earlier", V3.routing)

    def test_v3_is_listed_but_the_default_stays_v2(self):
        from examples.frontend_backend_live.prompts import DEFAULT_PROMPT_SET, PROMPT_SETS

        self.assertIs(PROMPT_SETS["v3"], v3.PROMPT_SET)
        self.assertEqual(DEFAULT_PROMPT_SET, "v2")


class CallerLabelTests(unittest.TestCase):
    HISTORY = [
        {"role": "user", "content": "Book the morning slot."},
        {"role": "assistant", "content": "One moment."},
        {"role": "user", "content": BACKEND_RESULT_PREFIX + "Morning slot is free. Shall I book it?"},
        {"role": "user", "content": "Yes."},
        {"role": "user", "content": "At nine."},
        {"role": "user", "content": CLIENT_CONTEXT_PREFIX + "Loyalty member."},
        {"role": "user", "content": "Great."},
    ]

    def test_v3_labels_each_caller_turn_right_after_reference_data(self):
        frontend = ScriptedFrontend([TalkerDecision("delegate", "")])
        asyncio.run(Talker(frontend, V3).decide("Be Sam.", self.HISTORY))
        sent = [item["content"] for item in frontend.calls[0]["history"]]
        self.assertEqual(sent[3], "Caller: Yes.")
        self.assertEqual(sent[4], "At nine.")  # follows the caller's own turn, not reference data
        self.assertEqual(sent[6], "Caller: Great.")
        self.assertEqual(sent[:3] + [sent[5]], [item["content"] for item in self.HISTORY[:3] + [self.HISTORY[5]]])

    def test_v1_and_v2_send_the_history_as_it_is(self):
        for prompts in (V1, V2):
            with self.subTest(prompt_set=prompts.name):
                frontend = ScriptedFrontend([TalkerDecision("delegate", "")])
                asyncio.run(Talker(frontend, prompts).decide("Be Remy.", self.HISTORY))
                self.assertEqual(frontend.calls[0]["history"], self.HISTORY)


class WriterFrameTests(unittest.TestCase):
    def render(self, prompts, answer):
        frontend = ScriptedFrontend([TalkerDecision("speak", "Said.")])
        asyncio.run(CommentaryWriter(frontend, prompts).render("Be Sam.", [], "Latte added.", answer=answer))
        return frontend.calls[0]

    def test_v1_and_v2_send_the_writer_todays_request(self):
        for prompts in (V1, V2):
            for answer in (True, False):
                with self.subTest(prompt_set=prompts.name, answer=answer):
                    call = self.render(prompts, answer)
                    expected = "Be Sam.\n" + prompts.commentary + (prompts.answer if answer else "")
                    self.assertEqual(call["instructions"], expected)
                    update = call["history"][-1]["content"]
                    self.assertEqual(update, "Commentary update (reference data):\nLatte added.")

    def test_v3_frames_the_writers_instructions_and_labels_each_source(self):
        result, commentary = self.render(V3, True), self.render(V3, False)
        self.assertEqual(result["instructions"], V3.writer_framed("Be Sam.") + "\n" + V3.commentary + V3.answer)
        self.assertIn("<application_instructions>\nBe Sam.\n</application_instructions>", result["instructions"])
        self.assertTrue(result["history"][-1]["content"].startswith("Backend result for the caller (reference data):"))
        self.assertTrue(commentary["history"][-1]["content"].startswith("The application's message for the caller:"))


class GuardTests(unittest.TestCase):
    def test_lead_in_never_claims_an_outcome_or_asks_a_question(self):
        self.assertEqual(safe_lead_in("Your order is confirmed."), SAFE_LEAD_IN)
        self.assertTrue(asserts_completion("It is all set"))
        self.assertEqual(safe_lead_in("Let me check. Which size?"), "Let me check.")
        self.assertEqual(safe_lead_in("Which size?"), SAFE_LEAD_IN)
        self.assertEqual(safe_lead_in("Let me look that up."), "Let me look that up.")
        self.assertEqual(plain_speech("- one\n- two"), "one. two.")

    def test_router_drops_backchannels_but_keeps_answers(self):
        router = TurnRouter()
        self.assertFalse(router.verdict("Okay.", "Your latte is ready.").keep)
        self.assertTrue(router.verdict("Yes.", "Please confirm the order.").keep)
        self.assertTrue(router.verdict("Okay.", None).keep)
        self.assertTrue(router.verdict("Two lattes", "Anything else?").keep)


if __name__ == "__main__":
    unittest.main()
