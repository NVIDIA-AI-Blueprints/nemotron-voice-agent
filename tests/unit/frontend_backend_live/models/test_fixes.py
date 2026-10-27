# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Talker and writer edge cases: fallback wording, the repeat guard, the holding line, JSON replies."""

import unittest
from dataclasses import replace

from examples.frontend_backend_live.cascade.agent import build_agent
from examples.frontend_backend_live.cascade.memory import BACKEND_RESULT_PREFIX
from examples.frontend_backend_live.cascade.ports import ReadyResult, WordingNotes
from examples.frontend_backend_live.cascade.talker.repeat_guard import repeated_speech
from examples.frontend_backend_live.cascade.talker.talker import Talker
from examples.frontend_backend_live.cascade.writer.guards import plain_speech
from examples.frontend_backend_live.cascade.writer.wording import WriterWording
from examples.frontend_backend_live.cascade.writer.writer import CommentaryWriter
from examples.frontend_backend_live.models.chat_completions import ChatCompletionsFrontend
from examples.frontend_backend_live.models.decision import TalkerDecision, decision_from_output_text
from examples.frontend_backend_live.prompts import get_prompt_set
from examples.frontend_backend_live.prompts.catalog import catalog_line
from tests.unit.cascade_helpers import CASCADE_ENDPOINT as ENDPOINT
from tests.unit.cascade_helpers import FAST, ScriptedFrontend, _config, _decision_chunks
from tests.unit.core_helpers import FakeChat

V2, V3 = get_prompt_set("v2"), get_prompt_set("v3")
HOLDING = "One moment, I'm still working on your earlier request."


class FallbackSpeechTests(unittest.IsolatedAsyncioTestCase):
    """When the writer fails, the result is said as written, and its numbers and ids survive."""

    def test_a_leading_number_or_a_minus_sign_is_no_list_marker(self):
        self.assertEqual(plain_speech("12.50 is your total"), "12.50 is your total.")
        self.assertEqual(plain_speech("-5 degrees tonight"), "-5 degrees tonight.")
        self.assertEqual(plain_speech("#5 is ready"), "#5 is ready.")

    def test_underscores_in_ids_stay(self):
        self.assertEqual(plain_speech("Found user mia_li_3668"), "Found user mia_li_3668.")

    def test_real_list_markers_and_emphasis_still_go(self):
        self.assertEqual(plain_speech("1. Latte\n2) Croissant\n- Tea\n* Scone"), "Latte. Croissant. Tea. Scone.")
        self.assertEqual(plain_speech("# Order\n**Total:** $5.00"), "Order. Total: $5.00.")

    async def test_a_failed_wording_says_numbers_and_ids_intact(self):
        frontend = ScriptedFrontend([RuntimeError("hub down")])
        wording = WriterWording(
            CommentaryWriter(frontend, V2), lambda: ("Be Remy.", []), timeout_seconds=1.0, observe=lambda *a, **k: None
        )
        result = ReadyResult("d1", "m1", "12.50 is the total for mia_li_3668")
        line = await wording.words([result], WordingNotes(told={}, newer_requests=()))
        self.assertEqual(line, "12.50 is the total for mia_li_3668.")


class RepeatGuardTests(unittest.TestCase):
    """The repeat guard is off by default, so a recap is answered, not delegated."""

    def test_the_guard_still_finds_a_near_repeat_when_it_is_on(self):
        stale = "I am still unable to locate your account with the information provided."
        history = [{"role": "assistant", "content": stale}]
        self.assertGreaterEqual(repeated_speech(stale, "It is 4472.", history), 0.85)
        self.assertEqual(repeated_speech(stale, "Can you repeat that?", history), 0.0)
        results = [{"role": "user", "content": BACKEND_RESULT_PREFIX + stale}]
        self.assertGreaterEqual(repeated_speech(stale, "Okay.", results), 0.85)

    def test_v3_tells_the_talker_to_repeat_a_still_current_result(self):
        self.assertIn("repeat it when the caller asks for it again", V3.routing)


class HoldingTests(unittest.IsolatedAsyncioTestCase):
    """While work is open, a failed, empty or unusable decision says the holding line, not a new delegation."""

    def talker(self, script, prompts=V2, holding=HOLDING):
        return Talker(ScriptedFrontend(script), prompts, holding=holding)

    async def test_unusable_output_while_work_is_open_holds(self):
        unusable = TalkerDecision("delegate", "", {"recovery": "invalid_json"})
        decision = await self.talker([unusable]).decide("Be Remy.", [], "WORK", work_open=True)
        self.assertEqual((decision.action, decision.speech), ("speak", HOLDING))
        self.assertEqual(
            (decision.diagnostics["recovery"], decision.diagnostics["failed"]), ("holding", "invalid_json")
        )

    async def test_an_empty_decision_while_work_is_open_holds_where_no_wait_is_declared(self):
        decision = await self.talker([TalkerDecision("listen", "")]).decide("x", [], "WORK", work_open=True)
        self.assertEqual((decision.action, decision.diagnostics["failed"]), ("speak", "listen_removed"))
        waits = await self.talker([TalkerDecision("listen", "")], V3).decide("x", [], "WORK", work_open=True)
        self.assertEqual(waits.action, "wait")  # a set that declares wait stays silent instead

    async def test_with_no_work_open_or_no_holding_line_the_turn_is_delegated_as_before(self):
        unusable = TalkerDecision("delegate", "", {"recovery": "empty_output"})
        for talker, work_open in ((self.talker([unusable]), False), (self.talker([unusable], holding=""), True)):
            with self.subTest(work_open=work_open, holding=bool(talker.holding)):
                decision = await talker.decide("x", [], "", work_open=work_open)
                self.assertEqual((decision.action, decision.diagnostics["recovery"]), ("delegate", "empty_output"))

    def test_a_raised_call_gets_the_same_answer_through_after_failure(self):
        talker = self.talker([])
        held = talker.after_failure({"recovery": "talker_failed"}, work_open=True)
        self.assertEqual((held.action, held.speech, held.diagnostics["failed"]), ("speak", HOLDING, "talker_failed"))
        self.assertEqual(talker.after_failure({"recovery": "talker_failed"}, work_open=False).action, "delegate")

    def test_the_agent_gives_the_talker_the_catalogs_holding_line(self):
        config = replace(_config(), provider="chat-completions")
        agent = build_agent(config, ENDPOINT, frontend_client=FakeChat())
        self.assertEqual(agent.talker.holding, catalog_line("holding"))
        self.assertEqual(catalog_line("holding"), HOLDING)


class JsonReplyTests(unittest.IsolatedAsyncioTestCase):
    """The first complete JSON object is the reply, not a greedy match from the first '{' to the last '}'."""

    def test_a_valid_reply_followed_by_more_braces_is_read(self):
        decision = decision_from_output_text('{"action":"speak","speech":"Hi."} and {"note": 1}', {})
        self.assertEqual((decision.action, decision.speech), ("speak", "Hi."))

    def test_prose_around_the_reply_and_braces_inside_the_speech_are_read(self):
        decision = decision_from_output_text('Sure: {"action":"speak","speech":"Use {x} here."} done', {})
        self.assertEqual(decision.speech, "Use {x} here.")

    async def test_the_chat_provider_reads_it_with_no_response_format(self):
        endpoint = replace(ENDPOINT, response_format="none")
        client = FakeChat(_decision_chunks('{"action":"delegate","speech":"One moment."} {"extra": true}'))
        decision = await Talker(ChatCompletionsFrontend(endpoint, FAST, client), V2).decide("Be Remy.", [])
        self.assertEqual((decision.action, decision.speech), ("delegate", "One moment."))
        self.assertNotIn("response_format", client.requests[0])


if __name__ == "__main__":
    unittest.main()
