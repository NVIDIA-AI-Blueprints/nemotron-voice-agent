# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Talker and writer edge cases: results said as written, a recap, the holding line, JSON replies."""

import unittest
from dataclasses import replace

from examples.frontend_backend_live.cascade.agent import build_agent
from examples.frontend_backend_live.cascade.talker.talker import Talker
from examples.frontend_backend_live.cascade.writer.plain_speech import plain_speech
from examples.frontend_backend_live.models.chat_completions import ChatCompletionsFrontend
from examples.frontend_backend_live.models.decision import TalkerDecision, decision_from_output_text
from examples.frontend_backend_live.prompts import get_prompt_set
from examples.frontend_backend_live.prompts.catalog import catalog_line
from tests.unit.cascade_helpers import CASCADE_ENDPOINT as ENDPOINT
from tests.unit.cascade_helpers import FAST, ScriptedFrontend, _config, _decision_chunks
from tests.unit.core_helpers import FakeChat

DEFAULT = get_prompt_set("default")
HOLDING = "One moment, I'm still working on your earlier request."


class FallbackSpeechTests(unittest.TestCase):
    """A result said as written loses its markup, and its numbers and ids survive."""

    def test_a_leading_number_or_a_minus_sign_is_no_list_marker(self):
        self.assertEqual(plain_speech("12.50 is your total"), "12.50 is your total.")
        self.assertEqual(plain_speech("-5 degrees tonight"), "-5 degrees tonight.")
        self.assertEqual(plain_speech("#5 is ready"), "#5 is ready.")

    def test_underscores_in_ids_stay(self):
        self.assertEqual(plain_speech("Found user Q91_T"), "Found user Q91_T.")

    def test_real_list_markers_and_emphasis_still_go(self):
        self.assertEqual(plain_speech("1. Latte\n2) Croissant\n- Tea\n* Scone"), "Latte. Croissant. Tea. Scone.")
        self.assertEqual(plain_speech("# Order\n**Total:** $5.00"), "Order. Total: $5.00.")

    def test_stacked_list_markers_and_headings_go_at_once_so_a_second_cleaning_changes_nothing(self):
        stacked = "## 1. Latte\n- - Tea\n1. - Scone"
        self.assertEqual(plain_speech(stacked), "Latte. Tea. Scone.")
        self.assertEqual(plain_speech(plain_speech(stacked)), plain_speech(stacked))


class RecapTests(unittest.TestCase):
    """Whether to say an earlier answer again is the talker's prompt's to decide."""

    def test_the_prompt_set_tells_the_talker_to_repeat_what_was_said_when_the_caller_asks(self):
        self.assertIn("request to repeat something said in this call", DEFAULT.routing)


class HoldingTests(unittest.IsolatedAsyncioTestCase):
    """While work is open, a failed or unusable decision says the holding line, not a new delegation."""

    def talker(self, script, holding=HOLDING):
        return Talker(ScriptedFrontend(script), DEFAULT, holding=holding)

    async def test_unusable_output_while_work_is_open_holds(self):
        unusable = TalkerDecision("delegate", "", {"recovery": "invalid_json"})
        decision = await self.talker([unusable]).decide("Be Remy.", [], "WORK", work_open=True)
        self.assertEqual((decision.action, decision.speech), ("speak", HOLDING))
        self.assertEqual(
            (decision.diagnostics["recovery"], decision.diagnostics["failed"]), ("holding", "invalid_json")
        )

    async def test_an_empty_reply_while_work_is_open_waits(self):
        decision = await self.talker([TalkerDecision("speak", "")]).decide("x", [], "WORK", work_open=True)
        self.assertEqual((decision.action, decision.diagnostics["recovery"]), ("wait", "empty_speak"))

    async def test_with_no_work_open_or_no_holding_line_an_unusable_turn_is_delegated(self):
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

    def test_prose_around_the_reply_and_braces_inside_the_speech_are_read(self):
        decision = decision_from_output_text('Sure: {"action":"speak","speech":"Use {x} here."} done', {})
        self.assertEqual(decision.speech, "Use {x} here.")

    async def test_the_chat_provider_reads_it_with_no_response_format(self):
        endpoint = replace(ENDPOINT, response_format="none")
        client = FakeChat(_decision_chunks('{"action":"delegate","speech":"One moment."} {"extra": true}'))
        decision = await Talker(ChatCompletionsFrontend(endpoint, FAST, client), DEFAULT).decide("Be Remy.", [])
        self.assertEqual((decision.action, decision.speech), ("delegate", "One moment."))
        self.assertNotIn("response_format", client.requests[0])


if __name__ == "__main__":
    unittest.main()
