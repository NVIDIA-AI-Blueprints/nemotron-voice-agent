# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The contracts check themselves: the event checker, the endings fixture, and the port contract on its own double."""

import unittest

from tests.unit.contracts.backend import a_round, call_events, check_events, message_events, text_round
from tests.unit.contracts.delegation_port import DelegationPortContract, ScriptedDelegationPort
from tests.unit.contracts.endings import load, normalized, official_events, timeline_problems


class EventCheckerTests(unittest.TestCase):
    def test_each_rule_finds_its_violation(self):
        no_phase = text_round("Done.")
        self.assertIn("E3", " ".join(check_events(no_phase)))
        self.assertEqual(check_events(no_phase, phases=False), [])
        late_message = a_round(call_events("get_menu", {}), message_events("Done.", index=1))
        self.assertIn("E4: a message follows a call in its round", check_events(late_message, phases=False))
        unannounced = [e for e in text_round("Done.") if e["type"] != "response.output_item.added"]
        self.assertIn("E2", " ".join(check_events(unannounced, phases=False)))
        after_terminal = [*text_round("Done."), {"type": "response.output_text.delta", "delta": "x"}]
        self.assertIn("E1", " ".join(check_events(after_terminal, phases=False)))
        cut = text_round("Done.")[:-1]
        self.assertIn("E5: the iteration ended inside an attempt", check_events(cut, phases=False))

    def test_a_round_with_calls_may_say_only_commentary(self):
        events = a_round(message_events("Let me look.", phase="commentary"), call_events("get_menu", {}, index=1))
        self.assertEqual(check_events(events), [])
        events = a_round(message_events("Done.", phase="final_answer"), call_events("get_menu", {}, index=1))
        self.assertIn("E4: a round with calls carries a final_answer", check_events(events))


class EndingsFixtureTests(unittest.TestCase):
    def test_every_scenario_keeps_the_wire_rules(self):
        for name, scenario in load().items():
            with self.subTest(name):
                self.assertEqual(timeline_problems(scenario), [])

    def test_the_official_events_read_back_as_the_timeline(self):
        texts = {"handover_line": "Handled together.", "cancelled_line": "Stopped.", "unconfirmed_line": "Not sure."}
        for name, scenario in load().items():
            with self.subTest(name):
                self.assertEqual(normalized(official_events(scenario, texts), texts), scenario["timeline"])

    def test_the_rules_catch_a_taker_named_after_the_older_response_ended(self):
        scenario = load()["handed_over_with_the_taker_named"]
        timeline = [step for step in scenario["timeline"] if step != ["R2", "in_progress previous=R1"]]
        late = [*timeline[:-2], ["R2", "in_progress previous=R1"], *timeline[-2:]]
        problems = timeline_problems({**scenario, "timeline": late})
        self.assertIn("R2: in_progress previous=R1: the taker must be named before the older response ends", problems)


class ScriptedPortContractTests(DelegationPortContract, unittest.IsolatedAsyncioTestCase):
    async def make_port(self):
        created = {"type": "response.created", "response": {"id": "resp_1", "status": "queued"}}
        completed = {"type": "response.completed", "response": {"id": "resp_1", "status": "completed", "output": []}}
        return ScriptedDelegationPort(answer_with=[created, completed])
