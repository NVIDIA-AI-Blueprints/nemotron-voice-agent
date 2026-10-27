# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The contracts check themselves: the event checker, the endings fixture, and two contracts on their own doubles."""

import unittest

from delegation_core.backend.port import ConductorInput, Place, Start
from tests.unit.contracts.backend import (
    ConductorContract,
    a_round,
    call_events,
    check_events,
    message_events,
    text_round,
)
from tests.unit.contracts.delegation_port import DelegationPortContract, ScriptedDelegationPort
from tests.unit.contracts.endings import load, normalized, official_events, timeline_problems

# Every ending the fixture must hold.
REQUIRED_ENDINGS = {
    "answered",
    "answered_with_progress_and_tool_items",
    "waits_for_the_app_then_answers",
    "queued_then_running",
    "handed_over_with_the_taker_named",
    "a_chain_of_handovers",
    "answered_by_the_conductor",
    "cancelled_for_a_newer_request_stopped",
    "cancelled_for_a_newer_request_unconfirmed",
    "failed_with_an_error",
    "late_continuation",
    "deadline_while_waiting",
    "cancelled_at_close",
    "a_typed_run",
    "stateless",
    "show_then_the_answer",
    "two_works_answering_at_once",
    "one_work_two_tasks",
    "a_task_waits_while_another_runs",
    "cut_by_max_output_tokens",
    "two_works_waiting_for_the_app",
}


class EventCheckerTests(unittest.TestCase):
    def test_a_round_that_keeps_the_rules_has_no_violation(self):
        events = a_round(message_events("Done.", phase="final_answer"))
        self.assertEqual(check_events(events), [])

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
    def test_the_fixture_holds_every_ending_the_plan_names(self):
        self.assertEqual(set(load()), REQUIRED_ENDINGS)

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


class StartEachArrival:
    """The smallest conductor: a new Work for every arrival, as the queue profile's rule does."""

    async def decide(self, given: ConductorInput, place: Place) -> None:
        for arrival in given.arrivals:
            await place(Start((arrival.delegation_id,)))


class ConductorContractSelfTests(ConductorContract, unittest.IsolatedAsyncioTestCase):
    def make_conductor(self):
        return StartEachArrival()


class ExpectedFailureTests(unittest.TestCase):
    def test_a_known_failure_marks_only_its_own_binding(self):
        from tests.unit.contracts.base import ContractCase

        class Contract(ContractCase):
            def test_it_works(self):
                pass

        class Failing(Contract):
            EXPECTED_FAILURES = {"test_it_works": "a later change"}

        class Passing(Contract):
            pass

        self.assertTrue(getattr(Failing.test_it_works, "__unittest_expecting_failure__", False))
        self.assertFalse(getattr(Passing.test_it_works, "__unittest_expecting_failure__", False))
        self.assertFalse(getattr(Contract.test_it_works, "__unittest_expecting_failure__", False))

    def test_a_known_failure_must_name_a_test_of_the_contract(self):
        from tests.unit.contracts.base import ContractCase

        with self.assertRaisesRegex(TypeError, "no test of the contract"):

            class Wrong(ContractCase):
                EXPECTED_FAILURES = {"test_missing": "a later change"}
