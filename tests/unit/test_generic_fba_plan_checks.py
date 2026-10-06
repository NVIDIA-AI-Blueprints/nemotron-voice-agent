# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D101, D102

"""Generic Frontend/Backend plan checks, the consent gate, re-spell questions and the detached-run cap.

Every schema and value here is invented; none comes from a benchmark.
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from examples.frontend_backend_agent.generic import plan_checks
from examples.frontend_backend_agent.generic.backend import GenericThinkerBackend
from examples.frontend_backend_agent.generic.client_tools import build_client_tool_specs
from examples.frontend_backend_agent.generic.consent import ConsentBook, is_affirmative
from examples.frontend_backend_agent.src.history import ConversationTranscript, DelegationLedger
from realtime.conversation import ConversationJournal

LOOKUP = {
    "type": "function",
    "name": "lookup_member",
    "description": "Look up a member by member id.",
    "parameters": {
        "type": "object",
        "properties": {
            "member_id": {"type": "string", "description": "The member id, such as 'jordan_lee_82'."},
        },
        "required": ["member_id"],
    },
}
RENEW = {
    "type": "function",
    "name": "renew_membership",
    "description": "Renew a membership for one year.",
    "parameters": {
        "type": "object",
        "properties": {"member_id": {"type": "string", "title": "Member Id"}},
        "required": ["member_id"],
        "additionalProperties": False,
    },
}
HANDOFF = {
    "type": "function",
    "name": "transfer_to_staff",
    "description": "Transfer the caller to a human staff member.",
    "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
}
INSTRUCTIONS = "Membership policy. Lapsed memberships older than two years cannot be renewed by phone."


class _Executor:
    def __init__(self, outputs: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.outputs = outputs or {}

    async def __call__(self, calls, _timeout):
        self.calls.extend(calls)
        return [self.outputs.get(name, {"ok": True, "member_id": "jordan_lee_82"}) for name, _arguments in calls]


class _Context:
    def get_messages(self) -> list[dict[str, Any]]:
        return [{"role": "system", "content": INSTRUCTIONS}]


class _Planner:
    """Return scripted plans in order; record every state it received."""

    session_instruction_context = _Context()

    def __init__(self, plans: list[dict[str, Any]] | None = None) -> None:
        self.plans = list(plans or [])
        self.states: list[dict[str, Any]] = []

    async def plan(self, *, query: str, state: dict, **_kwargs: Any) -> dict:
        self.states.append(dict(state))
        return self.plans.pop(0) if self.plans else {"complete": True}


def _session(
    plans: list[dict[str, Any]], *, tools: tuple[dict, ...] = (LOOKUP, RENEW), outputs: dict | None = None
) -> tuple[GenericThinkerBackend, _Planner, _Executor, ConversationTranscript, ConversationJournal]:
    journal = ConversationJournal()
    transcript = ConversationTranscript(lambda: journal)
    planner = _Planner(plans)
    executor = _Executor(outputs)
    specs = build_client_tool_specs(tools)
    backend = GenericThinkerBackend(
        planner=planner,
        tools={},
        enabled_tools=tuple(specs),
        client_tools=specs,
        client_tool_executor=executor,
        overall_timeout_seconds=5,
        planner_timeout_seconds=5,
        max_planning_rounds=3,
        conversation_ledger=DelegationLedger(transcript),
    )
    return backend, planner, executor, transcript, journal


def _lookup(member_id: str = "jordan_lee_82") -> dict:
    return {"tool": "lookup_member", "params": {"member_id": member_id}, "continue_after_results": True}


def _refusal(text: str = "That cannot be done.") -> dict:
    return {
        "tool": "response_hint",
        "reason": "unsupported_request",
        "action": "answer_directly",
        "context": "general",
        "response_text": text,
    }


def _assistant(journal: ConversationJournal, text: str, item_id: str, status: str = "completed") -> None:
    journal.add_item(
        {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "status": status,
            "content": [{"type": "output_audio", "transcript": text}],
        }
    )


class HelperTests(unittest.TestCase):
    def test_change_verbs_are_stemmed(self) -> None:
        self.assertTrue(plan_checks.is_change_request("Please process the cancellation of my hold."))
        self.assertTrue(plan_checks.is_change_request("Renew and update my membership"))
        self.assertFalse(plan_checks.is_change_request("What is the status of my membership?"))
        self.assertFalse(plan_checks.is_change_request("Check in on the address"))

    def test_candidates_match_name_tokens_or_description_opening(self) -> None:
        tools = [
            plan_checks.ToolInfo("renew_membership", "Renew a membership for one year.", "write"),
            plan_checks.ToolInfo("lookup_member", "Look up a member by member id.", "read"),
        ]
        self.assertEqual(
            plan_checks.candidate_tools("Renew membership for jordan", tools, writes_only=True), ["renew_membership"]
        )
        self.assertEqual(plan_checks.candidate_tools("Delete my photos", tools, writes_only=True), [])

    def test_a_handoff_tool_needs_its_name_and_a_person_in_its_description(self) -> None:
        self.assertTrue(plan_checks.is_handoff_tool(plan_checks.ToolInfo(HANDOFF["name"], HANDOFF["description"], "w")))
        self.assertFalse(
            plan_checks.is_handoff_tool(plan_checks.ToolInfo("transfer_funds", "Transfer money between funds.", "w"))
        )
        self.assertFalse(
            plan_checks.is_handoff_tool(plan_checks.ToolInfo("get_agent_details", "Get a human agent.", "read"))
        )

    def test_explicit_person_requests_ignore_negations(self) -> None:
        self.assertTrue(plan_checks.asks_for_person("Can I speak to someone, a real person?"))
        self.assertTrue(plan_checks.asks_for_person("Transfer me please"))
        self.assertFalse(plan_checks.asks_for_person("Please don't transfer me"))
        self.assertFalse(plan_checks.asks_for_person("I do not want a human"))

    def test_policy_quotes_are_verbatim_and_a_malformed_basis_is_rejected(self) -> None:
        quote = plan_checks.policy_quote({"basis": {"policy_quote": "cannot be renewed by phone"}})
        self.assertTrue(plan_checks.quote_in_instructions(quote, INSTRUCTIONS))
        self.assertFalse(plan_checks.quote_in_instructions("can be renewed by phone", INSTRUCTIONS))
        self.assertFalse(plan_checks.quote_in_instructions("by phone", INSTRUCTIONS))
        with self.assertRaises(plan_checks.MalformedBasisError):
            plan_checks.policy_quote({"basis": "policy"})


class ConsentBookTests(unittest.TestCase):
    def test_affirmatives(self) -> None:
        for text in ("Yes.", "Yes, go ahead and renew it.", "Sure", "That's right", "Please proceed."):
            with self.subTest(text=text):
                self.assertTrue(is_affirmative(text))
        for text in ("Yes, but change the date.", "No.", "Wait.", "okay", "yes, if it is free", ""):
            with self.subTest(text=text):
                self.assertFalse(is_affirmative(text))

    def test_the_next_turn_grants_once_and_later_turns_expire(self) -> None:
        book = ConsentBook()
        book.record(tool="t", fingerprint="f", run_id="r", generation=0, turn=3)
        book.observe_turn(turn=4, utterance="Yes please", reply_completed=True, generation=0)
        approval = book.take("f", 0)
        self.assertIsNotNone(approval)
        self.assertIsNone(book.take("other", 0))
        book.consume(approval)
        self.assertIsNone(book.take("f", 0))

    def test_a_cut_question_or_another_answer_approves_nothing(self) -> None:
        for completed, utterance in ((False, "Yes"), (True, "Hmm, what?")):
            with self.subTest(completed=completed, utterance=utterance):
                book = ConsentBook()
                book.record(tool="t", fingerprint="f", run_id="r", generation=0, turn=1)
                book.observe_turn(turn=2, utterance=utterance, reply_completed=completed, generation=0)
                self.assertIsNone(book.take("f", 0))

    def test_a_changed_request_voids_the_question_unless_it_answers_it(self) -> None:
        book = ConsentBook()
        book.record(tool="t", fingerprint="f", run_id="r", generation=0, turn=1)
        book.expire_for_new_request("yes go ahead")
        book.observe_turn(turn=2, utterance="yes go ahead", reply_completed=True, generation=0)
        self.assertIsNotNone(book.take("f", 0))
        book.expire_for_new_request("Actually renew a different one")
        self.assertIsNone(book.take("f", 0))


class ConsentGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_an_unapproved_write_becomes_a_consent_question(self) -> None:
        backend, _planner, executor, transcript, _journal = _session(
            [{"tool": "renew_membership", "params": {"member_id": "jordan_lee_82"}}]
        )
        transcript.record_user("Renew jordan_lee_82.")

        payload = await backend.call("Renew membership jordan_lee_82.")

        self.assertEqual(executor.calls, [])
        self.assertEqual(payload["reason"], "confirmation_needed")
        self.assertEqual(payload["params_resolved"], {"member_id": "jordan_lee_82"})

    async def test_an_affirmative_answer_lets_the_planned_write_through_once(self) -> None:
        write = {"tool": "renew_membership", "params": {"member_id": "jordan_lee_82"}}
        backend, planner, executor, transcript, journal = _session([write])
        transcript.record_user("Renew jordan_lee_82.")
        question = await backend.call("Renew membership jordan_lee_82.")
        _assistant(journal, question["response_text"], "a1")
        transcript.record_user("Yes, go ahead and renew it.")
        planner.plans = [write, {"complete": True}]

        await backend.call("The caller confirmed: renew membership jordan_lee_82.")

        self.assertEqual(executor.calls, [("renew_membership", {"member_id": "jordan_lee_82"})])

    async def test_a_different_argument_or_a_refusal_is_not_approved(self) -> None:
        for answer, write in (
            ("Yes please go ahead.", {"tool": "renew_membership", "params": {"member_id": "sam_rivera_44"}}),
            ("No, wait.", {"tool": "renew_membership", "params": {"member_id": "jordan_lee_82"}}),
        ):
            with self.subTest(answer=answer):
                backend, planner, executor, transcript, journal = _session(
                    [{"tool": "renew_membership", "params": {"member_id": "jordan_lee_82"}}]
                )
                transcript.record_user("Renew jordan_lee_82.")
                question = await backend.call("Renew membership jordan_lee_82.")
                _assistant(journal, question["response_text"], "a1")
                transcript.record_user(answer)
                planner.plans = [write]

                payload = await backend.call("Renew the membership.")

                self.assertEqual(executor.calls, [])
                self.assertEqual(payload["reason"], "confirmation_needed")


class AnswerCheckTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_refusal_labelled_answer_is_replanned_and_delivered_as_a_draft(self) -> None:
        backend, planner, _executor, transcript, _journal = _session(
            [
                _lookup(),
                _refusal("Your membership runs until May."),
                {"complete": True, "response_text": "Your membership runs until May."},
            ]
        )
        transcript.record_user("When does my membership end?")

        payload = await backend.call("When does membership jordan_lee_82 end?")

        self.assertIn("unsupported_request means you are declining", planner.states[2]["validation_error"])
        self.assertEqual(payload["draft_answer"], "Your membership runs until May.")
        self.assertEqual(payload["type"], "tool_result")

    async def test_a_refusal_repeated_after_the_check_keeps_refusal_semantics_with_its_reason(self) -> None:
        backend, planner, _executor, transcript, _journal = _session(
            [_lookup(), _refusal(), _refusal("Lapsed memberships cannot be renewed by phone.")]
        )
        transcript.record_user("Update my lapsed membership.")

        payload = await backend.call("Update lapsed membership jordan_lee_82.")

        reason = planner.states[2]["validation_error"]
        self.assertIn("Before refusing, consider: renew_membership", reason)
        self.assertIn("unsupported_request means", reason)
        self.assertEqual(payload["refusal_reason"], "Lapsed memberships cannot be renewed by phone.")
        self.assertNotIn("draft_answer", payload)
        self.assertEqual(len(planner.states), 3)

    async def test_a_refusal_without_results_or_with_a_policy_quote_is_accepted_at_once(self) -> None:
        quoted = {**_refusal(), "basis": {"policy_quote": "cannot be renewed by phone"}}
        for query, plan in (("What is the weather on the moon?", _refusal()), ("Renew membership now.", quoted)):
            with self.subTest(query=query):
                backend, planner, _executor, transcript, _journal = _session([plan])
                transcript.record_user(query)

                payload = await backend.call(query)

                self.assertEqual(payload["reason"], "unsupported_request")
                self.assertEqual(len(planner.states), 1)


class HandoffCheckTests(unittest.IsolatedAsyncioTestCase):
    def _handoff(self) -> dict:
        return {"tool": "transfer_to_staff", "params": {"summary": "Caller wants a renewal."}}

    async def test_an_unrequested_handoff_is_rechecked_once(self) -> None:
        backend, planner, executor, transcript, _journal = _session(
            [self._handoff(), self._handoff()], tools=(LOOKUP, RENEW, HANDOFF)
        )
        transcript.record_user("What does my member record say?")

        await backend.call("Look up member jordan_lee_82.")

        self.assertIn("Hand off only if", planner.states[1]["validation_error"])
        self.assertEqual([name for name, _arguments in executor.calls], ["transfer_to_staff"])

    async def test_a_requested_handoff_goes_ahead_without_a_consent_question(self) -> None:
        backend, planner, executor, transcript, _journal = _session([self._handoff()], tools=(LOOKUP, RENEW, HANDOFF))
        transcript.record_user("I want to speak to a real person.")

        await backend.call("Transfer the caller to staff about a renewal.")

        self.assertEqual(len(planner.states), 2)
        self.assertNotIn("validation_error", planner.states[0])
        self.assertEqual([name for name, _arguments in executor.calls], ["transfer_to_staff"])


class RespellTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_not_found_identifier_is_answered_with_a_bounded_respell_question(self) -> None:
        missing = {
            "tool": "response_hint",
            "reason": "params_missing",
            "context": "lookup_member",
            "params_needed": ["member_id"],
        }
        backend, planner, _executor, transcript, _journal = _session(
            [{"tool": "lookup_member", "params": {"member_id": "jordan_lea_82"}}, missing],
            outputs={"lookup_member": "Error: member jordan_lea_82 not found"},
        )
        transcript.record_user("It is jordan lea 82.")

        texts = [(await backend.call("Look up member jordan_lea_82."))["response_text"]]
        for _ in range(2):
            planner.plans = [missing]
            texts.append((await backend.call("Look up the member."))["response_text"])

        # The first call also carries the failed lookup, so the question follows its text.
        self.assertIn("I couldn't find that member id.", texts[0])
        self.assertIn("spell the whole member id", texts[1])
        self.assertTrue(texts[2].startswith("Please tell me"))

    async def test_a_new_value_in_the_request_is_looked_up_instead(self) -> None:
        backend, planner, _executor, _transcript, _journal = _session(
            [{"tool": "lookup_member", "params": {"member_id": "jordan_lea_82"}}],
            outputs={"lookup_member": "Error: member jordan_lea_82 not found"},
        )
        await backend.call("Look up member jordan_lea_82.")
        planner.plans = [
            {
                "tool": "response_hint",
                "reason": "params_missing",
                "context": "lookup_member",
                "params_needed": ["member_id"],
            }
        ]

        payload = await backend.call("Look up member jordan_lee_82.")

        self.assertTrue(payload["response_text"].startswith("Please tell me"))


class DetachedCapTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_oldest_superseded_run_stops_beyond_the_cap(self) -> None:
        release = asyncio.Event()

        class _Slow(_Planner):
            async def plan(self, *, query: str, state: dict, **kwargs: Any) -> dict:
                await release.wait()
                return {"complete": True}

        backend, _planner, _executor, _transcript, _journal = _session([])
        backend._planner = _Slow()
        first = asyncio.create_task(backend.call("first request"))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(backend.call("second request"))
        await asyncio.sleep(0.01)
        self.assertFalse(first.done())
        third = asyncio.create_task(backend.call("third request"))
        await asyncio.sleep(0.01)

        self.assertTrue(first.done())
        self.assertFalse(second.done())
        release.set()
        await asyncio.wait_for(third, timeout=2)
        for task in (first, second):
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)


if __name__ == "__main__":
    unittest.main()
