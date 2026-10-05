# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D101, D102

"""Generic Frontend/Backend behaviour: ledger kinds, retries, late answers, consent, guards and hints.

Every schema and value here is invented; none comes from a benchmark.
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from typing import Any
from unittest.mock import patch

from examples.frontend_backend_agent.generic.argument_screen import ArgumentScreen
from examples.frontend_backend_agent.generic.backend import GenericThinkerBackend
from examples.frontend_backend_agent.generic.client_tools import build_client_tool_specs, format_client_result
from examples.frontend_backend_agent.generic.planner import PlanTruncatedError
from examples.frontend_backend_agent.src.frontend_verdict import decide, is_progress_phrase
from examples.frontend_backend_agent.src.history import (
    MAX_HISTORY_CHARS,
    ConversationTranscript,
    DelegationLedger,
)
from examples.frontend_backend_agent.src.runtime_context import (
    session_clock_from_instructions,
    session_runtime_fields,
    session_today,
)
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
PHONE = {
    "type": "function",
    "name": "find_member_by_phone",
    "description": "Find a member by phone number.",
    "parameters": {
        "type": "object",
        "properties": {"phone_number": {"type": "string"}},
        "required": ["phone_number"],
    },
}


class _Executor:
    def __init__(self, outputs: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.outputs = outputs or {}

    async def __call__(self, calls, _timeout):
        self.calls.extend(calls)
        return [self.outputs.get(name, {"ok": True}) for name, _arguments in calls]


class _Planner:
    """Return scripted plans per round; record every state it received."""

    def __init__(self, plans: list[dict[str, Any]] | None = None) -> None:
        self.plans = list(plans or [])
        self.states: list[dict[str, Any]] = []

    async def plan(self, *, query: str, state: dict, **_kwargs: Any) -> dict:
        self.states.append(dict(state))
        return self.plans.pop(0) if self.plans else {"complete": True}

    def prepare_session_update(self, **_kwargs: Any) -> None:
        return None


def _backend(
    planner: Any,
    executor: _Executor,
    ledger: DelegationLedger | None,
    tools: tuple[dict, ...] = (LOOKUP, RENEW),
) -> GenericThinkerBackend:
    specs = build_client_tool_specs(tools)
    return GenericThinkerBackend(
        planner=planner,
        tools={},
        enabled_tools=tuple(specs),
        client_tools=specs,
        client_tool_executor=executor,
        overall_timeout_seconds=5,
        planner_timeout_seconds=5,
        max_planning_rounds=3,
        conversation_ledger=ledger,
    )


def _confirm(member_id: str = "jordan_lee_82") -> dict:
    return {
        "tool": "response_hint",
        "reason": "confirmation_needed",
        "context": "renew_membership",
        "params": {"member_id": member_id},
    }


class LedgerKindTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_lookup_is_a_read_and_an_action_is_a_write(self) -> None:
        ledger = DelegationLedger()
        planner = _Planner(
            [
                {"tool": "lookup_member", "params": {"member_id": "jordan_lee_82"}},
                {"tool": "renew_membership", "params": {"member_id": "jordan_lee_82"}},
            ]
        )
        backend = _backend(planner, _Executor(), ledger)

        await backend.call("Look up jordan_lee_82.")
        await backend.call("Renew jordan_lee_82.")

        history = ledger.render()
        kinds = [(call["tool"], call["kind"]) for entry in history for call in entry["tool_calls"]]
        self.assertEqual(kinds, [("lookup_member", "read"), ("renew_membership", "write")])

    async def test_a_confirmation_records_tool_params_and_exact_text(self) -> None:
        ledger = DelegationLedger()
        backend = _backend(_Planner([_confirm()]), _Executor(), ledger)

        payload = await backend.call("Renew jordan_lee_82.")
        record = ledger.latest_confirmation()

        self.assertEqual(record.tool, "renew_membership")
        self.assertEqual(record.params, {"member_id": "jordan_lee_82"})
        self.assertEqual(record.text, payload["response_text"])
        self.assertFalse(record.summarized)
        self.assertEqual(ledger.render()[0]["confirmation"]["state"], "pending")

    async def test_history_with_confirmations_stays_within_its_budget(self) -> None:
        ledger = DelegationLedger()
        backend = _backend(_Planner([_confirm("x" * 70) for _ in range(20)]), _Executor(), ledger)
        for _ in range(20):
            await backend.call("Renew it.")

        self.assertLessEqual(len(json.dumps(ledger.render(), ensure_ascii=False)), MAX_HISTORY_CHARS)


class VerdictProgressPhraseTests(unittest.TestCase):
    RUNNING = "Look up member jordan_lee_82."

    def test_progress_phrases_continue_even_with_a_reworded_query(self) -> None:
        for utterance in (
            "Any progress?",
            "How much longer?",
            "One second.",
            "Give me a moment.",
            "I'll be right with you.",
            "Okay, any update?",
            "Let me know when you're done.",
            "[cough] hold on",
        ):
            with self.subTest(utterance=utterance):
                verdict = decide("new", "Check the member lookup status.", self.RUNNING, utterance)
                self.assertEqual((verdict.decision, verdict.reason), ("continue", "progress_phrase"))

    def test_requests_and_consent_stay_new(self) -> None:
        for utterance in ("can you do it", "give me one", "tell me the price", "Mumbai please", "yes, one second"):
            with self.subTest(utterance=utterance):
                self.assertFalse(is_progress_phrase(utterance))
                self.assertEqual(decide("continue", "Something else.", self.RUNNING, utterance).decision, "new")


class PlanRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_truncated_plan_is_retried_once(self) -> None:
        class Planner(_Planner):
            async def plan(self, *, query: str, state: dict, **kwargs: Any) -> dict:
                if not self.states:
                    self.states.append(dict(state))
                    raise PlanTruncatedError("cut off")
                return await super().plan(query=query, state=state, **kwargs)

        executor = _Executor()
        planner = Planner([{"tool": "lookup_member", "params": {"member_id": "jordan_lee_82"}}])
        await _backend(planner, executor, None).call("Look up jordan_lee_82.")

        self.assertEqual([state["planner_attempt"] for state in planner.states[:2]], [1, 2])
        self.assertEqual(executor.calls, [("lookup_member", {"member_id": "jordan_lee_82"})])

    async def test_a_rejected_plan_is_replanned_with_the_reason_and_no_values(self) -> None:
        executor = _Executor()
        planner = _Planner(
            [
                {"tool": "unknown_tool", "params": {"secret": "jordan_lee_82"}},
                {"tool": "lookup_member", "params": {"member_id": "jordan_lee_82"}},
            ]
        )
        await _backend(planner, executor, None).call("Look up jordan_lee_82.")

        self.assertEqual(planner.states[1]["validation_error"], "unknown tool: unknown_tool")
        self.assertNotIn("jordan_lee_82", planner.states[1]["validation_error"])
        self.assertEqual(len(executor.calls), 1)

    async def test_a_second_rejection_settles_without_executing(self) -> None:
        executor = _Executor()
        planner = _Planner([{"tool": "unknown_tool"}, {"tool": "unknown_tool"}])
        payload = await _backend(planner, executor, None).call("Do something.")

        self.assertEqual(executor.calls, [])
        self.assertEqual(payload["reason"], "no_action_needed")


class _GatedPlanner:
    """Plan per query: a query starting with 'first' waits for ``release``."""

    def __init__(self, first_plan: dict, later_plan: dict) -> None:
        self.release = asyncio.Event()
        self.first_plan = first_plan
        self.later_plan = later_plan

    async def plan(self, *, query: str, state: dict, **_kwargs: Any) -> dict:
        if state["planning_round"] > 1:
            return {"complete": True}
        if query.startswith("first"):
            await self.release.wait()
            return self.first_plan
        return self.later_plan


class LateAnswerTests(unittest.IsolatedAsyncioTestCase):
    async def _superseded(self, later_plan: dict) -> tuple[GenericThinkerBackend, asyncio.Task, dict]:
        planner = _GatedPlanner(_confirm(), later_plan)
        backend = _backend(planner, _Executor(), DelegationLedger())
        first = asyncio.create_task(backend.call("first renew jordan_lee_82"))
        await asyncio.sleep(0.01)
        second = await backend.call("okay thanks")
        self.planner = planner
        return backend, first, second

    async def test_a_superseded_question_is_spoken_when_nothing_newer_was(self) -> None:
        backend, first, second = await self._superseded({"complete": True})
        self.planner.release.set()

        late = await asyncio.wait_for(first, timeout=2)

        self.assertEqual(second["reason"], "no_action_needed")
        self.assertEqual(late["reason"], "confirmation_needed")
        self.assertTrue(any(backend.is_late_answer(run) for run in backend._late_delivered))

    async def test_a_newer_call_on_the_same_tool_wins(self) -> None:
        _backend_, first, _second = await self._superseded(_confirm("sam_rivera_44"))
        self.planner.release.set()

        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(first, timeout=2)

    async def test_a_withdrawal_drops_the_late_answer(self) -> None:
        backend, first, _second = await self._superseded({"complete": True})
        backend.cancel_active("user_cancelled")
        self.planner.release.set()

        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(first, timeout=2)

    async def test_a_newer_call_still_running_is_waited_for(self) -> None:
        planner = _GatedPlanner(_confirm(), {"complete": True})
        newer_release = asyncio.Event()
        original = planner.plan

        async def plan(*, query: str, state: dict, **kwargs: Any) -> dict:
            if query.startswith("newer") and state["planning_round"] == 1:
                await newer_release.wait()
            return await original(query=query, state=state, **kwargs)

        planner.plan = plan  # type: ignore[method-assign]
        backend = _backend(planner, _Executor(), DelegationLedger())
        first = asyncio.create_task(backend.call("first renew jordan_lee_82"))
        await asyncio.sleep(0.01)
        newer = asyncio.create_task(backend.call("newer okay"))
        await asyncio.sleep(0.01)
        planner.release.set()
        await asyncio.sleep(0.05)
        self.assertFalse(first.done())

        newer_release.set()
        await asyncio.wait_for(newer, timeout=2)
        self.assertEqual((await asyncio.wait_for(first, timeout=2))["reason"], "confirmation_needed")


def _consent_session(*, confirmation_tool: dict = RENEW) -> tuple[GenericThinkerBackend, _Planner, _Executor, Any]:
    journal = ConversationJournal()
    transcript = ConversationTranscript(lambda: journal)
    ledger = DelegationLedger(transcript)
    planner = _Planner([_confirm()])
    executor = _Executor()
    backend = _backend(planner, executor, ledger, tools=(LOOKUP, confirmation_tool))
    return backend, planner, executor, (journal, transcript)


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


class DirectWriteTests(unittest.IsolatedAsyncioTestCase):
    async def _asked(self, *, status: str = "completed", heard: str | None = None):
        backend, planner, executor, (journal, transcript) = _consent_session()
        transcript.record_user("Renew my membership, jordan_lee_82.")
        question = await backend.call("Renew membership jordan_lee_82.")
        _assistant(journal, heard if heard is not None else question["response_text"], "a1", status)
        return backend, planner, executor, journal, transcript, question

    async def test_a_bare_yes_issues_the_confirmed_call_without_a_new_plan(self) -> None:
        backend, planner, executor, _journal, transcript, question = await self._asked()
        self.assertEqual(
            question["response_text"],
            "Just to confirm, I will renew a membership for one year, with member id: jordan_lee_82. Shall I go ahead?",
        )
        planned_before = len(planner.states)
        transcript.record_user("Yes.")

        await backend.call("Yes, go ahead and renew jordan_lee_82.")

        self.assertEqual(executor.calls, [("renew_membership", {"member_id": "jordan_lee_82"})])
        # Only the finalizing round asked the Thinker; the write itself was not planned.
        self.assertEqual([state["planning_round"] for state in planner.states[planned_before:]], [2])

    async def test_the_approval_is_single_use(self) -> None:
        backend, planner, executor, journal, transcript, _question = await self._asked()
        transcript.record_user("Yes.")
        await backend.call("Yes, go ahead.")
        _assistant(journal, "Your membership is renewed.", "a2")
        transcript.record_user("Yes.")

        await backend.call("Yes.")

        self.assertEqual(len(executor.calls), 1)

    async def test_each_failing_condition_plans_instead(self) -> None:
        cases = {
            "a qualified yes": dict(reply="Yes, but next month."),
            "a truncated question": dict(heard="Just to confirm, I will renew a membership"),
            "an incomplete response": dict(status="incomplete"),
        }
        for name, case in cases.items():
            with self.subTest(case=name):
                backend, planner, executor, _journal, transcript, _question = await self._asked(
                    status=case.get("status", "completed"), heard=case.get("heard")
                )
                planned_before = len(planner.states)
                transcript.record_user(case.get("reply", "Yes."))

                await backend.call("Yes, go ahead.")

                self.assertEqual(executor.calls, [])
                self.assertEqual(planner.states[planned_before]["planning_round"], 1)
                self.assertEqual(backend.conversation_ledger.latest_confirmation().outcome, "expired")

    async def test_another_user_turn_in_between_voids_the_approval(self) -> None:
        backend, _planner, executor, _journal, transcript, _question = await self._asked()
        transcript.record_user("Hmm, let me think.")
        transcript.record_user("Yes.")

        await backend.call("Yes.")

        self.assertEqual(executor.calls, [])

    async def test_a_summarized_confirmation_never_approves_on_a_bare_yes(self) -> None:
        listing = {
            "type": "function",
            "name": "renew_membership",
            "description": "Renew memberships.",
            "parameters": {"type": "object", "properties": {"member_id": {"type": "array"}}},
        }
        backend, planner, executor, (journal, transcript) = _consent_session(confirmation_tool=listing)
        planner.plans = [_confirm(["a1", "b2"])]  # type: ignore[list-item]
        question = await backend.call("Renew both.")
        self.assertTrue(question["summarized"])
        _assistant(journal, question["response_text"], "a1")
        transcript.record_user("Yes.")

        await backend.call("Yes.")

        self.assertEqual(executor.calls, [])


class DoneGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_lookup_carries_the_no_change_notice(self) -> None:
        planner = _Planner([{"tool": "lookup_member", "params": {"member_id": "jordan_lee_82"}}])
        payload = await _backend(planner, _Executor(), DelegationLedger()).call("Look up jordan_lee_82.")

        self.assertIn("No change has been made", payload["no_change_notice"])
        self.assertNotIn("completed_actions", payload)

    async def test_a_successful_write_names_only_that_action(self) -> None:
        planner = _Planner([{"tool": "renew_membership", "params": {"member_id": "jordan_lee_82"}}])
        payload = await _backend(planner, _Executor(), DelegationLedger()).call("Renew jordan_lee_82.")

        self.assertEqual(payload["completed_actions"], ["renew_membership"])
        self.assertNotIn("no_change_notice", payload)

    async def test_a_failed_write_claims_nothing(self) -> None:
        planner = _Planner([{"tool": "renew_membership", "params": {"member_id": "jordan_lee_82"}}])
        executor = _Executor({"renew_membership": {"ok": False, "error": "renewal refused"}})
        payload = await _backend(planner, executor, DelegationLedger()).call("Renew jordan_lee_82.")

        self.assertNotIn("completed_actions", payload)
        self.assertIn("no_change_notice", payload)


class ArgumentScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_dictated_identifier_reaches_a_lookup_in_its_example_shape(self) -> None:
        executor = _Executor()
        planner = _Planner([{"tool": "lookup_member", "params": {"member_id": "Jordan underscore Lee underscore 82"}}])
        await _backend(planner, executor, None).call("Look up the member.")

        self.assertEqual(executor.calls, [("lookup_member", {"member_id": "jordan_lee_82"})])

    async def test_an_unfinished_identifier_is_asked_for_locally(self) -> None:
        ledger = DelegationLedger()
        executor = _Executor()
        planner = _Planner([{"tool": "lookup_member", "params": {"member_id": "jordan_lee"}}])
        payload = await _backend(planner, executor, ledger).call("Look up the member.")

        self.assertEqual(executor.calls, [])
        self.assertEqual(payload["reason"], "params_invalid")
        self.assertIn("j o r d a n underscore l e e", payload["response_text"])
        [call] = ledger.render()[0]["tool_calls"]
        self.assertEqual((call["kind"], call["state"]), ("read", "answered_locally"))

    async def test_a_write_is_never_rewritten(self) -> None:
        executor = _Executor()
        planner = _Planner([{"tool": "renew_membership", "params": {"member_id": "Jordan Lee 82"}}])
        await _backend(planner, executor, None).call("Renew the member.")

        self.assertEqual(executor.calls, [("renew_membership", {"member_id": "Jordan Lee 82"})])

    async def test_a_switched_off_screen_sends_values_unchanged(self) -> None:
        executor = _Executor()
        planner = _Planner([{"tool": "lookup_member", "params": {"member_id": "Jordan underscore Lee underscore 82"}}])
        with patch.dict(os.environ, {"FRONTEND_BACKEND_NORMALIZATION": "false"}):
            backend = _backend(planner, executor, None)
        await backend.call("Look up the member.")

        self.assertEqual(executor.calls[0][1]["member_id"], "Jordan underscore Lee underscore 82")


class ResultHintTests(unittest.IsolatedAsyncioTestCase):
    def _result(self, tool: str, arguments: dict, output: object) -> dict:
        return format_client_result(tool, arguments, output)

    def test_a_person_not_found_gets_escalating_hints_up_to_three(self) -> None:
        screen = ArgumentScreen.for_tools([LOOKUP], enabled=True, phone_hints=True)
        hints = []
        for _ in range(4):
            payload = self._result("lookup_member", {"member_id": "jordan_lee_28"}, {"error": "Member not found"})
            screen.add_result_hints(payload, is_read=True)
            hints.append(payload.get("thinker_hint"))

        self.assertIn("letter by letter", hints[0])
        self.assertIn("S as in Sam", hints[1])
        self.assertIsNone(hints[3])

    def test_the_clients_result_is_unchanged(self) -> None:
        screen = ArgumentScreen.for_tools([LOOKUP], enabled=True, phone_hints=True)
        output = {"error": "Member not found"}
        payload = self._result("lookup_member", {"member_id": "x"}, output)
        screen.add_result_hints(payload, is_read=True)

        self.assertEqual(payload["data"]["result"], {"error": "Member not found"})
        self.assertNotIn("thinker_hint", json.dumps(payload["data"]))

    async def test_a_phone_miss_gets_the_other_form_and_no_automatic_second_call(self) -> None:
        executor = _Executor({"find_member_by_phone": {"error": "No records found for that number"}})
        planner = _Planner([{"tool": "find_member_by_phone", "params": {"phone_number": "555-123-4567"}}])
        backend = _backend(planner, executor, None, tools=(PHONE,))
        await backend.call("Find the member by phone.")

        self.assertEqual(len(executor.calls), 1)
        hint = planner.states[1]["prior_tool_results"][-1]["thinker_hint"]
        self.assertIn("5551234567", hint)

    def test_an_eleven_digit_number_gets_no_phone_hint(self) -> None:
        screen = ArgumentScreen.for_tools([PHONE], enabled=True, phone_hints=True)
        payload = self._result("find_member_by_phone", {"phone_number": "1-555-123-4567"}, {"error": "not found"})
        screen.add_result_hints(payload, is_read=True)

        self.assertNotIn("1-555", payload.get("thinker_hint", ""))


class BuiltInToolTests(unittest.TestCase):
    def test_a_session_with_client_tools_gets_no_built_in_tools(self) -> None:
        from examples.frontend_backend_agent.generic.tools import TOOLS

        specs = build_client_tool_specs((LOOKUP,))
        backend = GenericThinkerBackend(
            planner=_Planner(),
            tools=TOOLS,
            enabled_tools=("get_weather", *specs),
            client_tools=specs,
        )
        self.assertEqual(backend._enabled_tools, ("lookup_member",))

        update = backend.prepare_session_update(instructions="", client_tools=[])
        self.assertEqual(update.enabled_tools, ("get_weather",))


class SessionDateTests(unittest.TestCase):
    def test_the_first_sentence_stating_the_present_wins(self) -> None:
        clock = session_clock_from_instructions(
            "You help members. The policy changed on 2020-01-01.\n"
            "The current time is 2031-05-15 15:00:00 EST. Be brief."
        )
        self.assertEqual((clock.date.isoformat(), clock.time, clock.timezone), ("2031-05-15", "15:00:00", "EST"))

    def test_no_stated_date_falls_back_to_the_clock(self) -> None:
        self.assertIsNone(session_clock_from_instructions("Effective 2020-01-01, renewals are yearly."))
        self.assertIn("local_datetime", session_runtime_fields(None))

    def test_the_explicit_override_wins_over_the_session(self) -> None:
        clock = session_clock_from_instructions("Today is 2031-05-15.")
        with patch.dict(os.environ, {"FRONTEND_BACKEND_AGENT_TODAY": "2032-02-02"}):
            self.assertEqual(session_today(clock).isoformat(), "2032-02-02")
            self.assertEqual(session_runtime_fields(clock)["date"], "2032-02-02")
        with patch.dict(os.environ, {"FRONTEND_BACKEND_AGENT_TODAY": ""}):
            self.assertEqual(session_runtime_fields(clock)["date"], "2031-05-15")


class PhoneHintSwitchTests(unittest.TestCase):
    def test_the_phone_hint_works_with_normalization_off(self) -> None:
        screen = ArgumentScreen.for_tools([PHONE], enabled=False, phone_hints=True)
        payload = format_client_result("find_member_by_phone", {"phone_number": "5551234567"}, {"error": "not found"})
        screen.add_result_hints(payload, is_read=True)

        self.assertIn("555-123-4567", payload["thinker_hint"])
        self.assertEqual(screen.screen("find_member_by_phone", {"phone_number": "5551234567"})[1], None)
