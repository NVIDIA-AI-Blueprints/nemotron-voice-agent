# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""The delegate tool contract (RC1), LLMDecider repair/fallback, guards, the rule decider and the verbalizer."""

from __future__ import annotations

import asyncio
import unittest

from _fdh_voice_fakes import PACKAGE, FakeChatModel, delegate_reply

from examples.frontend_delegation_hermes.core.frontend.decider import (
    DecisionContext,
    LLMDecider,
    RuleDecider,
    apply_guards,
)
from examples.frontend_delegation_hermes.core.frontend.delegate_tool import (
    ContractError,
    delegate_tool,
    parse_arguments,
    parse_reply,
)
from examples.frontend_delegation_hermes.core.frontend.llm import ChatReply
from examples.frontend_delegation_hermes.core.frontend.prompts import load_prompts
from examples.frontend_delegation_hermes.core.frontend.status_verbalizer import LLMVerbalizer, TemplateVerbalizer

PROMPTS = load_prompts(PACKAGE / "config" / "prompts.yaml")


def context(state: str = "NO_SESSION", *, asked: bool = False, text: str = "hello") -> DecisionContext:
    return DecisionContext(
        system_prompt="sys", messages=[{"role": "user", "content": text}], user_text=text, state=state,
        backend_asked_question=asked,
    )  # fmt: skip


class ContractTests(unittest.TestCase):
    def test_tool_schema_has_exactly_the_three_fields(self) -> None:
        tool = delegate_tool(PROMPTS)["function"]
        self.assertEqual(tool["name"], "delegate")
        self.assertEqual(set(tool["parameters"]["properties"]), {"delegate", "filler_text", "request"})
        self.assertEqual(tool["parameters"]["required"], ["delegate", "filler_text"])
        self.assertEqual(tool["parameters"]["properties"]["request"]["enum"], ["task", "status"])

    def test_request_defaults_to_task_and_invalid_values_are_tasks(self) -> None:
        self.assertEqual(parse_arguments('{"delegate": true, "filler_text": "ok"}').request, "task")
        self.assertEqual(
            parse_arguments('{"delegate": true, "filler_text": "", "request": "STATUS"}').request, "status"
        )
        self.assertEqual(parse_arguments('{"delegate": true, "filler_text": "", "request": "nope"}').request, "task")

    def test_contract_violations(self) -> None:
        for bad in (
            '{"filler_text": "x"}',
            '{"delegate": 1, "filler_text": "x"}',
            '{"delegate": true}',
            "not json",
            "[]",
        ):
            with self.assertRaises(ContractError):
                parse_arguments(bad)
        with self.assertRaises(ContractError):
            parse_reply([("delegate", "{}"), ("delegate", "{}")], None)
        with self.assertRaises(ContractError):
            parse_reply([], "   ")

    def test_plain_text_is_a_direct_reply(self) -> None:
        decision = parse_reply([], "You're welcome!")
        self.assertEqual(
            (decision.delegate, decision.filler_text, decision.repair), (False, "You're welcome!", "text_as_direct")
        )


class LLMDeciderTests(unittest.IsolatedAsyncioTestCase):
    def decider(self, model: FakeChatModel, **kwargs) -> LLMDecider:
        settings = {"timeout_ms": 1000, "max_repair_attempts": 1, "on_contract_error": "delegate", **kwargs}
        return LLMDecider(model, PROMPTS, **settings)

    async def test_clean_decision_forces_the_named_tool(self) -> None:
        model = FakeChatModel([delegate_reply(True, "On it.", "status")])
        decision = await self.decider(model).decide(context("WORKING"))
        self.assertEqual(
            (decision.delegate, decision.filler_text, decision.request, decision.repair), (True, "On it.", "status", "")
        )
        self.assertEqual(model.calls[0]["tool_choice"], {"type": "function", "function": {"name": "delegate"}})
        self.assertIn("state: WORKING", model.calls[0]["messages"][0]["content"])
        self.assertEqual(decision.usage, {"input_tokens": 50, "output_tokens": 9})

    async def test_one_repair_then_success(self) -> None:
        bad = ChatReply(content=None, tool_calls=[("delegate", '{"delegate": "maybe"}')])
        model = FakeChatModel([bad, delegate_reply(False, "")])
        decision = await self.decider(model).decide(context())
        self.assertEqual((decision.delegate, decision.repair), (False, "repaired"))
        self.assertIn("did not follow the contract", model.calls[1]["messages"][-1]["content"])

    async def test_second_failure_falls_back(self) -> None:
        bad = ChatReply(content=None, tool_calls=[("other", "{}")])
        decision = await self.decider(FakeChatModel([bad, bad])).decide(context())
        self.assertEqual((decision.delegate, decision.filler_text, decision.repair), (True, "", "contract_fallback"))
        decision = await self.decider(FakeChatModel([bad, bad]), on_contract_error="direct_empty").decide(context())
        self.assertEqual((decision.delegate, decision.repair), (False, "contract_fallback"))

    async def test_timeout_and_endpoint_errors_fall_back(self) -> None:
        slow = FakeChatModel([delegate_reply(False)], delay=1.0)
        decision = await self.decider(slow, timeout_ms=50).decide(context())
        self.assertEqual((decision.delegate, decision.repair), (True, "timeout"))
        decision = await self.decider(FakeChatModel([])).decide(context())
        self.assertEqual(decision.repair, "error:RuntimeError")


class HedgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_hedged_request_wins_over_a_slow_first_one(self) -> None:
        class SlowThenFast(FakeChatModel):
            async def complete(self, messages, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    await asyncio.sleep(1.0)
                return delegate_reply(True, "On it.")

        model = SlowThenFast()
        decider = LLMDecider(
            model, PROMPTS, timeout_ms=800, max_repair_attempts=0, on_contract_error="delegate", hedge_after_ms=50
        )
        decision = await decider.decide(context())
        self.assertEqual((decision.filler_text, decision.repair), ("On it.", ""))
        self.assertEqual(len(model.calls), 2)


class GuardAndRuleTests(unittest.IsolatedAsyncioTestCase):
    def test_answer_to_a_backend_question_is_delegated(self) -> None:
        from examples.frontend_delegation_hermes.core.frontend.delegate_tool import Decision

        held = Decision(delegate=False, filler_text="Great.")
        guarded = apply_guards(held, context("IDLE", asked=True), backend_question_needs_delegate=True)
        self.assertEqual(
            (guarded.delegate, guarded.filler_text, guarded.repair), (True, "Great.", "guard_backend_question")
        )
        self.assertIs(apply_guards(held, context("IDLE", asked=False), backend_question_needs_delegate=True), held)
        self.assertIs(apply_guards(held, context("WORKING", asked=True), backend_question_needs_delegate=True), held)
        self.assertIs(apply_guards(held, context("IDLE", asked=True), backend_question_needs_delegate=False), held)

    def test_backchannel_guard(self) -> None:
        from examples.frontend_delegation_hermes.core.frontend.delegate_tool import Decision

        words = ("mm-hmm", "okay")
        loud = Decision(delegate=True, filler_text="Sure, let me check.")
        guarded = apply_guards(
            loud, context("WORKING", text="Mm-hmm."), backend_question_needs_delegate=True, backchannel_words=words
        )
        self.assertEqual((guarded.delegate, guarded.filler_text, guarded.repair), (False, "", "guard_backchannel"))
        # "okay" after a backend question is an answer, and a real request is never touched.
        answer = apply_guards(
            loud,
            context("IDLE", asked=True, text="okay"),
            backend_question_needs_delegate=True,
            backchannel_words=words,
        )
        self.assertTrue(answer.delegate)
        request = apply_guards(
            loud,
            context("WORKING", text="okay check order 5"),
            backend_question_needs_delegate=True,
            backchannel_words=words,
        )
        self.assertIs(request, loud)

    async def test_rule_decider_covers_the_four_query_types(self) -> None:
        rules = RuleDecider()
        self.assertTrue((await rules.decide(context(text="check order 1234"))).delegate)
        ack = await rules.decide(context("WORKING", text="Okay, sure."))
        self.assertEqual((ack.delegate, ack.filler_text), (False, ""))
        status = await rules.decide(context("WORKING", text="what's the update so far?"))
        self.assertEqual((status.delegate, status.request), (True, "status"))
        task = await rules.decide(context("WORKING", text="what's the progress of order 1235"))
        self.assertEqual(task.request, "task")  # a question about a specific thing is a task
        idle = await rules.decide(context("IDLE", text="what's the update so far?"))
        self.assertEqual(idle.request, "task")


class VerbalizerTests(unittest.IsolatedAsyncioTestCase):
    SUMMARY = {"task": "check order 1234", "current_tool": "get_order", "tools_done": [{"name": "find_user"}]}

    async def test_llm_verbalizer_and_fallback(self) -> None:
        good = LLMVerbalizer(
            FakeChatModel([ChatReply(content="  I'm looking up your order now. ")]), PROMPTS, timeout_ms=500
        )
        spoken = await good.verbalize(self.SUMMARY)
        self.assertEqual((spoken.text, spoken.source), ("I'm looking up your order now.", "llm"))
        failing = LLMVerbalizer(FakeChatModel([]), PROMPTS, timeout_ms=500)
        spoken = await failing.verbalize(self.SUMMARY)
        self.assertEqual(spoken.source, "template_fallback")
        self.assertIn("working on get order", spoken.text)

    async def test_template(self) -> None:
        spoken = await TemplateVerbalizer(PROMPTS).verbalize({"tools_done": [{}, {}]})
        self.assertEqual(
            spoken.text, "I'm still working on it, and I've finished 2 steps so far. It should not be long."
        )


if __name__ == "__main__":
    unittest.main()
