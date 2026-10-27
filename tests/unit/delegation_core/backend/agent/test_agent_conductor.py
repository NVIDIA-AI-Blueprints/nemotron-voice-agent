# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Tests for the backend LLM agent's conductor: one model call per ``decide()``, through the control tools."""

import asyncio
import json
import unittest

from delegation_core.backend.agent.memory import ConductorMemory, DecisionRound
from delegation_core.backend.agent.render import round_input
from delegation_core.backend.port import Answer, Cancel, ConductorInput, Request, Start, Steer
from delegation_core.backend.providers import ChatCompletionsBackend
from tests.unit.contracts.backend import ConductorContract, FakeDecisionTarget, a_round, message_events, text_round
from tests.unit.delegation_core.backend.agent.doubles import (
    ENDPOINT,
    LIMITS,
    TEXTS,
    VOCABULARY,
    Model,
    a_request,
    a_work,
    answer_status,
    cancel,
    conductor_on,
    decisions,
    round_data,
    start,
    start_each,
    steer,
)
from tests.unit.delegation_core.backend.llm.doubles import FakeChat, chunk, tool_fragment

TOOLS = ["start_work", "steer_work", "cancel_work", "answer_status"]


class LlmConductorContractTests(ConductorContract, unittest.IsolatedAsyncioTestCase):
    """The conductor contract, on a model that starts one Work for each new request."""

    def make_conductor(self):
        return conductor_on(Model(respond=start_each))


def given(*requests, works=(), conversation=None, new_from=0) -> ConductorInput:
    if conversation is None:
        conversation = tuple(item for request in requests for item in request.input)
    return ConductorInput(tuple(requests), tuple(works), tuple(conversation), new_from)


async def decide(conductor, given_input) -> list:
    target = FakeDecisionTarget()
    await asyncio.wait_for(conductor.decide(given_input, target.place), 5)
    return target.decisions


class RoundTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_model_call_with_the_prompt_the_four_tools_required_and_parallel(self):
        model = Model(decisions(start("D1", "D2")))
        placed = await decide(conductor_on(model), given(a_request("D1", "Teas?"), a_request("D2", "Hours?")))
        self.assertEqual(placed, [Start(("D1", "D2"))])
        [payload] = model.payloads
        self.assertEqual(payload["instructions"], TEXTS.conductor_prompt)
        self.assertEqual([tool["name"] for tool in payload["tools"]], TOOLS)
        self.assertEqual((payload["tool_choice"], payload["parallel_tool_calls"]), ("required", True))
        self.assertEqual(payload["max_output_tokens"], 2 * 256)  # the cap grows with the requests it places
        self.assertEqual((payload["reasoning"], payload["model"]), ({"effort": "none"}, ENDPOINT.model))

    async def test_the_round_reads_the_new_conversation_the_work_records_and_the_new_requests(self):
        earlier = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi."}]}
        said = {
            "type": "message",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": "Tea is $3."}],
        }
        call = {"type": "function_call", "call_id": "x", "name": "get_menu", "arguments": "{}"}
        model = Model(decisions(steer("W1", "D2")))
        works = (a_work("W1", "running", "answered"),)
        await decide(
            conductor_on(model),
            given(a_request("D2", "Make it green."), works=works, conversation=(earlier, said, call), new_from=1),
        )
        data = round_data(model.payloads[0])
        self.assertEqual(
            data["conversation"], [{"role": "assistant", "text": "Tea is $3."}]
        )  # from new_from, messages only
        self.assertEqual(
            data["new_requests"], [{"delegation_id": "D2", "messages": [{"role": "user", "text": "Make it green."}]}]
        )
        [work] = data["works"]
        self.assertEqual((work["work_id"], work["state"], work["seconds"]), ("W1", "running", 3.0))
        self.assertEqual(
            work["requests"], [{"delegation_id": "D1", "messages": [{"role": "user", "text": "Find teas."}]}]
        )
        self.assertEqual(
            work["tasks"][1],
            {
                "task_id": "W1.2",
                "state": "answered",
                "brief": None,
                "steps": ["get_menu: done"],
                "acted": True,
                "progress": "Looking.",
                "seconds": 2.2,
            },
        )

    async def test_each_control_tool_becomes_its_decision_and_every_call_is_placed(self):
        round_ = decisions(
            start("D1", tasks=["teas", "hours"]),
            steer("W1", "D2", task_id="W1.2"),
            cancel("W1", "W1.1"),
            answer_status("D3", "Still looking at the teas."),
            start("D1"),  # a repeat: placed too, and the IM keeps the first placement
        )
        requests = (a_request("D1", "Teas and hours."), a_request("D2", "Green only."), a_request("D3", "How long?"))
        placed = await decide(
            conductor_on(Model(round_)), given(*requests, works=(a_work("W1", "running", "running"),))
        )
        self.assertEqual(
            placed,
            [
                Start(("D1",), ("teas", "hours")),
                Steer("W1", ("D2",), "W1.2"),
                Cancel("W1", "W1.1"),
                Answer("D3", "Still looking at the teas."),
                Start(("D1",)),
            ],
        )

    async def test_no_new_request_makes_no_model_call(self):
        model = Model()
        self.assertEqual(await decide(conductor_on(model), given()), [])
        self.assertEqual(model.payloads, [])

    async def test_reasoning_left_out_keeps_the_models_own(self):
        model = Model(decisions(start("D1")))
        await decide(conductor_on(model, reasoning_effort=None), given(a_request("D1", "Teas?")))
        self.assertNotIn("reasoning", model.payloads[0])


class RulePlacementTests(unittest.IsolatedAsyncioTestCase):
    """What the model leaves, the backend LLM's steer rule places: a join while a Work is open, else a start."""

    async def check(self, round_, *, works=(), requests=("D2",), **settings) -> list:
        conductor = conductor_on(Model(round_), **settings)
        return await decide(conductor, given(*(a_request(d, "Also this.") for d in requests), works=works))

    async def test_a_failure_while_a_work_is_open_joins_it(self):
        placed = await self.check([RuntimeError("model down")], works=(a_work("W1"),))
        self.assertEqual(placed, [Steer("W1", ("D2",))])

    async def test_a_failure_with_no_work_open_starts_the_requests_together(self):
        self.assertEqual(await self.check([RuntimeError("model down")], requests=("D1", "D2")), [Start(("D1", "D2"))])
        ended = a_work("W0", state="answered")
        self.assertEqual(await self.check([RuntimeError("model down")], works=(ended,)), [Start(("D2",))])

    async def test_a_timeout_places_by_the_rule(self):
        never = asyncio.Event()
        placed = await self.check([never], works=(a_work("W1"),), timeout_seconds=0.05)
        self.assertEqual(placed, [Steer("W1", ("D2",))])

    async def test_a_round_whose_input_cannot_be_rendered_places_by_the_rule(self):
        # A request part whose text is null fails the round before its model call. The rule still places it (a join
        # while a Work is open), so decide() never raises and the IM never starts it beside the open Work.
        odd = Request("D2", ({"type": "message", "role": "user", "content": [{"type": "input_text", "text": None}]},))
        model = Model()
        conductor = conductor_on(model)
        self.assertEqual(await decide(conductor, given(odd, works=(a_work("W1"),))), [Steer("W1", ("D2",))])
        self.assertEqual(await decide(conductor, given(odd)), [Start(("D2",))])
        self.assertEqual((model.payloads, conductor.memory.items()), ([], []))

    async def test_a_round_with_no_call_places_by_the_rule(self):
        self.assertEqual(
            await self.check(text_round("I would start it."), works=(a_work("W1"),)), [Steer("W1", ("D2",))]
        )

    async def test_a_round_cut_by_its_output_cap_places_by_the_rule(self):
        cut = a_round(message_events("start_work start_work start_work"))
        cut[-1] = {
            "type": "response.incomplete",
            "response": {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
        }
        self.assertEqual(await self.check(cut), [Start(("D2",))])

    async def test_a_request_the_model_skipped_joins_the_open_work(self):
        placed = await self.check(decisions(start("D1")), works=(a_work("W1"),), requests=("D1", "D2"))
        self.assertEqual(placed, [Start(("D1",)), Steer("W1", ("D2",))])

    async def test_malformed_calls_are_skipped_and_the_rule_places_their_requests(self):
        round_ = decisions(
            ("steer_work", {"delegations": ["D2"]}), ("book_table", {}), ("start_work", {"delegations": "D2"})
        )
        self.assertEqual(await self.check(round_, works=(a_work("W1"),)), [Steer("W1", ("D2",))])

    async def test_a_decision_naming_an_unshown_work_or_task_leaves_its_request_to_the_rule(self):
        # The IM refuses both decisions (an unknown Work, an unknown task), so the request still waits for its place.
        unknown = await self.check(decisions(steer("W9", "D2")), works=(a_work("W1"),))
        self.assertEqual(unknown, [Steer("W9", ("D2",)), Steer("W1", ("D2",))])
        no_task = await self.check(decisions(steer("W1", "D2", task_id="W1.7")), works=(a_work("W1"),))
        self.assertEqual(no_task, [Steer("W1", ("D2",), "W1.7"), Steer("W1", ("D2",))])

    async def test_an_answered_or_steered_request_is_not_placed_again(self):
        placed = await self.check(decisions(answer_status("D2", "Nearly done.")), works=(a_work("W1"),))
        self.assertEqual(placed, [Answer("D2", "Nearly done.")])
        placed = await self.check(decisions(steer("W1", "D2", task_id="W1.1")), works=(a_work("W1"),))
        self.assertEqual(placed, [Steer("W1", ("D2",), "W1.1")])


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_outcome_line_is_the_calls_output_and_the_next_round_reads_it(self):
        model = Model(decisions(start("D1")), decisions(steer("W1", "D2")))
        conductor = conductor_on(model)
        await decide(conductor, given(a_request("D1", "Teas?")))
        await decide(conductor, given(a_request("D2", "Green."), works=(a_work("W1"),)))
        first, second = model.payloads
        self.assertEqual(len(first["input"]), 1)
        message, call, output, new = second["input"]
        self.assertEqual(message, first["input"][0])
        self.assertEqual(
            (call["type"], call["name"], json.loads(call["arguments"])),
            ("function_call", "start_work", {"delegations": ["D1"]}),
        )
        self.assertEqual(
            output, {"type": "function_call_output", "call_id": call["call_id"], "output": "Started W1 with 1 task(s)."}
        )
        self.assertEqual(round_data(second)["new_requests"][0]["delegation_id"], "D2")
        self.assertEqual(new["role"], "user")

    async def test_a_malformed_call_is_answered_with_why_so_every_call_has_its_output(self):
        model = Model(decisions(("steer_work", {"delegations": ["D1"]}), start("D1")), decisions(start("D2")))
        conductor = conductor_on(model)
        await decide(conductor, given(a_request("D1", "Teas?")))
        await decide(conductor, given(a_request("D2", "Hours?")))
        outputs = [item for item in model.payloads[1]["input"] if item["type"] == "function_call_output"]
        self.assertEqual(json.loads(outputs[0]["output"]), {"error": "steer_work: missing work_id"})
        self.assertEqual(outputs[1]["output"], "Started W1 with 1 task(s).")

    async def test_a_failed_round_keeps_its_input_so_its_conversation_is_not_lost(self):
        model = Model([RuntimeError("model down")], decisions(start("D2")))
        conductor = conductor_on(model)
        await decide(conductor, given(a_request("D1", "Teas?")))
        await decide(conductor, given(a_request("D2", "Hours?")))
        failed, new = model.payloads[1]["input"]
        self.assertEqual(failed, model.payloads[0]["input"][0])
        self.assertEqual(round_data({"input": [new]})["new_requests"][0]["delegation_id"], "D2")

    async def test_a_decide_cancelled_before_it_placed_leaves_the_memory_as_it_was(self):
        thinking = asyncio.Event()
        model = Model([thinking], decisions(start("D1", "D2")))
        conductor = conductor_on(model)
        first = asyncio.create_task(conductor.decide(given(a_request("D1", "Teas?")), FakeDecisionTarget().place))
        await asyncio.sleep(0.01)
        self.assertEqual(len(model.payloads), 1)  # it was waiting for its model
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        self.assertEqual(conductor.memory.items(), [])
        placed = await decide(conductor, given(a_request("D1", "Teas?"), a_request("D2", "Hours?")))
        self.assertEqual(placed, [Start(("D1", "D2"))])
        self.assertEqual(len(model.payloads[1]["input"]), 1)  # nothing of the cancelled round

    async def test_memory_keeps_whole_rounds_within_its_budget_and_always_the_newest(self):
        memory = ConductorMemory(budget=200)
        rounds = []
        for number in range(4):
            decision_round = DecisionRound(round_input(given(a_request(f"D{number}", "x" * 300))))
            decision_round.answered({"call_id": f"c{number}", "name": "start_work", "arguments": "{}"}, "Started.")
            memory.add(decision_round)
            rounds.append(decision_round.items())
        self.assertEqual(memory.items(), rounds[-1])  # every round is over the budget alone: only the newest stays
        roomy = ConductorMemory(budget=10_000)
        for items in rounds:
            decision_round = DecisionRound(items[0])
            decision_round.answered(items[1], items[2]["output"])
            roomy.add(decision_round)
        self.assertEqual(roomy.items(), [item for items in rounds for item in items])
        self.assertEqual(
            [item["type"] for item in roomy.items()[:3]], ["message", "function_call", "function_call_output"]
        )


class ChatProviderTests(unittest.IsolatedAsyncioTestCase):
    """The conductor on the real chat provider (template and translator), its client scripted."""

    async def test_parallel_calls_are_all_placed_and_the_memory_is_a_valid_chat_history(self):
        chat = FakeChat(
            [
                chunk(tool_calls=[tool_fragment(0, "p1", "start_work", '{"delegations": ["D1"]}')]),
                chunk(tool_calls=[tool_fragment(1, "p2", "answer_status", '{"delegation_id": "D2", "text": "Soon."}')]),
                chunk(finish="tool_calls"),
            ],
            [
                chunk(tool_calls=[tool_fragment(0, "p3", "steer_work", '{"work_id": "W1", "delegations": ["D3"]}')]),
                chunk(finish="tool_calls"),
            ],
        )
        provider = ChatCompletionsBackend(ENDPOINT, LIMITS, chat)
        conductor = conductor_on(None)
        conductor.make_provider = lambda: provider
        placed = await decide(conductor, given(a_request("D1", "Teas?"), a_request("D2", "How long?")))
        self.assertEqual(placed, [Start(("D1",)), Answer("D2", "Soon.")])
        await decide(conductor, given(a_request("D3", "Green."), works=(a_work("W1"),)))
        first, second = chat.requests
        self.assertEqual((first["tool_choice"], first["max_tokens"]), ("required", 512))
        self.assertNotIn("parallel_tool_calls", first)  # on by default; never sent to a chat endpoint
        self.assertEqual(
            [message["role"] for message in second["messages"]], ["system", "user", "assistant", "tool", "tool", "user"]
        )
        self.assertEqual(second["messages"][0]["content"], VOCABULARY.prompt)
        self.assertEqual(
            [call["function"]["name"] for call in second["messages"][2]["tool_calls"]], ["start_work", "answer_status"]
        )
        self.assertEqual(
            [message["content"] for message in second["messages"][3:5]], ["Started W1 with 1 task(s).", "Answered D2."]
        )

    async def test_close_closes_the_provider_once_a_round_made_it(self):
        model = Model(decisions(start("D1")))
        conductor = conductor_on(model)
        await conductor.close()
        self.assertEqual(model.closed, 0)
        await decide(conductor, given(a_request("D1", "Teas?")))
        await conductor.close()
        self.assertEqual(model.closed, 1)


if __name__ == "__main__":
    unittest.main()
