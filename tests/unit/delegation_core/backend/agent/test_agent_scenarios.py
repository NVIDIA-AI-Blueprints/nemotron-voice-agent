# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""End-to-end tests of the backend LLM agent on the real IM with scripted models.

Switching ``backend.adapter`` to ``llm-agent`` changes nothing else: the IM writes the same official endings it
writes for the backend LLM, and the agent's conductor only decides.
"""

import asyncio
import unittest
from dataclasses import replace

from delegation_core.api.local import LocalResponsesClient
from delegation_core.api.service import ResponsesService
from delegation_core.app import build_core
from delegation_core.backend.adapters import ADAPTERS
from delegation_core.backend.agent import AgentSettings, AgentSource
from delegation_core.config import CoreSettings
from delegation_core.im.input import BackendTask
from delegation_core.im.results import SpeakDisplay
from tests.unit.contracts.backend import call_round, text_round
from tests.unit.contracts.endings import load
from tests.unit.delegation_core.backend.agent.doubles import (
    FBL,
    SHIPPED,
    TEXTS,
    Model,
    answer_status,
    by_brief,
    cancel,
    decisions,
    lookup,
    round_data,
    source,
    start,
    start_each,
    steer,
    texts_of,
)
from tests.unit.delegation_core.backend.llm.doubles import RecordingTools
from tests.unit.delegation_core.helpers import IMCase, ask, read_all

SCENARIOS = load()
# A response that does not end within this many seconds fails its test instead of hanging the suite.
READ_LIMIT_SECONDS = 10


async def read(stream) -> list[dict]:
    """Every event of a response stream until it ends, within ``READ_LIMIT_SECONDS``."""
    return await asyncio.wait_for(read_all(stream), READ_LIMIT_SECONDS)


def gated(gate: asyncio.Event, round_: list) -> list:
    """``round_``, waiting at ``gate`` once it opened: a round still streaming until the test lets it go on."""
    return [round_[0], gate, *round_[1:]]


def answers(events: list[dict]) -> list[str]:
    """The texts of a response's ``final_answer`` items, in order."""
    return [
        event["item"]["content"][0]["text"]
        for event in events
        if event["type"] == "response.output_item.done" and event["item"].get("phase") == "final_answer"
    ]


def agent(conductor: Model, workers: Model, **kwargs):
    """One IM session's backend LLM agent, its conductor and its workers on scripted models."""
    return source(conductor, workers, **kwargs).new()


class AgentScenarioTests(IMCase):
    def assertScenario(self, name: str) -> None:
        self.assertEqual(self.timeline(), SCENARIOS[name]["timeline"])

    async def test_a_request_split_into_two_tasks_runs_two_workers_at_once_and_answers_twice(self):
        teas, hours = asyncio.Event(), asyncio.Event()
        conductor = Model(decisions(start("D1", tasks=["teas", "hours"])))
        workers = Model(
            respond=by_brief(
                {
                    "teas": [gated(teas, text_round("Green and black."))],
                    "hours": [gated(hours, text_round("Nine to five."))],
                }
            )
        )
        im = self.session(agent(conductor, workers))
        stream = im.admit(ask("Which teas, and when are you open?"), delegation_id="D1").stream
        await self.turns()
        self.assertEqual(len(workers.payloads), 2)  # both tasks run at once, each with its brief
        briefs = sorted(texts_of(payload)[-1] for payload in workers.payloads)
        self.assertEqual(briefs, [TEXTS.task_line.format(task="hours"), TEXTS.task_line.format(task="teas")])
        hours.set()
        teas.set()
        events = await read(stream)
        self.assertEqual(answers(events), ["Nine to five.", "Green and black."])  # each its own item, never merged
        self.assertEqual([task.task_id for task in im.ended_works[-1].tasks], ["W1.1", "W1.2"])
        self.assertScenario("one_work_two_tasks")

    async def test_a_steer_into_one_task_reaches_that_task_only(self):
        teas, hours = asyncio.Event(), asyncio.Event()
        conductor = Model(
            decisions(start("D1", tasks=["teas", "hours"])),
            decisions(steer("W1", "D2", task_id="W1.2")),
        )
        workers = Model(
            respond=by_brief(
                {
                    "teas": [gated(teas, text_round("Green and black."))],
                    "hours": [gated(hours, text_round("Nine to five.")), text_round("Ten to four on Sundays.")],
                }
            )
        )
        im = self.session(agent(conductor, workers))
        first = im.admit(ask("Which teas, and when are you open?"), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("And on Sundays?"), delegation_id="D2")
        await self.turns()
        self.assertEqual([w["tasks"][1]["task_id"] for w in round_data(conductor.payloads[1])["works"]], ["W1.2"])
        teas.set()
        events = [*await read(first.stream), *await read(second.stream)]
        hours_runs = [p for p in workers.payloads if TEXTS.task_line.format(task="hours") in texts_of(p)]
        teas_runs = [p for p in workers.payloads if TEXTS.task_line.format(task="teas") in texts_of(p)]
        self.assertEqual(len(hours_runs), 2)  # the streaming round was redone with the newer request (once)
        self.assertTrue(any("And on Sundays?" in text for text in texts_of(hours_runs[-1])))
        self.assertFalse(any("And on Sundays?" in text for p in teas_runs for text in texts_of(p)))
        self.assertEqual(sorted(answers(events)), ["Green and black.", "Ten to four on Sundays."])
        self.assertEqual((im.ledger.delegations["D1"].state, im.ledger.delegations["D2"].work), ("handed_over", "W1"))
        self.assertNotIn("W2", im.ledger.works)

    async def test_a_cancel_of_one_task_stops_it_and_the_other_task_answers(self):
        teas, hours = asyncio.Event(), asyncio.Event()
        conductor = Model(
            decisions(start("D1", tasks=["teas", "hours"])),
            decisions(cancel("W1", "W1.2"), answer_status("D2", "Stopped looking up the hours.")),
        )
        workers = Model(
            respond=by_brief(
                {
                    "teas": [gated(teas, text_round("Green and black."))],
                    "hours": [gated(hours, text_round("Nine to five."))],
                }
            )
        )
        im = self.session(agent(conductor, workers))
        first = im.admit(ask("Which teas, and when are you open?"), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("Skip the hours."), delegation_id="D2")
        self.assertEqual(answers(await read(second.stream)), ["Stopped looking up the hours."])
        await self.turns()  # the stop of W1.2 is confirmed before the other task answers
        teas.set()
        self.assertEqual(answers(await read(first.stream)), [TEXTS.cancelled_line, "Green and black."])
        self.assertEqual([task.record.state for task in im.ended_works[-1].tasks], ["answered", "cancelled"])
        self.assertEqual(im.ledger.works["W1"].state, "answered")  # one task stopped; the Work answered
        self.assertEqual(len(workers.payloads), 2)  # the stopped task's round was cut, and none followed it

    async def test_a_stop_is_a_cancel_and_an_answer_and_nothing_more_runs(self):
        gate = asyncio.Event()
        conductor = Model(
            decisions(start("D1")),
            decisions(cancel("W1"), answer_status("D2", "Stopped the tea search.")),
        )
        tools = RecordingTools(["book_table"])
        workers = Model(gated(gate, call_round("book_table", {})))
        im = self.session(agent(conductor, workers, tools=[tools]))
        first = im.admit(ask("Find teas."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("Never mind."), delegation_id="D2")
        self.assertEqual(answers(await read(second.stream)), ["Stopped the tea search."])
        self.assertEqual(answers(await read(first.stream)), [TEXTS.cancelled_line])
        gate.set()
        await self.turns()
        self.assertEqual(tools.calls, [])  # the cut round's call was never dispatched
        self.assertEqual(im.ledger.works["W1"].state, "cancelled")
        self.assertScenario("cancelled_for_a_newer_request_stopped")

    async def test_a_status_question_is_answered_from_the_records_and_the_work_runs_on(self):
        gate = asyncio.Event()
        conductor = Model(decisions(start("D1")), decisions(answer_status("D2", "Still looking at the teas.")))
        workers = Model(gated(gate, text_round("Green and black.")))
        im = self.session(agent(conductor, workers))
        first = im.admit(ask("Find teas."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("How is it going?"), delegation_id="D2")
        self.assertEqual(answers(await read(second.stream)), ["Still looking at the teas."])
        [work] = round_data(conductor.payloads[1])["works"]
        self.assertEqual((work["work_id"], work["state"]), ("W1", "running"))
        self.assertEqual(im.ledger.works["W1"].state, "running")
        gate.set()
        self.assertEqual(answers(await read(first.stream)), ["Green and black."])
        self.assertScenario("answered_by_the_conductor")

    async def test_a_conductor_failure_while_a_work_is_open_joins_it(self):
        gate = asyncio.Event()
        conductor = Model(decisions(start("D1")), [RuntimeError("the conductor's model is down")])
        workers = Model(gated(gate, text_round("Black tea.")), text_round("Green tea, as you asked."))
        im = self.session(agent(conductor, workers))
        first = im.admit(ask("Find teas."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("Green tea only."), delegation_id="D2")
        await read(first.stream)
        self.assertEqual(answers(await read(second.stream)), ["Green tea, as you asked."])
        self.assertEqual(im.ledger.delegations["D2"].work, "W1")
        self.assertNotIn("W2", im.ledger.works)  # no second Work beside the open one
        self.assertTrue(any("Green tea only." in text for text in texts_of(workers.payloads[-1])))
        self.assertScenario("handed_over_with_the_taker_named")

    async def test_a_conductor_failure_with_no_work_open_starts_the_request(self):
        conductor, workers = Model([RuntimeError("the conductor's model is down")]), Model(text_round("Done."))
        im = self.session(agent(conductor, workers))
        self.assertEqual(answers(await read(im.admit(ask("Find teas."), delegation_id="D1").stream)), ["Done."])
        self.assertEqual(im.ledger.delegations["D1"].work, "W1")
        self.assertScenario("answered")

    async def test_a_conductor_timeout_while_a_work_is_open_joins_it(self):
        gate, never = asyncio.Event(), asyncio.Event()
        conductor = Model(decisions(start("D1")), [never])
        workers = Model(gated(gate, text_round("Black tea.")), text_round("Green tea, as you asked."))
        settings = replace(source(conductor, workers).settings)
        settings = replace(settings, conductor=replace(settings.conductor, timeout_seconds=0.05))
        im = self.session(agent(conductor, workers, settings=settings))
        im.admit(ask("Find teas."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("Green tea only."), delegation_id="D2")
        self.assertEqual(answers(await read(second.stream)), ["Green tea, as you asked."])
        self.assertEqual(im.ledger.delegations["D2"].work, "W1")

    async def test_a_conductor_failure_after_the_open_work_ended_starts_the_request(self):
        # W1 ends while the model decides, then the model fails: the rule steers into W1 as it was shown, and the IM,
        # which re-checks every decision, starts a Work instead, since nothing joins an ended Work.
        round_gate, deciding = asyncio.Event(), asyncio.Event()
        conductor = Model(decisions(start("D1")), [deciding, RuntimeError("the conductor's model is down")])
        workers = Model(gated(round_gate, text_round("Black tea.")), text_round("Green tea."))
        im = self.session(agent(conductor, workers))
        first = im.admit(ask("Find teas."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("And a green one?"), delegation_id="D2")
        await self.turns()
        self.assertEqual([work["state"] for work in round_data(conductor.payloads[1])["works"]], ["running"])
        round_gate.set()
        self.assertEqual(answers(await read(first.stream)), ["Black tea."])
        deciding.set()
        self.assertEqual(answers(await read(second.stream)), ["Green tea."])
        self.assertEqual((im.ledger.works["W1"].state, im.ledger.delegations["D2"].work), ("answered", "W2"))
        outcomes = [fields["outcome"] for kind, fields in self.traces if kind == "im.placed"]
        self.assertEqual(outcomes[-1], TEXTS.outcomes["restarted"].format(work="W1", new="W2"))

    async def test_an_arrival_restarts_a_decide_that_placed_nothing_once(self):
        cancelled, restarted = asyncio.Event(), asyncio.Event()
        conductor = Model(
            [cancelled, *decisions(start("D1"))],
            [restarted, *decisions(start("D1", "D2"))],
            decisions(start("D3")),
        )
        workers = Model(text_round("One."), text_round("Two."))
        im = self.session(agent(conductor, workers))
        im.admit(ask("Find teas."), delegation_id="D1")
        await self.turns()
        im.admit(ask("Green only."), delegation_id="D2")  # the decide() placed nothing: it starts again with both
        await self.turns()
        third = im.admit(ask("And the hours?"), delegation_id="D3")  # the restarted call is not restarted again
        await self.turns()
        restarted.set()
        await read(third.stream)
        asked = [[r["delegation_id"] for r in round_data(p)["new_requests"]] for p in conductor.payloads]
        self.assertEqual(asked, [["D1"], ["D1", "D2"], ["D3"]])
        self.assertEqual([len(p["input"]) for p in conductor.payloads], [1, 1, 4])  # nothing of the cancelled round
        self.assertEqual([im.ledger.delegations[d].work for d in ("D1", "D2", "D3")], ["W1", "W1", "W2"])
        self.assertIn("im.decide.restarted", [kind for kind, _ in self.traces])

    async def test_no_call_is_dispatched_while_newer_input_waits(self):
        # After the run's one redo, a call written while a newer request waits to land is refused (the commit gate):
        # it is not run, and the next round reads its "not run" output and the request.
        first_round, second_round = asyncio.Event(), asyncio.Event()
        conductor = Model(decisions(start("D1")), decisions(steer("W1", "D2")), decisions(steer("W1", "D3")))
        tools = RecordingTools(["book_table"])
        workers = Model(
            gated(first_round, text_round("Old.")),
            gated(second_round, call_round("book_table", {}, call_id="call_book")),
            text_round("Booked for two, by the window."),
        )
        im = self.session(agent(conductor, workers, tools=[tools]))
        im.admit(ask("Book a table."), delegation_id="D1")
        await self.turns()
        im.admit(ask("For two."), delegation_id="D2")  # redoes the streaming round
        await self.turns()
        third = im.admit(ask("By the window."), delegation_id="D3")  # waits: the run had its one redo
        await self.turns()
        second_round.set()
        self.assertEqual(answers(await read(third.stream)), ["Booked for two, by the window."])
        self.assertEqual(tools.calls, [])
        last = workers.payloads[-1]["input"]
        outputs = [item["output"] for item in last if item.get("type") == "function_call_output"]
        self.assertEqual(outputs, [TEXTS.not_run_output])
        self.assertTrue(any("By the window." in text for text in texts_of(workers.payloads[-1])))

    async def test_no_call_is_dispatched_while_the_conductor_decides_on_newer_input(self):
        # The worker's round ends with a call while the conductor decides on a newer request: the IM refuses it and
        # holds the worker until the decision, so the conductor's Cancel stops it before another model round.
        round_gate, deciding = asyncio.Event(), asyncio.Event()
        conductor = Model(
            decisions(start("D1")),
            [deciding, *decisions(cancel("W1"), answer_status("D2", "Stopped the booking."))],
        )
        tools = RecordingTools(["book_table"])
        workers = Model(gated(round_gate, call_round("book_table", {})), text_round("Booked."))
        im = self.session(agent(conductor, workers, tools=[tools]))
        first = im.admit(ask("Book a table."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("Stop, do not book it."), delegation_id="D2")
        await self.turns()
        round_gate.set()  # the round ends with its call while the conductor still decides
        await self.turns()
        deciding.set()
        self.assertEqual(answers(await read(second.stream)), ["Stopped the booking."])
        self.assertEqual(tools.calls, [])
        self.assertEqual(answers(await read(first.stream)), [TEXTS.cancelled_line])
        self.assertEqual(len(workers.payloads), 1)  # no model round after the call was held

    async def test_the_envelope_reaches_every_worker_and_never_the_conductor(self):
        envelope = TEXTS.result_formats["speak_display"]
        conductor = Model(decisions(start("D1", tasks=["teas", "hours"])))
        workers = Model(respond=by_brief({"teas": [text_round("Green.")], "hours": [text_round("Nine.")]}))
        im = self.session(agent(conductor, workers), result_format=SpeakDisplay(envelope))
        await read(im.admit(ask("Teas and hours?"), delegation_id="D1").stream)
        self.assertEqual(len(workers.payloads), 2)
        for payload in workers.payloads:
            self.assertIn(envelope, payload["instructions"])
            self.assertIn("show", [tool["name"] for tool in payload["tools"]])
        [asked] = conductor.payloads
        self.assertEqual(asked["instructions"], TEXTS.conductor_prompt)
        self.assertNotIn(envelope, asked["instructions"])
        self.assertNotIn("show", [tool["name"] for tool in asked["tools"]])

    async def test_max_running_counts_the_tasks_of_every_work(self):
        gates = {brief: asyncio.Event() for brief in ("a", "b", "c", "d")}
        conductor = Model(decisions(start("D1", tasks=["a", "b"])), decisions(start("D2", tasks=["c", "d"])))
        workers = Model(respond=by_brief({brief: [gated(gate, text_round(brief))] for brief, gate in gates.items()}))
        im = self.session(agent(conductor, workers))
        first = im.admit(ask("A and B."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("C and D."), delegation_id="D2")
        await self.turns()
        self.assertEqual(len(workers.payloads), 3)  # max_running 3: the fourth task waits for a slot
        gates["a"].set()
        await self.turns()
        self.assertEqual(len(workers.payloads), 4)
        for gate in gates.values():
            gate.set()
        self.assertEqual(answers(await read(first.stream)), ["a", "b"])
        self.assertEqual(answers(await read(second.stream)), ["c", "d"])


class ProfileTests(unittest.IsolatedAsyncioTestCase):
    """``core.agent.yaml``: one key switches the core to the agent; the rest is ``core.yaml``'s, plus the envelope."""

    def setUp(self):
        self.agent = CoreSettings.load(FBL / "core.agent.yaml")

    def test_the_profile_changes_the_adapter_and_the_result_format_only(self):
        self.assertEqual((self.agent.backend.adapter, self.agent.backend.one_shot_adapter), ("llm-agent", "llm"))
        self.assertEqual(self.agent.delegation.result_format, "speak_display")
        self.assertEqual(replace(self.agent.delegation, result_format="plain"), SHIPPED.delegation)
        self.assertEqual(replace(self.agent.backend, adapter="llm"), SHIPPED.backend)
        self.assertEqual(self.agent.texts, SHIPPED.texts)

    async def test_the_core_built_from_the_profile_runs_the_agent_and_one_shot_rounds_on_llm(self):
        core = build_core(self.agent, endpoints=lookup, defaults=BackendTask("test/model", "Take orders.", []))
        self.addAsyncCleanup(core.aclose)
        self.assertIsInstance(core.source, AgentSource)
        self.assertEqual(type(core.one_shot_source).__name__, "LlmSource")
        self.assertEqual((core.describe()["adapter"], core.describe()["one_shot_adapter"]), ("llm-agent", "llm"))
        await core.check_model("test/model")

    async def test_a_conversation_is_answered_through_the_core_built_from_the_profile(self):
        conductor = Model(respond=start_each)
        workers = Model(respond=lambda payload: text_round("A latte is $4.50."))

        class Scripted(AgentSource):
            @classmethod
            def from_settings(cls, block, *, endpoints, tools, vocabulary):
                return cls(
                    AgentSettings.parse(block),
                    endpoints=endpoints,
                    tools=tools,
                    vocabulary=vocabulary,
                    make_provider=lambda endpoint: workers,
                    make_conductor_provider=lambda endpoint: conductor,
                )

        adapters = {**ADAPTERS, "llm-agent": lambda: Scripted}
        defaults = BackendTask("test/model", "Take orders.", [])
        core = build_core(self.agent, endpoints=lookup, defaults=defaults, adapters=adapters)
        self.addAsyncCleanup(core.aclose)
        client = LocalResponsesClient(
            ResponsesService(core.session, stateless_session=core.one_shot_session, backend=core)
        )
        conversation = (await client.create_conversation({}))["id"]
        self.addAsyncCleanup(client.delete_conversation, conversation)
        user = {"type": "message", "role": "user", "content": "A latte, please."}
        params = {
            "conversation": conversation,
            "input": [user],
            "metadata": {"live_delegation_id": "d1"},
            "stream": True,
        }
        stream = await client.create_response(params)
        async with asyncio.timeout(10):
            events = [event async for event in stream]
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(answers(events), ["A latte is $4.50."])
        self.assertEqual(conductor.payloads[0]["instructions"], TEXTS.conductor_prompt)
        self.assertIn(TEXTS.result_formats["speak_display"], workers.payloads[0]["instructions"])


if __name__ == "__main__":
    unittest.main()
