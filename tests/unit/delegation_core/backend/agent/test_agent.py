# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Tests for the backend LLM agent: its source, the adapter contract, its model conductor, and runs on the real IM.

Switching ``backend.adapter`` to ``llm-agent`` changes nothing else: the IM writes the same responses it writes for the
backend LLM, and the agent's conductor only decides.
"""

import asyncio
import itertools
import json
import unittest
from collections.abc import Callable, Sequence
from typing import Any

from examples.frontend_backend_live.delegation_core.api.local import LocalResponsesClient
from examples.frontend_backend_live.delegation_core.api.service import ResponsesService
from examples.frontend_backend_live.delegation_core.app import build_core
from examples.frontend_backend_live.delegation_core.backend.adapters import ADAPTERS
from examples.frontend_backend_live.delegation_core.backend.agent import (
    AgentSettings,
    AgentSource,
    ConductorSettings,
    LlmConductor,
)
from examples.frontend_backend_live.delegation_core.backend.control import ControlVocabulary
from examples.frontend_backend_live.delegation_core.backend.port import (
    Answer,
    Cancel,
    ConductorInput,
    Request,
    Start,
    Steer,
    TaskSummary,
    WorkSummary,
)
from examples.frontend_backend_live.delegation_core.backend.providers import ChatCompletionsBackend
from examples.frontend_backend_live.delegation_core.backend.providers.translation import item_text
from examples.frontend_backend_live.delegation_core.backend.tools import CafeTools
from examples.frontend_backend_live.delegation_core.config import BackendChoice, CoreSettings
from examples.frontend_backend_live.delegation_core.im.input import BackendTask
from examples.frontend_backend_live.delegation_core.im.results import SpeakDisplay
from examples.shared.delegation_contract import ModelEndpoint
from tests.unit.contracts.backend import (
    TEXTS,
    BackendAdapterContract,
    ConductorContract,
    FakeDecisionTarget,
    a_request,
    a_round,
    call_events,
    message_events,
    text_round,
)
from tests.unit.core_helpers import CORE_BLOCK, CORE_TEXTS, core_block
from tests.unit.delegation_core.backend.llm.doubles import (
    FAST,
    LIMITS,
    FakeChat,
    IMTask,
    ScriptedProvider,
    chunk,
    contract_rounds,
    contract_tools,
    drive,
    gated,
    tool_fragment,
)
from tests.unit.delegation_core.helpers import IMCase, ask, read_all

VOCABULARY = ControlVocabulary(TEXTS.control_tools, TEXTS.conductor_prompt, TEXTS.refusal("idle"), TEXTS.failed_output)
ENDPOINT = ModelEndpoint(model="test/model", base_url="http://localhost:1/v1", api_key="secret-key")
OTHER = ModelEndpoint(model="other/model", base_url="http://localhost:2/v1")
# The model conductor places every request, also while no Work is open, so the tests read its rounds.
CONDUCTOR = ConductorSettings()
# The shipped settings on the agent, with the envelope in every worker's instructions.
AGENT = core_block(backend={"adapter": "llm-agent"}, delegation={"result_format": "speak_display"})
SETTINGS = AgentSettings(conductor=CONDUCTOR, worker=FAST)


def lookup(model: str | None) -> ModelEndpoint:
    """The catalog: the default model and one other; any other model is not listed."""
    return {None: ENDPOINT, ENDPOINT.model: ENDPOINT, OTHER.model: OTHER}[model]  # a KeyError is a LookupError


# --------------------------------------------------------------------------------------------- the conductor's model
_rounds = itertools.count(1)


def decisions(*calls: tuple[str, dict]) -> list[dict]:
    """One conductor round whose model calls the control tools ``(name, arguments)`` at once, in this order."""
    number = next(_rounds)
    parts = [
        call_events(name, arguments, call_id=f"c{number}_{index}", index=index)
        for index, (name, arguments) in enumerate(calls)
    ]
    return a_round(*parts, response_id=f"resp_c{number}")


def start(*delegations: str, tasks: Sequence[str] | None = None) -> tuple[str, dict]:
    arguments: dict[str, Any] = {"delegations": list(delegations)}
    if tasks is not None:
        arguments["tasks"] = list(tasks)
    return "start_work", arguments


def steer(work_id: str, *delegations: str, task_id: str | None = None) -> tuple[str, dict]:
    arguments: dict[str, Any] = {"work_id": work_id, "delegations": list(delegations)}
    if task_id is not None:
        arguments["task_id"] = task_id
    return "steer_work", arguments


def cancel(work_id: str, task_id: str | None = None) -> tuple[str, dict]:
    return "cancel_work", {"work_id": work_id, **({"task_id": task_id} if task_id is not None else {})}


def answer_status(delegation_id: str, text: str) -> tuple[str, dict]:
    return "answer_status", {"delegation_id": delegation_id, "text": text}


def round_data(payload: dict) -> dict:
    """What a conductor round's input carries: the JSON of its last message."""
    return json.loads(payload["input"][-1]["content"][0]["text"])


def start_each(payload: dict) -> list[dict]:
    """A conductor model that starts one Work for each new request it is given."""
    return decisions(*(start(request["delegation_id"]) for request in round_data(payload)["new_requests"]))


def conductor_on(model: ScriptedProvider, **settings: Any) -> LlmConductor:
    """An ``LlmConductor`` on ``model``, with the shipped vocabulary and ``settings`` over ``CONDUCTOR``."""
    return LlmConductor(
        lambda: model, VOCABULARY, CONDUCTOR.model_copy(update=settings), ENDPOINT.model, SETTINGS.max_running
    )


# --------------------------------------------------------------------------------------------- the workers' model
def texts_of(payload: dict) -> list[str]:
    """The text of each message of a request's input, oldest first."""
    return [item_text(item.get("content")) for item in payload["input"] if item.get("type", "message") == "message"]


def by_brief(scripts: dict[str, list[list]]) -> Callable[[dict], list]:
    """A workers' model that plays each task the next round of its brief's script (the newest brief it reads)."""
    lines = {TEXTS.task_line.format(task=brief): brief for brief in scripts}

    def respond(payload: dict) -> list:
        brief = next(lines[text] for text in reversed(texts_of(payload)) if text in lines)
        return scripts[brief].pop(0)

    return respond


# --------------------------------------------------------------------------------------------- the agent
def source(conductor, workers, *, settings: AgentSettings = SETTINGS, tools=None) -> AgentSource:
    """An ``AgentSource`` whose conductor runs on ``conductor`` and whose workers run on ``workers``."""
    return AgentSource(
        settings,
        endpoints=lookup,
        tools=lambda: list(tools or ()),
        vocabulary=VOCABULARY,
        make_provider=lambda endpoint: workers,
        make_conductor_provider=lambda endpoint: conductor,
    )


def a_work(work_id: str, *task_states: str, state: str = "running", delegation_id: str = "D1") -> WorkSummary:
    """A Work carrying ``delegation_id``, with one task per state (``W1.1``, ``W1.2`` ...)."""
    tasks = tuple(
        TaskSummary(f"{work_id}.{number}", task_state, None, ("get_menu: done",), True, "Looking.", 2.25)
        for number, task_state in enumerate(task_states or (state,), start=1)
    )
    return WorkSummary(work_id, state, delegation_id, (a_request(delegation_id, "Find teas."),), tasks, 3.04)


def build(block, *, tools=lambda: [], endpoints=lookup, **makers) -> AgentSource:
    return AgentSource(
        AgentSettings.model_validate(block), endpoints=endpoints, tools=tools, vocabulary=VOCABULARY, **makers
    )


def given(*requests, works=(), conversation=None, new_from=0) -> ConductorInput:
    if conversation is None:
        conversation = tuple(item for request in requests for item in request.input)
    return ConductorInput(tuple(requests), tuple(works), tuple(conversation), new_from)


async def decide(conductor, given_input) -> list:
    target = FakeDecisionTarget()
    await asyncio.wait_for(conductor.decide(given_input, target.place), 5)
    return target.decisions


def control_call(call_id: str, name: str, arguments: str, index: int = 0) -> list:
    """A chat model's reply chunk that calls a control tool."""
    return chunk(tool_calls=[tool_fragment(index, call_id, name, arguments)])


class AgentAdapterContractTests(BackendAdapterContract, unittest.IsolatedAsyncioTestCase):
    """The backend adapter contract on the backend LLM agent."""

    def make_adapter(self, scenario):
        self.gate = asyncio.Event()
        self.workers = ScriptedProvider(*contract_rounds(scenario, self.gate))
        return source(ScriptedProvider(), self.workers, tools=contract_tools(scenario)).new()

    def release(self):
        self.gate.set()

    def sent_instructions(self):
        return [payload.get("instructions") for payload in self.workers.payloads]


class LlmConductorContractTests(ConductorContract, unittest.IsolatedAsyncioTestCase):
    def make_conductor(self):
        return conductor_on(ScriptedProvider(respond=start_each))


class SourceTests(unittest.IsolatedAsyncioTestCase):
    def test_the_block_names_what_it_serves_and_it_describes_that_without_credentials(self):
        conductor = {"provider": "openai-responses", "model": "other/model", "timeout_seconds": 4.0}
        conductor |= {"memory_tokens": 2000, "max_output_tokens_per_request": 128, "temperature": None}
        conductor |= {"reasoning_effort": "low"}
        worker = {"provider": "openai-responses", "max_tool_rounds": 5}
        block = {"max_running": 4, "conductor": conductor, "worker": worker}
        described = build(block, tools=lambda: [CafeTools()]).describe()
        self.assertEqual(
            [described[key] for key in ("adapter", "model", "max_running")], ["llm-agent", "test/model", 4]
        )
        self.assertEqual({key: described["conductor"][key] for key in conductor}, conductor)
        self.assertEqual(described["worker"], worker)
        self.assertEqual([tool["name"] for tool in described["tools"]], ["get_menu", "manage_cart", "place_order"])
        self.assertNotIn("secret-key", json.dumps(described))
        self.assertEqual(build({"max_running": 1}).describe()["conductor"], "steer")  # the rule places, no model

    def test_a_bad_block_fails_when_the_core_is_built(self):
        cases = {
            "not a mapping": ("fast", "backend.llm-agent\n.*valid dictionary"),
            "an unknown key": ({"workers": 3}, "backend.llm-agent\nworkers\n.*Extra inputs"),
            "no task may run": ({"max_running": 0}, "max_running\n.*greater than or equal to 1"),
            "an unknown conductor key": ({"conductor": {"tool_choice": "auto"}}, "conductor.tool_choice\n.*Extra"),
            "an unknown provider": ({"conductor": {"provider": "hermes"}}, "Unknown backend provider 'hermes'"),
            "a negative wait": ({"conductor": {"timeout_seconds": -1}}, "conductor.timeout_seconds\n.*greater than 0"),
            "no memory": ({"conductor": {"memory_tokens": 0}}, "conductor.memory_tokens\n.*equal to 1"),
            "a negative temperature": ({"conductor": {"temperature": -0.1}}, "temperature\n.*equal to 0"),
            "an empty effort": ({"conductor": {"reasoning_effort": " "}}, "reasoning_effort\n.*non-empty text"),
            "an unlisted model": ({"conductor": {"model": "unlisted/model"}}, "'unlisted/model' is not in the model"),
            "a rule for requests with no open Work": ({"conductor": {"when_idle": "start"}}, "when_idle\n.*Extra"),
            "a worker's conductor": ({"worker": {"conductor": "steer"}}, "worker.conductor\n.*Extra inputs"),
            "a worker's unknown key": ({"worker": {"fold": True}}, "worker.fold\n.*Extra inputs"),
        }
        for name, (block, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                build(block)
        one_shot = {"slot": "s", "prompt": "p", "adapter": "llm-agent", "one_shot_adapter": "llm-agent"}
        shipped = CoreSettings.parse(CORE_BLOCK, CORE_TEXTS)
        settings = shipped.model_copy(update={"backend": BackendChoice.model_validate(one_shot)})
        with self.assertRaisesRegex(ValueError, "cannot serve one-shot sessions"):
            build_core(settings, endpoints=lookup, defaults=BackendTask("test/model", "x", []))

    async def test_the_model_check_and_the_settings_check_are_the_workers(self):
        built = build({})
        await built.check_model("other/model")
        with self.assertRaisesRegex(ValueError, "does not serve the model 'unlisted/model'"):
            await built.check_model("unlisted/model")
        # What the benchmark client sends: functions, tool_choice auto, one call per round, an effort.
        body = {"tools": [{"type": "function", "name": "get_menu"}], "tool_choice": "auto", "reasoning": {}}
        self.assertEqual(built.unsupported({**body, "parallel_tool_calls": False}), [])
        self.assertEqual([param for param, _ in built.unsupported({"tools": [{"type": "web_search"}]})], ["tools"])

    async def test_the_conductor_runs_on_a_copy_of_its_endpoint_with_its_own_temperature_and_thinking_off(self):
        thinking = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}, "temperature": 0.7}
        endpoint = ModelEndpoint("test/model", "http://localhost:1/v1", extra_params=thinking)
        for block, temperature in (({}, 0.0), ({"conductor": {"temperature": None}}, 0.7)):
            chat = FakeChat([control_call("p1", "start_work", '{"delegations": ["D1"]}')])
            make = {"make_conductor_provider": lambda at, chat=chat: ChatCompletionsBackend(at, LIMITS, chat)}
            built = build(block, endpoints=lambda model: endpoint, **make)
            await decide(built.new().conductor(), given(a_request("D1", "Teas?"), works=(a_work("W1"),)))
            sent = chat.requests[0]
            self.assertEqual(sent["temperature"], temperature)
            self.assertFalse(sent["extra_body"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(endpoint.extra_params["temperature"], 0.7)  # the catalog's entry is unchanged

    async def test_close_releases_the_workers_and_the_conductors_providers(self):
        conductor, workers = ScriptedProvider(decisions(start("D1"))), ScriptedProvider(text_round("Done."))
        adapter, task = source(conductor, workers).new(), IMTask()
        await decide(adapter.conductor(), given(a_request("D1", "Teas?")))
        await drive(adapter.start(task), task)
        await adapter.close()
        self.assertEqual((conductor.closed, workers.closed), (1, 1))


class ConductorTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_model_call_per_decide_with_the_prompt_the_four_tools_required_and_one_call(self):
        model = ScriptedProvider(decisions(start("D1", "D2")))
        placed = await decide(conductor_on(model), given(a_request("D1", "Teas?"), a_request("D2", "Hours?")))
        self.assertEqual(placed, [Start(("D1", "D2"))])
        [payload] = model.payloads
        self.assertEqual(payload["instructions"], TEXTS.conductor_prompt.format(max_running=3))  # the agent's limit
        self.assertEqual([tool["name"] for tool in payload["tools"]], list(TEXTS.control_tools))
        self.assertEqual((payload["tool_choice"], payload["parallel_tool_calls"]), ("required", False))
        self.assertEqual(payload["max_output_tokens"], 2 * 256)  # the cap grows with the requests it places
        self.assertEqual((payload["reasoning"], payload["model"]), ({"effort": "none"}, ENDPOINT.model))
        self.assertEqual(await decide(conductor_on(model), given()), [])  # no new request: no model call
        unset = ScriptedProvider(decisions(start("D1")))
        await decide(conductor_on(unset, reasoning_effort=None), given(a_request("D1", "Teas?")))
        self.assertEqual((len(model.payloads), "reasoning" in unset.payloads[0]), (1, False))  # the model's own

    async def test_the_round_reads_the_new_conversation_the_work_records_and_the_new_requests(self):
        said = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Tea is $3."}]}
        call = {"type": "function_call", "call_id": "x", "name": "get_menu", "arguments": "{}"}
        conversation = ({"type": "message", "role": "user", "content": "Hi."}, said, call)
        model, works = ScriptedProvider(decisions(steer("W1", "D2"))), (a_work("W1", "running", "answered"),)
        await decide(
            conductor_on(model), given(a_request("D2", "Green."), works=works, conversation=conversation, new_from=1)
        )
        data = round_data(model.payloads[0])
        self.assertEqual(data["conversation"], [{"role": "assistant", "text": "Tea is $3."}])  # from new_from, messages
        self.assertEqual(
            data["new_requests"], [{"delegation_id": "D2", "messages": [{"role": "user", "text": "Green."}]}]
        )
        [work] = data["works"]
        self.assertEqual([work[key] for key in ("work_id", "state", "seconds")], ["W1", "running", 3.0])
        self.assertEqual(
            work["requests"], [{"delegation_id": "D1", "messages": [{"role": "user", "text": "Find teas."}]}]
        )
        task = {"task_id": "W1.2", "state": "answered", "brief": None, "steps": ["get_menu: done"], "acted": True}
        self.assertEqual(work["tasks"][1], {**task, "progress": "Looking.", "seconds": 2.2})

    async def test_each_control_call_becomes_its_decision_and_every_call_is_placed(self):
        round_ = decisions(
            start("D1", tasks=["teas", "hours"]),
            steer("W1", "D2", task_id="W1.2"),
            cancel("W1", "W1.1"),
            answer_status("D3", "Still looking at the teas."),
            start("D1"),  # a repeat: placed too, and the IM keeps the first placement
            steer("W9", "D2"),  # an unshown Work: the IM refuses it
        )
        requests = (a_request("D1", "Teas and hours."), a_request("D2", "Green only."), a_request("D3", "How long?"))
        placed = await decide(conductor_on(ScriptedProvider(round_)), given(*requests, works=(a_work("W1"),)))
        expected = [Start(("D1",), ("teas", "hours")), Steer("W1", ("D2",), "W1.2"), Cancel("W1", "W1.1")]
        expected += [Answer("D3", "Still looking at the teas."), Start(("D1",)), Steer("W9", ("D2",))]
        self.assertEqual(placed, expected)

    async def test_a_round_that_fails_times_out_cannot_be_read_or_makes_no_valid_call_places_nothing(self):
        cut = a_round(message_events("start_work start_work"))
        cut[-1] = {"type": "response.incomplete", "response": {"incomplete_details": {"reason": "max_output_tokens"}}}
        malformed = (("steer_work", {"delegations": ["D2"]}), ("book_table", {}), ("start_work", {"delegations": "D2"}))
        text_none = [{"type": "input_text", "text": None}]
        cases = {
            "a model that is down": ([RuntimeError("model down")], {}),
            "a round past its limit": ([asyncio.Event()], {"timeout_seconds": 0.05}),
            "an answer with no call": (text_round("I would start it."), {}),
            "a round its output cap cut": (cut, {}),
            "malformed calls": (decisions(*malformed), {}),
            "a request that cannot be written out": (None, {}),  # no model call
        }
        for name, (round_, settings) in cases.items():
            with self.subTest(name):
                model = ScriptedProvider(*([round_] if round_ else []))
                content = text_none if round_ is None else "Also this."
                request = Request("D2", ({"type": "message", "role": "user", "content": content},))
                self.assertEqual(
                    await decide(conductor_on(model, **settings), given(request, works=(a_work("W1"),))), []
                )
                self.assertEqual(len(model.payloads), int(round_ is not None))

    async def test_the_next_round_reads_each_calls_outcome_or_error_and_a_failed_rounds_input(self):
        first = decisions(("steer_work", {"delegations": ["D1"]}), start("D1"))
        model = ScriptedProvider(first, [RuntimeError("model down")], decisions(start("D3")))
        conductor = conductor_on(model)
        for delegation_id in ("D1", "D2", "D3"):
            await decide(conductor, given(a_request(delegation_id, "Teas?")))
        first, failed, third = model.payloads
        message, malformed, call, error, outcome, failed_input, new = third["input"]
        self.assertEqual((message, failed_input), (first["input"][0], failed["input"][-1]))
        self.assertEqual((malformed["type"], outcome["call_id"]), ("function_call", call["call_id"]))
        self.assertEqual(json.loads(error["output"]), {"error": "steer_work: missing work_id"})
        self.assertEqual(outcome["output"], "Started W1 with 1 task(s).")
        self.assertEqual(round_data(third)["new_requests"][0]["delegation_id"], "D3")

    async def test_memory_keeps_whole_rounds_within_its_budget_and_a_cancelled_decide_leaves_none(self):
        for budget, kept in ((1, 3), (8000, 6)):  # 1: every round is over the budget alone, so only the newest stays
            with self.subTest(budget=budget):
                model = ScriptedProvider(respond=start_each)
                conductor = conductor_on(model, memory_tokens=budget)
                for delegation_id in ("D1", "D2", "D3"):
                    await decide(conductor, given(a_request(delegation_id, "x" * 300)))
                self.assertEqual(len(model.payloads[-1]["input"]), kept + 1)
        model = ScriptedProvider([asyncio.Event()], decisions(start("D1", "D2")))
        conductor = conductor_on(model)
        cancelled = asyncio.ensure_future(conductor.decide(given(a_request("D1", "Teas?")), FakeDecisionTarget().place))
        while not model.payloads:
            await asyncio.sleep(0.01)
        cancelled.cancel()  # the IM's busy rule, before it placed anything
        await asyncio.gather(cancelled, return_exceptions=True)
        await decide(conductor, given(a_request("D1", "Teas?"), a_request("D2", "Green only.")))
        self.assertEqual(len(model.payloads[1]["input"]), 1)  # nothing of the cancelled round

    async def test_with_no_open_work_the_model_places_the_requests_too(self):
        model = ScriptedProvider(decisions(start("D1", "D2")))
        ended = a_work("W0", state="answered")  # the records list it; it takes no request
        placed = await decide(
            conductor_on(model), given(a_request("D1", "Teas?"), a_request("D2", "Hours?"), works=(ended,))
        )
        self.assertEqual((placed, len(model.payloads)), ([Start(("D1", "D2"))], 1))
        self.assertEqual([work["state"] for work in round_data(model.payloads[0])["works"]], ["answered"])


async def read(stream) -> list[dict]:
    """Every event of a response stream until it ends; one that does not end within 10 s fails the test."""
    return await asyncio.wait_for(read_all(stream), 10)


def answers(events: list[dict]) -> list[str]:
    """The texts of a response's ``final_answer`` items, in order."""
    return [
        event["item"]["content"][0]["text"]
        for event in events
        if event["type"] == "response.output_item.done" and event["item"].get("phase") == "final_answer"
    ]


def agent(conductor, workers, **kwargs):
    """One IM session's backend LLM agent, its conductor and its workers on scripted models."""
    return source(conductor, workers, **kwargs).new()


def two_tasks(teas: list[list], hours: list[list]) -> ScriptedProvider:
    """Workers that run the ``teas`` and ``hours`` tasks of a split request, each by its own script."""
    return ScriptedProvider(respond=by_brief({"teas": teas, "hours": hours}))


class AgentScenarioTests(IMCase):
    async def test_a_request_split_into_two_tasks_runs_two_workers_at_once_and_answers_twice(self):
        teas, hours = asyncio.Event(), asyncio.Event()
        conductor = ScriptedProvider(decisions(start("D1", tasks=["teas", "hours"])))
        workers = two_tasks([gated(teas, text_round("Green and black."))], [gated(hours, text_round("Nine to five."))])
        im = self.session(agent(conductor, workers))
        stream = im.admit(ask("Which teas, and when are you open?"), delegation_id="D1").stream
        await self.turns()
        briefs = sorted(texts_of(payload)[-1] for payload in workers.payloads)  # both run at once, each with its brief
        self.assertEqual(briefs, [TEXTS.task_line.format(task="hours"), TEXTS.task_line.format(task="teas")])
        hours.set()
        teas.set()
        self.assertEqual(answers(await read(stream)), ["Nine to five.", "Green and black."])  # each its own item

    async def test_a_steer_into_one_task_reaches_that_task_only(self):
        teas, hours = asyncio.Event(), asyncio.Event()
        decided = (decisions(start("D1", tasks=["teas", "hours"])), decisions(steer("W1", "D2", task_id="W1.2")))
        conductor = ScriptedProvider(*decided)
        later = text_round("Ten to four on Sundays.")
        workers = two_tasks([gated(teas, text_round("Green and black."))], [gated(hours, text_round("Nine.")), later])
        im = self.session(agent(conductor, workers))
        first = im.admit(ask("Which teas, and when are you open?"), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("And on Sundays?"), delegation_id="D2")
        await self.turns()
        self.assertEqual([work["tasks"][1]["task_id"] for work in round_data(conductor.payloads[1])["works"]], ["W1.2"])
        teas.set()
        events = [*await read(first.stream), *await read(second.stream)]
        self.assertEqual(sorted(answers(events)), ["Green and black.", "Ten to four on Sundays."])
        # The hours task's streaming round was redone with the newer request, once; the teas task never read it.
        read_it = [any("And on Sundays?" in text for text in texts_of(payload)) for payload in workers.payloads]
        self.assertEqual(sorted(read_it), [False, False, True])

    async def test_a_cancel_of_one_task_stops_it_with_the_conductors_answer_and_the_other_task_answers(self):
        teas, hours = asyncio.Event(), asyncio.Event()
        stopped = decisions(cancel("W1", "W1.2"), answer_status("D2", "Stopped looking up the hours."))
        conductor = ScriptedProvider(decisions(start("D1", tasks=["teas", "hours"])), stopped)
        workers = two_tasks([gated(teas, text_round("Green and black."))], [gated(hours, text_round("Nine to five."))])
        im = self.session(agent(conductor, workers))
        first = im.admit(ask("Which teas, and when are you open?"), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("Skip the hours."), delegation_id="D2")
        self.assertEqual(answers(await read(second.stream)), ["Stopped looking up the hours."])
        await self.turns()  # the stop of W1.2 is confirmed before the other task answers
        teas.set()
        self.assertEqual(answers(await read(first.stream)), [TEXTS.cancelled_line, "Green and black."])
        self.assertEqual(len(workers.payloads), 2)  # the stopped task's round was cut, and none followed it

    async def test_an_answer_while_no_work_is_open_is_refused_and_the_request_starts_work(self):
        # No records stand behind it, so the IM refuses the answer; the request is left, and the steer rule starts it.
        conductor = ScriptedProvider(decisions(answer_status("D1", "Your table is booked.")))
        workers = ScriptedProvider(text_round("Friday at seven is free; shall I book it?"))
        events = await read(
            self.session(agent(conductor, workers)).admit(ask("A table, Friday."), delegation_id="D1").stream
        )
        self.assertEqual(answers(events), ["Friday at seven is free; shall I book it?"])

    async def test_the_envelope_reaches_every_worker_and_never_the_conductor(self):
        envelope = TEXTS.result_formats["speak_display"]
        conductor = ScriptedProvider(decisions(start("D1", tasks=["teas", "hours"])))
        workers = two_tasks([text_round("Green.")], [text_round("Nine.")])
        im = self.session(agent(conductor, workers), result_format=SpeakDisplay(envelope))
        await read(im.admit(ask("Teas and hours?"), delegation_id="D1").stream)
        self.assertEqual(len(workers.payloads), 2)
        for payload in workers.payloads:
            self.assertIn(envelope, payload["instructions"])
            self.assertIn("show", [tool["name"] for tool in payload["tools"]])
        [asked] = conductor.payloads
        self.assertEqual(asked["instructions"], TEXTS.conductor_prompt.format(max_running=3))
        self.assertNotIn("show", [tool["name"] for tool in asked["tools"]])

    async def test_with_one_task_at_a_time_the_steer_rule_places_with_no_model_call_and_a_newer_request_joins(self):
        gate, conductor = asyncio.Event(), ScriptedProvider()  # never asked
        workers = ScriptedProvider(gated(gate, text_round("Old.")), text_round("Teas for two."))
        im = self.session(agent(conductor, workers, settings=SETTINGS.model_copy(update={"max_running": 1})))
        first = im.admit(ask("Find teas."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("For two."), delegation_id="D2")
        self.assertEqual(answers(await read(second.stream)), ["Teas for two."])
        gate.set()
        self.assertEqual((answers(await read(first.stream)), conductor.payloads), ([], []))  # handed over to D2


class CoreSwitchTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_conversation_is_answered_through_the_core_built_on_the_agent(self):
        # backend.adapter: llm-agent switches the core to the agent, with stateless rounds on llm; the envelope too.
        conductor = ScriptedProvider(respond=start_each)
        workers = ScriptedProvider(respond=lambda payload: text_round("A latte is $4.50."))

        class Scripted(AgentSource):
            @classmethod
            def from_settings(cls, block, *, endpoints, tools, vocabulary):
                makers = {"make_provider": lambda _: workers, "make_conductor_provider": lambda _: conductor}
                return cls(
                    AgentSettings.model_validate(block),
                    endpoints=endpoints,
                    tools=tools,
                    vocabulary=vocabulary,
                    **makers,
                )

        defaults = BackendTask("test/model", "Take orders.", [])
        adapters = {**ADAPTERS, "llm-agent": lambda: Scripted}
        core = build_core(CoreSettings.parse(AGENT, CORE_TEXTS), endpoints=lookup, defaults=defaults, adapters=adapters)
        self.addAsyncCleanup(core.aclose)
        self.assertEqual((core.describe()["adapter"], core.describe()["one_shot_adapter"]), ("llm-agent", "llm"))
        await core.check_model("test/model")
        client = LocalResponsesClient(
            ResponsesService(core.session, stateless_session=core.one_shot_session, backend=core)
        )
        conversation = (await client.create_conversation({}))["id"]
        self.addAsyncCleanup(client.delete_conversation, conversation)
        user = {"type": "message", "role": "user", "content": "A latte, please."}
        params = {"conversation": conversation, "input": [user], "metadata": {"live_delegation_id": "d1"}}
        stream = await client.create_response({**params, "stream": True})
        async with asyncio.timeout(10):
            events = [event async for event in stream]
        self.assertEqual((events[-1]["type"], answers(events)), ("response.completed", ["A latte is $4.50."]))
        self.assertEqual(len(conductor.payloads), 1)  # the model placed the request, though no Work was open
        self.assertIn(TEXTS.result_formats["speak_display"], workers.payloads[0]["instructions"])


if __name__ == "__main__":
    unittest.main()
