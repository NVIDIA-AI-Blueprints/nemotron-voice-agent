# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The backend LLM agent: its source, its model conductor, and its runs on the real IM."""

import asyncio
import itertools
import json
import unittest
from collections.abc import Callable
from dataclasses import replace
from typing import Any
from unittest import mock

import httpx
from openai import APIConnectionError

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
from examples.frontend_backend_live.delegation_core.backend.agent.memory import DecisionRound
from examples.frontend_backend_live.delegation_core.backend.agent.render import round_input
from examples.frontend_backend_live.delegation_core.backend.control import ControlVocabulary
from examples.frontend_backend_live.delegation_core.backend.port import (
    CallerMessage,
    Cancel,
    ConductorInput,
    DecisionRejected,
    Request,
    Start,
    Steer,
    TaskSpec,
    TaskSummary,
    WorkSummary,
    WorkTarget,
)
from examples.frontend_backend_live.delegation_core.backend.providers import (
    Backend,
    BrokenStreamError,
    ChatCompletionsBackend,
    OpenAIResponsesBackend,
    retry,
)
from examples.frontend_backend_live.delegation_core.backend.providers.translation import item_text
from examples.frontend_backend_live.delegation_core.backend.tools import CafeTools
from examples.frontend_backend_live.delegation_core.config import BackendChoice, CoreSettings
from examples.frontend_backend_live.delegation_core.im.input import BackendTask
from examples.shared.delegation_contract import ModelEndpoint
from tests.unit.core_helpers import CORE_BLOCK, CORE_TEXTS, FakeChat, chunk, core_block, tool_fragment
from tests.unit.delegation_core.helpers import (
    FAST,
    LIMITS,
    MENU,
    NO_RETRY,
    QUESTION,
    TEXTS,
    FakeDecisionTarget,
    FakeTaskContext,
    IMCase,
    ScriptedProvider,
    a_request,
    a_round,
    answers,
    app_outputs,
    ask,
    call_events,
    call_round,
    decide,
    drive,
    gated,
    logged,
    message_events,
    read_all,
    text_round,
)

VOCABULARY = ControlVocabulary(TEXTS.control_tools, TEXTS.conductor_prompt, TEXTS.refusal("idle"), TEXTS.failed_output)
ENDPOINT = ModelEndpoint(model="test/model", base_url="http://localhost:1/v1", api_key="secret-key")
OTHER = ModelEndpoint(model="other/model", base_url="http://localhost:2/v1", api="openai-responses")
CONDUCTOR = ConductorSettings()
SETTINGS = AgentSettings(conductor=CONDUCTOR, worker=FAST)


def lookup(model: str | None) -> ModelEndpoint:
    """Look up ``model`` in a two-model catalog; a ``KeyError`` (a ``LookupError``) for any other."""
    return {None: ENDPOINT, ENDPOINT.model: ENDPOINT, OTHER.model: OTHER}[model]


_rounds = itertools.count(1)


def decisions(*actions: dict) -> list[dict]:
    """Build one completed conductor response containing a complete action envelope."""
    number = next(_rounds)
    return a_round(
        call_events("submit_decision", {"actions": list(actions)}, call_id=f"c{number}"), response_id=f"resp_c{number}"
    )


def start(*delegations, tasks=None):
    return {
        "action": "start",
        "delegations": list(delegations),
        "tasks": [{"brief": brief} for brief in (tasks if tasks is not None else ["Caller request."])],
    }


def steer(work_id, *delegations, task_id=None):
    target = {"work_id": work_id, **({"task_ids": [task_id]} if task_id is not None else {})}
    return {"action": "steer", "delegations": list(delegations), "targets": [target]}


def cancel(work_id, task_id=None, delegations=("D2",)):
    target = {"work_id": work_id, **({"task_ids": [task_id]} if task_id is not None else {})}
    return {"action": "stop", "delegations": list(delegations), "targets": [target]}


def answer_status(delegation_id, *work_ids):
    return {
        "action": "caller_message",
        "kind": "status",
        "delegations": [delegation_id],
        "targets": [{"work_id": work_id} for work_id in work_ids],
    }


def round_data(payload: dict) -> dict:
    return json.loads(payload["input"][-1]["content"][0]["text"])


def start_each(payload: dict) -> list[dict]:
    return decisions(*(start(request["delegation_id"]) for request in round_data(payload)["new_requests"]))


def conductor_on(model: Backend, **settings: Any) -> LlmConductor:
    """Build an ``LlmConductor`` on ``model`` with the shipped vocabulary and ``settings`` over ``CONDUCTOR``."""
    return LlmConductor(
        lambda: model, VOCABULARY, CONDUCTOR.model_copy(update=settings), ENDPOINT.model, SETTINGS.max_running
    )


def given(*requests, works=(), conversation=None, new_from=0) -> ConductorInput:
    if conversation is None:
        conversation = tuple(item for request in requests for item in request.input)
    return ConductorInput(tuple(requests), tuple(works), tuple(conversation), new_from)


def a_work(work_id: str, *task_states: str, state: str = "running", delegation_id: str = "D1") -> WorkSummary:
    """Build a Work carrying ``delegation_id``, with one task per state (``W1.1``, ``W1.2`` ...)."""
    tasks = tuple(
        TaskSummary(f"{work_id}.{number}", task_state, None, ("get_menu: done",), True, "Looking.", 2.25)
        for number, task_state in enumerate(task_states or (state,), start=1)
    )
    return WorkSummary(work_id, state, delegation_id, (a_request(delegation_id, "Find teas."),), tasks, 3.04)


def texts_of(payload: dict) -> list[str]:
    return [item_text(item.get("content")) for item in payload["input"] if item.get("type", "message") == "message"]


def by_brief(scripts: dict[str, list[list]]) -> Callable[[dict], list]:
    """Build a workers' model that plays the next round for its recorded assignment."""

    def respond(payload: dict) -> list:
        brief = json.loads(texts_of(payload)[-1])["private_backend_state"]["initial_assignment"]
        return scripts[brief].pop(0)

    return respond


def source(conductor, workers, *, settings=SETTINGS):
    makers = {"make_provider": lambda endpoint: workers, "make_conductor_provider": lambda endpoint: conductor}
    return AgentSource(settings, endpoints=lookup, tools=lambda: [], vocabulary=VOCABULARY, **makers)


def build(block, *, tools=lambda: [], endpoints=lookup, **makers) -> AgentSource:
    settings = AgentSettings.model_validate(block)
    return AgentSource(settings, endpoints=endpoints, tools=tools, vocabulary=VOCABULARY, **makers)


def two_tasks(teas: list[list], hours: list[list]) -> ScriptedProvider:
    return ScriptedProvider(respond=by_brief({"teas": teas, "hours": hours}))


async def read(stream) -> list[dict]:
    return await read_all(stream, timeout=10)


class SourceTests(unittest.IsolatedAsyncioTestCase):
    def test_it_describes_what_it_serves_without_credentials(self):
        conductor = {"model": "other/model", "timeout_seconds": 4.0}
        conductor |= {"memory_tokens": 2000, "max_output_tokens_per_request": 128, "temperature": None}
        conductor |= {"reasoning_effort": "low"}
        worker = {"max_tool_rounds": 5}
        block = {"max_running": 4, "conductor": conductor, "worker": worker}
        built = build(block, tools=lambda: [CafeTools()])
        described = built.describe()
        self.assertEqual(
            [described[key] for key in ("adapter", "model", "api", "max_running")],
            ["llm-agent", "test/model", "chat-completions", 4],
        )
        self.assertEqual({key: described["conductor"][key] for key in conductor}, conductor)
        self.assertEqual(described["conductor"]["api"], "openai-responses")
        provider = built.conductor_provider()
        self.addAsyncCleanup(provider.close)
        self.assertIsInstance(provider, OpenAIResponsesBackend)
        self.assertEqual(described["worker"], worker)
        self.assertEqual([tool["name"] for tool in described["tools"]], ["get_menu", "manage_cart", "place_order"])
        self.assertNotIn("secret-key", json.dumps(described))
        self.assertEqual(build({"max_running": 1}).describe()["conductor"], "steer")
        self.assertEqual(build({}, endpoints=lambda model: OTHER).describe()["api"], "openai-responses")

    def test_a_bad_block_or_api_fails_when_the_core_is_built(self):
        cases = {
            "not a mapping": ("fast", "backend.llm-agent\n.*valid dictionary"),
            "an unknown key": ({"workers": 3}, "backend.llm-agent\nworkers\n.*Extra inputs"),
            "no task may run": ({"max_running": 0}, "max_running\n.*greater than or equal to 1"),
            "an unknown conductor key": ({"conductor": {"tool_choice": "auto"}}, "conductor.tool_choice\n.*Extra"),
            "a nonpositive conductor output cap": (
                {"conductor": {"max_output_tokens_per_request": 0}},
                "conductor.max_output_tokens_per_request\n.*greater than or equal to 1",
            ),
            "a provider key": ({"conductor": {"provider": "openai-responses"}}, "conductor.provider\n.*Extra"),
            "a negative wait": ({"conductor": {"timeout_seconds": -1}}, "conductor.timeout_seconds\n.*greater than 0"),
            "no memory": ({"conductor": {"memory_tokens": 0}}, "conductor.memory_tokens\n.*equal to 1"),
            "a negative temperature": ({"conductor": {"temperature": -0.1}}, "temperature\n.*equal to 0"),
            "an empty effort": ({"conductor": {"reasoning_effort": " "}}, "reasoning_effort\n.*non-empty text"),
            "an unlisted model": ({"conductor": {"model": "unlisted/model"}}, "'unlisted/model' is not in the model"),
            "a worker's unknown key": ({"worker": {"fold": True}}, "worker.fold\n.*Extra inputs"),
        }
        for name, (block, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                build(block)
        unknown = "Unknown backend API 'grpc'; known: chat-completions, openai-responses"
        catalogs = {
            "the default model's unknown API": {None: replace(ENDPOINT, api="grpc"), OTHER.model: OTHER},
            "the conductor model's unknown API": {None: ENDPOINT, OTHER.model: replace(OTHER, api="grpc")},
        }
        for name, catalog in catalogs.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, unknown):
                build({"conductor": {"model": OTHER.model}}, endpoints=catalog.__getitem__)
        one_shot = {"slot": "s", "prompt": "p", "adapter": "llm-agent", "one_shot_adapter": "llm-agent"}
        shipped = CoreSettings.parse(CORE_BLOCK, CORE_TEXTS)
        settings = shipped.model_copy(update={"backend": BackendChoice.model_validate(one_shot)})
        with self.assertRaisesRegex(ValueError, "cannot serve one-shot sessions"):
            build_core(settings, endpoints=lookup, defaults=BackendTask("test/model", "x", []))

    async def test_the_model_and_settings_checks_are_the_workers(self):
        built = build({})
        await built.check_model("other/model")
        with self.assertRaisesRegex(ValueError, "does not serve the model 'unlisted/model'"):
            await built.check_model("unlisted/model")
        unknown = build({}, endpoints={None: ENDPOINT, OTHER.model: replace(OTHER, api="grpc")}.__getitem__)
        with self.assertRaisesRegex(ValueError, "Unknown backend API 'grpc'"):
            await unknown.check_model(OTHER.model)
        # What a Live client sends: functions, tool_choice auto, one call per round, an effort.
        body = {"tools": [{"type": "function", "name": "get_menu"}], "tool_choice": "auto", "reasoning": {}}
        self.assertEqual(built.unsupported({**body, "parallel_tool_calls": False}), [])
        self.assertEqual([param for param, _ in built.unsupported({"tools": [{"type": "web_search"}]})], ["tools"])
        self.assertEqual(built.unsupported({"tools": [{"type": "web_search"}], "model": "other/model"}), [])

    async def test_the_conductor_runs_on_a_copy_of_its_endpoint(self):
        thinking = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}, "temperature": 0.7}
        endpoint = ModelEndpoint("test/model", "http://localhost:1/v1", extra_params=thinking)
        for block, temperature in (({}, 0.0), ({"conductor": {"temperature": None}}, 0.7)):
            chat = FakeChat(
                [
                    chunk(
                        tool_calls=[tool_fragment(0, "p1", "submit_decision", json.dumps({"actions": [start("D1")]}))]
                    ),
                    chunk(finish="tool_calls"),
                ]
            )
            make = {"make_conductor_provider": lambda at, chat=chat: ChatCompletionsBackend(at, NO_RETRY, chat)}
            built = build(block, endpoints=lambda model: endpoint, **make)
            adapter = built.new()
            self.addAsyncCleanup(adapter.close)
            placed = await decide(adapter.conductor(), given(a_request("D1", "Teas?"), works=(a_work("W1"),)))
            self.assertEqual(placed, [Start(("D1",), (TaskSpec("Caller request."),))])
            sent = chat.requests[0]
            self.assertNotIn("parallel_tool_calls", sent)
            self.assertEqual(sent["temperature"], temperature)
            self.assertFalse(sent["extra_body"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(endpoint.extra_params["temperature"], 0.7)

    async def test_close_releases_both_models_providers(self):
        conductor, workers = ScriptedProvider(decisions(start("D1"))), ScriptedProvider(text_round("Done."))
        adapter, task = source(conductor, workers).new(), FakeTaskContext()
        await decide(adapter.conductor(), given(a_request("D1", "Teas?")))
        await drive(adapter.start(task), task)
        await adapter.close()
        self.assertEqual((conductor.closed, workers.closed), (1, 1))

    async def test_workers_retry_a_round_and_the_conductor_only_an_opening(self):
        workers, task = ScriptedProvider(a_round(), text_round("Done.")), FakeTaskContext()
        settings = AgentSettings(conductor=CONDUCTOR, worker=FAST.model_copy(update={"provider_retries": 1}))
        with mock.patch.object(retry, "retry_delay", return_value=0.0):
            events = await drive(source(ScriptedProvider(), workers, settings=settings).new().start(task), task)
        self.assertEqual((answers(events), workers.opened), (["Done."], 2))
        dropped = APIConnectionError(request=httpx.Request("POST", "http://localhost:1/v1/chat/completions"))
        placing = decisions(start("D1"))
        cases = {
            "before its first event": ([dropped], LIMITS, [Start(("D1",), (TaskSpec("Caller request."),))], 2),
            "before completion": ([*placing[:-1], dropped], LIMITS, [], 1),
            "with no retry set": ([dropped], NO_RETRY, [], 1),
        }
        for name, (first, limits, placed, opened) in cases.items():
            with self.subTest(name), mock.patch.object(retry, "retry_delay", return_value=0.0) as delay:
                model = ScriptedProvider(first, decisions(start("D1")), limits=limits)
                if name == "before its first event":
                    decided = await decide(conductor_on(model), given(a_request("D1", "Teas?")))
                    self.assertEqual(decided, placed)
                else:
                    with self.assertRaises(APIConnectionError):
                        await decide(conductor_on(model), given(a_request("D1", "Teas?")))
                self.assertEqual((model.opened, delay.call_count), (opened, opened - 1))

    async def test_a_core_built_on_the_agent_repairs_a_cutoff_and_answers_a_conversation(self):
        def respond(payload):
            events = start_each(payload)
            return events if round_data(payload).get("placement_attempt") == 2 else events[:-1]

        conductor = ScriptedProvider(respond=respond)
        workers = ScriptedProvider(respond=lambda payload: text_round("A latte is $4.50."))

        class Scripted(AgentSource):
            @classmethod
            def from_settings(cls, block, *, endpoints, tools, vocabulary):
                makers = {"make_provider": lambda _: workers, "make_conductor_provider": lambda _: conductor}
                settings = AgentSettings.model_validate(block)
                return cls(settings, endpoints=endpoints, tools=tools, vocabulary=vocabulary, **makers)

        agent = core_block(backend={"adapter": "llm-agent"}, delegation={"result_format": "speak_display"})
        defaults, adapters = BackendTask("test/model", "Take orders.", []), {**ADAPTERS, "llm-agent": lambda: Scripted}
        core = build_core(CoreSettings.parse(agent, CORE_TEXTS), endpoints=lookup, defaults=defaults, adapters=adapters)
        self.addAsyncCleanup(core.aclose)
        self.assertEqual((core.describe()["adapter"], core.describe()["one_shot_adapter"]), ("llm-agent", "llm"))
        await core.check_model("test/model")
        service = ResponsesService(core.session, stateless_session=core.one_shot_session, backend=core)
        client = LocalResponsesClient(service, close_seconds=core.close_seconds, close_core=core.aclose)
        conversation = (await client.create_conversation({}))["id"]
        self.addAsyncCleanup(client.delete_conversation, conversation)
        user = {"type": "message", "role": "user", "content": "A latte, please."}
        params = {"conversation": conversation, "input": [user], "metadata": {"live_delegation_id": "d1"}}
        stream = await client.create_response({**params, "stream": True})
        async with asyncio.timeout(10):
            events = [event async for event in stream]
        self.assertEqual((events[-1]["type"], answers(events)), ("response.completed", ["A latte is $4.50."]))
        self.assertEqual(len(conductor.payloads), 2)
        self.assertIn("ended without a completed decision", round_data(conductor.payloads[1])["decision_rejection"])
        [worker_request] = workers.payloads
        self.assertIn(TEXTS.result_formats["speak_display"], worker_request["instructions"])


class ConductorTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_conductor_round_logs_one_correlated_usage_record_even_without_a_terminal(self):
        created = {"type": "response.created", "response": {"id": "observed_response", "model": "reported/model"}}
        counts = {"input_tokens": 11, "output_tokens": 3}
        completed = {"type": "response.completed", "response": {"id": "observed_response", "usage": counts}}
        incomplete = {
            "type": "response.incomplete",
            "response": {
                "id": "observed_response",
                "usage": counts,
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        }
        failed = {
            "type": "response.failed",
            "response": {"id": "observed_response", "usage": counts, "error": {"code": "test"}},
        }
        cases = (
            ("before first event", [RuntimeError("test stream failed")], "stream_error", RuntimeError),
            ("after response identity", [created, RuntimeError("test stream failed")], "stream_error", RuntimeError),
            ("deadline", [created, asyncio.Event()], "stream_error", TimeoutError),
            ("cancelled", [created, asyncio.Event()], "cancelled", asyncio.CancelledError),
            ("exhausted", [created], "stream_error", DecisionRejected),
            ("broken stream", [created, BrokenStreamError("test cutoff")], "stream_error", DecisionRejected),
            ("failed terminal", [created, failed], "response.failed", RuntimeError),
            ("incomplete terminal", [created, incomplete], "response.incomplete", ValueError),
            (
                "completed terminal",
                [created, *call_events("submit_decision", {"actions": [start("D1")]}, call_id="placed"), completed],
                "response.completed",
                None,
            ),
        )
        for name, script, terminal, error in cases:
            with self.subTest(ending=name):
                model = ScriptedProvider(script)
                conductor = conductor_on(model, **({"timeout_seconds": 0.02} if name == "deadline" else {}))
                request = replace(
                    given(a_request("D1", "A generic request.")), attempt=2, conversation_ref="trace_probe"
                )
                expected_payload = conductor.payload(request, DecisionRound(round_input(request)))
                del expected_payload["parallel_tool_calls"]
                task = asyncio.create_task(conductor.decide(request, FakeDecisionTarget().place))
                with logged("DEBUG") as lines:
                    try:
                        if name == "cancelled":
                            while not model.payloads:
                                await asyncio.sleep(0)
                            task.cancel()
                        if error is None:
                            await task
                        else:
                            with self.assertRaises(error):
                                await task
                        usage = [line for line in lines if line.startswith("conductor.round.usage")]
                        self.assertEqual(len(usage), 1)
                        row = usage[0]
                        for field in (
                            "backend=llm-agent role=conductor",
                            "round=1 attempt=2",
                            f"terminal={terminal}",
                            "conversation_ref=trace_probe",
                            "delegation_ids=['D1']",
                        ):
                            self.assertIn(field, row)
                        if name == "before first event":
                            self.assertIn("model=test/model model_source=configured_model", row)
                            self.assertIn("response_id=None", row)
                        else:
                            self.assertIn("model=reported/model model_source=translated_request", row)
                            self.assertIn("reported_model=None", row)
                            self.assertIn("response_id=observed_response", row)
                        if terminal in ("response.completed", "response.incomplete", "response.failed"):
                            self.assertIn(f"usage={counts} usage_missing=False", row)
                        else:
                            self.assertIn("usage=None usage_missing=True", row)
                        # Telemetry does not alter authored requests, placement-attempt identity or provider behavior.
                        self.assertEqual(model.payloads, [expected_payload])
                        self.assertEqual(conductor.model, ENDPOINT.model)
                    finally:
                        if not task.done():
                            task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                        await conductor.close()

    async def test_provider_setup_failure_does_not_invent_a_model_round_or_usage(self):
        lines = self.enterContext(logged("DEBUG"))
        model = ScriptedProvider()
        conductor = conductor_on(model)
        conductor.make_provider = mock.Mock(side_effect=RuntimeError("provider setup failed"))
        with self.assertRaisesRegex(RuntimeError, "provider setup failed"):
            await conductor.decide(given(a_request("D1", "A request.")), FakeDecisionTarget().place)
        self.assertEqual((conductor.round, model.payloads), (0, []))
        self.assertFalse(any(line.startswith("conductor.round.usage") for line in lines))

    async def test_one_model_call_per_decide_with_one_complete_tool(self):
        model = ScriptedProvider(decisions(start("D1", "D2")))
        ended = a_work("W0", state="answered")
        requests = (a_request("D1", "Teas?"), a_request("D2", "Hours?"))
        placed = await decide(conductor_on(model), given(*requests, works=(ended,)))
        self.assertEqual(placed, [Start(("D1", "D2"), (TaskSpec("Caller request."),))])
        [payload] = model.payloads
        self.assertEqual([work["state"] for work in round_data(payload)["works"]], ["answered"])
        self.assertEqual(payload["instructions"], TEXTS.conductor_prompt.format(max_running=3))
        self.assertEqual([tool["name"] for tool in payload["tools"]], list(TEXTS.control_tools))
        self.assertEqual(payload["tool_choice"], "required")
        self.assertNotIn("parallel_tool_calls", payload)
        self.assertNotIn("max_output_tokens", payload)
        self.assertEqual((payload["reasoning"], payload["model"]), ({"effort": "none"}, ENDPOINT.model))
        self.assertEqual(await decide(conductor_on(model), given()), [])
        unset = ScriptedProvider(decisions(start("D1")), endpoint=OTHER)
        await decide(conductor_on(unset, reasoning_effort=None), given(a_request("D1", "Teas?")))
        self.assertEqual((len(model.payloads), "reasoning" in unset.payloads[0]), (1, False))
        self.assertIs(unset.payloads[0]["parallel_tool_calls"], False)
        capped = ScriptedProvider(decisions(start("D1", "D2")))
        await decide(conductor_on(capped, max_output_tokens_per_request=128), given(*requests))
        self.assertEqual(capped.payloads[0]["max_output_tokens"], 2 * 128)

    async def test_the_round_reads_the_conversation_works_and_requests(self):
        said = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Tea is $3."}]}
        call = {"type": "function_call", "call_id": "x", "name": "get_menu", "arguments": "{}"}
        conversation = ({"type": "message", "role": "user", "content": "Hi."}, said, call)
        model, works = ScriptedProvider(decisions(steer("W1", "D2"))), (a_work("W1", "running", "answered"),)
        line = {"source_message_id": "D2", "sequence": 2, "text": "Green."}
        request = replace(a_request("D2", "Green."), caller_lines=(line,))
        await decide(conductor_on(model), given(request, works=works, conversation=conversation, new_from=1))
        data = round_data(model.payloads[0])
        self.assertEqual(data["conversation"], [{"role": "assistant", "text": "Tea is $3."}])
        green = [{"delegation_id": "D2", "messages": [{"role": "user", "text": "Green."}], "caller_lines": [line]}]
        self.assertEqual(data["new_requests"], green)
        [work] = data["works"]
        self.assertEqual([work[key] for key in ("work_id", "state", "seconds")], ["W1", "running", 3.0])
        teas = [{"delegation_id": "D1", "messages": [{"role": "user", "text": "Find teas."}]}]
        self.assertEqual(work["requests"], teas)
        task = {"task_id": "W1.2", "state": "answered", "brief": None, "steps": ["get_menu: done"], "acted": True}
        self.assertEqual(work["tasks"][1], {**task, "progress": "Looking.", "seconds": 2.2, "stop": "none"})

    async def test_a_tasks_open_question_is_read_only_while_it_holds_one(self):
        model = ScriptedProvider(decisions(steer("W1", "D2", task_id="W1.2")))
        work = a_work("W1", "running", "running")
        asked = replace(work, tasks=(work.tasks[0], replace(work.tasks[1], question=QUESTION)))
        await decide(conductor_on(model), given(a_request("D2", "A latte."), works=(asked,)))
        silent, asking = round_data(model.payloads[0])["works"][0]["tasks"]
        self.assertNotIn("question", silent)
        self.assertEqual(asking["question"], QUESTION)

    async def test_a_work_that_outlives_the_call_is_read_with_its_mark_and_answers(self):
        result = "The suppliers, ranked by price."
        model = ScriptedProvider(decisions(answer_status("D2", "W1")))
        work = a_work("W1", state="answered")
        kept = replace(work, after_call=True, tasks=(replace(work.tasks[0], answer=result),))
        await decide(conductor_on(model), given(a_request("D2", "Is my research ready?"), works=(kept,)))
        (read,) = round_data(model.payloads[0])["works"]
        self.assertEqual((read["continues_after_call"], read["tasks"][0]["answer"]), (True, result))

    async def test_each_control_call_becomes_its_decision(self):
        round_ = decisions(
            start("D1", tasks=["teas", "hours"]),
            steer("W1", "D2", task_id="W1.2"),
            cancel("W1", "W1.1"),
            answer_status("D3", "W1"),
            start("D1"),
            steer("W9", "D2"),
        )
        requests = (a_request("D1", "Teas and hours."), a_request("D2", "Green only."), a_request("D3", "How long?"))
        placed = await decide(conductor_on(ScriptedProvider(round_)), given(*requests, works=(a_work("W1"),)))
        expected = [
            Start(("D1",), (TaskSpec("teas"), TaskSpec("hours"))),
            Steer(("D2",), (WorkTarget("W1", ("W1.2",)),)),
            Cancel(("D2",), (WorkTarget("W1", ("W1.1",)),)),
        ]
        expected += [
            CallerMessage(("D3",), "status", targets=(WorkTarget("W1"),)),
            Start(("D1",), (TaskSpec("Caller request."),)),
            Steer(("D2",), (WorkTarget("W9"),)),
        ]
        self.assertEqual(placed, expected)

    async def test_a_round_that_fails_places_nothing(self):
        cut = a_round(message_events("start_work start_work"))
        cut[-1] = {"type": "response.incomplete", "response": {"incomplete_details": {"reason": "max_output_tokens"}}}
        malformed = (
            {"action": "steer", "delegations": ["D2"]},
            {"action": "book_table"},
            {"action": "start", "delegations": "D2"},
        )
        text_none = [{"type": "input_text", "text": None}]
        cases = {
            "a model that is down": ([RuntimeError("model down")], {}),
            "a round past its limit": ([asyncio.Event()], {"timeout_seconds": 0.05}),
            "an answer with no call": (text_round("I would start it."), {}),
            "a round its output cap cut": (cut, {}),
            "malformed calls": (decisions(start("D2"), *malformed), {}),
            "a request that cannot be written out": (None, {}),
        }
        for name, (round_, settings) in cases.items():
            with self.subTest(name):
                model = ScriptedProvider(*([round_] if round_ else []))
                content = text_none if round_ is None else "Also this."
                request = Request("D2", ({"type": "message", "role": "user", "content": content},))
                target = FakeDecisionTarget()
                with self.assertRaises((RuntimeError, TimeoutError, TypeError, ValueError)):
                    await conductor_on(model, **settings).decide(given(request, works=(a_work("W1"),)), target.place)
                self.assertEqual(target.decisions, [])
                self.assertEqual(len(model.payloads), int(round_ is not None))

        decision = json.dumps({"actions": [start("D2")]})
        calls = [
            chunk(tool_calls=[tool_fragment(0, "first", "submit_decision", decision)]),
            chunk(tool_calls=[tool_fragment(1, "second", "submit_decision", decision)]),
        ]
        for ending, rejection in (
            ([], "stream ended without a completed decision"),
            ([chunk(finish="tool_calls")], "must contain one complete decision call"),
        ):
            with self.subTest(rejection=rejection):
                client = FakeChat([*calls, *ending])
                provider = ChatCompletionsBackend(ENDPOINT, NO_RETRY, client)
                conductor = conductor_on(provider)
                self.addAsyncCleanup(conductor.close)
                target = FakeDecisionTarget()
                with self.assertRaisesRegex(DecisionRejected, rejection):
                    await conductor.decide(given(a_request("D2", "Also this.")), target.place)
                self.assertEqual(target.decisions, [])
                self.assertEqual(len(client.requests), 1)
                self.assertNotIn("parallel_tool_calls", client.requests[0])

    async def test_memory_keeps_whole_rounds_within_its_budget(self):
        # With a budget of 1, every round is over it alone, so only the newest stays.
        for budget, kept in ((1, 3), (8000, 6)):
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
        # The IM's busy rule cancels a decide that has placed nothing.
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)
        await decide(conductor, given(a_request("D1", "Teas?"), a_request("D2", "Green only.")))
        self.assertEqual(len(model.payloads[1]["input"]), 1)


class AgentTests(IMCase):
    def agent(self, conductor, workers, **kwargs):
        return source(conductor, workers, **kwargs).new()

    async def test_only_capacity_one_interprets_an_update_during_an_app_wait(self):
        updated = "Look for decaf too."
        settings = {"tools": [MENU]}
        for capacity in (1, 3):
            with self.subTest(capacity=capacity):
                conductor = ScriptedProvider(decisions(start("D1")), decisions(steer("W1", "D2")))
                rounds = [call_round("get_menu", {}, "call_menu")]
                if capacity == 1:
                    rounds.append(text_round("Still waiting for the menu."))
                workers = ScriptedProvider(*rounds, text_round("The menu has decaf."))
                im = self.session(
                    self.agent(conductor, workers, settings=SETTINGS.model_copy(update={"max_running": capacity}))
                )
                first = await read(im.admit(ask("Find a coffee."), delegation_id="D1", settings=settings).stream)
                im.admit(ask(updated), delegation_id="D2")
                await self.until(lambda im=im: not im.waits_for_decision, "placed caller update")
                if capacity == 1:
                    await self.until(lambda workers=workers: len(workers.payloads) == 2, "isolated control round")
                    tool_names = {tool["name"] for tool in workers.payloads[-1]["tools"]}
                    assert tool_names == {"cancel_work", "set_work_state", "caller_message", "resolve_questions"}
                    self.assertIn(updated, texts_of(workers.payloads[-1]))
                    self.assertFalse(conductor.payloads)
                else:
                    await self.turns()
                    self.assertEqual(len(workers.payloads), 1)
                    self.assertEqual(len(conductor.payloads), 2)

                outputs = app_outputs(first, output='{"decaf": true}')
                resumed_index = len(workers.payloads)
                im.admit_client_request(outputs, settings=settings)
                await self.until(
                    lambda workers=workers, start=resumed_index: len(workers.payloads) > start,
                    "normal business continuation",
                )
                self.assertIn(updated, texts_of(workers.payloads[resumed_index]))
                self.assertIn(outputs[0], workers.payloads[resumed_index]["input"])

    async def test_usage_correlation_is_stable_across_attachments_and_does_not_log_kept_authority(self):
        lines = self.enterContext(logged("DEBUG"))
        refs, authorities = [], []
        for _ in range(2):
            conductor = ScriptedProvider(decisions(start("D1")), decisions(start("D2")))
            workers = ScriptedProvider(text_round("One."), text_round("Two."))
            im = self.session(self.agent(conductor, workers))
            refs.append(im.conversation_ref)
            authorities.append(im.ledger.session.id)
            await read(im.admit(ask("Teas?"), delegation_id="D1").stream)
            await im.set_attachment(False)
            await im.set_attachment(True)
            await read(im.admit(ask("Hours?"), delegation_id="D2").stream)
            self.assertEqual(im.conversation_ref, refs[-1])
            self.assertTrue(all(refs[-1] not in json.dumps(payload) for payload in conductor.payloads))
        self.assertNotEqual(*refs)
        usage = self.traced("backend.round.usage")
        self.assertEqual([row["conversation_ref"] for row in usage], [refs[0], refs[0], refs[1], refs[1]])
        logged_usage = "\n".join(line for line in lines if line.startswith("conductor.round.usage"))
        for ref in refs:
            self.assertEqual(logged_usage.count(ref), 2)
        for authority in authorities:
            self.assertNotIn(authority, logged_usage)
            self.assertNotIn(authority, json.dumps(usage))

    async def test_reported_usage_is_correlated_on_every_worker_terminal(self):
        for terminal, tokens in (("failed", 7), ("incomplete", 7), ("completed", 0)):
            with self.subTest(terminal=terminal):
                self.traces.clear()
                kind = f"response.{terminal}"
                reply = "The teas are available."
                events = text_round(reply, response_id="observed_response")
                envelope = events[-1]["response"]
                envelope["usage"] = {"output_tokens": tokens}
                if terminal != "completed":
                    envelope.update(
                        error={"code": "server_error", "message": "failed"},
                        incomplete_details={"reason": "max_output_tokens"},
                    )
                events[-1]["type"] = kind
                backend = self.agent(
                    ScriptedProvider(),
                    ScriptedProvider(events),
                    settings=SETTINGS.model_copy(update={"max_running": 1}),
                )
                im = self.session(backend)
                result = await read(im.admit(ask("Teas?"), delegation_id="D1").stream)
                self.assertEqual(answers(result), [reply] if terminal == "completed" else [])
                [mirror] = self.traced("backend.round.usage")
                [attempt] = self.traced("backend.provider_attempt.ended")
                for row in (mirror, attempt):
                    self.assertEqual(row["usage"], {"output_tokens": tokens})
                    self.assertFalse(row["usage_missing"])
                    assert (row["conversation_ref"], row["response_id"]) == (im.conversation_ref, "observed_response")
                self.assertEqual((mirror["terminal"], mirror["round"]), (kind, 1))
                self.assertEqual(attempt["provider_terminal"], kind)
                self.assertIsNone(attempt["cost"])

    async def test_a_steer_into_one_task_reaches_that_task_only(self):
        for correction in (False, True):
            with self.subTest(correction=correction):
                self.traces.clear()
                await self.check_scoped_task_usage(correction)

    async def check_scoped_task_usage(self, correction):
        teas, hours = asyncio.Event(), asyncio.Event()
        decided = (decisions(start("D1", tasks=["teas", "hours"])), decisions(steer("W1", "D2", task_id="W1.2")))
        conductor = ScriptedProvider(*decided)
        later = text_round("Ten to four on Sundays.", response_id="round_hours")
        base = text_round("Green and black.", response_id="round_teas")
        base[-1]["response"]["usage"] = {"output_tokens": 4}
        hours_rounds = [gated(hours, text_round("Nine.")), later] if correction else [later]
        workers = two_tasks([gated(teas, base)], hours_rounds)
        im = self.session(self.agent(conductor, workers))
        first = im.admit(ask("Which teas, and when are you open?"), delegation_id="D1")
        await self.turns()
        streams = [first.stream]
        if correction:
            streams.append(im.admit(ask("And on Sundays?"), delegation_id="D2").stream)
            await self.turns()
            seen = round_data(conductor.payloads[1])["works"]
            self.assertEqual([work["tasks"][1]["task_id"] for work in seen], ["W1.2"])
        teas.set()
        events = [event for stream in streams for event in await read(stream)]
        self.assertEqual(sorted(answers(events)), ["Green and black.", "Ten to four on Sundays."])
        # The hours task's streaming round was redone with the newer request; the teas task never read it.
        read_it = [any("And on Sundays?" in text for text in texts_of(payload)) for payload in workers.payloads]
        self.assertEqual(sorted(read_it), [False, False, True] if correction else [False, False])
        completed = {
            row["task_id"]: row for row in self.traced("backend.round.usage") if row["terminal"] == "response.completed"
        }
        self.assertEqual(set(completed), {"W1.1", "W1.2"})
        for task_id, response, usage in (
            ("W1.1", "round_teas", {"output_tokens": 4}),
            ("W1.2", "round_hours", None),
        ):
            row = completed[task_id]
            self.assertEqual((row["backend"], row["role"], row["model"]), ("llm-agent", "worker", ENDPOINT.model))
            self.assertEqual((row["work_id"], row["conversation_ref"]), ("W1", im.conversation_ref))
            self.assertEqual((row["delegation_id"], row["response_id"]), ("D2" if correction else "D1", response))
            if not correction:
                self.assertEqual(row["round"], 1)
            self.assertEqual((row["usage"], row["usage_missing"]), (usage, usage is None))

    async def test_a_cancel_of_one_task_lets_the_other_answer(self):
        teas, hours = asyncio.Event(), asyncio.Event()
        stopped = decisions(cancel("W1", "W1.2"))
        conductor = ScriptedProvider(decisions(start("D1", tasks=["teas", "hours"])), stopped)
        workers = two_tasks(
            [gated(teas, text_round("Green and black."))],
            [gated(hours, text_round("Nine to five.")), text_round("The hours lookup stopped; no booking changed.")],
        )
        im = self.session(self.agent(conductor, workers))
        first = im.admit(ask("Which teas, and when are you open?"), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("Skip the hours."), delegation_id="D2")
        self.assertEqual(answers(await read(second.stream)), [TEXTS.control_requested_line])
        await self.turns()
        teas.set()
        first_answers = answers(await read(first.stream))
        self.assertIn("Green and black.", first_answers)
        self.assertNotIn("The hours lookup stopped; no booking changed.", first_answers)
        self.assertEqual(len(workers.payloads), 2)

    async def test_an_unknown_target_does_not_start_unrelated_work(self):
        conductor = ScriptedProvider(decisions(steer("W9", "D1")))
        workers = ScriptedProvider(text_round("Friday at seven is free; shall I book it?"))
        im = self.session(self.agent(conductor, workers))
        events = await read(im.admit(ask("A table, Friday."), delegation_id="D1").stream)
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertEqual((workers.payloads, im.ledger.works), ([], {}))

    async def test_with_one_task_at_a_time_the_steer_rule_places(self):
        gate, conductor = asyncio.Event(), ScriptedProvider()
        workers = ScriptedProvider(gated(gate, text_round("Old.")), text_round("Teas for two."))
        im = self.session(self.agent(conductor, workers, settings=SETTINGS.model_copy(update={"max_running": 1})))
        first = im.admit(ask("Find teas."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("For two."), delegation_id="D2")
        self.assertEqual(answers(await read(second.stream)), ["Teas for two."])
        gate.set()
        self.assertEqual((answers(await read(first.stream)), conductor.payloads), ([], []))
