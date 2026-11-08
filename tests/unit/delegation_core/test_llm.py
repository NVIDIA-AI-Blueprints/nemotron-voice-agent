# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The backend LLM: a task's run (its rounds, retries, calls, steering and stops), its source and the steer rule."""

import asyncio
import json
import time
import unittest
from unittest import mock

import httpx
from openai import APIConnectionError, APIError, BadRequestError, InternalServerError, RateLimitError

from examples.frontend_backend_live.delegation_core.backend.conductors import SteerOpenWork
from examples.frontend_backend_live.delegation_core.backend.llm import LlmSettings, LlmSource
from examples.frontend_backend_live.delegation_core.backend.llm.run import RoundFailedError
from examples.frontend_backend_live.delegation_core.backend.port import (
    ConductorInput,
    Start,
    Steer,
    TaskSummary,
    WorkSummary,
)
from examples.frontend_backend_live.delegation_core.backend.providers import (
    BrokenStreamError,
    ChatCompletionsBackend,
    OpenAIResponsesBackend,
    retry,
)
from examples.frontend_backend_live.delegation_core.backend.providers.retry import StreamErrorEvent
from examples.frontend_backend_live.delegation_core.backend.tools import TOOL_SOURCES, CafeTools
from examples.shared.delegation_contract import ModelEndpoint
from tests.unit.core_helpers import Sleep
from tests.unit.delegation_core.helpers import (
    FAST,
    MENU,
    STEERING,
    TEXTS,
    FakeTaskContext,
    Hang,
    IMCase,
    RecordingTools,
    ScriptedProvider,
    a_request,
    a_round,
    adapter,
    an_ended_work,
    answers,
    ask,
    call_events,
    call_round,
    check_events,
    decide,
    drive,
    function_calls,
    gated,
    items_of,
    kinds,
    message_events,
    read_all,
    said,
    source,
    text_round,
)

REQUEST = httpx.Request("POST", "http://localhost:1/v1/chat/completions")
DEFAULT = ModelEndpoint("test/model", "http://default:1/v1", api_key="secret-key", extra_params={"temperature": 0.2})
OTHER = ModelEndpoint("other/model", "http://other:2/v1", api="openai-responses")


def calls_round(*names, said=None):
    parts = [message_events(said, index=0)] if said else []
    for name in names:
        parts.append(call_events(name, {}, call_id=f"call_{name}", index=len(parts)))
    return a_round(*parts, response_id="resp_calls")


def failed_round():
    error = {"code": "server_error", "message": "The model fell over."}
    return [
        {"type": "response.created", "response": {"id": "resp_f", "status": "in_progress"}},
        {"type": "response.failed", "response": {"id": "resp_f", "status": "failed", "error": error}},
    ]


def stalled(text="Done."):
    """Return a round that goes silent past the idle limit after its first event."""
    round_ = text_round(text)
    return [round_[0], Sleep(1.0), *round_[1:]]


def status_error(kind, status):
    return kind("down", response=httpx.Response(status, request=REQUEST), body=None)


QUICK = LlmSettings(provider_retries=1, first_event_timeout_seconds=0.2, idle_timeout_seconds=0.2)
CREATED = text_round("x")[0]
ADDING = call_round("manage_cart", {"action": "add", "item_id": "lat"})
OVERLOADED = APIError("Overloaded", REQUEST, body=None)
# A transient error before a round's first event: the round never opened.
OPENING_FAILURES = {
    "a dropped connection": ([APIConnectionError(request=REQUEST)], APIConnectionError),
    "a rate limit": ([status_error(RateLimitError, 429)], RateLimitError),
    "a server error": ([status_error(InternalServerError, 503)], InternalServerError),
    "an error event first": ([OVERLOADED], StreamErrorEvent),
    "a network error": ([httpx.ConnectError("refused")], httpx.ConnectError),
}
# Every other failure that can clear up: a time limit, a broken or failed round, a transient error once it opened.
ROUND_FAILURES = {
    "no first event in time": ([Sleep(1.0)], TimeoutError),
    "a stream with no event": ([], BrokenStreamError),
    "an error event midway": ([CREATED, OVERLOADED], StreamErrorEvent),
    "a connection dropped after a call": ([*ADDING[:-1], APIConnectionError(request=REQUEST)], APIConnectionError),
    "a stall": (stalled(), TimeoutError),
    "no terminal": (text_round("x")[:-1], BrokenStreamError),
    "a second response.created": ([*text_round("x")[:2], *text_round("y")], BrokenStreamError),
    "a failed round": (failed_round(), RoundFailedError),
    "an empty round": (a_round(), RoundFailedError),
}


def outputs_sent(payload):
    return [item["call_id"] for item in payload["input"] if item.get("type") == "function_call_output"]


def count(payload, items):
    return sum(item in payload["input"] for item in items)


async def read(events, task, until=None):
    """Read ``events`` as the IM does, up to and including the first event of type ``until`` (or to the end)."""
    seen = []
    async for event in events:
        task.see(event)
        seen.append(event)
        if event["type"] == until:
            break
    return seen


def lookup(model):
    """Look up ``model`` in a two-model catalog; a ``KeyError`` (a ``LookupError``) for any other."""
    return {None: DEFAULT, DEFAULT.model: DEFAULT, OTHER.model: OTHER}[model]


def build(block, tools=lambda: [], make_provider=None):
    if make_provider is None:
        return LlmSource.from_settings(block, endpoints=lookup, tools=tools, vocabulary=None)
    return LlmSource(LlmSettings.model_validate(block), endpoints=lookup, tools=tools, make_provider=make_provider)


class TypedRunTask(FakeTaskContext):
    """A task whose steering is a typed run with its own settings: once it lands, they are the Work's."""

    def __init__(self, brings: dict, **kwargs):
        super().__init__(**kwargs)
        self.brings = brings

    def settle(self, steering, delivery):
        super().settle(steering, delivery)
        if delivery == "applied":
            self.instructions, self.declared = self.brings["instructions"], list(self.brings["tools"])


class ChoiceTask(FakeTaskContext):
    """A task whose request sets its own tool choice."""

    def __init__(self, tool_choice, **kwargs):
        super().__init__(**kwargs)
        self.tool_choice = tool_choice

    def request(self):
        return {**super().request(), "tool_choice": self.tool_choice}


class RoundTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Record each backoff, by the failure and the try it follows, and wait none of it."""
        self.backoffs = []

        def backoff(exc, attempt):
            self.backoffs.append((type(exc), attempt))
            return 0.0

        self.enterContext(mock.patch.object(retry, "retry_delay", backoff))

    async def run_task(self, *rounds, tools=(), settings=QUICK):
        # The provider times its rounds by the settings' limits, as in a core.
        provider = ScriptedProvider(*rounds, limits=settings.limits)
        task, events, error = FakeTaskContext(), [], None
        try:
            async for event in adapter(provider, settings=settings, tools=tools).start(task).events():
                task.see(event)
                events.append(event)
        except Exception as exc:
            error = exc
        return provider, events, error

    async def test_a_phase_is_kept_and_commentary_answers_with_its_last(self):
        for second in ("final_answer", "commentary"):
            first = message_events("First.", item_id="m1", phase="commentary")
            round_ = a_round(first, message_events("Second.", item_id="m2", index=1, phase=second))
            task = FakeTaskContext()
            events = await drive(adapter(ScriptedProvider(round_)).start(task), task)
            self.assertEqual(said(events), [("commentary", "First."), ("final_answer", "Second.")])

    async def test_the_capped_round_offers_no_tool(self):
        capped = LlmSettings(max_tool_rounds=2)
        first, second = call_round("lookup", {}, "call_1"), call_round("lookup", {}, "call_2")
        provider, task = ScriptedProvider(first, second, text_round("Here is what I have.")), FakeTaskContext()
        # get_menu is also the app's: it is offered once.
        clashing = RecordingTools(["get_menu", "lookup"])
        events = await drive(adapter(provider, tools=[clashing], settings=capped).start(task), task)
        self.assertEqual(said(events)[-1], ("final_answer", "Here is what I have."))
        self.assertEqual([payload["tool_choice"] for payload in provider.payloads], ["auto", "auto", "none"])
        offered = [[tool["name"] for tool in payload["tools"]] for payload in provider.payloads]
        self.assertEqual(offered, [["get_menu", "lookup"], ["get_menu", "lookup"], []])
        self.assertEqual(provider.payloads[2]["instructions"], f"COMPOSED-INSTRUCTIONS\n\n{task.texts.round_cap_note}")
        self.assertEqual(outputs_sent(provider.payloads[2]), ["call_1", "call_2"])
        provider = ScriptedProvider(first, second, call_round("lookup", {}, "call_3"))
        shop, task, events = RecordingTools(), FakeTaskContext(), []
        with self.assertRaisesRegex(RuntimeError, "past the round limit"):
            async for event in adapter(provider, tools=[shop], settings=capped).start(task).events():
                task.see(event)
                events.append(event)
        self.assertEqual((events[-1]["type"], len(shop.calls)), ("response.failed", 2))

    async def test_a_round_cut_short_is_never_retried(self):
        cut = a_round(message_events("Half an ans"))
        incomplete = {"id": "resp_1", "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
        cut[-1] = {"type": "response.incomplete", "response": incomplete}
        provider, events, error = await self.run_task(cut, text_round("Done."))
        self.assertRegex(str(error), "cut short: max_output_tokens")
        self.assertEqual((provider.opened, kinds(events)), (1, ["response.created", "response.incomplete"]))
        self.assertEqual(events[-1]["response"]["incomplete_details"], {"reason": "max_output_tokens"})
        self.assertEqual(events[-1]["response"]["output"], [])

    async def test_what_cannot_clear_up_or_fails_again_is_passed_on(self):
        refused = BadRequestError("bad request", response=httpx.Response(400, request=REQUEST), body=None)
        too_many = RateLimitError("slow down", response=httpx.Response(429, request=REQUEST), body=None)
        cases = {
            "a refused request": ([refused], text_round("Done."), BadRequestError, 1),
            "an unexpected error": ([ValueError("no such field")], text_round("Done."), ValueError, 1),
            "a rate limit twice": ([too_many], [too_many], RateLimitError, 2),
            "an empty round twice": (a_round(), a_round(), RoundFailedError, 2),
            "a stall twice": (stalled(), stalled(), TimeoutError, 2),
        }
        for name, (first, second, error_type, opened) in cases.items():
            with self.subTest(name):
                provider, _, error = await self.run_task(first, second, text_round("Never."))
                self.assertEqual((type(error), provider.opened), (error_type, opened))

    async def test_every_failure_that_can_clear_up_shares_one_count(self):
        settings = LlmSettings(provider_retries=2, first_event_timeout_seconds=0.2, idle_timeout_seconds=0.2)
        for name, (failing, error_type) in {**OPENING_FAILURES, **ROUND_FAILURES}.items():
            opened = ["response.created", "response.failed"] if failing and isinstance(failing[0], dict) else []
            for fails in (2, 3):
                with self.subTest(name, fails=fails):
                    self.backoffs.clear()
                    rounds, cafe = [*[failing] * fails, text_round("Done.", "resp_2")], CafeTools()
                    provider, events, error = await self.run_task(*rounds, settings=settings, tools=[cafe])
                    self.assertEqual((provider.opened, self.backoffs), (3, [(error_type, 1), (error_type, 2)]))
                    if fails == 2:
                        self.assertIsNone(error)
                        self.assertEqual(kinds(events)[: 2 * len(opened)], opened * 2)
                        self.assertEqual((said(events), check_events(events)), ([("final_answer", "Done.")], []))
                        # A failed round's calls never run.
                        self.assertEqual(cafe.snapshot()["item_count"], 0)
                    else:
                        self.assertEqual((type(error), kinds(events)), (error_type, opened * 3))
        with self.subTest("failures of different kinds"):
            self.backoffs.clear()
            dropped, empty = OPENING_FAILURES["a dropped connection"][0], ROUND_FAILURES["an empty round"][0]
            provider, _, error = await self.run_task(dropped, empty, dropped, text_round("Never."), settings=settings)
            self.assertEqual((type(error), provider.opened), (APIConnectionError, 3))
            self.assertEqual(self.backoffs, [(APIConnectionError, 1), (RoundFailedError, 2)])

    async def test_a_mixed_round_runs_own_calls_and_ends_for_the_apps(self):
        provider = ScriptedProvider(calls_round("get_menu", "lookup", "book", said="Checking."), text_round("Done."))
        shop, task = RecordingTools(["lookup", "book"]), FakeTaskContext()
        # Refused, as after a stop was requested.
        task.refused.add("call_book")
        run = adapter(provider, tools=[shop]).start(task)
        first = await drive(run, task)
        self.assertEqual(check_events(first), [])
        self.assertEqual([call["name"] for call in items_of(first, "function_call")], ["get_menu", "lookup", "book"])
        self.assertEqual(said(first), [("commentary", "Checking.")])
        self.assertEqual([name for _, name, _ in shop.calls], ["lookup"])
        self.assertEqual(task.order, ["started:call_lookup", "outcome:call_lookup"])
        self.assertEqual(task.started, [("call_lookup", "shop")])
        task.give_outputs([{"type": "function_call_output", "call_id": "call_get_menu", "output": "{}"}])
        await run.resume()
        self.assertEqual(said(await drive(run, task)), [("final_answer", "Done.")])
        self.assertEqual(outputs_sent(provider.payloads[1]), ["call_lookup", "call_get_menu"])

    async def test_each_own_call_ends_with_its_outcome(self):
        provider = ScriptedProvider(calls_round("lookup", "broken", "slow", "LookupTool"), text_round("Done."))
        tools = [RecordingTools(["lookup", "broken"], fail=["broken"]), RecordingTools(["slow"], delay=1.0)]
        task, settings = FakeTaskContext(), LlmSettings(server_tool_timeout_seconds=0.05)
        events = await drive(adapter(provider, tools=tools, settings=settings).start(task), task)
        self.assertEqual(events[-1]["type"], "response.completed")
        outcomes = {call_id: outcome for call_id, (_, outcome) in task.outcomes.items()}
        expected = {"call_lookup": "done", "call_broken": "failed", "call_slow": "unclear", "call_LookupTool": "failed"}
        self.assertEqual(outcomes, expected)
        failed = json.loads(task.outcomes["call_broken"][0])["error"]
        self.assertEqual(failed, "broken failed: RuntimeError: broken broke")
        self.assertIn("did not answer within 0.05 s", task.outcomes["call_slow"][0])
        # No name is guessed: the model reads the offered names next round.
        unknown = task.texts.unknown_tool_output.format(name="LookupTool", offered="get_menu, lookup, broken, slow")
        self.assertEqual(task.outcomes["call_LookupTool"][0], json.dumps({"error": unknown}))
        self.assertEqual(task.started[-1], ("call_LookupTool", "llm"))
        self.assertEqual(outputs_sent(provider.payloads[1]), list(expected))

    async def test_an_allowed_tools_choice_limits_what_is_offered_and_run(self):
        listed = [
            {"type": "function", "name": "get_menu"},
            {"type": "web_search"},
            {"type": "function", "name": "lookup"},
        ]
        decaf = {"type": "function", "name": "set_decaf", "parameters": {"type": "object", "properties": {}}}
        offered = ["get_menu", "web_search", "lookup", "show", "cancel_work"]
        for mode, sent in (("required", "required"), (None, "auto")):
            with self.subTest(mode=mode):
                provider = ScriptedProvider(calls_round("book"), text_round("Done."))
                choice = {"type": "allowed_tools", "mode": mode, "tools": listed}
                work_tools = RecordingTools(["show", "cancel_work"])
                shop, task = RecordingTools(["lookup", "book"]), ChoiceTask(choice, tools=work_tools)
                task.declared += [decaf, {"type": "web_search"}, {"type": "file_search"}]
                await drive(adapter(provider, tools=[shop]).start(task), task)
                for payload in provider.payloads:
                    self.assertEqual([tool.get("name") or tool["type"] for tool in payload["tools"]], offered)
                    self.assertEqual(payload["tool_choice"], sent)
                names = "get_menu, lookup, show, cancel_work"
                unknown = task.texts.unknown_tool_output.format(name="book", offered=names)
                self.assertEqual(task.outcomes["call_book"], (json.dumps({"error": unknown}), "failed"))
                self.assertEqual((shop.calls, task.started), ([], [("call_book", "llm")]))

    async def test_a_streaming_round_restarts_with_the_steering(self):
        gate, dropped = asyncio.Event(), APIConnectionError(request=REQUEST)
        provider = ScriptedProvider(gated(gate, text_round("One.")), [dropped], text_round("Two.", "resp_2"))
        decaf = {"type": "function", "name": "set_decaf", "parameters": {"type": "object", "properties": {}}}
        task = TypedRunTask({"instructions": "NEW instructions.", "tools": [decaf]}, instructions="OLD instructions.")
        run = adapter(provider, settings=LlmSettings(provider_retries=1)).start(task)
        events = run.events()
        await read(events, task, until="response.created")
        task.steer_with(STEERING)
        run.steer()
        gate.set()
        rest = await read(events, task)
        self.assertEqual(kinds(rest)[:2], ["response.incomplete", "response.created"])
        self.assertEqual(rest[0]["response"]["incomplete_details"], {"reason": "steered"})
        self.assertEqual(check_events([{"type": "response.created", "response": {}}, *rest]), [])
        self.assertEqual((said(rest), task.settled), ([("final_answer", "Two.")], [("steer_1", "applied")]))
        self.assertEqual([count(payload, STEERING) for payload in provider.payloads], [0, 1, 1])
        landing = provider.payloads[2]
        self.assertEqual((landing["instructions"], landing["tools"][-1]["name"]), ("NEW instructions.", "set_decaf"))
        with self.subTest("failures on both sides of a restart share one count"):
            gate, task = asyncio.Event(), FakeTaskContext()
            rounds = ([dropped], gated(gate, text_round("x")), [dropped], [dropped], text_round("No."))
            provider = ScriptedProvider(*rounds)
            run = adapter(provider, settings=LlmSettings(provider_retries=2)).start(task)
            events = run.events()
            await read(events, task, until="response.created")
            task.steer_with(STEERING)
            run.steer()
            with self.assertRaises(APIConnectionError):
                await read(events, task)
            self.assertEqual(provider.opened, 4)

    async def test_a_second_steer_restarts_too_and_a_crossed_one_goes_back(self):
        oat = [{"type": "message", "role": "user", "content": "With oat milk."}]
        first_gate, second_gate = asyncio.Event(), asyncio.Event()
        rounds = (gated(first_gate, text_round("One.")), gated(second_gate, text_round("Two.", "resp_2")))
        provider, task = ScriptedProvider(*rounds, text_round("Two with oat.", "resp_3")), FakeTaskContext()
        run = adapter(provider).start(task)
        events = run.events()
        await read(events, task, until="response.created")
        task.steer_with(STEERING, "steer_1")
        run.steer()
        first_gate.set()
        restarted = await read(events, task, until="response.created")
        self.assertEqual(kinds(restarted), ["response.incomplete", "response.created"])
        task.steer_with(oat, "steer_2")
        run.steer()
        second_gate.set()
        restarted = await read(events, task, until="response.created")
        self.assertEqual(kinds(restarted), ["response.incomplete", "response.created"])
        await read(events, task, until="response.output_item.added")
        task.steer_with([{"type": "message", "role": "user", "content": "And a muffin."}], "steer_3")
        run.steer()
        self.assertEqual(said(await read(events, task)), [("final_answer", "Two with oat.")])
        settled = [("steer_1", "applied"), ("steer_2", "applied")]
        self.assertEqual((task.settled, provider.opened), (settled, 3))
        sent = [(count(payload, STEERING), count(payload, oat)) for payload in provider.payloads]
        self.assertEqual(sent, [(0, 0), (1, 0), (1, 1)])

    async def test_steering_after_a_call_lands_at_the_next_round(self):
        provider = ScriptedProvider(call_round("lookup", {}), text_round("Done.", "resp_2"))
        shop, task = RecordingTools(delay=0.1), FakeTaskContext()
        run = adapter(provider, tools=[shop]).start(task)
        events = run.events()
        await read(events, task, until="response.completed")
        task.steer_with(STEERING)
        run.steer()
        rest = await read(events, task)
        self.assertEqual(task.outcomes["call_1"][1], "done")
        self.assertEqual((task.settled, count(provider.payloads[1], STEERING)), ([("steer_1", "applied")], 1))
        self.assertNotIn("response.incomplete", kinds(rest))


class ThroughTheImTests(IMCase):
    async def test_the_im_refuses_an_app_function_the_list_leaves_out(self):
        decaf = {"type": "function", "name": "set_decaf", "parameters": {"type": "object", "properties": {}}}
        listed = [{"type": "function", "name": name} for name in ("get_menu", "lookup")]
        settings = {"tools": [MENU, decaf], "tool_choice": {"type": "allowed_tools", "mode": "auto", "tools": listed}}
        provider = ScriptedProvider(calls_round("set_decaf", "lookup", "get_menu"), text_round("Only the menu."))
        shop = RecordingTools(["lookup"])
        im = self.session(adapter(provider, tools=[shop]))
        first = await read_all(im.admit(ask("Make it decaf."), delegation_id="D1", settings=settings).stream)
        offered = [tool["name"] for tool in provider.payloads[0]["tools"]]
        self.assertEqual(offered, ["get_menu", "lookup", "cancel_work"])
        self.assertEqual((function_calls(first), [name for _, name, _ in shop.calls]), (["call_get_menu"], ["lookup"]))
        rest = await self.answer_calls(im, first, settings=settings)
        refused = {"type": "function_call_output", "call_id": "call_set_decaf", "output": TEXTS.unlisted_output}
        self.assertIn(refused, provider.payloads[1]["input"])
        self.assertEqual((answers(rest), rest[-1]["type"]), (["Only the menu."], "response.completed"))

    async def test_a_list_the_outputs_bring_sets_what_is_offered_and_run(self):
        decaf = {"type": "function", "name": "set_decaf", "parameters": {"type": "object", "properties": {}}}
        menu_only, decaf_only = (
            {"tools": [MENU, decaf], "tool_choice": {"type": "allowed_tools", "mode": "auto", "tools": [listed]}}
            for listed in ({"type": "function", "name": "get_menu"}, {"type": "function", "name": "set_decaf"})
        )
        rounds = (call_round("get_menu", {}, "call_menu"), calls_round("get_menu", "set_decaf"), text_round("Decaf."))
        provider = ScriptedProvider(*rounds)
        im = self.session(adapter(provider))
        first = await read_all(im.admit(ask("The menu, then decaf."), delegation_id="D1", settings=menu_only).stream)
        rest = await self.answer_calls(im, first, settings=decaf_only)
        self.assertEqual([tool["name"] for tool in provider.payloads[1]["tools"]], ["set_decaf", "cancel_work"])
        self.assertEqual(function_calls(rest), ["call_set_decaf"])
        last = await self.answer_calls(im, rest, settings=decaf_only)
        refused = {"type": "function_call_output", "call_id": "call_get_menu", "output": TEXTS.unlisted_output}
        self.assertIn(refused, provider.payloads[2]["input"])
        self.assertEqual(answers(last), ["Decaf."])

    async def test_each_tool_choice_shape_sets_what_is_offered_and_run(self):
        menu = {"type": "function", "name": "get_menu"}
        shapes = [(None, []), ("get_menu", []), (menu, []), ([], []), ([None, menu], ["get_menu"])]
        choices = [({"type": "allowed_tools", "tools": tools}, listed) for tools, listed in shapes]
        for choice, listed in [*choices, (menu, ["get_menu", "lookup"])]:
            with self.subTest(choice=choice):
                provider = ScriptedProvider(call_round("get_menu", {}), text_round("No menu."))
                im = self.session(adapter(provider, tools=[RecordingTools()]))
                settings = {"tools": [MENU], "tool_choice": choice}
                first = await read_all(im.admit(ask("The menu."), delegation_id="D1", settings=settings).stream)
                self.assertEqual(function_calls(first), ["call_1"] if "get_menu" in listed else [])
                self.assertEqual([tool["name"] for tool in provider.payloads[0]["tools"]], [*listed, "cancel_work"])


class StopTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_stop_between_rounds_is_confirmed_at_once(self):
        for name, round_ in (
            ("before", None),
            ("waiting", call_round("get_menu", {})),
            ("answered", text_round("Done.")),
        ):
            with self.subTest(name):
                provider, task = ScriptedProvider(round_ or text_round("Never.")), FakeTaskContext()
                run = adapter(provider).start(task)
                if round_ is not None:
                    await drive(run, task)
                self.assertEqual(await run.stop(), "stopped")
                await run.resume()
                self.assertEqual((await drive(run, task), provider.opened), ([], int(round_ is not None)))

    async def test_a_stop_lets_started_calls_finish(self):
        provider = ScriptedProvider(call_round("lookup", {}), text_round("Done.", "resp_2"))
        shop, task = RecordingTools(delay=0.2), FakeTaskContext()
        run = adapter(provider, tools=[shop]).start(task)
        reading = asyncio.create_task(drive(run, task))
        while not shop.calls:
            await asyncio.sleep(0.01)
        self.assertEqual(await run.stop(), "stopped")
        self.assertEqual(task.outcomes["call_1"][1], "done")
        self.assertEqual((kinds(await reading)[-1], provider.opened), ("response.completed", 1))
        provider, shop, task = ScriptedProvider(call_round("lookup", {})), RecordingTools(), FakeTaskContext()
        run = adapter(provider, tools=[shop]).start(task)
        events = run.events()
        await read(events, task, until="response.completed")
        self.assertEqual(await run.stop(), "stopped")
        self.assertEqual(([event async for event in events], shop.calls, task.started), ([], [], []))

    async def test_a_stop_before_a_retry_opens_nothing(self):
        with self.subTest("at the failed round's end"):
            provider, task = ScriptedProvider(failed_round(), text_round("Done.")), FakeTaskContext()
            run = adapter(provider, settings=FAST.model_copy(update={"provider_retries": 1})).start(task)
            seen = []
            async for event in run.events():
                seen.append(event["type"])
                if event["type"] == "response.failed":
                    self.assertEqual(await run.stop(), "stopped")
            self.assertEqual((seen, provider.opened), (["response.created", "response.failed"], 1))
            self.assertNotIn("backend.recovered", [kind for kind, _ in task.traces])
        with self.subTest("in the wait before it"):
            limited = RateLimitError("slow", response=httpx.Response(429, request=REQUEST), body=None)
            settings = LlmSettings(provider_retries=1)
            provider, task = ScriptedProvider([limited], text_round("Done."), limits=settings.limits), FakeTaskContext()
            run = adapter(provider, settings=settings).start(task)
            with mock.patch.object(retry, "retry_delay", return_value=0.3):
                reader = asyncio.ensure_future(anext(run.events(), None))
                await asyncio.sleep(0.1)
                self.assertEqual(await run.stop(), "stopped")
                self.assertIsNone(await reader)
            recovered = [fields["error_type"] for kind, fields in task.traces if kind == "backend.recovered"]
            self.assertEqual((provider.opened, recovered), (1, ["RateLimitError"]))

    async def test_a_round_that_holds_on_leaves_the_stop_unconfirmed(self):
        hung = text_round("Old.")
        hung.insert(1, Hang(1.0))
        for steered in (False, True):
            with self.subTest(steered=steered):
                gate, task = asyncio.Event(), FakeTaskContext()
                provider = ScriptedProvider(list(hung), gated(gate, text_round("New.")))
                run = adapter(provider, settings=LlmSettings(stop_timeout_seconds=0.1)).start(task)
                events = run.events()
                await anext(events)
                if steered:
                    task.steer_with(STEERING)
                    run.steer()
                    kinds_now = [(await anext(events))["type"] for _ in range(2)]
                    self.assertEqual(kinds_now, ["response.incomplete", "response.created"])
                started = time.monotonic()
                self.assertEqual(await run.stop(), "unconfirmed")
                self.assertLess(time.monotonic() - started, 0.5)
                gate.set()
                self.assertEqual([event async for event in events], [])


def a_work(work_id: str, state: str, seconds: float) -> WorkSummary:
    task = TaskSummary(f"{work_id}.1", state, None, (), False, None, seconds)
    return WorkSummary(work_id, state, f"d_{work_id}", (a_request(f"d_{work_id}", "Earlier."),), (task,), seconds)


class SourceTests(unittest.IsolatedAsyncioTestCase):
    def test_it_describes_what_it_serves_without_credentials(self):
        block = {"max_tool_rounds": 4, "provider_retries": 1, "idle_timeout_seconds": 10.5}
        described = build(block, tools=lambda: [TOOL_SOURCES["cafe"].from_settings({})]).describe()
        expected = {"adapter": "llm", "model": "test/model", "api": "chat-completions", "max_running": 1}
        expected |= {"max_tool_rounds": 4}
        self.assertEqual({key: described[key] for key in expected}, expected)
        self.assertEqual([tool["name"] for tool in described["tools"]], ["get_menu", "manage_cart", "place_order"])
        self.assertNotIn("secret-key", json.dumps(described))
        with self.assertRaisesRegex(ValueError, "takes no settings"):
            TOOL_SOURCES["cafe"].from_settings({"url": "http://example"})
        described["tools"][0]["name"] = "changed"
        self.assertEqual(CafeTools().tools()[0]["name"], "get_menu")
        self.assertIsInstance(build({}).new().conductor(), SteerOpenWork)
        self.assertEqual(
            LlmSource(FAST, endpoints=lambda model: OTHER, tools=lambda: []).describe()["api"], "openai-responses"
        )

    def test_the_settings_its_provider_cannot_honor_are_named(self):
        settings = {"tools": [{"type": "web_search"}], "tool_choice": "auto", "service_tier": "priority"}
        self.assertEqual([param for param, _ in build({}).unsupported(settings)], ["tools", "service_tier"])
        # A model whose entry names the Responses API honors them: its server runs the hosted tool.
        self.assertEqual(build({}).unsupported({**settings, "model": OTHER.model}), [])
        hosted = [{"type": "web_search"}, {"type": "web_search"}, {"type": "file_search"}]
        problems = build({}).unsupported({"tools": hosted, "service_tier": "flex", "text": {"verbosity": "low"}})
        self.assertEqual([param for param, _ in problems], ["tools", "tools", "service_tier", "text.verbosity"])
        self.assertIn("web_search", problems[0][1])
        # On either API, each type of tool whose calls the client runs or approves; a server runs the others.
        client = [
            {"type": "custom", "name": "run_sql"},
            {"type": "shell"},
            {"type": "shell", "environment": {"type": "local"}},
            {"type": "tool_search", "execution": "client"},
            {"type": "mcp", "server_label": "docs"},
            *({"type": kind} for kind in ("computer", "computer_use_preview", "local_shell", "apply_patch")),
            *({"type": kind} for kind in ("namespace", "programmatic_tool_calling")),
        ]
        served = [
            {"type": "shell", "environment": {"type": "container_auto"}},
            {"type": "tool_search"},
            {"type": "mcp", "server_label": "docs", "require_approval": "never"},
        ]
        kinds = [
            "custom",
            "shell",
            "tool_search",
            "mcp",
            "computer",
            "computer_use_preview",
            "local_shell",
            "apply_patch",
            "namespace",
            "programmatic_tool_calling",
        ]
        refused = build({}).unsupported({"tools": client + served, "model": OTHER.model})
        self.assertEqual([message.split()[1] for _, message in refused], kinds)
        self.assertTrue(all("client to run or approve" in message for _, message in refused))
        self.assertEqual(build({}).unsupported({"tools": ["custom"], "model": OTHER.model}), [])
        on_chat = build({}).unsupported({"tools": client + served})
        self.assertEqual(on_chat[: len(kinds)], refused)
        self.assertEqual([message.split()[1] for _, message in on_chat[len(kinds) :]], ["shell", "tool_search", "mcp"])
        self.assertTrue(all("needs a Responses backend" in message for _, message in on_chat[len(kinds) :]))
        self.assertEqual(build({}).unsupported({"service_tier": "auto"}), [])
        reasoning = {"reasoning": {"effort": "high", "summary": "auto"}}
        # The endpoint's own settings set the model's reasoning.
        self.assertEqual(build({}).unsupported(reasoning), [])

    async def test_each_model_runs_on_its_own_endpoint_and_api(self):
        rounds = {
            DEFAULT.model: [text_round("Done.")],
            OTHER.model: [call_round("get_menu", {}), text_round("Done.", "resp_2"), text_round("Done.", "resp_3")],
        }
        made = []

        def make(endpoint):
            made.append(ScriptedProvider(*rounds[endpoint.model], endpoint=endpoint))
            return made[-1]

        session = build({}, make_provider=make).new()
        for model in (None, "other/model", "other/model"):
            task = FakeTaskContext(model=model)
            run = session.start(task)
            await drive(run, task)
            if task.calls_seen:
                task.give_outputs([{"type": "function_call_output", "call_id": "call_1", "output": "{}"}])
                await run.resume()
                await drive(run, task)
        self.assertEqual([(provider.endpoint, len(provider.payloads)) for provider in made], [(DEFAULT, 1), (OTHER, 3)])
        self.assertEqual({payload["model"] for payload in made[1].payloads}, {"other/model"})
        session = build({}).new()
        self.addAsyncCleanup(session.close)
        self.assertEqual(
            [type(session.provider(model)) for model in (None, OTHER.model)],
            [ChatCompletionsBackend, OpenAIResponsesBackend],
        )
        unknown = ModelEndpoint("test/model", "http://default:1/v1", api="grpc")
        with self.assertRaisesRegex(ValueError, "Unknown backend API 'grpc'"):
            LlmSource(FAST, endpoints=lambda model: unknown, tools=lambda: [])

    async def test_a_model_the_catalog_does_not_list_is_refused(self):
        made = []

        def make(endpoint):
            made.append(ScriptedProvider(text_round("Done."), endpoint=endpoint))
            return made[-1]

        source_ = build({}, make_provider=make)
        await source_.check_model("other/model")
        with self.assertRaisesRegex(ValueError, "not in the model catalog"):
            await source_.check_model("nvidia/unlisted")
        task = FakeTaskContext(model="nvidia/unlisted")
        with self.assertRaisesRegex(ValueError, "not in the model catalog"):
            await drive(source_.new().start(task), task)
        self.assertEqual(made, [])

    async def test_each_session_gets_fresh_tool_sources(self):
        providers = [
            ScriptedProvider(call_round("manage_cart", {"action": "add", "item_id": "lat"}), text_round("Added."))
            for _ in range(2)
        ]
        made = list(providers)
        built = LlmSource(FAST, endpoints=lookup, tools=lambda: [CafeTools()], make_provider=lambda _: made.pop(0))
        for provider in providers:
            session, task = built.new(), FakeTaskContext()
            run = session.start(task)
            self.assertEqual((provider.payloads, run.round), ([], 0))
            await drive(run, task)
            self.assertEqual(json.loads(task.outcomes["call_1"][0])["cart"]["item_count"], 1)
            await session.close()
            self.assertEqual(provider.closed, 1)

    async def test_a_one_shot_offers_no_tool_and_retries_only_its_opening(self):
        provider = ScriptedProvider(a_round(), text_round("Done."))
        task = FakeTaskContext()
        with self.assertRaisesRegex(RoundFailedError, "answered nothing"):
            await drive(source(provider, tools=[RecordingTools()]).new_one_shot().start(task), task)
        self.assertEqual([tool["name"] for tool in provider.payloads[0]["tools"]], ["get_menu"])
        settings = LlmSettings(provider_retries=2, first_event_timeout_seconds=0.2, idle_timeout_seconds=0.2)
        for failures, opened in ((OPENING_FAILURES, 3), (ROUND_FAILURES, 1)):
            for name, (failing, error_type) in failures.items():
                with self.subTest(name), mock.patch.object(retry, "retry_delay", return_value=0.0):
                    provider = ScriptedProvider(*[failing] * 3, text_round("Never."), limits=settings.limits)
                    task = FakeTaskContext()
                    with self.assertRaises(error_type):
                        await drive(source(provider, settings=settings).new_one_shot().start(task), task)
                    self.assertEqual(provider.opened, opened)

    async def test_steer_joins_the_newest_open_work_or_starts_one(self):
        running, queued = a_work("W1", "running", 9.0), a_work("W2", "queued", 0.0)
        waiting = a_work("W3", "waiting_client", 4.0)
        just_ended = a_work("W5", "answered", 0.5)
        cases = [
            ((an_ended_work("W0"), running, queued, an_ended_work("W4")), Steer("W2", ("d_1", "d_2"))),
            ((a_work("W1", "queued", 0.0), queued), Steer("W2", ("d_1", "d_2"))),
            ((waiting,), Steer("W3", ("d_1", "d_2"))),
            ((running, just_ended), Steer("W1", ("d_1", "d_2"))),
            ((just_ended, an_ended_work("W0")), Start(("d_1", "d_2"))),
        ]
        for works, decision in cases:
            with self.subTest(decision=decision, works=[work.work_id for work in works]):
                arrivals = (a_request("d_1", "Also this."), a_request("d_2", "Also this."))
                self.assertEqual(await decide(SteerOpenWork(), ConductorInput(arrivals, works, (), 0)), [decision])
        self.assertEqual(await decide(SteerOpenWork(), ConductorInput((), (running,), (), 0)), [])


class ClosedToolsTests(unittest.IsolatedAsyncioTestCase):
    """Port rule E9: a task whose tools the IM closed answers next, with no tool offered and no round-cap note."""

    async def test_a_closed_task_gets_a_round_with_no_tool_and_answers(self):
        provider, task = ScriptedProvider(text_round("I can save the note. Shall I?")), FakeTaskContext()
        task.tools_closed = True
        events = await drive(adapter(provider, tools=[RecordingTools()]).start(task), task)
        payload = provider.payloads[0]
        self.assertEqual((payload["tools"], payload["tool_choice"]), ([], "none"))
        self.assertNotIn(task.texts.round_cap_note, payload["instructions"])
        self.assertEqual(said(events)[-1], ("final_answer", "I can save the note. Shall I?"))

    async def test_a_call_written_in_a_closed_round_does_not_fail_the_response(self):
        provider = ScriptedProvider(call_round("get_menu", {}, "call_1"), text_round("Shall I save it?"))
        task = FakeTaskContext()
        task.tools_closed = True
        task.refused.add("call_1")  # the IM refuses a declared write of a closed task at once
        events = await drive(adapter(provider).start(task), task)
        self.assertEqual([payload["tool_choice"] for payload in provider.payloads], ["none", "none"])
        self.assertEqual(said(events)[-1], ("final_answer", "Shall I save it?"))

    async def test_a_round_sends_the_apps_tools_without_their_annotations(self):
        annotated = {**MENU, "annotations": {"readOnlyHint": True}}
        provider, task = ScriptedProvider(text_round("Done.")), FakeTaskContext(declared=[annotated])
        await drive(adapter(provider).start(task), task)
        sent = [tool for tool in provider.payloads[0]["tools"] if tool["name"] == "get_menu"]
        self.assertEqual(sent, [MENU])
        self.assertIn("annotations", task.request()["tools"][0])  # the task's request keeps them

    async def test_closed_tools_win_over_an_allowed_tools_choice(self):
        choice = {"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "get_menu"}]}
        provider, task = ScriptedProvider(text_round("Shall I save it?")), FakeTaskContext()
        task.tools_closed = True
        task.request = lambda: {**FakeTaskContext().request(), "tool_choice": choice}
        await drive(adapter(provider).start(task), task)
        self.assertEqual((provider.payloads[0]["tools"], provider.payloads[0]["tool_choice"]), ([], "none"))
