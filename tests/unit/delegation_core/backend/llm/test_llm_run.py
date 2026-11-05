# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Tests for a task's run on the backend LLM: its rounds, retries, calls, round cap, steering and stops."""

import asyncio
import json
import time
import unittest

import httpx
from openai import APIConnectionError, BadRequestError, RateLimitError

from examples.frontend_backend_live.delegation_core.backend.llm import LlmSettings
from examples.frontend_backend_live.delegation_core.backend.llm.run import RoundFailedError
from examples.frontend_backend_live.delegation_core.backend.tools import CafeTools
from tests.unit.contracts.backend import (
    BackendAdapterContract,
    ScriptedBackend,
    a_round,
    call_events,
    call_round,
    check_events,
    message_events,
    text_round,
)
from tests.unit.delegation_core.backend.llm.doubles import (
    Hang,
    IMTask,
    RecordingTools,
    ScriptedProvider,
    Sleep,
    WorkTools,
    adapter,
    contract_rounds,
    contract_tools,
    drive,
    gated,
    items_of,
    kinds,
)

REQUEST = httpx.Request("POST", "http://localhost:1/v1/chat/completions")
MENU_OUTPUT = [{"type": "function_call_output", "call_id": "call_1", "output": "{}"}]
TWO = [{"type": "message", "role": "user", "content": "Make it two."}]
OAT = [{"type": "message", "role": "user", "content": "With oat milk."}]


def calls_round(*names, said=None):
    parts = [message_events(said, index=0)] if said else []
    for name in names:
        parts.append(call_events(name, {}, call_id=f"call_{name}", index=len(parts)))
    return a_round(*parts, response_id="resp_calls")


def outputs_sent(payload):
    return [item["call_id"] for item in payload["input"] if item.get("type") == "function_call_output"]


def count(payload, items):
    return sum(item in payload["input"] for item in items)


def texts(events):
    return [(message["content"][0]["text"], message["phase"]) for message in items_of(events, "message")]


async def read(events, task, until=None):
    """Read ``events`` as the IM does, up to and including the first event of type ``until`` (or to the end)."""
    seen = []
    async for event in events:
        task.see(event)
        seen.append(event)
        if event["type"] == until:
            break
    return seen


class LlmAdapterContractTests(BackendAdapterContract, unittest.IsolatedAsyncioTestCase):
    """The backend adapter contract on the backend LLM."""

    def make_adapter(self, scenario):
        self.gate = asyncio.Event()
        self.backend = ScriptedBackend(contract_rounds(scenario, self.gate))
        return adapter(self.backend, tools=contract_tools(scenario))

    def release(self):
        self.gate.set()

    def sent_instructions(self):
        return [payload.get("instructions") for payload in self.backend.payloads]


class RoundTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_phase_the_model_gave_is_kept_and_a_round_of_commentary_answers_with_its_last_message(self):
        for second in ("final_answer", "commentary"):
            first = message_events("First.", item_id="m1", phase="commentary")
            round_ = a_round(first, message_events("Second.", item_id="m2", index=1, phase=second))
            task = IMTask()
            events = await drive(adapter(ScriptedProvider(round_)).start(task), task)
            self.assertEqual(texts(events), [("First.", "commentary"), ("Second.", "final_answer")])

    async def test_an_empty_round_is_retried_once_and_fails_the_task_the_second_time(self):
        provider, task = ScriptedProvider(a_round(), text_round("Done.")), IMTask()
        events = await drive(adapter(provider).start(task), task)
        self.assertEqual(kinds(events)[:3], ["response.created", "response.failed", "response.created"])
        self.assertEqual(events[1]["response"]["error"]["message"], "The backend's round answered nothing")
        self.assertEqual((check_events(events), texts(events)), ([], [("Done.", "final_answer")]))
        events = []
        with self.assertRaisesRegex(RoundFailedError, "answered nothing"):
            async for event in adapter(ScriptedProvider(a_round(), a_round())).start(IMTask()).events():
                events.append(event)
        self.assertEqual(kinds(events), ["response.created", "response.failed"] * 2)

    async def test_the_capped_round_offers_no_tool_and_calls_past_it_fail_the_task(self):
        capped = LlmSettings(max_tool_rounds=2)
        first, second = call_round("lookup", {}, "call_1"), call_round("lookup", {}, "call_2")
        provider, task = ScriptedProvider(first, second, text_round("Here is what I have.")), IMTask()
        events = await drive(adapter(provider, tools=[RecordingTools()], settings=capped).start(task), task)
        self.assertEqual(texts(events)[-1], ("Here is what I have.", "final_answer"))
        self.assertEqual([payload["tool_choice"] for payload in provider.payloads], ["auto", "auto", "none"])
        offered = [[tool["name"] for tool in payload["tools"]] for payload in provider.payloads]
        self.assertEqual(offered, [["get_menu", "lookup"], ["get_menu", "lookup"], []])  # none to call or write out
        self.assertEqual(provider.payloads[2]["instructions"], f"COMPOSED-INSTRUCTIONS\n\n{task.texts.round_cap_note}")
        self.assertEqual(outputs_sent(provider.payloads[2]), ["call_1", "call_2"])  # the last output is kept
        provider = ScriptedProvider(first, second, call_round("lookup", {}, "call_3"))
        shop, task, events = RecordingTools(), IMTask(), []
        with self.assertRaisesRegex(RuntimeError, "past the round limit"):
            async for event in adapter(provider, tools=[shop], settings=capped).start(task).events():
                task.see(event)
                events.append(event)
        self.assertEqual((events[-1]["type"], len(shop.calls)), ("response.failed", 2))


def failed_round():
    error = {"code": "server_error", "message": "The model fell over."}
    return [
        {"type": "response.created", "response": {"id": "resp_f", "status": "in_progress"}},
        {"type": "response.failed", "response": {"id": "resp_f", "status": "failed", "error": error}},
    ]


def stalled(text="Done."):
    round_ = text_round(text)
    return [round_[0], Sleep(1.0), *round_[1:]]


class RetryTests(unittest.IsolatedAsyncioTestCase):
    async def run_task(self, *rounds, tools=()):
        settings = LlmSettings(provider_retries=0, first_event_timeout_seconds=0.2, idle_timeout_seconds=0.2)
        provider = ScriptedProvider(*rounds, limits=settings.limits)
        task, events, error = IMTask(), [], None
        try:
            async for event in adapter(provider, settings=settings, tools=tools).start(task).events():
                task.see(event)
                events.append(event)
        except Exception as exc:
            error = exc
        return provider, events, error

    async def test_an_attempt_that_failed_broke_dropped_or_stalled_is_retried_once(self):
        cases = {
            "a failed round": failed_round(),
            "no terminal": text_round("x")[:-1],
            "a second response.created": [*text_round("x")[:2], *text_round("y")],
            "a dropped connection": [text_round("x")[0], APIConnectionError(request=REQUEST)],
            "a stall": stalled("Old."),
        }
        for name, first in cases.items():
            with self.subTest(name):
                provider, events, error = await self.run_task(first, text_round("Done.", "resp_2"))
                self.assertIsNone(error)
                self.assertEqual((provider.opened, kinds(events)[1], check_events(events)), (2, "response.failed", []))

    async def test_what_cannot_clear_up_or_fails_again_reaches_the_im_unchanged(self):
        refused = BadRequestError("bad request", response=httpx.Response(400, request=REQUEST), body=None)
        too_many = RateLimitError("slow down", response=httpx.Response(429, request=REQUEST), body=None)
        cases = {
            "a refused request": ([refused], text_round("Done."), BadRequestError, 1),
            "a rate limit twice": ([too_many], [too_many], RateLimitError, 2),  # a 429 the IM passes on as one
            "a stall twice": (stalled(), stalled(), TimeoutError, 2),
        }
        for name, (first, second, error_type, opened) in cases.items():
            with self.subTest(name):
                provider, _, error = await self.run_task(first, second, text_round("Never."))
                self.assertEqual((type(error), provider.opened), (error_type, opened))

    async def test_a_round_cut_short_is_passed_on_never_retried_and_ends_the_task(self):
        cut = a_round(message_events("Half an ans"))
        incomplete = {"id": "resp_1", "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
        cut[-1] = {"type": "response.incomplete", "response": incomplete}
        provider, events, error = await self.run_task(cut, text_round("Done."))
        self.assertRegex(str(error), "cut short: max_output_tokens")
        self.assertEqual((provider.opened, kinds(events)), (1, ["response.created", "response.incomplete"]))
        self.assertEqual(events[-1]["response"]["incomplete_details"], {"reason": "max_output_tokens"})
        self.assertEqual(events[-1]["response"]["output"], [])  # the cut message is not delivered

    async def test_a_round_dropped_after_a_call_is_retried_and_the_call_never_runs(self):
        cafe = CafeTools()
        added = a_round(call_events("manage_cart", {"action": "add", "item_id": "lat"}))
        dropped = APIConnectionError(request=REQUEST)
        provider, _, error = await self.run_task([*added[:-1], dropped], text_round("Done.", "resp_2"), tools=[cafe])
        self.assertIsNone(error)
        self.assertEqual((provider.opened, cafe.snapshot()["item_count"]), (2, 0))


class CallTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_mixed_round_runs_the_backends_calls_at_once_and_ends_for_the_apps(self):
        provider = ScriptedProvider(calls_round("get_menu", "lookup", "book", said="Checking."), text_round("Done."))
        shop, task = RecordingTools(["lookup", "book"]), IMTask()
        task.refused.add("call_book")  # e.g. after a stop was requested
        run = adapter(provider, tools=[shop]).start(task)
        first = await drive(run, task)
        self.assertEqual(check_events(first), [])
        self.assertEqual([call["name"] for call in items_of(first, "function_call")], ["get_menu", "lookup", "book"])
        self.assertEqual(texts(first), [("Checking.", "commentary")])
        self.assertEqual([name for _, name, _ in shop.calls], ["lookup"])  # neither the app's nor the refused one
        self.assertEqual(task.order, ["started:call_lookup", "outcome:call_lookup"])  # before the iteration ended
        self.assertEqual(task.started, [("call_lookup", "shop")])
        task.give_outputs([{"type": "function_call_output", "call_id": "call_get_menu", "output": "{}"}])
        await run.resume()
        self.assertEqual(texts(await drive(run, task)), [("Done.", "final_answer")])
        self.assertEqual(outputs_sent(provider.payloads[1]), ["call_lookup", "call_get_menu"])

    async def test_each_call_the_backend_runs_ends_with_its_outcome_and_the_task_goes_on(self):
        provider = ScriptedProvider(calls_round("lookup", "broken", "slow", "LookupTool"), text_round("Done."))
        tools = [RecordingTools(["lookup", "broken"], fail=["broken"]), RecordingTools(["slow"], delay=1.0)]
        task, settings = IMTask(), LlmSettings(server_tool_timeout_seconds=0.05)
        events = await drive(adapter(provider, tools=tools, settings=settings).start(task), task)
        self.assertEqual(events[-1]["type"], "response.completed")
        outcomes = {call_id: outcome for call_id, (_, outcome) in task.outcomes.items()}
        expected = {"call_lookup": "done", "call_broken": "failed", "call_slow": "unclear", "call_LookupTool": "failed"}
        self.assertEqual(outcomes, expected)
        self.assertEqual(
            json.loads(task.outcomes["call_broken"][0])["error"], "broken failed: RuntimeError: broken broke"
        )
        self.assertIn("did not answer within 0.05 s", task.outcomes["call_slow"][0])  # it may have taken effect
        # No name is guessed: the model reads the offered names next round.
        unknown = task.texts.unknown_tool_output.format(name="LookupTool", offered="get_menu, lookup, broken, slow")
        self.assertEqual(task.outcomes["call_LookupTool"][0], json.dumps({"error": unknown}))
        self.assertEqual(task.started[-1], ("call_LookupTool", "llm"))
        self.assertEqual(outputs_sent(provider.payloads[1]), list(expected))  # every output, in call order

    async def test_every_round_offers_the_apps_functions_the_sources_and_the_work_tools_each_name_once(self):
        provider = ScriptedProvider(calls_round("lookup"), text_round("Done."))
        clashing = RecordingTools(["get_menu", "lookup"])  # get_menu is also the app's: the app's wins
        task = IMTask(work_tools=WorkTools())
        cart = {"type": "function", "name": "manage_cart", "parameters": {"type": "object", "properties": {}}}
        async for event in adapter(provider, tools=[clashing]).start(task).events():
            task.see(event)
            if event["type"] == "response.completed" and len(task.requests) == 1:
                task.declared.append(cart)  # a session.update the IM took before the next round
        offered = [[tool["name"] for tool in payload["tools"]] for payload in provider.payloads]
        self.assertEqual(offered[0], ["get_menu", "lookup", "show", "cancel_work"])
        self.assertEqual(offered[1], ["get_menu", "manage_cart", "lookup", "show", "cancel_work"])
        self.assertEqual(task.started, [("call_lookup", "shop")])

    async def test_show_reaches_the_im_and_cancel_work_stops_the_work_after_its_step(self):
        show = a_round(message_events("Here is the menu."), call_events("show", {"display": "Latte $4.50"}, index=1))
        cancel = call_round("cancel_work", {}, "call_stop")
        late = call_round("lookup", {}, "call_late")
        provider = ScriptedProvider(show, cancel, late, text_round("I stopped that.", "resp_end"))
        shop, work = RecordingTools(), WorkTools()
        task = IMTask(work_tools=work)
        events = await drive(adapter(provider, tools=[shop]).start(task), task)
        self.assertEqual((work.shown, work.cancelled, shop.calls), (["Latte $4.50"], True, []))  # late was refused
        self.assertEqual(task.started, [("call_1", "delegation_core"), ("call_stop", "delegation_core")])
        self.assertEqual(json.loads(task.outcomes["call_stop"][0]), {"stopping": "after this step"})
        self.assertEqual(texts(events), [("Here is the menu.", "commentary"), ("I stopped that.", "final_answer")])


class TypedRunTask(IMTask):
    """A task whose steering is a typed run with its own settings: once it lands, they are the Work's."""

    def __init__(self, brings: dict, **kwargs):
        super().__init__(**kwargs)
        self.brings = brings

    def settle(self, steering, delivery):
        super().settle(steering, delivery)
        if delivery == "applied":
            self.instructions, self.declared = self.brings["instructions"], list(self.brings["tools"])


class SteerTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_streaming_round_is_restarted_once_with_the_steering_and_a_typed_runs_settings(self):
        # The restarted round's connection drops: its retry sends the steering once more, never twice.
        gate, dropped = asyncio.Event(), APIConnectionError(request=REQUEST)
        provider = ScriptedProvider(gated(gate, text_round("One.")), [dropped], text_round("Two.", "resp_2"))
        decaf = {"type": "function", "name": "set_decaf", "parameters": {"type": "object", "properties": {}}}
        task = TypedRunTask({"instructions": "NEW instructions.", "tools": [decaf]}, instructions="OLD instructions.")
        run = adapter(provider, settings=LlmSettings(provider_retries=0)).start(task)
        events = run.events()
        await read(events, task, until="response.created")
        task.steer_with(TWO)
        run.steer()
        gate.set()
        rest = await read(events, task)
        self.assertEqual(kinds(rest)[:2], ["response.incomplete", "response.created"])
        self.assertEqual(rest[0]["response"]["incomplete_details"], {"reason": "steered"})
        self.assertEqual(check_events([{"type": "response.created", "response": {}}, *rest]), [])
        self.assertEqual((texts(rest), task.settled), ([("Two.", "final_answer")], [("steer_1", "applied")]))
        self.assertEqual([count(payload, TWO) for payload in provider.payloads], [0, 1, 1])
        landing = provider.payloads[2]  # the round that reads the typed words is sent its settings
        self.assertEqual((landing["instructions"], landing["tools"][-1]["name"]), ("NEW instructions.", "set_decaf"))

    async def test_one_restart_per_run_and_steering_an_answer_crossed_goes_back_to_the_im(self):
        first_gate, second_gate = asyncio.Event(), asyncio.Event()
        rounds = (gated(first_gate, text_round("One.")), gated(second_gate, text_round("Two.", "resp_2")))
        provider, task = ScriptedProvider(*rounds, text_round("Two with oat.", "resp_3")), IMTask()
        run = adapter(provider).start(task)
        events = run.events()
        await read(events, task, until="response.created")
        task.steer_with(TWO, "steer_1")
        run.steer()
        first_gate.set()
        self.assertEqual(
            kinds(await read(events, task, until="response.created")), ["response.incomplete", "response.created"]
        )
        task.steer_with(OAT, "steer_2")
        run.steer()  # no second restart: this round goes on
        second_gate.set()
        self.assertEqual(texts(await read(events, task)), [("Two.", "final_answer")])
        self.assertEqual((task.settled, provider.opened), ([("steer_1", "applied")], 2))  # the IM starts steer_2 next
        self.assertEqual((count(provider.payloads[1], TWO), count(provider.payloads[1], OAT)), (1, 0))

    async def test_steering_after_a_call_lands_at_the_next_round_and_the_call_runs(self):
        provider = ScriptedProvider(call_round("lookup", {}), text_round("Done.", "resp_2"))
        shop, task = RecordingTools(delay=0.1), IMTask()
        run = adapter(provider, tools=[shop]).start(task)
        events = run.events()
        await read(events, task, until="response.completed")
        task.steer_with(TWO)
        run.steer()  # the round has ended: nothing to restart
        rest = await read(events, task)
        self.assertEqual(task.outcomes["call_1"][1], "done")  # its route was decided at the round's completion
        self.assertEqual((task.settled, count(provider.payloads[1], TWO)), ([("steer_1", "applied")], 1))
        self.assertNotIn("response.incomplete", kinds(rest))

    async def test_every_batch_waiting_for_the_app_lands_after_its_outputs_in_the_next_round_together(self):
        provider, task = ScriptedProvider(call_round("get_menu", {}), text_round("Two, with oat milk.")), IMTask()
        run = adapter(provider).start(task)
        await drive(run, task)
        task.steer_with(TWO)
        task.steer_with(OAT, "steer_2")
        run.steer()
        task.give_outputs(MENU_OUTPUT)
        await run.resume()
        await drive(run, task)
        sent = provider.payloads[1]["input"]
        self.assertEqual([item.get("type") for item in sent[-4:-2]], ["function_call", "function_call_output"])
        self.assertEqual((len(provider.payloads), sent[-2:]), (2, [TWO[0], OAT[0]]))
        self.assertEqual(task.settled, [("steer_1", "applied"), ("steer_2", "applied")])


class StopTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_stop_before_the_first_round_while_waiting_for_the_app_or_after_the_answer_is_confirmed_at_once(
        self,
    ):
        for name, round_ in (
            ("before", None),
            ("waiting", call_round("get_menu", {})),
            ("answered", text_round("Done.")),
        ):
            with self.subTest(name):
                provider, task = ScriptedProvider(round_ or text_round("Never.")), IMTask()
                run = adapter(provider).start(task)
                if round_ is not None:
                    await drive(run, task)
                self.assertEqual(await run.stop(), "stopped")
                await run.resume()
                self.assertEqual((await drive(run, task), provider.opened), ([], int(round_ is not None)))

    async def test_a_stop_lets_started_calls_finish_and_dispatches_nothing_more(self):
        provider = ScriptedProvider(call_round("lookup", {}), text_round("Done.", "resp_2"))
        shop, task = RecordingTools(delay=0.2), IMTask()
        run = adapter(provider, tools=[shop]).start(task)
        reading = asyncio.create_task(drive(run, task))
        while not shop.calls:
            await asyncio.sleep(0.01)
        self.assertEqual(await run.stop(), "stopped")
        self.assertEqual(task.outcomes["call_1"][1], "done")  # it finished before stop() returned
        self.assertEqual((kinds(await reading)[-1], provider.opened), ("response.completed", 1))
        # A stop once the round with a call was passed on, before its call was dispatched: the call never runs.
        provider, shop, task = ScriptedProvider(call_round("lookup", {})), RecordingTools(), IMTask()
        run = adapter(provider, tools=[shop]).start(task)
        events = run.events()
        await read(events, task, until="response.completed")
        self.assertEqual(await run.stop(), "stopped")
        self.assertEqual(([event async for event in events], shop.calls, task.started), ([], [], []))

    async def test_a_stop_between_an_attempt_and_its_retry_opens_nothing(self):
        provider, task = ScriptedProvider(failed_round(), text_round("Done.")), IMTask()
        run = adapter(provider).start(task)
        seen = []
        async for event in run.events():
            seen.append(event["type"])
            if event["type"] == "response.failed":
                self.assertEqual(await run.stop(), "stopped")  # the safe point before the retry
        self.assertEqual((seen, provider.opened), (["response.created", "response.failed"], 1))

    async def test_a_round_that_does_not_let_go_leaves_the_stop_unconfirmed_at_its_limit(self):
        hung = text_round("Old.")
        hung.insert(1, Hang(1.0))
        for steered in (False, True):  # a round cut for steering whose reader hangs still holds the stop
            with self.subTest(steered=steered):
                gate, task = asyncio.Event(), IMTask()
                provider = ScriptedProvider(list(hung), gated(gate, text_round("New.")))
                run = adapter(provider, settings=LlmSettings(stop_timeout_seconds=0.1)).start(task)
                events = run.events()
                await anext(events)
                if steered:
                    task.steer_with(TWO)
                    run.steer()
                    self.assertEqual(
                        [(await anext(events))["type"] for _ in range(2)], ["response.incomplete", "response.created"]
                    )
                started = time.monotonic()
                self.assertEqual(await run.stop(), "unconfirmed")
                self.assertLess(time.monotonic() - started, 0.5)
                gate.set()
                self.assertEqual([event async for event in events], [])


if __name__ == "__main__":
    unittest.main()
