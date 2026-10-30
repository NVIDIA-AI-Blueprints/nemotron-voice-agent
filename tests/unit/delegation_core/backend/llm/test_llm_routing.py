# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Tests for call routing: mixed rounds work, calls are reported at dispatch and no output is lost."""

import json
import unittest

from delegation_core.backend.llm import LlmSettings
from tests.unit.contracts.backend import a_round, call_events, call_round, check_events, message_events, text_round
from tests.unit.delegation_core.backend.llm.doubles import (
    IMTask,
    RecordingTools,
    ScriptedProvider,
    WorkTools,
    adapter,
    drive,
    items_of,
)
from tests.unit.delegation_core.helpers import TEXTS, IMCase, ask, read_all


def calls_round(*names, said=None, prefix="call"):
    parts = [message_events(said, index=0)] if said else []
    for name in names:
        parts.append(call_events(name, {}, call_id=f"{prefix}_{name}", index=len(parts)))
    return a_round(*parts, response_id=f"resp_{prefix}")


class OrderedTask(IMTask):
    """Notes when each call is routed and when each round's terminal is seen."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timeline = []

    def route(self, call):
        self.timeline.append(f"route:{call['name']}")
        return super().route(call)

    def see(self, event):
        if event["type"] == "response.completed":
            self.timeline.append("completed")
        super().see(event)


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_mixed_round_runs_the_backends_calls_and_ends_for_the_apps(self):
        provider = ScriptedProvider(
            calls_round("get_menu", "lookup", "book", said="Checking."), text_round("Done.", "resp_2")
        )
        shop = RecordingTools(["lookup", "book"])
        task = OrderedTask()
        task.refused.add("call_book")  # e.g. after a stop was requested
        run = adapter(provider, tools=[shop]).start(task)
        first = await drive(run, task)
        self.assertEqual(check_events(first), [])
        self.assertEqual([call["name"] for call in items_of(first, "function_call")], ["get_menu", "lookup", "book"])
        self.assertEqual(items_of(first, "message")[0]["phase"], "commentary")
        self.assertEqual(task.timeline, ["route:get_menu", "route:lookup", "route:book", "completed"])
        self.assertEqual([name for _, name, _ in shop.calls], ["lookup"])  # neither the app's nor the refused one
        self.assertEqual(
            task.order, ["started:call_lookup", "outcome:call_lookup"]
        )  # reported before the iteration ended
        self.assertEqual(task.started, [("call_lookup", "shop")])
        task.give_outputs([{"type": "function_call_output", "call_id": "call_get_menu", "output": "{}"}])
        await run.resume()
        rest = await drive(run, task)
        self.assertEqual(items_of(rest, "message")[0]["phase"], "final_answer")
        outputs = [
            item["call_id"] for item in provider.payloads[1]["input"] if item.get("type") == "function_call_output"
        ]
        self.assertEqual(outputs, ["call_lookup", "call_get_menu"])

    async def test_one_failing_tool_keeps_its_siblings_outputs_and_the_task_goes_on(self):
        provider = ScriptedProvider(calls_round("lookup", "broken"), text_round("Done.", "resp_2"))
        task = IMTask()
        events = await drive(
            adapter(provider, tools=[RecordingTools(["lookup", "broken"], fail=["broken"])]).start(task), task
        )
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(task.outcomes["call_lookup"][1], "done")
        output, outcome = task.outcomes["call_broken"]
        self.assertEqual(
            (outcome, json.loads(output)["error"]), ("failed", "broken failed: RuntimeError: broken broke")
        )
        sent = [item["call_id"] for item in provider.payloads[1]["input"] if item.get("type") == "function_call_output"]
        self.assertEqual(sent, ["call_lookup", "call_broken"])

    async def test_a_slow_tool_is_unclear_and_an_unknown_one_is_told_the_offered_names(self):
        provider = ScriptedProvider(calls_round("slow", "invented"), text_round("Done.", "resp_2"))
        task = IMTask()
        settings = LlmSettings(server_tool_timeout_seconds=0.05)
        await drive(adapter(provider, tools=[RecordingTools(["slow"], delay=1.0)], settings=settings).start(task), task)
        # Cut at its limit after its effect may have happened: whether it ran is not known.
        self.assertEqual(task.outcomes["call_slow"][1], "unclear")
        self.assertIn("did not answer within 0.05 s", task.outcomes["call_slow"][0])
        error = json.loads(task.outcomes["call_invented"][0])["error"]
        self.assertEqual(
            (error, task.outcomes["call_invented"][1]),
            ("Unknown tool: invented. The tools offered: get_menu, slow", "failed"),
        )
        self.assertIn(("call_invented", "llm"), task.started)

    async def test_a_name_the_round_did_not_offer_runs_as_written_even_close_to_an_offered_one(self):
        # No name is guessed: the unknown-tool output lists the offered names, and the model reads them next round.
        provider = ScriptedProvider(calls_round("LookupTool"), text_round("Done.", "resp_2"))
        shop, task = RecordingTools(["lookup"]), IMTask()
        events = await drive(adapter(provider, tools=[shop]).start(task), task)
        self.assertEqual([item["name"] for item in items_of(events, "function_call")], ["LookupTool"])
        self.assertEqual((shop.calls, task.started), ([], [("call_LookupTool", "llm")]))
        unknown = task.texts.unknown_tool_output.format(name="LookupTool", offered="get_menu, lookup")
        self.assertEqual(task.outcomes["call_LookupTool"], (json.dumps({"error": unknown}), "failed"))

    async def test_every_round_offers_the_apps_the_sources_and_the_work_tools_each_name_once(self):
        provider = ScriptedProvider(calls_round("lookup"), text_round("Done.", "resp_2"))
        clashing = RecordingTools(["get_menu", "lookup"])  # get_menu is also the app's: the app's wins
        task = IMTask(work_tools=WorkTools())
        await drive(adapter(provider, tools=[clashing]).start(task), task)
        for payload in provider.payloads:
            self.assertEqual([tool["name"] for tool in payload["tools"]], ["get_menu", "lookup", "show", "cancel_work"])
        self.assertEqual(task.started, [("call_lookup", "shop")])


USER_FUNCTION = {
    "type": "function",
    "name": "get_user_details",
    "description": "A user's profile.",
    "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}},
}


class RepeatedCallTests(IMCase):
    """Under the real IM, an app call that repeats the newest call that ran is answered from the record."""

    async def test_the_repeat_never_reaches_the_app_and_the_next_round_reads_the_recorded_answer(self):
        lookup = {"user_id": "noh_mlr_9847"}
        provider = ScriptedProvider(
            call_round("get_user_details", lookup, "call_1"),
            call_round("get_user_details", lookup, "call_2"),
            text_round("Please spell your user id.", "resp_3"),
        )
        im = self.session(adapter(provider))
        settings = {"tools": [USER_FUNCTION]}
        first = await read_all(im.admit(ask("Find my account."), delegation_id="D1", settings=settings).stream)
        self.assertEqual([call["call_id"] for call in items_of(first, "function_call")], ["call_1"])
        output = [{"type": "function_call_output", "call_id": "call_1", "output": "Error: user not found"}]
        rest = await read_all(im.admit_client_request(output, delegation_id="D1", settings=settings).stream)
        self.assertEqual(items_of(rest, "function_call"), [])  # the repeat never went to the app
        self.assertEqual(items_of(rest, "message")[-1]["content"][0]["text"], "Please spell your user id.")
        outputs = {item["call_id"]: item["output"] for item in provider.payloads[2]["input"] if "output" in item}
        repeated = TEXTS.repeated_call_output.format(output="Error: user not found")
        self.assertEqual(outputs, {"call_1": "Error: user not found", "call_2": repeated})


if __name__ == "__main__":
    unittest.main()
