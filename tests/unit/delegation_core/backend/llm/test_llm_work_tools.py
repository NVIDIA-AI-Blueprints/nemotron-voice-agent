# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Tests for the IM's work tools, ``show`` and ``cancel_work``, in the backend LLM's run."""

import json
import unittest

from tests.unit.contracts.backend import a_round, call_events, message_events, text_round
from tests.unit.delegation_core.backend.llm.doubles import (
    IMTask,
    RecordingTools,
    ScriptedProvider,
    WorkTools,
    adapter,
    drive,
    items_of,
)


def calling(name, arguments=None, call_id="call_1", said=None):
    parts = [message_events(said, index=0)] if said else []
    parts.append(call_events(name, arguments or {}, call_id=call_id, index=len(parts)))
    return a_round(*parts, response_id=f"resp_{call_id}")


class WorkToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_show_reaches_the_im_and_the_answer_follows(self):
        provider = ScriptedProvider(
            calling("show", {"display": "Latte $4.50"}, said="Here is the menu."), text_round("A latte is $4.50.")
        )
        work = WorkTools()
        task = IMTask(work_tools=work)
        events = await drive(adapter(provider).start(task), task)
        self.assertEqual(work.shown, ["Latte $4.50"])
        self.assertEqual((task.started, task.outcomes["call_1"][1]), ([("call_1", "delegation_core")], "done"))
        self.assertEqual([m["phase"] for m in items_of(events, "message")], ["commentary", "final_answer"])

    async def test_cancel_work_stops_its_own_work_after_the_step(self):
        provider = ScriptedProvider(
            calling("cancel_work", call_id="call_stop"),
            calling("lookup", call_id="call_late"),
            text_round("I stopped that.", "resp_end"),
        )
        shop, work = RecordingTools(), WorkTools()
        task = IMTask(work_tools=work)
        events = await drive(adapter(provider, tools=[shop]).start(task), task)
        self.assertTrue(work.cancelled)
        self.assertEqual(shop.calls, [])  # the later call was refused, never run
        self.assertEqual(task.started, [("call_stop", "delegation_core")])
        self.assertEqual(items_of(events, "message")[-1]["phase"], "final_answer")  # its next answer is delivered
        self.assertEqual(json.loads(task.outcomes["call_stop"][0]), {"stopping": "after this step"})


if __name__ == "__main__":
    unittest.main()
