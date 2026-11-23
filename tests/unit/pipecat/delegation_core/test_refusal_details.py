# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Runtime-owned refusal provenance, distinct from text, delivery, and a global placement gate."""

import asyncio
from dataclasses import FrozenInstanceError

from tests.unit.pipecat.delegation_core.helpers import (
    IMCase,
    ScriptedAdapter,
    answer,
    call_item,
    wait,
)
from tests.unit.pipecat.delegation_core.test_im_calls import MENU_APP


class RefusalDetailsTests(IMCase):
    async def test_restrictions_keep_precedence_over_owned_input(self):
        gate = asyncio.Event()
        restricted = {**MENU_APP, "tool_choice": {"type": "allowed_tools", "tools": []}}
        adapter = ScriptedAdapter([wait(gate), answer()])
        await self.admit_each(self.session(adapter), "Menu?", "For two.", settings=restricted)
        task = adapter.runs[0].task
        self.assertEqual(task.route(call_item("get_menu", {}, "restricted")), "refused")
        details = task.refusal_details("restricted")
        self.assertEqual((details.reason, details.input_scope), ("unlisted", "none"))
        self.assertFalse(details.owned_input_pending)
        gate.set()

    async def test_owned_snapshot_survives_take_settle_duplicate_and_later_stop(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([wait(gate), answer()])
        await self.admit_each(self.session(adapter), "Menu?", "For two.", settings=MENU_APP)
        task = adapter.runs[0].task
        call = call_item("get_menu", {}, "owned")
        self.assertEqual(task.route(call), "refused")
        details = task.refusal_details("owned")
        self.assertEqual((details.input_scope, details.reason, details.generation), ("owned", "not_run", 1))
        self.assertEqual((details.work_id, details.task_id, details.call_id), (task.work_id, task.task_id, "owned"))
        self.assertTrue(details.owned_input_pending)
        with self.assertRaises(FrozenInstanceError):
            details.input_scope = "global"
        taken = task.take_boundary_input()
        self.assertEqual(task.route(call_item("get_menu", {}, "taken")), "refused")
        self.assertTrue(task.refusal_details("taken").owned_input_pending)
        task.settle(taken.steering, "applied")
        self.assertEqual(task.route(call_item("get_menu", {}, "now-runs")), "app")
        task.lifecycle.own_cancel = True
        self.assertEqual(task.route(call), "refused")
        self.assertIs(task.refusal_details("owned"), details)
        self.assertEqual(task.route(call_item("get_menu", {}, "stopped")), "refused")
        stopped = task.refusal_details("stopped")
        self.assertEqual((stopped.reason, stopped.input_scope, stopped.generation), ("stopped", "none", 3))
        self.assertFalse(stopped.owned_input_pending)
        gate.set()
