# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Closing an IM session: what stops at once, the grace period, and the order of the last responses."""

import asyncio

from tests.unit.delegation_core.helpers import (
    MENU,
    IMCase,
    ScriptedAdapter,
    ScriptedConductor,
    StartEach,
    answer,
    app_call,
    ask,
    read_all,
    wait,
)


class CloseTests(IMCase):
    async def test_a_queued_delegation_ends_cancelled_with_no_reason_and_never_runs(self):
        never = asyncio.Event()
        adapter = ScriptedAdapter([wait(never)], [answer()], conductor=StartEach())
        im = self.session(adapter, close_grace_seconds=0.01)
        im.admit(ask("Busy."), delegation_id="D1")
        queued = im.admit(ask("Next."), delegation_id="D2")
        await self.turns()
        await im.close()
        events = await read_all(queued.stream)
        self.assertEqual([e["type"] for e in events], ["response.created", "response.queued", "response.incomplete"])
        self.assertEqual(events[-1]["response"]["status"], "cancelled")
        self.assertIsNone(events[-1]["response"]["incomplete_details"])
        self.assertEqual(len(adapter.runs), 1)
        self.assertEqual(im.ledger.delegations["D2"].state, "cancelled")

    async def test_a_running_work_that_ends_within_the_grace_answers(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([wait(gate), answer("Made it.")])
        im = self.session(adapter, close_grace_seconds=1.0)
        stream = im.admit(ask("Almost done."), delegation_id="D1").stream
        await self.turns()
        closing = asyncio.create_task(im.close())
        await self.turns()
        gate.set()
        await closing
        events = await read_all(stream)
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(adapter.runs[0].stops, 0)

    async def test_a_running_work_past_the_grace_is_stopped_and_ends_cancelled(self):
        never = asyncio.Event()
        adapter = ScriptedAdapter([wait(never), answer()])
        im = self.session(adapter, close_grace_seconds=0.02)
        stream = im.admit(ask("Long."), delegation_id="D1").stream
        await self.turns()
        await im.close()
        events = await read_all(stream)
        self.assertEqual((events[-1]["type"], events[-1]["response"]["status"]), ("response.incomplete", "cancelled"))
        self.assertEqual(adapter.runs[0].stops, 1)
        self.assertEqual(im.ledger.works["W1"].state, "cancelled")
        self.assertTrue(adapter.closed)

    async def test_a_work_waiting_for_the_app_stops_at_once(self):
        adapter = ScriptedAdapter([app_call("get_menu"), answer()])
        im = self.session(adapter, close_grace_seconds=60)
        await read_all(im.admit(ask("Menu?"), delegation_id="D1", settings={"tools": [MENU]}).stream)
        await asyncio.wait_for(im.close(), 5)
        self.assertEqual(adapter.runs[0].stops, 1)
        self.assertEqual(im.ledger.works["W1"].state, "cancelled")

    async def test_an_arrival_the_conductor_is_still_placing_ends_cancelled(self):
        thinking = asyncio.Event()

        async def slow(given, place):
            await thinking.wait()

        im = self.session(ScriptedAdapter(conductor=ScriptedConductor(slow)))
        stream = im.admit(ask("Hello?"), delegation_id="D1").stream
        await self.turns()
        await im.close()
        events = await read_all(stream)
        self.assertEqual((events[-1]["type"], events[-1]["response"]["status"]), ("response.incomplete", "cancelled"))

    async def test_without_a_grace_running_work_may_finish(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([wait(gate), answer("Finished.")])
        im = self.session(adapter, close_grace_seconds=None)
        stream = im.admit(ask("Long."), delegation_id="D1").stream
        await self.turns()
        closing = asyncio.create_task(im.close())
        await asyncio.sleep(0.05)
        self.assertFalse(closing.done())
        gate.set()
        await closing
        self.assertEqual((await read_all(stream))[-1]["type"], "response.completed")

    async def test_the_adapter_closes_even_when_a_stop_fails_and_close_is_safe_twice(self):
        class BrokenStop(ScriptedAdapter):
            def start(self, task):
                run = super().start(task)

                async def stop():
                    raise RuntimeError("stop failed")

                run.stop = stop
                return run

        never = asyncio.Event()
        adapter = BrokenStop([wait(never)])
        im = self.session(adapter, close_grace_seconds=0.01)
        stream = im.admit(ask("Long."), delegation_id="D1").stream
        await self.turns()
        await im.close()
        await im.close()
        self.assertTrue(adapter.closed)
        self.assertEqual(im.ledger.tasks["W1.1"].stop.state, "unconfirmed")
        self.assertEqual((await read_all(stream))[-1]["type"], "response.incomplete")
        self.assertEqual(im.ledger.session.state, "ended")
