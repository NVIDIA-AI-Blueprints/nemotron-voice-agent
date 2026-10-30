# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The handover always names its taker, in a fixed order of events."""

import asyncio

from delegation_core.backend.port import Cancel, Start, Steer
from tests.unit.delegation_core.helpers import (
    MENU,
    TEXTS,
    IMCase,
    ScriptedAdapter,
    ScriptedConductor,
    answer,
    app_call,
    app_outputs,
    ask,
    checkpoint,
    is_handover_line,
    read_all,
    wait,
)


class HandoverCase(IMCase):
    def handovers(self) -> list[str]:
        """The responses that ended with only the handover line (each one an older delegation handed over)."""
        ended = []
        for response_id, event in self.wire.events:
            if event["type"] == "response.completed":
                output = event["response"]["output"]
                texts = [part["text"] for item in output for part in item.get("content", [])]
                if len(texts) == 1 and is_handover_line(texts[0]):
                    ended.append(response_id)
        return ended

    def assertEveryHandoverNamesItsTaker(self) -> None:
        """Each handed-over response is named by exactly one taker's in_progress, written before it ended."""
        handed = self.handovers()
        self.assertTrue(handed, "the scenario hands nothing over")
        order = list(self.wire.events)

        def first(response_id: str, kind: str) -> int:
            return next(n for n, (rid, event) in enumerate(order) if rid == response_id and event["type"] == kind)

        for older in handed:
            named = [
                n
                for n, (_, event) in enumerate(order)
                if event["type"] == "response.in_progress" and event["response"].get("previous_response_id") == older
            ]
            line, ended = first(older, "response.output_item.done"), first(older, "response.completed")
            self.assertEqual(len(named), 1, f"{older} is named by {len(named)} takers")
            self.assertLess(named[0], line)
            self.assertLess(line, ended)


class HandoverTests(HandoverCase):
    async def test_a_steer_into_running_work_names_the_taker_then_ends_the_older_response(self):
        gate = asyncio.Event()
        im = self.session(ScriptedAdapter([wait(gate), answer("Green teas: two.")]))
        first = im.admit(ask("Find teas."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("Green only."), delegation_id="D2")
        await self.turns()
        gate.set()
        older, taker = await read_all(first.stream), await read_all(second.stream)
        in_progress = next(e for e in taker if e["type"] == "response.in_progress")
        self.assertEqual(in_progress["response"]["previous_response_id"], older[0]["response"]["id"])
        self.assertEqual(older[-1]["type"], "response.completed")
        self.assertEqual(older[-2]["item"]["phase"], "commentary")
        self.assertEqual(older[-2]["item"]["content"][0]["text"], TEXTS.handover_line.format(delegation="D2"))
        self.assertEqual(im.ledger.delegations["D1"].state, "handed_over")
        self.assertEqual(im.ledger.delegations["D1"].handed_to, "D2")
        self.assertEqual(im.ledger.works["W1"].carrier, "D2")
        self.assertEveryHandoverNamesItsTaker()

    async def test_a_merge_into_a_queued_work_names_the_taker_at_once(self):
        never, hold = asyncio.Event(), asyncio.Event()

        async def plan(given, place):
            ids = tuple(arrival.delegation_id for arrival in given.arrivals)
            if ids in (("D0",), ("D1",)):
                await place(Start(ids))
            else:
                queued = [work.work_id for work in given.works if work.state == "queued"]
                await place(Steer(queued[-1], ids))
                hold.set()

        adapter = ScriptedAdapter([wait(never)], [answer()], conductor=ScriptedConductor(plan))
        im = self.session(adapter)
        im.admit(ask("Busy."), delegation_id="D0")
        await self.turns()
        first = im.admit(ask("Teas."), delegation_id="D1")
        await self.turns()
        im.admit(ask("Green only."), delegation_id="D2")
        await hold.wait()
        older = await read_all(first.stream)
        self.assertEqual(
            [e["type"] for e in older if "response" in e],
            ["response.created", "response.queued", "response.in_progress", "response.completed"],
        )
        self.assertEqual(im.ledger.works["W2"].state, "queued")
        self.assertEqual(im.ledger.works["W2"].carrier, "D2")
        self.assertEveryHandoverNamesItsTaker()

    async def test_a_steer_into_work_waiting_for_the_app_lands_after_the_outputs(self):
        im = self.session(ScriptedAdapter([app_call("get_menu"), answer("Green: two.")]))
        menu = {"tools": [MENU]}
        first = await read_all(im.admit(ask("Find teas."), delegation_id="D1", settings=menu).stream)
        second = im.admit(ask("Green only."), delegation_id="D2", settings=menu)
        await self.turns()
        self.assertEqual(im.ledger.delegations["D2"].state, "queued")
        continuation = im.admit_client_request(app_outputs(first), delegation_id="D1", settings=menu)
        resumed, taker = await read_all(continuation.stream), await read_all(second.stream)
        named = next(e for e in taker if e["type"] == "response.in_progress")["response"]["previous_response_id"]
        self.assertEqual(named, resumed[0]["response"]["id"])
        self.assertEqual(resumed[-1]["type"], "response.completed")
        self.assertEqual(taker[-1]["response"]["output"][-1]["content"][0]["text"], "Green: two.")
        self.assertEveryHandoverNamesItsTaker()

    async def test_a_start_of_three_requests_hands_over_as_a_chain(self):
        async def plan(given, place):
            await place(Start(tuple(arrival.delegation_id for arrival in given.arrivals)))

        im = self.session(ScriptedAdapter([answer()], conductor=ScriptedConductor(plan)))
        streams = [im.admit(ask(f"Part {n}."), delegation_id=f"D{n}").stream for n in (1, 2, 3)]
        events = [await read_all(stream) for stream in streams]
        named = [next(e for e in each if e["type"] == "response.in_progress")["response"] for each in events]
        self.assertEqual(named[1]["previous_response_id"], named[0]["id"])
        self.assertEqual(named[2]["previous_response_id"], named[1]["id"])
        self.assertEqual([im.ledger.delegations[f"D{n}"].state for n in (1, 2, 3)], ["handed_over"] * 2 + ["answered"])
        self.assertEqual([im.ledger.delegations[f"D{n}"].handed_to for n in (1, 2)], ["D2", "D3"])
        self.assertEveryHandoverNamesItsTaker()
        # One Work carries the requests, oldest first; the newest carries it, and its task read them all.
        work = im.ledger.works["W1"]
        self.assertEqual((work.requests, work.carrier, work.state), (["D1", "D2", "D3"], "D3", "answered"))
        self.assertIn("Part 1.", str(im.ended_works[-1].requested))
        self.assertIn("Part 3.", str(im.ended_works[-1].requested))

    async def test_steering_from_one_task_while_another_waits_hands_over_when_the_carrier_reopens(self):
        gate = asyncio.Event()

        async def plan(given, place):
            ids = tuple(arrival.delegation_id for arrival in given.arrivals)
            if ids == ("D1",):
                await place(Start(ids, ("menu", "hours")))
            else:
                await place(Steer("W1", ids, "W1.2"))

        adapter = ScriptedAdapter(
            [app_call("get_menu"), answer("Menu.")],
            [wait(gate), checkpoint(), answer("Hours, for green tea.")],
            conductor=ScriptedConductor(plan),
            max_running=2,
        )
        im = self.session(adapter)
        menu = {"tools": [MENU]}
        first = await read_all(im.admit(ask("Menu and hours."), delegation_id="D1", settings=menu).stream)
        second = im.admit(ask("Green tea only."), delegation_id="D2")
        await self.turns()
        gate.set()
        await self.turns()
        self.assertEqual(im.works["W1"].deferred, ["D2"])  # landed, but D1's response waits for the app
        continuation = im.admit_client_request(app_outputs(first), delegation_id="D1", settings=menu)
        resumed, taker = await read_all(continuation.stream), await read_all(second.stream)
        named = next(e for e in taker if e["type"] == "response.in_progress")["response"]["previous_response_id"]
        self.assertEqual(named, resumed[0]["response"]["id"])
        self.assertEqual(
            [part["text"] for e in taker if e["type"] == "response.output_item.done" for part in e["item"]["content"]],
            ["Hours, for green tea.", "Menu."],
        )
        self.assertEveryHandoverNamesItsTaker()


class CancelForANewerRequestTests(IMCase):
    async def cancel(self, stop_outcome: str) -> list[dict]:
        gate = asyncio.Event()

        async def plan(given, place):
            ids = tuple(arrival.delegation_id for arrival in given.arrivals)
            if ids == ("D1",):
                await place(Start(ids))
            else:
                await place(Cancel("W1"))
                await place(Start(ids))

        adapter = ScriptedAdapter(
            [wait(gate), answer()], [answer()], conductor=ScriptedConductor(plan), stop_outcome=stop_outcome
        )
        im = self.session(adapter)
        first = im.admit(ask("Book it."), delegation_id="D1")
        await self.turns()
        im.admit(ask("Stop, never mind."), delegation_id="D2")
        older = await read_all(first.stream)
        self.im, self.adapter = im, adapter
        return older

    async def test_a_confirmed_stop_writes_the_cancelled_line_then_completed(self):
        older = await self.cancel("stopped")
        line = older[-2]["item"]
        self.assertEqual((line["phase"], line["content"][0]["text"]), ("final_answer", TEXTS.cancelled_line))
        self.assertEqual(older[-1]["type"], "response.completed")
        self.assertEqual(self.adapter.runs[0].stops, 1)
        work = self.im.ledger.works["W1"]
        self.assertEqual((work.state, self.im.ledger.tasks["W1.1"].stop.state), ("cancelled", "stopped"))

    async def test_a_stop_that_does_not_return_in_time_is_unconfirmed(self):
        gate = asyncio.Event()

        async def plan(given, place):
            ids = tuple(arrival.delegation_id for arrival in given.arrivals)
            await place(Start(ids) if ids == ("D1",) else Cancel("W1"))

        adapter = ScriptedAdapter([wait(gate), answer()], conductor=ScriptedConductor(plan), stop_delay=1.0)
        im = self.session(adapter, stop_timeout_seconds=0.05)
        first = im.admit(ask("Book it."), delegation_id="D1")
        await self.turns()
        im.admit(ask("Stop."), delegation_id="D2")
        older = await read_all(first.stream)
        self.assertEqual(older[-2]["item"]["content"][0]["text"], TEXTS.unconfirmed_line)
        self.assertEqual(im.ledger.tasks["W1.1"].stop.state, "unconfirmed")
