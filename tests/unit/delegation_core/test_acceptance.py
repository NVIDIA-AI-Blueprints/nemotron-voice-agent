# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Responses open at acceptance, and every response passes through ``in_progress`` before its terminal."""

import asyncio
import unittest

from delegation_core.api.local import LocalResponsesClient
from delegation_core.api.service import ResponsesService
from delegation_core.backend.port import Answer, Start
from delegation_core.im.session import DelegationError
from tests.unit.contracts.responses import ResponsesClientContract
from tests.unit.delegation_core.emulator import emulator_source
from tests.unit.delegation_core.helpers import (
    IMCase,
    ScriptedAdapter,
    ScriptedConductor,
    StartEach,
    answer,
    ask,
    im_session,
    read_all,
    wait,
)


class IMResponsesClientContractTests(ResponsesClientContract, unittest.IsolatedAsyncioTestCase):
    """The IM behind the in-process front door, with the API surface's ``input_tokens``."""

    async def make_client(self):
        new_session = lambda: im_session(ScriptedAdapter(), close_grace_seconds=0.01)  # noqa: E731
        return LocalResponsesClient(ResponsesService(new_session, backend=emulator_source()))


class AcceptanceTests(IMCase):
    async def test_the_first_two_events_are_written_at_admission(self):
        im = self.session(ScriptedAdapter([answer()]))
        stream = im.admit(ask("Find teas."), delegation_id="D1").stream
        self.assertEqual(stream.unread, 2)  # before anything else runs
        events = await read_all(stream)
        self.assertEqual(
            [e["type"] for e in events[:3]], ["response.created", "response.queued", "response.in_progress"]
        )
        self.assertEqual([e["response"]["status"] for e in events[:3]], ["queued", "queued", "in_progress"])
        self.assertEqual(len({e["response"]["id"] for e in events if "response" in e}), 1)

    async def test_the_seventeenth_waiting_request_is_refused_before_anything_changes(self):
        never = asyncio.Event()
        adapter = ScriptedAdapter([wait(never)], conductor=StartEach())
        im = self.session(adapter)
        im.admit(ask("Busy."), delegation_id="D0")
        await self.turns()
        for n in range(1, 17):
            im.admit(ask(f"Request {n}."), delegation_id=f"D{n}")
        before = len(im.ledger.delegations)
        with self.assertRaisesRegex(DelegationError, "full"):
            im.admit(ask("One too many."), delegation_id="D17")
        self.assertEqual(len(im.ledger.delegations), before)
        self.assertNotIn("D17", im.ledger.delegations)

    async def test_a_handover_and_an_answer_pass_through_in_progress(self):
        async def plan(given, place):
            ids = [arrival.delegation_id for arrival in given.arrivals]
            if ids == ["D1", "D2"]:
                await place(Start(("D1", "D2")))
            else:
                await place(Answer(ids[0], "Nothing runs now."))

        never = asyncio.Event()
        adapter = ScriptedAdapter([wait(never)], conductor=ScriptedConductor(plan))
        im = self.session(adapter)
        first = im.admit(ask("Teas."), delegation_id="D1")
        im.admit(ask("Green only."), delegation_id="D2")
        older = await read_all(first.stream)
        self.assertEqual(
            [e["type"] for e in older if e["type"].startswith("response.") and "response" in e],
            ["response.created", "response.queued", "response.in_progress", "response.completed"],
        )
        answered = await read_all(im.admit(ask("Status?"), delegation_id="D3").stream)
        self.assertEqual(
            [e["type"] for e in answered if "response" in e],
            ["response.created", "response.queued", "response.in_progress", "response.completed"],
        )

    async def test_a_reader_that_stops_reading_leaves_the_response_running(self):
        gate = asyncio.Event()
        im = self.session(ScriptedAdapter([wait(gate), answer()]))
        stream = im.admit(ask("Find teas."), delegation_id="D1").stream
        await stream.aclose()  # the API side's LocalStream.aclose() calls it on every stream it reads
        gate.set()
        await self.turns()
        self.assertEqual(im.ledger.delegations["D1"].state, "answered")

    async def test_a_closing_session_or_an_id_in_use_is_refused(self):
        im = self.session(ScriptedAdapter([answer()]))
        await read_all(im.admit(ask("One."), delegation_id="D1").stream)
        with self.assertRaisesRegex(DelegationError, "already exists"):
            im.admit(ask("Again."), delegation_id="D1")
        await im.close()
        with self.assertRaisesRegex(DelegationError, "closing"):
            im.admit(ask("Late."), delegation_id="D2")
