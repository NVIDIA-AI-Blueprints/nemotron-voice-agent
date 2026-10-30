# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Admission: responses open at acceptance, and what admission refuses."""

import asyncio

from delegation_core.im.session import DelegationError
from tests.unit.delegation_core.helpers import (
    IMCase,
    ScriptedAdapter,
    StartEach,
    answer,
    ask,
    read_all,
    wait,
)


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
