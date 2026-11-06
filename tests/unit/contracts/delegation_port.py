# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D102, D107

"""The contract of a ``DelegationPort`` (``live/ports.py``): where a frontend adapter hands its delegations.

Each delegation part's test subclasses :class:`DelegationPortContract` (``make_port()``). Only the standard library is
imported: a port is structural.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tests.unit.contracts.base import ContractCase

USER = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Find teas."}]}]


class Recorder:
    """A ``DelegationEvents`` listener that notes each event, and whether ``delegate()`` had returned by then."""

    def __init__(self):
        self.events: list[tuple[str, dict, bool]] = []
        self.lost: list[str] = []
        self.returned = False

    def on_delegation_event(self, delegation_id: str, event: dict) -> None:
        self.events.append((delegation_id, event, self.returned))

    def on_delegation_lost(self, delegation_id: str) -> None:
        self.lost.append(delegation_id)


class DelegationPortContract(ContractCase):
    """What every delegation part offers a frontend adapter.

    A subclass is an ``IsolatedAsyncioTestCase`` that implements ``async make_port()``: an opened part. When the part
    reports events, the backend it reaches answers each request with its events.
    """

    async def make_port(self) -> Any:
        raise NotImplementedError

    async def test_delegate_returns_an_id(self):
        port = await self.make_port()
        port.listen(Recorder())
        delegation_id = await port.delegate(USER)
        self.assertIsInstance(delegation_id, str)
        self.assertTrue(delegation_id)

    async def test_no_event_reaches_the_listener_before_delegate_returns(self):
        port = await self.make_port()
        recorder = Recorder()
        port.listen(recorder)
        await port.delegate(USER)
        recorder.returned = True
        for _ in range(20):  # let the part hand what follows
            await asyncio.sleep(0)
        self.assertTrue(all(returned for _, _, returned in recorder.events))
        if port.reports_events:
            self.assertTrue(recorder.events, "a part that reports events hands the delegation's events")

    async def test_a_part_that_reports_events_refuses_a_delegation_with_no_listener(self):
        port = await self.make_port()
        if not port.reports_events:
            self.skipTest("no event follows this part's delegations")
        with self.assertRaises(Exception):  # noqa: B017 - any refusal, before any event
            await port.delegate(USER)
