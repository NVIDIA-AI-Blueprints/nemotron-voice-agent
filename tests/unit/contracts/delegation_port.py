# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D102, D103, D107

"""The contract of a ``DelegationPort`` (``live/ports.py``): where a frontend adapter hands its delegations.

* :class:`ScriptedDelegationPort`: a port for the cascade's tests. ``delegate()`` returns at once, or, with ``hold``,
  waits until the test accepts or refuses it; the test then hands events to the listener, or ``answer_with`` hands the
  same events for every delegation.
* :class:`DelegationPortContract`: each part's test subclasses it (``make_port()``).

Only the standard library is imported: a port is structural.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tests.unit.contracts.base import ContractCase

USER = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Find teas."}]}]


class ScriptedDelegationPort:
    """A ``DelegationPort`` the test drives: it records each request, and hands the events the test emits."""

    target = "responses"

    def __init__(
        self,
        *,
        reports_events: bool = True,
        acknowledgement_delay_ms: int = 0,
        hold: bool = False,
        answer_with: list[dict] | None = None,
    ):
        self.reports_events, self.acknowledgement_delay_ms, self.hold = reports_events, acknowledgement_delay_ms, hold
        self.answer_with = answer_with  # events handed for each delegation once delegate() has returned
        self.requests: list[list[dict]] = []
        self.listener: Any = None
        self.arrived = asyncio.Event()
        self._waiting: list[asyncio.Future] = []
        self._minted = 0

    def listen(self, listener: Any) -> None:
        self.listener = listener

    async def delegate(self, input: list[dict]) -> str:
        if self.reports_events and self.listener is None:
            raise RuntimeError("listen() comes before the first delegate()")
        self.requests.append(input)
        self.arrived.set()
        self._minted += 1
        delegation_id = f"item_{self._minted}"
        if self.hold:
            decided = asyncio.get_running_loop().create_future()
            self._waiting.append(decided)
            await decided
        for event in self.answer_with or ():
            asyncio.get_running_loop().call_soon(self.emit, delegation_id, event)  # after delegate() returns
        return delegation_id

    def accept(self) -> None:
        """Let the oldest held ``delegate()`` return its id."""
        self._waiting.pop(0).set_result(None)

    def refuse(self, error: Exception) -> None:
        """Make the oldest held ``delegate()`` raise ``error``, before any event."""
        self._waiting.pop(0).set_exception(error)

    def emit(self, delegation_id: str, event: dict) -> None:
        """Hand one official event of a delegation to the listener."""
        self.listener.on_delegation_event(delegation_id, event)

    def lose(self, delegation_id: str) -> None:
        """The delegation's stream ended with no terminal."""
        self.listener.on_delegation_lost(delegation_id)


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

    async def test_the_port_states_its_data(self):
        port = await self.make_port()
        self.assertIsInstance(port.target, str)
        self.assertIsInstance(port.reports_events, bool)
        self.assertIsInstance(port.acknowledgement_delay_ms, int)
        self.assertGreaterEqual(port.acknowledgement_delay_ms, 0)

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
