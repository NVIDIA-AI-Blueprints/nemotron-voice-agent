# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103
"""Doubles of the airline desk's tests: a desk on a fixed clock, and a test case that calls its tools like the model."""

import json
import re
import unittest
from datetime import UTC, datetime

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.tools.airline.state import (
    DATA_NOTICE as DATA_NOTICE,  # re-exported: the tests say the notice the desk says
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.tools.airline.tools import (
    AirlineTools,
)

# A Wednesday noon, UTC: the desk's "today" for every test.
NOW = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
TOMORROW = "2026-11-05"
BUSY_DAY = "2026-11-09"
SEARCH = {"origin": "Boston", "destination": "Denver", "date": TOMORROW}
# A route with the most flights a day (five each way), and one with four.
FIVE_A_DAY = {"origin": "JFK", "destination": "LAX", "date": BUSY_DAY}
FOUR_A_DAY = {"origin": "JFK", "destination": "BOS", "date": BUSY_DAY}
# The sample trips, as the timetable makes them at the fixed clock (pinned in test_airline_network.TimetableTests).
K7_FLIGHT, K7_SAYS, K7_LEAVES, K7_DELAYED = "DA 213", "Demo Air flight two thirteen", "9:55 AM", "10:40 AM"
M4_FLIGHT, M4_LEAVES, M4_DAY = "DA 497", "4:00 PM", "2026-11-28"
R9_FLIGHT, R9_SAYS, R9_LEAVES = "DA 713", "Demo Air flight seven thirteen", "10:00 AM"
CODE = re.compile(r"^(?=.*[A-Z])(?=.*[2-9])[A-Z2-9]{6}$")


def desk(now: datetime = NOW) -> AirlineTools:
    """Return a fresh desk whose clock stands at ``now`` and whose booking codes repeat."""
    return AirlineTools(clock=lambda: now, seed=7)


class DeskCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tools = desk()
        self.calls = 0

    async def call(self, name: str, tools: AirlineTools | None = None, **arguments) -> dict:
        """Call a tool with a call id of its own, as the backend model does, and return its JSON result."""
        self.calls += 1
        return json.loads(await (tools or self.tools).execute(f"call_{self.calls}", name, json.dumps(arguments)))

    async def sign_in(self, user_id: str = "jane_doe_001") -> dict:
        return await self.call("sign_in", user_id=user_id)

    async def booked(self, flight: str = "1", **search) -> dict:
        """Sign in, search ``search`` (BOS to DEN tomorrow by default), and book ``flight`` after its read-back."""
        await self.sign_in()
        found = await self.call("flights_search", **{**SEARCH, **search})
        self.assertEqual(found["status"], "ok")
        read_back = await self.call("new_booking", flight=flight)
        self.assertEqual(read_back["status"], "needs_confirmation")
        done = await self.call("new_booking", flight=flight, confirmed=True)
        self.assertEqual(done["status"], "confirmed")
        return done["booking"]

    def bookings(self, tools: AirlineTools | None = None) -> list:
        return list((tools or self.tools).store.bookings_by_id.values())


def flight_on(clock: str, code: str = "DA", day: str = TOMORROW) -> dict:
    """Return a minimal searched flight that leaves at the local time ``clock`` ("HH:MM") on ``day``."""
    return {"depart_local": f"{day}T{clock}:00-05:00", "airline": {"code": code, "name": code}}
