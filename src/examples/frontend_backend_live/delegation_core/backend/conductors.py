# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The steer rule: the conductor that places requests without a model call."""

from __future__ import annotations

from typing import ClassVar, get_args

from examples.frontend_backend_live.delegation_core.backend.port import ConductorInput, OpenState, Place, Start, Steer

# A Work in these states is open; the records also list recently ended Works.
OPEN_STATES = frozenset(get_args(OpenState))


class SteerOpenWork:
    """``steer``: the arrivals join the newest open Work; with none open, they continue the Work that ended last.

    It places for a backend that runs one task at a time, so its Works are one line of work. A Steer to an ended Work
    starts a new Work that continues it, so a backend that keeps sessions can serve it from that Work's idle session.
    With no Work in the records (a call's first request, or ``records_max_works`` 0), the arrivals start one.
    """

    name: ClassVar[str] = "steer"

    async def decide(self, given: ConductorInput, place: Place) -> None:
        """Place every arrival with one decision: Steer the newest open Work, else the last ended one, else Start."""
        if not given.arrivals:
            return
        delegations = tuple(arrival.delegation_id for arrival in given.arrivals)
        open_works = [work for work in given.works if work.state in OPEN_STATES]
        if not given.works:
            await place(Start(delegations))
            return
        if not open_works:
            # The ended Works are listed in the order they ended.
            await place(Steer(given.works[-1].work_id, delegations))
            return
        # On a tie, the Work listed later.
        newest = min(reversed(open_works), key=lambda work: work.seconds)
        await place(Steer(newest.work_id, delegations))
