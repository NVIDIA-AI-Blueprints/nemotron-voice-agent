# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The steer rule: the conductor that places requests without a model call."""

from __future__ import annotations

from typing import ClassVar, get_args

from examples.frontend_backend_live.delegation_core.backend.port import ConductorInput, OpenState, Place, Start, Steer

# The records also list recently ended Works, which take no request.
OPEN_STATES = frozenset(get_args(OpenState))


class SteerOpenWork:
    """``steer``: the arrivals join the newest open Work, or start one together when none is open."""

    name: ClassVar[str] = "steer"

    async def decide(self, given: ConductorInput, place: Place) -> None:
        """Place every arrival with one decision: a Steer into the newest open Work, else a Start."""
        if not given.arrivals:
            return
        delegations = tuple(arrival.delegation_id for arrival in given.arrivals)
        open_works = [work for work in given.works if work.state in OPEN_STATES]
        if not open_works:
            await place(Start(delegations))
            return
        # On a tie, the Work listed later.
        newest = min(reversed(open_works), key=lambda work: work.seconds)
        await place(Steer(newest.work_id, delegations))
