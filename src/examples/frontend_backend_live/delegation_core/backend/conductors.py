# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The steer rule: the conductor that places requests without a model call.

:class:`SteerOpenWork` places the backend LLM's requests, and those of the LLM agent and the harness at one task at a
time: arrivals join the newest open Work, whether queued, running or waiting for the app's outputs; with none open they
start one together. It holds no state and places at once, so a ``decide()`` cancelled before it placed anything leaves
nothing behind. It names no task, since the backend LLM runs one task per Work; a caller's stop joins the Work like
any request.
"""

from __future__ import annotations

from typing import ClassVar, get_args

from examples.frontend_backend_live.delegation_core.backend.port import ConductorInput, OpenState, Place, Start, Steer

# The states of a Work a request can still join; the records also list recently ended Works, which take none.
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
        newest = min(reversed(open_works), key=lambda work: work.seconds)  # on a tie, the one listed later
        await place(Steer(newest.work_id, delegations))
