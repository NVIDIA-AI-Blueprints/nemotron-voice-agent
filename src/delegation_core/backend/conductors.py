# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rule conductors that place the backend LLM's requests without a model call.

``backend.llm.conductor`` names the rule:

* ``steer`` (:class:`SteerOpenWork`, the default): arrivals join the newest open Work, whether queued, running or
  waiting for the app's outputs; with none open they start one together.
* ``start_each`` (:class:`StartEach`, the queue profile): each arrival starts its own Work, which waits its turn.

Both hold no state and place at once, so a ``decide()`` cancelled before it placed anything leaves nothing behind.
Neither names a task, since the backend LLM runs one task per Work; a caller's stop joins the Work like any request.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, get_args

from delegation_core.backend.port import ConductorInput, OpenState, Place, Start, Steer

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


class StartEach:
    """``start_each``: each arrival starts its own Work (the queue profile)."""

    name: ClassVar[str] = "start_each"

    async def decide(self, given: ConductorInput, place: Place) -> None:
        """Place each arrival in a new Work of its own, oldest first."""
        for arrival in given.arrivals:
            await place(Start((arrival.delegation_id,)))


# Each rule conductor by its name in backend.llm.conductor.
RULE_CONDUCTORS: Mapping[str, type[SteerOpenWork] | type[StartEach]] = {
    rule.name: rule for rule in (SteerOpenWork, StartEach)
}
