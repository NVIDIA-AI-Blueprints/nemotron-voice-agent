# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The steer rule: the conductor that places requests without a model call."""

from __future__ import annotations

from typing import ClassVar, get_args

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    ConductorInput,
    OpenState,
    Place,
    Start,
    Steer,
    WorkTarget,
)

# A Work in these states is open; the records also list recently ended Works.
OPEN_STATES = frozenset(get_args(OpenState))


class SteerOpenWork:
    """``steer``: the arrivals join the newest open Work; with none open, they continue the Work that ended last.

    It places for a backend that runs one task at a time, so its Works are one line of work. A Steer to an ended Work
    starts a new Work that continues it, so a backend that keeps sessions can serve it from that Work's idle session.
    A zero-length records window can retain an exact eligible ended-thread hint. A cancelled or stopping
    thread is not a continuation target; subsequent requests start independent work.
    """

    name: ClassVar[str] = "steer"
    model_placement: ClassVar[bool] = False

    async def decide(self, given: ConductorInput, place: Place) -> None:
        """Place every arrival with one decision: Steer the newest open Work, else the last ended one, else Start."""
        if not given.arrivals:
            return
        delegations = tuple(arrival.delegation_id for arrival in given.arrivals)
        open_works = [
            work
            for work in given.works
            if work.state in OPEN_STATES
            and any(task.state in OPEN_STATES and task.stop == "none" for task in work.tasks)
        ]
        if not given.works:
            await place(
                Steer(delegations, (WorkTarget(given.last_ended_work, (given.last_ended_task,)),))
                if given.last_ended_work and given.last_ended_task
                else Start(delegations)
            )
            return
        if not open_works:
            latest = given.works[-1]
            eligible = [
                task.task_id
                for task in latest.tasks
                if task.state in ("answered", "failed")
                and task.stop == "none"
                and task.steering in (None, "continuable")
            ]
            await place(
                Steer(delegations, (WorkTarget(latest.work_id, tuple(eligible)),)) if eligible else Start(delegations)
            )
            return
        # On a tie, the Work listed later.
        newest = min(reversed(open_works), key=lambda work: work.seconds)
        tasks = tuple(task.task_id for task in newest.tasks if task.state in OPEN_STATES and task.stop == "none")
        await place(Steer(delegations, (WorkTarget(newest.work_id, tasks),)))
