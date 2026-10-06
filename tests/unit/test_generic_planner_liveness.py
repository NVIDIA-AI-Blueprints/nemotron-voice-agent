# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""A planning attempt ends when its stream stalls, not while it is still producing."""

# ruff: noqa: D101, D102

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from examples.frontend_backend_agent.generic.backend import GenericThinkerBackend, PlannerCeilingError
from examples.frontend_backend_agent.src.stage_metrics import STREAM_ACTIVITY

_PLAN = {"tool": "response_hint", "reason": "params_missing"}


class _StreamingPlanner:
    """Report a chunk every ``interval`` seconds for ``chunks`` chunks, then answer."""

    def __init__(self, *, chunks: int, interval: float, first_delay: float = 0.0) -> None:
        self.chunks = chunks
        self.interval = interval
        self.first_delay = first_delay
        self.calls = 0
        self.cancelled = 0

    async def plan(self, *, query, state, history=None):
        self.calls += 1
        touch = STREAM_ACTIVITY.get()
        try:
            await asyncio.sleep(self.first_delay)
            for _ in range(self.chunks):
                if touch is not None:
                    touch()
                await asyncio.sleep(self.interval)
            if self.chunks == 0:
                await asyncio.sleep(3600)
            return dict(_PLAN)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


def _backend(planner, *, liveness: bool = True, ceiling: float = 2.0) -> GenericThinkerBackend:
    return GenericThinkerBackend(
        planner=planner,
        tools={},
        enabled_tools=(),
        overall_timeout_seconds=10.0,
        planner_timeout_seconds=ceiling,
        planner_first_chunk_timeout_seconds=0.5 if liveness else None,
        planner_stall_timeout_seconds=0.5 if liveness else None,
    )


class PlannerLivenessTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_slow_but_live_stream_is_not_cut_off(self) -> None:
        # 1.2 s of steady chunks outlives both 0.5 s liveness limits combined.
        planner = _StreamingPlanner(chunks=12, interval=0.1)

        plan = await _backend(planner)._await_plan(planner.plan(query="q", state={}))

        self.assertEqual(plan, _PLAN)

    async def test_the_same_stream_under_a_fixed_limit_times_out(self) -> None:
        planner = _StreamingPlanner(chunks=12, interval=0.1)
        backend = _backend(planner, liveness=False, ceiling=1.0)

        with self.assertRaises(TimeoutError):
            await backend._await_plan(planner.plan(query="q", state={}))

    async def test_a_stalled_stream_times_out_and_is_cancelled(self) -> None:
        planner = _StreamingPlanner(chunks=1, interval=3600)

        with self.assertRaisesRegex(TimeoutError, "stalled"):
            await _backend(planner)._await_plan(planner.plan(query="q", state={}))
        self.assertEqual(planner.cancelled, 1)

    async def test_a_stream_that_never_starts_times_out(self) -> None:
        planner = _StreamingPlanner(chunks=0, interval=0.1)

        with self.assertRaisesRegex(TimeoutError, "did not start"):
            await _backend(planner)._await_plan(planner.plan(query="q", state={}))
        self.assertEqual(planner.cancelled, 1)

    async def test_a_live_stream_at_its_ceiling_is_not_asked_again(self) -> None:
        planner = _StreamingPlanner(chunks=100, interval=0.1)
        backend = _backend(planner, ceiling=0.6)

        with self.assertRaises(PlannerCeilingError):
            await backend._plan_with_retry("call-1", "q", planning_round=1, prior_tool_results=[])
        self.assertEqual(planner.calls, 1)

    async def test_a_stalled_attempt_is_retried(self) -> None:
        attempts: list[str] = []

        async def plan(*, query, state, history=None):
            attempts.append(query)
            touch = STREAM_ACTIVITY.get()
            touch()
            if len(attempts) == 1:
                await asyncio.sleep(3600)
            return dict(_PLAN)

        backend = _backend(SimpleNamespace(plan=plan))

        result = await backend._plan_with_retry("call-1", "q", planning_round=1, prior_tool_results=[])

        self.assertEqual(result, _PLAN)
        self.assertEqual(len(attempts), 2)

    async def test_cancelling_the_caller_cancels_the_attempt(self) -> None:
        planner = _StreamingPlanner(chunks=100, interval=0.1)
        task = asyncio.create_task(_backend(planner)._await_plan(planner.plan(query="q", state={})))
        await asyncio.sleep(0.2)

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(planner.cancelled, 1)


if __name__ == "__main__":
    unittest.main()
