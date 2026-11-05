# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The voice's lines, settled by the playout the edge after the output reports."""

import asyncio
import unittest

from examples.frontend_backend_live.spoken import AsRecorded, EstimatedHeard, Line, LineKind, SpokenLines

TEXT = "First part. Your order number is A B 12."


class Settled:
    def __init__(self):
        self.events: list[tuple] = []

    def heard(self, line, cut):
        self.events.append(("heard", line.heard, cut))

    def not_heard(self, line):
        self.events.append(("not_heard",))

    def say_again(self, line):
        self.events.append(("again",))


def playing(cut_memory=None, *, sentences=True, start_timeout=3.0, started=True):
    """Lines that settle into a :class:`Settled`, with :data:`TEXT` handed and reached, and started if ``started``."""
    settled = Settled()
    spoken = SpokenLines(
        settled,
        cut_memory=cut_memory or AsRecorded(),
        start_timeout_seconds=start_timeout,
        retry_silent_lines=True,
        reports_sentences=sentences,
        observe=lambda *a, **k: None,
    )
    spoken.handed(Line(LineKind.REPLY, TEXT, number=1))
    spoken.reached(1)
    if started:
        spoken.started(1)
    return spoken, settled


class SpokenLinesTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_line_that_has_not_started_by_the_start_timeout_is_said_once_more(self):
        for started, events in ((False, [("again",)]), (True, [])):
            with self.subTest(started=started):
                _, settled = playing(start_timeout=0.05, started=started)
                await asyncio.sleep(0.1)
                self.assertEqual(settled.events, events)

    async def test_a_cut_line_keeps_what_its_cut_memory_says_the_caller_heard(self):
        # 11 characters played in 1.1 s; the estimate keeps the words the audio past them likely carried.
        for memory, sentences, kept in (
            (AsRecorded(), True, "First part."),
            (EstimatedHeard(15.0), True, "First part. Your order"),
            (AsRecorded(), False, TEXT),  # a stitched line: its text comes only at its end, so it is heard whole
            (EstimatedHeard(15.0), False, "First part."),
        ):
            with self.subTest(memory=type(memory).__name__, sentences=sentences):
                spoken, settled = playing(memory, sentences=sentences)
                if sentences:
                    spoken.played(1, "First part.", 1100)
                spoken.cut(1, 1000, "Your order number is A B 12." if sentences else TEXT)
                self.assertEqual(spoken.kept("First part." if sentences else ""), kept)
                self.assertEqual(settled.events, [("heard", "First part." if sentences else TEXT, True)])
