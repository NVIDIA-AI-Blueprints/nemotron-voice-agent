# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The voice's lines, settled by the playout the edge after the output reports and the words the aggregator records."""

import unittest

from examples.frontend_backend_live.spoken import Line, LineKind, SpokenLines

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


def playing(*, text=TEXT, sentences=True, tells=(), started=True):
    """Return lines with ``text`` handed, reached and started if ``started``, and their :class:`Settled`."""
    settled = Settled()
    spoken = SpokenLines(settled, reports_sentences=sentences, observe=lambda *a, **k: None)
    spoken.handed(Line(LineKind.REPLY, text, tells=tells, number=1))
    spoken.reached(1)
    if started:
        spoken.started(1)
    return spoken, settled


class SpokenLinesTests(unittest.TestCase):
    def test_a_cut_line_keeps_the_words_that_played(self):
        # A stitched line's text comes only at its end, so it is heard whole.
        for sentences, played, kept in ((True, "First part.", "First part."), (False, "", TEXT)):
            with self.subTest(sentences=sentences):
                spoken, settled = playing(sentences=sentences)
                spoken.cut(1)
                self.assertEqual(spoken.cut_played(played), kept)
                self.assertEqual(settled.events, [("heard", kept, True)])

    def test_a_cut_settles_once_the_aggregator_reports_words(self):
        spoken, settled = playing(tells=("del_1",))
        spoken.cut(1)
        self.assertEqual(settled.events, [])
        self.assertEqual((spoken.in_flight(), spoken.told_text("del_1")), (True, TEXT))
        spoken.cut_played("First part.")
        self.assertEqual(settled.events, [("heard", "First part.", True)])
        self.assertEqual((spoken.in_flight(), spoken.told_text("del_1")), (False, "First part."))

    def test_a_cut_before_any_audio_played_settles_at_once(self):
        spoken, settled = playing(started=False)
        spoken.handed(Line(LineKind.REPLY, "Next line.", number=2))
        spoken.interrupted(2)
        spoken.cut(1)
        self.assertEqual(settled.events, [("not_heard",), ("not_heard",)])
        self.assertEqual(spoken.cut_played(""), "")

    def test_an_unreported_cut_settles_once_a_later_response_heads(self):
        spoken, settled = playing()
        spoken.cut(1)
        spoken.reached(2)
        self.assertEqual((settled.events, spoken.in_flight()), ([("not_heard",)], False))

    def test_a_whole_line_keeps_its_words_however_they_are_joined(self):
        spoken, _ = playing(text="First part.\nYour order number is A B 12.", tells=("del_1",))
        spoken.ended(1)
        self.assertIsNone(spoken.end_played("First part. \nYour order number is A B 12."))
        self.assertEqual(spoken.told_text("del_1"), "First part.\nYour order number is A B 12.")
