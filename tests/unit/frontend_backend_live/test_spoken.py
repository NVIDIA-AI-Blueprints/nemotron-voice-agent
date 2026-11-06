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
    """Lines that settle into a :class:`Settled`, with ``text`` handed and reached, and started if ``started``."""
    settled = Settled()
    spoken = SpokenLines(settled, reports_sentences=sentences, observe=lambda *a, **k: None)
    spoken.handed(Line(LineKind.REPLY, text, tells=tells, number=1))
    spoken.reached(1)
    if started:
        spoken.started(1)
    return spoken, settled


class SpokenLinesTests(unittest.TestCase):
    def test_a_cut_line_keeps_the_words_the_aggregator_recorded_and_a_stitched_one_is_heard_whole(self):
        for sentences, played, kept in (
            (True, "First part.", "First part."),
            (False, "", TEXT),  # a stitched line: its text comes only at its end, so it is heard whole
        ):
            with self.subTest(sentences=sentences):
                spoken, settled = playing(sentences=sentences)
                spoken.cut(1)
                self.assertEqual(spoken.cut_played(played), kept)
                self.assertEqual(settled.events, [("heard", kept, True)])

    def test_a_line_cut_in_its_first_sentence_keeps_nothing_and_is_not_heard(self):
        spoken, settled = playing()
        spoken.cut(1)  # its audio played, but no sentence of it finished
        self.assertEqual(spoken.cut_played(""), "")
        self.assertEqual(settled.events, [("not_heard",)])

    def test_a_cut_while_a_line_played_settles_with_the_words_the_aggregator_reports(self):
        spoken, settled = playing(tells=("del_1",))
        spoken.cut(1)
        self.assertEqual(settled.events, [])  # until the aggregator reports what played, the line is in flight, whole
        self.assertTrue(spoken.in_flight())
        self.assertEqual(spoken.told_text("del_1"), TEXT)
        spoken.cut_played("First part.")
        self.assertEqual(settled.events, [("heard", "First part.", True)])
        self.assertEqual((spoken.in_flight(), spoken.told_text("del_1")), (False, "First part."))

    def test_a_cut_before_any_audio_played_settles_at_once(self):
        spoken, settled = playing(started=False)
        spoken.handed(Line(LineKind.REPLY, "Next line.", number=2))  # dropped by the same interruption
        spoken.interrupted(2)
        spoken.cut(1)
        self.assertEqual(settled.events, [("not_heard",), ("not_heard",)])
        self.assertEqual(spoken.cut_played(""), "")  # the aggregator's report of that turn changes nothing

    def test_a_cut_the_aggregator_never_reports_settles_with_nothing_played_once_a_later_response_heads(self):
        spoken, settled = playing()
        spoken.cut(1)
        spoken.reached(2)
        self.assertEqual(settled.events, [("not_heard",)])
        self.assertFalse(spoken.in_flight())

    def test_a_line_a_sentence_of_which_got_no_audio_tells_only_the_words_the_aggregator_reports(self):
        spoken, settled = playing(tells=("del_1",))
        spoken.dropped(1)  # its first sentence got no audio
        spoken.ended(1)
        self.assertEqual(settled.events, [("heard", TEXT, False)])  # heard: never said again
        self.assertEqual(spoken.told_text("del_1"), TEXT)  # whole until the aggregator reports what played
        line = spoken.end_played("Your order number is A B 12.")
        self.assertEqual((line.number, spoken.told_text("del_1")), (1, "Your order number is A B 12."))

    def test_words_reported_once_a_later_response_heads_the_playout_leave_the_line_that_ended_whole(self):
        spoken, _ = playing(tells=("del_1",))
        spoken.dropped(1)
        spoken.ended(1)
        spoken.reached(2)
        self.assertIsNone(spoken.end_played("Your order number is A B 12."))
        self.assertEqual(spoken.told_text("del_1"), TEXT)

    def test_a_line_that_played_every_sentence_keeps_its_words_however_the_aggregator_joins_them(self):
        spoken, _ = playing(text="First part.\nYour order number is A B 12.", tells=("del_1",))
        spoken.ended(1)
        self.assertIsNone(spoken.end_played("First part. \nYour order number is A B 12."))  # the aggregator's join
        self.assertEqual(spoken.told_text("del_1"), "First part.\nYour order number is A B 12.")


if __name__ == "__main__":
    unittest.main()
