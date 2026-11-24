# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The voice's lines, settled by the playout the edge after the output reports and the words the aggregator records."""

import json
import unittest

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.spoken import Line, LineKind, Outcome, SpokenLines

TEXT = "First part. Your order number is A B 12."


class Settled:
    def __init__(self):
        self.events: list[tuple] = []

    def heard(self, line):
        self.events.append(("heard", line.heard, line.outcome is Outcome.CUT))

    def not_heard(self, line, *, audio_exhausted=False):
        self.events.append(("not_heard",))

    def say_again(self, line):
        self.events.append(("again",))


def playing(*, text=TEXT, tells=(), started=True):
    """Return lines with ``text`` handed and started if ``started``, and their :class:`Settled`."""
    settled = Settled()
    spoken = SpokenLines(settled, observe=lambda *a, **k: None)
    spoken.handed(Line(LineKind.REPLY, text, tells=tells, number=1))
    if started:
        spoken.started(1)
    return spoken, settled


class SpokenLinesTests(unittest.TestCase):
    def test_a_cut_with_audio_but_no_recorded_words_settles_with_an_empty_prefix(self):
        spoken, settled = playing()
        spoken.cut(1, 101)
        spoken.cut_played(101, "")
        self.assertEqual(settled.events, [("heard", "", True)])

    def test_a_cut_settles_once_the_aggregator_reports_words(self):
        spoken, settled = playing(tells=("del_1",))
        spoken.cut(1, 101)
        self.assertEqual(settled.events, [])
        self.assertTrue(spoken.in_flight())
        evidence = json.loads(spoken.told_text("del_1"))[0]
        self.assertEqual((evidence["recorded_sentence_prefix"], evidence["state"]), ("", "playing"))
        self.assertNotIn("text", evidence)
        self.assertEqual(spoken.presentation(next(spoken.in_flight_lines()))["text"], TEXT)
        self.assertFalse(evidence["complete_conveyance"])
        spoken.cut_played(101, "First part.")
        self.assertEqual(settled.events, [("heard", "First part.", True)])
        self.assertFalse(spoken.in_flight())
        evidence = json.loads(spoken.told_text("del_1"))[0]
        self.assertEqual((evidence["recorded_sentence_prefix"], evidence["state"]), ("First part.", "cut"))
        self.assertNotIn("text", evidence)

    def test_a_cut_before_any_audio_played_settles_at_once(self):
        spoken, settled = playing(started=False)
        spoken.handed(Line(LineKind.REPLY, "Next line.", number=2))
        spoken.handed(Line(LineKind.REPLY, "Future line.", number=3))
        spoken.cut(1, 101, newest=2)
        self.assertEqual(settled.events, [("not_heard",), ("not_heard",)])
        spoken.cut_played(101, "")
        self.assertEqual(settled.events, [("not_heard",), ("not_heard",)])
        self.assertEqual(tuple(spoken.lines), (3,))

    def test_a_whole_line_keeps_the_aggregator_words(self):
        spoken, _ = playing(text="First part.\nYour order number is A B 12.", tells=("del_1",))
        spoken.ended(1)
        spoken.handed(Line(LineKind.REPLY, "Second line.", tells=("del_2",), number=2))
        spoken.started(2)
        spoken.ended(2)
        spoken.end_played(1, "First part. \nYour order number is A B 12.")
        self.assertEqual(
            json.loads(spoken.told_text("del_1"))[0]["recorded_sentence_prefix"],
            "First part. \nYour order number is A B 12.",
        )
        self.assertFalse(json.loads(spoken.told_text("del_2"))[0]["complete_conveyance"])
        spoken.end_played(2, "Second line.")
        self.assertTrue(json.loads(spoken.told_text("del_2"))[0]["complete_conveyance"])

    def test_a_silent_sentence_never_promotes_generated_text_to_playback(self):
        spoken, settled = playing(tells=("del_1",))
        stamp = spoken.context_stamp(("del_1",), ())
        spoken.ended(1)
        self.assertEqual(settled.events, [])
        self.assertEqual(spoken.context_stamp(("del_1",), ()), stamp)
        spoken.end_played(1, "")
        self.assertEqual(settled.events, [("heard", "", False)])
        self.assertNotEqual(spoken.context_stamp(("del_1",), ()), stamp)
        evidence = json.loads(spoken.told_text("del_1"))[0]
        self.assertEqual(evidence["recorded_sentence_prefix"], "")
        self.assertTrue(evidence["audio_started"])
        self.assertFalse(evidence["complete_conveyance"])


class ScopedSpokenLinesTests(unittest.TestCase):
    def make(self):
        return SpokenLines(Settled(), observe=lambda *a, **kw: None)

    def heard(self, spoken, line):
        spoken.handed(line)
        spoken.started(line.number)
        spoken.ended(line.number)
        spoken.end_played(line.number, line.text)

    def test_consumed_inflight_or_queued_progress_does_not_resurrect_when_audio_settles(self):
        for already_handed in (False, True):
            with self.subTest(already_handed=already_handed):
                spoken = self.make()
                line = Line(LineKind.STATUS, "Old A progress.", ("D3",), number=1, work_tells=frozenset(("A",)))
                spoken.register(line)
                if already_handed:
                    spoken.handed(line)
                    spoken.started(1)
                spoken.forget_work("A")
                if not already_handed:
                    line.number = 9
                    spoken.handed(line)
                    spoken.started(9)
                spoken.ended(line.number)
                spoken.end_played(line.number, line.text)
                self.assertEqual(line.heard, line.text)
                self.assertIsNone(spoken.context_text(work_refs=("A",)))
                self.assertIsNone(spoken.told_text("D3"))
                later = Line(LineKind.STATUS, "New A evidence.", number=10, work_tells=frozenset(("A",)))
                self.heard(spoken, later)
                self.assertEqual(
                    json.loads(spoken.context_text(work_refs=("A",)))[0]["recorded_sentence_prefix"], later.text
                )

    def test_host_binding_associates_actual_unscoped_leadin_without_retagging_or_splitting(self):
        spoken = self.make()
        request = ((1, "Check both."),)
        line = Line(LineKind.LEAD_IN, "Checking both.", ("D3",), tells=("D3",), number=1, handoff_request=request)
        spoken.handed(line)
        spoken.started(1)
        spoken.cut(1, 10)
        spoken.cut_played(10, "Checking")
        self.assertFalse(spoken.heard_handoff(line.text, "D3", None, request))
        line = Line(LineKind.LEAD_IN, line.text, ("D3",), tells=("D3",), number=2, handoff_request=request)
        self.heard(spoken, line)
        spoken.bind_work("A", "D3")
        self.assertTrue(spoken.heard_handoff(line.text, "D3", "A", request))
        self.assertFalse(spoken.heard_handoff(line.text, "D3", "A", ((2, "Check both."),)))
        self.assertFalse(spoken.heard_handoff(line.text + " One is ready.", "D3", "A", request))
        spoken.bind_work("B", "D3")
        self.assertFalse(spoken.heard_handoff(line.text, "D3", "A", request))
        self.assertEqual(line.work_tells, frozenset())
        self.assertEqual(line.delegation_ids, ("D3",))
        self.assertEqual(
            [row["recorded_sentence_prefix"] for row in json.loads(spoken.context_text(work_refs=("A", "B")))],
            ["Checking", line.text],
        )
        spoken.forget_work("A")
        self.assertEqual(json.loads(spoken.context_text(work_refs=("B",)))[-1]["recorded_sentence_prefix"], line.text)
        self.assertEqual(json.loads(spoken.told_text("D3"))[-1]["recorded_sentence_prefix"], line.text)
        spoken.forget_work("B")
        self.assertIsNone(spoken.told_text("D3"))
