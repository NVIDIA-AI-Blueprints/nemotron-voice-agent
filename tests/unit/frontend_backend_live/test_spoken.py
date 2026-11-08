# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The voice's lines, settled by the playout the edge after the output reports and the words the aggregator records."""

import unittest

from examples.frontend_backend_live import spoken as said
from examples.frontend_backend_live.spoken import Line, LineKind, Outcome, SpokenLines, unheard_rest

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


class Order:
    """A ``Settled`` that notes the order in which lines settle not heard."""

    def __init__(self):
        self.unheard: list[int] = []

    def heard(self, line, cut):
        pass

    def not_heard(self, line):
        self.unheard.append(line.number)

    def say_again(self, line):
        pass


class SettleOrderTests(unittest.TestCase):
    def cut_two(self, requeued_as_said: bool) -> list[int]:
        order = Order()
        spoken = SpokenLines(order, requeued_as_said=requeued_as_said, observe=lambda *a, **k: None)
        for number in (1, 2):
            spoken.handed(Line(LineKind.RESULT, f"Line {number}.", number=number))
        spoken.interrupted(2)
        spoken.cut(None)  # nothing played: both settle not heard now
        return order.unheard

    def test_lines_one_cut_drops_settle_in_the_order_they_were_handed(self):
        # Newest first, so each line's items, put back at the front of what is owed, keep the order they were handed.
        self.assertEqual(self.cut_two(True), [2, 1])

    def test_settle_order_is_todays_with_requeued_as_said_off(self):
        self.assertEqual(self.cut_two(False), [1, 2])


class UnsettledTests(unittest.TestCase):
    def lines(self, *, clear_ended_on_cut=False):
        settled = Settled()
        spoken = SpokenLines(settled, clear_ended_on_cut=clear_ended_on_cut, observe=lambda *a, **k: None)
        return spoken, settled

    def test_unsettled_lists_a_cuts_lines_the_lines_in_flight_and_a_line_waiting_for_its_words(self):
        spoken, _ = self.lines()
        first, second = Line(LineKind.RESULT, "A. B?", number=1), Line(LineKind.RESULT, "C.", number=2)
        spoken.handed(first)
        spoken.handed(second)
        spoken.reached(1)
        spoken.started(1)
        spoken.interrupted(1)
        spoken.cut(1)  # the first line played: the cut waits for what played; the second was handed after it passed
        third = Line(LineKind.RESULT, "D. E?", number=3)
        spoken.handed(third)
        self.assertEqual(spoken.unsettled(), (first, second, third))
        spoken.cut_played("A.")
        spoken.reached(3)
        spoken.started(3)
        spoken.dropped(3)
        spoken.ended(3)  # a sentence got no audio: it waits for the words the aggregator recorded
        self.assertEqual((spoken.unsettled(), first.outcome), ((second, third), Outcome.CUT))

    def ended_waiting(self, clear_ended_on_cut):
        spoken, _ = self.lines(clear_ended_on_cut=clear_ended_on_cut)
        waiting = Line(LineKind.RESULT, "D. E?", number=1)
        spoken.handed(waiting)
        spoken.reached(1)
        spoken.started(1)
        spoken.dropped(1)
        spoken.ended(1)
        spoken.handed(Line(LineKind.REPLY, "Sure.", number=2))
        spoken.interrupted(2)
        spoken.cut(None)  # an interruption: the waiting line's words never come
        return spoken, waiting

    def test_a_cut_settles_a_line_waiting_for_its_words_when_clear_ended_on_cut_is_on(self):
        spoken, _ = self.ended_waiting(True)
        self.assertEqual(spoken.unsettled(), ())

    def test_with_clear_ended_on_cut_off_a_cut_leaves_the_waiting_line_as_today(self):
        spoken, waiting = self.ended_waiting(False)
        self.assertEqual(spoken.unsettled(), (waiting,))


class SpelledRunTests(unittest.TestCase):
    def test_single_characters_in_a_row_join_as_heard(self):
        self.assertEqual(said.spelled_runs("my name is J, O, H, N."), ["JOHN"])
        self.assertEqual(said.spelled_runs("last name e L L I S"), ["eLLIS"])

    def test_ordinary_speech_forms_no_run(self):
        for text in ("I need a new one", "plan A is fine", "June 15 2024"):
            with self.subTest(text=text):
                self.assertEqual(said.spelled_runs(text), [])

    def test_scripts_without_letter_case_form_no_run(self):
        for text in ("我 想 要 咖 啡", "コ ー ヒ ー", "ก า แ ฟ"):
            with self.subTest(text=text):
                self.assertEqual(said.spelled_runs(text), [])
        self.assertEqual(said.spelled_runs("我 想 1 2 3 咖 啡"), ["123"])

    def test_any_text_is_read_without_raising(self):
        for text in ("", "!!! ??? ...", "word " * 10_000):
            self.assertEqual(said.spelled_runs(text), [])
            self.assertFalse(said.ends_spelled(text))

    def test_ends_spelled(self):
        self.assertTrue(said.ends_spelled("J O H"))
        self.assertFalse(said.ends_spelled("thank you"))
        self.assertFalse(said.ends_spelled("my code is J"))

    def test_short_words_in_french_and_portuguese_are_pinned(self):
        # Pins today's behaviour: an additive note, no hold.
        for text, run in (("il y a un problème avec ma commande", "ya"), ("quero o café e o bolo de hoje", "eo")):
            with self.subTest(text=text):
                self.assertEqual(said.spelled_runs(text), [run])
                self.assertFalse(said.ends_spelled(text))
        self.assertEqual(said.spelled_runs("c'est bon") + said.spelled_runs("obrigado"), [])


class SentencesTests(unittest.TestCase):
    def test_sentences_split_by_the_speech_services_own_rule(self):
        text = "Your table is ready. It is by the window! Shall I seat you?"
        self.assertEqual(said.sentences(text), ["Your table is ready.", "It is by the window!", "Shall I seat you?"])
        self.assertEqual(said.sentences("Dr. Lee is in."), ["Dr. Lee is in."])

    def test_sentences_follow_the_speech_services_language(self):
        self.assertEqual(said.sentences("¿Está listo? Sí.", language="es"), ["¿Está listo?", "Sí."])
        self.assertEqual(said.sentences("準備はいいですか？はい。", language="ja"), ["準備はいいですか？", "はい。"])

    def test_a_last_part_with_no_sentence_end_is_a_sentence(self):
        self.assertEqual(said.sentences("Ready now. And the"), ["Ready now.", "And the"])


def line(text: str, heard: str) -> Line:
    return Line(LineKind.RESULT, text, heard=heard)


class UnheardRestTests(unittest.TestCase):
    THREE = "The gym opens at six. The pool opens at seven. Do you want a towel?"

    def test_the_unheard_rest_is_the_sentences_after_what_played(self):
        rest = unheard_rest(line(self.THREE, "The gym opens at six."))
        self.assertEqual(rest, "The pool opens at seven. Do you want a towel?")

    def test_the_sentence_playing_at_the_cut_counts_as_unheard(self):
        self.assertEqual(unheard_rest(line("Bike 4 is free. Shall I book it?", "Bike 4 is free.")), "Shall I book it?")

    def test_a_sentence_missing_from_the_middle_is_found(self):
        heard = "The gym opens at six. Do you want a towel?"
        self.assertEqual(unheard_rest(line(self.THREE, heard)), "The pool opens at seven.")

    def test_markup_and_case_never_tell_sentences_apart(self):
        self.assertEqual(unheard_rest(line("**Bike** 4 is free.", "bike 4 is free")), "")

    def test_a_line_heard_whole_has_no_rest(self):
        self.assertEqual(unheard_rest(line(self.THREE, self.THREE)), "")
        spoken, _ = playing(text=self.THREE, sentences=False)  # stitched: a line cut while it plays is heard whole
        stitched = spoken.lines[1]
        spoken.cut(1)
        spoken.cut_played("")
        self.assertEqual(unheard_rest(stitched), "")
