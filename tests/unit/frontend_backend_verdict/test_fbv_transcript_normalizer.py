# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""The transcript hook: anchored spans only, real 800 ms-run transcripts as a corpus."""

from __future__ import annotations

import unittest

from examples.frontend_backend_verdict.voice.normalization.transcript import (
    SpelledRunSettings,
    TranscriptNormalizer,
    TranscriptSettings,
)

LOWER = TranscriptNormalizer(TranscriptSettings(enabled=True, case="lower"))

#: Modelled on ASR finals of a voice evaluation run (token structure kept, values invented) and their written form.
CORPUS = (
    ("my user ID is Jordan underscore Lee underscore eight two six four.", "my user ID is jordan_lee_8264."),
    ("J O underscore L ee underscore three one eight five reason", "jo_lee_3185 reason"),
    ("Underscore three one eight five .", "_3185 ."),
    (
        "Yeah sorry user ID is JORDAN underscore LEE underscore eight two six four and the reservation code is Q"
        " four XB two T",
        "Yeah sorry user ID is jordan_lee_8264 and the reservation code is Q four XB two T",
    ),
    ("Q U E N T I N underscore M orro underscore two zero four eight", "quentin_morro_2048"),
    ("Sure it's Q U E N T I N underscore Mor r o underscore two zero four eight .", "Sure it's quentin_morro_2048 ."),
    ("Yeah , it's P E N E L O underscore V A R D A M", "Yeah , it's penelo_vardam"),
    ("Yeah T O B I A S underscore Wen Z O underscore five one seven three .", "Yeah tobias_wenzo_5173 ."),
    (
        "Yeah , it's R am os underscore F E R R I S underscore six zero two nine .",
        "Yeah , it's ramos_ferris_6029 .",
    ),
    ("A a lto underscore a h L E N underscore four four one seven", "aalto_ahlen_4417"),
    ("It's the same I V O underscore Q U I N N underscore nine three five two", "It's the same ivo_quinn_9352"),
    ("Underscore two zero four eight . Did you get that ?", "_2048 . Did you get that ?"),
    ("Yeah O D E T T E underscore", "Yeah odette_"),
    ("D E V R A underscore L ee underscore three one eight five", "devra_lee_3185"),
    ("Whole thing K A I underscore V.E.N.", "Whole thing kai_ven."),
    ("Yeah A A L T O underscore A.H.", "Yeah aalto_ah."),  # dotted letters are spelled, not the filler "ah"
)


class TranscriptNormalizerTests(unittest.TestCase):
    def test_corpus(self) -> None:
        for raw, expected in CORPUS:
            with self.subTest(raw=raw):
                self.assertEqual(LOWER.normalize(raw).text, expected)

    def test_idempotent(self) -> None:
        for raw, _ in CORPUS:
            once = LOWER.normalize(raw).text
            self.assertEqual(LOWER.normalize(once).text, once)
            self.assertFalse(LOWER.normalize(once).changed)

    def test_no_anchor_no_change(self) -> None:
        for text in ("I need two passengers", "Book four three nine seven", "underscore", "Underscore."):
            result = LOWER.normalize(text)
            self.assertEqual(result.text, text)
            self.assertFalse(result.changed)

    def test_spans_record_spoken_and_written_forms(self) -> None:
        result = LOWER.normalize("It's Jordan underscore Lee underscore double four nine seven, thanks.")
        self.assertEqual(result.text, "It's jordan_lee_4497, thanks.")
        (span,) = result.spans
        self.assertEqual(
            (span.spoken, span.written), ("Jordan underscore Lee underscore double four nine seven", "jordan_lee_4497")
        )

    def test_stop_words_bound_the_left_part(self) -> None:
        self.assertEqual(LOWER.normalize("my name is Em ma underscore Kim").text, "my name is emma_kim")

    def test_too_many_words_between_anchors_make_separate_spans(self) -> None:
        normalizer = TranscriptNormalizer(TranscriptSettings(enabled=True, case="lower", max_part_tokens=2))
        text = "Jordan underscore then we talked about many things underscore four"
        self.assertEqual(normalizer.normalize(text).text, "jordan_then we talked about many things_4")

    def test_case_keep_and_number_words_off(self) -> None:
        keep = TranscriptNormalizer(TranscriptSettings(enabled=True))
        self.assertEqual(keep.normalize("Jordan underscore Lee underscore four").text, "Jordan_Lee_4")
        words = TranscriptNormalizer(TranscriptSettings(enabled=True, number_words=False))
        self.assertEqual(words.normalize("Jordan underscore Lee underscore four").text, "Jordan_Lee_four")

    def test_custom_separator_words(self) -> None:
        normalizer = TranscriptNormalizer(TranscriptSettings(enabled=True, separator_words={"dash": "-"}))
        self.assertEqual(normalizer.normalize("code A B dash four two").text, "code AB-42")
        self.assertEqual(normalizer.normalize("Jordan underscore Lee").text, "Jordan underscore Lee")


SPELLED = TranscriptNormalizer(TranscriptSettings(enabled=True, case="lower", spelled_runs=SpelledRunSettings(True)))


class SpelledRunTests(unittest.TestCase):
    def test_runs_are_joined_with_the_asr_case_and_stop_at_words(self) -> None:
        cases = {
            "X, B, seven, T, P, two and Q, U, E, N, five, R": "XB7TP2 and QUEN5R",
            "Q U I N N": "QUINN",
            "my name is Q, uh, U I N N.": "my name is QUINN.",
            "Z P four K nine one": "ZP4K91",
            "from J F K to L A X": "from JFK to LAX",
        }
        for spoken, written in cases.items():
            with self.subTest(spoken=spoken):
                self.assertEqual(SPELLED.normalize(spoken).text, written)

    def test_unchanged_without_a_letter_or_below_min_tokens(self) -> None:
        for text in ("five zero zero", "one four four.", "I want a flight", "plan A or B", "A B"):
            with self.subTest(text=text):
                self.assertEqual(SPELLED.normalize(text).text, text)

    def test_anchored_spans_win_and_are_unchanged(self) -> None:
        for spoken, written in CORPUS:
            if not any(len(token.strip(".,")) == 1 for token in written.split()):
                with self.subTest(spoken=spoken):
                    self.assertEqual(SPELLED.normalize(spoken).text, written)
        self.assertEqual(SPELLED.normalize("Aalto underscore A H underscore one two three four").text, "aalto_ah_1234")

    def test_off_is_byte_identical(self) -> None:
        for spoken, _ in CORPUS:
            with self.subTest(spoken=spoken):
                self.assertEqual(
                    TranscriptNormalizer(TranscriptSettings(enabled=True, case="lower")).normalize(spoken),
                    LOWER.normalize(spoken),
                )
        self.assertEqual(LOWER.normalize("R O S S I").text, "R O S S I")
