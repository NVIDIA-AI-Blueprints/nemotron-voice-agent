# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Normalization token rules: digit words, compounds, fillers, case, rulesets."""

from __future__ import annotations

import unittest

from examples.frontend_backend_verdict.voice.normalization.rules import (
    EN,
    apply_case,
    digits_from_words,
    get_ruleset,
    join_part,
    spelled_out,
    tokenize,
)


def digits(text: str, *, compound: bool = True) -> str | None:
    return digits_from_words(text.lower().split(), EN, compound=compound)


class DigitWordTests(unittest.TestCase):
    def test_digit_words_numerals_and_repeats(self) -> None:
        self.assertEqual(digits("four three nine seven"), "4397")
        self.assertEqual(digits("double four nine seven"), "4497")
        self.assertEqual(digits("triple 5 one"), "5551")
        self.assertEqual(digits("4 three 9 seven"), "4397")

    def test_oh_only_inside_a_run(self) -> None:
        self.assertEqual(digits("four oh seven"), "407")
        self.assertIsNone(digits("oh"))

    def test_compound_numbers(self) -> None:
        self.assertEqual(digits("forty seven thirty nine"), "4739")
        self.assertEqual(digits("nineteen ninety"), "1990")
        self.assertEqual(digits("forty-three"), "43")
        self.assertIsNone(digits("forty seven", compound=False))

    def test_a_non_number_word_rejects_the_run(self) -> None:
        self.assertIsNone(digits("four three nine se"))
        self.assertIsNone(digits("two passengers"))


class TokenTests(unittest.TestCase):
    def test_edge_punctuation_is_outside_the_core(self) -> None:
        text = "Kim, underscore. (four) ."
        tokens = tokenize(text)
        self.assertEqual([t.core for t in tokens], ["Kim", "underscore", "four", ""])
        self.assertEqual(text[tokens[0].start : tokens[0].end], "Kim")

    def test_join_part_drops_fillers_and_converts_all_number_parts(self) -> None:
        self.assertEqual(join_part(tokenize("Em uh ma"), EN, number_words=True, compound=True), "Emma")
        self.assertEqual(join_part(tokenize("nine nine"), EN, number_words=True, compound=True), "99")
        self.assertEqual(join_part(tokenize("nine nine"), EN, number_words=False, compound=True), "ninenine")

    def test_case_and_spelled_out(self) -> None:
        self.assertEqual(apply_case("Jordan_Lee", "lower"), "jordan_lee")
        self.assertEqual(apply_case("Jordan_Lee", "upper"), "JORDAN_LEE")
        self.assertEqual(apply_case("Jordan_Lee", "keep"), "Jordan_Lee")
        self.assertEqual(spelled_out("mi_4", {"underscore": "_"}), "m, i, underscore, 4")


class RulesetTests(unittest.TestCase):
    def test_en_is_the_shipped_ruleset(self) -> None:
        self.assertIs(get_ruleset("en"), EN)
        with self.assertRaises(KeyError):
            get_ruleset("xx")

    def test_extra_words_extend_the_ruleset(self) -> None:
        extended = EN.extended(fillers=["Like"], stop_words=["Please"])
        self.assertIn("like", extended.fillers)
        self.assertIn("please", extended.stop_words)
        self.assertNotIn("like", EN.fillers)

    def test_letter_names_are_not_mapped(self) -> None:
        self.assertEqual(join_part(tokenize("em ma"), EN, number_words=True, compound=True), "emma")
