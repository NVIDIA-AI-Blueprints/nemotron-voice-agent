# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102
"""The delegation contract's shared text rule: sentences and questions, read by Unicode properties in any script."""

import unittest

from examples.shared.delegation_contract import ends_with_question, question_marks, sentence_spans, sentences


class TextRuleTests(unittest.TestCase):
    def test_question_marks_count_closing_marks_in_any_script(self):
        for text in ("?", "？", "؟", "⁉"):
            with self.subTest(text=text):
                self.assertEqual(question_marks(text), 1)
        self.assertEqual(question_marks("¿Listo?"), 1)
        self.assertEqual(question_marks(";"), 0)

    def test_a_sentence_asks_past_closing_quotes_emphasis_and_symbols(self):
        for sentence in ('He said "ready?"', "**Shall I book it?**", "Is that okay?!", "Ready? 🙂"):
            with self.subTest(sentence=sentence):
                self.assertTrue(ends_with_question(sentence))
        self.assertFalse(ends_with_question("Please confirm the address."))
        self.assertFalse(ends_with_question(""))

    def test_the_default_rule_splits_spaced_and_unspaced_scripts(self):
        for text, expected in (
            ("It is 2.5 units. Is that fine?", ["It is 2.5 units.", "Is that fine?"]),
            ("準備はいいですか？はい。", ["準備はいいですか？", "はい。"]),
            ("¿Está listo? Sí.", ["¿Está listo?", "Sí."]),
        ):
            with self.subTest(text=text):
                self.assertEqual(sentences(text), expected)
                spans = sentence_spans(text)
                self.assertEqual("".join(text[start:end] for start, end in spans), text)
                self.assertEqual([end for _, end in spans[:-1]], [start for start, _ in spans[1:]])

    def test_a_given_end_rule_is_used_in_place_of_the_default(self):
        self.assertEqual(sentences("One. Two? Three!", ends=lambda rest: 0), ["One. Two? Three!"])


if __name__ == "__main__":
    unittest.main()
