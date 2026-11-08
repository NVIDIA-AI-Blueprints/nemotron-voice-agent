# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102
"""A request's extra parts: the work records block, the questions block and the closing line."""

import json
import unittest

from examples.frontend_backend_live.delegation_core.im.input import QUESTION_TEXT_CHARS, questions_part, records_parts
from tests.unit.delegation_core.helpers import TEXTS

RECORD = {"work": "W1", "state": "answered"}


class RecordsPartsTests(unittest.TestCase):
    def test_the_questions_block_follows_the_records_block_and_is_cut_to_120_characters(self):
        long = "Is the address on file still correct, " * 6 + "?"
        block = questions_part(TEXTS.questions_block, [{"question": long, "asked": 2, "heard_in_full": 1}])
        parts = records_parts(TEXTS.records_block, TEXTS.closing_line, [RECORD], block)
        self.assertEqual(len(parts), 3)
        self.assertTrue(parts[0].startswith(TEXTS.records_block.split("{")[0]))
        self.assertEqual(parts[1:], (block, TEXTS.closing_line))
        line = json.loads(block.splitlines()[-1])
        self.assertEqual((len(line["question"]), line["asked"], line["heard_in_full"]), (QUESTION_TEXT_CHARS, 2, 1))

    def test_shown_zero_or_no_template_gives_no_block(self):
        self.assertEqual(questions_part(TEXTS.questions_block, []), "")
        self.assertEqual(questions_part(None, [{"question": "Anything else?", "asked": 1}]), "")
        parts = records_parts(TEXTS.records_block, TEXTS.closing_line, [RECORD], "")
        self.assertEqual(len(parts), 2)

    def test_no_record_gives_no_parts(self):
        self.assertEqual(records_parts(TEXTS.records_block, TEXTS.closing_line, [], "a block"), ())


if __name__ == "__main__":
    unittest.main()
