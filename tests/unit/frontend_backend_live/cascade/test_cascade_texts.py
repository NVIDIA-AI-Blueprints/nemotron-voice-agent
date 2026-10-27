# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The cascade reads its texts from the prompt set and its fixed lines from ``prompts.yaml``."""

import unittest
from dataclasses import replace

from examples.frontend_backend_live.cascade.lines import FixedLines
from examples.frontend_backend_live.cascade.worknote import WorkNote
from examples.frontend_backend_live.prompts import get_prompt_set
from tests.unit.frontend_backend_live.cascade.doubles import (
    ANSWER_READY_LABEL,
    NO_WORK_IN_PROGRESS,
    TYPED_REQUEST_LABEL,
    WORK_IN_PROGRESS,
    prompt_set,
)


def note(prompts):
    return WorkNote(prompts, tracks=True, clock=lambda: 0.0, observe=lambda kind, **fields: None)


def rendered(prompts) -> tuple[str, str]:
    """The note with nothing open, then with a running entry, a typed run and a result said next."""
    work = note(prompts)
    empty = work.line()
    work.confirm(work.reserve("Teas?"), "D1")
    work.running("D1")
    work.typed("T1")
    work.confirm(work.reserve("Cakes?"), "D2")
    work.answered("D2")
    return empty, work.line()


ITEMS = (
    '1. running for 0 s: "Teas?"',
    '2. waiting to start: "a typed request"',
    '3. answer ready, said next: "Cakes?"',
)
TODAY = (NO_WORK_IN_PROGRESS, WORK_IN_PROGRESS.format(items="\n".join(ITEMS)))


class PromptSetTextTests(unittest.TestCase):
    def test_the_work_note_reads_every_text_from_the_prompt_set(self):
        marked = replace(
            prompt_set(),
            work_in_progress="NOTE<{items}>",
            no_work_in_progress="NOTHING",
            answer_ready_label="READY",
            typed_request_label="TYPED",
        )
        self.assertEqual(
            rendered(marked),
            ("NOTHING", 'NOTE<1. running for 0 s: "Teas?"\n2. waiting to start: "TYPED"\n3. READY: "Cakes?">'),
        )

    def test_the_v1_and_v2_prompt_sets_carry_the_work_note_texts_unchanged(self):
        # PromptSet holds the work note's texts byte for byte (the doubles keep the pin).
        for version in ("v1", "v2"):
            with self.subTest(version):
                self.assertEqual(rendered(get_prompt_set(version)), TODAY)
        self.assertEqual((ANSWER_READY_LABEL, TYPED_REQUEST_LABEL), ("answer ready, said next", "a typed request"))


class LineTests(unittest.TestCase):
    def test_lines_refuse_unknown_and_missing_names(self):
        block = {name: "x" for name in FixedLines.__dataclass_fields__}
        self.assertEqual(FixedLines.parse(block).lost, "x")
        with self.assertRaisesRegex(ValueError, "unknown \\['extra'\\]"):
            FixedLines.parse({**block, "extra": "y"})
        with self.assertRaisesRegex(ValueError, "missing \\['lost'\\]"):
            FixedLines.parse({key: value for key, value in block.items() if key != "lost"})
        with self.assertRaisesRegex(ValueError, "must be text"):
            FixedLines.parse({**block, "lost": 3})
