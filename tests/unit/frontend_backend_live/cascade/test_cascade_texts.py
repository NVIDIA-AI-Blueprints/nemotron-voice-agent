# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The cascade reads its texts from the prompt set and its fixed lines from ``prompts.yaml``."""

import unittest
from dataclasses import replace

from examples.frontend_backend_live.cascade.lines import FixedLines
from examples.frontend_backend_live.cascade.worknote import WorkNote
from examples.frontend_backend_live.prompts import WorkTexts
from tests.unit.frontend_backend_live.cascade.doubles import prompt_set


def note(prompts):
    return WorkNote(prompts, tracks=True, clock=lambda: 0.0, observe=lambda kind, **fields: None)


def rendered(prompts) -> tuple[str, str]:
    """The note empty, then with a running entry (one call failed, one runs), a typed run and a result said next."""
    work = note(prompts)
    empty = work.line()
    work.confirm(work.reserve("Teas?"), "D1")
    work.running("D1")
    work.call_ended("D1", "resp_1", "mcp_1", "get_stock", "failed")
    work.call_started("D1", "resp_1", "mcp_2", "get_menu")
    work.progress("D1", "Checking the menu.")
    work.typed("T1")
    work.confirm(work.reserve("Cakes?"), "D2")
    work.answered("D2")
    return empty, work.line()


class PromptSetTextTests(unittest.TestCase):
    def test_the_work_note_reads_every_text_from_the_prompt_set(self):
        marked = replace(
            prompt_set(),
            work_in_progress="NOTE<{items}>",
            no_work_in_progress="NOTHING",
            answer_ready_label="READY",
            typed_request_label="TYPED",
            work_texts=WorkTexts(
                item="{number}|{state}|{request}|{update}",
                running="RUN {seconds}",
                ended=" ENDED {steps}",
                ended_call="{name}={status}",
                steps=" USING {steps}",
                waiting="WAIT",
                update="UPDATE {progress}",
            ),
        )
        first = "1|RUN 0 ENDED get stock=failed USING get menu|Teas?|UPDATE Checking the menu."
        self.assertEqual(rendered(marked), ("NOTHING", f"NOTE<{first}\n2|WAIT|TYPED|\n3|READY|Cakes?|>"))


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
