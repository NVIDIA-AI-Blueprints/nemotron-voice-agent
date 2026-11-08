# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102
"""What an app function does, as the app's MCP annotations or the operator's lists declare it."""

import unittest

from examples.frontend_backend_live.delegation_core.config import ToolKindSettings
from examples.frontend_backend_live.delegation_core.im.tools import (
    DeclaredKind,
    KindSource,
    ToolKind,
    declared_tool_kind,
    is_declared_read,
)

NONE = ToolKindSettings()


def tool(name: str, **hints) -> dict:
    """A function tool named ``name``, with ``hints`` as its MCP annotations when given."""
    made = {"type": "function", "name": name, "parameters": {"type": "object", "properties": {}}}
    return {**made, "annotations": hints} if hints else made


READ = DeclaredKind(ToolKind.READ, KindSource.ANNOTATIONS)
WRITE = DeclaredKind(ToolKind.WRITE, KindSource.ANNOTATIONS)


class ToolKindTests(unittest.TestCase):
    def test_read_only_hint_true_is_a_read_whatever_the_name(self):
        for name in ("read_note", "save_note", "purge_notes"):
            with self.subTest(name=name):
                self.assertEqual(declared_tool_kind(tool(name, readOnlyHint=True), NONE), READ)

    def test_read_only_hint_false_or_a_destructive_hint_is_a_write(self):
        self.assertEqual(declared_tool_kind(tool("save_note", readOnlyHint=False), NONE), WRITE)
        self.assertEqual(declared_tool_kind(tool("save_note", destructiveHint=True), NONE), WRITE)

    def test_read_only_hint_true_wins_over_a_destructive_hint(self):
        self.assertEqual(declared_tool_kind(tool("read_note", readOnlyHint=True, destructiveHint=True), NONE), READ)

    def test_a_hint_that_is_not_a_boolean_is_no_hint(self):
        self.assertIsNone(declared_tool_kind(tool("save_note", readOnlyHint="true"), NONE))
        self.assertIsNone(declared_tool_kind(tool("save_note", readOnlyHint=1, destructiveHint="yes"), NONE))

    def test_destructive_hint_false_on_a_write_is_reversible(self):
        kind = declared_tool_kind(tool("tag_note", readOnlyHint=False, destructiveHint=False), NONE)
        self.assertEqual(kind, DeclaredKind(ToolKind.WRITE, KindSource.ANNOTATIONS, reversible=True))

    def test_an_operator_pattern_makes_an_undeclared_tool_a_read_and_matches_case(self):
        listed = ToolKindSettings(read_tools=("list_*",))
        self.assertEqual(
            declared_tool_kind(tool("list_notes"), listed), DeclaredKind(ToolKind.READ, KindSource.OPERATOR)
        )
        self.assertIsNone(declared_tool_kind(tool("List_notes"), listed))

    def test_a_write_tools_entry_makes_an_undeclared_tool_a_write(self):
        listed = ToolKindSettings(write_tools=("purge_notes",))
        expected = DeclaredKind(ToolKind.WRITE, KindSource.OPERATOR)
        self.assertEqual(declared_tool_kind(tool("purge_notes"), listed), expected)

    def test_a_tool_matched_by_both_lists_is_a_write(self):
        listed = ToolKindSettings(read_tools=("*_notes",), write_tools=("purge_*",))
        self.assertEqual(declared_tool_kind(tool("purge_notes"), listed).kind, ToolKind.WRITE)

    def test_annotations_win_over_the_operator_lists(self):
        listed = ToolKindSettings(write_tools=("read_note",))
        self.assertEqual(declared_tool_kind(tool("read_note", readOnlyHint=True), listed), READ)

    def test_annotations_with_neither_hint_leave_the_tool_to_the_operator_lists(self):
        listed = ToolKindSettings(read_tools=("read_note",))
        titled = tool("read_note", title="Read a note")
        self.assertEqual(declared_tool_kind(titled, listed), DeclaredKind(ToolKind.READ, KindSource.OPERATOR))
        self.assertIsNone(declared_tool_kind(titled, NONE))

    def test_a_tool_nobody_declares_is_undeclared(self):
        self.assertIsNone(declared_tool_kind(tool("plain_tool"), NONE))

    def test_is_declared_read_is_true_only_for_a_declared_read(self):
        listed = ToolKindSettings(read_tools=("list_*",))
        self.assertTrue(is_declared_read(tool("read_note", readOnlyHint=True), NONE))
        self.assertTrue(is_declared_read(tool("list_notes"), listed))
        self.assertFalse(is_declared_read(tool("save_note", readOnlyHint=False), NONE))
        self.assertFalse(is_declared_read(tool("tag_note", readOnlyHint=False, destructiveHint=False), NONE))
        self.assertFalse(is_declared_read(tool("plain_tool"), NONE))


if __name__ == "__main__":
    unittest.main()
