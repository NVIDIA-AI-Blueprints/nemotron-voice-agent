# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102
"""The example's ``prompts.yaml`` as its reader reads it: a named prompt by group and name, the lines, the texts."""

import os
import re
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml

from examples.frontend_backend_live import prompts
from examples.frontend_backend_live.prompts import load_prompts

PROMPTS_PATH = Path(prompts.__file__).with_name("prompts.yaml")
GROUPS = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))["agent_prompts"]
LOADED = load_prompts()


class PromptFileTests(unittest.TestCase):
    def load(self, text: str):
        """Load the prompts of a server prompt file holding ``text``, in place of the example's own file."""
        folder = self.enterContext(TemporaryDirectory())
        path = Path(folder) / "other.yaml"
        path.write_text(text, encoding="utf-8")
        with patch.dict(os.environ, {"PROMPT_FILE_PATH": str(path)}):
            return load_prompts()

    def test_a_named_prompt_is_read_stripped_or_refused(self):
        self.assertEqual(LOADED.text("Channel", "cafe"), GROUPS["Channel"]["cafe"]["content"].strip())
        where = re.escape(str(PROMPTS_PATH))
        unknown = rf"agent_prompts\.Backend\.kiosk was not found in {where}; known: cafe, generic"
        with self.assertRaisesRegex(KeyError, unknown):
            LOADED.text("Backend", "kiosk")
        blank = replace(LOADED, agent_prompts={"Channel": {"cafe": {"content": " \n"}}})
        with self.assertRaisesRegex(KeyError, rf"agent_prompts\.Channel\.cafe in {where} has no text"):
            blank.text("Channel", "cafe")

    def test_the_lines_are_every_line_the_voice_says(self):
        lines = GROUPS["Lines"]
        for name, (given, message) in {
            "an unknown line": (
                {**lines, "farewell": {"content": "Bye."}},
                "farewell\n  Extra inputs are not permitted",
            ),
            "a missing line": ({k: v for k, v in lines.items() if k != "holding"}, "holding\n  Field required"),
            "an entry above its line": (
                {"greeting.extra": {"content": "Hi."}, **lines},
                "the entry greeting names a place that another text holds",
            ),
        }.items():
            refused = rf"agent_prompts\.Lines in .*: (?s:.*){message}"
            with self.subTest(name), self.assertRaisesRegex(ValueError, refused):
                self.load(yaml.safe_dump({"agent_prompts": {**GROUPS, "Lines": given}}, sort_keys=False))
        silent = {**lines, "holding": {"content": ""}}
        self.assertEqual(self.load(yaml.safe_dump({"agent_prompts": {**GROUPS, "Lines": silent}})).lines.holding, "")

    def test_a_file_may_omit_only_the_tools_style_texts_and_add_none(self):
        tools = ["tool_frame_before", "tool_routing", "tool_rules", "tools.delegate", "tools.wait"]
        json_only = {name: entry for name, entry in GROUPS["Cascade"].items() if name not in tools}
        cascade = self.load(yaml.safe_dump({"agent_prompts": {**GROUPS, "Cascade": json_only}})).cascade
        self.assertEqual((cascade.missing(tools), cascade.routing), (tools, LOADED.cascade.routing))
        self.assertEqual(LOADED.cascade.missing(tools), [])
        entry = {"content": "Text."}
        for name, (given, message) in {
            "a missing entry": ({k: v for k, v in json_only.items() if k != "routing"}, "routing\n  Field required"),
            "a missing nested entry": (
                {k: v for k, v in json_only.items() if k != "work_texts.item"},
                "work_texts.item\n  Field required",
            ),
            "an unknown entry": ({**json_only, "farewell": entry}, "farewell\n  Extra inputs are not permitted"),
            "an unknown nested entry": (
                {**json_only, "work_texts.extra": entry, "tools.extra": entry},
                "work_texts.extra\n  Extra inputs(?s:.*)tools.extra\n  Extra inputs are not permitted",
            ),
            "an entry under a text": (
                {**json_only, "routing.extra": entry},
                "the entry routing.extra names a place that another text holds",
            ),
            "an entry above its text": (
                {"routing.extra": entry, **json_only},
                "the entry routing names a place that another text holds",
            ),
        }.items():
            refused = rf"agent_prompts\.Cascade in .*: (?s:.*){message}"
            with self.subTest(name), self.assertRaisesRegex(ValueError, refused):
                self.load(yaml.safe_dump({"agent_prompts": {**GROUPS, "Cascade": given}}, sort_keys=False))

    def test_a_missing_or_malformed_file_is_refused(self):
        with TemporaryDirectory() as folder:
            missing, malformed, listed = (Path(folder) / name for name in ("missing.yaml", "bad.yaml", "list.yaml"))
            malformed.write_text("agent_prompts: [\n", encoding="utf-8")
            listed.write_text("- agent_prompts\n", encoding="utf-8")
            for path, cause in ((missing, FileNotFoundError), (malformed, yaml.YAMLError)):
                with self.subTest(path.name), patch.dict(os.environ, {"PROMPT_FILE_PATH": str(path)}):
                    failed = rf"Failed to load YAML from {re.escape(str(path))}: "
                    with self.assertRaisesRegex(RuntimeError, failed) as caught:
                        load_prompts()
                    self.assertIsInstance(caught.exception.__cause__, cause)
            root = rf"YAML root must be a mapping: {re.escape(str(listed))}"
            with patch.dict(os.environ, {"PROMPT_FILE_PATH": str(listed)}), self.assertRaisesRegex(RuntimeError, root):
                load_prompts()
        with self.assertRaisesRegex(KeyError, r"agent_prompts\.Cascade was not found in .*other\.yaml"):
            self.load("custom:\n  content: not the talker prompt\n")
