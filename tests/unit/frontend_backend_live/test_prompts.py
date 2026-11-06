# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102
"""The example's ``prompts.yaml`` as its reader reads it: a named prompt by group and name, and the core's texts."""

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
LOADED = load_prompts()


class NamedPromptTests(unittest.TestCase):
    FILE = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))

    def test_a_named_prompt_is_read_stripped_and_an_unknown_one_names_the_file_and_the_known_ones(self):
        channel = self.FILE["agent_prompts"]["Channel"]["cafe"]["content"]
        self.assertEqual(LOADED.text("Channel", "cafe"), channel.strip())
        known = f"was not found in {re.escape(str(PROMPTS_PATH))}; known: cafe, generic"
        with self.assertRaisesRegex(KeyError, rf"agent_prompts\.Backend\.kiosk {known}"):
            LOADED.text("Backend", "kiosk")

    def test_a_blank_named_prompt_is_refused_and_names_the_file(self):
        blank = replace(LOADED, agent_prompts={"Channel": {"cafe": {"content": " \n"}}})
        where = re.escape(str(PROMPTS_PATH))
        with self.assertRaisesRegex(KeyError, rf"agent_prompts\.Channel\.cafe in {where} has no text"):
            blank.text("Channel", "cafe")

    def test_the_cores_texts_are_its_group_as_written(self):
        core = self.FILE["agent_prompts"]["Core"]
        self.assertEqual(dict(LOADED.core), {name: entry["content"] for name, entry in core.items()})
        self.assertTrue(LOADED.core["texts.caller_notes.default"].startswith("\n## Working"))  # not stripped

    def test_a_prompt_file_without_the_cores_texts_names_the_group_and_the_file(self):
        groups = {name: group for name, group in self.FILE["agent_prompts"].items() if name != "Core"}
        with TemporaryDirectory() as folder:
            other = Path(folder) / "other.yaml"
            other.write_text(yaml.safe_dump({"agent_prompts": groups}), encoding="utf-8")
            lacks = rf"agent_prompts\.Core was not found in {re.escape(str(other))}"
            with patch.dict(os.environ, {"PROMPT_FILE_PATH": str(other)}), self.assertRaisesRegex(KeyError, lacks):
                load_prompts()


class PromptFileTests(unittest.TestCase):
    """The prompt file is read whole: a missing or malformed one is refused, and the lines are the voice's, no other."""

    def load(self, path: Path):
        with patch.dict(os.environ, {"PROMPT_FILE_PATH": str(path)}):
            return load_prompts()

    def test_the_lines_are_every_line_the_voice_says_and_no_other(self):
        groups = NamedPromptTests.FILE["agent_prompts"]
        lines = groups["Lines"]
        cases = {
            "an unknown line": ({**lines, "farewell": {"content": "Bye."}}, r"unknown \['farewell'\], missing \[\]"),
            "a missing line": (
                {name: line for name, line in lines.items() if name != "holding"},
                r"unknown \[\], missing \['holding'\]",
            ),
        }
        with TemporaryDirectory() as folder:
            other = Path(folder) / "other.yaml"
            for name, (given, message) in cases.items():
                other.write_text(yaml.safe_dump({"agent_prompts": {**groups, "Lines": given}}), encoding="utf-8")
                with self.subTest(name), self.assertRaisesRegex(ValueError, rf"agent_prompts\.Lines: {message}"):
                    self.load(other)

    def test_a_missing_or_malformed_file_is_refused_with_its_cause(self):
        with TemporaryDirectory() as folder:
            missing, malformed, listed = (Path(folder) / name for name in ("missing.yaml", "bad.yaml", "list.yaml"))
            malformed.write_text("agent_prompts: [\n", encoding="utf-8")
            listed.write_text("- agent_prompts\n", encoding="utf-8")
            for path, cause in ((missing, FileNotFoundError), (malformed, yaml.YAMLError)):
                with self.subTest(path.name):
                    failed = rf"Failed to load YAML from {re.escape(str(path))}: "
                    with self.assertRaisesRegex(RuntimeError, failed) as caught:
                        self.load(path)
                    self.assertIsInstance(caught.exception.__cause__, cause)
            with self.assertRaisesRegex(RuntimeError, rf"YAML root must be a mapping: {re.escape(str(listed))}"):
                self.load(listed)


if __name__ == "__main__":
    unittest.main()
