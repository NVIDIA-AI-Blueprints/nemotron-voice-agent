# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102
"""The example's ``prompts.yaml`` as its reader reads it: a named prompt by group and name, the lines, the texts."""

import os
import re
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml
from loguru import logger

from examples.frontend_backend_live import prompts
from examples.frontend_backend_live.prompts import _group, load_prompts, optional
from examples.shared.strict_settings import StrictSettings

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


class CoreTextTests(unittest.TestCase):
    def test_unheard_write_output_is_plain_text_with_no_field(self):
        for key in ("texts.unheard_write_output", "texts.unheard_undeclared_output"):
            with self.subTest(key=key):
                self.assertNotIn("{", LOADED.core[key])
        question = LOADED.core["texts.unheard_question_output"]
        self.assertEqual(question.format(question="Which list?").count("Which list?"), 1)


# The caller note's situation bullet, byte for byte as it shipped before this iteration: it must not change.
SITUATION_BULLET = """- Some causes of a problem depend on the caller's situation, which no tool shows. Ask about it in
  one short question right after you have identified the caller and before any checks, unless they
  have already said, and only once. For example:
  - a card that stopped working: whether they are using it abroad;
  - a phone or internet service that stopped working: whether they are at home or travelling right now;
  - a device that will not turn on: whether it was dropped or got wet.
  Do not hand the caller off or give up while such a question is unasked.
"""


class TalkerPromptTests(unittest.TestCase):
    def test_the_lead_in_rule_forbids_repeating_a_question_and_choosing(self):
        rule = " ".join(LOADED.cascade.routing.split())
        self.assertIn(
            "Never repeat a question you or the backend just asked, and never choose an option for the caller", rule
        )


class CallerNoteTests(unittest.TestCase):
    NOTE = LOADED.core["texts.caller_note"]

    def test_the_situation_bullet_is_unchanged(self):
        self.assertIn(SITUATION_BULLET, self.NOTE)

    def test_the_caller_note_has_at_most_three_examples(self):
        examples = [line for line in self.NOTE.splitlines() if line.startswith("  - ")]
        self.assertLessEqual(len(examples), 3)

    def test_the_voice_note_names_the_spelled_note(self):
        self.assertIn(f"[{LOADED.cascade.transcript.spelled}:", LOADED.core["texts.voice_note"])

    def bullets(self) -> list[str]:
        """The note's top-level bullets, each joined into one line."""
        return [" ".join(part.split()) for part in self.NOTE.split("\n- ")[1:]]

    def test_the_consent_bullet_says_a_question_or_more_detail_is_not_a_yes(self):
        consent = next(b for b in self.bullets() if b.startswith("Before any change"))
        self.assertIn("Nor is a question, even one about the change, or more detail about the request", consent)

    def test_the_caller_note_asks_for_a_choice_the_caller_has_not_made(self):
        bullets = self.bullets()
        index = next(i for i, b in enumerate(bullets) if b.startswith("Before any change"))
        self.assertTrue(bullets[index + 1].startswith("Before a change, ask for every detail of it"))

    def test_the_choice_bullet_quotes_no_example_and_holds_no_number(self):
        choice = next(b for b in self.bullets() if b.startswith("Before a change, ask"))
        self.assertFalse(any(char in choice for char in '"“”0123456789'))  # an apostrophe is no quotation

    def test_the_choice_bullet_exempts_defaults_the_instructions_give(self):
        choice = next(b for b in self.bullets() if b.startswith("Before a change, ask"))
        self.assertIn("give no default", choice)
        self.assertIn("including defaults", choice)

    def test_the_claims_bullet_comes_last(self):
        self.assertTrue(self.bullets()[-1].startswith("Never tell the caller you did something that no tool call"))

    def test_the_voice_note_asks_for_a_short_first_sentence(self):
        note = LOADED.core["texts.voice_note"]
        question = note.index("- Put any question or request for confirmation at the end.")
        self.assertGreater(note.index("- Start with one short sentence that holds what the caller asked for"), question)

    def test_the_writer_keeps_the_first_sentence_short(self):
        self.assertIn(
            "Keep the first sentence short and make it carry what the caller asked for", LOADED.cascade.answer
        )

    def test_the_questions_block_holds_only_its_field(self):
        block = LOADED.core["texts.questions_block"]
        self.assertIn("{questions}", block)
        block.format(questions="x")  # names no other field


@contextmanager
def warnings_logged():
    """The loguru warnings logged inside the block, each as its message."""
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING", format="{message}")
    try:
        yield messages
    finally:
        logger.remove(sink)


class Notes(StrictSettings):
    """A made-up text group: one existing text and one added later, optional in a prompt file."""

    header: str
    added: str | None = optional("the added note")


class CascadeTemplateTests(unittest.TestCase):
    # Each new Cascade template holds its field and names no other.
    FIELDS = {"not_said": ("answer",), "now_heard": ("rest",), "cut_question": ("questions",), "spelled": ()}

    def test_the_cascade_templates_hold_their_fields_and_no_other(self):
        transcript = LOADED.cascade.transcript
        for name, fields in self.FIELDS.items():
            with self.subTest(name=name):
                text = getattr(transcript, name)
                for field in fields:
                    self.assertIn("{" + field + "}", text)
                text.format(**dict.fromkeys(fields, "x"))  # names no other field

    def test_a_prompt_file_without_the_new_transcript_texts_loads_with_a_warning(self):
        groups = GROUPS
        cascade = {name: entry for name, entry in groups["Cascade"].items() if name.split(".")[-1] not in self.FIELDS}
        with TemporaryDirectory() as folder, warnings_logged() as warnings:
            other = Path(folder) / "other.yaml"
            other.write_text(yaml.safe_dump({"agent_prompts": {**groups, "Cascade": cascade}}), encoding="utf-8")
            with patch.dict(os.environ, {"PROMPT_FILE_PATH": str(other)}):
                loaded = load_prompts()
        for name in self.FIELDS:
            self.assertIsNone(getattr(loaded.cascade.transcript, name))
            self.assertEqual(sum(f"key=Cascade.transcript.{name} " in warning for warning in warnings), 1)


class OptionalTextTests(unittest.TestCase):
    """README rule 6: a text added after a prompt file was written is optional in it; an existing one stays required."""

    def test_a_missing_optional_text_loads_as_none_with_one_warning(self):
        groups = {"Library": {"header": {"content": "Since the last request:"}}}
        with warnings_logged() as warnings:
            notes = _group(Notes, groups, "Library", PROMPTS_PATH)
        self.assertEqual(notes, Notes(header="Since the last request:", added=None))
        self.assertEqual(len(warnings), 1)
        self.assertIn("prompts.missing_text key=Library.added feature=the added note", warnings[0])

    def test_a_missing_existing_text_still_fails(self):
        groups = {"Library": {"added": {"content": "A note."}}}
        with self.assertRaisesRegex(ValueError, r"agent_prompts\.Library in .*\nheader\n  Field required"):
            _group(Notes, groups, "Library", PROMPTS_PATH)
