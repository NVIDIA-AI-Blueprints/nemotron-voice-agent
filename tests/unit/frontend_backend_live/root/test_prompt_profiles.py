# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The prompt profiles: their reader, a UI bot session's choice of one, and the UI's prompt list."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from fastapi.testclient import TestClient

from examples.frontend_backend_live import pipeline
from examples.frontend_backend_live.prompts import catalog
from server import create_app

EXAMPLE = "frontend-backend-live"


class ProfilesCase(unittest.TestCase):
    """A copy of the shipped profiles in a temporary folder, which the reader reads in place of ``prompts/``."""

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())) / "prompts"
        shutil.copytree(catalog.PROMPTS_DIR, self.root, ignore=shutil.ignore_patterns("*.py", "__pycache__"))
        self.enterContext(patch.object(catalog, "PROMPTS_DIR", self.root))
        self.enterContext(patch.object(catalog, "LINES_PATH", self.root / "shared" / "lines.yaml"))
        self.enterContext(patch.dict(os.environ))
        for name in ("PROMPT_FILE_PATH", "PROMPT_SELECTOR"):
            os.environ.pop(name, None)

    def write(self, relative: str, contents) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents if isinstance(contents, str) else yaml.safe_dump(contents), encoding="utf-8")

    def add_profile(self, name: str, *, channel: bool = True, **settings) -> None:
        """A profile ``name`` whose role texts say whose they are."""
        self.write(f"{name}/profile.yaml", settings)
        for role in ("talker", "thinker", "channel") if channel else ("talker", "thinker"):
            self.write(f"{name}/{role}.yaml", {"description": f"The {name} {role}.", "content": f"{name} {role}.\n"})


class ReaderTests(ProfilesCase):
    def test_the_profiles_are_the_folders_with_settings_sorted_and_the_internal_ones_unlisted(self):
        self.add_profile("kiosk")
        (self.root / "notes").mkdir()  # a folder without profile.yaml is no profile
        self.assertEqual(catalog.profile_names(), ["cafe", "generic", "kiosk"])
        self.assertEqual(catalog.listed_profile_names(), ["cafe", "kiosk"])
        self.assertEqual(catalog.default_profile().name, "cafe")

    def test_a_profile_reads_its_roles_its_tools_and_its_presets(self):
        cafe = catalog.get_profile("cafe")
        self.assertEqual((cafe.default, cafe.internal, dict(cafe.tools)), (True, False, {"cafe": {}}))
        self.assertTrue(cafe.console_presets)
        self.assertTrue(cafe.content("talker").endswith("\n"))  # as the file holds it
        self.assertEqual(cafe.prompt("talker"), cafe.content("talker").strip())
        self.assertEqual(cafe.instructions("thinker"), f"{cafe.prompt('thinker')}\n\n{cafe.prompt('channel')}")
        generic = catalog.get_profile("generic")
        self.assertEqual((generic.internal, dict(generic.tools), generic.console_presets), (True, {}, {}))
        self.assertEqual(generic.instructions("thinker"), generic.prompt("thinker"))  # no channel follows it

    def test_an_edit_takes_effect_at_the_next_read(self):
        self.write("cafe/talker.yaml", {"description": "Edited.", "content": "An edited talker.\n"})
        self.assertEqual(catalog.get_profile("cafe").prompt("talker"), "An edited talker.")
        self.assertEqual(catalog.ui_catalog()["cafe"]["content"], "An edited talker.\n")

    def test_an_unknown_profile_or_a_missing_or_empty_role_is_a_key_error(self):
        with self.assertRaisesRegex(KeyError, "Unknown prompt profile 'kiosk'; known: cafe, generic"):
            catalog.get_profile("kiosk")
        self.add_profile("kiosk")
        (self.root / "kiosk" / "thinker.yaml").unlink()
        self.write("kiosk/talker.yaml", {"description": "Empty.", "content": "  \n"})
        kiosk = catalog.get_profile("kiosk")
        for role in ("talker", "thinker"):
            with self.subTest(role=role), self.assertRaisesRegex(KeyError, f"Prompt '{role}' was not found"):
                kiosk.prompt(role)

    def test_a_listed_profile_needs_a_channel_and_an_internal_one_does_not(self):
        self.add_profile("kiosk", channel=False)
        with self.assertRaisesRegex(ValueError, "'kiosk' is listed, so it needs a channel.yaml"):
            catalog.get_profile("kiosk")
        self.add_profile("kiosk", channel=False, internal=True)
        self.assertEqual(catalog.get_profile("kiosk").prompt("thinker"), "kiosk thinker.")

    def test_malformed_settings_are_value_errors(self):
        for settings, message in (
            ("- not a mapping\n", "Prompt profile 'kiosk'"),
            ({"description": "A surface text."}, "unknown settings description"),
            ({"internal": "yes"}, "internal must be true or false"),
            ({"tools": ["cafe"]}, "tools must map each tool source's name to its block"),
        ):
            with self.subTest(settings=settings):
                self.add_profile("kiosk")
                self.write("kiosk/profile.yaml", settings)
                with self.assertRaisesRegex(ValueError, message):
                    catalog.get_profile("kiosk")

    def test_exactly_one_listed_profile_is_the_default(self):
        self.add_profile("kiosk", default=True)
        with self.assertRaisesRegex(ValueError, r"Exactly one prompt profile must set default: true; found \['cafe'"):
            catalog.default_profile()
        self.write("cafe/profile.yaml", {"tools": {"cafe": {}}})
        self.write("kiosk/profile.yaml", {"default": True, "internal": True})
        with self.assertRaisesRegex(ValueError, "'kiosk' cannot be internal"):
            catalog.default_profile()
        self.write("kiosk/profile.yaml", {})
        with self.assertRaisesRegex(ValueError, "found none"):
            catalog.default_profile()

    def test_the_ui_catalog_lists_the_listed_profiles_talkers_as_their_files_hold_them(self):
        self.add_profile("kiosk")
        listed = catalog.ui_catalog()
        self.assertEqual(list(listed), ["cafe", "kiosk"])
        talker = yaml.safe_load((self.root / "cafe" / "talker.yaml").read_text(encoding="utf-8"))
        self.assertEqual(listed["cafe"], {**talker, "default": True})
        self.assertEqual(
            listed["kiosk"], {"description": "The kiosk talker.", "content": "kiosk talker.\n", "default": False}
        )

    def test_a_broken_internal_profile_leaves_the_ui_catalog_and_the_prompt_list_working(self):
        self.write("generic/talker.yaml", "content: [unclosed\n")
        self.assertEqual(list(catalog.ui_catalog()), ["cafe"])
        with TestClient(create_app()) as client:
            listed = client.get("/api/prompts", params={"pipeline_mode": EXAMPLE}).json()
        self.assertEqual([(entry["key"], entry["default"]) for entry in listed], [("cafe", True)])


class SessionProfileTests(ProfilesCase):
    """A UI bot session's talker and profile (``pipeline.resolve_profile``), in today's precedence."""

    def setUp(self):
        super().setUp()
        self.add_profile("kiosk")
        self.prompt_file = self.root.parent / "prompt_file.yaml"
        self.prompt_file.write_text(
            yaml.safe_dump({"from_file": {"default": True, "content": "From a file.\n"}}), encoding="utf-8"
        )

    def resolve(self, body: dict, **env) -> tuple[str, str, str]:
        os.environ.update(env)
        name, talker, profile = pipeline.resolve_profile(body)
        return name, talker, profile.name

    def test_a_session_selects_a_listed_profile_by_its_key_or_the_selector_and_otherwise_gets_the_default(self):
        self.assertEqual(self.resolve({}), ("cafe", catalog.get_profile("cafe").content("talker"), "cafe"))
        self.assertEqual(self.resolve({"prompt_key": "kiosk"}), ("kiosk", "kiosk talker.\n", "kiosk"))
        self.assertEqual(self.resolve({}, PROMPT_SELECTOR="kiosk")[2], "kiosk")
        self.assertEqual(self.resolve({"prompt_key": "cafe"}, PROMPT_SELECTOR="kiosk")[2], "cafe")  # the key first

    def test_an_internal_legacy_or_unknown_name_falls_back_to_the_default_with_a_warning(self):
        for name in ("generic", "talker", "voice", "no_such_profile"):
            with self.subTest(name=name), patch.object(pipeline.logger, "warning") as warning:
                self.assertEqual(self.resolve({"prompt_key": name})[2], "cafe")
                self.assertIn(
                    f"Prompt '{name}' not found in the listed prompt profiles; using 'cafe'", str(warning.call_args)
                )

    def test_the_sessions_own_text_is_the_talker_and_its_key_selects_a_listed_profile_only(self):
        custom = {"prompt_content": "A custom talker.\n"}
        self.assertEqual(self.resolve(custom), ("custom", "A custom talker.\n", "cafe"))
        self.assertEqual(self.resolve({**custom, "prompt_key": "kiosk"}), ("kiosk", "A custom talker.\n", "kiosk"))
        self.assertEqual(self.resolve({**custom, "prompt_key": "generic"})[1:], ("A custom talker.\n", "cafe"))

    def test_a_prompt_file_gives_the_talker_and_the_default_profile_and_the_sessions_text_still_wins(self):
        env = {"PROMPT_FILE_PATH": str(self.prompt_file)}
        self.assertEqual(self.resolve({}, **env), ("from_file", "From a file.\n", "cafe"))
        self.assertEqual(self.resolve({"prompt_key": "kiosk"}, **env)[2], "cafe")  # the key names the file's entries
        self.assertEqual(
            self.resolve({"prompt_content": "Mine.\n", "prompt_key": "kiosk"}, **env), ("kiosk", "Mine.\n", "cafe")
        )


if __name__ == "__main__":
    unittest.main()
