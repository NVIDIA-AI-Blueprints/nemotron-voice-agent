# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103
"""An example's optional ``prompt_catalog`` hook: the prompt catalog in place of the ``prompts.yaml`` beside its bot."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

import examples_registry
from utils import load_prompt_catalog

HOOK = "examples.frontend_backend_live.prompts.catalog:ui_catalog"


def entry(**changes) -> dict:
    """A minimal raw registry entry."""
    return {"label": "An example", "bot": "examples.generic_assistant.pipeline:bot", **changes}


class PromptCatalogHookTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ))
        os.environ.pop("PROMPT_FILE_PATH", None)

    def test_the_hook_is_parsed_and_an_example_without_one_has_none(self):
        self.assertEqual(
            examples_registry._parse_example("x", entry(prompt_catalog=f" {HOOK} "))["prompt_catalog"], HOOK
        )
        self.assertIsNone(examples_registry._parse_example("x", entry())["prompt_catalog"])
        with self.assertRaisesRegex(RuntimeError, "Example 'x' prompt_catalog must be 'module.path:attr'"):
            examples_registry._parse_example("x", entry(prompt_catalog="no.callable"))

    def test_only_the_frontend_backend_live_example_names_a_hook(self):
        hooks = {key: example["prompt_catalog"] for key, example in examples_registry.EXAMPLES.items()}
        self.assertEqual({key: hook for key, hook in hooks.items() if hook}, {"frontend-backend-live": HOOK})

    def test_the_hook_builds_the_catalog_and_an_example_without_one_reads_its_prompts_yaml(self):
        live = examples_registry.find("frontend-backend-live", ignore_lock=True)
        self.assertEqual(list(examples_registry.prompt_catalog(live)), ["cafe"])
        generic = examples_registry.find("generic-assistant", ignore_lock=True)
        module_file = examples_registry.example_module_file(generic)
        self.assertEqual(examples_registry.prompt_catalog(generic), load_prompt_catalog(module_file))
        self.assertEqual(
            examples_registry.prompt_catalog(dict(generic, prompt_catalog=None)), load_prompt_catalog(module_file)
        )

    def test_a_prompt_file_the_server_was_started_with_wins_over_the_hook(self):
        with tempfile.TemporaryDirectory() as folder:
            prompt_file = Path(folder) / "prompts.yaml"
            prompt_file.write_text(yaml.safe_dump({"from_file": {"content": "From a file.\n"}}), encoding="utf-8")
            os.environ["PROMPT_FILE_PATH"] = str(prompt_file)
            live = examples_registry.find("frontend-backend-live", ignore_lock=True)
            self.assertEqual(examples_registry.prompt_catalog(live), {"from_file": {"content": "From a file.\n"}})

    def test_the_registry_default_prompt_is_the_default_profiles_talker(self):
        live = examples_registry.find("frontend-backend-live", ignore_lock=True)
        (default,) = examples_registry.metadata(live)["defaults"]["prompt"]
        talker = examples_registry.prompt_catalog(live)["cafe"]
        self.assertEqual(
            (default["key"], default["description"], default["content"]),
            ("cafe", talker["description"], talker["content"]),
        )


if __name__ == "__main__":
    unittest.main()
