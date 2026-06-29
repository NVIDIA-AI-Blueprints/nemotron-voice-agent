# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

if "timeutils" not in sys.modules:
    timeutils = types.ModuleType("timeutils")
    timeutils.TOOL_HANDLERS = {}
    sys.modules["timeutils"] = timeutils

from utils import (
    PROJECT_ROOT,
    default_prompt_key,
    load_prompt_catalog,
    load_yaml_file,
    resolve_prompt,
    resolve_prompt_catalog_path,
)


@contextmanager
def _example_with_catalog(contents: str):
    """Yield a fake example module path whose sibling ``prompts.yaml`` holds ``contents``."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "prompts.yaml").write_text(contents, encoding="utf-8")
        yield Path(tmpdir) / "pipeline.py"


class PromptCatalogTests(unittest.TestCase):
    def test_builtin_catalogs_have_expected_prompts(self) -> None:
        cases = {
            PROJECT_ROOT / "src/examples/generic/prompts.yaml": {
                "flowershop",
                "generic_assistant",
                "generic_assistant_without_tools",
            },
            PROJECT_ROOT / "src/examples/multilingual/prompts.yaml": {
                "auto_detect_language_addon",
                "fixed_session_language_addon",
                "multilingual_voice_assistant",
            },
        }
        for path, expected in cases.items():
            self.assertEqual(set(load_yaml_file(path).keys()), expected, path)

    def test_perf_prompt_catalog_defaults_to_prompt_1000_tokens(self) -> None:
        catalog = load_yaml_file(PROJECT_ROOT / "benchmarking_tools/scaling-perf/perf_prompts.yaml")
        self.assertEqual(default_prompt_key(catalog), "prompt_1000_tokens")

    def test_resolves_to_package_local_prompts_yaml(self) -> None:
        module_file = PROJECT_ROOT / "src/examples/generic/pipeline.py"
        self.assertEqual(
            resolve_prompt_catalog_path(module_file),
            PROJECT_ROOT / "src/examples/generic/prompts.yaml",
        )

    def test_env_override_wins_over_package_local_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            override = Path(tmpdir) / "custom.yaml"
            override.write_text("custom:\n  content: hi\n", encoding="utf-8")
            with patch.dict(os.environ, {"PROMPT_FILE_PATH": str(override)}, clear=True):
                self.assertEqual(load_prompt_catalog("/anywhere/pipeline.py"), {"custom": {"content": "hi"}})

    def test_resolve_prompt_returns_first_entry_by_default(self) -> None:
        with (
            _example_with_catalog("first:\n  content: a\nsecond:\n  content: b\n") as module_file,
            patch.dict(os.environ, {}, clear=True),
        ):
            self.assertEqual(resolve_prompt(module_file), ("first", "a"))

    def test_resolve_prompt_honors_explicit_default_flag(self) -> None:
        with (
            _example_with_catalog("first:\n  content: a\nsecond:\n  default: true\n  content: b\n") as module_file,
            patch.dict(os.environ, {}, clear=True),
        ):
            self.assertEqual(resolve_prompt(module_file), ("second", "b"))

    def test_resolve_prompt_honors_explicit_key(self) -> None:
        with (
            _example_with_catalog("first:\n  content: a\nsecond:\n  content: b\n") as module_file,
            patch.dict(os.environ, {}, clear=True),
        ):
            self.assertEqual(resolve_prompt(module_file, prompt_key="second"), ("second", "b"))

    def test_resolve_prompt_falls_back_when_key_missing(self) -> None:
        with (
            _example_with_catalog("first:\n  content: a\n") as module_file,
            patch.dict(os.environ, {}, clear=True),
        ):
            self.assertEqual(resolve_prompt(module_file, prompt_key="missing"), ("first", "a"))

    def test_prompt_selector_env_overrides_default(self) -> None:
        with (
            _example_with_catalog("first:\n  content: a\nsecond:\n  content: b\n") as module_file,
            patch.dict(os.environ, {"PROMPT_SELECTOR": "second"}, clear=True),
        ):
            self.assertEqual(resolve_prompt(module_file), ("second", "b"))

    def test_resolve_prompt_passes_through_client_content(self) -> None:
        self.assertEqual(
            resolve_prompt("/anywhere/pipeline.py", prompt_content="hello"),
            ("custom", "hello"),
        )


if __name__ == "__main__":
    unittest.main()
