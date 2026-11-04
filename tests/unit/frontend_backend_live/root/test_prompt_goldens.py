# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""What a model, the caller or the Live Console reads from the example's prompts, observed through the code paths.

``prompt_goldens.json`` holds a SHA-256 and a length for every observation below. Each observation comes from the real
code path: the UI bot's ``bot()`` (its talker seed, and the settings and defaults it builds its core from), the Live
core's build, the Live sessions' default talker, the lines, the Live Console's sample and the UI's prompt list. A
prompt edited on purpose changes its hashes: rerun the observations and update the file in the same change.
"""

import asyncio
import hashlib
import json
import os
import re
import tempfile
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from fastapi.testclient import TestClient

import examples_registry
from examples.frontend_backend_live import factory, pipeline
from examples.frontend_backend_live.cascade.lines import FixedLines
from examples.frontend_backend_live.config_manager.loader import CONFIG_PATH, EXAMPLE_DIR, load_live_config
from examples.frontend_backend_live.config_manager.schema import LiveConfig
from examples.frontend_backend_live.delegation_core.app import tool_sources
from examples.frontend_backend_live.delegation_core.config import CoreSettings
from examples.frontend_backend_live.prompts import load_prompts
from server import create_app
from tests.unit.frontend_backend_live.root.root_helpers import clear_root_caches, stub_root
from tests.unit.frontend_backend_live.root.test_root_ui_bot import BotCase

FIXTURE = Path(__file__).with_name("prompt_goldens.json")
EXAMPLE = "frontend-backend-live"
# The example's prompt file as written.
PROMPTS_PATH = CONFIG_PATH.with_name("prompts.yaml")
# The talker of a prompt file the server may be started with (--prompt-file), marked as that file's default.
FROM_FILE = {"from_file": {"default": True, "description": "A prompt file's prompt.", "content": "From a file.\n"}}
# A prompt file replaces the example's whole file, so it holds the example's entries too, as this one does.
PROMPT_FILE = {**yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8")), **FROM_FILE}

# The UI bot's sessions, by case: the session body and the environment. Each must match the golden as it is.
SESSIONS = {
    "default": ({}, {}),
    "selected": ({"prompt_key": "cafe"}, {}),
    "custom": ({"prompt_content": "A custom talker.\n", "prompt_key": "my_prompt"}, {}),
    "custom_with_internal_key": ({"prompt_content": "A custom talker.\n", "prompt_key": "generic"}, {}),
    "selector": ({}, {"PROMPT_SELECTOR": "cafe"}),
    "prompt_file": ({}, {"PROMPT_FILE_PATH": True}),
    "custom_with_prompt_file": ({"prompt_content": "A custom talker.\n"}, {"PROMPT_FILE_PATH": True}),
}
# Sessions that name no prompt the UI lists: each falls back to the default one, so it observes as ``default`` does.
FALLBACKS = {
    "legacy_key": ({"prompt_key": "voice"}, {}),
    "internal_key": ({"prompt_key": "generic"}, {}),
    "unknown_key": ({"prompt_key": "no_such_prompt"}, {}),
}
# The delegation core's own files, each loaded as its own CORE_CONFIG_PATH would load it.
CORE_FILES = ("core.yaml", "core.agent.yaml", "core.harness.yaml")


def digest(value) -> dict:
    """The SHA-256 and length of ``value`` as canonical JSON (``repr`` for what JSON cannot hold)."""
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=repr)
    return {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "length": len(text)}


def without_prompt_name(settings: CoreSettings) -> str:
    """``settings`` with ``backend.prompt`` blanked: the name is an identifier, the text it names is observed apart."""
    return repr(replace(settings, backend=replace(settings.backend, prompt="")))


def built_tools(settings: CoreSettings) -> list[dict]:
    """The tools that the tool sources ``backend.tools`` names build, in order."""
    return [tool for source in tool_sources(settings.backend.tools)() for tool in source.tools()]


class PromptGoldenCase(BotCase):
    """Every observation, through the code paths the example runs."""

    def setUp(self):
        super().setUp()
        self.builds = []
        self.enterContext(patch.object(pipeline, "build_core", self.record_build))
        self.enterContext(patch.dict(os.environ))
        prompt_file = Path(self.enterContext(tempfile.TemporaryDirectory())) / "prompts.yaml"
        prompt_file.write_text(json.dumps(PROMPT_FILE), encoding="utf-8")
        self.prompt_file = str(prompt_file)

    def record_build(self, settings, *, endpoints, defaults):
        self.builds.append((settings, defaults))
        return SimpleNamespace(aclose=self.no_close)

    async def no_close(self):
        pass

    async def bot_session(self, body: dict, env: dict) -> dict:
        """One UI bot session: its talker seed, and the core settings, instructions and tools it builds a core from."""
        for name in ("PROMPT_FILE_PATH", "PROMPT_SELECTOR"):
            os.environ.pop(name, None)
        os.environ.update({name: self.prompt_file if value is True else value for name, value in env.items()})
        self.cascades.clear()
        self.builds.clear()
        self.body = {"pipeline_mode": EXAMPLE, **body}
        cascade = await self.run_bot(self.client_ready)
        ((settings, defaults),) = self.builds
        return {
            "seed": cascade.conversation.messages(),
            "core": without_prompt_name(settings),
            "instructions": defaults.instructions,
            "tools": built_tools(settings),
        }

    def live_core(self) -> dict:
        """The Live core's build: its settings and the default instructions of a request that sets none."""
        clear_root_caches(self)
        builds = []
        with (
            patch.object(factory, "root", return_value=stub_root()),
            patch.object(
                factory, "build_core", lambda settings, *, endpoints, defaults: builds.append((settings, defaults))
            ),
        ):
            factory.delegation_core()
        ((settings, defaults),) = builds
        return {"core": without_prompt_name(settings), "instructions": defaults.instructions}

    def live_talker(self) -> str:
        """The talker a Live session with no or blank instructions starts from (``frontend.cascade.prompt``)."""
        config = load_live_config(CONFIG_PATH)
        core = CoreSettings.load(EXAMPLE_DIR / "core.yaml")
        name = LiveConfig.parse(factory.with_display_label(factory.adapter_block(config), core)).prompt
        return load_prompts().text("Talker", name)

    def console(self) -> dict:
        """The Live Console's sample in ``GET /v1/live/info``: its prefills, tools, lines and presets."""
        for name in ("PROMPT_FILE_PATH", "PROMPT_SELECTOR"):
            os.environ.pop(name, None)
        clear_root_caches(self)
        with (
            patch.object(factory, "root", return_value=stub_root()),
            patch.object(factory, "delegation_core", return_value=SimpleNamespace(describe=dict)),
        ):
            info = asyncio.run(asyncio.to_thread(factory._describe))
        return {key: info[key] for key in ("instructions", "sample_tools", "lines", "presets")}

    def prompt_list(self) -> dict:
        """The UI's prompt list for the example (``/api/prompts``) and its registry default, without the keys."""
        for name in ("PROMPT_FILE_PATH", "PROMPT_SELECTOR"):
            os.environ.pop(name, None)
        with TestClient(create_app()) as client:
            listed = client.get("/api/prompts", params={"pipeline_mode": EXAMPLE}).json()
        (default,) = [entry for entry in listed if entry["default"]]
        registry = examples_registry.metadata(examples_registry.find(EXAMPLE, ignore_lock=True))["defaults"]["prompt"]
        (registry_default,) = registry
        drop = ("key",)
        return {
            "listed_default": {k: v for k, v in default.items() if k not in drop},
            "registry_default": {k: v for k, v in registry_default.items() if k not in drop},
        }

    async def observe(self, sessions: dict) -> dict:
        """Every observation by name."""
        observed = {}
        for case, (body, env) in sessions.items():
            for part, value in (await self.bot_session(body, env)).items():
                observed[f"bot.{case}.{part}"] = value
        for part, value in (await asyncio.to_thread(self.live_core)).items():
            observed[f"live_core.{part}"] = value
        observed["live_talker"] = self.live_talker()
        lines = load_prompts().lines
        observed["lines.parsed"] = repr(lines)
        observed["lines.by_name"] = {field.name: getattr(lines, field.name) for field in fields(FixedLines)}
        for part, value in (await asyncio.to_thread(self.console)).items():
            observed[f"console.{part}"] = value
        for part, value in self.prompt_list().items():
            observed[f"prompt_list.{part}"] = value
        for name in CORE_FILES:
            observed[f"core_file.{name}"] = without_prompt_name(CoreSettings.load(EXAMPLE_DIR / name))
        return observed


class PromptGoldenTests(PromptGoldenCase):
    async def test_every_observation_matches_its_golden(self):
        golden = json.loads(FIXTURE.read_text(encoding="utf-8"))
        observed = {name: digest(value) for name, value in (await self.observe(SESSIONS)).items()}
        self.assertEqual(sorted(observed), sorted(golden))
        for name, value in observed.items():
            with self.subTest(observation=name):
                self.assertEqual(value, golden[name])

    async def test_a_session_that_names_no_listed_prompt_gets_the_default_ones(self):
        default = await self.bot_session({}, {})
        for case, (body, env) in FALLBACKS.items():
            with self.subTest(case=case):
                self.assertEqual(await self.bot_session(body, env), default)

    async def test_a_prompt_file_that_copies_the_examples_changes_no_session_and_no_live_default(self):
        copy = Path(self.enterContext(tempfile.TemporaryDirectory())) / "copy.yaml"
        copy.write_bytes(PROMPTS_PATH.read_bytes())
        shipped = (await self.bot_session({}, {}), await asyncio.to_thread(self.live_core), self.live_talker())
        session = await self.bot_session({}, {"PROMPT_FILE_PATH": str(copy)})  # the copy stays set for the rest
        self.assertEqual((session, await asyncio.to_thread(self.live_core), self.live_talker()), shipped)

    async def test_a_prompt_file_with_a_talker_only_refuses_the_session_and_names_what_it_lacks(self):
        talker_only = Path(self.enterContext(tempfile.TemporaryDirectory())) / "talker-only.yaml"
        talker_only.write_text(json.dumps(FROM_FILE), encoding="utf-8")
        lacks = f"agent_prompts.Cascade was not found in {re.escape(str(talker_only))}"
        with self.assertRaisesRegex(KeyError, lacks):
            await self.bot_session({}, {"PROMPT_FILE_PATH": str(talker_only)})
