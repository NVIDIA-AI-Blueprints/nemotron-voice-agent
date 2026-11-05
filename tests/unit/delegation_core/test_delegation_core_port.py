# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The core's settings file and its profiles, the conductor's control vocabulary, and the names a core is built from."""

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from examples.frontend_backend_live.delegation_core.app import build_core
from examples.frontend_backend_live.delegation_core.backend.control import ControlVocabulary
from examples.frontend_backend_live.delegation_core.backend.port import Answer, Cancel, Start, Steer
from examples.frontend_backend_live.delegation_core.config import BackendChoice, CoreSettings, _placed
from tests.unit.core_helpers import CORE_TEXTS
from tests.unit.delegation_core.helpers import CORE_SETTINGS, DEFAULTS, TEXTS

WORDS = _placed(CORE_TEXTS)  # the shipped texts, placed as the core's settings hold them
BACKEND = {"slot": "s", "prompt": "p"}


def vocabulary(schemas=TEXTS.control_tools) -> ControlVocabulary:
    return ControlVocabulary(schemas, TEXTS.conductor_prompt, TEXTS.refusal("idle"), TEXTS.failed_output)


class ControlVocabularyTests(unittest.TestCase):
    def test_each_control_call_makes_its_decision_and_a_malformed_one_is_refused(self):
        self.assertEqual([tool["name"] for tool in vocabulary().conductor_tools()], list(TEXTS.control_tools))
        cases = [
            ("start_work", {"delegations": ["d1"]}, Start(("d1",))),
            ("start_work", {"delegations": ["d1"], "tasks": ["menu", "hours"]}, Start(("d1",), ("menu", "hours"))),
            ("steer_work", {"work_id": "W1", "delegations": ["d2"], "task_id": "W1.2"}, Steer("W1", ("d2",), "W1.2")),
            ("steer_work", {"work_id": "W1", "delegations": ["d2"]}, Steer("W1", ("d2",))),
            ("cancel_work", {"work_id": "W1"}, Cancel("W1")),
            ("answer_status", {"delegation_id": "d3", "text": "Still checking."}, Answer("d3", "Still checking.")),
        ]
        for name, arguments, decision in cases:
            with self.subTest(name=name, arguments=arguments):
                self.assertEqual(vocabulary().decision(name, json.dumps(arguments)), decision)
        for name, arguments in (
            ("start_work", "{}"),
            ("start_work", '{"delegations": []}'),
            ("steer_work", '{"work_id": "W1"}'),
            ("answer_status", '{"delegation_id": "d1", "text": ""}'),
            ("cancel_work", "not json"),
            ("run_shell", "{}"),
        ):
            with self.subTest(name=name, arguments=arguments), self.assertRaises(ValueError):
                vocabulary().decision(name, arguments)

    def test_schemas_that_drop_a_parameter_or_a_tool_are_refused(self):
        schemas = copy.deepcopy({name: dict(schema) for name, schema in TEXTS.control_tools.items()})
        del schemas["steer_work"]["parameters"]["properties"]["work_id"]
        with self.assertRaisesRegex(ValueError, "steer_work"):
            vocabulary(schemas)
        with self.assertRaisesRegex(ValueError, "exactly"):
            vocabulary({name: schema for name, schema in schemas.items() if name != "cancel_work"})


class CoreSettingsTests(unittest.TestCase):
    def parsed(self, delegation=None, backend=None, **texts) -> CoreSettings:
        """The shipped texts with ``texts`` over them, and the ``delegation`` and ``backend`` sections given."""
        data = {"backend": backend or BACKEND}
        words = {**WORDS, "texts": {**WORDS["texts"], **texts}}
        return CoreSettings.parse(data if delegation is None else {**data, "delegation": delegation}, words)

    def test_a_file_that_sets_little_gets_the_defaults_and_keeps_each_adapters_block(self):
        settings = self.parsed(backend={**BACKEND, "llm": {"conductor": "steer"}, "tools": {"cafe": {}}})
        self.assertEqual((settings.backend.adapter, settings.backend.one_shot_adapter), ("llm", "llm"))
        self.assertEqual(dict(settings.backend.blocks["llm"]), {"conductor": "steer"})
        self.assertEqual(list(settings.backend.tools), ["cafe"])
        defaults = {"client_tool_timeout_seconds": 60.0, "work_timeout_seconds": 600.0, "backend_calls": "items"}
        defaults |= {"display_label": "delegation_core", "records_max_works": 5}
        defaults |= {"context_tokens": 24_000, "tool_output_chars": 8000}
        self.assertEqual({name: getattr(settings.delegation, name) for name in defaults}, defaults)
        limits = {"client_tool_timeout_seconds": None, "close_grace_seconds": None, "records_max_works": 0}
        delegation = self.parsed(limits).delegation  # no deadline, no grace, and records that list no ended Work
        self.assertEqual({name: getattr(delegation, name) for name in limits}, limits)

    def test_unknown_keys_wrong_values_and_missing_names_are_refused(self):
        cases = {
            "unknown key": ({"max_pending": 3}, None),
            "a fraction": ({"max_pending_delegations": 2.5}, None),
            "a boolean count": ({"context_tokens": True}, None),
            "no room for a tool output": ({"tool_output_chars": 0}, None),
            "a negative wait": ({"stop_timeout_seconds": -1}, None),
            "an unknown caller note": ({"caller_note": "c9"}, None),
            "an unknown voice note": ({"voice_note": "v9"}, None),
            "an unknown format": ({"result_format": "html"}, None),
            "an unknown work tool": ({"work_tools": ["run_shell"]}, None),
            "a scalar adapter block": (None, {**BACKEND, "llm": "fast"}),
            "a scalar step block": (None, {**BACKEND, "app_output_steps": {"spell": "on"}}),
            "no slot": (None, {"prompt": "p"}),
        }
        for name, (delegation, backend) in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.parsed(delegation, backend)

    def test_the_texts_keep_their_template_fields(self):
        outcomes, reasons = WORDS["texts"]["outcomes"], WORDS["texts"]["refusal_reasons"]
        cases = {
            "handover_line": ("Handled together.", "must hold {delegation}"),
            "task_line": ("Task:", "must hold {task}"),
            "unknown_tool_output": ("No {name}.", "must hold {offered}"),
            "failed_output": ("{name}: {error} {why}", "not given"),
            "misrouted_output": ("The app's.", "must hold {name}"),
            "repeated_call_output": ("Again.", "must hold {output}"),
            "conductor_prompt": ("Place them.", "must hold {max_running}"),
            "cut_output": ("[cut]", "must hold {chars}"),
            "outcomes": ({**outcomes, "refused": "Not applied."}, "outcomes.refused must hold {reason}"),
            "refusal_reasons": ({"ended": "ended"}, "refusal_reasons must hold exactly"),
            "no_work": ({**reasons, "no_work": "no such work"}, "no_work must hold {work}"),
            "over_limit": ({**reasons, "over_limit": "too many"}, "over_limit must hold {max_running}"),
            "round_cap_note": (" ", "round_cap_note must be non-empty"),
        }
        for key, (value, message) in cases.items():
            with self.subTest(key), self.assertRaisesRegex(ValueError, message):
                self.parsed(**{"refusal_reasons" if key in reasons else key: value})
        settings = self.parsed(outcomes={**outcomes, "refused": "Not applied: {reason}"})
        self.assertEqual(settings.texts.refusal("no_task", task="W1.7", work="W1"), "Not applied: no task W1.7 in W1")

    def test_a_profile_merges_mappings_over_the_file_it_extends_and_replaces_the_rest(self):
        with tempfile.TemporaryDirectory() as folder:
            files = {
                "core.yaml": {"backend": {**BACKEND, "llm": {"conductor": "steer"}}},
                "core.agent.yaml": {
                    "extends": "core.yaml",
                    "delegation": {"result_format": "speak_display", "client_tool_timeout_seconds": None},
                    "backend": {"adapter": "llm-agent", "llm-agent": {"max_running": 3}},
                },
                "core.harness.yaml": {"extends": "core.agent.yaml", "backend": {"adapter": "harness"}},
                "a.yaml": {"extends": "b.yaml"},
                "b.yaml": {"extends": "a.yaml"},
                "c.yaml": {"extends": "missing.yaml"},
            }
            for name, data in files.items():
                (Path(folder) / name).write_text(json.dumps(data), encoding="utf-8")  # JSON is YAML
            settings = CoreSettings.load(Path(folder) / "core.harness.yaml", CORE_TEXTS)
            for name, message in (("a.yaml", "extends itself"), ("c.yaml", "Cannot read")):
                with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                    CoreSettings.load(Path(folder) / name, CORE_TEXTS)
        self.assertEqual((settings.backend.adapter, sorted(settings.backend.blocks)), ("harness", ["llm", "llm-agent"]))
        self.assertEqual(settings.texts.conductor_prompt, TEXTS.conductor_prompt)  # the texts come from prompts.yaml
        self.assertEqual(settings.texts.handover_line, TEXTS.handover_line)
        self.assertEqual(settings.delegation.result_format, "speak_display")
        self.assertIsNone(settings.delegation.client_tool_timeout_seconds)

    def test_a_core_that_names_an_unknown_adapter_or_app_output_step_fails_when_it_is_built(self):
        cases = {
            "Unknown backend adapter 'hermes'; known: harness, llm, llm-agent": {**BACKEND, "adapter": "hermes"},
            "Unknown app output step 'spell'": {**BACKEND, "app_output_steps": {"spell": {}}},
        }
        for message, backend in cases.items():
            settings = replace(CORE_SETTINGS, backend=BackendChoice.parse(backend))
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                build_core(settings, endpoints=lambda model: None, defaults=DEFAULTS)
