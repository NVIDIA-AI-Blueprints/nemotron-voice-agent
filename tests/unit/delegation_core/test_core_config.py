# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The core's settings and texts, and the conductor's control vocabulary."""

import copy
import json
import unittest

from examples.frontend_backend_live.delegation_core.backend.control import ControlVocabulary
from examples.frontend_backend_live.delegation_core.backend.port import Answer, Cancel, Start, Steer
from examples.frontend_backend_live.delegation_core.config import CoreSettings
from examples.shared.strict_settings import placed
from tests.unit.core_helpers import CORE_TEXTS
from tests.unit.delegation_core.helpers import TEXTS

WORDS = placed(CORE_TEXTS, "the core's text")
BACKEND = {"slot": "s", "prompt": "p"}


def vocabulary(schemas=TEXTS.control_tools) -> ControlVocabulary:
    return ControlVocabulary(schemas, TEXTS.conductor_prompt, TEXTS.refusal("idle"), TEXTS.failed_output)


def parsed(delegation=None, backend=None, **texts) -> CoreSettings:
    """Parse the shipped texts with ``texts`` over them, and the ``delegation`` and ``backend`` sections given."""
    data = {"backend": backend or BACKEND}
    words = {**WORDS, "texts": {**WORDS["texts"], **texts}}
    return CoreSettings.parse(data if delegation is None else {**data, "delegation": delegation}, words)


class CoreConfigTests(unittest.TestCase):
    def test_each_control_call_makes_its_decision(self):
        self.assertEqual([tool["name"] for tool in vocabulary().conductor_tools()], list(TEXTS.control_tools))
        cases = [
            ("start_work", {"delegations": ["d1"]}, Start(("d1",))),
            # A key the tool does not take is ignored.
            (
                "start_work",
                {"delegations": ["d1"], "tasks": ["menu", "hours"], "task_id": "W1.1"},
                Start(("d1",), ("menu", "hours")),
            ),
            ("steer_work", {"work_id": "W1", "delegations": ["d2"], "task_id": "W1.2"}, Steer("W1", ("d2",), "W1.2")),
            ("steer_work", {"work_id": "W1", "delegations": ["d2"]}, Steer("W1", ("d2",))),
            # An empty task_id means every task of the work.
            ("cancel_work", {"work_id": "W1", "task_id": ""}, Cancel("W1")),
            ("answer_status", {"delegation_id": "d3", "text": "Still checking."}, Answer("d3", "Still checking.")),
        ]
        for name, arguments, decision in cases:
            with self.subTest(name=name, arguments=arguments):
                self.assertEqual(vocabulary().decision(name, json.dumps(arguments)), decision)
        for name, arguments, error in (
            ("start_work", "{}", "start_work: missing delegations"),
            ("start_work", '{"delegations": []}', "delegations must name at least one"),
            ("start_work", '{"delegations": ["d1", 2]}', "delegations must be a list of non-empty strings"),
            ("steer_work", '{"work_id": "W1"}', "steer_work: missing delegations"),
            ("steer_work", '{"work_id": 7, "delegations": [""]}', "work_id must be a non-empty string"),
            ("answer_status", '{"delegation_id": "d1", "text": ""}', "text must be non-empty text"),
            ("answer_status", '{"delegation_id": "d1", "text": null}', "text must be non-empty text"),
            ("cancel_work", "not json", "cancel_work: the arguments are not JSON"),
            ("run_shell", "{}", "Unknown control tool 'run_shell'"),
        ):
            with self.subTest(name=name, arguments=arguments), self.assertRaisesRegex(ValueError, f"^{error}$"):
                vocabulary().decision(name, arguments)

    def test_schemas_that_drop_a_parameter_or_a_tool_are_refused(self):
        schemas = copy.deepcopy({name: dict(schema) for name, schema in TEXTS.control_tools.items()})
        # An optional parameter may be left out; a parameter the tool does not take is refused.
        del schemas["start_work"]["parameters"]["properties"]["tasks"]
        del schemas["steer_work"]["parameters"]["properties"]["work_id"]
        schemas["steer_work"]["parameters"]["properties"]["tasks"] = {"type": "array"}
        with self.assertRaisesRegex(ValueError, r"steer_work: parameters missing \['work_id'\], not taken \['tasks'\]"):
            vocabulary(schemas)
        with self.assertRaisesRegex(ValueError, "exactly"):
            vocabulary({name: schema for name, schema in schemas.items() if name != "cancel_work"})

    def test_unknown_keys_and_wrong_values_are_refused(self):
        cases = {
            "unknown key": ({"max_pending": 3}, None),
            "a fraction": ({"max_pending_delegations": 2.5}, None),
            "a boolean count": ({"context_tokens": True}, None),
            "no room for a tool output": ({"tool_output_chars": 0}, None),
            "a negative wait": ({"stop_timeout_seconds": -1}, None),
            "an unknown format": ({"result_format": "html"}, None),
            "a scalar adapter block": (None, {**BACKEND, "llm": "fast"}),
            "a scalar step block": (None, {**BACKEND, "app_output_steps": {"spell": "on"}}),
            "no slot": (None, {"prompt": "p"}),
        }
        for name, (delegation, backend) in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                parsed(delegation, backend)

    def test_the_texts_keep_their_template_fields(self):
        outcomes, reasons = WORDS["texts"]["outcomes"], WORDS["texts"]["refusal_reasons"]
        cases = {
            "handover_line": ("Handled together.", "must hold {delegation}"),
            "task_line": ("Task:", "must hold {task}"),
            "unknown_tool_output": ("No {name}.", "must hold {offered}"),
            "failed_output": ("{name}: {error} {why}", "not given"),
            "records_block": ("{records} ({records.total})", "records_block names a field it is not given"),
            "misrouted_output": ("The app's.", "must hold {name}"),
            "repeated_call_output": ("Again.", "must hold {output}"),
            "conductor_prompt": ("Place them.", "must hold {max_running}"),
            "cut_output": ("[cut]", "must hold {chars}"),
            "no_task": (
                {**reasons, "no_task": "no {task} in {work} {work[id]}"},
                "no_task names a field it is not given",
            ),
            "no_work": ({**reasons, "no_work": "no such work"}, "no_work must hold {work}"),
            "over_limit": ({**reasons, "over_limit": "too many"}, "over_limit must hold {max_running}"),
            "refusal_reasons": ({"ended": "ended"}, "refusal_reasons must hold exactly"),
            "outcomes.started": (
                {**outcomes, "started": "Started {work_id} with {tasks}."},
                r"outcomes.started must hold \{work\}",
            ),
            "outcomes.refused": ({**outcomes, "refused": "Not applied."}, "outcomes.refused must hold {reason}"),
            "round_cap_note": (" ", "round_cap_note\n.*must be non-empty"),
        }
        for key, (value, message) in cases.items():
            with self.subTest(key), self.assertRaisesRegex(ValueError, message):
                parsed(**{"refusal_reasons" if key in reasons else key.partition(".")[0]: value})
        settings = parsed(outcomes={**outcomes, "refused": "Not applied: {reason}"})
        self.assertEqual(settings.texts.refusal("no_task", task="W1.7", work="W1"), "Not applied: no task W1.7 in W1")
