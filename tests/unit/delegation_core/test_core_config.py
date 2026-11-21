# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The core's settings and texts, and the conductor's control vocabulary."""

import copy
import json
import unittest

from examples.frontend_backend_live.delegation_core.backend import port
from examples.frontend_backend_live.delegation_core.backend.control import ControlVocabulary
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
    def test_selected_agent_receives_only_its_prompt_defaults(self):
        plain = {**BACKEND, "harness": {"agent": "acp"}}
        self.assertNotIn("agents", parsed(backend=plain).backend.blocks["harness"])
        acp = parsed(backend={**BACKEND, "harness": {"agent": "hermes"}})
        self.assertTrue(acp.backend.blocks["harness"]["agents"]["hermes"]["command_guard"])
        explicit = {**plain, "harness": {**plain["harness"], "agents": {"wrong-agent": {"x": 1}}}}
        self.assertEqual(parsed(backend=explicit).backend.blocks["harness"]["agents"], {"wrong-agent": {"x": 1}})

    def test_one_control_envelope_retains_each_typed_action(self):
        self.assertEqual([tool["name"] for tool in vocabulary().conductor_tools()], ["submit_decision"])
        actions = [
            {
                "action": "start",
                "delegations": ["d1"],
                "tasks": [{"brief": "menu"}, {"brief": "hours", "continues": "W0.1"}],
            },
            {"action": "steer", "delegations": ["d2"], "targets": [{"work_id": "W1", "task_ids": ["W1.1", "W1.3"]}]},
            {"action": "stop", "delegations": ["d2"], "targets": [{"work_id": "W2"}]},
            {"action": "caller_message", "kind": "status", "delegations": ["d3"], "targets": [{"work_id": "W1"}]},
        ]
        expected = port.CompleteDecision(
            (
                port.Start(("d1",), (port.TaskSpec("menu"), port.TaskSpec("hours", "W0.1"))),
                port.Steer(("d2",), (port.WorkTarget("W1", ("W1.1", "W1.3")),)),
                port.Cancel(("d2",), (port.WorkTarget("W2"),)),
                port.CallerMessage(("d3",), "status", targets=(port.WorkTarget("W1"),)),
            )
        )
        self.assertEqual(vocabulary().decision("submit_decision", json.dumps({"actions": actions})), expected)
        submitted_value = "private submitted value"
        invalid = [
            ({}, "actions: Field required"),
            ({"actions": []}, "actions must contain 1 or more valid entries"),
            ({"actions": submitted_value}, "actions must be a JSON array. Received a string."),
            ({"actions": [{"action": submitted_value}]}, "actions.0 requires 'action' to be one of"),
            ({"actions": [actions[0], {"action": "stop", "delegations": ["d2"]}]}, "stop.targets: Field required"),
            ({"actions": [{**actions[1], "targets": [{"work_id": "W1", "task_ids": []}]}]}, "targets.0.task_ids"),
            ({"actions": [{**actions[0], "delegations": ["d1", "d1"]}]}, "delegations: identities must be unique"),
            (
                {"actions": [{**actions[0], "tasks": [{"brief": 1}]}]},
                "tasks.0.brief must be a string. Received an integer.",
            ),
            ({"actions": [{**actions[0], "more_actions": True}]}, "start.more_actions: Extra inputs are not permitted"),
            (
                {"actions": [{**actions[3], "message": submitted_value}]},
                "status.message: Extra inputs are not permitted",
            ),
        ]
        for arguments, problem in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises(port.DecisionRejected) as refused:
                    vocabulary().decision("submit_decision", json.dumps(arguments))
                feedback = str(refused.exception)
                self.assertIn(problem, feedback)
                self.assertNotIn(submitted_value, feedback)
                self.assertNotIn("https://errors.pydantic.dev", feedback)
        for name, arguments in (("submit_decision", "not json"), ("run_shell", "{}")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                vocabulary().decision(name, arguments)

    def test_schemas_that_drop_a_parameter_or_a_tool_are_refused(self):
        schemas = copy.deepcopy({name: dict(schema) for name, schema in TEXTS.control_tools.items()})
        del schemas["submit_decision"]["parameters"]["properties"]["actions"]
        with self.assertRaisesRegex(ValueError, "complete actions envelope"):
            vocabulary(schemas)
        with self.assertRaisesRegex(ValueError, "exactly"):
            vocabulary({})

    def test_unknown_keys_and_wrong_values_are_refused(self):
        cases = {
            "unknown key": ({"max_pending": 3}, None),
            "a fraction": ({"max_pending_delegations": 2.5}, None),
            "a boolean count": ({"context_tokens": True}, None),
            "no room for a tool output": ({"tool_output_chars": 0}, None),
            "a negative wait": ({"stop_timeout_seconds": -1}, None),
            "no keep time": ({"after_call_seconds": 0}, None),
            "a fraction of a conversation": ({"max_after_call_conversations": 1.5}, None),
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
        tools = WORDS["texts"]["work_tools"]
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
            "refusal_reasons": ({"ended": "ended"}, "refusal_reasons must hold exactly"),
            "outcomes.cancel_requested": (
                {**outcomes, "cancel_requested": "Stopping {work_id} with {tasks}."},
                r"outcomes.cancel_requested must hold \{work\}",
            ),
            "outcomes.refused": ({**outcomes, "refused": "Not applied."}, "outcomes.refused must hold {reason}"),
            "round_cap_note": (" ", "round_cap_note\n.*must be non-empty"),
            "work_tools": (
                {name: tool for name, tool in tools.items() if name != "caller_message"},
                "work_tools must hold exactly",
            ),
        }
        for key, (value, message) in cases.items():
            with self.subTest(key), self.assertRaisesRegex(ValueError, message):
                parsed(**{"refusal_reasons" if key in reasons else key.partition(".")[0]: value})
        settings = parsed(outcomes={**outcomes, "refused": "Not applied: {reason}"})
        self.assertEqual(settings.texts.refusal("ended"), f"Not applied: {reasons['ended']}")
