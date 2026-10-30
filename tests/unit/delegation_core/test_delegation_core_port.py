# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The backend port against the core: its helpers, the control vocabulary, the core's settings and the tables."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from delegation_core.api.local import LocalResponsesClient
from delegation_core.api.service import ResponsesService
from delegation_core.app import build_core
from delegation_core.backend.adapters import ADAPTERS
from delegation_core.backend.control import ControlVocabulary
from delegation_core.backend.port import (
    Answer,
    Cancel,
    OneShotSource,
    Start,
    Steer,
    calls_a_function,
    function_call_item,
    message_phase,
)
from delegation_core.config import BackendChoice, CoreSettings, ModelEndpoint
from delegation_core.im.input import BackendTask
from delegation_core.tables import pick
from tests.unit.contracts.responses import ResponsesClientContract
from tests.unit.delegation_core.emulator import EmulatorSource

CORE_YAML = Path(__file__).resolve().parents[3] / "src/examples/frontend_backend_live/core.yaml"


# ------------------------------------------------------------------------------------------ the core's client
def emulator_core():
    """A core that ``build_core`` makes from the shipped file, on the emulator: one source serves every session."""
    raw = {"slot": "thinker-llm", "prompt": "thinker", "adapter": "emulator", "one_shot_adapter": "emulator"}
    settings = replace(CoreSettings.load(CORE_YAML), backend=BackendChoice.parse(raw))
    endpoint = ModelEndpoint(model="test/model", base_url="http://localhost:1/v1")
    return build_core(
        settings,
        endpoints=lambda model: endpoint,
        defaults=BackendTask("test/model", "Do the task.", []),
        adapters={"emulator": lambda: EmulatorSource},
    )


def local_client(core):
    """The in-process client of ``core``, as a root puts them together: it carries the core's close."""
    service = ResponsesService(core.session, stateless_session=core.one_shot_session, backend=core)
    return LocalResponsesClient(service, close_core=core.aclose)


class LocalResponsesClientContractTests(ResponsesClientContract, unittest.IsolatedAsyncioTestCase):
    """The in-process client of a core that ``build_core`` makes from the shipped file, on the emulator."""

    async def make_client(self):
        client = local_client(emulator_core())
        self.addAsyncCleanup(client.aclose)  # runs after the conversation's delete: the core closes last
        return client


class LocalResponsesClientCloseTests(unittest.IsolatedAsyncioTestCase):
    async def test_closing_the_client_closes_its_core_whose_source_releases_what_it_holds(self):
        core = emulator_core()
        await local_client(core).aclose()
        self.assertTrue(core.source.closed)

    async def test_a_client_whose_root_handed_it_no_close_has_nothing_to_release(self):
        core = emulator_core()
        self.addAsyncCleanup(core.aclose)
        await LocalResponsesClient(ResponsesService(core.session, backend=core)).aclose()
        self.assertFalse(core.source.closed)


# --------------------------------------------------------------------------------------------- the port's helpers
class PortHelperTests(unittest.TestCase):
    def test_a_function_call_is_seen_in_its_item_and_its_arguments(self):
        item = {"type": "function_call", "call_id": "c1", "name": "get_menu", "arguments": "{}"}
        done = {"type": "response.output_item.done", "item": item}
        self.assertTrue(calls_a_function(done))
        self.assertTrue(calls_a_function({"type": "response.function_call_arguments.delta", "delta": "{"}))
        self.assertFalse(calls_a_function({"type": "response.output_item.done", "item": {"type": "message"}}))
        self.assertEqual(function_call_item(done), item)
        self.assertIsNone(function_call_item({"type": "response.output_item.added", "item": item}))

    def test_a_message_without_a_phase_counts_as_the_answer(self):
        self.assertEqual(message_phase({"type": "message"}), "final_answer")
        self.assertEqual(message_phase({"type": "message", "phase": "commentary"}), "commentary")


# ------------------------------------------------------------------------------------------- the control vocabulary
FAILED = "{name} broke: {error}"  # the answer to a control call that raised (texts.failed_output)


def control_schemas():
    def schema(required, optional=()):
        properties = {name: {"type": "string"} for name in (*required, *optional)}
        return {"description": "A control tool.", "parameters": {"type": "object", "properties": properties}}

    return {
        "start_work": schema(("delegations",), ("tasks",)),
        "steer_work": schema(("work_id", "delegations"), ("task_id",)),
        "cancel_work": schema(("work_id",), ("task_id",)),
        "answer_status": schema(("delegation_id", "text")),
    }


class ControlVocabularyTests(unittest.TestCase):
    def test_each_control_call_makes_its_decision(self):
        vocabulary = ControlVocabulary(control_schemas(), "Place the requests.", "Not applied: idle.", FAILED)
        self.assertEqual(
            (vocabulary.prompt, vocabulary.idle, vocabulary.failed),
            ("Place the requests.", "Not applied: idle.", FAILED),
        )
        self.assertEqual([tool["name"] for tool in vocabulary.conductor_tools()], list(control_schemas()))
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
                self.assertEqual(vocabulary.decision(name, json.dumps(arguments)), decision)

    def test_a_malformed_call_is_refused(self):
        vocabulary = ControlVocabulary(control_schemas(), "Place the requests.", "Not applied: idle.", FAILED)
        for name, arguments in (
            ("start_work", "{}"),
            ("start_work", '{"delegations": []}'),
            ("steer_work", '{"work_id": "W1"}'),
            ("answer_status", '{"delegation_id": "d1", "text": ""}'),
            ("cancel_work", "not json"),
            ("run_shell", "{}"),
        ):
            with self.subTest(name=name, arguments=arguments), self.assertRaises(ValueError):
                vocabulary.decision(name, arguments)

    def test_a_schema_that_drops_a_parameter_is_refused_when_the_core_is_built(self):
        schemas = control_schemas()
        del schemas["steer_work"]["parameters"]["properties"]["work_id"]
        with self.assertRaisesRegex(ValueError, "steer_work"):
            ControlVocabulary(schemas, "Place the requests.", "Not applied: idle.", FAILED)
        with self.assertRaisesRegex(ValueError, "exactly"):
            ControlVocabulary(
                {name: schema for name, schema in control_schemas().items() if name != "cancel_work"},
                "Place them.",
                "Not applied: idle.",
                FAILED,
            )


# ------------------------------------------------------------------------------------------------ the core's settings
def texts():
    return {
        "voice_notes": {"default": "\n## Voice channel\n"},
        "result_formats": {"plain": "", "speak_display": "Two parts."},
        "handover_line": "Handled together ({delegation}).",
        "cancelled_line": "Stopped.",
        "unconfirmed_line": "Not confirmed.",
        "owed_output": "Outcome unknown.",
        "not_run_output": "Not run.",
        "stopped_output": "Stopped.",
        "clash_output": "Two of a name.",
        "unknown_tool_output": "No {name}; try {offered}.",
        "failed_output": "{name}: {error}",
        "misrouted_output": "{name} is the app's.",
        "repeated_call_output": "{output} (again)",
        "cut_output": "[{chars} cut]",
        "none": "none",
        "records_block": "Work so far:\n{records}",
        "closing_line": "Say where it stands.",
        "task_line": "Task: {task}",
        "round_cap_note": "Answer now.",
        "conductor_prompt": "Place the requests; {max_running} run at once.",
        "outcomes": {
            **{key: key for key in ("started", "steered", "restarted", "cancel_requested", "ended", "answered")},
            "refused": "refused: {reason}",
        },
        "refusal_reasons": {
            "ended": "ended",
            "unplaced": "not among {delegations}",
            "no_work": "no {work}",
            "no_task": "no {task} in {work}",
            "no_open_work": "no open work",
            "over_limit": "over {max_running}",
            "idle": "idle",
        },
        "control_tools": control_schemas(),
        "work_tools": {
            "show": {"description": "Show it.", "parameters": {"type": "object", "properties": {}}},
            "cancel_work": {"description": "Stop it.", "parameters": {"type": "object", "properties": {}}},
        },
    }


class CoreSettingsTests(unittest.TestCase):
    def settings(self, **changes):
        data = {"texts": texts(), "backend": {"slot": "thinker-llm", "prompt": "thinker"}}
        for key, value in changes.items():
            data[key] = value
        return CoreSettings.parse(data)

    def test_the_defaults_and_the_adapter_blocks(self):
        settings = self.settings(
            backend={"slot": "s", "prompt": "p", "llm": {"conductor": "steer"}, "tools": {"cafe": {}}}
        )
        self.assertEqual((settings.backend.adapter, settings.backend.one_shot_adapter), ("llm", "llm"))
        self.assertEqual(dict(settings.backend.blocks["llm"]), {"conductor": "steer"})
        self.assertEqual(list(settings.backend.tools), ["cafe"])
        delegation = settings.delegation
        self.assertEqual(
            (
                delegation.client_tool_timeout_seconds,
                delegation.work_timeout_seconds,
                delegation.backend_calls,
                delegation.display_label,
                delegation.records_max_works,
                delegation.context_tokens,
                delegation.tool_output_chars,
            ),
            (60.0, 600.0, "items", "delegation_core", 5, 24_000, 8000),
        )
        # The records may list no ended Work.
        self.assertEqual(self.settings(delegation={"records_max_works": 0}).delegation.records_max_works, 0)

    def test_unknown_keys_fractions_and_missing_names_are_refused(self):
        cases = {
            "unknown key": {"delegation": {"max_pending": 3}},
            "a fraction": {"delegation": {"max_pending_delegations": 2.5}},
            "a boolean count": {"delegation": {"context_tokens": True}},
            "no room for a tool output": {"delegation": {"tool_output_chars": 0}},
            "a negative wait": {"delegation": {"stop_timeout_seconds": -1}},
            "an unknown voice note": {"delegation": {"voice_note": "v9"}},
            "an unknown format": {"delegation": {"result_format": "html"}},
            "an unknown work tool": {"delegation": {"work_tools": ["run_shell"]}},
            "a scalar adapter block": {"backend": {"slot": "s", "prompt": "p", "llm": "fast"}},
            "no slot": {"backend": {"prompt": "p"}},
        }
        for name, changes in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.settings(**changes)

    def test_no_deadline_and_no_grace_are_written_null(self):
        settings = self.settings(
            delegation={"client_tool_timeout_seconds": None, "close_grace_seconds": None, "work_timeout_seconds": None}
        )
        self.assertIsNone(settings.delegation.client_tool_timeout_seconds)
        self.assertIsNone(settings.delegation.close_grace_seconds)
        self.assertIsNone(settings.delegation.work_timeout_seconds)

    def test_the_texts_keep_their_template_fields(self):
        cases = {
            "a handover with no taker": ("handover_line", "Handled together.", "must hold {delegation}"),
            "a brief with no task": ("task_line", "Task:", "must hold {task}"),
            "an unknown tool with no names offered": ("unknown_tool_output", "No {name}.", "must hold {offered}"),
            "a failure naming a field it is not given": ("failed_output", "{name}: {error} {why}", "not given"),
            "a misrouted call with no name": ("misrouted_output", "The app's.", "must hold {name}"),
            "a repeated call with no output": ("repeated_call_output", "Again.", "must hold {output}"),
            "a conductor prompt with no limit": ("conductor_prompt", "Place them.", "must hold {max_running}"),
            "a cut with no count": ("cut_output", "[cut]", "must hold {chars}"),
            "a refusal with no reason": (
                "outcomes",
                {**texts()["outcomes"], "refused": "Not applied."},
                "outcomes.refused must hold {reason}",
            ),
            "a refusal reason left out": ("refusal_reasons", {"ended": "ended"}, "refusal_reasons must hold exactly"),
            "a reason with no work": (
                "refusal_reasons",
                {**texts()["refusal_reasons"], "no_work": "no such work"},
                "no_work must hold {work}",
            ),
            "a limit with no number": (
                "refusal_reasons",
                {**texts()["refusal_reasons"], "over_limit": "too many"},
                "over_limit must hold {max_running}",
            ),
            "no round-cap note": ("round_cap_note", " ", "round_cap_note must be non-empty"),
        }
        for name, (key, value, message) in cases.items():
            broken = {**texts(), key: value}
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                CoreSettings.parse({"texts": broken, "backend": {"slot": "s", "prompt": "p"}})

    def test_a_refusal_is_the_refused_outcome_with_its_reason(self):
        given = {**texts(), "outcomes": {**texts()["outcomes"], "refused": "Not applied: {reason}"}}
        settings = CoreSettings.parse({"texts": given, "backend": {"slot": "s", "prompt": "p"}})
        self.assertEqual(settings.texts.refusal("no_task", task="W1.7", work="W1"), "Not applied: no W1.7 in W1")
        self.assertEqual(settings.texts.refusal("ended"), "Not applied: ended")


class ProfileTests(unittest.TestCase):
    """A profile extends another file and sets only what differs."""

    def write(self, folder: Path, name: str, data: dict) -> Path:
        path = folder / name
        path.write_text(json.dumps(data), encoding="utf-8")  # JSON is YAML
        return path

    def test_a_profile_merges_mappings_and_replaces_the_rest(self):
        with tempfile.TemporaryDirectory() as folder:
            base = {"texts": texts(), "backend": {"slot": "s", "prompt": "p", "llm": {"conductor": "steer"}}}
            self.write(Path(folder), "core.yaml", base)
            self.write(
                Path(folder),
                "core.agent.yaml",
                {
                    "extends": "core.yaml",
                    "delegation": {"result_format": "speak_display", "client_tool_timeout_seconds": None},
                    "texts": {"conductor_prompt": "Place them by the records, {max_running} at once."},
                    "backend": {"adapter": "llm-agent", "llm-agent": {"max_running": 3}},
                },
            )
            harness = self.write(
                Path(folder), "core.harness.yaml", {"extends": "core.agent.yaml", "backend": {"adapter": "harness"}}
            )
            settings = CoreSettings.load(harness)
        self.assertEqual(settings.backend.adapter, "harness")
        self.assertEqual(sorted(settings.backend.blocks), ["llm", "llm-agent"])
        self.assertEqual(settings.texts.conductor_prompt, "Place them by the records, {max_running} at once.")
        self.assertEqual(settings.texts.handover_line, "Handled together ({delegation}).")
        self.assertEqual(settings.delegation.result_format, "speak_display")
        self.assertIsNone(settings.delegation.client_tool_timeout_seconds)

    def test_a_loop_or_a_missing_file_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            first = self.write(Path(folder), "a.yaml", {"extends": "b.yaml"})
            self.write(Path(folder), "b.yaml", {"extends": "a.yaml"})
            with self.assertRaisesRegex(ValueError, "extend themselves"):
                CoreSettings.load(first)
            lost = self.write(Path(folder), "c.yaml", {"extends": "missing.yaml"})
            with self.assertRaisesRegex(ValueError, "Cannot read"):
                CoreSettings.load(lost)


# ------------------------------------------------------------------------------------- the tables and the builder
class TableTests(unittest.TestCase):
    def test_pick_names_what_is_known(self):
        self.assertEqual(pick({"llm": 1}, "llm", "backend adapter"), 1)
        with self.assertRaisesRegex(ValueError, r"Unknown backend adapter 'hermes'; known: harness, llm"):
            pick({"llm": 1, "harness": 2}, "hermes", "backend adapter")

    def test_every_adapter_of_the_table_loads_its_source(self):
        # Each adapter is built, so each entry loads the source class its package exports.
        sources = {"llm": "LlmSource", "llm-agent": "AgentSource", "harness": "HarnessSource"}
        for name, source in sources.items():
            with self.subTest(name):
                self.assertEqual(pick(ADAPTERS, name, "backend adapter")().__name__, source)

    def test_the_one_shot_capability_is_a_method(self):
        class OneShot:
            def new_one_shot(self):
                return None

        self.assertIsInstance(OneShot(), OneShotSource)
        self.assertNotIsInstance(object(), OneShotSource)
