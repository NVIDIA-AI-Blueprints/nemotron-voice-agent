# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The ID-shape hints, an app output step: a "not found" output gets the failed id in the shape the app's ids take."""

import json
import unittest

from examples.frontend_backend_live.delegation_core.app import app_output_steps, log_trace
from examples.frontend_backend_live.delegation_core.backend.recovery import ShapeHints
from examples.frontend_backend_live.delegation_core.config import CoreSettings
from tests.unit.core_helpers import CORE_TEXTS, core_block

# The step is off in the shipped settings; a deployment turns it on with its block. Its block (with the hint's words
# from prompts.yaml), and the hint it adds.
NOT_FOUND_PATTERN = r"(?i)\b(not found|does not exist|no such)\b"


def _shape_hints_block() -> dict:
    steps = {"shape_hints": {"not_found_pattern": NOT_FOUND_PATTERN}}
    settings = CoreSettings.parse(core_block(backend={"app_output_steps": steps}), CORE_TEXTS)
    return dict(settings.backend.app_output_steps["shape_hints"])


SHAPE_HINTS = _shape_hints_block()
HINT = SHAPE_HINTS["hint"]


def tool(name: str, **descriptions: str) -> dict:
    properties = {parameter: {"type": "string", "description": text} for parameter, text in descriptions.items()}
    return {"type": "function", "name": name, "parameters": {"type": "object", "properties": properties}}


USER = tool("get_member", user_id="The user id, such as 'lee_park_42'.")
PHONE = tool("get_customer_by_phone", phone_number="The customer's phone number.")


def pair(name: str, call_id: str, text, **arguments) -> tuple[dict, dict]:
    """A call of ``name`` with ``arguments``, and the app's output ``text`` for it."""
    call = {"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(arguments)}
    return call, {"type": "function_call_output", "call_id": call_id, "output": text}


class ShapeHintsTests(unittest.TestCase):
    def setUp(self):
        self.traces = []
        self.fresh()

    def fresh(self):  # a new session's step
        self.hints = ShapeHints.from_settings(SHAPE_HINTS, observe=lambda kind, **fields: self.traces.append(fields))

    def read(self, *pairs, tools=(USER, PHONE)):
        """What the backend reads of each output, by call id."""
        return {call_id: item["output"] for call_id, item in self.hints.outputs(list(pairs), list(tools)).items()}

    def test_a_not_found_id_gets_the_shape_its_tool_shows(self):
        cases = {
            # the example the description gives, the value not found, the form hinted (None: no hint)
            "the case": ("lee_park_42", "Maya_Ross_3817", "maya_ross_3817"),
            "the separator": ("lee_park_42", "maya ross 3817", "maya_ross_3817"),
            "a leading mark": ("#A1234567", "A1234567", "#A1234567"),
            "a form without joiners": ("#A1234567", "a 123 4567", "#A1234567"),
            "digits grouped as the example": ("555-123-2002", "5551232002", "555-123-2002"),
            "digits the example does not group": ("5551232002", "555 123 2002", "5551232002"),
            "groups that do not fit": ("555-123-2002", "55512320", None),
            "the example's form already": ("lee_park_42", "maya_ross_3817", None),
            "another kind of value": ("lee_park_42", "5551232002", None),
            "text that is no id": ("lee_park_42", "maya.ross@example.com", None),
        }
        for name, (example, value, hinted) in cases.items():
            with self.subTest(name):
                self.fresh()
                failed = f"Error: Record {value} not found"
                lookup = tool("lookup", key=f"The key, such as '{example}'.")
                read = self.read(pair("lookup", name, failed, key=value, date="2024-05-17"), tools=[lookup])[name]
                self.assertEqual(read, failed if hinted is None else f"{failed}\n{HINT.format(value=hinted)}")
        self.assertEqual({fields["source"] for fields in self.traces}, {"description"})

    def test_the_failed_value_is_the_named_or_only_string(self):
        failed = "Error: User Maya_Ross_3817 not found (page 10 of 12)"
        read = self.read(pair("get_member", "c1", failed, user_id="Maya_Ross_3817", page="1"))["c1"]
        self.assertEqual(read, f"{failed}\n{HINT.format(value='maya_ross_3817')}")
        read = self.read(pair("get_member", "c2", "Error: user not found", user_id="A B 1"))["c2"]
        self.assertEqual(read, f"Error: user not found\n{HINT.format(value='a_b_1')}")
        read = self.read(pair("get_member", "c3", "Error: user not found", first="Ann", user_id="Ann Lee 10"))["c3"]
        self.assertEqual(read, "Error: user not found")  # two string arguments: neither is named

    def test_the_hint_is_given_once_per_form(self):
        hint = HINT.format(value="maya_ross_3817")
        for call_id, value, hinted in (
            ("c1", "Maya_Ross_3817", True),
            ("c2", "Maya_Ross_3817", False),
            ("c3", "maya_ross_3817", False),
            ("c4", "MAYA_ROSS_3817", False),
        ):
            failed = f"Error: User {value} not found"
            read = self.read(pair("get_member", call_id, failed, user_id=value))[call_id]
            self.assertEqual(read, f"{failed}\n{hint}" if hinted else failed)

    def test_the_shape_is_learned_from_what_worked_not_errors(self):
        for index, failure in enumerate(("Error: the service is unavailable", '{"error": "rate limited"}')):
            self.read(pair("get_customer_by_phone", f"b{index}", failure, phone_number="5559870000"))
        for index, empty in enumerate(({"error": None}, {"errors": []}, {"error": ""})):  # an empty error is none
            found = json.dumps({"customer_id": "C1001", **empty})
            self.read(pair("get_customer_by_phone", f"f{index}", found, phone_number="555-123-2002"))
        self.read(pair("get_customer_by_phone", "deep", "[" * 100_000 + "]" * 100_000, phone_number="555-123-2003"))
        failed = "Customer with phone number 5551110000 not found"
        read = self.read(pair("get_customer_by_phone", "c1", failed, phone_number="5551110000"))
        self.assertEqual(read["c1"], f"{failed}\n{HINT.format(value='555-111-0000')}")
        self.assertEqual(self.traces[-1]["source"], "session")
        self.read(pair("find_member", "c2", json.dumps({"user": {"user_id": "ann_lee_10"}}), email="ann@example.com"))
        failed = "Error: User Bob Ray 22 not found"
        read = self.read(pair("get_member", "c3", failed, user_id="Bob Ray 22"), tools=())
        self.assertEqual(read["c3"], f"{failed}\n{HINT.format(value='bob_ray_22')}")

    def test_an_output_with_nothing_to_hint_passes_as_it_is(self):
        busy = "Error: the service is unavailable, try again"
        self.assertEqual(self.read(pair("get_member", "c1", busy, user_id="Maya_Ross_3817")), {"c1": busy})
        no_shape = pair("get_customer_by_phone", "c2", "Customer not found", phone_number="5551232002")
        self.assertEqual(self.read(no_shape), {"c2": "Customer not found"})
        parts = [{"type": "input_text", "text": "Error: User Maya_Ross_3817 not found"}]
        not_text = pair("get_member", "c3", parts, user_id="Maya_Ross_3817")
        self.assertEqual(self.hints.outputs([not_text], [USER]), {"c3": not_text[1]})
        given = pair("get_member", "c4", "Error: User Maya_Ross_3817 not found", user_id="Maya_Ross_3817")
        for tools in (None, ["get_member", {"name": "get_member", "parameters": "a user id"}]):  # none, or malformed
            self.assertEqual(self.hints.outputs([given], tools), {"c4": given[1]})
        self.assertEqual(self.traces, [])
        self.hints.outputs([given], [USER])
        self.assertEqual(given[1]["output"], "Error: User Maya_Ross_3817 not found")  # a hinted item is a copy

    def test_a_bad_block_fails_and_each_session_has_its_own_step(self):
        cases = {
            "a key left out": ({"hint": HINT}, "must set not_found_pattern and hint"),
            "an empty pattern": ({**SHAPE_HINTS, "not_found_pattern": " "}, "non-empty text"),
            "a pattern that does not compile": ({**SHAPE_HINTS, "not_found_pattern": "(not found"}, "regular expr"),
            "a hint with no value": ({**SHAPE_HINTS, "hint": "Try another form."}, r"must hold \{value\}"),
            "a hint naming another field": ({**SHAPE_HINTS, "hint": "{value} or {other}"}, "not given"),
            "a hint reading past its value": ({**SHAPE_HINTS, "hint": "{value} or {value[x]}"}, "not given"),
        }
        for name, (block, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                app_output_steps({"shape_hints": block})
        steps = app_output_steps({"shape_hints": SHAPE_HINTS})
        first, second = steps(log_trace), steps(log_trace)
        failed = pair("get_member", "c1", "Error: User Maya_Ross_3817 not found", user_id="Maya_Ross_3817")
        for step in (first[0], second[0]):  # what one session tried, another has not
            self.assertIn(HINT.format(value="maya_ross_3817"), step.outputs([failed], [USER])["c1"]["output"])
