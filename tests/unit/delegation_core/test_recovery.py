# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""ID recovery: the shape of an id, the failed value of a "not found" output, and the hint the backend reads."""

import json
import unittest

from examples.frontend_backend_live.delegation_core.im.recovery import (
    ShapeHints,
    apply_shape,
    description_examples,
    failed_value,
    shape_of,
    skeleton,
)
from tests.unit.delegation_core.helpers import TEXTS

NOT_FOUND = r"(?i)\b(not found|does not exist|no such)\b"


def tool(name: str, **descriptions: str) -> dict:
    """A Responses function tool whose parameters have ``descriptions``."""
    properties = {parameter: {"type": "string", "description": text} for parameter, text in descriptions.items()}
    return {"type": "function", "name": name, "parameters": {"type": "object", "properties": properties}}


USER = tool("get_member", user_id="The user id, such as 'lee_park_42'.")
BOOKING = tool("get_booking", booking_id="The booking id, such as '#A1234567'.")
PHONE = tool("get_customer_by_phone", phone_number="The customer's phone number.")


def call(name: str, call_id: str, **arguments) -> dict:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(arguments)}


def output(call_id: str, text: str) -> dict:
    return {"type": "function_call_output", "call_id": call_id, "output": text}


class ShapeTests(unittest.TestCase):
    def suggest(self, example: str, value: str) -> str | None:
        return apply_shape(value, shape_of(example))

    def test_the_case_and_the_separator_follow_the_example(self):
        self.assertEqual(self.suggest("lee_park_42", "Maya_Ross_3817"), "maya_ross_3817")
        self.assertEqual(self.suggest("lee_park_42", "maya ross 3817"), "maya_ross_3817")
        self.assertEqual(self.suggest("lee_park_42", "MAYA-ROSS-3817"), "maya_ross_3817")

    def test_a_leading_mark_and_a_joiner_less_form_follow_the_example(self):
        self.assertEqual(self.suggest("#A1234567", "A1234567"), "#A1234567")
        self.assertEqual(self.suggest("#A1234567", "a 123 4567"), "#A1234567")

    def test_digits_are_grouped_only_as_a_grouped_example_shows(self):
        self.assertEqual(self.suggest("555-123-2002", "5551232002"), "555-123-2002")
        self.assertIsNone(self.suggest("555-123-2002", "55512320"))  # the groups do not fit
        self.assertEqual(self.suggest("5551232002", "555 123 2002"), "5551232002")

    def test_no_suggestion_when_the_form_is_already_the_examples(self):
        self.assertIsNone(self.suggest("lee_park_42", "maya_ross_3817"))

    def test_no_suggestion_across_kinds_of_value(self):
        self.assertIsNone(self.suggest("lee_park_42", "5551232002"))  # digits only against letters and digits
        self.assertIsNone(self.suggest("5551232002", "Maya"))

    def test_text_that_is_no_id_has_no_shape(self):
        for text in ("maya.ross@example.com", "a" * 65, "", "one/two"):
            with self.subTest(text=text):
                self.assertIsNone(shape_of(text))

    def test_a_suggestion_never_changes_a_letter_or_a_digit(self):
        for example, value in (("lee_park_42", "K_L_7_RIVERA_5520"), ("#A1234567", "a7654321"), ("1-2", "34")):
            with self.subTest(value=value):
                suggestion = self.suggest(example, value)
                if suggestion is not None:
                    self.assertEqual(skeleton(suggestion), skeleton(value))

    def test_the_skeleton_ignores_case_separators_and_marks(self):
        self.assertEqual(skeleton("#A-12_3 4.567"), skeleton("a1234567"))


class FailedValueTests(unittest.TestCase):
    def test_the_quoted_argument_is_the_failed_one(self):
        arguments = {"user_id": "Maya_Ross_3817", "date": "2024-05-17"}
        self.assertEqual(failed_value(arguments, "Error: User Maya_Ross_3817 not found"), ("user_id", "Maya_Ross_3817"))

    def test_a_value_counts_as_quoted_only_as_a_whole_word(self):
        arguments = {"user_id": "maya_ross_3817", "page": "1"}
        output_text = "Error: User maya_ross_3817 not found (page 10 of 12)"
        self.assertEqual(failed_value(arguments, output_text), ("user_id", "maya_ross_3817"))
        self.assertEqual(
            failed_value({"booking_id": "#A1234567"}, "Booking #A1234567 not found"), ("booking_id", "#A1234567")
        )

    def test_an_unquoted_failure_names_the_only_string_argument_or_nothing(self):
        self.assertEqual(failed_value({"user_id": "a_b_1"}, "Error: user not found"), ("user_id", "a_b_1"))
        self.assertIsNone(failed_value({"first": "Ann", "zip": "02110"}, "Error: user not found"))


class DescriptionTests(unittest.TestCase):
    def test_the_first_quoted_example_of_each_parameter_in_either_tool_form(self):
        chat_form = {"type": "function", "function": BOOKING}
        examples = description_examples([USER, chat_form, PHONE])
        self.assertEqual(
            examples,
            {("get_member", "user_id"): "lee_park_42", ("get_booking", "booking_id"): "#A1234567"},
        )


class ShapeHintsTests(unittest.TestCase):
    def setUp(self):
        self.traces = []
        self.hints = ShapeHints(NOT_FOUND, TEXTS.shape_hint, lambda kind, **fields: self.traces.append((kind, fields)))

    def read(self, pairs, tools=(USER, BOOKING, PHONE)):
        return {call_id: item["output"] for call_id, item in self.hints.outputs(pairs, list(tools)).items()}

    def test_a_not_found_gets_the_form_the_description_shows(self):
        lookup = call("get_member", "c1", user_id="Maya_Ross_3817")
        read = self.read([(lookup, output("c1", "Error: User Maya_Ross_3817 not found"))])
        hint = TEXTS.shape_hint.format(value="maya_ross_3817")
        self.assertEqual(read, {"c1": f"Error: User Maya_Ross_3817 not found\n{hint}"})
        self.assertEqual(self.traces[0][0], "tools.shape_hint")
        self.assertEqual(self.traces[0][1]["source"], "description")

    def test_the_hint_is_given_once_and_never_for_a_form_already_tried(self):
        first = call("get_member", "c1", user_id="Maya_Ross_3817")
        self.read([(first, output("c1", "Error: User Maya_Ross_3817 not found"))])
        second = call("get_member", "c2", user_id="maya_ross_3817")
        read = self.read([(second, output("c2", "Error: User maya_ross_3817 not found"))])
        self.assertEqual(read, {"c2": "Error: User maya_ross_3817 not found"})
        third = call("get_member", "c3", user_id="MAYA_ROSS_3817")
        read = self.read([(third, output("c3", "Error: User MAYA_ROSS_3817 not found"))])
        self.assertEqual(read, {"c3": "Error: User MAYA_ROSS_3817 not found"})  # its form was tried in c2

    def test_a_value_that_worked_in_the_session_teaches_the_shape(self):
        found = call("get_customer_by_phone", "c1", phone_number="555-123-2002")
        self.read([(found, output("c1", '{"customer_id": "C1001"}'))])
        failed = call("get_customer_by_phone", "c2", phone_number="5559870000")
        read = self.read([(failed, output("c2", "Customer with phone number 5559870000 not found"))])
        self.assertIn(TEXTS.shape_hint.format(value="555-987-0000"), read["c2"])
        self.assertEqual(self.traces[-1][1]["source"], "session")

    def test_a_field_of_the_same_name_in_an_earlier_output_teaches_the_shape(self):
        found = call("find_member_by_email", "c1", email="ann@example.com")
        self.read([(found, output("c1", json.dumps({"user": {"user_id": "ann_lee_10"}})))], tools=())
        failed = call("get_member", "c2", user_id="Bob Ray 22")
        read = self.read([(failed, output("c2", "Error: User Bob Ray 22 not found"))], tools=())
        self.assertIn(TEXTS.shape_hint.format(value="bob_ray_22"), read["c2"])

    def test_nothing_is_added_without_a_shape_or_without_a_not_found(self):
        failed = call("get_customer_by_phone", "c1", phone_number="5551232002")
        self.assertEqual(self.read([(failed, output("c1", "Customer not found"))]), {"c1": "Customer not found"})
        refused = call("get_member", "c2", user_id="Maya_Ross_3817")
        busy = "Error: the service is unavailable, try again"
        self.assertEqual(self.read([(refused, output("c2", busy))]), {"c2": busy})
        self.assertEqual(self.traces, [])

    def test_a_retry_of_the_same_failed_form_gets_no_second_hint(self):
        hint = TEXTS.shape_hint.format(value="maya_ross_3817")
        for call_id in ("c1", "c2"):
            lookup = call("get_member", call_id, user_id="Maya_Ross_3817")
            read = self.read([(lookup, output(call_id, "Error: User Maya_Ross_3817 not found"))])
            self.assertEqual(hint in read[call_id], call_id == "c1")
        self.assertEqual(len(self.traces), 1)

    def test_an_output_that_reports_an_error_teaches_no_shape(self):
        for index, failure in enumerate(("Error: the service is unavailable", '{"error": "rate limited"}')):
            busy = call("get_customer_by_phone", f"b{index}", phone_number="5559870000")
            self.read([(busy, output(f"b{index}", failure))])
        found = call("get_customer_by_phone", "c1", phone_number="555-123-2002")
        self.read([(found, output("c1", '{"customer_id": "C1001"}'))])
        failed = call("get_customer_by_phone", "c2", phone_number="5551110000")
        read = self.read([(failed, output("c2", "Customer with phone number 5551110000 not found"))])
        self.assertIn(TEXTS.shape_hint.format(value="555-111-0000"), read["c2"])  # the error forms were not learned
        busy = call("get_customer_by_phone", "c3", phone_number="5552220000")
        self.read([(busy, output("c3", "Error: the service is unavailable"))])
        failed = call("get_customer_by_phone", "c4", phone_number="5553330000")
        read = self.read([(failed, output("c4", "Customer with phone number 5553330000 not found"))])
        self.assertIn(TEXTS.shape_hint.format(value="555-333-0000"), read["c4"])  # the dashed form still holds

    def test_an_output_too_deeply_nested_to_read_passes_and_still_counts_as_found(self):
        deep = "[" * 100_000 + "]" * 100_000
        found = call("get_customer_by_phone", "c1", phone_number="555-123-2002")
        self.assertEqual(self.read([(found, output("c1", deep))]), {"c1": deep})
        failed = call("get_customer_by_phone", "c2", phone_number="5559870000")
        read = self.read([(failed, output("c2", "Customer with phone number 5559870000 not found"))])
        self.assertIn(TEXTS.shape_hint.format(value="555-987-0000"), read["c2"])

    def test_an_output_that_is_not_text_passes_unchanged(self):
        lookup = call("get_member", "c1", user_id="Maya_Ross_3817")
        parts = [{"type": "input_text", "text": "Error: User Maya_Ross_3817 not found"}]
        item = {"type": "function_call_output", "call_id": "c1", "output": parts}
        self.assertEqual(self.hints.outputs([(lookup, item)], [USER]), {"c1": item})

    def test_the_hinted_item_is_a_copy_and_the_given_one_is_left_as_it_was(self):
        lookup = call("get_member", "c1", user_id="Maya_Ross_3817")
        given = output("c1", "Error: User Maya_Ross_3817 not found")
        self.hints.outputs([(lookup, given)], [USER])
        self.assertEqual(given["output"], "Error: User Maya_Ross_3817 not found")


if __name__ == "__main__":
    unittest.main()
