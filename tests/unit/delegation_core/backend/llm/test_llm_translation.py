# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Tests for the chat translator: tool choices, cut-off calls, typed images and call ids."""

import json
import unittest

from delegation_core.backend.providers import BrokenStreamError
from delegation_core.backend.providers.translation import (
    ChatResponseTranslator,
    responses_items_to_messages,
    to_chat_tool_choice,
)
from tests.unit.delegation_core.backend.llm.doubles import chunk
from tests.unit.delegation_core.backend.llm.doubles import tool_fragment as fragment


def run(chunks, **options):
    translator = ChatResponseTranslator("test/model", **options)
    events = translator.start()
    for each in chunks:
        events += translator.feed(each)
    return events + translator.finish()


def items(events, item_type):
    return [e["item"] for e in events if e["type"] == "response.output_item.done" and e["item"]["type"] == item_type]


class ToolChoiceTests(unittest.TestCase):
    def test_every_form_maps_and_none_crashes(self):
        cases = [
            ({"type": "function", "name": "get_menu"}, {"type": "function", "function": {"name": "get_menu"}}),
            ({"type": "function"}, "auto"),  # no name
            ({"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "a"}]}, "auto"),
            ({"type": "web_search"}, "auto"),  # a hosted tool
            ("required", "required"),
            ("none", "none"),
            ("sometimes", "auto"),
            (None, "auto"),
            (["auto"], "auto"),
        ]
        for choice, expected in cases:
            with self.subTest(choice=choice):
                self.assertEqual(to_chat_tool_choice(choice), expected)


class CutCallTests(unittest.TestCase):
    def test_a_call_in_a_reply_cut_short_by_the_token_limit_or_a_filter_is_never_emitted(self):
        cases = {
            "length": ('{"seat": "1', "max_output_tokens"),  # the call itself was cut
            "content_filter": ('{"seat": "1A"}', "content_filter"),  # a whole call, never emitted either
        }
        for finish, (arguments, reason) in cases.items():
            with self.subTest(finish):
                call = chunk(tool_calls=[fragment(0, "c1", "book", arguments)])
                events = run([chunk("Booking"), call, chunk(finish=finish)])
                self.assertEqual(items(events, "function_call"), [])
                self.assertEqual(events[-1]["type"], "response.incomplete")
                self.assertEqual(events[-1]["response"]["incomplete_details"], {"reason": reason})
                self.assertEqual(events[-1]["response"]["output"][0]["type"], "message")

    def test_a_stream_that_ends_inside_a_call_is_broken(self):
        translator = ChatResponseTranslator("test/model")
        translator.start()
        translator.feed(chunk(tool_calls=[fragment(0, "c1", "book", '{"seat": ')]))
        with self.assertRaises(BrokenStreamError):
            translator.finish()

    def test_with_one_call_per_reply_a_cut_second_call_never_costs_the_whole_first(self):
        chunks = [
            chunk(tool_calls=[fragment(0, "c1", "book", '{"seat": "1A"}')]),
            chunk(tool_calls=[fragment(1, "c2", "pay", '{"card": "4')]),
            chunk(finish="length"),  # after the reply ended with its first call: it changes nothing
        ]
        events = run(chunks, single_call=True)
        [call] = items(events, "function_call")
        self.assertEqual((call["name"], json.loads(call["arguments"])), ("book", {"seat": "1A"}))
        self.assertEqual(events[-1]["type"], "response.completed")


class ImageTests(unittest.TestCase):
    def test_images_in_typed_input_reach_the_model(self):
        image = {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "low"}
        typed = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What is this?"}, image]}
        spoken = {"type": "message", "role": "user", "content": "And the price?"}
        messages = responses_items_to_messages("Be brief.", [typed, spoken])
        self.assertEqual(
            messages[1],
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "low"}},
                    {"type": "text", "text": "And the price?"},
                ],
            },
        )

    def test_text_only_input_stays_a_string_and_a_file_id_image_is_left_out(self):
        typed = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "a"}, {"type": "input_image", "file_id": "file_1"}],
        }
        self.assertEqual(responses_items_to_messages(None, [typed]), [{"role": "user", "content": "a"}])


class CallIdTests(unittest.TestCase):
    def test_our_own_call_ids_replace_the_providers(self):
        reused = []
        for _ in range(2):  # the provider sends the same id in two rounds
            events = run([chunk(tool_calls=[fragment(0, "call_0", "get_menu", "{}")]), chunk(finish="tool_calls")])
            reused += [call["call_id"] for call in items(events, "function_call")]
        self.assertEqual(len(set(reused)), 2)
        self.assertNotIn("call_0", reused)
        self.assertTrue(all(call_id.startswith("call_") for call_id in reused))

    def test_two_calls_of_one_reply_get_distinct_ids_and_the_arguments_match_the_deltas(self):
        events = run(
            [
                chunk(tool_calls=[fragment(0, "x", "get_menu", '{"cate'), fragment(1, None, "manage_cart", "{}")]),
                chunk(tool_calls=[fragment(0, arguments='gory": "tea"}')]),
                chunk(finish="tool_calls"),
            ]
        )
        calls = items(events, "function_call")
        self.assertEqual([call["name"] for call in calls], ["get_menu", "manage_cart"])
        self.assertNotEqual(calls[0]["call_id"], calls[1]["call_id"])
        self.assertEqual(json.loads(calls[0]["arguments"]), {"category": "tea"})
        added = [
            e["item"]["call_id"] for e in events if e["type"] == "response.output_item.added" and "call_id" in e["item"]
        ]
        self.assertEqual(added, [call["call_id"] for call in calls])


if __name__ == "__main__":
    unittest.main()
