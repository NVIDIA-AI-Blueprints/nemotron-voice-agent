# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103

"""Chat-completions to Responses translation and the chat-completions backend provider."""

import json
import unittest
from dataclasses import replace

import httpx
from openai import APIError, APITimeoutError, BadRequestError

from delegation_core.backend.providers.chat_completions import ChatCompletionsBackend
from delegation_core.backend.providers.translation import (
    ChatResponseTranslator,
    responses_items_to_messages,
    split_reasoning,
    strip_reasoning,
    thinking_enabled,
    to_chat_tool_choice,
    to_chat_tools,
)
from delegation_core.backend.tools.cafe.schemas import CAFE_TOOLS
from tests.unit.core_helpers import ENDPOINT, LIMITS, FakeChat, _timeout, chunk, tool_fragment


def run(translator, chunks):
    events = translator.start()
    for item in chunks:
        events += translator.feed(item)
    return events + translator.finish()


def kinds(events):
    return [e["type"] for e in events]


# ---------------------------------------------------------------- request building
def test_items_become_chat_messages_with_tool_calls_and_results():
    items = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "a"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "b"}]},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_menu",
            "arguments": '{"category":"pastry"}',
        },
        {"type": "function_call", "call_id": "call_2", "name": "manage_cart", "arguments": ""},
        {"type": "function_call_output", "call_id": "call_1", "output": '{"items":[]}'},
        {"type": "function_call_output", "call_id": "call_2", "output": "{}"},
        {"type": "reasoning", "summary": []},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Done."}],
        },
    ]
    messages = responses_items_to_messages("Be brief.", items)
    assert messages[0] == {"role": "system", "content": "Be brief."}
    assert messages[1] == {"role": "user", "content": "a\nb"}  # adjacent users merge
    calls = messages[2]["tool_calls"]
    assert [c["id"] for c in calls] == ["call_1", "call_2"]
    assert calls[1]["function"]["arguments"] == "{}"  # empty arguments stay valid JSON
    assert messages[3] == {"role": "tool", "tool_call_id": "call_1", "content": '{"items":[]}'}
    assert messages[4]["tool_call_id"] == "call_2"
    assert messages[5] == {"role": "assistant", "content": "Done."}
    assert len(messages) == 6  # the reasoning item is dropped


def test_assistant_text_then_call_share_one_message():
    items = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Checking."}],
        },
        {"type": "function_call", "call_id": "c", "name": "get_menu", "arguments": "{}"},
    ]
    messages = responses_items_to_messages(None, items)
    assert len(messages) == 1
    assert messages[0]["content"] == "Checking."
    assert messages[0]["tool_calls"][0]["id"] == "c"


def test_hosted_tools_are_dropped_and_function_tools_converted():
    tools, dropped = to_chat_tools(
        [
            {"type": "web_search"},
            {
                "type": "function",
                "name": "get_menu",
                "description": "Look up the menu",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            },
        ]
    )
    assert dropped == ["web_search"]
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "get_menu",
                "description": "Look up the menu",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assert to_chat_tool_choice({"type": "function", "name": "get_menu"}) == {
        "type": "function",
        "function": {"name": "get_menu"},
    }
    assert to_chat_tool_choice("required") == "required"
    assert to_chat_tool_choice(None) == "auto"


def test_reasoning_split_handles_templates_that_open_think_in_the_prompt():
    assert split_reasoning("plan the answer</think>The total is $4.")[0] == "The total is $4."
    assert strip_reasoning("<think>hmm</think>Hello") == "Hello"
    assert strip_reasoning("<think>never closed") == ""
    assert strip_reasoning("plain") == "plain"


def test_thinking_flag_is_read_from_chat_template_kwargs():
    on = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
    off = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    assert thinking_enabled(on) and not thinking_enabled(off) and not thinking_enabled({})


# ----------------------------------------------------------------- event stream
def test_text_stream_emits_the_responses_lifecycle_in_order():
    t = ChatResponseTranslator("nvidia/test", response_id="resp_x")
    events = run(
        t,
        [
            chunk("The pastries"),
            chunk(" are croissants."),
            chunk(finish="stop"),
            chunk(usage={"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}),
        ],
    )
    assert kinds(events) == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [e["sequence_number"] for e in events] == list(range(1, len(events) + 1))
    done = next(e for e in events if e["type"] == "response.output_item.done")
    assert done["item"]["content"][0]["text"] == "The pastries are croissants."
    assert t.answer_text == "The pastries are croissants."
    completed = events[-1]["response"]
    assert completed["id"] == "resp_x" and completed["status"] == "completed"
    assert completed["usage"] == {"input_tokens": 7, "output_tokens": 5, "total_tokens": 12}


def test_streamed_tool_call_fragments_become_one_function_call_item():
    t = ChatResponseTranslator("nvidia/test")
    events = run(
        t,
        [
            chunk(tool_calls=[tool_fragment(0, "call_abc", "get_menu", "")]),
            chunk(tool_calls=[tool_fragment(0, arguments='{"cate')]),
            chunk(tool_calls=[tool_fragment(0, arguments='gory":"pastry"}')]),
            chunk(finish="tool_calls"),
        ],
    )
    done = [e for e in events if e["type"] == "response.output_item.done"]
    assert len(done) == 1
    item = done[0]["item"]
    # The call carries our own id, not the provider's (delegation_core/backend/llm/test_llm_translation.py).
    assert item["type"] == "function_call" and item["call_id"].startswith("call_") and item["call_id"] != "call_abc"
    assert item["name"] == "get_menu" and json.loads(item["arguments"]) == {"category": "pastry"}
    deltas = "".join(e["delta"] for e in events if e["type"] == "response.function_call_arguments.delta")
    assert deltas == item["arguments"]
    args_done = next(e for e in events if e["type"] == "response.function_call_arguments.done")
    assert args_done["arguments"] == item["arguments"]
    assert events[-1]["type"] == "response.completed"


def test_parallel_calls_get_distinct_ids_of_our_own():
    t = ChatResponseTranslator("nvidia/test")
    events = run(
        t,
        [
            chunk(
                tool_calls=[
                    tool_fragment(0, "call_1", "get_menu", "{}"),
                    tool_fragment(1, None, "manage_cart", '{"action":"view"}'),
                ]
            ),
            chunk(finish="tool_calls"),
        ],
    )
    items = [e["item"] for e in events if e["type"] == "response.output_item.done"]
    assert [i["name"] for i in items] == ["get_menu", "manage_cart"]
    assert all(i["call_id"].startswith("call_") and i["call_id"] != "call_1" for i in items)  # ours
    assert items[0]["call_id"] != items[1]["call_id"]
    assert [e["output_index"] for e in events if e["type"] == "response.output_item.added"] == [
        0,
        1,
    ]


def test_text_before_a_call_gets_the_first_output_index():
    t = ChatResponseTranslator("nvidia/test")
    events = run(
        t,
        [
            chunk("One moment."),
            chunk(tool_calls=[tool_fragment(0, "call_1", "get_menu", "{}")]),
            chunk(finish="tool_calls"),
        ],
    )
    added = {e["item"]["type"]: e["output_index"] for e in events if "added" in e["type"] and "item" in e}
    assert added == {"message": 0, "function_call": 1}


def test_buffered_thinking_never_reaches_the_wire():
    t = ChatResponseTranslator("nvidia/test", buffer_text=True)
    events = run(
        t,
        [
            chunk(None, reasoning_content="private plan"),
            chunk("hidden steps</think>The total is "),
            chunk("$4.07."),
            chunk(finish="stop"),
        ],
    )
    wire = json.dumps(events)
    assert "private plan" not in wire and "hidden steps" not in wire
    text = [e["delta"] for e in events if e["type"] == "response.output_text.delta"]
    assert text == ["The total is $4.07."]
    assert t.reasoning_chars == len("private plan")
    assert t.answer_text == "The total is $4.07."


def test_reasoning_only_output_emits_no_message():
    t = ChatResponseTranslator("nvidia/test", buffer_text=True)
    events = run(t, [chunk("still thinking with no end"), chunk(finish="length")])
    assert "response.output_item.added" not in kinds(events)
    assert events[-1]["type"] == "response.incomplete"


def test_truncated_output_is_reported_incomplete_not_completed():
    t = ChatResponseTranslator("nvidia/test")
    events = run(t, [chunk("Half an ans"), chunk(finish="length")])
    last = events[-1]
    assert last["type"] == "response.incomplete"
    assert last["response"]["incomplete_details"] == {"reason": "max_output_tokens"}


def test_nameless_call_fragment_is_not_surfaced_as_a_runnable_call():
    t = ChatResponseTranslator("nvidia/test")
    events = run(t, [chunk(tool_calls=[tool_fragment(0, "call_1", None, "{}")]), chunk(finish="stop")])
    assert "response.output_item.done" not in kinds(events)


# --------------------------------------------------------------------- the chat-completions provider
class ChatBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_round_sends_tools_history_and_translates_a_function_call(self):
        client = FakeChat(
            [
                chunk(tool_calls=[tool_fragment(0, "call_1", "get_menu", '{"category":')]),
                chunk(tool_calls=[tool_fragment(0, arguments='"tea"}')], finish="tool_calls"),
            ]
        )
        backend = ChatCompletionsBackend(ENDPOINT, LIMITS, client)
        payload = {
            "model": "test/model",
            "instructions": "Use tools.",
            "tools": CAFE_TOOLS + [{"type": "web_search"}],
            "tool_choice": "auto",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "teas?"}]}],
        }
        events = [e async for e in backend.stream(payload)]
        request = client.requests[0]
        self.assertEqual((request["temperature"], request["max_tokens"], request["tool_choice"]), (0.7, 1024, "auto"))
        self.assertEqual([t["function"]["name"] for t in request["tools"]], ["get_menu", "manage_cart", "place_order"])
        self.assertEqual([m["role"] for m in request["messages"]], ["system", "user"])
        calls = [
            e["item"]
            for e in events
            if e["type"] == "response.output_item.done" and e["item"]["type"] == "function_call"
        ]
        self.assertEqual((calls[0]["name"], json.loads(calls[0]["arguments"])), ("get_menu", {"category": "tea"}))
        self.assertEqual(events[-1]["type"], "response.completed")

    async def test_failure_before_the_first_chunk_is_retried_but_a_later_one_is_not(self):
        backend = ChatCompletionsBackend(ENDPOINT, LIMITS, FakeChat(_timeout(), [chunk("Hello", finish="stop")]))
        events = [e async for e in backend.stream({"input": []})]
        self.assertIn("response.output_text.delta", [e["type"] for e in events])

        failing = ChatCompletionsBackend(ENDPOINT, replace(LIMITS, provider_retries=0), FakeChat(_timeout()))
        with self.assertRaises(APITimeoutError):
            [e async for e in failing.stream({"input": []})]

    async def test_an_error_event_that_opens_the_stream_is_retried_but_a_rejected_request_is_not(self):
        request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
        overloaded = APIError("Service temporarily overloaded", request, body=None)
        backend = ChatCompletionsBackend(ENDPOINT, LIMITS, FakeChat(overloaded, [chunk("Hello", finish="stop")]))
        events = [e async for e in backend.stream({"input": []})]
        self.assertIn("response.output_text.delta", [e["type"] for e in events])

        rejected = BadRequestError("bad request", response=httpx.Response(400, request=request), body=None)
        refused = ChatCompletionsBackend(ENDPOINT, LIMITS, FakeChat(rejected, [chunk("Hello", finish="stop")]))
        with self.assertRaises(BadRequestError):
            [e async for e in refused.stream({"input": []})]
