# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import pytest

from examples.shared.vllm_streaming_input import realtime

TOOL = {"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {}}}


def _session(**fields) -> dict:
    return {**realtime.new_session("m"), **fields}


def test_function_calls_join_the_answer_they_follow() -> None:
    items = [
        realtime.user_item("Weather?"),
        realtime.normalize_item(
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking."}]}
        ),
        realtime.normalize_item({"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{}"}),
        realtime.normalize_item({"type": "function_call_output", "call_id": "c1", "output": "Sunny"}),
    ]

    assert realtime.chat_messages(items, "Be brief.") == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "Sunny"},
    ]


def test_session_options_map_to_chat_completion_fields() -> None:
    session = _session(
        tools=[TOOL],
        tool_choice={"type": "function", "name": "get_weather"},
        max_output_tokens=64,
        reasoning={"effort": "minimal"},
        chat_completion_options={"temperature": 0.2, "response_format": {"type": "json_object"}},
    )

    fields = realtime.chat_request_fields(session)

    assert fields["tools"][0]["function"]["name"] == "get_weather"
    assert fields["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}
    assert fields["max_completion_tokens"] == 64
    assert fields["reasoning_effort"] == "minimal"
    assert fields["chat_template_kwargs"] == {"enable_thinking": False}
    assert fields["temperature"] == 0.2
    assert fields["response_format"] == {"type": "json_object"}


def test_explicit_template_kwargs_win_over_reasoning_effort() -> None:
    session = _session(
        reasoning={"effort": "high"}, chat_completion_options={"chat_template_kwargs": {"enable_thinking": False}}
    )

    assert realtime.chat_request_fields(session)["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize(
    "fields",
    [
        {"type": "transcription"},
        {"output_modalities": ["audio"]},
        {"max_output_tokens": 0},
        {"reasoning": {"effort": "none"}},
        {"chat_completion_options": {"logprobs": True}},
    ],
)
def test_unsupported_sessions_are_rejected(fields) -> None:
    with pytest.raises(ValueError):
        realtime.validate_session(_session(**fields))


def test_response_events_stream_text_and_calls_then_close_them() -> None:
    response = realtime.ResponseEvents("item_user")
    call = {"index": 0, "id": "c1", "function": {"name": "get_weather", "arguments": "{}"}}
    events = response.started()
    events += response.chunk({"choices": [{"delta": {"content": "Checking."}}]})
    events += response.chunk({"choices": [{"delta": {"tool_calls": [call]}, "finish_reason": "tool_calls"}]})
    events += response.chunk({"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}})
    closing, items = response.finished("completed")

    assert [event["type"] for event in events] == [
        "response.created",
        "response.output_item.added",
        "conversation.item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_item.added",
        "conversation.item.added",
        "response.function_call_arguments.delta",
    ]
    assert [event["type"] for event in closing] == [
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "conversation.item.done",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "conversation.item.done",
        "response.done",
    ]
    assert response.finish_reason == "tool_calls"
    assert [item["type"] for item in items] == ["message", "function_call"]
    assert closing[3]["previous_item_id"] == "item_user"
    assert closing[-1]["response"]["usage"] == {"input_tokens": 9, "output_tokens": 3, "total_tokens": 12}
