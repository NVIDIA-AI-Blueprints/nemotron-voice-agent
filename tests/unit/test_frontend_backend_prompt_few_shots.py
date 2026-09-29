# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D103

"""Tests for repository-owned Frontend/Backend Talker few-shot prompts."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from examples.frontend_backend_agent import pipeline


def test_generic_prompt_loads_subject_neutral_native_repeat_example() -> None:
    messages = pipeline._load_prompt_few_shots("generic_talker", custom_prompt=False)

    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "assistant"]
    tool_call = messages[1]["tool_calls"][0]
    assert tool_call["function"]["name"] == "call_backend"
    arguments = json.loads(tool_call["function"]["arguments"])
    assert "Nairobi" not in arguments["query"]
    assert "location" in arguments["query"]
    assert arguments["filler_text"] == "Let me check that weather again."
    assert messages[2]["tool_call_id"] == tool_call["id"]


def test_custom_prompt_never_inherits_catalog_few_shots() -> None:
    with patch.object(pipeline, "load_prompt_catalog", side_effect=AssertionError("catalog must not be read")):
        assert pipeline._load_prompt_few_shots("generic_talker", custom_prompt=True) == []


def test_few_shot_loader_returns_a_deep_copy() -> None:
    catalog = {"example": {"few_shots": [{"role": "assistant", "content": None, "tool_calls": [{"id": "one"}]}]}}

    with patch.object(pipeline, "load_prompt_catalog", return_value=catalog):
        messages = pipeline._load_prompt_few_shots("example", custom_prompt=False)

    messages[0]["tool_calls"][0]["id"] = "changed"
    assert catalog["example"]["few_shots"][0]["tool_calls"][0]["id"] == "one"


@pytest.mark.parametrize(
    "messages, error",
    [
        ("not-a-list", "must be a list"),
        ([{"role": "system", "content": "override"}], "unsupported role"),
        ([{"role": "user", "content": 7}], "content must be text or null"),
        ([{"role": "tool", "content": "{}"}], "needs tool_call_id"),
    ],
)
def test_invalid_catalog_few_shots_fail_closed(messages: object, error: str) -> None:
    with (
        patch.object(pipeline, "load_prompt_catalog", return_value={"example": {"few_shots": messages}}),
        pytest.raises(ValueError, match=error),
    ):
        pipeline._load_prompt_few_shots("example", custom_prompt=False)
