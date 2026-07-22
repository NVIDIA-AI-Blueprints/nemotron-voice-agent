# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

from examples.shared.json_parsing import extract_json_object


def test_extracts_first_object_before_trailing_object() -> None:
    assert extract_json_object('{"tts":"ok"} trailing {"debug":true}') == {"tts": "ok"}


def test_extracts_fenced_object_with_trailing_text() -> None:
    assert extract_json_object('```json\n{"observation":"waving"}\n```\nextra') == {"observation": "waving"}


def test_skips_malformed_prefix_before_valid_object() -> None:
    assert extract_json_object('broken {not json} then {"valid":1}') == {"valid": 1}


def test_returns_empty_for_no_object() -> None:
    assert extract_json_object("plain text") == {}
