# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D103, D107

from types import SimpleNamespace

from examples.generic_assistant_livekit.errors import describe_error


class APIStatusError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


def test_a_recoverable_rate_limit_is_summarized_briefly():
    error = SimpleNamespace(error=APIStatusError("Error code: 429", 429), recoverable=True, timestamp=1.0, label="x")
    assert describe_error(error) == {
        "message": "APIStatusError: Error code: 429",
        "status": 429,
        "recoverable": True,
    }


def test_a_failure_is_not_recoverable_and_long_text_is_cut():
    error = SimpleNamespace(error=RuntimeError("boom " * 100), recoverable=False)
    summary = describe_error(error)
    assert summary["recoverable"] is False
    assert summary["status"] is None
    assert len(summary["message"]) == 200
    assert "\n" not in summary["message"]


def test_a_plain_exception_without_the_wrapper_still_works():
    summary = describe_error(ValueError("bad\nvalue"))
    assert summary == {"message": "ValueError: bad value", "status": None, "recoverable": False}


def test_non_integer_status_is_ignored():
    assert describe_error(SimpleNamespace(error=SimpleNamespace(status_code="429"), recoverable=True))["status"] is None
