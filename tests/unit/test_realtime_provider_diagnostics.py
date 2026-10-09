# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Keep provider diagnostics useful without copying private service output."""

from realtime.provider_diagnostics import provider_error_param


def test_provider_error_param_classifies_known_failure_without_copying_details() -> None:
    """Only fixed labels leave the process for recognized tool and RPC failures."""
    assert (
        provider_error_param(
            "PRIVATE: Forced tool completion did not return a structured call for 'PRIVATE'", ValueError("PRIVATE")
        )
        == "provider.tool_call_missing"
    )
    assert (
        provider_error_param("PRIVATE: status = StatusCode.DEADLINE_EXCEEDED; credentials=PRIVATE", None)
        == "provider.grpc.deadline_exceeded"
    )
    assert provider_error_param("PRIVATE arbitrary provider body", RuntimeError("PRIVATE")) is None


def test_provider_error_param_does_not_stringify_exceptions() -> None:
    """Exception values and arbitrary objects cannot become diagnostic text."""

    class PrivateError(Exception):
        def __str__(self) -> str:
            raise AssertionError("exception body must not be read")

    assert provider_error_param(object(), PrivateError()) is None
