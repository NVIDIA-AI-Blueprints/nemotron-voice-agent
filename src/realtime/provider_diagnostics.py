# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Classify provider failures without forwarding exception bodies or credentials."""

from __future__ import annotations


def provider_error_param(error: object, exception: BaseException | None) -> str | None:
    """Return only fixed diagnostic labels for known provider failure signatures."""
    text = error[:2048] if isinstance(error, str) else ""
    tool_failures = {
        "Forced tool completion did not finish with a complete tool call": "provider.tool_completion_incomplete",
        "Forced tool completion did not return a structured call": "provider.tool_call_missing",
        "Forced tool completion returned parallel calls": "provider.parallel_tool_calls",
        "Forced tool completion returned invalid JSON arguments": "provider.tool_arguments_invalid",
        "Forced tool completion returned unavailable function": "provider.tool_unavailable",
    }
    for signature, label in tool_failures.items():
        if signature in text:
            return label
    for status in (
        "DEADLINE_EXCEEDED",
        "RESOURCE_EXHAUSTED",
        "UNAUTHENTICATED",
        "PERMISSION_DENIED",
        "INVALID_ARGUMENT",
        "UNAVAILABLE",
        "INTERNAL",
        "CANCELLED",
    ):
        if f"StatusCode.{status}" in text or f"gRPC {status}" in text:
            return "provider.grpc." + status.lower()
    names = {
        "APITimeoutError": "provider.timeout",
        "RateLimitError": "provider.rate_limit",
        "AuthenticationError": "provider.authentication",
        "PermissionDeniedError": "provider.permission_denied",
        "BadRequestError": "provider.bad_request",
        "APIConnectionError": "provider.connection",
    }
    return names.get(type(exception).__name__)
