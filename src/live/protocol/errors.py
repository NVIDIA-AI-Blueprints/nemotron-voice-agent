# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared primitives of the live session protocol: ids, size estimates and the rejection error."""

from uuid import uuid4


def uid(prefix: str) -> str:
    """Return a unique id such as ``event_3f9a...``."""
    return f"{prefix}_{uuid4().hex}"


def approx_tokens(text: str) -> int:
    """Return a cheap token estimate (about four characters per token) for the protocol's size limits."""
    return (len(text) + 3) // 4


class ProtocolError(Exception):
    """A request the service rejects. ``code`` and ``param`` are sent to the client in the error event."""

    def __init__(self, message: str, code: str = "invalid_value", param: str | None = None):
        """Create the error with its protocol ``code`` and the offending ``param``."""
        super().__init__(message)
        self.code, self.param = code, param
