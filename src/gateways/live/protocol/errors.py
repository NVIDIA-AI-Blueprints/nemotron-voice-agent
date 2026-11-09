# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared primitives of the live session protocol: the rejection error."""

from pydantic import ValidationError


class ProtocolError(Exception):
    """A request the service rejects. ``code`` and ``param`` are sent to the client in the error event."""

    def __init__(self, message: str, code: str = "invalid_value", param: str | None = None):
        """Create the error with its protocol ``code`` and the offending ``param``."""
        super().__init__(message)
        self.code, self.param = code, param


def validation_error(refused: ValidationError, prefix: str) -> ProtocolError:
    """Return a schema check's first problem as the protocol's error, its ``param`` the dotted path under ``prefix``."""
    problem = refused.errors(include_url=False)[0]
    param = ".".join([prefix, *(str(part) for part in problem["loc"])])
    if problem["type"] == "extra_forbidden":
        return ProtocolError(f"Unknown parameter: '{param}'.", "unknown_parameter", param)
    return ProtocolError(f"Invalid value for '{param}': {problem['msg'].removeprefix('Value error, ')}", param=param)
