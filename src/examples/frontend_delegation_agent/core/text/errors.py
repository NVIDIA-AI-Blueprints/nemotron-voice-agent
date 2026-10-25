# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed errors raised by the text Frontend/Backend Agent prototype."""

from __future__ import annotations


class FrontendBackendAgentError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(FrontendBackendAgentError):
    """Raised when a configuration file is missing, malformed, or inconsistent."""


class ToolProtocolError(FrontendBackendAgentError):
    """Raised when tool results violate the call/result protocol (a driver bug)."""


class FrontendContractError(FrontendBackendAgentError):
    """Raised when the frontend LLM breaks its tool contract and the policy is error."""


class StateReplayError(FrontendBackendAgentError):
    """Raised when a foreign transcript cannot be replayed deterministically."""


class ToolResultSerializationError(FrontendBackendAgentError):
    """Raised when a tool result is not JSON-serializable."""
