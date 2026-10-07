# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic text-only Frontend/Backend Agent prototype.

A domain-agnostic, text-in/text-out reimplementation of the voice
Frontend/Backend Agent example, built so a scaffold evaluation harness (tau2-bench)
can drive it: explicit session state, structured tool calls, and exactly one
user-visible payload per step.
"""

from examples.frontend_backend_verdict.text.agent import FrontendBackendAgent, assemble_agent, build_agent
from examples.frontend_backend_verdict.text.config import Config, load_config
from examples.frontend_backend_verdict.text.errors import (
    ConfigError,
    FrontendBackendAgentError,
    FrontendContractError,
    StateReplayError,
    ToolProtocolError,
    ToolResultSerializationError,
)
from examples.frontend_backend_verdict.text.events import CollectingSink, EventSink, JsonlSink, LoggingSink, NullSink
from examples.frontend_backend_verdict.text.messages import AgentTurn, Message, ToolCall, ToolResult, UsageTotals
from examples.frontend_backend_verdict.text.session import PendingTurn, SessionState
from examples.frontend_backend_verdict.text.tools import ToolSpec

__all__ = [
    "AgentTurn",
    "CollectingSink",
    "Config",
    "ConfigError",
    "EventSink",
    "FrontendBackendAgent",
    "FrontendBackendAgentError",
    "FrontendContractError",
    "JsonlSink",
    "LoggingSink",
    "Message",
    "NullSink",
    "PendingTurn",
    "SessionState",
    "StateReplayError",
    "ToolCall",
    "ToolProtocolError",
    "ToolResult",
    "ToolResultSerializationError",
    "ToolSpec",
    "UsageTotals",
    "assemble_agent",
    "build_agent",
    "load_config",
]
