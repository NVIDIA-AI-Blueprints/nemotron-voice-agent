# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Canonical-current OpenAI Realtime scaling client.

The client speaks JSON events over ``/v1/realtime`` directly. It does not
translate RTVI frames and it does not retry a failed Realtime session through
another protocol.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import datetime as dt
import hashlib
import json
import math
import os
import ssl
import time
import uuid
import wave
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode, urlsplit

import aiohttp
import numpy as np
import resampy
import websockets
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for
from websockets.exceptions import ConnectionClosed

REALTIME_PCM_RATE = 24000
REALTIME_CHUNK_MS = 32
WS_CONNECT_TIMEOUT = 30
HARD_DEADLINE_BUFFER = 60
CLIENT_TOOL_HANDLER_MARGIN_RATIO = 0.1
CLIENT_TOOL_HANDLER_MARGIN_MAX = 10.0
MAX_HTTP_TOOL_RESPONSE_BYTES = 1024 * 1024

_FUNCTION_TOOL_FIELDS = frozenset({"type", "name", "description", "parameters"})
_MCP_TOOL_FIELDS = frozenset(
    {
        "type",
        "server_label",
        "server_url",
        "connector_id",
        "tunnel_id",
        "authorization",
        "headers",
        "allowed_tools",
        "require_approval",
        "server_description",
        "allowed_callers",
        "defer_loading",
    }
)
_MCP_ENDPOINT_FIELDS = ("server_url", "connector_id", "tunnel_id")
_MCP_TOOL_FILTER_FIELDS = frozenset({"tool_names", "read_only"})
_MCP_APPROVAL_FILTER_FIELDS = frozenset({"always", "never"})
_MCP_APPROVAL_POLICY_FIELDS = frozenset({"default", "rules"})
_MCP_APPROVAL_RULE_FIELDS = frozenset({"server_label", "name", "approve", "reason"})
_CLIENT_TOOL_CONFIG_FIELDS = frozenset({"tools", "handlers", "tool_choice", "parallel_tool_calls", "mcp_approval"})
_SCRIPT_HANDLER_FIELDS = frozenset({"type", "steps", "repeat_last"})
_SCRIPT_STEP_FIELDS = frozenset({"output", "error", "delay_ms", "expected_arguments"})
_SCRIPT_ERROR_FIELDS = frozenset({"code", "message"})
_HTTP_HANDLER_FIELDS = frozenset({"type", "url", "timeout_seconds"})
_LATE_CLIENT_TOOL_OUTPUT_ERROR_CODES = frozenset({"client_tool_timeout", "duplicate_tool_output"})
_MISSING = object()
_REDACTED = "[REDACTED]"

_TERMINAL_RESPONSE_STATUSES = frozenset({"completed", "cancelled", "failed", "incomplete"})
_CANONICAL_PASSIVE_EVENTS = frozenset(
    {
        "input_audio_buffer.cleared",
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.timeout_triggered",
        "conversation.item.added",
        "conversation.item.done",
        "conversation.item.deleted",
        "conversation.item.truncated",
        "conversation.item.retrieved",
        "conversation.item.input_audio_transcription.delta",
        "conversation.item.input_audio_transcription.completed",
        "conversation.item.input_audio_transcription.failed",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_audio.done",
        "response.output_audio_transcript.done",
        "response.output_text.done",
        "response.mcp_call_arguments.delta",
        "response.mcp_call_arguments.done",
        "response.mcp_call.in_progress",
        "response.mcp_call.completed",
        "response.mcp_call.failed",
        "mcp_list_tools.in_progress",
        "mcp_list_tools.completed",
        "mcp_list_tools.failed",
    }
)
_NON_GA_EVENT_TYPES = frozenset(
    {
        "conversation.item.created",
        "response.audio.delta",
        "response.audio.done",
        "response.audio_transcript.delta",
        "response.audio_transcript.done",
        "response.text.delta",
        "response.text.done",
    }
)
_INPUT_STREAM_BOUNDARY_EVENTS = frozenset(
    {
        "error",
        "input_audio_buffer.cleared",
        "input_audio_buffer.committed",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.timeout_triggered",
        "response.created",
    }
)


def _contains_user_visible_text(delta: str) -> bool:
    """Return whether a text delta contains more than formatting whitespace."""
    return bool(delta.strip())


def _redact_mcp_credentials(value: Any) -> None:
    """Redact MCP secrets in a copied protocol event before persistence."""
    if isinstance(value, list):
        for item in value:
            _redact_mcp_credentials(item)
        return
    if not isinstance(value, dict):
        return
    if value.get("type") == "mcp":
        if value.get("authorization") is not None:
            value["authorization"] = _REDACTED
        headers = value.get("headers")
        if isinstance(headers, dict):
            value["headers"] = {name: _REDACTED for name in headers}
        server_url = value.get("server_url")
        if isinstance(server_url, str):
            parsed_url = urlsplit(server_url)
            if parsed_url.query:
                value["server_url"] = parsed_url._replace(query=_REDACTED).geturl()
    for item in value.values():
        _redact_mcp_credentials(item)


def _public_tool_view(tools: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Match the server's public session view without MCP connection secrets."""
    public_tools = copy.deepcopy(list(tools))
    for tool in public_tools:
        if tool.get("type") == "mcp":
            tool.pop("authorization", None)
            tool.pop("headers", None)
    return tuple(public_tools)


class AsyncRunLogger(Protocol):
    """Logger surface shared with ``benchmark.RunLogger``."""

    async def log(self, message: str) -> None:
        """Write one diagnostic line."""


class RealtimeClientError(RuntimeError):
    """A connection-level protocol or server failure."""


class RealtimeTurnError(RuntimeError):
    """A terminal failure limited to one benchmark turn."""


class RealtimeClientStopped(RuntimeError):
    """A graceful local shutdown requested by the benchmark orchestrator."""


class _ScriptedToolError(RuntimeError):
    """A deterministic exception requested by a scripted tool step."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _HttpToolError(RuntimeError):
    """A model-visible failure returned by an HTTP client-tool handler."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _MCPDiscoveryPhase(Enum):
    ANNOUNCED = "announced"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class _MCPCallPhase(Enum):
    ANNOUNCED = "announced"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ScriptedToolStep:
    """One deterministic client-tool invocation result."""

    output: Any = _MISSING
    error_code: str | None = None
    error_message: str | None = None
    delay_ms: int = 0
    expected_arguments: Any = _MISSING


@dataclass(frozen=True, slots=True)
class ScriptedToolHandler:
    """Ordered per-tool results consumed independently by every worker."""

    steps: tuple[ScriptedToolStep, ...]
    repeat_last: bool


@dataclass(frozen=True, slots=True)
class HttpToolHandler:
    """POST tool arguments to an explicitly configured application endpoint."""

    url: str
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class HttpToolInvocation:
    """One HTTP handler invocation with validated function arguments."""

    handler: HttpToolHandler
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MCPApprovalDecision:
    """One deterministic tester-owned response to an MCP approval request."""

    approve: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class MCPApprovalPolicy:
    """Exact MCP approval decisions plus a terminal default."""

    default: MCPApprovalDecision
    rules: dict[tuple[str, str], MCPApprovalDecision] = field(default_factory=dict)

    def resolve(self, server_label: str, name: str) -> MCPApprovalDecision:
        """Return the exact rule or the terminal default decision."""
        return self.rules.get((server_label, name), self.default)


def _default_mcp_approval_policy() -> MCPApprovalPolicy:
    return MCPApprovalPolicy(default=MCPApprovalDecision(approve=False))


@dataclass(frozen=True, slots=True)
class RealtimeClientToolsConfig:
    """Validated Realtime tool declarations and tester-owned execution policy."""

    tools: tuple[dict[str, Any], ...]
    handlers: dict[str, ScriptedToolHandler | HttpToolHandler]
    argument_validators: dict[str, Validator] = field(repr=False, compare=False)
    tool_choice: str | dict[str, Any]
    parallel_tool_calls: bool
    sha256: str
    mcp_approval: MCPApprovalPolicy = field(default_factory=_default_mcp_approval_policy)


def _parse_finite_json_float(raw: str) -> float:
    """Decode one JSON float without permitting overflow to infinity."""
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number {raw!r}")
    return value


def _reject_nonfinite_json_constant(raw: str) -> None:
    """Reject the non-standard NaN and Infinity constants accepted by Python."""
    raise ValueError(f"non-finite JSON constant {raw!r}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while rejecting ambiguous duplicate member names."""
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = child
    return value


@dataclass(slots=True)
class OpenAIRealtimeOutcome:
    """Metrics and protocol diagnostics returned to ``benchmark.py``."""

    latency_values: list[float]
    valid_latency_values: list[float]
    failed_turns: int
    reverse_barge_ins_count: int
    glitch_detected: bool
    server_metric_samples: dict[str, list[float]]
    protocol_events: list[dict[str, Any]]
    event_correlations: list[dict[str, Any]]
    response_transcripts: list[dict[str, Any]]
    response_status_counts: dict[str, int]
    tool_calls: list[dict[str, Any]]
    error: str | None


@dataclass(slots=True)
class _ReceivedEvent:
    event: dict[str, Any]
    received_at: float


@dataclass(slots=True)
class _ResponseTrace:
    response_id: str
    created_at: float
    transcript: str = ""
    output_bytes: int = 0
    has_text: bool = False
    has_function_call: bool = False
    has_mcp_call: bool = False
    status: str = "in_progress"
    output_items: dict[int, _OutputItemTrace] = field(default_factory=dict)
    item_indexes: dict[str, int] = field(default_factory=dict)
    followup_requested: bool = False


@dataclass(slots=True)
class _ConversationItemTrace:
    item_id: str
    item_type: str
    added: dict[str, Any]
    done: dict[str, Any] | None = None
    deleted: bool = False


@dataclass(slots=True)
class _ContentPartTrace:
    content_index: int
    part_type: str
    audio_done: bool = False
    transcript_done: bool = False
    text_done: bool = False
    part_done: bool = False
    streamed_text: str = ""
    terminal_text: str | None = None


@dataclass(slots=True)
class _OutputItemTrace:
    output_index: int
    item_id: str
    item_type: str
    added: dict[str, Any]
    content_parts: dict[int, _ContentPartTrace] = field(default_factory=dict)
    function_arguments: str = ""
    function_arguments_done: bool = False
    output_done: bool = False
    mcp_status: _MCPCallPhase | None = None


@dataclass(slots=True)
class _MCPDiscoveryTrace:
    item_id: str
    server_label: str
    phase: _MCPDiscoveryPhase = _MCPDiscoveryPhase.ANNOUNCED
    item_done: bool = False


@dataclass(slots=True)
class _MCPApprovalResponseTrace:
    request_id: str
    item: dict[str, Any]
    event_id: str
    item_done: bool = False


@dataclass(slots=True)
class _TurnInputTrace:
    speech_item_id: str | None = None
    speech_start_ms: int | None = None
    speech_stopped: bool = False
    item_id: str | None = None
    submitted_text: str | None = None


@dataclass(slots=True)
class _TurnResult:
    input_finished_at: float
    first_output_at: float
    response_ids: list[str]
    transcript: str
    output_bytes: int
    tool_call_ids: list[str]


def _new_event_id() -> str:
    return f"event_{uuid.uuid4().hex}"


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RealtimeClientError(f"{label} must be a JSON object")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RealtimeClientError(f"{label} must be a non-empty string")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RealtimeClientError(f"{label} must be a non-negative integer")
    return value


def _require_finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RealtimeClientError(f"{label} must be a finite number")
    try:
        numeric = float(value)
    except OverflowError as exc:
        raise RealtimeClientError(f"{label} must be a finite number") from exc
    if not math.isfinite(numeric) or (nonnegative and numeric < 0):
        qualifier = "non-negative finite" if nonnegative else "finite"
        raise RealtimeClientError(f"{label} must be a {qualifier} number")
    return numeric


def _reject_unknown_keys(value: dict[str, Any], allowed: frozenset[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise RealtimeClientError(f"{label} contains unknown field {unknown[0]!r}")


def _validate_finite_json(value: Any, label: str) -> None:
    """Require a JSON-compatible tree with finite numeric values and string keys."""
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RealtimeClientError(f"{label} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite_json(child, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise RealtimeClientError(f"{label} contains a non-string object key")
            _validate_finite_json(child, f"{label}.{key}")
        return
    raise RealtimeClientError(f"{label} contains non-JSON value {type(value).__name__}")


def _validate_mcp_tool_names(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise RealtimeClientError(f"{label} must be a JSON array")
    names: list[str] = []
    for index, raw_name in enumerate(value):
        name = _require_nonempty_string(raw_name, f"{label}[{index}]")
        if name in names:
            raise RealtimeClientError(f"{label} contains duplicate tool name {name!r}")
        names.append(name)
    return names


def _validate_mcp_tool_filter(value: Any, label: str) -> dict[str, Any]:
    tool_filter = _require_object(value, label)
    _reject_unknown_keys(tool_filter, _MCP_TOOL_FILTER_FIELDS, label)
    if not tool_filter:
        raise RealtimeClientError(f"{label} must include tool_names or read_only")
    normalized = copy.deepcopy(tool_filter)
    if "tool_names" in normalized:
        normalized["tool_names"] = _validate_mcp_tool_names(normalized["tool_names"], f"{label}.tool_names")
    if "read_only" in normalized and not isinstance(normalized["read_only"], bool):
        raise RealtimeClientError(f"{label}.read_only must be a boolean")
    return normalized


def _validate_mcp_tool(raw_tool: dict[str, Any], label: str, *, allow_reference: bool) -> dict[str, Any]:
    _reject_unknown_keys(raw_tool, _MCP_TOOL_FIELDS, label)
    server_label = _require_nonempty_string(raw_tool.get("server_label"), f"{label}.server_label")
    endpoint_fields = [field_name for field_name in _MCP_ENDPOINT_FIELDS if field_name in raw_tool]
    if not endpoint_fields:
        if allow_reference and set(raw_tool) == {"type", "server_label"}:
            return copy.deepcopy(raw_tool)
        raise RealtimeClientError(f"{label} must provide exactly one of server_url, connector_id, or tunnel_id")
    if len(endpoint_fields) != 1:
        raise RealtimeClientError(f"{label} must provide exactly one of server_url, connector_id, or tunnel_id")
    endpoint_field = endpoint_fields[0]
    endpoint = _require_nonempty_string(raw_tool[endpoint_field], f"{label}.{endpoint_field}")
    if endpoint_field == "server_url":
        parsed_url = urlsplit(endpoint)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise RealtimeClientError(f"{label}.server_url must be an absolute HTTP or HTTPS URL")
        if parsed_url.username is not None or parsed_url.password is not None:
            raise RealtimeClientError(f"{label}.server_url must not contain credentials")
        if parsed_url.fragment:
            raise RealtimeClientError(f"{label}.server_url must not contain a fragment")

    normalized = copy.deepcopy(raw_tool)
    if "authorization" in normalized:
        _require_nonempty_string(normalized["authorization"], f"{label}.authorization")
    if "headers" in normalized and normalized["headers"] is not None:
        headers = _require_object(normalized["headers"], f"{label}.headers")
        canonical_names: set[str] = set()
        for header_name, header_value in headers.items():
            name = _require_nonempty_string(header_name, f"{label}.headers key")
            if not isinstance(header_value, str):
                raise RealtimeClientError(f"{label}.headers.{name} must be a string")
            canonical_name = name.casefold()
            if canonical_name in canonical_names:
                raise RealtimeClientError(f"{label}.headers contains duplicate header name {name!r}")
            canonical_names.add(canonical_name)
        if normalized.get("authorization") is not None and "authorization" in canonical_names:
            raise RealtimeClientError(f"{label} cannot provide authorization in both authorization and headers")
    if "allowed_tools" in normalized and normalized["allowed_tools"] is not None:
        allowed_tools = normalized["allowed_tools"]
        normalized["allowed_tools"] = (
            _validate_mcp_tool_names(allowed_tools, f"{label}.allowed_tools")
            if isinstance(allowed_tools, list)
            else _validate_mcp_tool_filter(allowed_tools, f"{label}.allowed_tools")
        )
    if "require_approval" in normalized and normalized["require_approval"] is not None:
        approval = normalized["require_approval"]
        if isinstance(approval, str):
            if approval not in {"always", "never"}:
                raise RealtimeClientError(f"{label}.require_approval must be always, never, or an approval filter")
        else:
            approval_filter = _require_object(approval, f"{label}.require_approval")
            _reject_unknown_keys(approval_filter, _MCP_APPROVAL_FILTER_FIELDS, f"{label}.require_approval")
            if not approval_filter:
                raise RealtimeClientError(f"{label}.require_approval must include an always or never filter")
            normalized["require_approval"] = {
                rule: _validate_mcp_tool_filter(value, f"{label}.require_approval.{rule}")
                for rule, value in approval_filter.items()
            }
    if "server_description" in normalized and not isinstance(normalized["server_description"], str):
        raise RealtimeClientError(f"{label}.server_description must be a string")
    if "allowed_callers" in normalized and normalized["allowed_callers"] is not None:
        callers = normalized["allowed_callers"]
        if not isinstance(callers, list):
            raise RealtimeClientError(f"{label}.allowed_callers must be a JSON array")
        seen_callers: set[str] = set()
        for index, caller in enumerate(callers):
            caller = _require_nonempty_string(caller, f"{label}.allowed_callers[{index}]")
            if caller not in {"direct", "programmatic"}:
                raise RealtimeClientError(f"{label}.allowed_callers[{index}] must be direct or programmatic")
            if caller in seen_callers:
                raise RealtimeClientError(f"{label}.allowed_callers contains duplicate caller {caller!r}")
            seen_callers.add(caller)
    if "defer_loading" in normalized and not isinstance(normalized["defer_loading"], bool):
        raise RealtimeClientError(f"{label}.defer_loading must be a boolean")
    _validate_finite_json(normalized, label)
    if normalized["server_label"] != server_label:
        raise RuntimeError("MCP server label changed during validation")
    return normalized


def _validate_realtime_tools(
    value: Any,
    label: str,
    *,
    allow_mcp_references: bool = True,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise RealtimeClientError(f"{label} must be a JSON array")
    tools: list[dict[str, Any]] = []
    function_names: set[str] = set()
    mcp_server_labels: set[str] = set()
    for index, raw_tool in enumerate(value):
        tool_label = f"{label}[{index}]"
        tool = _require_object(raw_tool, tool_label)
        tool_type = tool.get("type")
        if tool_type == "mcp":
            normalized = _validate_mcp_tool(tool, tool_label, allow_reference=allow_mcp_references)
            server_label = normalized["server_label"]
            if server_label in mcp_server_labels:
                raise RealtimeClientError(f"{label} contains duplicate MCP server label {server_label!r}")
            mcp_server_labels.add(server_label)
            tools.append(normalized)
            continue
        _reject_unknown_keys(tool, _FUNCTION_TOOL_FIELDS, tool_label)
        if tool_type != "function":
            raise RealtimeClientError(f"{tool_label}.type must be 'function' or 'mcp'")
        name = _require_nonempty_string(tool.get("name"), f"{tool_label}.name")
        if name in function_names:
            raise RealtimeClientError(f"{label} contains duplicate function name {name!r}")
        function_names.add(name)
        if "description" in tool and not isinstance(tool["description"], str):
            raise RealtimeClientError(f"{tool_label}.description must be a string")
        parameters = tool.get("parameters", {})
        if not isinstance(parameters, dict):
            raise RealtimeClientError(f"{tool_label}.parameters must be a JSON object")
        normalized = copy.deepcopy(tool)
        normalized.setdefault("parameters", {})
        _validate_finite_json(normalized, tool_label)
        tools.append(normalized)
    return tuple(tools)


def _compile_client_tool_argument_validators(
    tools: tuple[dict[str, Any], ...],
    label: str,
) -> dict[str, Validator]:
    """Compile each configured function's declared JSON Schema once."""
    validators: dict[str, Validator] = {}
    for index, tool in enumerate(tools):
        if tool["type"] != "function":
            continue
        schema = copy.deepcopy(tool["parameters"])
        schema_label = f"{label}[{index}].parameters"
        if "$schema" in schema:
            if not isinstance(schema["$schema"], str):
                raise RealtimeClientError(f"{schema_label}.$schema must be a string")
            validator_class = validator_for(schema, default=None)
            if validator_class is None:
                raise RealtimeClientError(f"{schema_label} declares an unsupported JSON Schema dialect")
        else:
            validator_class = Draft202012Validator
        try:
            validator_class.check_schema(schema)
        except SchemaError as exc:
            raise RealtimeClientError(f"{schema_label} is not a valid JSON Schema") from exc
        validators[tool["name"]] = validator_class(
            schema,
            format_checker=validator_class.FORMAT_CHECKER,
        )
    return validators


def _validate_tool_choice(
    value: Any,
    tools: tuple[dict[str, Any], ...],
    label: str,
) -> str | dict[str, Any]:
    function_names = {tool["name"] for tool in tools if tool["type"] == "function"}
    mcp_server_labels = {tool["server_label"] for tool in tools if tool["type"] == "mcp"}
    if isinstance(value, str):
        if value not in {"auto", "none", "required"}:
            raise RealtimeClientError(f"{label} must be auto, none, required, or a forced tool object")
        if value == "required" and not tools:
            raise RealtimeClientError(f"{label}=required needs at least one configured tool")
        return value
    choice = _require_object(value, label)
    choice_type = choice.get("type")
    if choice_type == "function":
        _reject_unknown_keys(choice, frozenset({"type", "name"}), label)
        name = _require_nonempty_string(choice.get("name"), f"{label}.name")
        if name not in function_names:
            raise RealtimeClientError(f"{label} selects unavailable function {name!r}")
        return {"type": "function", "name": name}
    if choice_type == "mcp":
        _reject_unknown_keys(choice, frozenset({"type", "server_label", "name"}), label)
        server_label = _require_nonempty_string(choice.get("server_label"), f"{label}.server_label")
        if server_label not in mcp_server_labels:
            raise RealtimeClientError(f"{label} selects unavailable MCP server {server_label!r}")
        normalized: dict[str, Any] = {"type": "mcp", "server_label": server_label}
        if "name" in choice:
            normalized["name"] = (
                None if choice["name"] is None else _require_nonempty_string(choice["name"], f"{label}.name")
            )
        return normalized
    raise RealtimeClientError(f"{label}.type must be 'function' or 'mcp'")


def _validate_mcp_approval_policy(value: Any, mcp_server_labels: set[str]) -> MCPApprovalPolicy:
    if value is None:
        return _default_mcp_approval_policy()
    policy = _require_object(value, "client-tools config.mcp_approval")
    _reject_unknown_keys(policy, _MCP_APPROVAL_POLICY_FIELDS, "client-tools config.mcp_approval")
    default = policy.get("default", "reject")
    if default not in {"approve", "reject"}:
        raise RealtimeClientError("client-tools config.mcp_approval.default must be approve or reject")
    raw_rules = policy.get("rules", [])
    if not isinstance(raw_rules, list):
        raise RealtimeClientError("client-tools config.mcp_approval.rules must be a JSON array")
    rules: dict[tuple[str, str], MCPApprovalDecision] = {}
    for index, raw_rule in enumerate(raw_rules):
        label = f"client-tools config.mcp_approval.rules[{index}]"
        rule = _require_object(raw_rule, label)
        _reject_unknown_keys(rule, _MCP_APPROVAL_RULE_FIELDS, label)
        server_label = _require_nonempty_string(rule.get("server_label"), f"{label}.server_label")
        if server_label not in mcp_server_labels:
            raise RealtimeClientError(f"{label}.server_label selects unavailable MCP server {server_label!r}")
        name = _require_nonempty_string(rule.get("name"), f"{label}.name")
        approve = rule.get("approve")
        if not isinstance(approve, bool):
            raise RealtimeClientError(f"{label}.approve must be a boolean")
        reason = rule.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise RealtimeClientError(f"{label}.reason must be a string")
        key = (server_label, name)
        if key in rules:
            raise RealtimeClientError(f"client-tools config.mcp_approval.rules repeats {server_label}.{name}")
        rules[key] = MCPApprovalDecision(approve=approve, reason=reason)
    return MCPApprovalPolicy(
        default=MCPApprovalDecision(approve=default == "approve"),
        rules=rules,
    )


def load_client_tools_config(path: Path) -> RealtimeClientToolsConfig:
    """Load and strictly validate one client-tool registry JSON file."""
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise RealtimeClientError(f"Could not read Realtime client-tools config {path}: {exc}") from exc
    try:
        raw = json.loads(
            raw_bytes,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except (json.JSONDecodeError, ValueError, OverflowError) as exc:
        raise RealtimeClientError(f"Realtime client-tools config {path} is invalid JSON: {exc}") from exc
    config = _require_object(raw, "client-tools config")
    _reject_unknown_keys(config, _CLIENT_TOOL_CONFIG_FIELDS, "client-tools config")
    tools = _validate_realtime_tools(
        config.get("tools"),
        "client-tools config.tools",
        allow_mcp_references=False,
    )
    argument_validators = _compile_client_tool_argument_validators(tools, "client-tools config.tools")
    function_names = {tool["name"] for tool in tools if tool["type"] == "function"}
    mcp_server_labels = {tool["server_label"] for tool in tools if tool["type"] == "mcp"}

    raw_handlers = _require_object(config.get("handlers"), "client-tools config.handlers")
    handler_names = set(raw_handlers)
    if not all(isinstance(name, str) and name for name in handler_names):
        raise RealtimeClientError("client-tools config.handlers keys must be non-empty strings")
    if handler_names != function_names:
        missing = sorted(function_names - handler_names)
        extra = sorted(handler_names - function_names)
        raise RealtimeClientError(
            "client-tools config.handlers must match tools exactly; "
            f"missing={missing or 'none'} extra={extra or 'none'}"
        )

    handlers: dict[str, ScriptedToolHandler | HttpToolHandler] = {}
    for name in sorted(handler_names):
        handler_label = f"client-tools config.handlers.{name}"
        raw_handler = _require_object(raw_handlers[name], handler_label)
        handler_type = raw_handler.get("type", "scripted")
        if handler_type == "http":
            _reject_unknown_keys(raw_handler, _HTTP_HANDLER_FIELDS, handler_label)
            url = _require_nonempty_string(raw_handler.get("url"), f"{handler_label}.url")
            parsed_url = urlsplit(url)
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
                raise RealtimeClientError(f"{handler_label}.url must be an absolute HTTP or HTTPS URL")
            if parsed_url.username is not None or parsed_url.password is not None:
                raise RealtimeClientError(f"{handler_label}.url must not contain credentials")
            if parsed_url.fragment:
                raise RealtimeClientError(f"{handler_label}.url must not contain a fragment")
            timeout_seconds = _require_finite_number(
                raw_handler.get("timeout_seconds", 60),
                f"{handler_label}.timeout_seconds",
            )
            if timeout_seconds <= 0:
                raise RealtimeClientError(f"{handler_label}.timeout_seconds must be greater than zero")
            handlers[name] = HttpToolHandler(url=url, timeout_seconds=timeout_seconds)
            continue
        if handler_type != "scripted":
            raise RealtimeClientError(f"{handler_label}.type must be 'scripted' or 'http'")
        _reject_unknown_keys(raw_handler, _SCRIPT_HANDLER_FIELDS, handler_label)
        raw_steps = raw_handler.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise RealtimeClientError(f"{handler_label}.steps must be a non-empty JSON array")
        repeat_last = raw_handler.get("repeat_last", True)
        if not isinstance(repeat_last, bool):
            raise RealtimeClientError(f"{handler_label}.repeat_last must be a boolean")
        steps: list[ScriptedToolStep] = []
        for index, raw_step in enumerate(raw_steps):
            step_label = f"{handler_label}.steps[{index}]"
            step = _require_object(raw_step, step_label)
            _reject_unknown_keys(step, _SCRIPT_STEP_FIELDS, step_label)
            has_output = "output" in step
            has_error = "error" in step
            if has_output == has_error:
                raise RealtimeClientError(f"{step_label} must contain exactly one of output or error")
            delay_ms = step.get("delay_ms", 0)
            if isinstance(delay_ms, bool) or not isinstance(delay_ms, int) or delay_ms < 0:
                raise RealtimeClientError(f"{step_label}.delay_ms must be a non-negative integer")
            expected_arguments = step.get("expected_arguments", _MISSING)
            if expected_arguments is not _MISSING and not isinstance(expected_arguments, dict):
                raise RealtimeClientError(f"{step_label}.expected_arguments must be a JSON object")
            if expected_arguments is not _MISSING:
                _validate_finite_json(expected_arguments, f"{step_label}.expected_arguments")
            if has_output:
                output = step["output"]
                _validate_finite_json(output, f"{step_label}.output")
                steps.append(
                    ScriptedToolStep(
                        output=copy.deepcopy(output),
                        delay_ms=delay_ms,
                        expected_arguments=(
                            _MISSING if expected_arguments is _MISSING else copy.deepcopy(expected_arguments)
                        ),
                    )
                )
                continue
            error = _require_object(step["error"], f"{step_label}.error")
            _reject_unknown_keys(error, _SCRIPT_ERROR_FIELDS, f"{step_label}.error")
            code = _require_nonempty_string(error.get("code"), f"{step_label}.error.code")
            message = _require_nonempty_string(error.get("message"), f"{step_label}.error.message")
            steps.append(
                ScriptedToolStep(
                    error_code=code,
                    error_message=message,
                    delay_ms=delay_ms,
                    expected_arguments=(
                        _MISSING if expected_arguments is _MISSING else copy.deepcopy(expected_arguments)
                    ),
                )
            )
        handlers[name] = ScriptedToolHandler(steps=tuple(steps), repeat_last=repeat_last)

    tool_choice = _validate_tool_choice(
        config.get("tool_choice", "auto"),
        tools,
        "client-tools config.tool_choice",
    )
    parallel_tool_calls = config.get("parallel_tool_calls", True)
    if not isinstance(parallel_tool_calls, bool):
        raise RealtimeClientError("client-tools config.parallel_tool_calls must be a boolean")
    mcp_approval = _validate_mcp_approval_policy(config.get("mcp_approval"), mcp_server_labels)
    return RealtimeClientToolsConfig(
        tools=tools,
        handlers=handlers,
        argument_validators=argument_validators,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        mcp_approval=mcp_approval,
    )


class OpenAIRealtimePerfClient:
    """Drive one strict canonical OpenAI Realtime WebSocket session."""

    def __init__(
        self,
        *,
        stream_id: str,
        host: str,
        port: int,
        path: str,
        scheme: str,
        insecure: bool,
        ca_file: Path | None,
        model: str,
        voice: str,
        instructions: str,
        output_modality: str,
        turn_mode: str,
        vad_silence_ms: int,
        api_key_env: str,
        audio_files: list[Path],
        start_delay: float,
        metrics_start_time: float | None,
        session_end_time: float | None,
        test_duration: float,
        reverse_barge_in_threshold: float,
        turn_response_timeout: float,
        tool_completion_timeout: float,
        max_tool_rounds: int,
        client_tools_config: RealtimeClientToolsConfig | None,
        session_init_timeout: float,
        audio_output_path: Path | None,
        logger: AsyncRunLogger,
        server_metric_keys: tuple[str, ...],
        shutdown_requested: Callable[[], bool],
        input_mode: str = "audio",
        text_inputs: list[str] | None = None,
    ) -> None:
        """Store explicit protocol and benchmark configuration."""
        if input_mode not in {"audio", "text"}:
            raise ValueError("input_mode must be audio or text")
        if output_modality not in {"audio", "text"}:
            raise ValueError("output_modality must be audio or text")
        if turn_mode not in {"automatic", "manual"}:
            raise ValueError("turn_mode must be automatic or manual")
        if not host:
            raise ValueError("Realtime WebSocket host must be non-empty")
        if isinstance(port, bool) or not isinstance(port, int) or port < 1 or port > 65535:
            raise ValueError("Realtime WebSocket port must be between 1 and 65535")
        if not path.startswith("/"):
            raise ValueError("Realtime WebSocket path must start with '/'")
        if scheme not in {"ws", "wss"}:
            raise ValueError("Realtime WebSocket scheme must be ws or wss")
        if scheme == "ws" and api_key_env:
            raise ValueError("Bearer credentials cannot be sent over an unencrypted ws connection")
        if scheme == "ws" and insecure:
            raise ValueError("realtime insecure mode applies only to wss connections")
        if scheme == "ws" and ca_file is not None:
            raise ValueError("a Realtime CA file applies only to wss connections")
        if insecure and ca_file is not None:
            raise ValueError("realtime insecure mode and a custom CA file are mutually exclusive")
        if ca_file is not None and not ca_file.is_file():
            raise ValueError(f"Realtime CA file does not exist: {ca_file}")
        if isinstance(vad_silence_ms, bool) or not isinstance(vad_silence_ms, int) or vad_silence_ms < 0:
            raise ValueError("vad_silence_ms must be non-negative")
        if not math.isfinite(start_delay) or start_delay < 0:
            raise ValueError("start_delay must be a non-negative finite number")
        if not math.isfinite(test_duration) or test_duration <= 0:
            raise ValueError("test_duration must be a positive finite number")
        if not math.isfinite(reverse_barge_in_threshold) or reverse_barge_in_threshold < 0:
            raise ValueError("reverse_barge_in_threshold must be a non-negative finite number")
        if metrics_start_time is not None and not math.isfinite(metrics_start_time):
            raise ValueError("metrics_start_time must be finite when set")
        if session_end_time is not None and not math.isfinite(session_end_time):
            raise ValueError("session_end_time must be finite when set")
        if not math.isfinite(turn_response_timeout) or turn_response_timeout <= 0:
            raise ValueError("turn_response_timeout must be greater than zero")
        if not math.isfinite(tool_completion_timeout) or tool_completion_timeout <= 0:
            raise ValueError("tool_completion_timeout must be greater than zero")
        if isinstance(max_tool_rounds, bool) or not isinstance(max_tool_rounds, int) or max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be a positive integer")
        if not math.isfinite(session_init_timeout) or session_init_timeout <= 0:
            raise ValueError("session_init_timeout must be greater than zero")
        text_inputs = list(text_inputs or [])
        if input_mode == "audio" and not audio_files:
            raise ValueError("audio_files must contain at least one WAV file in audio input mode")
        if input_mode == "text" and not text_inputs:
            raise ValueError("text_inputs must contain at least one prompt in text input mode")
        if any(not isinstance(value, str) or not value.strip() for value in text_inputs):
            raise ValueError("text_inputs must contain only non-empty prompts")

        self.stream_id = stream_id
        self.host = host
        self.port = port
        self.path = path
        self.scheme = scheme
        self.insecure = insecure
        self.ca_file = ca_file
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.input_mode = input_mode
        self.output_modality = output_modality
        self.turn_mode = turn_mode
        self.vad_silence_ms = vad_silence_ms
        self.api_key_env = api_key_env
        self.audio_files = audio_files
        self.text_inputs = text_inputs
        self.start_delay = start_delay
        self.metrics_start_time = metrics_start_time
        self.session_end_time = session_end_time
        self.test_duration = test_duration
        self.reverse_barge_in_threshold = reverse_barge_in_threshold
        self.turn_response_timeout = turn_response_timeout
        self.tool_completion_timeout = tool_completion_timeout
        # The first observed function-call event arms one absolute round
        # deadline. Client handlers stop this much earlier so their terminal
        # output and acknowledgement normally beat the gateway's own clock.
        self.client_tool_handler_margin = min(
            CLIENT_TOOL_HANDLER_MARGIN_MAX,
            tool_completion_timeout * CLIENT_TOOL_HANDLER_MARGIN_RATIO,
        )
        self.client_tool_handler_timeout = tool_completion_timeout - self.client_tool_handler_margin
        self.max_tool_rounds = max_tool_rounds
        self.client_tools_config = client_tools_config
        self.session_init_timeout = session_init_timeout
        self.audio_output_path = audio_output_path
        self.logger = logger
        self.shutdown_requested = shutdown_requested

        metric_keys = (
            *server_metric_keys,
            "response_total_tokens",
            "response_input_tokens",
            "response_output_tokens",
            "response_audio_bytes",
            "response_lifecycle",
        )
        self.server_metric_samples: dict[str, list[float]] = {key: [] for key in dict.fromkeys(metric_keys)}
        self.latency_values: list[float] = []
        self.valid_latency_values: list[float] = []
        self.failed_turns = 0
        self.reverse_barge_ins_count = 0
        self.glitch_detected = False
        self.protocol_events: list[dict[str, Any]] = []
        self.event_correlations: list[dict[str, Any]] = []
        self.response_transcripts: list[dict[str, Any]] = []
        self.response_status_counts: dict[str, int] = {}
        self.rate_limits: list[dict[str, Any]] = []
        self._tool_calls: dict[str, dict[str, Any]] = {}
        self._mcp_discoveries: dict[str, _MCPDiscoveryTrace] = {}
        self._mcp_approval_requests: dict[str, dict[str, Any]] = {}
        self._mcp_approval_requests_handled: set[str] = set()
        self._mcp_approval_response_items: dict[str, _MCPApprovalResponseTrace] = {}
        self._pending_events: dict[str, dict[str, Any]] = {}
        self._seen_server_event_ids: set[str] = set()
        self._response_traces: dict[str, _ResponseTrace] = {}
        self._active_response_id: str | None = None
        self._conversation_items: dict[str, _ConversationItemTrace] = {}
        self._conversation_tail_id: str | None = None
        self._event_queue: asyncio.Queue[_ReceivedEvent | BaseException] | None = None
        self._reader_task: asyncio.Task | None = None
        self._audio_writer: wave.Wave_write | None = None
        self._session_id = ""
        self._conversation_id = ""
        self._trusted_tool_names: set[str] = set()
        self._active_client_tool_names: set[str] = set()
        self._selected_tool_schemas: tuple[dict[str, Any], ...] = ()
        self._selected_tool_choice: str | dict[str, Any] = "auto"
        self._selected_parallel_tool_calls = True
        self._tool_handler_invocations: dict[str, int] = {}
        self._collecting_metrics = False
        self._input_append_count = 0
        self._input_audio_bytes = 0
        self._high_volume_protocol_events: dict[tuple[Any, ...], dict[str, Any]] = {}

    @property
    def uri(self) -> str:
        """Return the Realtime endpoint, including an explicit model query when set."""
        query = f"?{urlencode({'model': self.model})}" if self.model else ""
        return f"{self.scheme}://{self.host}:{self.port}{self.path}{query}"

    async def run(self) -> OpenAIRealtimeOutcome:
        """Run until the common turn-admission interval closes between turns."""
        await self.logger.log(f"{self.stream_id} starting OpenAI Realtime client uri={self.uri}")
        if self.start_delay > 0:
            await asyncio.sleep(self.start_delay)

        error: str | None = None
        hard_deadline = self._hard_deadline()
        try:
            await asyncio.wait_for(self._run_connection(), timeout=hard_deadline)
        except RealtimeClientStopped:
            pass
        except TimeoutError:
            error = f"Hard deadline reached ({hard_deadline:.0f}s)"
        except ConnectionClosed as exc:
            error = f"WebSocket connection closed: code={exc.code} reason={exc.reason or '-'}"
        except Exception as exc:  # noqa: BLE001
            error = str(exc) or exc.__class__.__name__
        finally:
            if self._reader_task is not None:
                self._reader_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, ConnectionClosed):
                    await self._reader_task
            if self._audio_writer is not None:
                self._audio_writer.close()
                self._audio_writer = None

        if error:
            await self.logger.log(f"{self.stream_id} OpenAI Realtime client failed: {error}")
        else:
            await self.logger.log(f"{self.stream_id} OpenAI Realtime client finished successfully")
        return OpenAIRealtimeOutcome(
            latency_values=self.latency_values,
            valid_latency_values=self.valid_latency_values,
            failed_turns=self.failed_turns,
            reverse_barge_ins_count=self.reverse_barge_ins_count,
            glitch_detected=self.glitch_detected,
            server_metric_samples=self.server_metric_samples,
            protocol_events=self.protocol_events,
            event_correlations=self.event_correlations,
            response_transcripts=self.response_transcripts,
            response_status_counts=self.response_status_counts,
            tool_calls=[copy.deepcopy(call) for call in self._tool_calls.values()],
            error=error,
        )

    def _hard_deadline(self) -> float:
        """Return an outer safety bound that cannot preempt inner phase deadlines."""
        max_input_seconds = (
            max(self._wav_duration_seconds(path) for path in self.audio_files) if self.input_mode == "audio" else 0.0
        )
        if self.input_mode == "audio" and self.turn_mode == "automatic":
            max_input_seconds += self.vad_silence_ms / 1000.0
        # One admitted turn may consume an initial response phase, then as many
        # bounded tool rounds as configured. Each round budgets a client handler
        # wait, a correlated item-ack wait, and the next response phase.
        tool_round_tail = self.max_tool_rounds * ((2 * self.tool_completion_timeout) + self.turn_response_timeout)
        turn_tail = max_input_seconds + self.turn_response_timeout + tool_round_tail + 10
        # Initialization has three separately bounded waits: session.created,
        # conversation.created, and the session.updated acknowledgement.
        connection_tail = WS_CONNECT_TIMEOUT + (3 * self.session_init_timeout) + 10
        safety_tail = max(HARD_DEADLINE_BUFFER, turn_tail, connection_tail)

        if self.session_end_time is not None:
            remaining = max(self.session_end_time - time.time(), 0.0)
            return remaining + safety_tail
        if self.metrics_start_time is not None:
            admission_end = self.metrics_start_time + self.test_duration
            remaining = max(admission_end - time.time(), 0.0)
            return remaining + safety_tail
        return self.test_duration + safety_tail

    @staticmethod
    def _wav_duration_seconds(path: Path) -> float:
        """Read one WAV duration for the outer deadline's input-stream budget."""
        try:
            with wave.open(str(path), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                if sample_rate <= 0:
                    raise RealtimeClientError(f"WAV file has an invalid sample rate: {path}")
                return wav_file.getnframes() / sample_rate
        except (OSError, EOFError, wave.Error) as exc:
            raise RealtimeClientError(f"Could not inspect WAV duration for {path}: {exc}") from exc

    async def _run_connection(self) -> None:
        ssl_context: ssl.SSLContext | None = None
        if self.scheme == "wss":
            if self.insecure:
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            else:
                ssl_context = ssl.create_default_context()
                if self.ca_file is not None:
                    ssl_context.load_verify_locations(cafile=str(self.ca_file))
        headers: dict[str, str] = {}
        if self.api_key_env:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RealtimeClientError("Configured Realtime API key environment variable is not set")
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            websocket = await websockets.connect(
                self.uri,
                open_timeout=WS_CONNECT_TIMEOUT,
                ssl=ssl_context,
                subprotocols=["realtime"],
                additional_headers=headers or None,
                compression=None,
            )
        except TimeoutError as exc:
            raise RealtimeClientError(f"WebSocket connection did not open within {WS_CONNECT_TIMEOUT:.1f}s") from exc

        try:
            await self.logger.log(f"{self.stream_id} OpenAI Realtime websocket connected")
            await self._initialize_session(websocket)
            self._event_queue = asyncio.Queue()
            self._reader_task = asyncio.create_task(self._reader_loop(websocket))
            await self._continuous_turn_loop(websocket)
        finally:
            await websocket.close()

    async def _receive_initialization_event(self, websocket, phase: str) -> _ReceivedEvent:
        """Receive one initialization event with an actionable phase timeout."""
        try:
            return await asyncio.wait_for(self._receive_wire_event(websocket), timeout=self.session_init_timeout)
        except TimeoutError as exc:
            raise RealtimeClientError(f"Timed out after {self.session_init_timeout:.1f}s waiting for {phase}") from exc

    async def _initialize_session(self, websocket) -> None:
        session: dict[str, Any] | None = None
        conversation: dict[str, Any] | None = None
        while session is None or conversation is None:
            received = await self._receive_initialization_event(
                websocket,
                "session.created and conversation.created",
            )
            event = received.event
            event_type = event["type"]
            if event_type == "error":
                raise self._server_error(event)
            if event_type == "session.created":
                if session is not None:
                    raise RealtimeClientError("Received duplicate session.created")
                session = self._validate_session(event.get("session"), expected_id=None)
                if self.model and session.get("model") != self.model:
                    raise RealtimeClientError(
                        f"session.created model {session.get('model')!r} does not match requested {self.model!r}"
                    )
                self._session_id = session["id"]
                self._configure_tool_ownership(session)
                continue
            if event_type == "conversation.created":
                if conversation is not None:
                    raise RealtimeClientError("Received duplicate conversation.created")
                body = _require_object(event.get("conversation"), "conversation.created.conversation")
                self._conversation_id = _require_nonempty_string(body.get("id"), "conversation.id")
                if body.get("object") != "realtime.conversation":
                    raise RealtimeClientError("conversation.object must be 'realtime.conversation'")
                conversation = body
                continue
            raise RealtimeClientError(
                f"Expected session.created/conversation.created during handshake, received {event_type!r}"
            )

        advertised_audio = _require_object(session.get("audio"), "session.created.audio")
        advertised_input = _require_object(advertised_audio.get("input"), "session.created.audio.input")
        advertised_turn_detection = advertised_input.get("turn_detection")
        if self.turn_mode == "automatic":
            if not isinstance(advertised_turn_detection, dict) or advertised_turn_detection.get("type") not in {
                "server_vad",
                "semantic_vad",
            }:
                raise RealtimeClientError("session.created did not advertise a supported automatic turn detection mode")
            requested_turn_detection = copy.deepcopy(advertised_turn_detection)
        else:
            requested_turn_detection = None

        update: dict[str, Any] = {
            "type": "realtime",
            "output_modalities": [self.output_modality],
            "tools": [copy.deepcopy(tool) for tool in self._selected_tool_schemas],
            "tool_choice": copy.deepcopy(self._selected_tool_choice),
            "parallel_tool_calls": self._selected_parallel_tool_calls,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": REALTIME_PCM_RATE},
                    "turn_detection": requested_turn_detection,
                },
                "output": {"format": {"type": "audio/pcm", "rate": REALTIME_PCM_RATE}},
            },
        }
        if self.voice:
            update["audio"]["output"]["voice"] = self.voice
        if self.instructions:
            update["instructions"] = self.instructions
        update_event_id = await self._send_event(
            websocket,
            "session.update",
            {"session": update},
            ack="session.updated",
        )

        expected_mcp_labels = {tool["server_label"] for tool in self._selected_tool_schemas if tool["type"] == "mcp"}
        updated: dict[str, Any] | None = None
        while updated is None or not self._mcp_discovery_complete(expected_mcp_labels):
            received = await self._receive_initialization_event(
                websocket,
                "session.updated and MCP discovery completion",
            )
            event = received.event
            event_type = event["type"]
            if event_type == "error":
                raise self._server_error(event)
            if event_type == "session.updated":
                if updated is not None:
                    raise RealtimeClientError("Received duplicate session.updated")
                updated = self._validate_session(event.get("session"), expected_id=self._session_id)
                continue
            if event_type in {"conversation.item.added", "conversation.item.done"}:
                item = _require_object(event.get("item"), f"{event_type}.item")
                if item.get("type") != "mcp_list_tools":
                    raise RealtimeClientError(
                        f"Unexpected {item.get('type')!r} conversation item during session initialization"
                    )
                self._capture_mcp_discovery_item_event(event)
                continue
            if event_type.startswith("mcp_list_tools."):
                self._capture_mcp_discovery_event(event)
                continue
            raise RealtimeClientError(
                f"Expected session.updated or MCP discovery during initialization, received {event_type!r}"
            )

        assert updated is not None
        if self.model and updated.get("model") != self.model:
            raise RealtimeClientError(
                f"session.updated model {updated.get('model')!r} does not match requested {self.model!r}"
            )
        if updated.get("output_modalities") != [self.output_modality]:
            raise RealtimeClientError("session.updated did not apply the requested output modality")
        updated_audio = _require_object(updated.get("audio"), "session.audio")
        input_audio = _require_object(updated_audio.get("input"), "session.audio.input")
        output_audio = _require_object(updated_audio.get("output"), "session.audio.output")
        if self.voice and output_audio.get("voice") != self.voice:
            raise RealtimeClientError(
                f"session.updated voice {output_audio.get('voice')!r} does not match requested {self.voice!r}"
            )
        if self.instructions and updated.get("instructions") != self.instructions:
            raise RealtimeClientError("session.updated did not apply the requested instructions exactly")
        updated_tools = _validate_realtime_tools(updated.get("tools"), "session.updated.tools")
        if updated_tools != _public_tool_view(self._selected_tool_schemas):
            raise RealtimeClientError("session.updated did not apply the requested tools exactly")
        updated_tool_choice = _validate_tool_choice(
            updated.get("tool_choice"),
            updated_tools,
            "session.updated.tool_choice",
        )
        if updated_tool_choice != self._selected_tool_choice:
            raise RealtimeClientError("session.updated did not apply the requested tool_choice exactly")
        if updated.get("parallel_tool_calls") != self._selected_parallel_tool_calls:
            raise RealtimeClientError("session.updated did not apply parallel_tool_calls exactly")
        if self._trusted_names_from_session(updated) != self._trusted_tool_names:
            raise RealtimeClientError("session.updated changed the trusted NVIDIA tool ownership extension")
        actual_turn_detection = input_audio.get("turn_detection")
        if self.turn_mode == "manual" and actual_turn_detection is not None:
            raise RealtimeClientError("session.updated did not apply the requested turn detection mode")
        if self.turn_mode == "automatic" and actual_turn_detection != requested_turn_detection:
            raise RealtimeClientError("session.updated changed the advertised automatic turn detection mode")
        self._correlate(update_event_id, "session.updated", resource_id=self._session_id)

    def _capture_mcp_discovery_item_event(self, event: dict[str, Any]) -> None:
        """Validate one hosted-MCP discovery conversation item lifecycle event."""
        event_type = event["type"]
        item = self._validate_conversation_item_snapshot(event.get("item"), f"{event_type}.item")
        item_id = item["id"]
        server_label = item["server_label"]
        configured_labels = {tool["server_label"] for tool in self._selected_tool_schemas if tool["type"] == "mcp"}
        if server_label not in configured_labels:
            raise RealtimeClientError(f"MCP discovery item {item_id!r} references unconfigured server {server_label!r}")
        if event_type == "conversation.item.added":
            if server_label in self._mcp_discoveries:
                raise RealtimeClientError(f"MCP server {server_label!r} emitted more than one discovery item")
            self._capture_item_event(event, [], set(), _TurnInputTrace())
            self._mcp_discoveries[server_label] = _MCPDiscoveryTrace(
                item_id=item_id,
                server_label=server_label,
            )
            return
        if event_type != "conversation.item.done":
            raise RealtimeClientError(f"Unsupported MCP discovery item event {event_type!r}")
        discovery = self._mcp_discoveries.get(server_label)
        if discovery is None or discovery.item_id != item_id:
            raise RealtimeClientError(f"MCP discovery item {item_id!r} was not announced")
        if discovery.item_done:
            raise RealtimeClientError(f"MCP discovery item {item_id!r} repeated conversation.item.done")
        self._capture_item_event(event, [], set(), _TurnInputTrace())
        discovery.item_done = True

    def _capture_mcp_discovery_event(self, event: dict[str, Any]) -> None:
        """Advance one canonical unprefixed hosted-MCP discovery lifecycle."""
        event_type = event["type"]
        item_id = _require_nonempty_string(event.get("item_id"), f"{event_type}.item_id")
        matches = [trace for trace in self._mcp_discoveries.values() if trace.item_id == item_id]
        if len(matches) != 1:
            raise RealtimeClientError(f"{event_type} references unknown MCP discovery item {item_id!r}")
        discovery = matches[0]
        if event_type == "mcp_list_tools.in_progress":
            if discovery.phase is not _MCPDiscoveryPhase.ANNOUNCED:
                raise RealtimeClientError(f"{event_type} repeated or followed a terminal discovery event")
            discovery.phase = _MCPDiscoveryPhase.IN_PROGRESS
            return
        if event_type == "mcp_list_tools.completed":
            if discovery.phase is not _MCPDiscoveryPhase.IN_PROGRESS or not discovery.item_done:
                raise RealtimeClientError(f"{event_type} must follow in_progress and the terminal discovery item")
            discovery.phase = _MCPDiscoveryPhase.COMPLETED
            return
        if event_type == "mcp_list_tools.failed":
            if discovery.phase not in {_MCPDiscoveryPhase.ANNOUNCED, _MCPDiscoveryPhase.IN_PROGRESS}:
                raise RealtimeClientError(f"{event_type} repeated a terminal discovery event")
            discovery.phase = _MCPDiscoveryPhase.FAILED
            raise RealtimeClientError(f"MCP tool discovery failed for configured server {discovery.server_label!r}")
        raise RealtimeClientError(f"Unsupported MCP discovery event {event_type!r}")

    def _mcp_discovery_complete(self, expected_labels: set[str]) -> bool:
        """Return whether every configured MCP server completed exactly one discovery."""
        observed_labels = set(self._mcp_discoveries)
        unexpected_labels = observed_labels - expected_labels
        if unexpected_labels:
            raise RealtimeClientError(f"MCP discovery included unconfigured servers: {sorted(unexpected_labels)}")
        return expected_labels == observed_labels and all(
            trace.phase is _MCPDiscoveryPhase.COMPLETED and trace.item_done for trace in self._mcp_discoveries.values()
        )

    def _configure_tool_ownership(self, session: dict[str, Any]) -> None:
        created_tools = _validate_realtime_tools(session.get("tools"), "session.created.tools")
        created_names = {tool["name"] for tool in created_tools if tool["type"] == "function"}
        trusted_names = self._trusted_names_from_session(session)
        unknown_trusted = trusted_names - created_names
        if unknown_trusted:
            raise RealtimeClientError(
                f"session.created NVIDIA ownership names have no matching tool schema: {sorted(unknown_trusted)}"
            )
        self._trusted_tool_names = trusted_names

        if self.client_tools_config is None:
            ambiguous = created_names - trusted_names
            if ambiguous:
                raise RealtimeClientError(
                    "session.created exposed tools without NVIDIA server/delegate ownership and no "
                    f"client-tools config was supplied: {sorted(ambiguous)}"
                )
            self._selected_tool_schemas = created_tools
            self._selected_tool_choice = _validate_tool_choice(
                session.get("tool_choice", "auto"),
                created_tools,
                "session.created.tool_choice",
            )
            parallel = session.get("parallel_tool_calls", True)
            if not isinstance(parallel, bool):
                raise RealtimeClientError("session.created.parallel_tool_calls must be a boolean")
            self._selected_parallel_tool_calls = parallel
            return

        client_names = {tool["name"] for tool in self.client_tools_config.tools if tool["type"] == "function"}
        conflicts = client_names & trusted_names
        if conflicts:
            raise RealtimeClientError(
                f"Client-tools config conflicts with trusted NVIDIA server/delegate names: {sorted(conflicts)}"
            )
        self._active_client_tool_names = client_names
        self._selected_tool_schemas = self.client_tools_config.tools
        self._selected_tool_choice = copy.deepcopy(self.client_tools_config.tool_choice)
        self._selected_parallel_tool_calls = self.client_tools_config.parallel_tool_calls

    @staticmethod
    def _trusted_names_from_session(session: dict[str, Any]) -> set[str]:
        nvidia = session.get("nvidia")
        if nvidia is None:
            return set()
        nvidia = _require_object(nvidia, "session.nvidia")
        trusted: set[str] = set()
        for field_name in ("server_tools", "delegate_tools"):
            raw_names = nvidia.get(field_name, [])
            if not isinstance(raw_names, list):
                raise RealtimeClientError(f"session.nvidia.{field_name} must be an array")
            for index, name in enumerate(raw_names):
                name = _require_nonempty_string(name, f"session.nvidia.{field_name}[{index}]")
                if name in trusted:
                    raise RealtimeClientError(f"session.nvidia repeats trusted tool name {name!r}")
                trusted.add(name)
        return trusted

    def _validate_session(self, value: Any, *, expected_id: str | None) -> dict[str, Any]:
        session = _require_object(value, "session")
        session_id = _require_nonempty_string(session.get("id"), "session.id")
        if expected_id is not None and session_id != expected_id:
            raise RealtimeClientError(f"session ID changed from {expected_id!r} to {session_id!r}")
        if session.get("object") != "realtime.session" or session.get("type") != "realtime":
            raise RealtimeClientError("Server returned a non-Realtime session object")
        modalities = session.get("output_modalities")
        if modalities not in (["audio"], ["text"]):
            raise RealtimeClientError("session.output_modalities must be exactly ['audio'] or ['text']")
        audio = _require_object(session.get("audio"), "session.audio")
        for side in ("input", "output"):
            block = _require_object(audio.get(side), f"session.audio.{side}")
            fmt = _require_object(block.get("format"), f"session.audio.{side}.format")
            if fmt != {"type": "audio/pcm", "rate": REALTIME_PCM_RATE}:
                raise RealtimeClientError(f"session.audio.{side}.format is not canonical 24 kHz PCM")
        return session

    async def _reader_loop(self, websocket) -> None:
        assert self._event_queue is not None
        try:
            while True:
                await self._event_queue.put(await self._receive_wire_event(websocket))
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001
            await self._event_queue.put(exc)

    async def _receive_wire_event(self, websocket) -> _ReceivedEvent:
        raw = await websocket.recv()
        received_at = time.time()
        if not isinstance(raw, str):
            raise RealtimeClientError("OpenAI Realtime server sent a binary WebSocket message")
        try:
            event = json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_constant,
                parse_float=_parse_finite_json_float,
            )
        except (json.JSONDecodeError, ValueError, OverflowError) as exc:
            raise RealtimeClientError(f"OpenAI Realtime server sent invalid or non-finite JSON: {exc}") from exc
        event = _require_object(event, "server event")
        event_type = _require_nonempty_string(event.get("type"), "server event.type")
        if event_type in _NON_GA_EVENT_TYPES:
            raise RealtimeClientError(f"Server emitted non-canonical event type {event_type!r}")
        event_id = _require_nonempty_string(event.get("event_id"), f"{event_type}.event_id")
        if event_id in self._seen_server_event_ids:
            raise RealtimeClientError(f"Server reused event_id {event_id!r}")
        self._seen_server_event_ids.add(event_id)
        await self._record_protocol_event("server", event, received_at)
        return _ReceivedEvent(event=event, received_at=received_at)

    async def _next_queued_event(self) -> _ReceivedEvent:
        assert self._event_queue is not None
        queued = await self._event_queue.get()
        if isinstance(queued, BaseException):
            raise queued
        return queued

    async def _next_event(self, timeout: float) -> _ReceivedEvent:
        return await asyncio.wait_for(self._next_queued_event(), timeout=timeout)

    async def _stream_pcm_with_event_prefetch(
        self,
        websocket,
        pcm: bytes,
        prefetched: deque[_ReceivedEvent],
    ) -> tuple[float, bool]:
        """Stream audio while preserving and reacting to concurrently received events.

        Automatic turn detection can finish a turn before a source WAV has been fully sent. Once
        that boundary is visible, cooperatively stop the stream before another
        chunk can turn the WAV's remainder into a second utterance. Every event
        consumed here remains ordered for the normal lifecycle validator.
        """
        stop_event = asyncio.Event()
        stream_task = asyncio.create_task(self._stream_pcm(websocket, pcm, stop_event=stop_event))
        event_task: asyncio.Task[_ReceivedEvent] | None = asyncio.create_task(self._next_queued_event())
        boundary_at: float | None = None
        try:
            while True:
                assert event_task is not None
                done, _ = await asyncio.wait(
                    {stream_task, event_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if event_task in done:
                    completed_event_task = event_task
                    event_task = None
                    received = completed_event_task.result()
                    prefetched.append(received)
                    if received.event["type"] in _INPUT_STREAM_BOUNDARY_EVENTS:
                        boundary_at = received.received_at
                        stop_event.set()

                if stream_task in done or boundary_at is not None:
                    stream_ended_at = await stream_task
                    if event_task is not None:
                        raced_event = await self._cancel_and_harvest_event(event_task)
                        event_task = None
                        if raced_event is not None:
                            prefetched.append(raced_event)
                            if raced_event.event["type"] in _INPUT_STREAM_BOUNDARY_EVENTS:
                                boundary_at = raced_event.received_at
                    return (boundary_at if boundary_at is not None else stream_ended_at), boundary_at is not None

                event_task = asyncio.create_task(self._next_queued_event())
        finally:
            stop_event.set()
            if not stream_task.done():
                stream_task.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await stream_task
            if event_task is not None:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    raced_event = await self._cancel_and_harvest_event(event_task)
                    if raced_event is not None:
                        prefetched.append(raced_event)

    @staticmethod
    async def _cancel_and_harvest_event(
        event_task: asyncio.Task[_ReceivedEvent],
    ) -> _ReceivedEvent | None:
        """Cancel a queue getter without dropping an event that won the race."""
        event_task.cancel()
        try:
            return await event_task
        except asyncio.CancelledError:
            return None

    async def _send_event(
        self,
        websocket,
        event_type: str,
        payload: dict[str, Any],
        *,
        ack: str | None = None,
    ) -> str:
        event_id = _new_event_id()
        event = {"event_id": event_id, "type": event_type, **payload}
        sent_at = time.time()
        if ack is not None:
            self._pending_events[event_id] = {"type": event_type, "ack": ack, "sent_at": sent_at}
        try:
            await websocket.send(json.dumps(event, separators=(",", ":")))
        except BaseException:
            self._pending_events.pop(event_id, None)
            raise
        await self._record_protocol_event("client", event, time.time())
        return event_id

    async def _send_audio_chunk(self, websocket, chunk: bytes) -> None:
        event = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(chunk).decode("ascii"),
        }
        await websocket.send(json.dumps(event, separators=(",", ":")))
        await self._record_protocol_event("client", event, time.time())
        self._input_append_count += 1
        self._input_audio_bytes += len(chunk)

    async def _record_protocol_event(self, direction: str, event: dict[str, Any], timestamp: float) -> None:
        sanitized = copy.deepcopy(event)
        _redact_mcp_credentials(sanitized)
        event_type = str(sanitized.get("type") or "")
        payload_field = {
            "input_audio_buffer.append": "audio",
            "response.output_audio.delta": "delta",
        }.get(event_type)
        if payload_field is not None and isinstance(sanitized.get(payload_field), str):
            encoded = sanitized.pop(payload_field)
            try:
                payload_bytes = len(base64.b64decode(encoded, validate=True))
            except Exception as exc:  # noqa: BLE001
                raise RealtimeClientError(f"{event_type}.{payload_field} contains invalid base64: {exc}") from exc
            correlation_fields = {
                key: sanitized[key]
                for key in ("response_id", "item_id", "output_index", "content_index")
                if key in sanitized
            }
            aggregate_key = (direction, event_type)
            aggregate = self._high_volume_protocol_events.get(aggregate_key)
            if aggregate is None:
                aggregate = {
                    "timestamp": dt.datetime.fromtimestamp(timestamp).isoformat(),
                    "direction": direction,
                    "type": event_type,
                    "data": {
                        "capture": "aggregated",
                        "event_count": 0,
                        "payload_base64_chars": 0,
                        "payload_bytes": 0,
                        "first_event_id": sanitized.get("event_id"),
                        "last_event_id": sanitized.get("event_id"),
                        "first_correlation": copy.deepcopy(correlation_fields),
                        "last_correlation": copy.deepcopy(correlation_fields),
                    },
                }
                self._high_volume_protocol_events[aggregate_key] = aggregate
                self.protocol_events.append(aggregate)
            data = aggregate["data"]
            data["event_count"] += 1
            data["payload_base64_chars"] += len(encoded)
            data["payload_bytes"] += payload_bytes
            data["last_event_id"] = sanitized.get("event_id")
            data["last_correlation"] = copy.deepcopy(correlation_fields)
            data["last_timestamp"] = dt.datetime.fromtimestamp(timestamp).isoformat()
            return
        self.protocol_events.append(
            {
                "timestamp": dt.datetime.fromtimestamp(timestamp).isoformat(),
                "direction": direction,
                "type": event_type,
                "data": sanitized,
            }
        )
        await self.logger.log(
            f"{self.stream_id} {direction} event type={event_type} event_id={event.get('event_id', '-')}"
        )

    async def _continuous_turn_loop(self, websocket) -> None:
        turn_index = 0
        while not self.shutdown_requested():
            now = time.time()
            if self.session_end_time is not None and now >= self.session_end_time:
                break
            self._collecting_metrics = bool(self.metrics_start_time is None or now >= self.metrics_start_time)
            if self.metrics_start_time is not None and now >= self.metrics_start_time + self.test_duration:
                break

            turn_input: Path | str
            if self.input_mode == "audio":
                turn_input = self.audio_files[turn_index % len(self.audio_files)]
            else:
                turn_input = self.text_inputs[turn_index % len(self.text_inputs)]
            try:
                turn = await self._run_turn(websocket, turn_input)
            except (TimeoutError, RealtimeTurnError) as exc:
                if self._collecting_metrics:
                    self.failed_turns += 1
                await self.logger.log(f"{self.stream_id} Realtime turn failed: {exc}")
                raise RealtimeClientError(
                    f"Realtime turn stopped the session to preserve response/event correlation: {exc}"
                ) from exc

            latency = turn.first_output_at - turn.input_finished_at
            self.response_transcripts.append(
                {
                    "turn": turn_index,
                    "response_ids": turn.response_ids,
                    "transcript": turn.transcript,
                    "output_audio_bytes": turn.output_bytes,
                    "tool_call_ids": turn.tool_call_ids,
                    "latency": latency,
                }
            )
            if self._collecting_metrics:
                self.latency_values.append(latency)
                is_audio_playback_turn = self.output_modality == "audio" and turn.output_bytes > 0
                if is_audio_playback_turn and latency < self.reverse_barge_in_threshold:
                    self.reverse_barge_ins_count += 1
                else:
                    self.valid_latency_values.append(latency)
            await self.logger.log(
                f"{self.stream_id} Realtime turn complete latency={latency:.3f}s "
                f"responses={turn.response_ids} output_bytes={turn.output_bytes}"
            )
            turn_index += 1
            await asyncio.sleep(0.1)

    async def _run_turn(self, websocket, turn_input_value: Path | str) -> _TurnResult:
        prefetched_events: deque[_ReceivedEvent] = deque()
        pending_ids: list[str] = []
        turn_input = _TurnInputTrace()
        if self.input_mode == "audio":
            if not isinstance(turn_input_value, Path):
                raise RealtimeClientError("Audio input mode selected a non-path turn input")
            pcm = self._read_wav_as_pcm_24khz(turn_input_value)
            input_boundary_observed = False
            if self.turn_mode == "automatic":
                input_finished_at, input_boundary_observed = await self._stream_pcm_with_event_prefetch(
                    websocket,
                    pcm,
                    prefetched_events,
                )
            else:
                input_finished_at = await self._stream_pcm(websocket, pcm)

            if self.turn_mode == "automatic" and not input_boundary_observed and self.vad_silence_ms:
                silence = b"\x00\x00" * (REALTIME_PCM_RATE * self.vad_silence_ms // 1000)
                _, input_boundary_observed = await self._stream_pcm_with_event_prefetch(
                    websocket,
                    silence,
                    prefetched_events,
                )
            if self.turn_mode == "manual":
                pending_ids.append(
                    await self._send_event(
                        websocket,
                        "input_audio_buffer.commit",
                        {},
                        ack="input_audio_buffer.committed",
                    )
                )
                pending_ids.append(await self._send_event(websocket, "response.create", {}, ack="response.created"))
        else:
            if not isinstance(turn_input_value, str):
                raise RealtimeClientError("Text input mode selected a non-text turn input")
            input_finished_at, text_pending_ids = await self._submit_text_turn(
                websocket,
                turn_input_value,
                turn_input,
            )
            pending_ids.extend(text_pending_ids)

        deadline_at = input_finished_at + self.turn_response_timeout
        tool_deadline_at: float | None = None

        response_ids: list[str] = []
        turn_tool_call_ids: set[str] = set()
        pending_client_output_calls: set[str] = set()
        tool_rounds = 0
        first_output_at: float | None = None
        turn_transcript = ""
        output_bytes = 0
        playback_buffer_seconds = 0.0
        last_audio_at: float | None = None

        try:
            while True:
                if prefetched_events:
                    received = prefetched_events.popleft()
                else:
                    active_deadline = tool_deadline_at if tool_deadline_at is not None else deadline_at
                    remaining = active_deadline - time.time()
                    if remaining <= 0:
                        if tool_deadline_at is not None:
                            raise TimeoutError(f"no correlated tool output within {self.tool_completion_timeout:.1f}s")
                        phase = "response completion" if first_output_at is not None else "first response output"
                        raise TimeoutError(f"no {phase} activity within {self.turn_response_timeout:.1f}s")
                    try:
                        received = await self._next_event(remaining)
                    except TimeoutError as exc:
                        if tool_deadline_at is not None:
                            message = f"no correlated tool output within {self.tool_completion_timeout:.1f}s"
                        else:
                            phase = "response completion" if first_output_at is not None else "first response output"
                            message = f"no {phase} activity within {self.turn_response_timeout:.1f}s"
                        raise TimeoutError(message) from exc
                event = received.event
                event_type = event["type"]
                if event_type == "error":
                    if self._capture_tool_failure_error(event, turn_tool_call_ids):
                        continue
                    raise self._server_error(event)
                if first_output_at is not None and event_type.startswith("response."):
                    deadline_at = max(deadline_at, received.received_at + self.turn_response_timeout)
                if event_type == "rate_limits.updated":
                    limits = event.get("rate_limits")
                    if not isinstance(limits, list):
                        raise RealtimeClientError("rate_limits.updated.rate_limits must be an array")
                    self.rate_limits.append(copy.deepcopy(event))
                    continue
                if event_type == "nvidia.metrics.updated":
                    self._capture_nvidia_metrics(event)
                    continue
                if event_type == "input_audio_buffer.committed":
                    item_id = _require_nonempty_string(event.get("item_id"), "input_audio_buffer.committed.item_id")
                    self._capture_input_lifecycle_event(event, turn_input)
                    self._correlate_oldest("input_audio_buffer.committed", resource_id=item_id)
                    continue
                if event_type in {
                    "input_audio_buffer.cleared",
                    "input_audio_buffer.speech_started",
                    "input_audio_buffer.speech_stopped",
                    "input_audio_buffer.timeout_triggered",
                }:
                    self._capture_input_lifecycle_event(event, turn_input)
                    continue
                if event_type == "response.created":
                    response = _require_object(event.get("response"), "response.created.response")
                    response_id = _require_nonempty_string(response.get("id"), "response.id")
                    if turn_input.item_id is None:
                        raise RealtimeClientError(
                            "Received an unsolicited response before this benchmark turn's input was submitted; "
                            "disable the pipeline welcome message for scaling-perf"
                        )
                    if self.input_mode == "text" and not response_ids:
                        user_item = self._conversation_items.get(turn_input.item_id)
                        response_request_sent_at = self._pending_ack_sent_at("response.created")
                        if user_item is None or user_item.done is None or response_request_sent_at is None:
                            raise RealtimeClientError(
                                "Text response started before the submitted conversation item was terminal"
                            )
                    previous_trace = self._response_traces[response_ids[-1]] if response_ids else None
                    requires_explicit_followup = bool(
                        previous_trace is not None and self._response_requires_client_followup(previous_trace)
                    )
                    if requires_explicit_followup:
                        response_request_sent_at = self._pending_ack_sent_at("response.created")
                        if response_request_sent_at is None:
                            raise RealtimeClientError(
                                "Tool recovery response started without the client's explicit response.create"
                            )
                        if received.received_at < response_request_sent_at:
                            raise RealtimeClientError("Tool recovery response was emitted before response.create")
                    self._validate_recovery_response_start(response_id, response_ids, turn_tool_call_ids)
                    self._start_response_graph(response, received.received_at)
                    response_ids.append(response_id)
                    self._correlate_oldest("response.created", resource_id=response_id)
                    continue
                if event_type == "response.output_audio.delta":
                    if self.output_modality != "audio":
                        raise RealtimeClientError("Text-only session received response.output_audio.delta")
                    self._capture_response_stream_event(event)
                    trace = self._response_trace_for_event(event)
                    encoded = _require_nonempty_string(event.get("delta"), f"{event_type}.delta")
                    try:
                        audio = base64.b64decode(encoded, validate=True)
                    except Exception as exc:  # noqa: BLE001
                        raise RealtimeClientError(f"Invalid base64 output audio: {exc}") from exc
                    if len(audio) % 2:
                        raise RealtimeClientError("Output PCM delta contains an odd number of bytes")
                    if first_output_at is None:
                        first_output_at = received.received_at
                    deadline_at = received.received_at + self.turn_response_timeout
                    trace.output_bytes += len(audio)
                    output_bytes += len(audio)
                    self._write_output_audio(audio)
                    if last_audio_at is not None:
                        arrival_gap_seconds = received.received_at - last_audio_at
                        buffered_before_gap = playback_buffer_seconds
                        playback_buffer_seconds -= arrival_gap_seconds
                        if playback_buffer_seconds < -0.020:
                            self.glitch_detected = True
                            await self.logger.log(
                                f"{self.stream_id} audio glitch detected: response={trace.response_id} "
                                f"arrival_gap_ms={arrival_gap_seconds * 1000:.1f} "
                                f"buffered_ms={buffered_before_gap * 1000:.1f} "
                                f"underrun_ms={-playback_buffer_seconds * 1000:.1f} "
                                f"next_chunk_bytes={len(audio)}"
                            )
                            playback_buffer_seconds = 0.0
                    playback_buffer_seconds += len(audio) / (2 * REALTIME_PCM_RATE)
                    last_audio_at = received.received_at
                    continue
                if event_type in {"response.output_audio_transcript.delta", "response.output_text.delta"}:
                    self._capture_response_stream_event(event)
                    trace = self._response_trace_for_event(event)
                    delta = event.get("delta")
                    if not isinstance(delta, str):
                        raise RealtimeClientError(f"{event_type}.delta must be a string")
                    if event_type == "response.output_text.delta" and self.output_modality != "text":
                        raise RealtimeClientError("Audio session received response.output_text.delta")
                    if (
                        event_type == "response.output_text.delta"
                        and first_output_at is None
                        and _contains_user_visible_text(delta)
                    ):
                        first_output_at = received.received_at
                        deadline_at = received.received_at + self.turn_response_timeout
                    trace.transcript += delta
                    trace.has_text = True
                    turn_transcript += delta
                    continue
                if event_type in {"response.function_call_arguments.delta", "response.function_call_arguments.done"}:
                    trace, call_id = self._capture_function_arguments_event(event)
                    trace.has_function_call = True
                    turn_tool_call_ids.add(call_id)
                    if tool_deadline_at is None:
                        tool_deadline_at = received.received_at + self.tool_completion_timeout
                    call = self._tool_call(call_id, event, trace.response_id)
                    if event_type.endswith(".delta"):
                        call["arguments"] += event["delta"]
                    else:
                        call["arguments"] = event["arguments"]
                        call["status"] = "completed"
                    continue
                if event_type in {
                    "conversation.item.added",
                    "conversation.item.done",
                    "response.output_item.added",
                    "response.output_item.done",
                }:
                    self._capture_item_event(event, response_ids, turn_tool_call_ids, turn_input)
                    item = event.get("item")
                    item_type = item.get("type") if isinstance(item, dict) else None
                    if (
                        event_type == "conversation.item.done"
                        and isinstance(item, dict)
                        and item.get("id") == turn_input.item_id
                        and turn_input.submitted_text is not None
                    ):
                        self._correlate_oldest("conversation.item.done", resource_id=item["id"])
                    if item_type in {"function_call", "mcp_call"} and tool_deadline_at is None:
                        tool_deadline_at = received.received_at + self.tool_completion_timeout
                    if item_type == "mcp_approval_request" and event_type == "conversation.item.done":
                        await self._answer_mcp_approval_request(websocket, item, pending_ids)
                    if item_type == "function_call_output" and event_type == "conversation.item.done":
                        call_id = _require_nonempty_string(item.get("call_id"), "function_call_output.call_id")
                        call = self._tool_calls[call_id]
                        if call_id in pending_client_output_calls:
                            if not call.get("server_generated_output"):
                                output_event_id = _require_nonempty_string(
                                    call.get("output_event_id"),
                                    f"client tool {call_id}.output_event_id",
                                )
                                self._correlate(
                                    output_event_id,
                                    "conversation.item.done",
                                    resource_id=item["id"],
                                )
                            pending_client_output_calls.discard(call_id)
                        response_trace = self._response_traces[str(call["response_id"])]
                        followup_sent = await self._request_tool_followup_if_ready(
                            websocket,
                            response_trace,
                            pending_client_output_calls,
                            pending_ids,
                        )
                        response_functions_done = all(
                            self._tool_calls[response_call_id].get("output_item_id")
                            for response_call_id in self._response_call_ids(response_trace.response_id)
                        )
                        response_mcp_done = all(
                            output.output_done for output in self._response_mcp_outputs(response_trace.response_id)
                        )
                        if response_functions_done and response_mcp_done:
                            tool_deadline_at = None
                            deadline_at = received.received_at + self.turn_response_timeout
                        elif not followup_sent and tool_deadline_at is None:
                            tool_deadline_at = received.received_at + self.tool_completion_timeout
                    if item_type == "mcp_call" and event_type == "response.output_item.done":
                        response_id = _require_nonempty_string(
                            event.get("response_id"),
                            "response.output_item.done.response_id",
                        )
                        response_trace = self._response_traces[response_id]
                        followup_sent = await self._request_tool_followup_if_ready(
                            websocket,
                            response_trace,
                            pending_client_output_calls,
                            pending_ids,
                        )
                        response_functions_done = all(
                            self._tool_calls[response_call_id].get("output_item_id")
                            for response_call_id in self._response_call_ids(response_id)
                        )
                        response_mcp_done = all(
                            output.output_done for output in self._response_mcp_outputs(response_id)
                        )
                        if response_functions_done and response_mcp_done:
                            tool_deadline_at = None
                            deadline_at = received.received_at + self.turn_response_timeout
                        elif not followup_sent and tool_deadline_at is None:
                            tool_deadline_at = received.received_at + self.tool_completion_timeout
                    continue
                if event_type == "response.done":
                    response = _require_object(event.get("response"), "response.done.response")
                    trace = self._finish_response_graph(response)
                    response_id = trace.response_id
                    status = trace.status
                    self.response_status_counts[status] = self.response_status_counts.get(status, 0) + 1
                    self._capture_usage(response.get("usage"))
                    if self._collecting_metrics:
                        self.server_metric_samples["response_lifecycle"].append(received.received_at - trace.created_at)
                        self.server_metric_samples["response_audio_bytes"].append(float(trace.output_bytes))
                    self._capture_done_output(response, trace, turn_tool_call_ids)
                    if status != "completed":
                        raise RealtimeTurnError(
                            f"response {response_id} ended with status={status} "
                            f"details={response.get('status_details')}"
                        )
                    round_call_ids = self._response_call_ids(response_id)
                    round_mcp_outputs = self._response_mcp_outputs(response_id)
                    if trace.has_function_call and not round_call_ids:
                        raise RealtimeClientError(
                            f"response {response_id} reported a function call without a correlated call item"
                        )
                    if trace.has_mcp_call and not round_mcp_outputs:
                        raise RealtimeClientError(
                            f"response {response_id} reported an MCP call without a correlated call item"
                        )
                    if round_call_ids or round_mcp_outputs:
                        tool_rounds += 1
                        if tool_rounds > self.max_tool_rounds:
                            raise RealtimeTurnError(f"tool round limit exceeded ({self.max_tool_rounds})")
                        ownership = self._classify_tool_round(round_call_ids, tool_rounds) if round_call_ids else None
                        if len(round_call_ids) + len(round_mcp_outputs) > 1 and not self._selected_parallel_tool_calls:
                            raise RealtimeClientError(
                                "Server emitted parallel tool calls while session.parallel_tool_calls=false"
                            )
                        if ownership == "client":
                            if pending_client_output_calls:
                                raise RealtimeClientError(
                                    "A new client tool round started before the prior output acknowledgements"
                                )
                            if tool_deadline_at is None:
                                raise RealtimeClientError(
                                    "Client tool round has no absolute correlated-output deadline"
                                )
                            outputs = await self._execute_client_tool_round(
                                round_call_ids,
                                tool_deadline_at=tool_deadline_at,
                            )
                            for call_id, output in outputs:
                                call = self._tool_calls[call_id]
                                item_id = f"item_{uuid.uuid4().hex}"
                                remaining_tool_time = tool_deadline_at - time.time()
                                if remaining_tool_time <= 0:
                                    raise TimeoutError(
                                        f"client tool round exceeded its absolute "
                                        f"{self.tool_completion_timeout:.1f}s deadline before output send"
                                    )
                                output_event_id = await asyncio.wait_for(
                                    self._send_event(
                                        websocket,
                                        "conversation.item.create",
                                        {
                                            "item": {
                                                "id": item_id,
                                                "type": "function_call_output",
                                                "call_id": call_id,
                                                "output": output,
                                            }
                                        },
                                        ack="conversation.item.done",
                                    ),
                                    timeout=remaining_tool_time,
                                )
                                call["expected_output"] = output
                                call["expected_output_item_id"] = item_id
                                call["output_event_id"] = output_event_id
                                pending_ids.append(output_event_id)
                                pending_client_output_calls.add(call_id)
                        followup_sent = await self._request_tool_followup_if_ready(
                            websocket,
                            trace,
                            pending_client_output_calls,
                            pending_ids,
                        )
                        response_functions_done = all(
                            self._tool_calls[call_id].get("output_item_id") for call_id in round_call_ids
                        )
                        response_mcp_done = all(output.output_done for output in round_mcp_outputs)
                        if response_functions_done and response_mcp_done:
                            tool_deadline_at = None
                            deadline_at = received.received_at + self.turn_response_timeout
                        elif not followup_sent and tool_deadline_at is None:
                            tool_deadline_at = received.received_at + self.tool_completion_timeout
                        continue
                    has_expected_output = trace.output_bytes > 0 if self.output_modality == "audio" else trace.has_text
                    if has_expected_output:
                        if first_output_at is None:
                            first_output_at = received.received_at
                        if not turn_transcript:
                            turn_transcript = trace.transcript
                        missing_tool_outputs = [
                            call_id
                            for call_id in turn_tool_call_ids
                            if not self._tool_calls.get(call_id, {}).get("output_item_id")
                        ]
                        if missing_tool_outputs:
                            raise RealtimeTurnError(
                                "final response arrived before correlated function_call_output items for "
                                f"{sorted(missing_tool_outputs)}"
                            )
                        return _TurnResult(
                            input_finished_at=input_finished_at,
                            first_output_at=first_output_at,
                            response_ids=response_ids,
                            transcript=turn_transcript,
                            output_bytes=output_bytes,
                            tool_call_ids=sorted(turn_tool_call_ids),
                        )
                    raise RealtimeTurnError(f"response {response_id} completed without text, audio, or a tool call")
                if event_type in {"response.content_part.added", "response.content_part.done"}:
                    self._capture_content_part_event(event)
                    continue
                if event_type in {
                    "response.output_audio.done",
                    "response.output_audio_transcript.done",
                    "response.output_text.done",
                }:
                    self._capture_response_stream_event(event)
                    continue
                if event_type.startswith("conversation.item.input_audio_transcription.") or event_type in {
                    "conversation.item.deleted",
                    "conversation.item.truncated",
                    "conversation.item.retrieved",
                }:
                    self._capture_conversation_passive_event(event)
                    continue
                if event_type.startswith("response.mcp_"):
                    self._capture_mcp_event(event)
                    if tool_deadline_at is None:
                        tool_deadline_at = received.received_at + self.tool_completion_timeout
                    continue
                if event_type in _CANONICAL_PASSIVE_EVENTS:
                    raise RealtimeClientError(f"Passive event {event_type!r} was not validated")
                    continue
                if event_type in {"session.created", "session.updated", "conversation.created"}:
                    raise RealtimeClientError(f"Unexpected lifecycle event {event_type!r} after session initialization")
                raise RealtimeClientError(f"Unsupported server event type {event_type!r}")
        finally:
            for event_id in pending_ids:
                self._pending_events.pop(event_id, None)

    async def _submit_text_turn(
        self,
        websocket,
        text: str,
        turn_input: _TurnInputTrace,
    ) -> tuple[float, list[str]]:
        """Submit one canonical client-created user text item and request its response."""
        item_id = f"item_{uuid.uuid4().hex}"
        turn_input.item_id = item_id
        turn_input.submitted_text = text
        item_event_id = await self._send_event(
            websocket,
            "conversation.item.create",
            {
                "item": {
                    "id": item_id,
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                }
            },
            ack="conversation.item.done",
        )
        try:
            response_event_id = await self._send_event(
                websocket,
                "response.create",
                {},
                ack="response.created",
            )
        except BaseException:
            self._pending_events.pop(item_event_id, None)
            raise
        return time.time(), [item_event_id, response_event_id]

    def _pending_ack_sent_at(self, ack: str) -> float | None:
        matches = [pending["sent_at"] for pending in self._pending_events.values() if pending.get("ack") == ack]
        if len(matches) > 1:
            raise RealtimeClientError(f"More than one client event is awaiting {ack!r}")
        return matches[0] if matches else None

    def _response_call_ids(self, response_id: str) -> list[str]:
        trace = self._response_traces[response_id]
        call_ids: list[str] = []
        for output_index in sorted(trace.output_items):
            output = trace.output_items[output_index]
            if output.item_type != "function_call":
                continue
            call_id = _require_nonempty_string(
                output.added.get("call_id"),
                f"response {response_id} output[{output_index}].call_id",
            )
            if call_id in call_ids:
                raise RealtimeClientError(f"response {response_id} repeated function call ID {call_id!r}")
            call_ids.append(call_id)
        return call_ids

    def _response_mcp_outputs(self, response_id: str) -> list[_OutputItemTrace]:
        trace = self._response_traces[response_id]
        return [
            trace.output_items[output_index]
            for output_index in sorted(trace.output_items)
            if trace.output_items[output_index].item_type == "mcp_call"
        ]

    def _response_requires_client_followup(self, trace: _ResponseTrace) -> bool:
        if trace.has_mcp_call:
            return True
        return any(
            self._tool_calls[call_id].get("owner") == "client" for call_id in self._response_call_ids(trace.response_id)
        )

    async def _request_tool_followup_if_ready(
        self,
        websocket,
        trace: _ResponseTrace,
        pending_client_output_calls: set[str],
        pending_ids: list[str],
    ) -> bool:
        """Request Response B once all application and hosted-MCP results are terminal."""
        if trace.status == "in_progress" or trace.followup_requested:
            return False
        if not self._response_requires_client_followup(trace):
            return False
        response_call_ids = self._response_call_ids(trace.response_id)
        if any(not self._tool_calls[call_id].get("output_item_id") for call_id in response_call_ids):
            return False
        if pending_client_output_calls.intersection(response_call_ids):
            return False
        if any(not output.output_done for output in self._response_mcp_outputs(trace.response_id)):
            return False
        event_id = await self._send_event(
            websocket,
            "response.create",
            self._client_tool_recovery_response_payload(),
            ack="response.created",
        )
        trace.followup_requested = True
        pending_ids.append(event_id)
        return True

    async def _answer_mcp_approval_request(
        self,
        websocket,
        item: dict[str, Any],
        pending_ids: list[str],
    ) -> None:
        """Answer one native MCP approval request with the tester's exact-match policy."""
        request_id = item["id"]
        known = self._mcp_approval_requests.get(request_id)
        if known is None or known != item:
            raise RealtimeClientError(f"MCP approval request {request_id!r} has no matching added item")
        if request_id in self._mcp_approval_requests_handled:
            raise RealtimeClientError(f"MCP approval request {request_id!r} was already answered")
        server_label = item["server_label"]
        configured_labels = {tool["server_label"] for tool in self._selected_tool_schemas if tool["type"] == "mcp"}
        if server_label not in configured_labels:
            raise RealtimeClientError(
                f"MCP approval request {request_id!r} references unconfigured server {server_label!r}"
            )
        policy = (
            self.client_tools_config.mcp_approval
            if self.client_tools_config is not None
            else _default_mcp_approval_policy()
        )
        decision = policy.resolve(server_label, item["name"])
        response_item: dict[str, Any] = {
            "id": f"item_{uuid.uuid4().hex}",
            "type": "mcp_approval_response",
            "approval_request_id": request_id,
            "approve": decision.approve,
        }
        if decision.reason is not None:
            response_item["reason"] = decision.reason
        event_id = await self._send_event(
            websocket,
            "conversation.item.create",
            {"item": response_item},
            ack="conversation.item.done",
        )
        self._mcp_approval_requests_handled.add(request_id)
        self._mcp_approval_response_items[response_item["id"]] = _MCPApprovalResponseTrace(
            request_id=request_id,
            item=copy.deepcopy(response_item),
            event_id=event_id,
        )
        pending_ids.append(event_id)

    def _classify_tool_round(self, call_ids: list[str], round_number: int) -> str:
        owners: set[str] = set()
        selected_names = {tool["name"] for tool in self._selected_tool_schemas if tool["type"] == "function"}
        for call_id in call_ids:
            call = self._tool_calls[call_id]
            name = _require_nonempty_string(call.get("name"), f"function call {call_id}.name")
            if name not in selected_names:
                raise RealtimeClientError(f"Function call {call_id!r} selected tool {name!r} outside session.tools")
            if name in self._trusted_tool_names:
                owner = "backend"
            elif name in self._active_client_tool_names:
                owner = "client"
            else:
                raise RealtimeClientError(f"Function call {call_id!r} selected undeclared or ownerless tool {name!r}")
            call["owner"] = owner
            call["tool_round"] = round_number
            owners.add(owner)
        if len(owners) != 1:
            raise RealtimeClientError(
                "A response mixed trusted backend and client-owned function calls; mixed ownership is unsupported"
            )
        return owners.pop()

    def _client_tool_recovery_response_payload(self) -> dict[str, Any]:
        """Let Response B answer after a session-level forced tool call."""
        if self._selected_tool_choice == "required" or isinstance(self._selected_tool_choice, dict):
            return {"response": {"tool_choice": "none"}}
        return {}

    async def _execute_client_tool_round(
        self,
        call_ids: list[str],
        *,
        tool_deadline_at: float,
    ) -> list[tuple[str, str]]:
        if self.client_tools_config is None:
            raise RealtimeClientError("Client-owned tool call arrived without a client-tools config")
        handler_deadline_at = tool_deadline_at - self.client_tool_handler_margin
        reserved: list[tuple[str, ScriptedToolStep | HttpToolInvocation | BaseException]] = []
        for call_id in call_ids:
            try:
                reserved.append((call_id, self._reserve_client_tool_invocation(call_id)))
            except Exception as exc:  # noqa: BLE001
                reserved.append((call_id, exc))
        outputs = await asyncio.gather(
            *(
                self._execute_client_tool_invocation(
                    call_id,
                    invocation,
                    handler_deadline_at=handler_deadline_at,
                )
                for call_id, invocation in reserved
            )
        )
        return list(zip(call_ids, outputs, strict=True))

    def _reserve_client_tool_invocation(self, call_id: str) -> ScriptedToolStep | HttpToolInvocation:
        assert self.client_tools_config is not None
        call = self._tool_calls[call_id]
        name = _require_nonempty_string(call.get("name"), f"function call {call_id}.name")
        handler = self.client_tools_config.handlers[name]
        invocation_index = self._tool_handler_invocations.get(name, 0)
        self._tool_handler_invocations[name] = invocation_index + 1
        call["handler_invocation"] = invocation_index
        arguments_json = call.get("arguments")
        if not isinstance(arguments_json, str):
            raise _ScriptedToolError(
                "client_tool_arguments_invalid",
                f"Client tool {name!r} arguments were not a JSON string",
            )
        try:
            arguments = json.loads(
                arguments_json,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_constant,
                parse_float=_parse_finite_json_float,
            )
        except (json.JSONDecodeError, ValueError, OverflowError) as exc:
            raise _ScriptedToolError(
                "client_tool_arguments_invalid",
                f"Client tool {name!r} arguments were not finite JSON: {exc}",
            ) from exc
        if not isinstance(arguments, dict):
            raise _ScriptedToolError(
                "client_tool_arguments_invalid",
                f"Client tool {name!r} arguments must decode to a JSON object",
            )
        call["parsed_arguments"] = copy.deepcopy(arguments)
        validator = self.client_tools_config.argument_validators[name]
        try:
            validation_error = next(validator.iter_errors(arguments), None)
        except Exception as exc:  # noqa: BLE001
            raise _ScriptedToolError(
                "client_tool_schema_error",
                f"Client tool {name!r} arguments could not be checked against its declared JSON Schema",
            ) from exc
        if validation_error is not None:
            constraint = validation_error.validator if isinstance(validation_error.validator, str) else "constraint"
            raise _ScriptedToolError(
                "client_tool_arguments_invalid",
                f"Client tool {name!r} arguments did not satisfy its declared JSON Schema (constraint: {constraint})",
            )
        if isinstance(handler, HttpToolHandler):
            return HttpToolInvocation(handler=handler, arguments=arguments)

        if invocation_index < len(handler.steps):
            step = handler.steps[invocation_index]
        elif handler.repeat_last:
            step = handler.steps[-1]
        else:
            raise _ScriptedToolError(
                "client_tool_script_exhausted",
                f"No scripted output remains for client tool {name!r}",
            )
        if step.expected_arguments is not _MISSING and arguments != step.expected_arguments:
            raise _ScriptedToolError(
                "client_tool_arguments_mismatch",
                f"Client tool {name!r} arguments did not match the scripted expectation",
            )
        return step

    async def _execute_client_tool_invocation(
        self,
        call_id: str,
        invocation_or_error: ScriptedToolStep | HttpToolInvocation | BaseException,
        *,
        handler_deadline_at: float,
    ) -> str:
        call = self._tool_calls[call_id]
        name = _require_nonempty_string(call.get("name"), f"function call {call_id}.name")
        started = time.monotonic()
        call["handler_status"] = "in_progress"
        handler_budget = max(0.0, handler_deadline_at - time.time())
        try:
            if isinstance(invocation_or_error, BaseException):
                raise invocation_or_error
            result = await asyncio.wait_for(
                self._invoke_client_tool(invocation_or_error),
                timeout=handler_budget,
            )
            _validate_finite_json(result, f"Client tool {name!r} output")
            output = json.dumps(result, separators=(",", ":"), allow_nan=False)
            call["handler_status"] = "completed"
            call["handler_error"] = None
            return output
        except TimeoutError:
            code = "client_tool_timeout"
            message = (
                f"Client tool {name!r} exceeded its remaining {handler_budget:.1f}s execution budget "
                f"before the absolute {self.tool_completion_timeout:.1f}s tool deadline"
            )
        except _ScriptedToolError as exc:
            code = exc.code
            message = str(exc)
        except _HttpToolError as exc:
            code = exc.code
            message = str(exc)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            code = "client_tool_exception"
            message = str(exc) or exc.__class__.__name__
        finally:
            call["handler_latency"] = time.monotonic() - started
        call["handler_status"] = "failed"
        call["handler_error"] = {"code": code, "message": message}
        return json.dumps(
            {"ok": False, "error": {"code": code, "message": message}},
            separators=(",", ":"),
            allow_nan=False,
        )

    @staticmethod
    async def _invoke_client_tool(invocation: ScriptedToolStep | HttpToolInvocation) -> Any:
        if isinstance(invocation, ScriptedToolStep):
            if invocation.delay_ms:
                await asyncio.sleep(invocation.delay_ms / 1000.0)
            if invocation.error_code is not None:
                raise _ScriptedToolError(
                    invocation.error_code,
                    invocation.error_message or "Scripted client tool error",
                )
            return copy.deepcopy(invocation.output)

        timeout = aiohttp.ClientTimeout(total=invocation.handler.timeout_seconds)
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(
                    invocation.handler.url,
                    json=invocation.arguments,
                    allow_redirects=False,
                ) as response,
            ):
                if 300 <= response.status < 400:
                    raise _HttpToolError(
                        "client_tool_http_redirect",
                        f"HTTP client tool redirects are not allowed (status {response.status})",
                    )
                try:
                    raw = await response.content.readexactly(MAX_HTTP_TOOL_RESPONSE_BYTES + 1)
                except asyncio.IncompleteReadError as exc:
                    raw = exc.partial
                if len(raw) > MAX_HTTP_TOOL_RESPONSE_BYTES:
                    raise _HttpToolError(
                        "client_tool_http_response_too_large",
                        "HTTP client tool response exceeded 1 MiB",
                    )
                if response.status < 200 or response.status >= 300:
                    detail = raw.decode("utf-8", errors="replace").strip()[:512]
                    suffix = f": {detail}" if detail else ""
                    raise _HttpToolError(
                        "client_tool_http_status",
                        f"HTTP client tool returned status {response.status}{suffix}",
                    )
                if not raw:
                    return None
                try:
                    return json.loads(
                        raw,
                        object_pairs_hook=_reject_duplicate_json_keys,
                        parse_constant=_reject_nonfinite_json_constant,
                        parse_float=_parse_finite_json_float,
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError, OverflowError) as exc:
                    raise _HttpToolError(
                        "client_tool_http_invalid_json",
                        "HTTP client tool returned invalid JSON",
                    ) from exc
        except _HttpToolError:
            raise
        except TimeoutError as exc:
            raise _HttpToolError(
                "client_tool_http_timeout",
                f"HTTP client tool exceeded its {invocation.handler.timeout_seconds:.1f}s endpoint timeout",
            ) from exc
        except aiohttp.ClientError as exc:
            raise _HttpToolError(
                "client_tool_http_request_failed",
                str(exc) or exc.__class__.__name__,
            ) from exc

    def _response_trace_for_event(
        self,
        event: dict[str, Any],
        *,
        allow_terminal_mcp: bool = False,
    ) -> _ResponseTrace:
        response_id = _require_nonempty_string(event.get("response_id"), f"{event['type']}.response_id")
        try:
            trace = self._response_traces[response_id]
        except KeyError as exc:
            raise RealtimeClientError(f"{event['type']} references unknown response {response_id!r}") from exc
        active = trace.status == "in_progress" and self._active_response_id == response_id
        if not active and not (allow_terminal_mcp and trace.status in _TERMINAL_RESPONSE_STATUSES):
            raise RealtimeClientError(f"{event['type']} references inactive response {response_id!r}")
        return trace

    def _response_output_for_event(
        self,
        event: dict[str, Any],
        *,
        require_content: bool = False,
        allow_terminal_mcp: bool = False,
    ) -> tuple[_ResponseTrace, _OutputItemTrace, _ContentPartTrace | None]:
        trace = self._response_trace_for_event(event, allow_terminal_mcp=allow_terminal_mcp)
        output_index = _require_nonnegative_int(event.get("output_index"), f"{event['type']}.output_index")
        item_id = _require_nonempty_string(event.get("item_id"), f"{event['type']}.item_id")
        output = trace.output_items.get(output_index)
        if output is None or output.item_id != item_id:
            raise RealtimeClientError(
                f"{event['type']} does not match response {trace.response_id!r} output index {output_index}"
            )
        if output.output_done:
            raise RealtimeClientError(f"{event['type']} references terminal output item {item_id!r}")
        if not require_content:
            return trace, output, None
        content_index = _require_nonnegative_int(event.get("content_index"), f"{event['type']}.content_index")
        content = output.content_parts.get(content_index)
        if content is None:
            raise RealtimeClientError(
                f"{event['type']} references unknown content index {content_index} for item {item_id!r}"
            )
        if content.part_done:
            raise RealtimeClientError(f"{event['type']} references terminal content part {content_index}")
        return trace, output, content

    def _validate_conversation_item_snapshot(self, value: Any, label: str) -> dict[str, Any]:
        item = _require_object(value, label)
        _require_nonempty_string(item.get("id"), f"{label}.id")
        item_type = _require_nonempty_string(item.get("type"), f"{label}.type")
        mcp_item_types = {
            "mcp_call",
            "mcp_list_tools",
            "mcp_approval_request",
            "mcp_approval_response",
        }
        if item_type not in {
            "message",
            "function_call",
            "function_call_output",
            *mcp_item_types,
        }:
            raise RealtimeClientError(f"{label}.type {item_type!r} is not supported by the scaling client")
        status = item.get("status")
        if item_type in {"message", "function_call", "function_call_output"} and item.get("object") != "realtime.item":
            raise RealtimeClientError(f"{label}.object must be 'realtime.item'")
        if item_type in mcp_item_types and item.get("object") not in {None, "realtime.item"}:
            raise RealtimeClientError(f"{label}.object must be absent or 'realtime.item'")
        if status not in {"in_progress", "completed", "incomplete"} and not (
            item_type in mcp_item_types and status is None
        ):
            raise RealtimeClientError(f"{label}.status must be in_progress, completed, or incomplete")
        if item_type == "message":
            role = item.get("role")
            if role not in {"system", "user", "assistant"}:
                raise RealtimeClientError(f"{label}.role is invalid")
            content = item.get("content")
            if not isinstance(content, list):
                raise RealtimeClientError(f"{label}.content must be an array")
            allowed_content = {
                "system": {"input_text"},
                "user": {"input_text", "input_audio"},
                "assistant": {"output_text", "output_audio"},
            }[role]
            for content_index, part in enumerate(content):
                if not isinstance(part, dict) or part.get("type") not in allowed_content:
                    raise RealtimeClientError(f"{label}.content[{content_index}] has an invalid type")
                text_field = "text" if part["type"].endswith("text") else "transcript"
                if text_field in part and part[text_field] is not None and not isinstance(part[text_field], str):
                    raise RealtimeClientError(f"{label}.content[{content_index}].{text_field} must be a string")
        elif item_type in {"function_call", "function_call_output"}:
            _require_nonempty_string(item.get("call_id"), f"{label}.call_id")
            if item_type == "function_call":
                _require_nonempty_string(item.get("name"), f"{label}.name")
                if not isinstance(item.get("arguments"), str):
                    raise RealtimeClientError(f"{label}.arguments must be a string")
            elif not isinstance(item.get("output"), str):
                raise RealtimeClientError(f"{label}.output must be a string")
        elif item_type == "mcp_call":
            for field_name in ("name", "server_label"):
                _require_nonempty_string(item.get(field_name), f"{label}.{field_name}")
            if not isinstance(item.get("arguments"), str):
                raise RealtimeClientError(f"{label}.arguments must be a string")
            approval_request_id = item.get("approval_request_id")
            if approval_request_id is not None:
                _require_nonempty_string(approval_request_id, f"{label}.approval_request_id")
            if item.get("output") is not None and not isinstance(item.get("output"), str):
                raise RealtimeClientError(f"{label}.output must be a string or null")
            if item.get("error") is not None and not isinstance(item.get("error"), dict):
                raise RealtimeClientError(f"{label}.error must be an object or null")
        elif item_type == "mcp_list_tools":
            _require_nonempty_string(item.get("server_label"), f"{label}.server_label")
            tools = item.get("tools")
            if not isinstance(tools, list):
                raise RealtimeClientError(f"{label}.tools must be an array")
            names: set[str] = set()
            for tool_index, raw_tool in enumerate(tools):
                tool_label = f"{label}.tools[{tool_index}]"
                tool = _require_object(raw_tool, tool_label)
                name = _require_nonempty_string(tool.get("name"), f"{tool_label}.name")
                if name in names:
                    raise RealtimeClientError(f"{label}.tools repeats MCP tool name {name!r}")
                names.add(name)
                if not isinstance(tool.get("input_schema"), dict):
                    raise RealtimeClientError(f"{tool_label}.input_schema must be an object")
                if tool.get("description") is not None and not isinstance(tool.get("description"), str):
                    raise RealtimeClientError(f"{tool_label}.description must be a string or null")
                if tool.get("annotations") is not None and not isinstance(tool.get("annotations"), dict):
                    raise RealtimeClientError(f"{tool_label}.annotations must be an object or null")
        elif item_type == "mcp_approval_request":
            for field_name in ("server_label", "name"):
                _require_nonempty_string(item.get(field_name), f"{label}.{field_name}")
            if not isinstance(item.get("arguments"), str) or not item["arguments"]:
                raise RealtimeClientError(f"{label}.arguments must be a non-empty string")
        elif item_type == "mcp_approval_response":
            _require_nonempty_string(item.get("approval_request_id"), f"{label}.approval_request_id")
            if not isinstance(item.get("approve"), bool):
                raise RealtimeClientError(f"{label}.approve must be a boolean")
            if item.get("reason") is not None and not isinstance(item.get("reason"), str):
                raise RealtimeClientError(f"{label}.reason must be a string or null")
        return item

    @staticmethod
    def _assert_item_identity_matches(current: dict[str, Any], incoming: dict[str, Any], label: str) -> None:
        for field_name in (
            "id",
            "object",
            "type",
            "role",
            "call_id",
            "name",
            "server_label",
            "approve",
            "reason",
        ):
            if current.get(field_name) != incoming.get(field_name):
                raise RealtimeClientError(f"{label}.{field_name} changed during the item lifecycle")
        item_type = current.get("type")
        if item_type == "mcp_approval_response" and (
            current.get("approval_request_id") != incoming.get("approval_request_id")
        ):
            raise RealtimeClientError(f"{label}.approval_request_id changed during the item lifecycle")
        if item_type == "mcp_approval_request" and current.get("arguments") != incoming.get("arguments"):
            raise RealtimeClientError(f"{label}.arguments changed during the item lifecycle")

    def _capture_item_event(
        self,
        event: dict[str, Any],
        response_ids: list[str],
        turn_tool_call_ids: set[str],
        turn_input: _TurnInputTrace,
    ) -> None:
        event_type = event["type"]
        item = self._validate_conversation_item_snapshot(event.get("item"), f"{event_type}.item")
        item_id = item["id"]
        item_type = item["type"]
        if item_type == "message" and item.get("role") == "user" and turn_input.submitted_text is not None:
            expected_content = [{"type": "input_text", "text": turn_input.submitted_text}]
            if item_id != turn_input.item_id or item.get("content") != expected_content:
                raise RealtimeClientError("Server changed the client-submitted text item")

        if event_type == "conversation.item.added":
            if item_id in self._conversation_items:
                raise RealtimeClientError(f"conversation.item.added reused item ID {item_id!r}")
            if "previous_item_id" not in event:
                raise RealtimeClientError("conversation.item.added.previous_item_id is required")
            previous_item_id = event.get("previous_item_id")
            if previous_item_id is not None and (
                not isinstance(previous_item_id, str) or previous_item_id not in self._conversation_items
            ):
                raise RealtimeClientError("conversation.item.added.previous_item_id references an unknown item")
            if previous_item_id != self._conversation_tail_id:
                raise RealtimeClientError(
                    f"conversation.item.added predecessor {previous_item_id!r} does not match tail "
                    f"{self._conversation_tail_id!r}"
                )
            client_text_item = bool(
                item_type == "message"
                and item.get("role") == "user"
                and turn_input.submitted_text is not None
                and turn_input.item_id == item_id
            )
            atomically_added_item = (
                client_text_item
                or item_type == "function_call_output"
                or item_type in {"mcp_approval_request", "mcp_approval_response"}
                or (item_type == "message" and item.get("role") == "system")
            )
            if atomically_added_item:
                if item.get("status") not in {"completed", "incomplete"} and not (
                    item_type in {"mcp_approval_request", "mcp_approval_response"} and item.get("status") is None
                ):
                    raise RealtimeClientError("atomic conversation.item.added item must be terminal")
            elif item.get("status") != "in_progress" and not (
                item_type in {"mcp_call", "mcp_list_tools"} and item.get("status") is None
            ):
                raise RealtimeClientError("streamed conversation.item.added item must be in_progress")
            if item_type == "message" and item.get("role") == "user" and turn_input.item_id != item_id:
                raise RealtimeClientError(
                    f"conversation user item {item_id!r} does not match this turn's committed item"
                )
            self._conversation_items[item_id] = _ConversationItemTrace(
                item_id=item_id,
                item_type=item_type,
                added=copy.deepcopy(item),
            )
            self._conversation_tail_id = item_id
        elif event_type == "conversation.item.done":
            known = self._conversation_items.get(item_id)
            if known is None:
                raise RealtimeClientError(f"conversation.item.done references unknown item {item_id!r}")
            if known.done is not None:
                raise RealtimeClientError(f"conversation.item.done repeated terminal item {item_id!r}")
            self._assert_item_identity_matches(known.added, item, "conversation.item.done.item")
            if item_type == "function_call_output" and known.added.get("output") != item.get("output"):
                raise RealtimeClientError("conversation.item.done changed function_call_output.output")
            if item.get("status") not in {"completed", "incomplete"} and not (
                item_type
                in {
                    "mcp_call",
                    "mcp_list_tools",
                    "mcp_approval_request",
                    "mcp_approval_response",
                }
                and item.get("status") is None
            ):
                raise RealtimeClientError("conversation.item.done item must be terminal")
            if known.added.get("status") in {"completed", "incomplete"} and (
                known.added.get("status") != item.get("status")
            ):
                raise RealtimeClientError("conversation.item.done changed an atomically added terminal status")
            known.done = copy.deepcopy(item)
        elif event_type in {"response.output_item.added", "response.output_item.done"}:
            trace = self._response_trace_for_event(
                event,
                allow_terminal_mcp=event_type == "response.output_item.done",
            )
            output_index = _require_nonnegative_int(event.get("output_index"), f"{event_type}.output_index")
            known_item = self._conversation_items.get(item_id)
            if known_item is not None:
                self._assert_item_identity_matches(known_item.added, item, f"{event_type}.item")
            if event_type == "response.output_item.added":
                if output_index in trace.output_items or item_id in trace.item_indexes:
                    raise RealtimeClientError(
                        f"response.output_item.added repeated output {output_index} or item {item_id!r}"
                    )
                if item.get("status") != "in_progress" and not (
                    item_type in {"mcp_call", "mcp_list_tools"} and item.get("status") is None
                ):
                    raise RealtimeClientError("response.output_item.added item must be in_progress")
                trace.output_items[output_index] = _OutputItemTrace(
                    output_index=output_index,
                    item_id=item_id,
                    item_type=item_type,
                    added=copy.deepcopy(item),
                    mcp_status=(_MCPCallPhase.ANNOUNCED if item_type == "mcp_call" else None),
                )
                trace.item_indexes[item_id] = output_index
                if item_type == "mcp_call":
                    trace.has_mcp_call = True
            else:
                output = trace.output_items.get(output_index)
                if output is None or output.item_id != item_id:
                    raise RealtimeClientError("response.output_item.done does not match an announced output item")
                if output.output_done:
                    raise RealtimeClientError(f"response.output_item.done repeated item {item_id!r}")
                self._assert_item_identity_matches(output.added, item, "response.output_item.done.item")
                if known_item is None or known_item.done is None or known_item.done != item:
                    raise RealtimeClientError(
                        "response.output_item.done snapshot must match the terminal conversation item"
                    )
                if any(not content.part_done for content in output.content_parts.values()):
                    raise RealtimeClientError("response.output_item.done arrived before all content parts were done")
                if output.item_type == "function_call" and not output.function_arguments_done:
                    raise RealtimeClientError("response.output_item.done arrived before function arguments were done")
                if output.item_type == "function_call" and item.get("arguments") != output.function_arguments:
                    raise RealtimeClientError(
                        "response.output_item.done arguments do not match function argument events"
                    )
                if output.item_type == "mcp_call" and output.mcp_status not in {
                    _MCPCallPhase.COMPLETED,
                    _MCPCallPhase.FAILED,
                }:
                    raise RealtimeClientError("response.output_item.done arrived before the MCP item was terminal")
                if trace.status != "in_progress" and output.item_type != "mcp_call":
                    raise RealtimeClientError("Only an MCP call may finish its output item after response.done")
                if output.item_type == "message":
                    snapshot_content = item.get("content")
                    if (
                        not isinstance(snapshot_content, list)
                        or sorted(output.content_parts) != list(range(len(output.content_parts)))
                        or len(snapshot_content) != len(output.content_parts)
                    ):
                        raise RealtimeClientError(
                            "response.output_item.done content does not match announced content parts"
                        )
                    for content_index, content in output.content_parts.items():
                        part = snapshot_content[content_index]
                        expected_type = "output_audio" if content.part_type == "audio" else "output_text"
                        text_field = "transcript" if content.part_type == "audio" else "text"
                        if part.get("type") != expected_type or part.get(text_field) != content.terminal_text:
                            raise RealtimeClientError(
                                f"response.output_item.done content index {content_index} changed terminal output"
                            )
                output.output_done = True
        else:
            raise RealtimeClientError(f"Unsupported item lifecycle event {event_type!r}")

        if item_type == "function_call_output":
            call_id = item["call_id"]
            call = self._tool_calls.get(call_id)
            if call is None or call_id not in turn_tool_call_ids:
                raise RealtimeClientError(f"Function output {call_id!r} has no active function call")
            response_trace = self._response_traces.get(str(call.get("response_id")))
            if response_trace is None or response_trace.status == "in_progress":
                raise RealtimeClientError(f"Function output {call_id!r} arrived before Response A was terminal")
            if call.get("output_item_id") not in {None, item_id}:
                raise RealtimeClientError(f"Function call {call_id!r} produced more than one output item")
            owner = call.get("owner")
            if owner == "client":
                expected_item_id = call.get("expected_output_item_id")
                expected_output = call.get("expected_output")
                output_mismatch = expected_item_id != item_id or expected_output != item["output"]
                server_timeout = self._client_tool_timeout_error(item["output"])
                if output_mismatch and server_timeout is not None:
                    if call.get("server_generated_output") and (
                        expected_item_id != item_id or expected_output != item["output"]
                    ):
                        raise RealtimeClientError(f"Client tool timeout {call_id!r} changed its terminal output")
                    call["server_generated_output"] = True
                    call["expected_output_item_id"] = item_id
                    call["expected_output"] = item["output"]
                    call["handler_status"] = "failed"
                    call["handler_error"] = copy.deepcopy(server_timeout)
                elif expected_item_id is None or expected_output is None:
                    raise RealtimeClientError(f"Client tool output {call_id!r} arrived before the client sent it")
                elif output_mismatch:
                    if item_id != expected_item_id:
                        raise RealtimeClientError(
                            f"Client tool output {call_id!r} changed item ID from {expected_item_id!r} to {item_id!r}"
                        )
                    raise RealtimeClientError(f"Client tool output {call_id!r} changed its scripted payload")
            elif owner != "backend":
                raise RealtimeClientError(f"Function output {call_id!r} has no established tool owner")
            turn_tool_call_ids.add(call_id)
            call["output"] = item["output"]
            if event_type == "conversation.item.done":
                call["output_item_id"] = item_id
                call["output_status"] = item.get("status")
            return
        if item_type == "mcp_approval_request":
            if event_type == "conversation.item.added":
                if item_id in self._mcp_approval_requests:
                    raise RealtimeClientError(f"MCP approval request {item_id!r} was repeated")
                self._mcp_approval_requests[item_id] = copy.deepcopy(item)
            return
        if item_type == "mcp_approval_response":
            approval = self._mcp_approval_response_items.get(item_id)
            if approval is None or approval.item != item:
                raise RealtimeClientError(f"MCP approval response {item_id!r} was not sent by this client")
            if event_type == "conversation.item.done":
                if approval.item_done:
                    raise RealtimeClientError(f"MCP approval response {item_id!r} repeated conversation.item.done")
                self._correlate(
                    approval.event_id,
                    "conversation.item.done",
                    resource_id=item_id,
                )
                approval.item_done = True
            return
        if item_type != "function_call":
            return
        response_id = event.get("response_id") or (response_ids[-1] if response_ids else None)
        if not isinstance(response_id, str) or response_id != self._active_response_id:
            raise RealtimeClientError(f"{event_type} function call is not correlated to the active response")
        call_id = item["call_id"]
        trace = self._response_traces[response_id]
        trace.has_function_call = True
        turn_tool_call_ids.add(call_id)
        call = self._tool_call(call_id, item, response_id)
        call["name"] = item["name"]
        if item["arguments"]:
            call["arguments"] = item["arguments"]
        call["item_id"] = item_id
        if event_type.endswith(".done") or item.get("status") == "completed":
            call["status"] = "completed"

    def _capture_content_part_event(self, event: dict[str, Any]) -> None:
        event_type = event["type"]
        _trace, output, content = self._response_output_for_event(
            event,
            require_content=event_type == "response.content_part.done",
        )
        part = _require_object(event.get("part"), f"{event_type}.part")
        part_type = _require_nonempty_string(part.get("type"), f"{event_type}.part.type")
        if part_type not in {"audio", "text"}:
            raise RealtimeClientError(f"{event_type}.part.type must be audio or text")
        expected_part_type = self.output_modality
        if part_type != expected_part_type:
            raise RealtimeClientError(
                f"{event_type} part type {part_type!r} does not match session modality {self.output_modality!r}"
            )
        if output.item_type != "message":
            raise RealtimeClientError(f"{event_type} references non-message item {output.item_id!r}")
        known_item = self._conversation_items.get(output.item_id)
        if known_item is None:
            raise RealtimeClientError(f"{event_type} arrived before conversation.item.added")
        content_index = _require_nonnegative_int(event.get("content_index"), f"{event_type}.content_index")
        if event_type == "response.content_part.added":
            if content_index in output.content_parts:
                raise RealtimeClientError(f"response.content_part.added repeated content index {content_index}")
            if part_type == "audio" and not isinstance(part.get("transcript"), str):
                raise RealtimeClientError("response.content_part.added audio transcript must be a string")
            if part_type == "text" and not isinstance(part.get("text"), str):
                raise RealtimeClientError("response.content_part.added text must be a string")
            output.content_parts[content_index] = _ContentPartTrace(content_index, part_type)
            return

        assert content is not None
        if content.part_type != part_type:
            raise RealtimeClientError("response.content_part.done changed the content part type")
        if part_type == "audio":
            if not content.audio_done or not content.transcript_done:
                raise RealtimeClientError(
                    "response.content_part.done arrived before audio and transcript terminal events"
                )
            if not isinstance(part.get("transcript"), str):
                raise RealtimeClientError("response.content_part.done audio transcript must be a string")
            if content.terminal_text is not None and part["transcript"] != content.terminal_text:
                raise RealtimeClientError("response.content_part.done transcript changed the terminal transcript")
        else:
            if not content.text_done:
                raise RealtimeClientError("response.content_part.done arrived before response.output_text.done")
            if not isinstance(part.get("text"), str):
                raise RealtimeClientError("response.content_part.done text must be a string")
            if content.terminal_text is not None and part["text"] != content.terminal_text:
                raise RealtimeClientError("response.content_part.done text changed the terminal output text")
        content.part_done = True

    def _capture_response_stream_event(self, event: dict[str, Any]) -> None:
        event_type = event["type"]
        _trace, output, content = self._response_output_for_event(event, require_content=True)
        assert content is not None
        if event_type.startswith("response.output_audio") and content.part_type != "audio":
            raise RealtimeClientError(f"{event_type} references a non-audio content part")
        if event_type.startswith("response.output_text") and content.part_type != "text":
            raise RealtimeClientError(f"{event_type} references a non-text content part")
        if event_type == "response.output_audio.done":
            if content.audio_done:
                raise RealtimeClientError("response.output_audio.done was emitted more than once")
            content.audio_done = True
        elif event_type == "response.output_audio_transcript.done":
            if content.transcript_done:
                raise RealtimeClientError("response.output_audio_transcript.done was emitted more than once")
            transcript = event.get("transcript")
            if not isinstance(transcript, str):
                raise RealtimeClientError("response.output_audio_transcript.done.transcript must be a string")
            if content.streamed_text and content.streamed_text != transcript:
                raise RealtimeClientError(
                    "response.output_audio_transcript.done.transcript does not match accumulated deltas"
                )
            content.terminal_text = transcript
            content.transcript_done = True
        elif event_type == "response.output_text.done":
            if content.text_done:
                raise RealtimeClientError("response.output_text.done was emitted more than once")
            text = event.get("text")
            if not isinstance(text, str):
                raise RealtimeClientError("response.output_text.done.text must be a string")
            if content.streamed_text and content.streamed_text != text:
                raise RealtimeClientError("response.output_text.done.text does not match accumulated deltas")
            content.terminal_text = text
            content.text_done = True
        elif event_type in {
            "response.output_audio.delta",
            "response.output_audio_transcript.delta",
            "response.output_text.delta",
        }:
            if output.output_done:
                raise RealtimeClientError(f"{event_type} arrived after response.output_item.done")
            if event_type != "response.output_audio.delta":
                delta = event.get("delta")
                if not isinstance(delta, str):
                    raise RealtimeClientError(f"{event_type}.delta must be a string")
                content.streamed_text += delta
        else:
            raise RealtimeClientError(f"Unsupported response stream event {event_type!r}")

    def _capture_function_arguments_event(self, event: dict[str, Any]) -> tuple[_ResponseTrace, str]:
        event_type = event["type"]
        trace, output, _content = self._response_output_for_event(event)
        if output.item_type != "function_call":
            raise RealtimeClientError(f"{event_type} references non-function item {output.item_id!r}")
        call_id = _require_nonempty_string(event.get("call_id"), f"{event_type}.call_id")
        item_call_id = _require_nonempty_string(output.added.get("call_id"), "function_call.call_id")
        if call_id != item_call_id:
            raise RealtimeClientError(f"{event_type}.call_id does not match the output item")
        if output.function_arguments_done:
            raise RealtimeClientError(f"{event_type} arrived after function arguments were done")
        if event_type.endswith(".delta"):
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise RealtimeClientError(f"{event_type}.delta must be a string")
            output.function_arguments += delta
        else:
            arguments = event.get("arguments")
            if not isinstance(arguments, str):
                raise RealtimeClientError(f"{event_type}.arguments must be a string")
            name = _require_nonempty_string(event.get("name"), f"{event_type}.name")
            if name != output.added.get("name"):
                raise RealtimeClientError(f"{event_type}.name does not match the output item")
            if output.function_arguments and output.function_arguments != arguments:
                raise RealtimeClientError(f"{event_type}.arguments does not match accumulated deltas")
            output.function_arguments = arguments
            output.function_arguments_done = True
        return trace, call_id

    def _capture_mcp_event(self, event: dict[str, Any]) -> None:
        event_type = event["type"]
        item_id = _require_nonempty_string(event.get("item_id"), f"{event_type}.item_id")
        output_index = _require_nonnegative_int(event.get("output_index"), f"{event_type}.output_index")
        if "response_id" in event:
            _trace, output, _content = self._response_output_for_event(
                event,
                allow_terminal_mcp=True,
            )
        else:
            matches = [
                trace.output_items[output_index]
                for trace in self._response_traces.values()
                if output_index in trace.output_items
                and trace.output_items[output_index].item_id == item_id
                and not trace.output_items[output_index].output_done
            ]
            if len(matches) != 1:
                raise RealtimeClientError(f"{event_type} does not identify exactly one unfinished MCP output item")
            output = matches[0]
        if output.item_type != "mcp_call":
            raise RealtimeClientError(f"{event_type} references non-MCP item {item_id!r}")
        if event_type.endswith("arguments.delta"):
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise RealtimeClientError(f"{event_type}.delta must be a string")
            if output.function_arguments_done:
                raise RealtimeClientError(f"{event_type} arrived after MCP arguments were done")
            output.function_arguments += delta
        elif event_type.endswith("arguments.done"):
            arguments = event.get("arguments")
            if not isinstance(arguments, str):
                raise RealtimeClientError(f"{event_type}.arguments must be a string")
            if output.function_arguments_done:
                raise RealtimeClientError(f"{event_type} repeated MCP arguments.done")
            if output.function_arguments and output.function_arguments != arguments:
                raise RealtimeClientError(f"{event_type}.arguments does not match accumulated deltas")
            output.function_arguments = arguments
            output.function_arguments_done = True
        elif event_type.endswith(".in_progress"):
            if output.mcp_status is not _MCPCallPhase.ANNOUNCED:
                raise RealtimeClientError(f"{event_type} repeated the MCP lifecycle start")
            if not output.function_arguments_done:
                raise RealtimeClientError(f"{event_type} arrived before MCP arguments.done")
            output.mcp_status = _MCPCallPhase.IN_PROGRESS
        elif event_type.endswith(".completed"):
            if output.mcp_status is not _MCPCallPhase.IN_PROGRESS:
                raise RealtimeClientError(f"{event_type} did not follow MCP in_progress")
            output.mcp_status = _MCPCallPhase.COMPLETED
        elif event_type.endswith(".failed"):
            if output.mcp_status not in {_MCPCallPhase.ANNOUNCED, _MCPCallPhase.IN_PROGRESS}:
                raise RealtimeClientError(f"{event_type} followed an invalid MCP phase")
            if not output.function_arguments_done:
                raise RealtimeClientError(f"{event_type} arrived before MCP arguments.done")
            output.mcp_status = _MCPCallPhase.FAILED
        else:
            raise RealtimeClientError(f"Unsupported MCP event {event_type!r}")

    def _capture_input_lifecycle_event(self, event: dict[str, Any], turn_input: _TurnInputTrace) -> None:
        event_type = event["type"]
        if event_type == "input_audio_buffer.speech_started":
            item_id = _require_nonempty_string(event.get("item_id"), f"{event_type}.item_id")
            audio_start_ms = _require_nonnegative_int(event.get("audio_start_ms"), f"{event_type}.audio_start_ms")
            if turn_input.speech_item_id is not None:
                raise RealtimeClientError("Received more than one speech_started event for a benchmark turn")
            turn_input.speech_item_id = item_id
            turn_input.speech_start_ms = audio_start_ms
            return
        if event_type in {"input_audio_buffer.speech_stopped", "input_audio_buffer.timeout_triggered"}:
            item_id = _require_nonempty_string(event.get("item_id"), f"{event_type}.item_id")
            audio_end_ms = _require_nonnegative_int(event.get("audio_end_ms"), f"{event_type}.audio_end_ms")
            if turn_input.speech_item_id != item_id or turn_input.speech_start_ms is None:
                raise RealtimeClientError(f"{event_type} does not match speech_started for this turn")
            if audio_end_ms < turn_input.speech_start_ms:
                raise RealtimeClientError(f"{event_type}.audio_end_ms precedes speech_started.audio_start_ms")
            if event_type == "input_audio_buffer.timeout_triggered":
                timeout_start = _require_nonnegative_int(
                    event.get("audio_start_ms"),
                    "input_audio_buffer.timeout_triggered.audio_start_ms",
                )
                if timeout_start != turn_input.speech_start_ms:
                    raise RealtimeClientError("input_audio_buffer.timeout_triggered changed audio_start_ms")
            if turn_input.speech_stopped:
                raise RealtimeClientError(f"{event_type} repeated the terminal speech boundary")
            turn_input.speech_stopped = True
            return
        if event_type == "input_audio_buffer.committed":
            item_id = _require_nonempty_string(event.get("item_id"), f"{event_type}.item_id")
            if turn_input.item_id is not None:
                raise RealtimeClientError("Received more than one input_audio_buffer.committed for a turn")
            if "previous_item_id" not in event or event.get("previous_item_id") != self._conversation_tail_id:
                raise RealtimeClientError(
                    "input_audio_buffer.committed.previous_item_id must match the conversation tail"
                )
            if self.turn_mode == "automatic" and (
                not turn_input.speech_stopped or turn_input.speech_item_id != item_id
            ):
                raise RealtimeClientError(
                    "input_audio_buffer.committed must follow speech_started/stopped for the same item"
                )
            turn_input.item_id = item_id
            return
        if event_type == "input_audio_buffer.cleared":
            raise RealtimeClientError("Received unsolicited input_audio_buffer.cleared")
        raise RealtimeClientError(f"Unsupported input audio lifecycle event {event_type!r}")

    def _capture_conversation_passive_event(self, event: dict[str, Any]) -> None:
        event_type = event["type"]
        if event_type.startswith("conversation.item.input_audio_transcription."):
            item_id = _require_nonempty_string(event.get("item_id"), f"{event_type}.item_id")
            content_index = _require_nonnegative_int(event.get("content_index"), f"{event_type}.content_index")
            item = self._conversation_items.get(item_id)
            if item is None or item.item_type != "message" or item.added.get("role") != "user":
                raise RealtimeClientError(f"{event_type} references unknown non-user item {item_id!r}")
            snapshot = item.done or item.added
            content = snapshot.get("content")
            if not isinstance(content, list) or content_index >= len(content):
                raise RealtimeClientError(f"{event_type}.content_index is outside the user item content")
            if event_type.endswith(".delta"):
                if not isinstance(event.get("delta"), str):
                    raise RealtimeClientError(f"{event_type}.delta must be a string")
            elif event_type.endswith(".completed"):
                if not isinstance(event.get("transcript"), str):
                    raise RealtimeClientError(f"{event_type}.transcript must be a string")
                if not isinstance(event.get("usage"), dict):
                    raise RealtimeClientError(f"{event_type}.usage must be an object")
            elif not isinstance(event.get("error"), dict):
                raise RealtimeClientError(f"{event_type}.error must be an object")
            return
        if event_type == "conversation.item.retrieved":
            item = self._validate_conversation_item_snapshot(event.get("item"), f"{event_type}.item")
            known = self._conversation_items.get(item["id"])
            if known is None or item != (known.done or known.added):
                raise RealtimeClientError("conversation.item.retrieved returned an unknown or divergent snapshot")
            return
        item_id = _require_nonempty_string(event.get("item_id"), f"{event_type}.item_id")
        known = self._conversation_items.get(item_id)
        if known is None:
            raise RealtimeClientError(f"{event_type} references unknown item {item_id!r}")
        if event_type == "conversation.item.truncated":
            content_index = _require_nonnegative_int(event.get("content_index"), f"{event_type}.content_index")
            _require_nonnegative_int(event.get("audio_end_ms"), f"{event_type}.audio_end_ms")
            snapshot = known.done or known.added
            if not isinstance(snapshot.get("content"), list) or content_index >= len(snapshot["content"]):
                raise RealtimeClientError("conversation.item.truncated.content_index is outside the item content")
            return
        if event_type == "conversation.item.deleted":
            if known.deleted:
                raise RealtimeClientError(f"conversation.item.deleted repeated item {item_id!r}")
            known.deleted = True
            return
        raise RealtimeClientError(f"Unsupported conversation lifecycle event {event_type!r}")

    def _tool_call(self, call_id: str, source: dict[str, Any], response_id: str) -> dict[str, Any]:
        call = self._tool_calls.setdefault(
            call_id,
            {
                "call_id": call_id,
                "response_id": response_id,
                "item_id": source.get("item_id"),
                "name": source.get("name"),
                "arguments": "",
                "status": "in_progress",
                "output": None,
                "output_item_id": None,
                "output_status": None,
                "error": None,
                "owner": None,
                "tool_round": None,
                "handler_invocation": None,
                "handler_status": None,
                "handler_latency": None,
                "handler_error": None,
                "parsed_arguments": None,
                "expected_output": None,
                "expected_output_item_id": None,
                "output_event_id": None,
                "server_generated_output": False,
                "late_output_error": None,
            },
        )
        if call["response_id"] != response_id:
            raise RealtimeClientError(f"Function call {call_id!r} changed response correlation")
        return call

    def _capture_tool_failure_error(
        self,
        event: dict[str, Any],
        turn_tool_call_ids: set[str],
    ) -> bool:
        """Record a correlated tool failure or rejected late client output."""
        error = _require_object(event.get("error"), "error.error")
        client_event_id = error.get("event_id")
        if error.get("code") in _LATE_CLIENT_TOOL_OUTPUT_ERROR_CODES and isinstance(client_event_id, str):
            for call_id in turn_tool_call_ids:
                call = self._tool_calls[call_id]
                if call.get("output_event_id") != client_event_id or not call.get("server_generated_output"):
                    continue
                handler_error = call.get("handler_error")
                if (
                    not isinstance(handler_error, dict)
                    or handler_error.get("code") != "client_tool_timeout"
                    or not isinstance(call.get("output_item_id"), str)
                ):
                    continue
                correlation = self._pending_events.pop(client_event_id, None)
                if (
                    correlation is None
                    or correlation.get("type") != "conversation.item.create"
                    or correlation.get("ack") != "conversation.item.done"
                ):
                    raise RealtimeClientError(
                        f"Rejected late client tool output {call_id!r} has no pending output event correlation"
                    )
                self.event_correlations.append(
                    {
                        "client_event_id": client_event_id,
                        "client_event_type": correlation["type"],
                        "server_event_type": "error",
                        "resource_id": call.get("output_item_id"),
                        "latency": time.time() - correlation["sent_at"],
                        "origin": "client",
                        "error_code": error["code"],
                    }
                )
                call["late_output_error"] = copy.deepcopy(error)
                return True
        metadata = error.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("kind") != "tool_failure":
            return False
        call_id = _require_nonempty_string(metadata.get("call_id"), "error.metadata.call_id")
        call = self._tool_calls.get(call_id)
        if call is None or call_id not in turn_tool_call_ids:
            raise RealtimeClientError(f"Tool failure {call_id!r} has no active response correlation")
        call["status"] = "failed"
        call["error"] = copy.deepcopy(error)
        return True

    @staticmethod
    def _client_tool_timeout_error(output: str) -> dict[str, str] | None:
        """Recognize the gateway's terminal structured client-tool timeout."""
        try:
            value = json.loads(
                output,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_constant,
                parse_float=_parse_finite_json_float,
            )
        except (json.JSONDecodeError, ValueError, OverflowError):
            return None
        if not isinstance(value, dict) or value.get("ok") is not False:
            return None
        error = value.get("error")
        if not isinstance(error, dict) or error.get("code") != "client_tool_timeout":
            return None
        message = error.get("message")
        if not isinstance(message, str) or not message:
            return None
        return {"code": "client_tool_timeout", "message": message}

    def _start_response_graph(self, response: dict[str, Any], received_at: float) -> _ResponseTrace:
        response_id = _require_nonempty_string(response.get("id"), "response.id")
        if response.get("object") != "realtime.response":
            raise RealtimeClientError("response.created.response.object must be 'realtime.response'")
        if response.get("conversation_id") != self._conversation_id:
            raise RealtimeClientError("response.created.response.conversation_id does not match the session")
        if response.get("status") != "in_progress":
            raise RealtimeClientError("response.created status must be in_progress")
        if response.get("status_details") is not None or response.get("usage") is not None:
            raise RealtimeClientError("response.created status_details and usage must be null")
        if response.get("output") != []:
            raise RealtimeClientError("response.created output must be empty")
        if self._active_response_id is not None:
            raise RealtimeClientError(
                f"response.created started {response_id!r} while {self._active_response_id!r} is active"
            )
        if response_id in self._response_traces:
            raise RealtimeClientError(f"Duplicate response.created for {response_id!r}")
        trace = _ResponseTrace(response_id, received_at)
        self._response_traces[response_id] = trace
        self._active_response_id = response_id
        return trace

    def _validate_recovery_response_start(
        self,
        response_id: str,
        response_ids: list[str],
        turn_tool_call_ids: set[str],
    ) -> None:
        """Require the exact Response A -> tool output -> Response B boundary."""
        if not response_ids:
            return
        previous = self._response_traces[response_ids[-1]]
        if previous.status == "in_progress":
            raise RealtimeClientError(
                f"response.created started {response_id!r} before response {previous.response_id!r} was terminal"
            )
        if not previous.has_function_call and not previous.has_mcp_call:
            raise RealtimeClientError(
                f"response.created started {response_id!r} after a terminal response with no tool call"
            )
        previous_call_ids = self._response_call_ids(previous.response_id)
        pending_outputs = [
            call_id
            for call_id in previous_call_ids
            if call_id in turn_tool_call_ids and not self._tool_calls.get(call_id, {}).get("output_item_id")
        ]
        if pending_outputs:
            raise RealtimeClientError(
                f"response.created started {response_id!r} before correlated "
                f"function_call_output items for {sorted(pending_outputs)}"
            )
        pending_mcp_items = [
            output.item_id for output in self._response_mcp_outputs(previous.response_id) if not output.output_done
        ]
        if pending_mcp_items:
            raise RealtimeClientError(
                f"response.created started {response_id!r} before terminal MCP items {sorted(pending_mcp_items)}"
            )

    def _finish_response_graph(self, response: dict[str, Any]) -> _ResponseTrace:
        response_id = _require_nonempty_string(response.get("id"), "response.id")
        if response_id != self._active_response_id:
            raise RealtimeClientError(f"response.done references inactive response {response_id!r}")
        try:
            trace = self._response_traces[response_id]
        except KeyError as exc:
            raise RealtimeClientError(f"response.done references unknown response {response_id!r}") from exc
        if response.get("object") != "realtime.response":
            raise RealtimeClientError("response.done.response.object must be 'realtime.response'")
        if response.get("conversation_id") != self._conversation_id:
            raise RealtimeClientError("response.done.response.conversation_id changed")
        status = response.get("status")
        if status not in _TERMINAL_RESPONSE_STATUSES:
            raise RealtimeClientError(f"response.done has invalid terminal status {status!r}")
        status_details = response.get("status_details")
        if status == "completed" and status_details is not None:
            raise RealtimeClientError("completed response.done must have null status_details")
        if status != "completed" and not isinstance(status_details, dict):
            raise RealtimeClientError("non-completed response.done requires status_details")

        response_output = response.get("output")
        if not isinstance(response_output, list):
            raise RealtimeClientError("response.done.response.output must be an array")
        expected_indexes = list(range(len(trace.output_items)))
        if sorted(trace.output_items) != expected_indexes or len(response_output) != len(expected_indexes):
            raise RealtimeClientError("response.done output indexes are incomplete or non-contiguous")
        for output_index, item in enumerate(response_output):
            item = self._validate_conversation_item_snapshot(item, f"response.output[{output_index}]")
            output = trace.output_items[output_index]
            known = self._conversation_items.get(output.item_id)
            if item["id"] != output.item_id or item["type"] != output.item_type or known is None:
                raise RealtimeClientError(
                    f"response.done output index {output_index} does not match its conversation item snapshot"
                )
            self._assert_item_identity_matches(output.added, item, f"response.output[{output_index}]")
            if output.output_done:
                if known.done is None or item != known.done:
                    raise RealtimeClientError(
                        f"response.done output index {output_index} does not match its terminal item snapshot"
                    )
                continue
            if output.item_type != "mcp_call":
                raise RealtimeClientError(
                    f"response.done arrived before response.output_item.done for index {output_index}"
                )
            if not output.function_arguments_done or item.get("arguments") != output.function_arguments:
                raise RealtimeClientError(
                    f"response.done MCP output index {output_index} does not match its terminal arguments"
                )
        trace.status = status
        self._active_response_id = None
        return trace

    def _capture_done_output(
        self,
        response: dict[str, Any],
        trace: _ResponseTrace,
        turn_tool_call_ids: set[str],
    ) -> None:
        output = response.get("output", [])
        if not isinstance(output, list):
            raise RealtimeClientError("response.done.response.output must be an array")
        for item in output:
            if not isinstance(item, dict):
                raise RealtimeClientError("response.done output item must be an object")
            if item.get("type") == "function_call":
                call_id = _require_nonempty_string(item.get("call_id"), "response.output.call_id")
                name = _require_nonempty_string(item.get("name"), "response.output.name")
                arguments = item.get("arguments")
                if not isinstance(arguments, str):
                    raise RealtimeClientError("response.output.arguments must be a string")
                trace.has_function_call = True
                turn_tool_call_ids.add(call_id)
                call = self._tool_call(call_id, item, trace.response_id)
                call["name"] = name
                call["arguments"] = arguments
                call["item_id"] = item.get("id")
                call["status"] = item.get("status", "completed")
                continue
            if item.get("type") != "message":
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                raise RealtimeClientError("response message content must be an array")
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    trace.has_text = True
                    if not trace.transcript:
                        trace.transcript = part["text"]
                if part.get("type") == "output_audio" and isinstance(part.get("transcript"), str):
                    trace.has_text = True
                    if not trace.transcript:
                        trace.transcript = part["transcript"]

    def _capture_usage(self, usage: Any) -> None:
        if usage is None:
            return
        if not isinstance(usage, dict):
            raise RealtimeClientError("response.usage must be an object or null")
        for source_key, metric_key in (
            ("total_tokens", "response_total_tokens"),
            ("input_tokens", "response_input_tokens"),
            ("output_tokens", "response_output_tokens"),
        ):
            value = usage.get(source_key)
            if value is not None:
                numeric = _require_finite_number(
                    value,
                    f"response.usage.{source_key}",
                    nonnegative=True,
                )
                if self._collecting_metrics:
                    self.server_metric_samples[metric_key].append(numeric)

    def _capture_nvidia_metrics(self, event: dict[str, Any]) -> None:
        metrics = event.get("metrics", event.get("data"))
        if not isinstance(metrics, dict):
            raise RealtimeClientError("nvidia.metrics.updated must contain a metrics object")
        for key, value in metrics.items():
            if key not in self.server_metric_samples:
                continue
            numeric = _require_finite_number(
                value,
                f"nvidia.metrics.updated.{key}",
                nonnegative=True,
            )
            if self._collecting_metrics:
                self.server_metric_samples[key].append(numeric)

    def _correlate_oldest(self, ack: str, *, resource_id: str) -> None:
        for event_id, pending in self._pending_events.items():
            if pending["ack"] == ack:
                self._correlate(event_id, ack, resource_id=resource_id)
                return
        self.event_correlations.append(
            {
                "client_event_id": None,
                "client_event_type": None,
                "server_event_type": ack,
                "resource_id": resource_id,
                "origin": "server",
            }
        )

    def _correlate(self, event_id: str, ack: str, *, resource_id: str) -> None:
        pending = self._pending_events.pop(event_id, None)
        if pending is None:
            raise RealtimeClientError(f"No pending client event {event_id!r} for {ack}")
        if pending["ack"] != ack:
            raise RealtimeClientError(f"Client event {event_id!r} expected {pending['ack']!r}, received {ack!r}")
        self.event_correlations.append(
            {
                "client_event_id": event_id,
                "client_event_type": pending["type"],
                "server_event_type": ack,
                "resource_id": resource_id,
                "latency": time.time() - pending["sent_at"],
                "origin": "client",
            }
        )

    def _server_error(self, event: dict[str, Any]) -> RealtimeClientError:
        error = _require_object(event.get("error"), "error.error")
        client_event_id = error.get("event_id")
        correlation = self._pending_events.pop(client_event_id, None) if isinstance(client_event_id, str) else None
        if correlation is not None:
            self.event_correlations.append(
                {
                    "client_event_id": client_event_id,
                    "client_event_type": correlation["type"],
                    "server_event_type": "error",
                    "resource_id": None,
                    "latency": time.time() - correlation["sent_at"],
                    "origin": "client",
                    "error_code": error.get("code"),
                }
            )
        context = f" client_event={correlation['type']}:{client_event_id}" if correlation else ""
        return RealtimeClientError(
            f"Realtime server error type={error.get('type', '-')} code={error.get('code', '-')} "
            f"param={error.get('param', '-')} message={error.get('message', '-')}{context}"
        )

    def _read_wav_as_pcm_24khz(self, path: Path) -> bytes:
        try:
            with wave.open(str(path), "rb") as wav_file:
                if wav_file.getnchannels() != 1:
                    raise RealtimeClientError(f"{path.name}: expected mono WAV")
                if wav_file.getsampwidth() != 2:
                    raise RealtimeClientError(f"{path.name}: expected 16-bit PCM WAV")
                if wav_file.getcomptype() != "NONE":
                    raise RealtimeClientError(f"{path.name}: compressed WAV is not supported")
                source_rate = wav_file.getframerate()
                raw = wav_file.readframes(wav_file.getnframes())
        except wave.Error as exc:
            raise RealtimeClientError(f"{path.name}: invalid WAV file: {exc}") from exc
        if not raw:
            raise RealtimeClientError(f"{path.name}: WAV contains no audio")
        if source_rate == REALTIME_PCM_RATE:
            return raw
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        resampled = resampy.resample(samples, source_rate, REALTIME_PCM_RATE)
        return np.clip(np.rint(resampled * 32767.0), -32768, 32767).astype("<i2").tobytes()

    async def _stream_pcm(
        self,
        websocket,
        pcm: bytes,
        *,
        stop_event: asyncio.Event | None = None,
    ) -> float:
        chunk_bytes = REALTIME_PCM_RATE * 2 * REALTIME_CHUNK_MS // 1000
        started = time.monotonic()
        sent_samples = 0
        for offset in range(0, len(pcm), chunk_bytes):
            if self.shutdown_requested():
                raise RealtimeClientStopped
            if stop_event is not None and stop_event.is_set():
                break
            chunk = pcm[offset : offset + chunk_bytes]
            await self._send_audio_chunk(websocket, chunk)
            sent_samples += len(chunk) // 2
            target = started + sent_samples / REALTIME_PCM_RATE
            delay = max(target - time.monotonic(), 0.0)
            if stop_event is None:
                await asyncio.sleep(delay)
                continue
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=delay)
        return time.time()

    def _write_output_audio(self, audio: bytes) -> None:
        if self.audio_output_path is None:
            return
        if self._audio_writer is None:
            self.audio_output_path.parent.mkdir(parents=True, exist_ok=True)
            self._audio_writer = wave.open(str(self.audio_output_path), "wb")  # noqa: SIM115
            self._audio_writer.setnchannels(1)
            self._audio_writer.setsampwidth(2)
            self._audio_writer.setframerate(REALTIME_PCM_RATE)
        self._audio_writer.writeframes(audio)
