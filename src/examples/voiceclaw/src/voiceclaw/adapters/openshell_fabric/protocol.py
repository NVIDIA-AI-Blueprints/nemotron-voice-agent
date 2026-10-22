# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Strict Fabric 0.5 runtime-server request and response codec."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

MAX_FABRIC_REQUEST_BYTES = 512 * 1024
MAX_FABRIC_RESPONSE_BYTES = 4 * 1024 * 1024
FABRIC_ADAPTER_KINDS = frozenset({"http", "native_plugin", "process", "python"})
_OUTER_FIELDS = frozenset({"operation", "status", "changed", "result", "error"})
_ERROR_FIELDS = frozenset({"code", "stage", "message", "effects"})
_RUN_REQUIRED_FIELDS = frozenset(
    {
        "agent_name",
        "harness",
        "adapter_kind",
        "adapter_id",
        "runtime_id",
        "invocation_id",
        "request_id",
        "status",
        "output",
    }
)
_RUN_ERROR_REQUIRED_FIELDS = frozenset({"stage", "code", "message"})


class FabricProtocolError(ValueError):
    """The Fabric runtime server emitted malformed or contradictory data."""


class FabricStatus(StrEnum):
    """Terminal statuses emitted by the runtime-server command envelope."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class FabricRunStatus(StrEnum):
    """Terminal statuses emitted by a Fabric invocation."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FabricEffects(StrEnum):
    """Whether a failed runtime-server command may have affected its target."""

    NONE = "none"
    APPLIED = "applied"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FabricFailure:
    """Validated command failure evidence safe to expose beyond the adapter."""

    code: str
    effects: FabricEffects


@dataclass(frozen=True, slots=True)
class FabricResponse:
    """One validated five-field runtime-server response envelope."""

    operation: str
    status: FabricStatus
    changed: bool | None
    result: dict[str, Any] | None
    error: FabricFailure | None


@dataclass(frozen=True, slots=True)
class FabricRunResult:
    """Validated identity and terminal state from one direct Fabric RunResult."""

    agent_name: str
    harness: str
    adapter_kind: str
    adapter_id: str
    runtime_id: str
    invocation_id: str
    request_id: str
    status: FabricRunStatus
    value: dict[str, Any]


def encode_run_request(
    *,
    input_text: str,
    request_id: str,
    relay_session_root: str,
    context: dict[str, Any],
) -> bytes:
    """Encode one complete public Fabric RunRequest for ``--request -``."""
    _required_string(input_text, "Fabric invocation input")
    _required_identity(request_id, "Fabric request ID")
    _required_identity(relay_session_root, "Fabric relay session root")
    if not isinstance(context, dict):
        raise FabricProtocolError("Fabric request context must be an object")
    return _encode_json_request(
        {
            "input": input_text,
            "request_id": request_id,
            "relay_session_root": relay_session_root,
            "context": context,
        }
    )


def decode_response(*, operation: str, exit_code: int, stdout: str) -> FabricResponse:
    """Decode one newline-terminated runtime-server response without fallback."""
    if operation != "invoke":
        raise FabricProtocolError("unsupported Fabric operation")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise FabricProtocolError("invalid Fabric exit code")
    if not isinstance(stdout, str):
        raise FabricProtocolError("invalid Fabric output")
    try:
        encoded = stdout.encode("utf-8")
    except UnicodeEncodeError as error:
        raise FabricProtocolError("invalid Fabric output") from error
    if not encoded or len(encoded) > MAX_FABRIC_RESPONSE_BYTES or not encoded.endswith(b"\n"):
        raise FabricProtocolError("invalid Fabric output framing")
    try:
        value = json.loads(
            stdout,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
            parse_int=_parse_integer,
        )
    except (ValueError, RecursionError) as error:
        raise FabricProtocolError("invalid Fabric JSON") from error
    _validate_json_strings(value)
    if not isinstance(value, dict) or set(value) != _OUTER_FIELDS:
        raise FabricProtocolError("invalid Fabric response fields")
    if value["operation"] != operation:
        raise FabricProtocolError("Fabric operation mismatch")
    try:
        status = FabricStatus(value["status"])
    except (TypeError, ValueError) as error:
        raise FabricProtocolError("invalid Fabric status") from error
    changed = value["changed"]
    if changed is not None and not isinstance(changed, bool):
        raise FabricProtocolError("invalid Fabric changed value")
    result = value["result"]
    if result is not None and not isinstance(result, dict):
        raise FabricProtocolError("invalid Fabric result")
    failure = _decode_failure(value["error"])
    if exit_code == 0:
        if status is not FabricStatus.SUCCEEDED or changed is not None or result is None or failure is not None:
            raise FabricProtocolError("contradictory successful Fabric response")
    elif exit_code == 1:
        if status not in {FabricStatus.FAILED, FabricStatus.UNSUPPORTED} or failure is None or result is not None:
            raise FabricProtocolError("contradictory failed Fabric response")
        if changed not in {None, False}:
            raise FabricProtocolError("failed Fabric invoke has invalid changed value")
        if changed is False and failure.effects is not FabricEffects.NONE:
            raise FabricProtocolError("only a rejected invoke may report unchanged")
    else:
        raise FabricProtocolError("invalid Fabric exit code")
    return FabricResponse(operation=operation, status=status, changed=changed, result=result, error=failure)


def parse_run_result(
    response: FabricResponse,
    *,
    agent_name: str,
    runtime_id: str,
    adapter_id: str,
    harness: str,
    adapter_kind: str,
    request_id: str,
) -> FabricRunResult:
    """Validate a direct Fabric RunResult against the provisioned binding."""
    if response.operation != "invoke" or response.status is not FabricStatus.SUCCEEDED:
        raise FabricProtocolError("Fabric invoke did not return a RunResult")
    value = response.result
    if not isinstance(value, dict) or not _RUN_REQUIRED_FIELDS.issubset(value):
        raise FabricProtocolError("Fabric RunResult fields are invalid")
    expected = {
        "agent_name": agent_name,
        "runtime_id": runtime_id,
        "adapter_id": adapter_id,
        "harness": harness,
        "adapter_kind": adapter_kind,
        "request_id": request_id,
    }
    for field, selected in expected.items():
        observed = value.get(field)
        _required_identity(observed, f"Fabric RunResult {field}")
        if observed != selected:
            raise FabricProtocolError(f"Fabric RunResult {field} mismatch")
    invocation_id = value.get("invocation_id")
    _required_identity(invocation_id, "Fabric invocation ID")
    try:
        status = FabricRunStatus(value.get("status"))
    except (TypeError, ValueError) as error:
        raise FabricProtocolError("invalid Fabric RunResult status") from error
    error_value = value.get("error")
    if status is FabricRunStatus.SUCCEEDED:
        if error_value is not None:
            raise FabricProtocolError("successful Fabric RunResult contains an error")
    elif error_value is not None:
        _validate_run_error(error_value)
    usage = value.get("usage")
    if usage is not None and not isinstance(usage, dict):
        raise FabricProtocolError("Fabric RunResult usage is invalid")
    artifacts = value.get("artifacts", {})
    artifact_entries = artifacts.get("artifacts", []) if isinstance(artifacts, dict) else None
    if (
        not isinstance(artifacts, dict)
        or not isinstance(artifact_entries, list)
        or any(not isinstance(artifact, dict) for artifact in artifact_entries)
        or (artifacts.get("root") is not None and not isinstance(artifacts.get("root"), str))
    ):
        raise FabricProtocolError("Fabric RunResult artifacts are invalid")
    telemetry = value.get("telemetry")
    if telemetry is not None and not isinstance(telemetry, (dict, list)):
        raise FabricProtocolError("Fabric RunResult telemetry is invalid")
    if isinstance(telemetry, list) and any(not isinstance(reference, dict) for reference in telemetry):
        raise FabricProtocolError("Fabric RunResult telemetry is invalid")
    events = value.get("events", [])
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise FabricProtocolError("Fabric RunResult events are invalid")
    if not isinstance(value.get("metadata", {}), dict):
        raise FabricProtocolError("Fabric RunResult metadata is invalid")
    return FabricRunResult(
        agent_name=agent_name,
        harness=harness,
        adapter_kind=adapter_kind,
        adapter_id=adapter_id,
        runtime_id=runtime_id,
        invocation_id=invocation_id,
        request_id=request_id,
        status=status,
        value=value,
    )


def resolve_json_pointer(value: Any, pointer: str) -> Any:
    """Resolve one RFC 6901 JSON Pointer without implicit coercion."""
    current = value
    for token in parse_json_pointer(pointer):
        if isinstance(current, dict):
            if token not in current:
                raise FabricProtocolError("result pointer does not exist")
            current = current[token]
            continue
        if isinstance(current, list):
            if not token.isascii() or not token.isdecimal() or (len(token) > 1 and token.startswith("0")):
                raise FabricProtocolError("result pointer contains an invalid array index")
            index = int(token)
            if index >= len(current):
                raise FabricProtocolError("result pointer does not exist")
            current = current[index]
            continue
        raise FabricProtocolError("result pointer traverses a scalar")
    return current


def parse_json_pointer(pointer: str) -> tuple[str, ...]:
    """Validate and tokenize one JSON Pointer."""
    if not isinstance(pointer, str) or "\x00" in pointer or len(pointer.encode("utf-8")) > 1024:
        raise FabricProtocolError("invalid result pointer")
    if pointer == "":
        return ()
    if not pointer.startswith("/"):
        raise FabricProtocolError("result pointer must be an RFC 6901 pointer")
    tokens: list[str] = []
    for raw in pointer[1:].split("/"):
        decoded: list[str] = []
        index = 0
        while index < len(raw):
            if raw[index] != "~":
                decoded.append(raw[index])
                index += 1
                continue
            if index + 1 >= len(raw) or raw[index + 1] not in {"0", "1"}:
                raise FabricProtocolError("invalid result pointer escape")
            decoded.append("~" if raw[index + 1] == "0" else "/")
            index += 2
        tokens.append("".join(decoded))
    return tuple(tokens)


def _encode_json_request(value: dict[str, Any]) -> bytes:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError) as error:
        raise FabricProtocolError("invalid Fabric invocation request") from error
    if len(encoded) > MAX_FABRIC_REQUEST_BYTES:
        raise FabricProtocolError("Fabric invocation request exceeds the request limit")
    return encoded


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise FabricProtocolError(f"{name} is invalid")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise FabricProtocolError(f"{name} is invalid") from error
    return value


def _required_identity(value: object, name: str) -> str:
    resolved = _required_string(value, name)
    encoded = resolved.encode("utf-8")
    if len(encoded) > 512 or any(ord(character) < 0x20 or ord(character) == 0x7F for character in resolved):
        raise FabricProtocolError(f"{name} is invalid")
    return resolved


def _validate_run_error(value: object) -> None:
    if not isinstance(value, dict) or not _RUN_ERROR_REQUIRED_FIELDS.issubset(value):
        raise FabricProtocolError("failed Fabric RunResult has invalid error metadata")
    for field in ("stage", "code", "message"):
        _required_string(value.get(field), f"Fabric RunResult error {field}")
    if "retryable" in value and not isinstance(value["retryable"], bool):
        raise FabricProtocolError("failed Fabric RunResult has invalid error metadata")
    if "metadata" in value and not isinstance(value["metadata"], dict):
        raise FabricProtocolError("failed Fabric RunResult has invalid error metadata")


def _decode_failure(value: Any) -> FabricFailure | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != _ERROR_FIELDS:
        raise FabricProtocolError("invalid Fabric error fields")
    for field in ("code", "stage", "message"):
        item = value[field]
        _required_string(item, f"Fabric error {field}")
        if len(item.encode("utf-8")) > 4096:
            raise FabricProtocolError("Fabric error is too large")
    try:
        effects = FabricEffects(value["effects"])
    except (TypeError, ValueError) as error:
        raise FabricProtocolError("invalid Fabric effects") from error
    return FabricFailure(code=value["code"], effects=effects)


def _validate_json_strings(value: Any) -> None:
    pending = [(value, 0)]
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if depth > 64 or nodes > 100_000:
            raise FabricProtocolError("Fabric JSON exceeds structural limits")
        if isinstance(current, str):
            try:
                current.encode("utf-8")
            except UnicodeEncodeError as error:
                raise FabricProtocolError("invalid Fabric JSON string") from error
        elif isinstance(current, dict):
            pending.extend((item, depth + 1) for item in current)
            pending.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            pending.extend((item, depth + 1) for item in current)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FabricProtocolError("duplicate JSON member")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise FabricProtocolError("non-finite JSON number")


def _parse_integer(value: str) -> int:
    if len(value) > 128:
        raise FabricProtocolError("JSON integer exceeds the numeric limit")
    return int(value)


def _parse_float(value: str) -> float:
    if len(value) > 128:
        raise FabricProtocolError("JSON number exceeds the numeric limit")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise FabricProtocolError("non-finite JSON number")
    return parsed


__all__ = [
    "FABRIC_ADAPTER_KINDS",
    "FabricEffects",
    "FabricFailure",
    "FabricProtocolError",
    "FabricResponse",
    "FabricRunResult",
    "FabricRunStatus",
    "FabricStatus",
    "MAX_FABRIC_REQUEST_BYTES",
    "MAX_FABRIC_RESPONSE_BYTES",
    "decode_response",
    "encode_run_request",
    "parse_json_pointer",
    "parse_run_result",
    "resolve_json_pointer",
]
