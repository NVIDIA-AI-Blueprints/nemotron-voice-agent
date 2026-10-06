# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Strict codec for the deployed Fabric bridge command contract."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

MAX_FABRIC_REQUEST_BYTES = 512 * 1024
MAX_FABRIC_RESPONSE_BYTES = 4 * 1024 * 1024
_OUTER_FIELDS = frozenset({"operation", "status", "changed", "result", "error"})
_ERROR_FIELDS = frozenset({"code", "stage", "message", "effects"})


class FabricProtocolError(ValueError):
    """The bridge emitted a malformed or contradictory response."""


class FabricFirstTurnQualificationError(FabricProtocolError):
    """The selected runtime contained conversation history from another turn."""


class FabricStatus(StrEnum):
    """Terminal statuses emitted by a bridge command."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class FabricEffects(StrEnum):
    """Whether a failed bridge command may have affected its target."""

    NONE = "none"
    APPLIED = "applied"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FabricFailure:
    """Validated failure evidence safe to expose beyond the adapter."""

    code: str
    effects: FabricEffects


@dataclass(frozen=True, slots=True)
class FabricResponse:
    """One validated five-field bridge response."""

    operation: str
    status: FabricStatus
    changed: bool | None
    result: dict[str, Any] | None
    error: FabricFailure | None


@dataclass(frozen=True, slots=True)
class FabricAdapterCodec:
    """Qualified request and result binding for one selected Fabric adapter."""

    adapter_id: str
    revision: str
    expected_harness: str
    expected_adapter_kind: str
    expected_output_harness: str
    default_result_pointer: str

    def encode_input(self, *, native_agent: str, prompt: str) -> bytes:
        """Encode one request through the qualified adapter input shape."""
        if not isinstance(native_agent, str) or not native_agent or "\x00" in native_agent:
            raise FabricProtocolError("invalid Fabric native agent")
        if not isinstance(prompt, str) or not prompt or "\x00" in prompt:
            raise FabricProtocolError("invalid Fabric invocation input")
        return _encode_json_request({"agent": native_agent, "message": prompt})

    def validate_first_turn_result(
        self,
        *,
        fabric_result: dict[str, Any],
        native_agent: str,
        runtime_id: str,
        prompt: str,
    ) -> str:
        """Validate the fail-closed first-turn shape and return its session key."""
        output = fabric_result.get("output")
        if not isinstance(output, dict):
            raise FabricProtocolError("Fabric result output is invalid")
        if output.get("harness") != self.expected_output_harness:
            raise FabricProtocolError("Fabric result output harness mismatch")
        expected_session_key = f"agent:{native_agent}:fabric-{runtime_id}"
        if output.get("session_key") != expected_session_key:
            raise FabricProtocolError("Fabric result session identity mismatch")
        messages = output.get("messages")
        if not isinstance(messages, list):
            raise FabricProtocolError("Fabric result omitted context-isolation evidence")
        if any(not isinstance(message, dict) for message in messages):
            raise FabricProtocolError("Fabric result contains malformed context-isolation evidence")
        if len(messages) > 2:
            raise FabricFirstTurnQualificationError("Fabric result does not satisfy first-turn qualification")
        if len(messages) != 2:
            raise FabricProtocolError("Fabric result does not satisfy first-turn qualification")
        user_message, assistant_message = messages
        if set(user_message) != {"role", "content"} or user_message != {"role": "user", "content": prompt}:
            raise FabricProtocolError("Fabric result does not satisfy first-turn qualification")
        response = output.get("response")
        if (
            not isinstance(response, str)
            or not response
            or set(assistant_message) != {"role", "content", "tool_calls"}
            or assistant_message.get("role") != "assistant"
            or assistant_message.get("content") != response
            or assistant_message.get("tool_calls") != []
        ):
            raise FabricProtocolError("Fabric result does not satisfy first-turn qualification")
        return expected_session_key


def adapter_codec(adapter_id: str) -> FabricAdapterCodec:
    """Resolve only an adapter contract with checked request and result evidence."""
    if adapter_id != "nvidia.fabric.openclaw":
        raise FabricProtocolError("incompatible Fabric adapter codec")
    return FabricAdapterCodec(
        adapter_id="nvidia.fabric.openclaw",
        revision="v1",
        expected_harness="nvidia.fabric.openclaw",
        expected_adapter_kind="python",
        expected_output_harness="openclaw",
        default_result_pointer="/output/response",
    )


def _encode_json_request(value: dict[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError) as error:
        raise FabricProtocolError("invalid Fabric invocation input") from error
    if len(encoded) > MAX_FABRIC_REQUEST_BYTES:
        raise FabricProtocolError("Fabric invocation input exceeds the request limit")
    return encoded


def decode_response(*, operation: str, exit_code: int, stdout: str) -> FabricResponse:
    """Decode one newline-terminated bridge response without lossy fallback."""
    if operation not in {"check", "invoke"}:
        raise FabricProtocolError("unsupported bridge operation")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise FabricProtocolError("invalid bridge exit code")
    if not isinstance(stdout, str):
        raise FabricProtocolError("invalid bridge output")
    try:
        encoded = stdout.encode("utf-8")
    except UnicodeEncodeError as error:
        raise FabricProtocolError("invalid bridge output") from error
    if not encoded or len(encoded) > MAX_FABRIC_RESPONSE_BYTES or not encoded.endswith(b"\n"):
        raise FabricProtocolError("invalid bridge output framing")
    try:
        value = json.loads(
            stdout,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
            parse_int=_parse_integer,
        )
    except (ValueError, RecursionError) as error:
        raise FabricProtocolError("invalid bridge JSON") from error
    _validate_json_strings(value)
    if not isinstance(value, dict) or set(value) != _OUTER_FIELDS:
        raise FabricProtocolError("invalid bridge response fields")
    if value["operation"] != operation:
        raise FabricProtocolError("bridge operation mismatch")
    try:
        status = FabricStatus(value["status"])
    except (TypeError, ValueError) as error:
        raise FabricProtocolError("invalid bridge status") from error
    changed = value["changed"]
    if changed is not None and not isinstance(changed, bool):
        raise FabricProtocolError("invalid bridge changed value")
    result = value["result"]
    if result is not None and not isinstance(result, dict):
        raise FabricProtocolError("invalid bridge result")
    failure = _decode_failure(value["error"])
    if exit_code == 0:
        if status is not FabricStatus.SUCCEEDED or failure is not None or result is None:
            raise FabricProtocolError("contradictory successful bridge response")
    elif exit_code == 1:
        if status not in {FabricStatus.FAILED, FabricStatus.UNSUPPORTED} or failure is None:
            raise FabricProtocolError("contradictory failed bridge response")
    else:
        raise FabricProtocolError("invalid bridge exit code")
    if operation == "check":
        if changed is not False:
            raise FabricProtocolError("check must not report a mutation")
        if failure is not None and failure.effects is not FabricEffects.NONE:
            raise FabricProtocolError("check failure must not report target effects")
    if operation == "invoke":
        if status is FabricStatus.SUCCEEDED and changed is not None:
            raise FabricProtocolError("successful invoke must not report changed")
        if status is not FabricStatus.SUCCEEDED and changed not in {None, False}:
            raise FabricProtocolError("failed invoke has invalid changed value")
        if changed is False and (failure is None or failure.effects is not FabricEffects.NONE):
            raise FabricProtocolError("only a rejected invoke may report unchanged")
    return FabricResponse(
        operation=operation,
        status=status,
        changed=changed,
        result=result,
        error=failure,
    )


def successful_fabric_result(response: FabricResponse) -> tuple[str, dict[str, Any]]:
    """Return the runtime identity and successful public Fabric result."""
    if response.operation != "invoke" or response.status is not FabricStatus.SUCCEEDED:
        raise FabricProtocolError("invoke did not succeed")
    result = response.result
    if not isinstance(result, dict):
        raise FabricProtocolError("invoke result is missing")
    runtime_id = result.get("runtime_id")
    fabric_result = result.get("fabric_result")
    if (
        not isinstance(runtime_id, str)
        or not runtime_id.strip()
        or "\x00" in runtime_id
        or not isinstance(fabric_result, dict)
        or fabric_result.get("status") != "succeeded"
    ):
        raise FabricProtocolError("invalid successful invoke result")
    return runtime_id, fabric_result


def resolve_json_pointer(value: Any, pointer: str) -> Any:
    """Resolve one RFC 6901 JSON Pointer without implicit coercion."""
    tokens = parse_json_pointer(pointer)
    current = value
    for token in tokens:
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


def _decode_failure(value: Any) -> FabricFailure | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != _ERROR_FIELDS:
        raise FabricProtocolError("invalid bridge error fields")
    code = value["code"]
    stage = value["stage"]
    message = value["message"]
    if not all(isinstance(item, str) and item and "\x00" not in item for item in (code, stage, message)):
        raise FabricProtocolError("invalid bridge error")
    if any(len(item.encode("utf-8")) > 4096 for item in (code, stage, message)):
        raise FabricProtocolError("bridge error is too large")
    try:
        effects = FabricEffects(value["effects"])
    except (TypeError, ValueError) as error:
        raise FabricProtocolError("invalid bridge effects") from error
    return FabricFailure(code=code, effects=effects)


def _validate_json_strings(value: Any) -> None:
    """Reject decoded JSON strings that cannot cross the UTF-8 boundary."""
    pending = [(value, 0)]
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if depth > 64 or nodes > 100_000:
            raise FabricProtocolError("bridge JSON exceeds structural limits")
        if isinstance(current, str):
            try:
                current.encode("utf-8")
            except UnicodeEncodeError as error:
                raise FabricProtocolError("invalid bridge JSON string") from error
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
    "FabricAdapterCodec",
    "FabricEffects",
    "FabricFailure",
    "FabricProtocolError",
    "FabricResponse",
    "FabricStatus",
    "MAX_FABRIC_REQUEST_BYTES",
    "MAX_FABRIC_RESPONSE_BYTES",
    "adapter_codec",
    "decode_response",
    "parse_json_pointer",
    "resolve_json_pointer",
    "successful_fabric_result",
]
