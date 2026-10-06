# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

from __future__ import annotations

import json

import pytest

from voiceclaw.adapters.openshell_fabric.protocol import (
    MAX_FABRIC_REQUEST_BYTES,
    MAX_FABRIC_RESPONSE_BYTES,
    FabricEffects,
    FabricFirstTurnQualificationError,
    FabricProtocolError,
    FabricStatus,
    adapter_codec,
    decode_response,
    resolve_json_pointer,
    successful_fabric_result,
)


def _wire(value: object) -> str:
    return json.dumps(value, separators=(",", ":")) + "\n"


def _success() -> dict[str, object]:
    return {
        "operation": "invoke",
        "status": "succeeded",
        "changed": None,
        "result": {
            "runtime_id": "runtime-1",
            "fabric_result": {"status": "succeeded", "output": {"response": "result"}},
        },
        "error": None,
    }


def _first_turn_result(
    *,
    messages: object | None = None,
    response: object = "done",
) -> dict[str, object]:
    prompt = "do the work"
    return {
        "output": {
            "harness": "openclaw",
            "response": response,
            "session_key": "agent:default:fabric-runtime-1",
            "messages": (
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response, "tool_calls": []},
                ]
                if messages is None
                else messages
            ),
        }
    }


def test_openclaw_codec_uses_the_adapter_owned_request_shape() -> None:
    encoded = adapter_codec("nvidia.fabric.openclaw").encode_input(
        native_agent="default",
        prompt="do the work",
    )
    assert json.loads(encoded) == {"agent": "default", "message": "do the work"}
    assert len(encoded) <= MAX_FABRIC_REQUEST_BYTES
    codec = adapter_codec("nvidia.fabric.openclaw")
    assert codec.default_result_pointer == "/output/response"
    assert codec.expected_harness == "nvidia.fabric.openclaw"
    assert codec.expected_adapter_kind == "python"


def test_openclaw_codec_accepts_only_fresh_single_turn_history() -> None:
    codec = adapter_codec("nvidia.fabric.openclaw")
    prompt = "do the work"
    result = _first_turn_result()

    assert (
        codec.validate_first_turn_result(
            fabric_result=result,
            native_agent="default",
            runtime_id="runtime-1",
            prompt=prompt,
        )
        == "agent:default:fabric-runtime-1"
    )

    result["output"]["harness"] = "wrong"
    with pytest.raises(FabricProtocolError, match="output harness"):
        codec.validate_first_turn_result(
            fabric_result=result,
            native_agent="default",
            runtime_id="runtime-1",
            prompt=prompt,
        )
    result["output"]["harness"] = "openclaw"

    result["output"]["session_key"] = "agent:default:fabric-wrong"
    with pytest.raises(FabricProtocolError, match="session identity"):
        codec.validate_first_turn_result(
            fabric_result=result,
            native_agent="default",
            runtime_id="runtime-1",
            prompt=prompt,
        )
    result["output"]["session_key"] = "agent:default:fabric-runtime-1"


@pytest.mark.parametrize(
    ("messages", "response", "expected_error"),
    [
        ([], "done", FabricProtocolError),
        ([{"role": "user", "content": "do the work"}], "done", FabricProtocolError),
        (
            [
                {"role": "system", "content": "stale policy"},
                {"role": "user", "content": "do the work"},
                {"role": "assistant", "content": "done", "tool_calls": []},
            ],
            "done",
            FabricFirstTurnQualificationError,
        ),
        (
            [
                {"role": "user", "content": "stale turn"},
                {"role": "assistant", "content": "stale answer", "tool_calls": []},
                {"role": "user", "content": "do the work"},
                {"role": "assistant", "content": "done", "tool_calls": []},
            ],
            "done",
            FabricFirstTurnQualificationError,
        ),
        (
            [
                {"role": "user", "content": "do the work"},
                {"role": "tool", "content": "stale tool result"},
                {"role": "assistant", "content": "done", "tool_calls": []},
            ],
            "done",
            FabricFirstTurnQualificationError,
        ),
        (
            [
                {"role": "user", "content": "do the work"},
                {"role": "assistant", "content": "done", "tool_calls": []},
                {"role": "assistant", "content": "stale trailing response", "tool_calls": []},
            ],
            "done",
            FabricFirstTurnQualificationError,
        ),
        (
            ["malformed", {"role": "assistant", "content": "done", "tool_calls": []}],
            "done",
            FabricProtocolError,
        ),
        (
            [
                {"role": "user", "content": "different prompt"},
                {"role": "assistant", "content": "done", "tool_calls": []},
            ],
            "done",
            FabricProtocolError,
        ),
        (
            [
                {"role": "user", "content": "do the work", "extra": True},
                {"role": "assistant", "content": "done", "tool_calls": []},
            ],
            "done",
            FabricProtocolError,
        ),
        (
            [
                {"role": "user", "content": "do the work"},
                {"role": "assistant", "content": "different", "tool_calls": []},
            ],
            "done",
            FabricProtocolError,
        ),
        (
            [
                {"role": "user", "content": "do the work"},
                {"role": "assistant", "content": "done"},
            ],
            "done",
            FabricProtocolError,
        ),
        (
            [
                {"role": "user", "content": "do the work"},
                {"role": "assistant", "content": "done", "tool_calls": [{"name": "stale"}]},
            ],
            "done",
            FabricProtocolError,
        ),
        (
            [
                {"role": "user", "content": "do the work"},
                {"role": "assistant", "content": "done", "tool_calls": [], "extra": True},
            ],
            "done",
            FabricProtocolError,
        ),
        (
            [
                {"role": "user", "content": "do the work"},
                {"role": "assistant", "content": "", "tool_calls": []},
            ],
            "",
            FabricProtocolError,
        ),
    ],
    ids=(
        "empty-history",
        "missing-assistant",
        "stale-system",
        "stale-turn",
        "tool-message",
        "trailing-assistant",
        "malformed-message",
        "wrong-user-content",
        "extra-user-field",
        "assistant-result-mismatch",
        "missing-tool-calls",
        "nonempty-tool-calls",
        "extra-assistant-field",
        "empty-result",
    ),
)
def test_openclaw_codec_rejects_nonfresh_or_ambiguous_first_turn_evidence(
    messages: object,
    response: object,
    expected_error: type[FabricProtocolError],
) -> None:
    with pytest.raises(expected_error, match="qualification|context-isolation"):
        adapter_codec("nvidia.fabric.openclaw").validate_first_turn_result(
            fabric_result=_first_turn_result(messages=messages, response=response),
            native_agent="default",
            runtime_id="runtime-1",
            prompt="do the work",
        )


@pytest.mark.parametrize("messages", [None, {}, "history"])
def test_openclaw_codec_requires_a_message_list(messages: object) -> None:
    result = _first_turn_result()
    result["output"]["messages"] = messages

    with pytest.raises(FabricProtocolError, match="context-isolation evidence"):
        adapter_codec("nvidia.fabric.openclaw").validate_first_turn_result(
            fabric_result=result,
            native_agent="default",
            runtime_id="runtime-1",
            prompt="do the work",
        )


def test_unknown_adapter_never_guesses_a_request_codec() -> None:
    with pytest.raises(FabricProtocolError):
        adapter_codec("vendor.unknown")


def test_success_response_and_pointer_are_strictly_decoded() -> None:
    response = decode_response(operation="invoke", exit_code=0, stdout=_wire(_success()))
    assert response.status is FabricStatus.SUCCEEDED
    runtime_id, fabric_result = successful_fabric_result(response)
    assert runtime_id == "runtime-1"
    assert resolve_json_pointer(fabric_result, "/output/response") == "result"


def test_confirmed_rejection_preserves_effects_without_exposing_message() -> None:
    value = {
        "operation": "invoke",
        "status": "failed",
        "changed": False,
        "result": None,
        "error": {"code": "wrong_agent", "stage": "request", "message": "private", "effects": "none"},
    }
    response = decode_response(operation="invoke", exit_code=1, stdout=_wire(value))
    assert response.error is not None
    assert response.error.effects is FabricEffects.NONE


@pytest.mark.parametrize(
    "stdout,exit_code",
    [
        (json.dumps(_success()), 0),
        (_wire({**_success(), "extra": True}), 0),
        (
            '{"operation":"invoke","operation":"invoke","status":"succeeded",'
            '"changed":null,"result":{},"error":null}\n',
            0,
        ),
        (_wire({**_success(), "changed": False}), 0),
        (_wire({**_success(), "result": None}), 0),
        (
            _wire(
                {
                    "operation": "invoke",
                    "status": "failed",
                    "changed": False,
                    "result": None,
                    "error": None,
                }
            ),
            1,
        ),
        (_wire(_success()), 1),
        (_wire(_success()), 2),
        ("x" * MAX_FABRIC_RESPONSE_BYTES + "\n", 0),
    ],
    ids=(
        "missing-newline",
        "extra-field",
        "duplicate-key",
        "success-reports-unchanged",
        "success-without-result",
        "failure-without-error",
        "success-body-with-failure-exit",
        "invalid-exit-code",
        "oversized-response",
    ),
)
def test_malformed_or_contradictory_response_is_rejected(stdout: str, exit_code: int) -> None:
    with pytest.raises(FabricProtocolError):
        decode_response(operation="invoke", exit_code=exit_code, stdout=stdout)


@pytest.mark.parametrize(
    "stdout",
    [
        (
            '{"operation":"invoke","status":"succeeded","changed":null,"result":{"value":'
            + ("9" * 5_000)
            + '},"error":null}\n'
        ),
        (
            '{"operation":"invoke","status":"succeeded","changed":null,"result":{"value":'
            + ("[" * 1_100)
            + "0"
            + ("]" * 1_100)
            + '},"error":null}\n'
        ),
        (
            '{"operation":"invoke","status":"failed","changed":null,"result":null,'
            '"error":{"code":"failed","stage":"invoke","message":"\\ud800","effects":"unknown"}}\n'
        ),
        '{"operation":"invoke","status":"succeeded","changed":null,"result":{"value":1e100000},"error":null}\n',
    ],
    ids=["oversized-integer", "excessive-nesting", "lone-surrogate", "non-finite-number"],
)
def test_hostile_json_failures_are_normalized(stdout: str) -> None:
    with pytest.raises(FabricProtocolError):
        decode_response(operation="invoke", exit_code=1 if '"status":"failed"' in stdout else 0, stdout=stdout)


def test_check_accepts_a_well_formed_unsupported_snapshot() -> None:
    value = {
        "operation": "check",
        "status": "unsupported",
        "changed": False,
        "result": {
            "runtime_id": "runtime-1",
            "runtime_state": "running",
            "generation": "generation-1",
            "applied_config": {},
            "health": None,
        },
        "error": {"code": "health_unsupported", "stage": "health", "message": "private", "effects": "none"},
    }
    response = decode_response(operation="check", exit_code=1, stdout=_wire(value))
    assert response.status is FabricStatus.UNSUPPORTED


@pytest.mark.parametrize("status", ["failed", "unsupported"])
@pytest.mark.parametrize("effects", ["applied", "unknown"])
def test_check_failure_cannot_claim_side_effects(status: str, effects: str) -> None:
    value = {
        "operation": "check",
        "status": status,
        "changed": False,
        "result": None,
        "error": {"code": "check_failed", "stage": "health", "message": "private", "effects": effects},
    }
    with pytest.raises(FabricProtocolError, match="effects"):
        decode_response(operation="check", exit_code=1, stdout=_wire(value))


@pytest.mark.parametrize("changed", [None, True])
def test_check_response_must_be_observational(changed: bool | None) -> None:
    value = {
        "operation": "check",
        "status": "failed",
        "changed": changed,
        "result": None,
        "error": {"code": "check_failed", "stage": "health", "message": "private", "effects": "none"},
    }
    with pytest.raises(FabricProtocolError, match="must not report a mutation"):
        decode_response(operation="check", exit_code=1, stdout=_wire(value))


def test_json_pointer_rejects_missing_and_invalid_paths() -> None:
    with pytest.raises(FabricProtocolError):
        resolve_json_pointer({"output": {}}, "/output/response")
    with pytest.raises(FabricProtocolError):
        resolve_json_pointer({"output": {}}, "/output/~2response")
