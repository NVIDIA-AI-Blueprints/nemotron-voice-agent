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
    FabricProtocolError,
    FabricRunStatus,
    FabricStatus,
    decode_response,
    encode_run_request,
    parse_run_result,
    resolve_json_pointer,
)


def _wire(value: object) -> str:
    return json.dumps(value, separators=(",", ":")) + "\n"


def _run_result(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "agent_name": "researcher",
        "harness": "nvidia.fabric.hermes",
        "adapter_kind": "python",
        "adapter_id": "nvidia.fabric.hermes",
        "runtime_id": "runtime-1",
        "invocation_id": "invocation-1",
        "request_id": "commit-1",
        "status": "succeeded",
        "output": {"response": "result"},
        "error": None,
        "usage": None,
        "artifacts": {"root": None, "artifacts": []},
        "telemetry": [],
        "events": [],
        "metadata": {},
    }
    result.update(overrides)
    return result


def _success(**result_overrides: object) -> dict[str, object]:
    return {
        "operation": "invoke",
        "status": "succeeded",
        "changed": None,
        "result": _run_result(**result_overrides),
        "error": None,
    }


def test_run_request_carries_fabric_and_voiceclaw_correlation() -> None:
    encoded = encode_run_request(
        input_text="do the work",
        request_id="commit-1",
        relay_session_root="conversation-1",
        context={
            "voiceclaw_conversation_id": "conversation-1",
            "voiceclaw_commit_id": "commit-1",
        },
    )

    assert json.loads(encoded) == {
        "input": "do the work",
        "request_id": "commit-1",
        "relay_session_root": "conversation-1",
        "context": {
            "voiceclaw_conversation_id": "conversation-1",
            "voiceclaw_commit_id": "commit-1",
        },
    }
    assert len(encoded) <= MAX_FABRIC_REQUEST_BYTES


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("input_text", ""),
        lambda value: value.__setitem__("request_id", ""),
        lambda value: value.__setitem__("request_id", "x" * 513),
        lambda value: value.__setitem__("relay_session_root", "bad\x00root"),
        lambda value: value.__setitem__("relay_session_root", "bad\nroot"),
        lambda value: value.__setitem__("context", []),
    ],
)
def test_run_request_rejects_invalid_correlation(mutation) -> None:
    values: dict[str, object] = {
        "input_text": "do the work",
        "request_id": "commit-1",
        "relay_session_root": "conversation-1",
        "context": {"voiceclaw_conversation_id": "conversation-1"},
    }
    mutation(values)

    with pytest.raises(FabricProtocolError):
        encode_run_request(**values)


def test_direct_run_result_is_decoded_and_validated() -> None:
    response = decode_response(operation="invoke", exit_code=0, stdout=_wire(_success()))
    assert response.status is FabricStatus.SUCCEEDED

    result = parse_run_result(
        response,
        agent_name="researcher",
        runtime_id="runtime-1",
        adapter_id="nvidia.fabric.hermes",
        harness="nvidia.fabric.hermes",
        adapter_kind="python",
        request_id="commit-1",
    )

    assert result.status is FabricRunStatus.SUCCEEDED
    assert result.invocation_id == "invocation-1"
    assert resolve_json_pointer(result.value, "/output/response") == "result"


def test_sparse_run_result_and_additive_fields_are_accepted() -> None:
    result = _run_result(future_fabric_field={"version": 1})
    for field in ("error", "usage", "telemetry", "events", "metadata"):
        result.pop(field)
    result["artifacts"] = {}
    response = decode_response(operation="invoke", exit_code=0, stdout=_wire({**_success(), "result": result}))

    parsed = parse_run_result(
        response,
        agent_name="researcher",
        runtime_id="runtime-1",
        adapter_id="nvidia.fabric.hermes",
        harness="nvidia.fabric.hermes",
        adapter_kind="python",
        request_id="commit-1",
    )

    assert parsed.status is FabricRunStatus.SUCCEEDED
    assert parsed.value["future_fabric_field"] == {"version": 1}


@pytest.mark.parametrize(
    "field",
    [
        "agent_name",
        "harness",
        "adapter_kind",
        "adapter_id",
        "runtime_id",
        "invocation_id",
        "request_id",
        "status",
        "output",
    ],
)
def test_run_result_requires_binding_correlation_status_and_output(field: str) -> None:
    result = _run_result()
    result.pop(field)
    response = decode_response(operation="invoke", exit_code=0, stdout=_wire({**_success(), "result": result}))

    with pytest.raises(FabricProtocolError):
        parse_run_result(
            response,
            agent_name="researcher",
            runtime_id="runtime-1",
            adapter_id="nvidia.fabric.hermes",
            harness="nvidia.fabric.hermes",
            adapter_kind="python",
            request_id="commit-1",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("agent_name", "wrong"),
        ("runtime_id", "wrong"),
        ("adapter_id", "wrong"),
        ("harness", "wrong"),
        ("adapter_kind", "wrong"),
        ("request_id", "wrong"),
    ],
)
def test_run_result_rejects_binding_or_correlation_mismatch(field: str, value: object) -> None:
    response = decode_response(operation="invoke", exit_code=0, stdout=_wire(_success(**{field: value})))

    with pytest.raises(FabricProtocolError, match="mismatch"):
        parse_run_result(
            response,
            agent_name="researcher",
            runtime_id="runtime-1",
            adapter_id="nvidia.fabric.hermes",
            harness="nvidia.fabric.hermes",
            adapter_kind="python",
            request_id="commit-1",
        )


def test_terminal_harness_failure_is_not_mistaken_for_transport_success() -> None:
    response = decode_response(
        operation="invoke",
        exit_code=0,
        stdout=_wire(
            _success(
                status="failed",
                error={
                    "stage": "invoke",
                    "code": "harness_failed",
                    "message": "private",
                },
            )
        ),
    )

    result = parse_run_result(
        response,
        agent_name="researcher",
        runtime_id="runtime-1",
        adapter_id="nvidia.fabric.hermes",
        harness="nvidia.fabric.hermes",
        adapter_kind="python",
        request_id="commit-1",
    )

    assert result.status is FabricRunStatus.FAILED


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_unsuccessful_run_result_may_omit_error(status: str) -> None:
    result = _run_result(status=status)
    result.pop("error")
    response = decode_response(
        operation="invoke",
        exit_code=0,
        stdout=_wire({**_success(), "result": result}),
    )

    result = parse_run_result(
        response,
        agent_name="researcher",
        runtime_id="runtime-1",
        adapter_id="nvidia.fabric.hermes",
        harness="nvidia.fabric.hermes",
        adapter_kind="python",
        request_id="commit-1",
    )

    assert result.status is FabricRunStatus(status)


def test_confirmed_bridge_rejection_preserves_effects_without_exposing_message() -> None:
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
        (_wire(_success()), 1),
        (_wire(_success()), 2),
        ("x" * MAX_FABRIC_RESPONSE_BYTES + "\n", 0),
    ],
)
def test_malformed_or_contradictory_response_is_rejected(stdout: str, exit_code: int) -> None:
    with pytest.raises(FabricProtocolError):
        decode_response(operation="invoke", exit_code=exit_code, stdout=stdout)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: result.pop("invocation_id"),
        lambda result: result.__setitem__("invocation_id", ""),
        lambda result: result.__setitem__("invocation_id", "x" * 513),
        lambda result: result.__setitem__("invocation_id", "line\nbreak"),
        lambda result: result.__setitem__("status", "running"),
        lambda result: result.pop("output"),
        lambda result: result.__setitem__("error", "failed"),
        lambda result: result.__setitem__("status", "succeeded") or result.__setitem__("error", {}),
        lambda result: result.__setitem__("status", "cancelled") or result.__setitem__("error", "bad"),
        lambda result: (
            result.__setitem__("status", "failed") or result.__setitem__("error", {"stage": "invoke", "code": "failed"})
        ),
        lambda result: (
            result.__setitem__("status", "failed")
            or result.__setitem__(
                "error", {"stage": "invoke", "code": "failed", "message": "private", "retryable": "no"}
            )
        ),
        lambda result: (
            result.__setitem__("status", "cancelled")
            or result.__setitem__(
                "error", {"stage": "invoke", "code": "cancelled", "message": "private", "metadata": []}
            )
        ),
        lambda result: result.__setitem__("usage", []),
        lambda result: result.__setitem__("artifacts", []),
        lambda result: result.__setitem__("artifacts", {"root": None, "artifacts": ["bad"]}),
        lambda result: result.__setitem__("telemetry", "bad"),
        lambda result: result.__setitem__("telemetry", ["bad"]),
        lambda result: result.__setitem__("events", {}),
        lambda result: result.__setitem__("events", ["bad"]),
        lambda result: result.__setitem__("metadata", []),
    ],
)
def test_malformed_direct_run_result_is_rejected(mutation) -> None:
    result = _run_result()
    mutation(result)
    response = decode_response(operation="invoke", exit_code=0, stdout=_wire({**_success(), "result": result}))

    with pytest.raises(FabricProtocolError):
        parse_run_result(
            response,
            agent_name="researcher",
            runtime_id="runtime-1",
            adapter_id="nvidia.fabric.hermes",
            harness="nvidia.fabric.hermes",
            adapter_kind="python",
            request_id="commit-1",
        )


def test_json_pointer_rejects_missing_and_invalid_paths() -> None:
    with pytest.raises(FabricProtocolError):
        resolve_json_pointer({"output": {}}, "/output/response")
    with pytest.raises(FabricProtocolError):
        resolve_json_pointer({"output": {}}, "/output/~2response")
