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


def _tool_history() -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "content": "\n\n",
            "tool_calls": [
                {
                    "id": "call-memory",
                    "type": "function",
                    "function": {"name": "memory_get", "arguments": '{"path":"USER.md","from":1,"lines":1}'},
                }
            ],
        },
        {"role": "tool", "content": '{"text":"# USER.md"}', "tool_call_id": "call-memory", "is_error": False},
    ]


@pytest.mark.parametrize("is_error", [False, True])
def test_openclaw_codec_accepts_a_completed_memory_tool_in_the_first_turn(is_error: bool) -> None:
    result = _first_turn_result()
    history = _tool_history()
    history[1]["is_error"] = is_error
    result["output"]["messages"][1:1] = history

    assert (
        adapter_codec("nvidia.fabric.openclaw").validate_first_turn_result(
            fabric_result=result, native_agent="default", runtime_id="runtime-1", prompt="do the work"
        )
        == "agent:default:fabric-runtime-1"
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_openclaw_codec_matches_every_tool_result_across_batches(parallel: bool) -> None:
    result = _first_turn_result()
    first, second = _tool_history(), _tool_history()
    second[0]["tool_calls"][0]["id"] = "call-second"
    second[1]["tool_call_id"] = "call-second"
    if parallel:
        first[0]["tool_calls"].extend(second[0]["tool_calls"])
        history = [first[0], second[1], first[1]]
    else:
        history = first + second
    result["output"]["messages"][1:1] = history

    assert (
        adapter_codec("nvidia.fabric.openclaw").validate_first_turn_result(
            fabric_result=result, native_agent="default", runtime_id="runtime-1", prompt="do the work"
        )
        == "agent:default:fabric-runtime-1"
    )


@pytest.mark.parametrize(
    "case",
    [
        "missing-result",
        "unknown-result",
        "duplicate-result",
        "result-before-call",
        "reused-call-id",
        "duplicate-call-id",
        "unfinished-batch",
        "extra-user",
        "extra-system",
        "extra-assistant",
        "missing-calls",
        "calls-not-list",
        "call-not-object",
        "extra-call-field",
        "wrong-call-type",
        "empty-call-id",
        "nonstring-call-id",
        "null-call-id",
        "function-not-object",
        "extra-function-field",
        "empty-function-name",
        "nonstring-arguments",
        "nonstring-assistant-content",
        "extra-assistant-field",
        "missing-error-flag",
        "extra-result-field",
        "nonboolean-error-flag",
        "nonstring-result-id",
        "nonstring-result-content",
        "wrong-final-response",
        "unfinished-final-call",
        "wrong-session",
        "wrong-prompt",
    ],
)
def test_openclaw_codec_rejects_unqualified_tool_chains(case: str) -> None:
    result = _first_turn_result()
    history = _tool_history()
    assistant, tool = history
    call = assistant["tool_calls"][0]
    function = call["function"]
    if case == "missing-result":
        history.pop()
    elif case == "unknown-result":
        tool["tool_call_id"] = "unrelated"
    elif case == "duplicate-result":
        history.append(dict(tool))
    elif case == "result-before-call":
        history.reverse()
    elif case == "reused-call-id":
        history.extend(_tool_history())
    elif case == "duplicate-call-id":
        assistant["tool_calls"].append(dict(call))
    elif case == "unfinished-batch":
        history.insert(1, _tool_history()[0])
    elif case in {"extra-user", "extra-system", "extra-assistant"}:
        history.append({"role": case.removeprefix("extra-"), "content": "unrelated", "tool_calls": []})
    elif case == "missing-calls":
        assistant.pop("tool_calls")
    elif case == "calls-not-list":
        assistant["tool_calls"] = {}
    elif case == "call-not-object":
        assistant["tool_calls"] = ["call-memory"]
    elif case == "extra-call-field":
        call["extra"] = True
    elif case == "wrong-call-type":
        call["type"] = "custom"
    elif case == "empty-call-id":
        call["id"] = ""
    elif case == "nonstring-call-id":
        call["id"] = []
    elif case == "null-call-id":
        call["id"] = "call\x00memory"
    elif case == "function-not-object":
        call["function"] = "memory_get"
    elif case == "extra-function-field":
        function["extra"] = True
    elif case == "empty-function-name":
        function["name"] = ""
    elif case == "nonstring-arguments":
        function["arguments"] = {}
    elif case == "nonstring-assistant-content":
        assistant["content"] = None
    elif case == "extra-assistant-field":
        assistant["extra"] = True
    elif case == "missing-error-flag":
        tool.pop("is_error")
    elif case == "extra-result-field":
        tool["extra"] = True
    elif case == "nonboolean-error-flag":
        tool["is_error"] = 0
    elif case == "nonstring-result-id":
        tool["tool_call_id"] = []
    elif case == "nonstring-result-content":
        tool["content"] = {}
    elif case == "wrong-final-response":
        result["output"]["messages"][-1]["content"] = "different"
    elif case == "unfinished-final-call":
        result["output"]["messages"][-1]["tool_calls"] = [dict(call)]
    elif case == "wrong-session":
        result["output"]["session_key"] = "agent:default:fabric-wrong"
    elif case == "wrong-prompt":
        result["output"]["messages"][0]["content"] = "different"
    else:
        raise AssertionError(case)
    result["output"]["messages"][1:1] = history

    with pytest.raises(FabricProtocolError):
        adapter_codec("nvidia.fabric.openclaw").validate_first_turn_result(
            fabric_result=result, native_agent="default", runtime_id="runtime-1", prompt="do the work"
        )


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


@pytest.mark.parametrize(
    ("response_padding", "history_padding"),
    [("", "\n"), ("\n", ""), (" \t\r\n", "\n\r\t "), ("", "\t\r\n ")],
)
def test_openclaw_codec_accepts_only_outer_json_whitespace_differences(
    response_padding: str, history_padding: str
) -> None:
    body = '{"schema":"voiceclaw.result.v1","speech":null,"display":"Done."}'
    response = response_padding + body + response_padding
    result = _first_turn_result(response=response)
    result["output"]["messages"][1]["content"] = history_padding + body + history_padding

    assert (
        adapter_codec("nvidia.fabric.openclaw").validate_first_turn_result(
            fabric_result=result, native_agent="default", runtime_id="runtime-1", prompt="do the work"
        )
        == "agent:default:fabric-runtime-1"
    )
    assert result["output"]["response"] == response


@pytest.mark.parametrize(
    "history",
    [
        '{"display":"different"}',
        '{ "display":"Done."}',
        '{"display":" Done."}',
        '\u00a0{"display":"Done."}\u00a0',
        '\v{"display":"Done."}\v',
        '\f{"display":"Done."}\f',
        None,
        [],
    ],
    ids=("different-value", "internal-format", "inside-string", "nbsp", "vertical-tab", "form-feed", "null", "list"),
)
def test_openclaw_codec_rejects_body_changes_and_non_json_whitespace(history: object) -> None:
    result = _first_turn_result(response='{"display":"Done."}')
    result["output"]["messages"][1]["content"] = history

    with pytest.raises(FabricProtocolError, match="qualification"):
        adapter_codec("nvidia.fabric.openclaw").validate_first_turn_result(
            fabric_result=result, native_agent="default", runtime_id="runtime-1", prompt="do the work"
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
