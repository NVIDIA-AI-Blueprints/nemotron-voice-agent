"""Protocol conformance tests."""

from __future__ import annotations

import json

import pytest

from webex_byova_ws_adapter.protocol import (
    OutboundSequence,
    ProtocolError,
    error,
    parse_request,
    parse_text,
    pong,
)


def request(*, seq: int = 1, conversation_id: str = "call-1", input_value: dict | None = None) -> dict:
    """Build a minimal valid request."""
    return {
        "type": "VOICE_VA_REQUEST",
        "seq": seq,
        "ts": "2026-09-07T00:00:00Z",
        "conversation_id": conversation_id,
        "payload": {
            "conversation_id": conversation_id,
            "customer_org_id": "org-1",
            "voice_va_input_type": {
                "event_input": input_value or {"event_type": "SESSION_START"},
            },
        },
    }


def test_request_is_strict_and_preserves_input() -> None:
    parsed = parse_request(request(input_value={"event_type": "CUSTOM_EVENT", "name": "x", "parameters": {}}))
    assert parsed.input_kind == "event_input"
    assert parsed.input_value["event_type"] == "CUSTOM_EVENT"
    bad = request()
    bad["payload"]["unexpected"] = True
    with pytest.raises(ProtocolError, match="unknown fields"):
        parse_request(bad)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("ts",), "not-a-date", "RFC 3339"),
        (("metadata",), [], "metadata"),
        (("payload", "allow_partial_responses"), 1, "boolean"),
        (("payload", "additional_info"), {"key": 1}, "values must be strings"),
    ],
)
def test_request_rejects_schema_type_mismatches(path: tuple[str, ...], value: object, message: str) -> None:
    invalid = request()
    target = invalid
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ProtocolError, match=message):
        parse_request(invalid)


def test_json_rejects_duplicates_and_constants() -> None:
    with pytest.raises(ProtocolError, match="duplicate"):
        parse_text('{"type":"PING","type":"PONG"}')
    with pytest.raises(ProtocolError, match="valid JSON"):
        parse_text('{"seq":NaN}')


def test_pong_echoes_ping_sequence_without_consuming_server_sequence() -> None:
    sequence = OutboundSequence()
    assert sequence.next() == 1
    result = pong(
        {"type": "PING", "seq": 91, "ts": "2026-09-07T00:00:00Z", "conversation_id": "call-1"},
        "call-1",
    )
    assert result["seq"] == 91
    assert sequence.next() == 2


def test_error_is_schema_shaped_and_bounded() -> None:
    result = error(2, "call-1", "bad_request", "x" * 1000)
    assert result.keys() == {"type", "seq", "ts", "conversation_id", "code", "status", "detail"}
    assert len(result["detail"]) == 256
    json.dumps(result)


@pytest.mark.parametrize(
    "digit",
    ["DTMF_DIGIT_ZERO", "DTMF_DIGIT_A", "DTMF_DIGIT_STAR", "DTMF_DIGIT_POUND"],
)
def test_dtmf_schema_values_are_accepted(digit: str) -> None:
    value = request()
    value["payload"]["voice_va_input_type"] = {"dtmf_input": {"dtmf_events": [digit]}}
    assert parse_request(value).input_kind == "dtmf_input"
