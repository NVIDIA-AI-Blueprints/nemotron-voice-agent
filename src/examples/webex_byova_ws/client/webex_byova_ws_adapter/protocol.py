"""Strict parsing and construction of Cisco AsyncAPI envelopes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

ENVELOPE_KEYS = {"type", "seq", "ts", "conversation_id", "metadata", "payload"}
REQUEST_KEYS = {
    "conversation_id",
    "customer_org_id",
    "virtual_agent_id",
    "allow_partial_responses",
    "vendor_specific_config",
    "voice_va_input_type",
    "additional_info",
}
INPUT_KEYS = {"audio_input", "dtmf_input", "event_input"}
EVENT_TYPES = {"SESSION_START", "SESSION_END", "NO_INPUT", "START_OF_DTMF", "CUSTOM_EVENT"}
DTMF_DIGITS = {
    *(
        f"DTMF_DIGIT_{name}"
        for name in ("ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE")
    ),
    *(f"DTMF_DIGIT_{letter}" for letter in "ABCD"),
    "DTMF_DIGIT_STAR",
    "DTMF_DIGIT_POUND",
}
ERROR_STATUS = {
    "unauthorized": 401,
    "bad_request": 400,
    "unsupported_media": 415,
    "upstream_error": 502,
    "rate_limit": 429,
    "timeout": 504,
}
MAX_ERROR_DETAIL = 256


class ProtocolError(ValueError):
    """A safe protocol failure that can be returned to Webex."""

    def __init__(self, detail: str, code: str = "bad_request") -> None:
        """Create a bounded peer-safe protocol error."""
        super().__init__(detail)
        self.code = code
        self.detail = detail[:MAX_ERROR_DETAIL]


@dataclass(frozen=True, slots=True)
class Request:
    """Validated request envelope."""

    seq: int
    conversation_id: str
    payload: dict[str, Any]
    input_kind: str
    input_value: dict[str, Any]


class OutboundSequence:
    """Independent monotonic server sequence."""

    def __init__(self) -> None:
        """Initialize the sequence before its first value."""
        self._value = 0

    def next(self) -> int:
        """Return the next sequence value."""
        self._value += 1
        return self._value


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{name} must be an object")
    return value


def _only_keys(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ProtocolError(f"{name} contains unknown fields: {', '.join(unknown)}")


def _date_time(value: Any, name: str) -> None:
    if not isinstance(value, str):
        raise ProtocolError(f"{name} must be a date-time string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolError(f"{name} must be an RFC 3339 date-time") from exc
    if parsed.utcoffset() is None:
        raise ProtocolError(f"{name} must include a timezone")


def parse_text(raw: str) -> dict[str, Any]:
    """Decode one JSON object without accepting constants or duplicate keys."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ProtocolError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except ProtocolError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise ProtocolError("message must be valid JSON") from exc
    return _object(value, "message")


def parse_request(message: dict[str, Any]) -> Request:
    """Validate a VOICE_VA_REQUEST against the published schema."""
    _only_keys(message, ENVELOPE_KEYS, "envelope")
    required = {"type", "seq", "ts", "conversation_id", "payload"}
    if not required <= message.keys():
        raise ProtocolError("request envelope is missing required fields")
    if message["type"] != "VOICE_VA_REQUEST":
        raise ProtocolError("expected VOICE_VA_REQUEST")
    if not isinstance(message["seq"], int) or isinstance(message["seq"], bool) or message["seq"] < 1:
        raise ProtocolError("seq must be a positive integer")
    conversation_id = message["conversation_id"]
    if not isinstance(conversation_id, str) or not conversation_id:
        raise ProtocolError("conversation_id must be a nonempty string")
    _date_time(message["ts"], "ts")
    if "metadata" in message:
        _object(message["metadata"], "metadata")
    payload = _object(message["payload"], "payload")
    _only_keys(payload, REQUEST_KEYS, "payload")
    if not {"conversation_id", "customer_org_id", "voice_va_input_type"} <= payload.keys():
        raise ProtocolError("payload is missing required fields")
    if payload["conversation_id"] != conversation_id:
        raise ProtocolError("payload conversation_id does not match envelope")
    if not isinstance(payload["customer_org_id"], str) or not payload["customer_org_id"]:
        raise ProtocolError("customer_org_id must be a nonempty string")
    for name in ("virtual_agent_id", "vendor_specific_config"):
        if name in payload and not isinstance(payload[name], str):
            raise ProtocolError(f"{name} must be a string")
    if "allow_partial_responses" in payload and not isinstance(payload["allow_partial_responses"], bool):
        raise ProtocolError("allow_partial_responses must be a boolean")
    if "additional_info" in payload:
        additional_info = _object(payload["additional_info"], "additional_info")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in additional_info.items()):
            raise ProtocolError("additional_info values must be strings")
    wrapper = _object(payload["voice_va_input_type"], "voice_va_input_type")
    if len(wrapper) != 1 or next(iter(wrapper), None) not in INPUT_KEYS:
        raise ProtocolError("voice_va_input_type must contain exactly one supported input")
    kind, value = next(iter(wrapper.items()))
    value = _object(value, kind)
    _validate_input(kind, value)
    return Request(message["seq"], conversation_id, payload, kind, value)


def _validate_input(kind: str, value: dict[str, Any]) -> None:
    if kind == "audio_input":
        allowed = {
            "caller_audio_b64",
            "encoding",
            "sample_rate_hertz",
            "audio_timestamp",
            "language_code",
            "is_single_utterance",
        }
        _only_keys(value, allowed, kind)
        if not {"encoding", "sample_rate_hertz"} <= value.keys():
            raise ProtocolError("audio_input is missing encoding or sample_rate_hertz")
        if "caller_audio_b64" in value and not isinstance(value["caller_audio_b64"], str):
            raise ProtocolError("caller_audio_b64 must be a string")
        if value["encoding"] not in {
            "UNSPECIFIED_FORMAT",
            "LINEAR16_FORMAT",
            "MULAW_FORMAT",
            "ALAW_FORMAT",
        }:
            raise ProtocolError("audio_input contains an invalid encoding")
        if not isinstance(value["sample_rate_hertz"], int) or isinstance(value["sample_rate_hertz"], bool):
            raise ProtocolError("sample_rate_hertz must be an integer")
        if "audio_timestamp" in value:
            _date_time(value["audio_timestamp"], "audio_timestamp")
        if "language_code" in value and not isinstance(value["language_code"], str):
            raise ProtocolError("language_code must be a string")
        if "is_single_utterance" in value and not isinstance(value["is_single_utterance"], bool):
            raise ProtocolError("is_single_utterance must be a boolean")
    elif kind == "dtmf_input":
        _only_keys(value, {"dtmf_events"}, kind)
        events = value.get("dtmf_events")
        if not isinstance(events, list) or any(item not in DTMF_DIGITS for item in events):
            raise ProtocolError("dtmf_events contains an invalid digit")
    else:
        _only_keys(value, {"event_type", "name", "parameters"}, kind)
        if value.get("event_type") not in EVENT_TYPES:
            raise ProtocolError("event_input contains an invalid event_type")
        if "name" in value and not isinstance(value["name"], str):
            raise ProtocolError("event_input name must be a string")
        if "parameters" in value:
            _object(value["parameters"], "event_input parameters")


def timestamp() -> str:
    """Return an RFC 3339 UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def response(seq: int, conversation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Build a schema-conformant response envelope."""
    return {
        "type": "VOICE_VA_RESPONSE",
        "seq": seq,
        "ts": timestamp(),
        "conversation_id": conversation_id,
        "payload": payload,
    }


def error(seq: int, conversation_id: str, code: str, detail: str) -> dict[str, Any]:
    """Build a bounded, non-sensitive error envelope."""
    safe_code = code if code in ERROR_STATUS else "upstream_error"
    return {
        "type": "ERROR",
        "seq": seq,
        "ts": timestamp(),
        "conversation_id": conversation_id or "unknown",
        "code": safe_code,
        "status": ERROR_STATUS[safe_code],
        "detail": detail[:MAX_ERROR_DETAIL],
    }


def pong(ping: dict[str, Any], conversation_id: str) -> dict[str, Any]:
    """Build a PONG that echoes the PING sequence as required."""
    _only_keys(ping, ENVELOPE_KEYS - {"payload"}, "PING envelope")
    if not {"type", "seq", "ts", "conversation_id"} <= ping.keys() or ping.get("type") != "PING":
        raise ProtocolError("invalid PING envelope")
    seq = ping.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 1:
        raise ProtocolError("PING seq must be a positive integer")
    _date_time(ping.get("ts"), "PING ts")
    if "metadata" in ping:
        _object(ping["metadata"], "PING metadata")
    if ping.get("conversation_id") != conversation_id:
        raise ProtocolError("PING conversation_id does not match session")
    return {"type": "PONG", "seq": seq, "ts": timestamp(), "conversation_id": conversation_id}
