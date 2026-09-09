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
EVENT_TYPES = {
    "UNSPECIFIED_INPUT",
    "SESSION_START",
    "SESSION_END",
    "NO_INPUT",
    "START_OF_DTMF",
    "CUSTOM_EVENT",
}
RESPONSE_KEYS = {
    "prompts",
    "output_events",
    "input_sensitive",
    "input_mode",
    "input_handling_config",
    "session_transcript",
    "session_summary",
    "disable_prompt_cancellation",
    "response_type",
}
RESPONSE_TYPES = {"PARTIAL", "CHUNK", "FINAL"}
INPUT_MODES = {
    "INPUT_VOICE_MODE_UNSPECIFIED",
    "INPUT_VOICE",
    "INPUT_EVENT_DTMF",
    "INPUT_VOICE_DTMF",
}
OUTPUT_EVENT_TYPES = {
    "UNSPECIFIED_EVENT",
    "START_OF_INPUT",
    "END_OF_INPUT",
    "NO_INPUT",
    "NO_MATCH",
    "TRANSFER_TO_AGENT",
    "CUSTOM_EVENT",
    "SESSION_END",
}
PROMPT_KEYS = {"audio_content_b64", "audio_uri", "text", "is_barge_in_enabled"}
OUTPUT_EVENT_KEYS = {"event_type", "name", "metadata"}
DTMF_DIGITS = {
    *(
        f"DTMF_DIGIT_{name}"
        for name in ("ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE")
    ),
    *(f"DTMF_DIGIT_{letter}" for letter in "ABCD"),
    "DTMF_DIGIT_STAR",
    "DTMF_DIGIT_POUND",
    "DTMF_EVENT_UNSPECIFIED",
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
    if message.get("metadata") is not None:
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
        if payload.get(name) is not None and not isinstance(payload[name], str):
            raise ProtocolError(f"{name} must be a string")
    if payload.get("allow_partial_responses") is not None and not isinstance(payload["allow_partial_responses"], bool):
        raise ProtocolError("allow_partial_responses must be a boolean")
    if payload.get("additional_info") is not None:
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
        if value.get("caller_audio_b64") is not None and not isinstance(value["caller_audio_b64"], str):
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
        if value.get("audio_timestamp") is not None:
            _date_time(value["audio_timestamp"], "audio_timestamp")
        if value.get("language_code") is not None and not isinstance(value["language_code"], str):
            raise ProtocolError("language_code must be a string")
        if value.get("is_single_utterance") is not None and not isinstance(value["is_single_utterance"], bool):
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
        if value.get("name") is not None and not isinstance(value["name"], str):
            raise ProtocolError("event_input name must be a string")
        if value.get("parameters") is not None:
            _object(value["parameters"], "event_input parameters")


def timestamp() -> str:
    """Return an RFC 3339 UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _validate_text_content(value: Any, name: str) -> None:
    content = _object(value, name)
    _only_keys(content, {"text", "ssml", "language_code"}, name)
    present = [field for field in ("text", "ssml") if field in content]
    if len(present) != 1 or not isinstance(content[present[0]], str):
        raise ProtocolError(f"{name} must contain exactly one string text or ssml field")
    if content.get("language_code") is not None and not isinstance(content["language_code"], str):
        raise ProtocolError(f"{name}.language_code must be a string")


def _validate_response_payload(payload: dict[str, Any]) -> None:
    _only_keys(payload, RESPONSE_KEYS, "response payload")
    if payload.get("response_type") not in RESPONSE_TYPES:
        raise ProtocolError("response_type must be PARTIAL, CHUNK, or FINAL")
    if payload.get("input_mode") is not None and payload["input_mode"] not in INPUT_MODES:
        raise ProtocolError("input_mode contains an invalid value")
    for name in ("input_sensitive", "disable_prompt_cancellation"):
        if payload.get(name) is not None and not isinstance(payload[name], bool):
            raise ProtocolError(f"{name} must be a boolean")
    if payload.get("input_handling_config") is not None:
        config = _object(payload["input_handling_config"], "input_handling_config")
        _only_keys(config, {"dtmf_config", "speech_timers"}, "input_handling_config")
        if config.get("dtmf_config") is not None:
            dtmf = _object(config["dtmf_config"], "dtmf_config")
            _only_keys(
                dtmf,
                {"inter_digit_timeout_msec", "termchar", "dtmf_input_length"},
                "dtmf_config",
            )
            for name in ("inter_digit_timeout_msec", "dtmf_input_length"):
                if dtmf.get(name) is not None and (
                    not isinstance(dtmf[name], int) or isinstance(dtmf[name], bool)
                ):
                    raise ProtocolError(f"dtmf_config.{name} must be an integer")
            if dtmf.get("termchar") is not None and dtmf["termchar"] not in DTMF_DIGITS:
                raise ProtocolError("dtmf_config.termchar contains an invalid digit")
        if config.get("speech_timers") is not None:
            timers = _object(config["speech_timers"], "speech_timers")
            timer_fields = {
                "max_speech_timeout_msec",
                "complete_timeout_msec",
                "incomplete_timeout_msec",
            }
            _only_keys(timers, timer_fields, "speech_timers")
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in timers.values()
            ):
                raise ProtocolError("speech_timers values must be integers")
    for name in ("session_transcript", "session_summary"):
        if payload.get(name) is not None:
            _validate_text_content(payload[name], name)

    prompts = payload.get("prompts")
    if prompts is not None:
        if not isinstance(prompts, list):
            raise ProtocolError("prompts must be an array")
        for prompt_value in prompts:
            prompt = _object(prompt_value, "prompt")
            _only_keys(prompt, PROMPT_KEYS, "prompt")
            for field in ("audio_content_b64", "audio_uri", "text"):
                if prompt.get(field) is not None and not isinstance(prompt[field], str):
                    raise ProtocolError(f"prompt.{field} must be a string")
            if not isinstance(prompt.get("audio_content_b64"), str) and not isinstance(prompt.get("text"), str):
                raise ProtocolError("prompt must contain text or audio_content_b64")
            if prompt.get("is_barge_in_enabled") is not None and not isinstance(
                prompt["is_barge_in_enabled"], bool
            ):
                raise ProtocolError("prompt.is_barge_in_enabled must be a boolean")

    events = payload.get("output_events")
    if events is not None:
        if not isinstance(events, list):
            raise ProtocolError("output_events must be an array")
        for event_value in events:
            event = _object(event_value, "output_event")
            _only_keys(event, OUTPUT_EVENT_KEYS, "output_event")
            if event.get("event_type") not in OUTPUT_EVENT_TYPES:
                raise ProtocolError("output_event contains an invalid event_type")
            if event.get("name") is not None and not isinstance(event["name"], str):
                raise ProtocolError("output_event.name must be a string")
            if event.get("metadata") is not None:
                _object(event["metadata"], "output_event.metadata")


def response(seq: int, conversation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Build a schema-conformant response envelope."""
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 1:
        raise ProtocolError("response seq must be a positive integer")
    if not isinstance(conversation_id, str) or not conversation_id:
        raise ProtocolError("response conversation_id must be a nonempty string")
    _validate_response_payload(payload)
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
    if ping.get("metadata") is not None:
        _object(ping["metadata"], "PING metadata")
    if ping.get("conversation_id") != conversation_id:
        raise ProtocolError("PING conversation_id does not match session")
    return {"type": "PONG", "seq": seq, "ts": timestamp(), "conversation_id": conversation_id}
