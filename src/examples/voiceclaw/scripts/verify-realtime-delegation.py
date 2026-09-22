#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Verify arbitrary agent delegation through VoiceClaw's public Realtime socket."""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import math
import os
import ssl
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlsplit

from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from voiceclaw.domain import (
    REALTIME_PROJECTION_SCHEMA,
    ResponseOnlyRequestState,
    ResponseOnlyResultEventKind,
    ResponseOnlySpeechSource,
    ResponseOnlyUpdateKind,
)
from voiceclaw.ports.runtime import FrontendResponsePurpose

_PROJECTION_ENVELOPE_FIELDS = frozenset({"schema", "session_id", "kind", "phase", "title", "request_summary"})
_BACKEND_RESPONSE_IDENTITY_FIELDS = frozenset({"backend_session_id", "turn_id", "response_id"})
_MAX_TERMINAL_DIAGNOSTIC_BYTES = 80
_TERMINAL_DIAGNOSTIC_PUNCTUATION = frozenset("._:-")
_PLAYBACK_RECEIPT_PROTOCOL = "conversation.item.truncate.v1"
_PLAYBACK_RECEIPT_METADATA_FIELDS = frozenset(
    {
        "voiceclaw_presentation_id",
        "voiceclaw_playback_receipt",
        "voiceclaw_playback_receipt_id",
        "voiceclaw_playback_receipt_required",
    }
)
_FRONTEND_SPEECH_PURPOSES = frozenset(purpose.value for purpose in FrontendResponsePurpose)
_MAX_PUBLIC_IDENTIFIER_CHARACTERS = 512


class _PlaybackReceiptContract(NamedTuple):
    """Server-owned metadata authorizing one standard zero-playback receipt."""

    response_id: str
    presentation_id: str
    receipt_id: str


class _AudioTarget(NamedTuple):
    """Public standard Realtime identity for one generated audio part."""

    item_id: str
    content_index: int


def _public_identifier(value: object, label: str) -> str:
    """Validate one bounded public correlation identity."""
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_PUBLIC_IDENTIFIER_CHARACTERS
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise RuntimeError(f"the playback receipt {label} is invalid")
    return value


def _playback_receipt_contract(
    response_id: str,
    metadata: object,
) -> _PlaybackReceiptContract:
    """Parse the complete server-advertised standard playback receipt contract."""
    if not isinstance(metadata, Mapping):
        raise RuntimeError("the frontend speech response omitted playback receipt metadata")
    missing = sorted(field for field in _PLAYBACK_RECEIPT_METADATA_FIELDS if field not in metadata)
    if missing:
        raise RuntimeError("the frontend speech response omitted playback receipt metadata: " + ", ".join(missing))
    if metadata.get("voiceclaw_playback_receipt") != _PLAYBACK_RECEIPT_PROTOCOL:
        raise RuntimeError("the frontend speech response advertised an unsupported playback receipt protocol")
    if metadata.get("voiceclaw_playback_receipt_required") != "true":
        raise RuntimeError("the frontend speech response advertised a malformed playback receipt requirement")
    return _PlaybackReceiptContract(
        response_id=_public_identifier(response_id, "response identity"),
        presentation_id=_public_identifier(
            metadata.get("voiceclaw_presentation_id"),
            "presentation identity",
        ),
        receipt_id=_public_identifier(metadata.get("voiceclaw_playback_receipt_id"), "token"),
    )


def _json_object_without_duplicate_receipt_metadata(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate receipt fields instead of accepting JSON's last value."""
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value and key in _PLAYBACK_RECEIPT_METADATA_FIELDS:
            raise RuntimeError(f"VoiceClaw sent duplicate playback receipt metadata: {key}")
        value[key] = item
    return value


class _PlaybackReceiptTracker:
    """Validate speech receipt authority and correlate standard zero-playback acks."""

    def __init__(self) -> None:
        self._contracts: dict[str, _PlaybackReceiptContract] = {}
        self._response_by_receipt_id: dict[str, str] = {}
        self._audio_targets: dict[str, _AudioTarget] = {}
        self._audio_done: set[str] = set()
        self._sent_responses: set[str] = set()
        self._pending_by_target: dict[_AudioTarget, str] = {}
        self._acknowledged_responses: set[str] = set()
        self._acknowledged_targets: set[_AudioTarget] = set()

    def register_response(self, response_id: str, metadata: object) -> None:
        """Register exactly one unique receipt token for a speech response."""
        if response_id in self._contracts:
            raise RuntimeError("the frontend emitted duplicate playback receipt metadata for one response")
        contract = _playback_receipt_contract(response_id, metadata)
        owner = self._response_by_receipt_id.get(contract.receipt_id)
        if owner is not None:
            raise RuntimeError("the frontend reused a playback receipt token across responses")
        self._contracts[response_id] = contract
        self._response_by_receipt_id[contract.receipt_id] = response_id

    def validate_terminal_metadata(self, response_id: str, metadata: object) -> None:
        """Require response.done to repeat, not replace, its receipt authority."""
        expected = self._contracts.get(response_id)
        if expected is None:
            raise RuntimeError("the completed frontend speech response had no playback receipt contract")
        if _playback_receipt_contract(response_id, metadata) != expected:
            raise RuntimeError("the completed frontend speech response changed its playback receipt metadata")

    def has_response(self, response_id: str) -> bool:
        """Return whether a speech response advertised a receipt contract."""
        return response_id in self._contracts

    def observe_audio_delta(self, event: Mapping[str, object]) -> None:
        """Bind a receipt to the first public standard output-audio identity."""
        response_id = event.get("response_id")
        if not isinstance(response_id, str) or response_id not in self._contracts:
            return
        if not isinstance(event.get("delta"), str) or not event["delta"]:
            raise RuntimeError("the frontend speech response emitted an invalid output-audio delta")
        target = self._audio_target(event)
        existing = self._audio_targets.setdefault(response_id, target)
        if existing != target:
            raise RuntimeError("the frontend speech response changed its public audio identity")
        if response_id in self._audio_done:
            raise RuntimeError("the frontend speech response emitted audio after output-audio completion")

    def observe_audio_done(self, event: Mapping[str, object]) -> None:
        """Close only the exact public audio part established by a prior delta."""
        response_id = event.get("response_id")
        if not isinstance(response_id, str) or response_id not in self._contracts:
            return
        target = self._audio_target(event)
        if self._audio_targets.get(response_id) != target:
            raise RuntimeError("the frontend speech response completed an unrecognized public audio part")
        if response_id in self._audio_done:
            raise RuntimeError("the frontend speech response emitted duplicate output-audio completion")
        self._audio_done.add(response_id)

    def zero_playback_event(self, response_id: str) -> dict[str, object]:
        """Create one token-authorized standard truncate at the truthful zero boundary."""
        contract = self._contracts.get(response_id)
        target = self._audio_targets.get(response_id)
        if contract is None:
            raise RuntimeError("the completed frontend speech response had no playback receipt contract")
        if target is None or response_id not in self._audio_done:
            raise RuntimeError("the completed frontend speech response omitted its public output-audio boundary")
        if response_id in self._sent_responses:
            raise RuntimeError("the verifier attempted a duplicate playback receipt")
        if target in self._pending_by_target or target in self._acknowledged_targets:
            raise RuntimeError("the frontend reused a public audio identity for playback receipts")
        self._sent_responses.add(response_id)
        self._pending_by_target[target] = response_id
        return {
            "event_id": contract.receipt_id,
            "type": "conversation.item.truncate",
            "item_id": target.item_id,
            "content_index": target.content_index,
            "audio_end_ms": 0,
        }

    def acknowledge(self, event: Mapping[str, object]) -> str:
        """Acknowledge only the exact public zero-boundary truncation tuple."""
        target = self._audio_target(event)
        audio_end_ms = event.get("audio_end_ms")
        if isinstance(audio_end_ms, bool) or not isinstance(audio_end_ms, int) or audio_end_ms != 0:
            raise RuntimeError("VoiceClaw acknowledged an invalid zero-playback boundary")
        response_id = self._pending_by_target.pop(target, None)
        if response_id is None:
            if target in self._acknowledged_targets:
                raise RuntimeError("VoiceClaw acknowledged a playback receipt more than once")
            raise RuntimeError("VoiceClaw acknowledged an uncorrelated playback receipt")
        self._acknowledged_targets.add(target)
        self._acknowledged_responses.add(response_id)
        return response_id

    def all_acknowledged(self, response_ids: set[str]) -> bool:
        """Return whether every required completed speech response has a matching ack."""
        return bool(response_ids) and response_ids <= self._acknowledged_responses

    @property
    def acknowledged_count(self) -> int:
        """Return the number of exact zero-playback receipts acknowledged by VoiceClaw."""
        return len(self._acknowledged_responses)

    @staticmethod
    def _audio_target(event: Mapping[str, object]) -> _AudioTarget:
        item_id = _public_identifier(event.get("item_id"), "public audio item identity")
        content_index = event.get("content_index")
        if isinstance(content_index, bool) or not isinstance(content_index, int) or content_index < 0:
            raise RuntimeError("the playback receipt public audio content index is invalid")
        return _AudioTarget(item_id=item_id, content_index=content_index)


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number of seconds")
    return seconds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Send one arbitrary typed turn through VoiceClaw's public WSS endpoint and require the "
            "frontend to delegate it and produce the asynchronous response-only lifecycle."
        ),
    )
    parser.add_argument(
        "--url",
        default=os.getenv("VOICECLAW_REALTIME_URL", "wss://127.0.0.1:7860/v1/realtime"),
        help="public VoiceClaw WSS URL (default: %(default)s)",
    )
    parser.add_argument("--query", required=True, help="arbitrary finalized user request to delegate")
    parser.add_argument(
        "--client-secret-file",
        type=Path,
        help="optional file containing a short-lived VoiceClaw ek_ client secret",
    )
    parser.add_argument(
        "--ca-file",
        type=Path,
        help="CA certificate for the public VoiceClaw TLS endpoint",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="disable TLS verification for a local self-signed development endpoint",
    )
    parser.add_argument(
        "--allow-loopback-ws",
        action="store_true",
        help="allow plaintext ws only for a literal loopback artifact test endpoint",
    )
    parser.add_argument(
        "--forbid-value-file",
        action="append",
        default=[],
        type=Path,
        help="fail if the exact UTF-8 value from this file appears in any received public event",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_seconds,
        default=120.0,
        help="overall completion timeout in seconds (default: %(default)s)",
    )
    return parser


def _validate_url(raw_url: str, *, allow_loopback_ws: bool = False) -> str:
    url = raw_url.strip()
    parsed = urlsplit(url)
    if not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("--url must have a host and no user information or fragment")
    if parsed.scheme == "ws":
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = False
        if not allow_loopback_ws or not loopback:
            raise ValueError("plaintext ws requires --allow-loopback-ws and a literal loopback host")
    elif parsed.scheme != "wss":
        raise ValueError("--url must use wss")
    return url


def _client_secret(arguments: argparse.Namespace) -> str:
    if arguments.client_secret_file is not None:
        secret = arguments.client_secret_file.read_text(encoding="utf-8").strip()
    else:
        secret = os.getenv("VOICECLAW_CLIENT_SECRET", "").strip()
    if secret and (
        not secret.startswith("ek_") or len(secret) > 4096 or any(character.isspace() for character in secret)
    ):
        raise ValueError("the VoiceClaw client secret must be one whitespace-free ek_ value")
    return secret


def _forbidden_values(arguments: argparse.Namespace) -> tuple[str, ...]:
    values: list[str] = []
    for path in arguments.forbid_value_file:
        value = path.read_text(encoding="utf-8").strip()
        if not value or len(value) > 4096 or any(character.isspace() for character in value):
            raise ValueError("each forbidden-value file must contain one bounded, whitespace-free UTF-8 value")
        if value not in values:
            values.append(value)
    return tuple(values)


def _tls_context(arguments: argparse.Namespace, url: str) -> ssl.SSLContext | None:
    if urlsplit(url).scheme == "ws":
        if arguments.insecure or arguments.ca_file is not None:
            raise ValueError("TLS options cannot be used with loopback ws")
        return None
    if arguments.insecure and arguments.ca_file is not None:
        raise ValueError("--insecure and --ca-file are mutually exclusive")
    if arguments.insecure:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
    if arguments.ca_file is not None:
        return ssl.create_default_context(cafile=str(arguments.ca_file))
    return ssl.create_default_context()


def _event_id() -> str:
    return f"event_smoke_{uuid.uuid4().hex}"


def _item_id() -> str:
    return f"item_smoke_{uuid.uuid4().hex}"


def _projection(metadata: object) -> dict[str, str] | None:
    if not isinstance(metadata, dict):
        return None
    schema = metadata.get("voiceclaw_schema")
    if schema is None:
        return None
    if schema != REALTIME_PROJECTION_SCHEMA:
        raise RuntimeError(f"VoiceClaw sent an unsupported projection schema: {schema!r}")
    projection = {
        key.removeprefix("voiceclaw_"): value
        for key, value in metadata.items()
        if key.startswith("voiceclaw_") and isinstance(value, str)
    }
    missing = [field for field in ("session_id", "kind", "phase", "title") if not projection.get(field)]
    if missing:
        raise RuntimeError("VoiceClaw sent a projection without required metadata: " + ", ".join(missing))
    return projection


def _projection_correlation(projection: Mapping[str, str]) -> dict[str, str]:
    """Return only application correlation from one projection envelope."""
    return {key: value for key, value in projection.items() if key not in _PROJECTION_ENVELOPE_FIELDS}


def _response_content_text(response: object, *, audio: bool) -> str:
    if not isinstance(response, dict):
        return ""
    parts: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            content_type = content.get("type")
            if audio and content_type not in {"audio", "output_audio"}:
                continue
            if not audio and content_type in {"audio", "output_audio"}:
                continue
            text = content.get("transcript") if audio else content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _audio_transcript(response: object, streams: Mapping[str, str]) -> str:
    streamed = streams.get("output_audio_transcript", "").strip()
    return streamed or _response_content_text(response, audio=True).strip()


def _output_text(response: object, streams: Mapping[str, str]) -> str:
    streamed = streams.get("output_text", "").strip()
    return streamed or _response_content_text(response, audio=False).strip()


def _response_text(response: object, streams: Mapping[str, str]) -> str:
    return _audio_transcript(response, streams) or _output_text(response, streams)


def _spoken_response_text(response: object, streams: Mapping[str, str], *, purpose: str) -> str:
    transcript = _audio_transcript(response, streams)
    if not transcript:
        raise RuntimeError(f"the frontend {purpose} completed without an audio transcript")
    return transcript


def _queue_depth(projection: Mapping[str, str], text: str) -> int | None:
    candidate: object = projection.get("queue_depth")
    if candidate is None and text:
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            body = None
        if isinstance(body, dict):
            candidate = body.get("queue_depth")
    if isinstance(candidate, bool):
        return None
    try:
        depth = int(candidate) if candidate is not None else None
    except (TypeError, ValueError):
        return None
    return depth if depth is not None and depth >= 0 else None


def _metadata_true(value: object) -> bool:
    return value is True or value == "true"


def _bounded_terminal_diagnostic(value: object) -> str | None:
    """Return one bounded identifier-like terminal detail without echoing prose."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if (
        not candidate
        or len(candidate.encode("utf-8")) > _MAX_TERMINAL_DIAGNOSTIC_BYTES
        or any(
            not (character.isascii() and (character.isalnum() or character in _TERMINAL_DIAGNOSTIC_PUNCTUATION))
            for character in candidate
        )
    ):
        return None
    return candidate


def _noncompleted_frontend_response_message(
    response: Mapping[str, object],
    *,
    speech_purpose: object,
) -> str:
    """Describe a non-completed speech response using only bounded standard fields."""
    labels = {
        FrontendResponsePurpose.DELEGATION_ACK.value: "delegation acknowledgement",
        FrontendResponsePurpose.RESULT_DELIVERY.value: "result delivery",
        FrontendResponsePurpose.FAILURE_DELIVERY.value: "failure delivery",
    }
    purpose = labels.get(speech_purpose, "frontend response")
    evidence: list[str] = []
    status = _bounded_terminal_diagnostic(response.get("status"))
    if status is not None:
        evidence.append(f"status={status}")
    status_details = response.get("status_details")
    if isinstance(status_details, Mapping):
        detail_type = _bounded_terminal_diagnostic(status_details.get("type"))
        reason = _bounded_terminal_diagnostic(status_details.get("reason"))
        if detail_type is not None:
            evidence.append(f"type={detail_type}")
        if reason is not None:
            evidence.append(f"reason={reason}")
    suffix = f" ({', '.join(evidence)})" if evidence else ""
    return f"the frontend {purpose} did not complete{suffix}"


def _delivery_completion(
    *,
    speech_source: ResponseOnlySpeechSource | None,
    succeeded_at: int | None,
    acknowledgement_completed_at: int | None,
    result_delivery_response_id: str | None,
    result_delivery_completed_at: int | None,
    queue_depth_transitions: list[tuple[int, int]],
) -> tuple[int, int] | None:
    """Return the drained-queue event once the advertised delivery path is terminal."""
    if speech_source is None or succeeded_at is None or acknowledgement_completed_at is None:
        return None
    if speech_source is ResponseOnlySpeechSource.NONE:
        if result_delivery_response_id is not None or result_delivery_completed_at is not None:
            raise RuntimeError("VoiceClaw delivered result speech after advertising a display-only result")
        delivery_boundary = max(succeeded_at, acknowledgement_completed_at)
    elif speech_source is ResponseOnlySpeechSource.BACKEND_AUTHORED:
        if result_delivery_response_id is None or result_delivery_completed_at is None:
            return None
        delivery_boundary = result_delivery_completed_at
    else:  # pragma: no cover - keeps future enum additions fail closed
        raise RuntimeError(f"unsupported result speech source: {speech_source.value}")
    if not queue_depth_transitions or queue_depth_transitions[-1][0] <= delivery_boundary:
        return None
    queue_drained_at, final_queue_depth = queue_depth_transitions[-1]
    if final_queue_depth != 0:
        raise RuntimeError("the speech delivery queue did not drain after its final delivery")
    if speech_source is ResponseOnlySpeechSource.BACKEND_AUTHORED and not any(
        succeeded_at < observed_at < result_delivery_completed_at and depth > 0
        for observed_at, depth in queue_depth_transitions
    ):
        raise RuntimeError("the result speech slot was not projected before result delivery")
    return queue_drained_at, final_queue_depth


async def _send(websocket: Any, event: dict[str, object]) -> None:
    value = {"event_id": _event_id(), **event}
    await websocket.send(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


async def _verify(arguments: argparse.Namespace) -> dict[str, object]:
    url = _validate_url(arguments.url, allow_loopback_ws=arguments.allow_loopback_ws)
    secret = _client_secret(arguments)
    forbidden_values = _forbidden_values(arguments)
    protocols = ["realtime"]
    if secret:
        protocols.append(f"openai-insecure-api-key.{secret}")

    session_id: str | None = None
    advertised_turn_detection: dict[str, object] | None = None
    negotiated_turn_detection: dict[str, object] | None = None
    session_updated = False
    gateway_reachable = False
    query_submitted = False
    projection_by_response: dict[str, dict[str, str]] = {}
    metadata_by_response: dict[str, dict[str, object]] = {}
    streams_by_response: dict[str, dict[str, str]] = {}
    observed: list[dict[str, str]] = []
    backend_turn_phases: list[ResponseOnlyRequestState] = []
    backend_phase_completed_at: dict[ResponseOnlyRequestState, int] = {}
    backend_request_id: str | None = None
    backend_call_id: str | None = None
    backend_target_identity: tuple[str | None, str | None, str | None] | None = None
    backend_base_correlation: dict[str, str] | None = None
    result_display_response_id: str | None = None
    result_display_correlation: dict[str, str] | None = None
    result_display_deltas: list[str] = []
    result_display_done_text: str | None = None
    result_display_started_at: int | None = None
    result_display_completed_at: int | None = None
    event_sequence = 0
    queued_speech_depth = 0
    queue_depth_transitions: list[tuple[int, int]] = []
    queue_reserved_before_dispatch_at: int | None = None
    acknowledgement_response_id: str | None = None
    acknowledgement_created_at: int | None = None
    acknowledgement_transcript = ""
    acknowledgement_completed_at: int | None = None
    succeeded_projection: dict[str, str] | None = None
    result_speech_source: ResponseOnlySpeechSource | None = None
    backend_result = ""
    result_delivery_response_id: str | None = None
    result_speech_transcript: str | None = None
    result_delivery_completed_at: int | None = None
    playback_receipts = _PlaybackReceiptTracker()
    required_playback_response_ids: set[str] = set()
    pending_success_result: dict[str, object] | None = None

    async with connect(
        url,
        ssl=_tls_context(arguments, url),
        subprotocols=protocols,
        max_size=16 * 1024 * 1024,
        open_timeout=min(arguments.timeout, 30.0),
    ) as websocket:
        async with asyncio.timeout(arguments.timeout):
            while True:
                raw_event = await websocket.recv()
                if not isinstance(raw_event, str):
                    raise RuntimeError("VoiceClaw sent a binary event on its JSON Realtime channel")
                if any(value in raw_event for value in forbidden_values):
                    raise RuntimeError("VoiceClaw exposed a forbidden private value in a public Realtime event")
                event = json.loads(raw_event, object_pairs_hook=_json_object_without_duplicate_receipt_metadata)
                if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                    raise RuntimeError("VoiceClaw sent an invalid Realtime event")

                event_sequence += 1
                event_type = event["type"]
                if event_type == "error":
                    error = event.get("error")
                    if isinstance(error, dict):
                        detail = error.get("message", error.get("code"))
                        evidence = ", ".join(
                            f"{key}={error[key]}"
                            for key in ("code", "param", "event_id")
                            if isinstance(error.get(key), str) and error[key]
                        )
                        if evidence:
                            detail = f"{detail or 'unknown error'} ({evidence})"
                    else:
                        detail = error
                    raise RuntimeError(f"VoiceClaw protocol error: {detail or 'unknown error'}")

                if event_type == "session.created":
                    session = event.get("session")
                    if not isinstance(session, dict) or not isinstance(session.get("id"), str):
                        raise RuntimeError("session.created did not contain a session identity")
                    session_id = session["id"]
                    created_audio = session.get("audio")
                    created_input = created_audio.get("input") if isinstance(created_audio, dict) else None
                    created_turn_detection = (
                        created_input.get("turn_detection") if isinstance(created_input, dict) else None
                    )
                    if isinstance(created_turn_detection, dict):
                        advertised_turn_detection = dict(created_turn_detection)
                    await _send(
                        websocket,
                        {
                            "type": "session.update",
                            "session": {
                                "type": "realtime",
                                "output_modalities": ["audio"],
                                "audio": {
                                    "input": {
                                        "format": {"type": "audio/pcm", "rate": 24_000},
                                    },
                                    "output": {
                                        "format": {"type": "audio/pcm", "rate": 24_000},
                                    },
                                },
                            },
                        },
                    )

                elif event_type == "session.updated":
                    session = event.get("session")
                    audio = session.get("audio") if isinstance(session, dict) else None
                    input_audio = audio.get("input") if isinstance(audio, dict) else None
                    updated_turn_detection = (
                        input_audio.get("turn_detection") if isinstance(input_audio, dict) else None
                    )
                    if advertised_turn_detection is not None:
                        if not isinstance(updated_turn_detection, dict):
                            raise RuntimeError("session.updated discarded advertised automatic turn detection")
                        if updated_turn_detection.get("type") != advertised_turn_detection.get("type"):
                            raise RuntimeError("session.updated changed the advertised turn-detection mode")
                        negotiated_turn_detection = dict(updated_turn_detection)
                    session_updated = True

                elif event_type == "response.created":
                    response = event.get("response")
                    if isinstance(response, dict) and isinstance(response.get("id"), str):
                        response_id = response["id"]
                        metadata = response.get("metadata")
                        if isinstance(metadata, dict):
                            metadata_by_response[response_id] = dict(metadata)
                        streams_by_response.setdefault(response_id, {})
                        projection = _projection(metadata)
                        if projection is not None:
                            projection_by_response[response_id] = projection
                            if projection.get("kind") == ResponseOnlyUpdateKind.RESULT_DISPLAY.value:
                                if projection.get("phase") != ResponseOnlyResultEventKind.DISPLAY_DELTA.value:
                                    raise RuntimeError("the result display did not start in display_delta phase")
                                if result_display_response_id is not None:
                                    raise RuntimeError("VoiceClaw started more than one result display stream")
                                if (
                                    backend_base_correlation is None
                                    or ResponseOnlyRequestState.WAITING_FOR_RESPONSE not in backend_phase_completed_at
                                ):
                                    raise RuntimeError("the result display started before its backend request")
                                if (
                                    response.get("status") != "in_progress"
                                    or response.get("conversation_id") is not None
                                    or response.get("output_modalities") != ["text"]
                                    or response.get("output") != []
                                ):
                                    raise RuntimeError(
                                        "the result display did not use an out-of-band Realtime text response"
                                    )
                                correlation = _projection_correlation(projection)
                                if projection.get("work_id"):
                                    raise RuntimeError("the response-only display claimed a durable Work ID")
                                if any(
                                    correlation.get(key) != value for key, value in backend_base_correlation.items()
                                ):
                                    raise RuntimeError("the result display did not correlate with its backend request")
                                if set(correlation).difference(backend_base_correlation) != (
                                    _BACKEND_RESPONSE_IDENTITY_FIELDS
                                ) or any(not correlation.get(key) for key in _BACKEND_RESPONSE_IDENTITY_FIELDS):
                                    raise RuntimeError("the result display omitted exact backend response correlation")
                                result_display_response_id = response_id
                                result_display_correlation = correlation
                                result_display_started_at = event_sequence
                        elif query_submitted:
                            speech_purpose = metadata_by_response.get(response_id, {}).get("voiceclaw_speech_purpose")
                            if speech_purpose in _FRONTEND_SPEECH_PURPOSES:
                                playback_receipts.register_response(response_id, metadata)
                            if speech_purpose == FrontendResponsePurpose.DELEGATION_ACK.value:
                                if acknowledgement_response_id is not None:
                                    raise RuntimeError("the frontend started more than one delegation acknowledgement")
                                if _metadata_true(metadata_by_response.get(response_id, {}).get("voiceclaw_delivery")):
                                    raise RuntimeError("the delegation acknowledgement was marked as result delivery")
                                acknowledgement_response_id = response_id
                                acknowledgement_created_at = event_sequence
                                print("delegation acknowledgement started", file=sys.stderr)
                            elif speech_purpose == FrontendResponsePurpose.RESULT_DELIVERY.value:
                                if result_speech_source is not ResponseOnlySpeechSource.BACKEND_AUTHORED:
                                    raise RuntimeError(
                                        "the frontend started result speech without advertising backend-authored "
                                        "presentation material"
                                    )
                                if result_delivery_response_id is not None:
                                    raise RuntimeError("the frontend started more than one result delivery")
                                if not _metadata_true(
                                    metadata_by_response.get(response_id, {}).get("voiceclaw_delivery")
                                ):
                                    raise RuntimeError("the result response omitted its delivery marker")
                                result_delivery_response_id = response_id
                                print("result-delivery speech started", file=sys.stderr)
                            else:
                                raise RuntimeError(
                                    "the frontend emitted an unmarked conversational response during delegation"
                                )

                elif event_type == "response.output_audio.delta":
                    playback_receipts.observe_audio_delta(event)

                elif event_type == "response.output_audio.done":
                    playback_receipts.observe_audio_done(event)

                elif event_type in {"response.output_text.delta", "response.output_audio_transcript.delta"}:
                    response_id = event.get("response_id")
                    delta = event.get("delta")
                    if isinstance(response_id, str) and isinstance(delta, str):
                        if response_id == result_display_response_id:
                            if event_type != "response.output_text.delta" or not delta:
                                raise RuntimeError("the result display emitted an invalid Realtime text delta")
                            result_display_deltas.append(delta)
                        stream_name = (
                            "output_audio_transcript"
                            if event_type == "response.output_audio_transcript.delta"
                            else "output_text"
                        )
                        streams = streams_by_response.setdefault(response_id, {})
                        streams[stream_name] = streams.get(stream_name, "") + delta

                elif event_type in {"response.output_text.done", "response.output_audio_transcript.done"}:
                    response_id = event.get("response_id")
                    final_text = event.get(
                        "transcript" if event_type == "response.output_audio_transcript.done" else "text"
                    )
                    if isinstance(response_id, str) and isinstance(final_text, str):
                        if response_id == result_display_response_id:
                            if event_type != "response.output_text.done" or result_display_done_text is not None:
                                raise RuntimeError("the result display emitted an invalid terminal text event")
                            if final_text != "".join(result_display_deltas):
                                raise RuntimeError("the result display terminal text contradicted its deltas")
                            result_display_done_text = final_text
                        stream_name = (
                            "output_audio_transcript"
                            if event_type == "response.output_audio_transcript.done"
                            else "output_text"
                        )
                        streams_by_response.setdefault(response_id, {})[stream_name] = final_text

                elif event_type == "response.done":
                    response = event.get("response")
                    if not isinstance(response, dict) or not isinstance(response.get("id"), str):
                        continue
                    response_id = response["id"]
                    response_metadata = metadata_by_response.get(response_id, {})
                    terminal_metadata = response.get("metadata")
                    if isinstance(terminal_metadata, dict):
                        response_metadata = {**response_metadata, **terminal_metadata}
                    if playback_receipts.has_response(response_id):
                        playback_receipts.validate_terminal_metadata(response_id, terminal_metadata)
                    projection = _projection(response_metadata) or projection_by_response.get(response_id)
                    response_streams = streams_by_response.get(response_id, {})
                    text = _response_text(response, response_streams)

                    if projection is not None:
                        if session_id is not None and projection.get("session_id") != session_id:
                            streams_by_response.pop(response_id, None)
                            metadata_by_response.pop(response_id, None)
                            projection_by_response.pop(response_id, None)
                            continue

                        kind = projection.get("kind", "projection")
                        phase = projection.get("phase", "updated")
                        title = projection.get("title", kind)
                        observation = {"kind": kind, "phase": phase, "title": title}
                        observed.append(observation)
                        print(f"projection {kind}/{phase}: {title}", file=sys.stderr)

                        if kind == ResponseOnlyUpdateKind.RESULT_DISPLAY.value:
                            initial_projection = projection_by_response.get(response_id)
                            if (
                                response_id != result_display_response_id
                                or initial_projection is None
                                or initial_projection.get("phase") != ResponseOnlyResultEventKind.DISPLAY_DELTA.value
                                or phase != ResponseOnlyResultEventKind.COMPLETED.value
                            ):
                                raise RuntimeError(
                                    "the result display did not transition from display_delta to completed"
                                )
                            if result_display_completed_at is not None:
                                raise RuntimeError("VoiceClaw completed more than one result display stream")
                            correlation = _projection_correlation(projection)
                            if correlation != result_display_correlation:
                                raise RuntimeError("the result display changed correlation before completion")
                            if (
                                response.get("status") != "completed"
                                or response.get("conversation_id") is not None
                                or response.get("output_modalities") != ["text"]
                            ):
                                raise RuntimeError(
                                    "the completed result display was not an out-of-band Realtime text response"
                                )
                            if not result_display_deltas or result_display_done_text is None:
                                raise RuntimeError("the result display completed without standard text events")
                            body_text = _response_content_text(response, audio=False)
                            if body_text != result_display_done_text:
                                raise RuntimeError("the result display response body contradicted its text stream")
                            if not body_text.strip():
                                raise RuntimeError("the completed result display was empty")
                            backend_result = body_text
                            result_display_completed_at = event_sequence

                        if kind == ResponseOnlyUpdateKind.BACKEND_TARGET.value:
                            gateway_reachable = phase == "reachable"
                            if phase == "unavailable":
                                raise RuntimeError("the configured agent gateway is unavailable")

                        if kind == ResponseOnlyUpdateKind.DELIVERY_QUEUE.value:
                            depth = _queue_depth(projection, text)
                            if depth is None:
                                raise RuntimeError("the speech delivery queue projection did not contain a valid depth")
                            queued_speech_depth = max(queued_speech_depth, depth)
                            if not queue_depth_transitions or queue_depth_transitions[-1][1] != depth:
                                queue_depth_transitions.append((event_sequence, depth))
                            if (
                                depth > 0
                                and ResponseOnlyRequestState.DISPATCHING not in backend_phase_completed_at
                                and queue_reserved_before_dispatch_at is None
                            ):
                                queue_reserved_before_dispatch_at = event_sequence

                        if kind == ResponseOnlyUpdateKind.BACKEND_TURN.value:
                            try:
                                request_state = ResponseOnlyRequestState(phase)
                            except ValueError as error:
                                raise RuntimeError(
                                    f"the response-only lifecycle used an unknown phase: {phase!r}"
                                ) from error
                            if projection.get("work_id"):
                                raise RuntimeError("the response-only backend projection claimed a durable Work ID")
                            local_request_id = projection.get("local_request_id")
                            if not local_request_id or projection.get("commit_id") != local_request_id:
                                raise RuntimeError(
                                    "the response-only lifecycle omitted its VoiceClaw-local request identity"
                                )
                            if projection.get("identity_authority") != "voiceclaw_local":
                                raise RuntimeError("the response-only lifecycle claimed a non-local identity authority")
                            correlation = _projection_correlation(projection)
                            if backend_request_id is None:
                                backend_request_id = local_request_id
                                backend_call_id = projection.get("call_id")
                                backend_target_identity = (
                                    projection.get("backend_name"),
                                    projection.get("backend_mode"),
                                    projection.get("target_ref"),
                                )
                            elif (
                                local_request_id != backend_request_id
                                or projection.get("call_id") != backend_call_id
                                or (
                                    projection.get("backend_name"),
                                    projection.get("backend_mode"),
                                    projection.get("target_ref"),
                                )
                                != backend_target_identity
                            ):
                                raise RuntimeError(
                                    "the response-only lifecycle changed correlation identity between phases"
                                )
                            if request_state in {
                                ResponseOnlyRequestState.LOCALLY_QUEUED,
                                ResponseOnlyRequestState.DISPATCHING,
                                ResponseOnlyRequestState.WAITING_FOR_RESPONSE,
                            }:
                                if backend_base_correlation is None:
                                    backend_base_correlation = correlation
                                elif correlation != backend_base_correlation:
                                    raise RuntimeError(
                                        "the response-only lifecycle changed request correlation before completion"
                                    )
                            if request_state is ResponseOnlyRequestState.FAILED:
                                raise RuntimeError(text or title or "the delegated backend turn failed")
                            backend_turn_phases.append(request_state)
                            backend_phase_completed_at.setdefault(request_state, event_sequence)

                            if (
                                request_state is ResponseOnlyRequestState.DISPATCHING
                                and queue_reserved_before_dispatch_at is None
                            ):
                                raise RuntimeError(
                                    "the acknowledgement speech slot was not reserved before backend dispatch"
                                )

                            if request_state is ResponseOnlyRequestState.SUCCEEDED:
                                required_phases = (
                                    ResponseOnlyRequestState.LOCALLY_QUEUED,
                                    ResponseOnlyRequestState.DISPATCHING,
                                    ResponseOnlyRequestState.WAITING_FOR_RESPONSE,
                                    ResponseOnlyRequestState.SUCCEEDED,
                                )
                                if tuple(backend_turn_phases) != required_phases:
                                    raise RuntimeError(
                                        "the delegated turn lifecycle did not follow the required order: "
                                        + " -> ".join(state.value for state in required_phases)
                                    )
                                missing_backend_identity = [
                                    key
                                    for key in ("backend_session_id", "turn_id", "response_id")
                                    if not projection.get(key)
                                ]
                                if missing_backend_identity:
                                    raise RuntimeError(
                                        "the succeeded backend turn omitted response correlation: "
                                        + ", ".join(missing_backend_identity)
                                    )
                                if acknowledgement_created_at is None or acknowledgement_created_at >= event_sequence:
                                    raise RuntimeError(
                                        "the delegation acknowledgement did not start before backend success"
                                    )
                                if (
                                    result_display_completed_at is None
                                    or result_display_completed_at >= event_sequence
                                    or result_display_correlation is None
                                ):
                                    raise RuntimeError("the backend succeeded before its result display was committed")
                                speech_source_value = correlation.get("speech_source")
                                try:
                                    advertised_speech_source = ResponseOnlySpeechSource(speech_source_value)
                                except (TypeError, ValueError) as error:
                                    raise RuntimeError(
                                        "the succeeded backend turn omitted its typed speech source"
                                    ) from error
                                result_correlation = {
                                    key: value for key, value in correlation.items() if key != "speech_source"
                                }
                                if result_correlation != result_display_correlation:
                                    raise RuntimeError(
                                        "the succeeded backend turn did not match its result display correlation"
                                    )
                                result_speech_source = advertised_speech_source
                                succeeded_projection = projection

                    else:
                        status = response.get("status")
                        speech_purpose = response_metadata.get("voiceclaw_speech_purpose")
                        is_delivery = _metadata_true(response_metadata.get("voiceclaw_delivery"))
                        if status != "completed":
                            raise RuntimeError(
                                _noncompleted_frontend_response_message(
                                    response,
                                    speech_purpose=speech_purpose,
                                )
                            )
                        if speech_purpose == FrontendResponsePurpose.RESULT_DELIVERY.value:
                            if result_speech_source is not ResponseOnlySpeechSource.BACKEND_AUTHORED:
                                raise RuntimeError(
                                    "the frontend completed result speech without advertising backend-authored "
                                    "presentation material"
                                )
                            if response_id != result_delivery_response_id or not is_delivery:
                                raise RuntimeError("the frontend completed an uncorrelated result delivery")
                            succeeded_at = backend_phase_completed_at.get(ResponseOnlyRequestState.SUCCEEDED)
                            if succeeded_projection is None or succeeded_at is None or event_sequence <= succeeded_at:
                                raise RuntimeError("result speech completed before the backend result was committed")
                            text = _spoken_response_text(
                                response,
                                response_streams,
                                purpose="result delivery",
                            )
                            if (
                                acknowledgement_completed_at is None
                                or not acknowledgement_transcript
                                or acknowledgement_completed_at >= event_sequence
                            ):
                                raise RuntimeError(
                                    "the delegation acknowledgement did not complete before result delivery"
                                )
                            result_speech_transcript = text
                            result_delivery_completed_at = event_sequence
                            print("result-delivery speech completed", file=sys.stderr)
                        elif speech_purpose == FrontendResponsePurpose.DELEGATION_ACK.value:
                            if response_id != acknowledgement_response_id or is_delivery:
                                raise RuntimeError("the frontend completed an uncorrelated delegation acknowledgement")
                            if acknowledgement_completed_at is not None:
                                raise RuntimeError("the frontend emitted more than one delegation acknowledgement")
                            text = _spoken_response_text(
                                response,
                                response_streams,
                                purpose="delegation acknowledgement",
                            )
                            for required_phase in (
                                ResponseOnlyRequestState.DISPATCHING,
                                ResponseOnlyRequestState.WAITING_FOR_RESPONSE,
                            ):
                                completed_at = backend_phase_completed_at.get(required_phase)
                                if completed_at is None or completed_at >= event_sequence:
                                    raise RuntimeError(
                                        "backend dispatch was blocked on delegation acknowledgement speech"
                                    )
                            acknowledgement_transcript = text
                            acknowledgement_completed_at = event_sequence
                            print("delegation acknowledgement completed", file=sys.stderr)
                        elif query_submitted:
                            raise RuntimeError(
                                "the frontend emitted an unmarked conversational response during delegation"
                            )

                    if playback_receipts.has_response(response_id):
                        receipt_event = playback_receipts.zero_playback_event(response_id)
                        await _send(websocket, receipt_event)
                        print(f"zero-playback receipt sent for {response_id}", file=sys.stderr)

                    streams_by_response.pop(response_id, None)
                    metadata_by_response.pop(response_id, None)
                    projection_by_response.pop(response_id, None)

                    delivery_completion = _delivery_completion(
                        speech_source=result_speech_source,
                        succeeded_at=backend_phase_completed_at.get(ResponseOnlyRequestState.SUCCEEDED),
                        acknowledgement_completed_at=acknowledgement_completed_at,
                        result_delivery_response_id=result_delivery_response_id,
                        result_delivery_completed_at=result_delivery_completed_at,
                        queue_depth_transitions=queue_depth_transitions,
                    )
                    if delivery_completion is not None:
                        assert succeeded_projection is not None
                        local_receipt_at = backend_phase_completed_at[ResponseOnlyRequestState.LOCALLY_QUEUED]
                        dispatching_at = backend_phase_completed_at[ResponseOnlyRequestState.DISPATCHING]
                        waiting_at = backend_phase_completed_at[ResponseOnlyRequestState.WAITING_FOR_RESPONSE]
                        succeeded_at = backend_phase_completed_at[ResponseOnlyRequestState.SUCCEEDED]
                        assert queue_reserved_before_dispatch_at is not None
                        assert acknowledgement_created_at is not None
                        assert acknowledgement_completed_at is not None
                        assert result_display_started_at is not None
                        assert result_display_completed_at is not None
                        assert result_speech_source is not None
                        queue_drained_at, final_queue_depth = delivery_completion
                        pending_success_result = {
                            "status": "succeeded",
                            "query": arguments.query,
                            "session_id": session_id,
                            "turn_detection": (
                                negotiated_turn_detection.get("type")
                                if negotiated_turn_detection is not None
                                else "manual"
                            ),
                            "backend": succeeded_projection.get("backend_name"),
                            "backend_mode": succeeded_projection.get("backend_mode"),
                            "target_ref": succeeded_projection.get("target_ref"),
                            "local_request_id": succeeded_projection.get("local_request_id"),
                            "turn_id": succeeded_projection.get("turn_id"),
                            "backend_response_id": succeeded_projection.get("response_id"),
                            "result": backend_result,
                            "result_display_delta_count": len(result_display_deltas),
                            "acknowledgement_transcript": acknowledgement_transcript,
                            "speech_source": result_speech_source.value,
                            "result_speech_transcript": result_speech_transcript,
                            "result_speech_generated": (
                                result_speech_source is ResponseOnlySpeechSource.BACKEND_AUTHORED
                            ),
                            "result_speech_delivered": False,
                            "audio_played_ms": 0,
                            "playback_receipt_protocol": _PLAYBACK_RECEIPT_PROTOCOL,
                            "playback_receipts_acknowledged": playback_receipts.acknowledged_count,
                            "early_local_receipt_observed": True,
                            "acknowledgement_slot_reserved_before_dispatch": (
                                queue_reserved_before_dispatch_at < dispatching_at
                            ),
                            "acknowledgement_created_before_backend_success": (
                                acknowledgement_created_at < succeeded_at
                            ),
                            "acknowledgement_completed_before_backend_success": (
                                acknowledgement_completed_at < succeeded_at
                            ),
                            "acknowledgement_completed_before_result_delivery": (
                                acknowledgement_completed_at < result_delivery_completed_at
                                if result_delivery_completed_at is not None
                                else None
                            ),
                            "result_delivery_completed_after_backend_success": (
                                result_delivery_completed_at > succeeded_at
                                if result_delivery_completed_at is not None
                                else None
                            ),
                            "result_display_completed_before_backend_success": (
                                result_display_completed_at < succeeded_at
                            ),
                            "backend_dispatch_started_before_acknowledgement_completed": (
                                dispatching_at < acknowledgement_completed_at
                            ),
                            "peak_speech_queue_depth": queued_speech_depth,
                            "speech_queue_depth_transitions": [
                                depth for _observed_at, depth in queue_depth_transitions
                            ],
                            "speech_queue_final_depth": final_queue_depth,
                            "durable_work_id_issued": False,
                            "backend_turn_phases": [phase.value for phase in backend_turn_phases],
                            "event_order": {
                                "locally_queued": local_receipt_at,
                                "delivery_queue_nonempty": queue_reserved_before_dispatch_at,
                                "dispatching": dispatching_at,
                                "waiting_for_response": waiting_at,
                                "acknowledgement_created": acknowledgement_created_at,
                                "acknowledgement_completed": acknowledgement_completed_at,
                                "result_display_started": result_display_started_at,
                                "result_display_completed": result_display_completed_at,
                                "succeeded": succeeded_at,
                                "result_delivery_completed": result_delivery_completed_at,
                                "delivery_queue_drained": queue_drained_at,
                            },
                            "projections": observed,
                        }
                        required_playback_response_ids = {acknowledgement_response_id}
                        if result_speech_source is ResponseOnlySpeechSource.BACKEND_AUTHORED:
                            assert result_delivery_response_id is not None
                            required_playback_response_ids.add(result_delivery_response_id)
                        if playback_receipts.all_acknowledged(required_playback_response_ids):
                            pending_success_result["playback_receipts_acknowledged"] = (
                                playback_receipts.acknowledged_count
                            )
                            return pending_success_result

                elif event_type == "conversation.item.truncated":
                    acknowledged_response_id = playback_receipts.acknowledge(event)
                    print(f"zero-playback receipt acknowledged for {acknowledged_response_id}", file=sys.stderr)
                    if pending_success_result is not None and playback_receipts.all_acknowledged(
                        required_playback_response_ids
                    ):
                        pending_success_result["playback_receipts_acknowledged"] = playback_receipts.acknowledged_count
                        return pending_success_result

                if session_updated and gateway_reachable and not query_submitted:
                    query_submitted = True
                    await _send(
                        websocket,
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "id": _item_id(),
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": arguments.query}],
                            },
                        },
                    )
                    await _send(
                        websocket,
                        {
                            "type": "response.create",
                            "response": {},
                        },
                    )


def main() -> None:
    """Run one public-contract delegation and emit its succeeded result as JSON."""
    arguments = _parser().parse_args()
    try:
        result = asyncio.run(_verify(arguments))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (OSError, RuntimeError, TimeoutError, ValueError, WebSocketException) as error:
        raise SystemExit(f"Realtime delegation verification failed: {error}") from error
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
