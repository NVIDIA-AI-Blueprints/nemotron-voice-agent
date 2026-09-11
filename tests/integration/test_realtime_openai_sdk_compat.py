# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

r"""Live OpenAI Realtime compatibility checks against ``WS /v1/realtime``.

Not collected by CI (``pytest tests/unit``). Opt in with ``RUN_REALTIME_COMPAT=1``.

Covers:

* OpenAI Python SDK multi-turn over the canonical Realtime event contract
* Deploy-agnostic initial updates that echo the advertised model, voice, audio
  formats, turn detection, and fixed transcription producer
* Real audio and text response lifecycles with canonical terminal events
* Structured rejection of pre-GA or unsupported fields and unavailable voices
* Initial manual-turn configuration on cascaded pipelines,
  post-handoff session changes, and per-response instruction overrides
* Fast client-tool output staging and ordered Response A/output/Response B flow

Run::

    PIPELINE_TLS=false \
      uv run python src/server.py --host 127.0.0.1 --port 7860
    OPENAI_REALTIME_WS_BASE=ws://127.0.0.1:7860/v1 \
      RUN_REALTIME_COMPAT=1 \
      uv run pytest tests/integration/test_realtime_openai_sdk_compat.py -v -s
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REALTIME_COMPAT", "").strip().lower() not in {"1", "true", "yes"},
    reason="Set RUN_REALTIME_COMPAT=1 to run live OpenAI Realtime SDK compat",
)

DEFAULT_WS_BASE = os.getenv("OPENAI_REALTIME_WS_BASE", "wss://127.0.0.1:7860/v1")
DEFAULT_WS_URL = os.getenv("OPENAI_REALTIME_WS_URL", f"{DEFAULT_WS_BASE.rstrip('/')}/realtime")
DEFAULT_TEXTS = (
    "Say hello in one short sentence.",
    "What is two plus two? Reply with just the number.",
    "Thanks. Reply with goodbye in one short sentence.",
)
FEATURE_INSTRUCTIONS = "Keep each compatibility-check reply to one short sentence."
FAST_CLIENT_TOOL = {
    "type": "function",
    "name": "compat_fast_echo",
    "description": "Return a deterministic compatibility-test marker supplied by the client.",
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


def _event_type(event: Any) -> str:
    if isinstance(event, dict):
        return str(event.get("type") or "")
    return str(getattr(event, "type", "") or "")


def _event_as_dict(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    for attr in ("model_dump", "to_dict"):
        fn = getattr(event, attr, None)
        if callable(fn):
            with contextlib.suppress(Exception):
                data = fn(exclude_none=True) if attr == "model_dump" else fn()
                if isinstance(data, dict):
                    return data
    with contextlib.suppress(Exception):
        return json.loads(event.model_dump_json())  # type: ignore[attr-defined]
    return {"type": _event_type(event), "raw": repr(event)}


def _require_object(value: Any, *, label: str) -> dict[str, Any]:
    assert isinstance(value, dict), f"{label} must be an object: {value!r}"
    return value


def _require_nonempty_string(value: Any, *, label: str) -> str:
    assert isinstance(value, str) and value, f"{label} must be a non-empty string: {value!r}"
    return value


def _validate_server_event(event: dict[str, Any], *, expected_type: str) -> None:
    assert event.get("type") == expected_type, event
    _require_nonempty_string(event.get("event_id"), label=f"{expected_type}.event_id")


def _validate_handshake(first: Any, second: Any) -> dict[str, Any]:
    """Validate the canonical two-event handshake and return its session."""
    created = _event_as_dict(first)
    conversation = _event_as_dict(second)
    _validate_server_event(created, expected_type="session.created")
    _validate_server_event(conversation, expected_type="conversation.created")
    assert created["event_id"] != conversation["event_id"], (created, conversation)

    session = _require_object(created.get("session"), label="session.created.session")
    _require_nonempty_string(session.get("id"), label="session.id")
    assert session.get("object") == "realtime.session", session
    assert session.get("type") == "realtime", session
    _require_nonempty_string(session.get("model"), label="session.model")

    audio = _require_object(session.get("audio"), label="session.audio")
    audio_input = _require_object(audio.get("input"), label="session.audio.input")
    audio_output = _require_object(audio.get("output"), label="session.audio.output")
    assert audio_input.get("format") == {"type": "audio/pcm", "rate": 24000}, audio_input
    turn_detection = _require_object(
        audio_input.get("turn_detection"),
        label="session.audio.input.turn_detection",
    )
    if turn_detection.get("type") == "server_vad":
        assert turn_detection == {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500,
            "create_response": True,
            "interrupt_response": True,
            "idle_timeout_ms": None,
        }, audio_input
    else:
        assert turn_detection == {
            "type": "semantic_vad",
            "eagerness": "auto",
            "create_response": True,
            "interrupt_response": True,
        }, audio_input
    assert audio_output.get("format") == {"type": "audio/pcm", "rate": 24000}, audio_output
    _require_nonempty_string(audio_output.get("voice"), label="session.audio.output.voice")

    conversation_obj = _require_object(
        conversation.get("conversation"),
        label="conversation.created.conversation",
    )
    _require_nonempty_string(conversation_obj.get("id"), label="conversation.id")
    assert conversation_obj.get("object") == "realtime.conversation", conversation_obj
    return session


def _effective_instructions(advertised: dict[str, Any], requested: str) -> str:
    """Keep pipeline-specific contracts while applying the client instruction."""
    requested = requested.strip()
    current = advertised.get("instructions")
    if isinstance(current, str) and current.strip():
        if not requested or requested in current:
            return current
        return f"{current.rstrip()}\n\n{requested}"
    return requested


def _initial_session_patch(
    advertised: dict[str, Any],
    *,
    output_modality: str,
    instructions: str,
) -> dict[str, Any]:
    """Build a strict update from values advertised by the selected deployment."""
    assert output_modality in {"audio", "text"}
    audio = _require_object(advertised.get("audio"), label="session.audio")
    audio_input = _require_object(audio.get("input"), label="session.audio.input")
    audio_output = _require_object(audio.get("output"), label="session.audio.output")
    transcription = audio_input.get("transcription")

    input_patch: dict[str, Any] = {
        "format": copy.deepcopy(audio_input.get("format")),
        "turn_detection": copy.deepcopy(audio_input.get("turn_detection")),
    }
    if transcription is not None:
        input_patch["transcription"] = copy.deepcopy(transcription)

    return {
        "type": "realtime",
        "model": _require_nonempty_string(advertised.get("model"), label="session.model"),
        "instructions": _effective_instructions(advertised, instructions),
        "output_modalities": [output_modality],
        "max_output_tokens": 256,
        "tools": [],
        "tool_choice": "none",
        "parallel_tool_calls": True,
        "include": [],
        "prompt": None,
        "tracing": None,
        "audio": {
            "input": input_patch,
            "output": {
                "format": copy.deepcopy(audio_output.get("format")),
                "voice": _require_nonempty_string(
                    audio_output.get("voice"),
                    label="session.audio.output.voice",
                ),
            },
        },
    }


def _assert_updated_session(
    updated: dict[str, Any],
    *,
    advertised: dict[str, Any],
    patch: dict[str, Any],
) -> None:
    _validate_server_event(updated, expected_type="session.updated")
    session = _require_object(updated.get("session"), label="session.updated.session")
    assert session.get("id") == advertised.get("id"), session
    assert session.get("model") == advertised.get("model"), session
    assert session.get("instructions") == patch.get("instructions"), session
    assert session.get("output_modalities") == patch.get("output_modalities"), session
    assert session.get("max_output_tokens") == patch.get("max_output_tokens"), session
    assert session.get("tools") == patch.get("tools"), session
    assert session.get("tool_choice") == patch.get("tool_choice"), session
    assert session.get("parallel_tool_calls") == patch.get("parallel_tool_calls"), session
    assert session.get("include") == patch.get("include"), session
    assert session.get("prompt") == patch.get("prompt"), session
    assert session.get("tracing") == patch.get("tracing"), session
    audio = _require_object(session.get("audio"), label="session.audio")
    audio_input = _require_object(audio.get("input"), label="session.audio.input")
    output = _require_object(audio.get("output"), label="session.audio.output")
    assert audio_input.get("format") == patch["audio"]["input"]["format"], audio_input
    assert audio_input.get("turn_detection") == patch["audio"]["input"]["turn_detection"], audio_input
    if "transcription" in patch["audio"]["input"]:
        assert audio_input.get("transcription") == patch["audio"]["input"]["transcription"], audio_input
    assert output.get("format") == patch["audio"]["output"]["format"], output
    assert output.get("voice") == patch["audio"]["output"]["voice"], output


async def _recv_event(connection: Any) -> Any:
    """Receive one event through the SDK so parser incompatibilities fail the test."""
    return await connection.recv()


@dataclass
class TurnResult:
    """Capture one assistant response lifecycle for assertions."""

    label: str
    matched: bool = False
    saw_done: bool = False
    status: str | None = None
    transcript_deltas: list[str] = field(default_factory=list)
    transcript_done: str | None = None
    audio_deltas: int = 0
    function_calls: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    event_types: list[str] = field(default_factory=list)
    last_event: dict[str, Any] | None = None
    response_created: dict[str, Any] | None = None
    response_done: dict[str, Any] | None = None
    _seen_call_ids: set[str] = field(default_factory=set, repr=False)
    _response_id: str | None = field(default=None, repr=False)

    @property
    def transcript(self) -> str:
        """Best-effort assistant transcript for this turn."""
        if self.transcript_done:
            return self.transcript_done.strip()
        return "".join(self.transcript_deltas).strip()

    def _record_function_call(self, call: dict[str, Any]) -> None:
        call_id = call.get("call_id")
        if not isinstance(call_id, str) or not call_id or call_id in self._seen_call_ids:
            return
        self._seen_call_ids.add(call_id)
        self.function_calls.append(call)

    def handle(self, event: Any) -> None:
        """Fold one Realtime server event into this turn."""
        data = _event_as_dict(event)
        et = str(data.get("type") or "")
        response = data.get("response") if isinstance(data.get("response"), dict) else None
        if et == "response.created":
            response_id = _require_nonempty_string(
                response.get("id") if response is not None else None,
                label="response.created.response.id",
            )
            assert self._response_id is None, f"{self.label}: multiple response.created events"
            self._response_id = response_id
        elif et.startswith("response."):
            response_id = (
                response.get("id") if et == "response.done" and response is not None else data.get("response_id")
            )
            if response_id is not None:
                assert self._response_id is not None, f"{self.label}: {et} preceded response.created"
                assert response_id == self._response_id, (
                    f"{self.label}: {et} belongs to {response_id!r}, expected {self._response_id!r}"
                )
        self.last_event = data
        self.event_types.append(et)
        if et in {"response.output_audio_transcript.delta", "response.output_text.delta"}:
            delta = str(data.get("delta") or "")
            if delta:
                self.transcript_deltas.append(delta)
        elif et in {"response.output_audio_transcript.done", "response.output_text.done"}:
            text = data.get("transcript")
            if text is None:
                text = data.get("text")
            if isinstance(text, str) and text.strip():
                self.transcript_done = text
        elif et == "response.output_audio.delta":
            if data.get("delta"):
                self.audio_deltas += 1
        elif et == "response.function_call_arguments.done":
            self._record_function_call(
                {
                    "call_id": data.get("call_id"),
                    "name": data.get("name"),
                    "arguments": data.get("arguments"),
                }
            )
        elif et == "response.output_item.done":
            item = data.get("item") if isinstance(data.get("item"), dict) else {}
            if item.get("type") == "function_call":
                self._record_function_call(
                    {
                        "call_id": item.get("call_id"),
                        "name": item.get("name"),
                        "arguments": item.get("arguments"),
                    }
                )
        elif et == "response.created":
            response = data.get("response")
            if isinstance(response, dict):
                self.response_created = response
        elif et == "response.done":
            response = data.get("response") or {}
            if isinstance(response, dict):
                self.response_done = response
                self.status = str(response.get("status") or "")
                for item in response.get("output") or []:
                    if isinstance(item, dict) and item.get("type") == "function_call":
                        self._record_function_call(
                            {
                                "call_id": item.get("call_id"),
                                "name": item.get("name"),
                                "arguments": item.get("arguments"),
                            }
                        )
            self.saw_done = True
        elif et == "error":
            err = data.get("error") if isinstance(data.get("error"), dict) else data
            self.errors.append(err if isinstance(err, dict) else {"message": str(err)})


def _raise_if_error(turn: TurnResult, *, allow_codes: frozenset[str] | None = None) -> None:
    """Fail on Realtime ``error`` events unless the code is explicitly allowed."""
    allow = allow_codes or frozenset()
    for err in turn.errors:
        code = str(err.get("code") or "")
        if code in allow:
            continue
        raise AssertionError(err.get("message") or f"Realtime error: {err}")


async def _wait_until(
    connection: Any,
    *,
    predicate: Callable[[TurnResult, Any], bool],
    timeout_s: float,
    label: str,
    allow_error_codes: frozenset[str] | None = None,
) -> TurnResult:
    turn = TurnResult(label=label)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            event = await asyncio.wait_for(_recv_event(connection), timeout=min(30.0, remaining))
        except TimeoutError:
            continue
        turn.handle(event)
        _raise_if_error(turn, allow_codes=allow_error_codes)
        if predicate(turn, event):
            turn.matched = True
            return turn
    return turn


async def _ws_recv_json(ws: Any, *, timeout_s: float = 30.0) -> dict[str, Any]:
    raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
    data = json.loads(raw)
    assert isinstance(data, dict), data
    return data


async def _ws_wait_until(
    ws: Any,
    *,
    predicate: Callable[[TurnResult, dict[str, Any]], bool],
    timeout_s: float,
    label: str,
    allow_error_codes: frozenset[str] | None = None,
) -> TurnResult:
    turn = TurnResult(label=label)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            event = await asyncio.wait_for(_ws_recv_json(ws), timeout=min(30.0, remaining))
        except TimeoutError:
            continue
        turn.handle(event)
        _raise_if_error(turn, allow_codes=allow_error_codes)
        if predicate(turn, event):
            turn.matched = True
            return turn
    return turn


async def _open_realtime_ws() -> Any:
    import ssl

    import websockets

    kwargs: dict[str, Any] = {"open_timeout": 30, "max_size": 8 * 1024 * 1024}
    api_key = os.getenv("OPENAI_REALTIME_API_KEY", "").strip()
    if api_key:
        kwargs["additional_headers"] = {"Authorization": f"Bearer {api_key}"}
    if DEFAULT_WS_URL.startswith("wss://"):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs["ssl"] = ctx
    return await websockets.connect(DEFAULT_WS_URL, **kwargs)


async def _raw_handshake(ws: Any) -> dict[str, Any]:
    first = await _ws_recv_json(ws)
    second = await _ws_recv_json(ws)
    return _validate_handshake(first, second)


def _assert_error_event(
    event: dict[str, Any],
    *,
    client_event_id: str,
    code: str,
    param: str,
) -> None:
    _validate_server_event(event, expected_type="error")
    error = _require_object(event.get("error"), label="error.error")
    assert error.get("type") == "invalid_request_error", error
    assert error.get("code") == code, error
    assert error.get("param") == param, error
    assert error.get("event_id") == client_event_id, error


def _assert_subsequence(values: list[str], expected: list[str], *, label: str) -> None:
    cursor = 0
    for value in values:
        if cursor < len(expected) and value == expected[cursor]:
            cursor += 1
    assert cursor == len(expected), f"{label}: missing ordered {expected[cursor:]}; got {values}"


def _assert_response_modality(turn: TurnResult, *, output_modality: str) -> None:
    assert turn.saw_done and turn.status == "completed", turn
    assert turn.transcript, f"{turn.label}: assistant output was empty"
    common_tail = [
        "response.content_part.done",
        "conversation.item.done",
        "response.output_item.done",
        "response.done",
    ]
    if output_modality == "audio":
        assert turn.audio_deltas > 0, f"{turn.label}: no response.output_audio.delta"
        _assert_subsequence(
            turn.event_types,
            [
                "response.created",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_audio.delta",
                "response.output_audio.done",
                "response.output_audio_transcript.done",
                *common_tail,
            ],
            label=turn.label,
        )
        assert "response.output_text.delta" not in turn.event_types, turn.event_types
    else:
        assert turn.audio_deltas == 0, f"{turn.label}: text session emitted audio"
        _assert_subsequence(
            turn.event_types,
            [
                "response.created",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                *common_tail,
            ],
            label=turn.label,
        )
        assert "response.output_audio.delta" not in turn.event_types, turn.event_types


async def run_openai_sdk_compat(
    *,
    ws_base: str = DEFAULT_WS_BASE,
    api_key: str | None = None,
    texts: tuple[str, ...] | list[str] = DEFAULT_TEXTS,
    instructions: str = "You are a helpful voice assistant. Keep every reply to one short sentence.",
    turn_timeout_s: float = 120.0,
    turn_gap_s: float = 0.5,
) -> list[tuple[str, str]]:
    """Run one canonical audio session; return ``[(user, assistant), ...]``.

    Keep the helper's public arguments and return shape stable for live
    OpenAI SDK compatibility checks.
    """
    from openai import AsyncOpenAI

    key = api_key or os.getenv("OPENAI_REALTIME_API_KEY") or "sk-realtime-compat"
    turns = [t.strip() for t in texts if str(t).strip()]
    assert turns, "need at least one user text"

    client = AsyncOpenAI(
        api_key=key,
        websocket_base_url=ws_base.rstrip("/"),
        base_url=ws_base.rstrip("/").replace("ws://", "http://").replace("wss://", "https://"),
    )

    pairs: list[tuple[str, str]] = []
    ws_opts: dict[str, Any] = {}
    if ws_base.startswith("wss://"):
        import ssl

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ws_opts["ssl"] = ctx

    async with client.realtime.connect(websocket_connection_options=ws_opts) as connection:
        created = await asyncio.wait_for(_recv_event(connection), timeout=30.0)
        conversation = await asyncio.wait_for(_recv_event(connection), timeout=30.0)
        advertised = _validate_handshake(created, conversation)
        patch = _initial_session_patch(
            advertised,
            output_modality="audio",
            instructions=instructions,
        )

        await connection.send(
            {
                "event_id": "event_sdk_configure",
                "type": "session.update",
                "session": patch,
            }
        )

        updated = await _wait_until(
            connection,
            predicate=lambda _t, event: _event_type(event) == "session.updated",
            timeout_s=120.0,
            label="session.update",
        )
        assert updated.matched, "timed out waiting for session.updated"
        assert updated.last_event is not None
        _assert_updated_session(updated.last_event, advertised=advertised, patch=patch)

        for i, user_text in enumerate(turns, start=1):
            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_text}],
                }
            )
            await connection.response.create()
            turn = await _wait_until(
                connection,
                predicate=lambda t, _event: t.saw_done,
                timeout_s=turn_timeout_s,
                label=f"turn-{i}",
            )
            assert turn.matched and turn.saw_done, f"turn {i}: no response.done"
            assert turn.status == "completed", f"turn {i}: response.status={turn.status!r}"
            assert turn.transcript, f"turn {i}: empty assistant transcript"
            _assert_response_modality(turn, output_modality="audio")
            pairs.append((user_text, turn.transcript))
            await asyncio.sleep(turn_gap_s)

    return pairs


async def _send_user_text(ws: Any, text: str) -> None:
    await ws.send(
        json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
    )
    await ws.send(json.dumps({"type": "response.create"}))


async def _settle(ws: Any, *, transcript: str = "", min_s: float = 1.0) -> None:
    """Drain only late metric events while the selected pipeline becomes idle."""
    wait_s = max(min_s, min(4.0, 0.045 * max(len(transcript), 1)))
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            event = await asyncio.wait_for(_ws_recv_json(ws), timeout=min(0.4, remaining))
        except TimeoutError:
            continue
        assert event.get("type") == "nvidia.metrics.updated", f"unexpected event while settling: {event}"


async def _wait_for_completed_response(
    ws: Any,
    *,
    label: str,
    output_modality: str,
    timeout_s: float,
) -> TurnResult:
    turn = await _ws_wait_until(
        ws,
        predicate=lambda candidate, _event: candidate.saw_done,
        timeout_s=timeout_s,
        label=label,
    )
    assert turn.matched and turn.saw_done, f"{label}: no response.done"
    _assert_response_modality(turn, output_modality=output_modality)
    return turn


async def _send_session_update(
    ws: Any,
    session: dict[str, Any],
    *,
    event_id: str,
    label: str,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    await ws.send(json.dumps({"event_id": event_id, "type": "session.update", "session": session}))
    result = await _ws_wait_until(
        ws,
        predicate=lambda _turn, event: event.get("type") == "session.updated",
        timeout_s=timeout_s,
        label=label,
    )
    assert result.matched and result.last_event is not None, f"{label}: no session.updated"
    return result.last_event


async def _send_response_create(ws: Any, response: dict[str, Any], *, event_id: str) -> None:
    await ws.send(json.dumps({"event_id": event_id, "type": "response.create", "response": response}))


async def _run_text_output_checks(
    *,
    turn_timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Exercise text output plus live session and response-local controls."""
    summary: dict[str, Any] = {"turns": []}

    async with await _open_realtime_ws() as ws:
        advertised = await _raw_handshake(ws)
        patch = _initial_session_patch(
            advertised,
            output_modality="text",
            instructions=FEATURE_INSTRUCTIONS,
        )
        updated = await _send_session_update(
            ws,
            patch,
            event_id="event_text_configure",
            timeout_s=120.0,
            label="text-session.update",
        )
        _assert_updated_session(updated, advertised=advertised, patch=patch)
        summary["session"] = {
            "model": advertised.get("model"),
            "voice": patch["audio"]["output"]["voice"],
            "output_modalities": ["text"],
            "tools": [],
        }

        user_text = "Confirm briefly that this text Realtime response works."
        await _send_user_text(ws, user_text)
        response = await _wait_for_completed_response(
            ws,
            label="text-turn",
            output_modality="text",
            timeout_s=turn_timeout_s,
        )
        summary["turns"].append({"kind": "text", "user": user_text, "assistant": response.transcript})
        await _settle(ws, transcript=response.transcript, min_s=1.0)

        live_event_id = "event_live_instructions_update"
        live_instructions = f"{patch['instructions']}\nChanged after handoff."
        live_update = await _send_session_update(
            ws,
            {"instructions": live_instructions},
            event_id=live_event_id,
            label="live-instructions-update",
        )
        live_session = _require_object(live_update.get("session"), label="live session.updated.session")
        assert live_session.get("instructions") == live_instructions, live_session

        override_event_id = "event_response_override"
        override_instructions = "Reply with exactly OVERRIDE-OK."
        override_max_output_tokens = 64
        await _send_response_create(
            ws,
            {"instructions": override_instructions, "max_output_tokens": override_max_output_tokens},
            event_id=override_event_id,
        )
        override = await _wait_for_completed_response(
            ws,
            label="response-override",
            output_modality="text",
            timeout_s=turn_timeout_s,
        )
        for label, effective in (
            ("response.created.response", override.response_created),
            ("response.done.response", override.response_done),
        ):
            response_object = _require_object(effective, label=label)
            assert "instructions" not in response_object, response_object
            assert "parallel_tool_calls" not in response_object, response_object
            assert response_object.get("max_output_tokens") == override_max_output_tokens, response_object
        # The deterministic unit coverage below this live suite inspects the
        # immutable provider context and proves that response instructions
        # replace only the response-local prompt.  Do not treat a generative
        # model's exact wording as a wire-protocol guarantee here: the live
        # contract is the completed, non-empty response lifecycle plus the
        # requested response fields and session isolation asserted below.
        summary["response_override_text"] = override.transcript

        await _send_response_create(
            ws,
            {
                "conversation": "auto",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Return the marker INPUT-CONTEXT-OK.",
                            }
                        ],
                    }
                ],
                "instructions": "Answer the response input with only the marker it requests.",
                "max_output_tokens": 64,
                "output_modalities": ["text"],
            },
            event_id="event_custom_response_input",
        )
        custom_input = await _wait_for_completed_response(
            ws,
            label="custom-response-input",
            output_modality="text",
            timeout_s=turn_timeout_s,
        )
        assert "INPUT-CONTEXT-OK" in custom_input.transcript, custom_input.transcript
        custom_response = _require_object(
            custom_input.response_done,
            label="custom response.done.response",
        )
        assert custom_response.get("conversation_id"), custom_response
        assert custom_response.get("output_modalities") == ["text"], custom_response
        assert custom_response.get("audio") is None, custom_response

        advertised_audio = _require_object(advertised.get("audio"), label="advertised.audio")
        advertised_output = _require_object(advertised_audio.get("output"), label="advertised.audio.output")
        await _send_response_create(
            ws,
            {
                "conversation": "auto",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Say AUDIO-OVERRIDE-OK briefly.",
                            }
                        ],
                    }
                ],
                "instructions": "Include the marker AUDIO-OVERRIDE-OK.",
                "max_output_tokens": "inf",
                "output_modalities": ["audio"],
                "audio": {
                    "output": {
                        "format": copy.deepcopy(advertised_output["format"]),
                        "voice": advertised_output["voice"],
                    }
                },
            },
            event_id="event_response_audio_override",
        )
        audio_override = await _wait_for_completed_response(
            ws,
            label="response-audio-override",
            output_modality="audio",
            timeout_s=turn_timeout_s,
        )
        response_audio = _require_object(
            _require_object(audio_override.response_done, label="audio response.done.response").get("audio"),
            label="audio response.audio",
        )
        assert response_audio.get("output") == {
            "format": advertised_output["format"],
            "voice": advertised_output["voice"],
        }, response_audio

        inherited = await _send_session_update(
            ws,
            {},
            event_id="event_restore_session_defaults",
            label="session-defaults-after-response-override",
        )
        inherited_session = _require_object(inherited.get("session"), label="inherited session.updated.session")
        assert inherited_session.get("instructions") == live_instructions, inherited_session
        assert inherited_session.get("max_output_tokens") == patch["max_output_tokens"], inherited_session

    return summary


async def _run_initial_rejection_checks() -> list[str]:
    """Assert exact canonical errors before any pipeline handoff."""
    async with await _open_realtime_ws() as ws:
        advertised = await _raw_handshake(ws)
        audio = _require_object(advertised.get("audio"), label="session.audio")
        audio_input = _require_object(audio.get("input"), label="session.audio.input")
        voice = _require_nonempty_string(
            _require_object(audio.get("output"), label="session.audio.output").get("voice"),
            label="session.audio.output.voice",
        )
        transcription = audio_input.get("transcription")
        alternate_transcription = {"model": "compat-unavailable-transcription"}
        if transcription == alternate_transcription:
            alternate_transcription = {"model": "compat-other-unavailable-transcription"}

        cases: list[tuple[str, dict[str, Any], str, str]] = [
            ("pre_ga_voice", {"voice": voice}, "unknown_parameter", "session.voice"),
            ("pre_ga_temperature", {"temperature": 0.8}, "unknown_parameter", "session.temperature"),
            ("pre_ga_modalities", {"modalities": ["text"]}, "unknown_parameter", "session.modalities"),
            (
                "combined_modalities",
                {"output_modalities": ["text", "audio"]},
                "invalid_value",
                "session.output_modalities",
            ),
            (
                "unknown_voice",
                {"audio": {"output": {"voice": f"{voice}-unavailable"}}},
                "unsupported_capability",
                "session.audio.output.voice",
            ),
            (
                "changed_transcription",
                {"audio": {"input": {"transcription": alternate_transcription}}},
                "unsupported_capability",
                "session.audio.input.transcription.model",
            ),
        ]
        checked: list[str] = []
        for name, session_patch, code, param in cases:
            client_event_id = f"event_reject_{name}"
            await ws.send(
                json.dumps(
                    {
                        "event_id": client_event_id,
                        "type": "session.update",
                        "session": session_patch,
                    }
                )
            )
            event = await _ws_recv_json(ws)
            _assert_error_event(
                event,
                client_event_id=client_event_id,
                code=code,
                param=param,
            )
            checked.append(_require_object(event.get("error"), label="error")["code"])
        return checked


async def _run_manual_mode_config_check() -> None:
    """Confirm an eligible cascaded deployment accepts manual turn control initially."""
    async with await _open_realtime_ws() as ws:
        advertised = await _raw_handshake(ws)
        patch = _initial_session_patch(
            advertised,
            output_modality="text",
            instructions=FEATURE_INSTRUCTIONS,
        )
        patch["audio"]["input"]["turn_detection"] = None
        await ws.send(
            json.dumps(
                {
                    "event_id": "event_manual_configure",
                    "type": "session.update",
                    "session": patch,
                }
            )
        )
        result = await _ws_wait_until(
            ws,
            predicate=lambda turn, _event: "session.updated" in turn.event_types or bool(turn.errors),
            timeout_s=120.0,
            label="manual-session.update",
            allow_error_codes=frozenset({"unsupported_capability"}),
        )
        if result.errors:
            error = result.errors[-1]
            if error.get("code") == "unsupported_capability":
                pytest.skip("The selected deployment does not expose cascaded manual input")
            raise AssertionError(error)
        assert result.last_event is not None
        _assert_updated_session(result.last_event, advertised=advertised, patch=patch)
        input_config = _require_object(
            _require_object(result.last_event["session"].get("audio"), label="session.audio").get("input"),
            label="session.audio.input",
        )
        assert input_config.get("turn_detection") is None, input_config


async def _run_turn_detection_config_check() -> dict[str, Any]:
    """Apply the exact supported fields for the deployment-advertised VAD type."""
    async with await _open_realtime_ws() as ws:
        advertised = await _raw_handshake(ws)
        patch = _initial_session_patch(
            advertised,
            output_modality="text",
            instructions=FEATURE_INSTRUCTIONS,
        )
        audio_input = _require_object(
            _require_object(advertised.get("audio"), label="session.audio").get("input"),
            label="session.audio.input",
        )
        advertised_turn_detection = _require_object(
            audio_input.get("turn_detection"),
            label="session.audio.input.turn_detection",
        )
        turn_type = advertised_turn_detection.get("type")
        if turn_type == "server_vad":
            turn_detection: dict[str, Any] = {
                "type": "server_vad",
                "threshold": 0.61,
                "prefix_padding_ms": 240,
                "silence_duration_ms": 640,
                "create_response": True,
                "interrupt_response": True,
                "idle_timeout_ms": 5_000,
            }
        elif turn_type == "semantic_vad":
            turn_detection = {
                "type": "semantic_vad",
                "eagerness": "auto",
                "create_response": True,
                "interrupt_response": True,
            }
        else:
            raise AssertionError(f"unsupported advertised turn detection: {advertised_turn_detection!r}")
        patch["audio"]["input"]["turn_detection"] = turn_detection

        await ws.send(
            json.dumps(
                {
                    "event_id": "event_turn_detection_configure",
                    "type": "session.update",
                    "session": patch,
                }
            )
        )
        result = await _ws_wait_until(
            ws,
            predicate=lambda turn, _event: "session.updated" in turn.event_types or bool(turn.errors),
            timeout_s=120.0,
            label="turn-detection-session.update",
        )
        assert result.matched and result.last_event is not None, "no session.updated"
        _assert_updated_session(result.last_event, advertised=advertised, patch=patch)
        echoed_input = _require_object(
            _require_object(result.last_event["session"].get("audio"), label="session.audio").get("input"),
            label="session.audio.input",
        )
        assert echoed_input.get("turn_detection") == turn_detection, echoed_input
        return copy.deepcopy(echoed_input["turn_detection"])


def _response_id(event: dict[str, Any]) -> str | None:
    response = event.get("response")
    response_id = response.get("id") if isinstance(response, dict) else event.get("response_id")
    return response_id if isinstance(response_id, str) and response_id else None


def _event_index(
    events: list[tuple[float, dict[str, Any]]],
    predicate: Callable[[dict[str, Any]], bool],
    *,
    label: str,
) -> int:
    for index, (_received_at, event) in enumerate(events):
        if predicate(event):
            return index
    raise AssertionError(f"missing {label}: {[event.get('type') for _, event in events]}")


def _function_call_identity(
    value: dict[str, Any],
    *,
    item_id_key: str = "id",
) -> tuple[str, str, str]:
    return (
        _require_nonempty_string(value.get(item_id_key), label=f"function_call.{item_id_key}"),
        _require_nonempty_string(value.get("call_id"), label="function_call.call_id"),
        _require_nonempty_string(value.get("name"), label="function_call.name"),
    )


async def _run_fast_client_tool_check(*, timeout_s: float = 120.0) -> dict[str, Any]:
    """Verify an early client output is staged into an exact A/output/B lifecycle."""
    run_token = uuid.uuid4().hex
    test_id = f"tool_test_{run_token}"
    marker = f"FAST_OUTPUT_ACCEPTED_{run_token[:12].upper()}"
    response_a_metadata = {
        "rva411_tool_test_id": test_id,
        "rva411_tool_test_phase": "response_a",
    }
    response_b_metadata = {
        "rva411_tool_test_id": test_id,
        "rva411_tool_test_phase": "response_b",
    }

    async with await _open_realtime_ws() as ws:
        advertised = await _raw_handshake(ws)
        patch = _initial_session_patch(
            advertised,
            output_modality="text",
            instructions=(
                "Call compat_fast_echo exactly once only when the user explicitly requests the compatibility "
                "client-tool check. Do not answer that request before its function output exists. After the "
                "output, do not call any tool again; answer in text and include its marker exactly."
            ),
        )
        patch["tools"] = [copy.deepcopy(FAST_CLIENT_TOOL)]
        patch["tool_choice"] = "auto"
        await ws.send(
            json.dumps(
                {
                    "event_id": f"event_fast_tool_configure_{run_token}",
                    "type": "session.update",
                    "session": patch,
                }
            )
        )
        updated = await _ws_recv_json(ws, timeout_s=120.0)
        if updated.get("type") == "error":
            error = _require_object(updated.get("error"), label="error.error")
            if error.get("code") == "unsupported_capability" and error.get("param") == "session.tools":
                pytest.skip("The selected pipeline does not support client-owned tools")
            raise AssertionError(error)
        _assert_updated_session(updated, advertised=advertised, patch=patch)

        user_item_id = f"item_fast_user_{run_token}"
        await ws.send(
            json.dumps(
                {
                    "event_id": f"event_fast_user_{run_token}",
                    "type": "conversation.item.create",
                    "item": {
                        "id": user_item_id,
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Run the compatibility client-tool check now. Call compat_fast_echo exactly "
                                    "once with an empty object, then copy the marker from its output into your "
                                    "text answer."
                                ),
                            }
                        ],
                    },
                }
            )
        )
        await ws.send(
            json.dumps(
                {
                    "event_id": f"event_fast_response_a_{run_token}",
                    "type": "response.create",
                    "response": {
                        "metadata": response_a_metadata,
                        "tool_choice": {"type": "function", "name": FAST_CLIENT_TOOL["name"]},
                    },
                }
            )
        )

        events: list[tuple[float, dict[str, Any]]] = []
        response_a_id: str | None = None
        call_id: str | None = None
        function_identity: tuple[str, str, str] | None = None
        output_item_id = f"item_fast_client_output_{run_token}"
        output_value = json.dumps(
            {"ok": True, "marker": marker, "source": "integration-client"},
            separators=(",", ":"),
        )
        fast_send_at: float | None = None
        function_done_at: float | None = None
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline and call_id is None:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                event = await _ws_recv_json(ws, timeout_s=min(30.0, remaining))
            except TimeoutError:
                continue
            received_at = time.monotonic()
            events.append((received_at, event))
            if event.get("type") == "error":
                raise AssertionError(event.get("error"))
            if event.get("type") == "response.created":
                response = _require_object(event.get("response"), label="Response A response.created.response")
                assert response.get("metadata") == response_a_metadata, response
                response_a_id = _response_id(event)
                assert response_a_id is not None
                continue
            if event.get("response_id") == response_a_id and event.get("type") == "response.output_item.added":
                item = _require_object(event.get("item"), label="Response A response.output_item.added.item")
                assert item.get("type") == "function_call", item
                function_identity = _function_call_identity(item)
                continue
            if (
                event.get("response_id") == response_a_id
                and event.get("type") == "response.function_call_arguments.done"
            ):
                assert function_identity is not None
                assert _function_call_identity(event, item_id_key="item_id") == function_identity, event
                assert (
                    json.loads(_require_nonempty_string(event.get("arguments"), label="function_call.arguments")) == {}
                )
                continue
            if event.get("type") != "response.output_item.done":
                continue
            assert event.get("response_id") == response_a_id, event
            item = event.get("item") if isinstance(event.get("item"), dict) else {}
            assert item.get("type") == "function_call", item
            assert item.get("status") == "completed", item
            assert function_identity is not None
            assert _function_call_identity(item) == function_identity, item
            call_id = _require_nonempty_string(item.get("call_id"), label="function_call.call_id")
            assert function_identity[2] == FAST_CLIENT_TOOL["name"], item
            assert json.loads(_require_nonempty_string(item.get("arguments"), label="function_call.arguments")) == {}
            assert response_a_id is not None and event.get("response_id") == response_a_id, event
            assert not any(
                candidate.get("type") == "response.done" and _response_id(candidate) == response_a_id
                for _, candidate in events
            ), "Response A was already observed before the fast output"

            function_done_at = received_at
            fast_send_at = time.monotonic()
            await ws.send(
                json.dumps(
                    {
                        "event_id": f"event_fast_tool_output_{run_token}",
                        "type": "conversation.item.create",
                        "item": {
                            "id": output_item_id,
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": output_value,
                        },
                    }
                )
            )
            await ws.send(
                json.dumps(
                    {
                        "event_id": f"event_fast_response_b_{run_token}",
                        "type": "response.create",
                        "response": {
                            "metadata": response_b_metadata,
                            "tool_choice": "none",
                            "max_output_tokens": "inf",
                        },
                    }
                )
            )

        assert response_a_id is not None
        assert call_id is not None
        assert function_identity is not None
        assert function_done_at is not None
        assert fast_send_at is not None
        assert function_done_at <= fast_send_at

        deadline = time.monotonic() + timeout_s
        response_b_id: str | None = None
        response_a_done_at: float | None = None
        response_a_done_event: dict[str, Any] | None = None
        response_b_done_event: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                event = await _ws_recv_json(ws, timeout_s=min(30.0, remaining))
            except TimeoutError:
                continue
            received_at = time.monotonic()
            events.append((received_at, event))
            if event.get("type") == "error":
                raise AssertionError(event.get("error"))
            event_response_id = _response_id(event)
            if event.get("type") == "response.done" and event_response_id == response_a_id:
                response_a_done_at = received_at
                response_a_done_event = event
            elif event.get("type") == "response.created" and event_response_id != response_a_id:
                response = _require_object(event.get("response"), label="Response B response.created.response")
                assert response.get("metadata") == response_b_metadata, response
                assert response.get("max_output_tokens") == "inf", response
                response_b_id = event_response_id
                assert response_b_id is not None
            elif (
                event.get("type") == "response.done"
                and response_b_id is not None
                and event_response_id == response_b_id
            ):
                response_b_done_event = event
                break
        else:
            raise AssertionError("client-tool follow-up response did not complete")

        assert response_a_done_at is not None
        assert response_a_done_event is not None
        assert response_b_id is not None
        assert response_b_done_event is not None
        assert fast_send_at < response_a_done_at

        response_a = _require_object(response_a_done_event.get("response"), label="Response A response.done.response")
        assert response_a.get("metadata") == response_a_metadata, response_a
        response_a_output = response_a.get("output")
        assert isinstance(response_a_output, list) and len(response_a_output) == 1, response_a
        response_a_call = _require_object(response_a_output[0], label="Response A output[0]")
        assert response_a_call.get("type") == "function_call", response_a_call
        assert response_a_call.get("status") == "completed", response_a_call
        assert _function_call_identity(response_a_call) == function_identity, response_a_call
        assert (
            json.loads(
                _require_nonempty_string(response_a_call.get("arguments"), label="Response A function arguments")
            )
            == {}
        )

        response_a_done = _event_index(
            events,
            lambda event: event.get("type") == "response.done" and _response_id(event) == response_a_id,
            label="Response A response.done",
        )
        output_added = _event_index(
            events,
            lambda event: (
                event.get("type") == "conversation.item.added"
                and isinstance(event.get("item"), dict)
                and event["item"].get("id") == output_item_id
            ),
            label="function output conversation.item.added",
        )
        output_added_item = _require_object(events[output_added][1].get("item"), label="output item added")
        assert output_added_item == {
            "id": output_item_id,
            "object": "realtime.item",
            "type": "function_call_output",
            "status": "completed",
            "call_id": call_id,
            "output": output_value,
        }, output_added_item
        output_done = _event_index(
            events,
            lambda event: (
                event.get("type") == "conversation.item.done"
                and isinstance(event.get("item"), dict)
                and event["item"].get("id") == output_item_id
            ),
            label="function output conversation.item.done",
        )
        output_done_item = _require_object(events[output_done][1].get("item"), label="output item done")
        assert output_done_item == {
            "id": output_item_id,
            "object": "realtime.item",
            "type": "function_call_output",
            "status": "completed",
            "call_id": call_id,
            "output": output_value,
        }, output_done_item
        assert output_done_item == output_added_item
        response_b_created = _event_index(
            events,
            lambda event: event.get("type") == "response.created" and _response_id(event) == response_b_id,
            label="Response B response.created",
        )
        response_b_done = _event_index(
            events,
            lambda event: event.get("type") == "response.done" and _response_id(event) == response_b_id,
            label="Response B response.done",
        )
        assert response_a_done < output_added < output_done < response_b_created < response_b_done

        response_a_events = [
            event
            for _, event in events[: response_a_done + 1]
            if event.get("response_id") == response_a_id or _response_id(event) == response_a_id
        ]
        assert not any(
            event.get("type")
            in {
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.output_audio.delta",
                "response.output_audio.done",
                "response.output_audio_transcript.delta",
                "response.output_audio_transcript.done",
            }
            for event in response_a_events
        ), response_a_events
        assert (
            sum(1 for event in response_a_events if event.get("type") == "response.function_call_arguments.done") == 1
        ), response_a_events

        response_b_turn = TurnResult(label="fast-client-tool-response-b")
        for _, event in events[response_b_created : response_b_done + 1]:
            response_b_turn.handle(event)
        _assert_response_modality(response_b_turn, output_modality="text")
        assert not response_b_turn.function_calls, response_b_turn.function_calls
        assert "<tool_call>" not in response_b_turn.transcript, response_b_turn.transcript

        response_b = _require_object(response_b_done_event.get("response"), label="Response B response.done.response")
        assert response_b.get("metadata") == response_b_metadata, response_b
        assert response_b.get("max_output_tokens") == "inf", response_b

        return {
            "model": advertised.get("model"),
            "test_id": test_id,
            "marker": marker,
            "call_id": call_id,
            "response_a_id": response_a_id,
            "response_b_id": response_b_id,
            "response_b_text": response_b_turn.transcript,
            "function_call_count": len(response_b_turn.function_calls) + 1,
            "send_to_response_a_done_ms": round((response_a_done_at - fast_send_at) * 1000, 3),
        }


def test_openai_realtime_sdk_three_turn_conversation() -> None:
    """Run three canonical audio turns through the official OpenAI SDK."""
    pairs = asyncio.run(run_openai_sdk_compat())
    assert len(pairs) == 3
    print("\n=== Realtime SDK compat conversation ===")
    for i, (user, assistant) in enumerate(pairs, start=1):
        assert user
        assert assistant
        print(f"turn {i}")
        print(f"  user:      {user}")
        print(f"  assistant: {assistant}")
    print("=== end ===\n")


def test_realtime_text_output_and_prompt_controls() -> None:
    """Run text output, then assert native live and response-local controls."""
    summary = asyncio.run(_run_text_output_checks())
    print("\n=== Realtime text-output conversation ===")
    for i, turn in enumerate(summary.get("turns") or [], start=1):
        print(f"turn {i} ({turn.get('kind')})")
        print(f"  user:      {turn.get('user')}")
        print(f"  assistant: {turn.get('assistant')}")
    print("--- checks ---")
    print(json.dumps({k: v for k, v in summary.items() if k != "turns"}, indent=2))
    print("=== end ===\n")
    assert summary.get("session", {}).get("tools") == []
    assert summary.get("session", {}).get("output_modalities") == ["text"]


def test_realtime_rejects_pre_ga_and_unavailable_session_fields() -> None:
    """Reject pre-GA aliases and unsupported current-profile capabilities."""
    checked = asyncio.run(_run_initial_rejection_checks())
    assert checked == [
        "unknown_parameter",
        "unknown_parameter",
        "unknown_parameter",
        "invalid_value",
        "unsupported_capability",
        "unsupported_capability",
    ]


def test_realtime_accepts_manual_turn_control_on_cascaded_pipeline() -> None:
    """Accept turn_detection=null during the initial cascaded session update."""
    asyncio.run(_run_manual_mode_config_check())


def test_realtime_applies_advertised_turn_detection_configuration() -> None:
    """Echo exact supported VAD fields without assuming one deployed VAD type."""
    configured = asyncio.run(_run_turn_detection_config_check())
    assert configured["type"] in {"server_vad", "semantic_vad"}
    assert configured["create_response"] is True
    assert configured["interrupt_response"] is True


def test_realtime_fast_client_tool_output_is_ordered() -> None:
    """Validate early output staging and distinct function-only A/text-only B responses."""
    summary = asyncio.run(_run_fast_client_tool_check())
    assert summary["response_a_id"] != summary["response_b_id"]
    assert summary["function_call_count"] == 1
    assert summary["marker"] in summary["response_b_text"]
    print("\n=== Realtime fast client-tool lifecycle ===")
    print(json.dumps(summary, indent=2))
    print("=== end ===\n")


def main() -> int:
    """CLI entry for manual runs without pytest."""
    if os.getenv("RUN_REALTIME_COMPAT", "").strip().lower() not in {"1", "true", "yes"}:
        print("Set RUN_REALTIME_COMPAT=1 to run this live compat check", flush=True)
        return 2
    pairs = asyncio.run(run_openai_sdk_compat())
    print("PASS  OpenAI Realtime SDK compatibility (generic client)")
    for i, (user, assistant) in enumerate(pairs, start=1):
        print(f"  {i}. user={user!r}")
        print(f"     assistant={assistant!r}")
    summary = asyncio.run(_run_text_output_checks())
    print("PASS  text output and prompt controls")
    for i, turn in enumerate(summary.get("turns") or [], start=1):
        print(f"  {i}. ({turn.get('kind')}) user={turn.get('user')!r}")
        print(f"     assistant={turn.get('assistant')!r}")
    checked = asyncio.run(_run_initial_rejection_checks())
    print(f"PASS  strict initial rejects ({len(checked)} cases)")
    asyncio.run(_run_manual_mode_config_check())
    print("PASS  cascaded manual turn configuration")
    configured_turn_detection = asyncio.run(_run_turn_detection_config_check())
    print(f"PASS  {configured_turn_detection['type']} configuration")
    tool_summary = asyncio.run(_run_fast_client_tool_check())
    print(f"PASS  fast client-tool ordering ({tool_summary['send_to_response_a_done_ms']} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
