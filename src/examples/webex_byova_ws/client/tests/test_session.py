"""Conversation lifecycle and event mapping tests."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from webex_byova_ws_adapter.completion import ReplyLifecycle
from webex_byova_ws_adapter.config import AdapterConfig
from webex_byova_ws_adapter.protocol import ProtocolError, parse_request
from webex_byova_ws_adapter.session import Conversation, SessionPhase

OUTBOUND_VALIDATOR = Draft202012Validator(
    json.loads((Path(__file__).parent / "fixtures" / "outbound.schema.json").read_text(encoding="utf-8")),
    format_checker=FormatChecker(),
)


class FakeWebSocket:
    """Capture application messages without a network listener."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed = False
        self.send_json = AsyncMock(side_effect=self.sent.append)

    async def close(self, **_: object) -> None:
        self.closed = True


def conversation() -> tuple[Conversation, FakeWebSocket]:
    """Build a local-test conversation."""
    websocket = FakeWebSocket()
    value = Conversation(
        AdapterConfig(
            allow_plaintext_local=True,
            auth_enabled=False,
            allow_auth_disabled_local=True,
        ),
        websocket,
    )
    value.conversation_id = "call-1"
    value.phase = SessionPhase.LISTENING
    return value, websocket


def cisco_request(seq: int, input_kind: str, input_value: dict) -> dict:
    """Build a same-conversation Cisco request."""
    return {
        "type": "VOICE_VA_REQUEST",
        "seq": seq,
        "ts": "2026-09-07T00:00:00Z",
        "conversation_id": "call-1",
        "payload": {
            "conversation_id": "call-1",
            "customer_org_id": "org-1",
            "voice_va_input_type": {input_kind: input_value},
        },
    }


def test_reply_lifecycle_is_explicit_and_exactly_once() -> None:
    reply = ReplyLifecycle()
    reply.begin("caller-turn")
    reply.note_audio()
    assert reply.finish()
    assert not reply.finish()
    reply.begin("caller-turn")
    assert reply.ordinal == 2


@pytest.mark.asyncio
async def test_rtvi_input_boundary_mapping() -> None:
    value, websocket = conversation()
    await value._handle_backend_event("user-started-speaking")
    await value._handle_backend_event("user-turn-finalized")
    assert [message["payload"]["output_events"][0]["event_type"] for message in websocket.sent] == [
        "START_OF_INPUT",
        "END_OF_INPUT",
    ]
    assert all(message["payload"]["response_type"] == "CHUNK" for message in websocket.sent)
    assert value.phase is SessionPhase.ASSISTANT_REPLY


@pytest.mark.asyncio
async def test_audio_is_chunked_with_barge_in_then_exactly_one_final() -> None:
    value, websocket = conversation()
    value.reply.begin("assistant-output")
    value.reply.note_audio()
    for chunk in value.bot_audio.feed(b"\x00\x00" * 100, 22_050, 1):
        await value._send_audio(chunk)
    await value._finish_reply()
    await value._finish_reply()
    response_types = [message["payload"]["response_type"] for message in websocket.sent]
    assert response_types == ["CHUNK", "FINAL"]
    prompt = websocket.sent[0]["payload"]["prompts"][0]
    assert prompt["is_barge_in_enabled"] is True
    assert websocket.sent[1]["payload"]["prompts"] == [
        {"audio_content_b64": "", "is_barge_in_enabled": True}
    ]


@pytest.mark.asyncio
async def test_empty_audio_frame_is_ignored_without_closing_session() -> None:
    value, websocket = conversation()
    bridge = SimpleNamespace(send_audio=AsyncMock())
    value.bridge = bridge
    await value._handle(
        {
            "type": "VOICE_VA_REQUEST",
            "seq": 1,
            "ts": "2026-09-07T00:00:00Z",
            "conversation_id": "call-1",
            "payload": {
                "conversation_id": "call-1",
                "customer_org_id": "org-1",
                "voice_va_input_type": {
                    "audio_input": {
                        "caller_audio_b64": None,
                        "encoding": "MULAW_FORMAT",
                        "sample_rate_hertz": 8000,
                    }
                },
            },
        }
    )
    bridge.send_audio.assert_not_awaited()
    assert not websocket.closed


@pytest.mark.asyncio
async def test_same_socket_accepts_multiple_audio_requests() -> None:
    value, _ = conversation()
    bridge = SimpleNamespace(send_audio=AsyncMock())
    value.bridge = bridge
    audio = {
        "caller_audio_b64": base64.b64encode(b"\xff" * 160).decode(),
        "encoding": "MULAW_FORMAT",
        "sample_rate_hertz": 8000,
    }
    await value._handle(cisco_request(1, "audio_input", audio))
    await value._handle(cisco_request(2, "audio_input", audio))
    assert bridge.send_audio.await_count == 2
    assert value.last_inbound_seq == 2


@pytest.mark.asyncio
async def test_continuation_socket_resumes_bridge_and_resets_sequences() -> None:
    value, first_websocket = conversation()
    value.bridge = SimpleNamespace(send_audio=AsyncMock())
    value.last_inbound_seq = 9
    value.sequence.next()
    replacement = FakeWebSocket()
    audio = {
        "caller_audio_b64": base64.b64encode(b"\xff" * 160).decode(),
        "encoding": "MULAW_FORMAT",
        "sample_rate_hertz": 8000,
    }
    await value.attach(replacement, cisco_request(1, "audio_input", audio))
    assert value.websocket is replacement
    assert first_websocket.closed
    assert value.last_inbound_seq == 1
    assert value.sequence.next() == 1
    value.bridge.send_audio.assert_awaited_once()


@pytest.mark.asyncio
async def test_continuation_waits_until_active_reply_is_final() -> None:
    value, _ = conversation()
    value.reply.begin("assistant-output")
    value.reply_complete.clear()
    replacement = FakeWebSocket()
    value.bridge = SimpleNamespace(send_audio=AsyncMock())
    audio = {
        "caller_audio_b64": base64.b64encode(b"\xff" * 160).decode(),
        "encoding": "MULAW_FORMAT",
        "sample_rate_hertz": 8000,
    }
    attach_task = asyncio.create_task(
        value.attach(replacement, cisco_request(1, "audio_input", audio))
    )
    await asyncio.sleep(0)
    assert not attach_task.done()
    await value._send_audio_final()
    await attach_task
    assert value.websocket is replacement


@pytest.mark.asyncio
async def test_continuation_timeout_closes_logical_conversation() -> None:
    value, _ = conversation()
    value.config.first_audio_timeout_seconds = 0.001
    value.config.completion_timeout_seconds = 0.001
    value.reply.begin("assistant-output")
    value.reply_complete.clear()
    replacement = FakeWebSocket()
    with pytest.raises(TimeoutError):
        await value.attach(
            replacement,
            cisco_request(
                1,
                "audio_input",
                {
                    "caller_audio_b64": None,
                    "encoding": "MULAW_FORMAT",
                    "sample_rate_hertz": 8000,
                },
            ),
        )
    assert value.closed
    assert value.closed_event.is_set()


@pytest.mark.asyncio
async def test_twelve_turns_share_one_socket_with_one_final_each() -> None:
    value, websocket = conversation()
    for _ in range(12):
        await value._handle_backend_event("user-started-speaking")
        await value._handle_backend_event("user-turn-finalized")
        await value._handle_backend_event("bot-started-speaking")
        for chunk in value.bot_audio.feed(b"\x00\x00" * 22_050, 22_050, 1):
            value.reply.note_audio()
            await value._send_audio(chunk)
        await value._finish_reply()

    assert value.phase is SessionPhase.LISTENING
    assert value.reply.ordinal == 12
    assert [message["seq"] for message in websocket.sent] == list(range(1, 49))
    assert sum(message["payload"]["response_type"] == "FINAL" for message in websocket.sent) == 12
    assert not websocket.closed
    for message in websocket.sent:
        OUTBOUND_VALIDATOR.validate(message)


@pytest.mark.asyncio
async def test_barge_in_finishes_interrupted_reply_then_keeps_caller_active() -> None:
    value, websocket = conversation()
    value.phase = SessionPhase.ASSISTANT_REPLY
    value.reply.begin("assistant-output")
    value.reply.note_audio()
    await value._handle_backend_event("user-started-speaking")
    await value._finish_reply()
    assert value.phase is SessionPhase.CALLER_ACTIVE
    assert [message["payload"]["response_type"] for message in websocket.sent] == ["CHUNK", "FINAL"]

    await value._handle_backend_event("user-turn-finalized")
    assert value.phase is SessionPhase.ASSISTANT_REPLY
    assert value.reply.active
    assert websocket.sent[-1]["payload"]["output_events"] == [{"event_type": "END_OF_INPUT"}]


@pytest.mark.asyncio
async def test_dtmf_is_forwarded_on_the_existing_session() -> None:
    value, _ = conversation()
    bridge = SimpleNamespace(send_input=AsyncMock())
    value.bridge = bridge
    dtmf = {"dtmf_events": ["DTMF_DIGIT_ONE", "DTMF_DIGIT_POUND"]}
    await value._handle(cisco_request(1, "dtmf_input", dtmf))
    bridge.send_input.assert_awaited_once_with("dtmf", dtmf)


@pytest.mark.asyncio
async def test_no_input_is_forwarded_as_an_event_turn() -> None:
    value, _ = conversation()
    bridge = SimpleNamespace(send_input=AsyncMock())
    value.bridge = bridge
    no_input = {"event_type": "NO_INPUT"}
    await value._handle(cisco_request(1, "event_input", no_input))
    bridge.send_input.assert_awaited_once_with("no-input", no_input)
    assert value.reply.active


@pytest.mark.asyncio
async def test_transfer_event_is_a_final_response() -> None:
    value, websocket = conversation()
    await value._send_event("TRANSFER_TO_AGENT")
    assert websocket.sent[0]["payload"] == {
        "output_events": [{"event_type": "TRANSFER_TO_AGENT"}],
        "response_type": "FINAL",
    }
    assert value.phase is SessionPhase.TERMINAL


@pytest.mark.asyncio
async def test_application_ping_uses_same_socket_and_echoes_sequence() -> None:
    value, websocket = conversation()
    await value._handle(
        {
            "type": "PING",
            "seq": 7,
            "ts": "2026-09-07T00:00:00Z",
            "conversation_id": "call-1",
        }
    )
    assert websocket.sent[0]["type"] == "PONG"
    assert websocket.sent[0]["seq"] == 7


@pytest.mark.asyncio
async def test_session_end_is_terminal_and_closes_socket() -> None:
    value, websocket = conversation()
    await value._session_end()
    assert websocket.sent[0]["payload"] == {
        "output_events": [{"event_type": "SESSION_END"}],
        "response_type": "FINAL",
    }
    assert websocket.closed


@pytest.mark.asyncio
async def test_reply_timeout_sends_bounded_error_and_final() -> None:
    value, websocket = conversation()
    value.reply.begin("caller-turn")
    await value._reply_timeout(0)
    assert [message["type"] for message in websocket.sent] == ["ERROR", "VOICE_VA_RESPONSE"]
    assert websocket.sent[-1]["payload"]["response_type"] == "FINAL"
    assert value.phase is SessionPhase.LISTENING


@pytest.mark.asyncio
async def test_transport_disconnect_releases_terminal_state() -> None:
    value, websocket = conversation()
    bridge = SimpleNamespace(close=AsyncMock())
    value.bridge = bridge
    await value.close()
    bridge.close.assert_awaited_once()
    assert value.phase is SessionPhase.TERMINAL
    assert websocket.closed


@pytest.mark.asyncio
async def test_session_start_opens_welcome_reply_once() -> None:
    websocket = FakeWebSocket()
    value = Conversation(
        AdapterConfig(allow_plaintext_local=True, auth_enabled=False, allow_auth_disabled_local=True),
        websocket,
    )

    class FakeBridge:
        def __init__(self, _: AdapterConfig, __: str) -> None:
            self.events = asyncio.Queue()
            self.start = AsyncMock()
            self.close = AsyncMock()

    start = parse_request(cisco_request(1, "event_input", {"event_type": "SESSION_START"}))
    with patch("webex_byova_ws_adapter.session.NemotronBridge", FakeBridge):
        await value._start(start)
        assert value.phase is SessionPhase.WELCOME_REPLY
        assert value.reply.active
        assert value.reply.ordinal == 1
        await value.close()


@pytest.mark.asyncio
async def test_server_sequence_is_independent_and_monotonic() -> None:
    value, websocket = conversation()
    await value._send_event("START_OF_INPUT")
    await value._send_event("END_OF_INPUT")
    assert [message["seq"] for message in websocket.sent] == [1, 2]


@pytest.mark.asyncio
async def test_request_org_must_match_authenticated_org() -> None:
    websocket = FakeWebSocket()
    value = Conversation(
        AdapterConfig(auth_enabled=True, org_uuid="org-allowed"),
        websocket,
    )
    request = {
        "type": "VOICE_VA_REQUEST",
        "seq": 1,
        "ts": "2026-09-07T00:00:00Z",
        "conversation_id": "call-1",
        "payload": {
            "conversation_id": "call-1",
            "customer_org_id": "org-other",
            "voice_va_input_type": {"event_input": {"event_type": "SESSION_START"}},
        },
    }
    with pytest.raises(ProtocolError, match="authenticated organization"):
        await value._handle(request)
