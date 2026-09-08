"""Conversation lifecycle and event mapping tests."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from webex_byova_ws_adapter.completion import BaselineCompletionPolicy
from webex_byova_ws_adapter.config import AdapterConfig
from webex_byova_ws_adapter.protocol import ProtocolError
from webex_byova_ws_adapter.session import Conversation


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
    value.session_started = True
    return value, websocket


def test_baseline_completion_is_deduplicated_and_requires_audio() -> None:
    policy = BaselineCompletionPolicy()
    policy.begin_reply()
    assert not policy.should_finalize("bot-stopped-speaking")
    policy.note_audio()
    assert policy.should_finalize("bot-stopped-speaking")
    assert not policy.should_finalize("bot-stopped-speaking")
    policy.begin_reply()
    assert not policy.should_finalize("bot-started-speaking")


@pytest.mark.asyncio
async def test_rtvi_input_boundary_mapping() -> None:
    value, websocket = conversation()
    await value._handle_backend_event("user-started-speaking")
    await value._handle_backend_event("user-turn-finalized")
    assert [message["payload"]["output_events"][0]["event_type"] for message in websocket.sent] == [
        "START_OF_INPUT",
        "END_OF_INPUT",
    ]


@pytest.mark.asyncio
async def test_audio_is_chunked_with_barge_in_then_exactly_one_final() -> None:
    value, websocket = conversation()
    value.completion.begin_reply()
    value.completion.note_audio()
    for chunk in value.bot_audio.feed(b"\x00\x00" * 100):
        await value._send_audio(chunk)
    await value._handle_backend_event("bot-stopped-speaking")
    await value._handle_backend_event("bot-stopped-speaking")
    response_types = [message["payload"]["response_type"] for message in websocket.sent]
    assert response_types == ["CHUNK", "FINAL"]
    prompt = websocket.sent[0]["payload"]["prompts"][0]
    assert prompt["is_barge_in_enabled"] is True


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
