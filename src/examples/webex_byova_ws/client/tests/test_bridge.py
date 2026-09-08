"""Nemotron RTVI bridge tests."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from pipecat.frames.protobufs import frames_pb2

from webex_byova_ws_adapter.bridge import NemotronBridge
from webex_byova_ws_adapter.config import AdapterConfig


class FakeBackend:
    """Minimal protobuf WebSocket."""

    def __init__(self) -> None:
        self.frames: list[bytes] = []
        self.send = AsyncMock(side_effect=self.frames.append)
        self.close = AsyncMock()

    def __aiter__(self):
        async def empty():
            if False:
                yield b""

        return empty()


@pytest.mark.asyncio
async def test_bridge_opens_api_ws_and_sends_client_ready() -> None:
    backend = FakeBackend()
    config = AdapterConfig(
        backend_ws_url="ws://nemotron:7860/api/ws",
        allow_plaintext_local=True,
        auth_enabled=False,
        allow_auth_disabled_local=True,
    )
    bridge = NemotronBridge(config, "call-1")
    with patch("webex_byova_ws_adapter.bridge.websockets.connect", AsyncMock(return_value=backend)) as connect:
        await bridge.start()
        await bridge.reader_task
    assert connect.await_args.args[0] == "ws://nemotron:7860/api/ws"
    frame = frames_pb2.Frame.FromString(backend.frames[0])
    message = json.loads(frame.message.data)
    assert message["type"] == "client-ready"


@pytest.mark.asyncio
async def test_bridge_maps_required_rtvi_events() -> None:
    config = AdapterConfig(
        allow_plaintext_local=True,
        auth_enabled=False,
        allow_auth_disabled_local=True,
    )
    bridge = NemotronBridge(config, "call-1")
    await bridge._handle_message(json.dumps({"type": "user-started-speaking"}))
    await bridge._handle_message(json.dumps({"type": "user-stopped-speaking"}))
    await bridge._handle_message(json.dumps({"type": "server-message", "data": {"type": "user-turn-finalized"}}))
    await bridge._handle_message(json.dumps({"type": "bot-started-speaking"}))
    await bridge._handle_message(json.dumps({"type": "bot-stopped-speaking"}))
    assert [(await bridge.events.get())["type"] for _ in range(4)] == [
        "user-started-speaking",
        "user-turn-finalized",
        "bot-started-speaking",
        "bot-stopped-speaking",
    ]
    assert bridge.events.empty()


@pytest.mark.asyncio
async def test_dtmf_and_custom_events_are_forwarded_as_client_messages() -> None:
    backend = FakeBackend()
    config = AdapterConfig(
        allow_plaintext_local=True,
        auth_enabled=False,
        allow_auth_disabled_local=True,
    )
    bridge = NemotronBridge(config, "call-1", websocket=backend)
    await bridge.send_input("dtmf", {"dtmf_events": ["DTMF_DIGIT_ONE"]})
    await bridge.send_input("custom-event", {"name": "crm", "parameters": {"x": 1}})
    decoded = [json.loads(frames_pb2.Frame.FromString(raw).message.data)["data"]["type"] for raw in backend.frames]
    assert decoded == ["webex-dtmf", "webex-custom-event"]
