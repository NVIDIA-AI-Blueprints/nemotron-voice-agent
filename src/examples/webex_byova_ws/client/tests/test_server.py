"""Listener endpoint tests."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer

from webex_byova_ws_adapter.config import AdapterConfig
from webex_byova_ws_adapter.server import create_app


def local_config() -> AdapterConfig:
    """Build explicit local-only listener settings."""
    return AdapterConfig(
        allow_plaintext_local=True,
        auth_enabled=False,
        allow_auth_disabled_local=True,
    )


def session_start(conversation_id: str) -> dict:
    """Build the first canonical request for a Cisco socket."""
    return {
        "type": "VOICE_VA_REQUEST",
        "seq": 1,
        "ts": "2026-09-07T00:00:00Z",
        "conversation_id": conversation_id,
        "payload": {
            "conversation_id": conversation_id,
            "customer_org_id": "org-1",
            "voice_va_input_type": {"event_input": {"event_type": "SESSION_START"}},
        },
    }


async def test_health_is_marked_local_operator() -> None:
    """Keep the non-Cisco health endpoint clearly scoped."""
    async with TestClient(TestServer(create_app(local_config()))) as client:
        response = await client.get("/health")
        assert response.status == 200
        assert await response.json() == {"status": "ok", "scope": "local-operator"}


async def test_catalog_uses_unwrapped_short_lived_schema() -> None:
    """Return the official unwrapped discovery response."""
    async with TestClient(TestServer(create_app(local_config()))) as client:
        websocket = await client.ws_connect("/v1/listVirtualAgents")
        await websocket.send_json({"customer_org_id": "org-1"})
        response = await websocket.receive_json()
        assert response == {
            "virtual_agents": [
                {
                    "id": "nemotron-generic",
                    "name": "Nemotron Generic",
                    "description": "NVIDIA Nemotron voice agent",
                }
            ]
        }
        await websocket.close()


async def test_catalog_rejects_non_boolean_default_flag() -> None:
    """Reject discovery requests that violate the optional field type."""
    async with TestClient(TestServer(create_app(local_config()))) as client:
        websocket = await client.ws_connect("/v1/listVirtualAgents")
        await websocket.send_json(
            {
                "customer_org_id": "org-1",
                "is_default_virtual_agent_enabled": "true",
            }
        )
        message = await websocket.receive()
        assert message.type.name == "CLOSE"


async def test_voice_socket_requires_session_start_first() -> None:
    """Reject media sockets that try to attach without canonical startup."""
    async with TestClient(TestServer(create_app(local_config()))) as client:
        websocket = await client.ws_connect("/v1/va")
        request = session_start("call-media-first")
        request["payload"]["voice_va_input_type"] = {
            "audio_input": {
                "caller_audio_b64": None,
                "encoding": "MULAW_FORMAT",
                "sample_rate_hertz": 8000,
            }
        }
        await websocket.send_json(request)
        assert (await websocket.receive()).type.name == "CLOSE"


async def test_second_active_socket_for_conversation_is_rejected() -> None:
    """Keep mutable conversation state owned by the original socket."""
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingConversation:
        def __init__(self, *_: object) -> None:
            self.closed = False
            self.closed_event = asyncio.Event()

        async def run(self, _: dict, __: object) -> None:
            entered.set()
            await release.wait()
            self.closed = True
            self.closed_event.set()

    async with TestClient(TestServer(create_app(local_config()))) as client:
        with patch("webex_byova_ws_adapter.server.Conversation", BlockingConversation):
            first = await client.ws_connect("/v1/va")
            await first.send_json(session_start("call-duplicate"))
            await entered.wait()

            second = await client.ws_connect("/v1/va")
            await second.send_json(session_start("call-duplicate"))
            assert (await second.receive()).type.name == "CLOSE"

            release.set()
            await first.close()
