"""Listener endpoint tests."""

from __future__ import annotations

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
