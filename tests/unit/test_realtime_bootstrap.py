# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

from __future__ import annotations

import base64
import json
import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
from realtime_helpers import FakeWebSocket

from realtime.auth import (
    RealtimeAuthenticationError,
    authenticate_realtime_master_key,
    authenticate_realtime_websocket,
    issue_realtime_client_secret,
    verify_realtime_client_secret,
)
from realtime.bootstrap import parse_realtime_client_secret_request
from realtime.gateway import RealtimeModelRoute, _select_realtime_subprotocol, handle_realtime_websocket
from realtime.protocol import RealtimeProtocolError
from realtime.session import RealtimeSessionCapabilities

_API_KEY = "sk-test-realtime-master-key-with-enough-entropy"
_MODEL = "test-realtime-model"
_VOICE = "test-realtime-voice"


def _sanitize(data: dict[str, Any], **_: Any) -> dict[str, Any]:
    config = dict(data)
    config.setdefault("pipeline_mode", "generic-assistant")
    config.setdefault("model_id", _MODEL)
    config.setdefault("tts_voice_id", _VOICE)
    config.setdefault("prompt_key", "generic_assistant")
    config.setdefault("prompt_content", "Default instructions")
    return config


@contextmanager
def _standalone_realtime_test_app():
    import server

    async def resolve_capabilities(
        config: dict[str, Any],
        discover_output_voices: bool,  # noqa: ARG001
    ) -> RealtimeSessionCapabilities:
        if config["tts_voice_id"] != _VOICE:
            raise AssertionError("the sanitized voice must reach capability resolution")
        return RealtimeSessionCapabilities(voices=frozenset({_VOICE}))

    with (
        patch.dict(
            os.environ,
            {
                "REALTIME_API_KEY": _API_KEY,
                "REALTIME_SERVICE_PLATFORM": "singlegpu",
            },
        ),
        patch.object(server, "_sanitize_realtime_config", side_effect=_sanitize),
        patch.object(server, "_validate_realtime_bootstrap_selectors"),
        patch.object(server, "_resolve_realtime_server_tools", return_value=[]),
        patch.object(server, "_resolve_realtime_delegate_tools", return_value=[]),
        patch.object(server, "_resolve_realtime_server_tool_schemas", return_value=[]),
        patch.object(server, "_resolve_realtime_delegate_tool_schemas", return_value=[]),
        patch.object(server, "_resolve_realtime_session_capabilities", side_effect=resolve_capabilities),
        patch.object(server, "resolve_voice_for_language", return_value=_VOICE),
    ):
        yield server.create_realtime_app()


class RealtimeClientSecretAuthTests(unittest.TestCase):
    def test_signed_secret_round_trip_is_stateless_and_reusable(self) -> None:
        session = {
            "type": "realtime",
            "instructions": "Bound before connect",
            "nvidia": {"pipeline_mode": "generic-assistant", "llm_id": "cloud-nim:model"},
        }
        secret = issue_realtime_client_secret(
            api_key=_API_KEY,
            session=session,
            issued_at=1_000,
            expires_at=1_600,
        )

        first = verify_realtime_client_secret(secret, api_key=_API_KEY, now=1_001)
        second = verify_realtime_client_secret(secret, api_key=_API_KEY, now=1_599)

        self.assertTrue(secret.startswith("ek_"))
        self.assertEqual(first, second)
        self.assertEqual(first.session, session)
        self.assertEqual(first.expires_at, 1_600)

    def test_client_secret_encrypts_bound_mcp_credentials(self) -> None:
        authorization = "private-mcp-bearer"
        header_value = "private-mcp-header"
        session = {
            "type": "realtime",
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "private",
                    "server_url": "https://mcp.example.test/rpc",
                    "authorization": authorization,
                    "headers": {"X-Service-Key": header_value},
                }
            ],
        }

        secret = issue_realtime_client_secret(
            api_key=_API_KEY,
            session=session,
            issued_at=1_000,
            expires_at=1_600,
        )

        decoded_segments = [
            base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))
            for segment in secret.removeprefix("ek_").split(".")
        ]
        self.assertFalse(any(authorization.encode() in segment for segment in decoded_segments))
        self.assertFalse(any(header_value.encode() in segment for segment in decoded_segments))
        self.assertEqual(
            verify_realtime_client_secret(secret, api_key=_API_KEY, now=1_001).session,
            session,
        )

    def test_expired_tampered_and_wrong_deployment_secrets_fail_closed(self) -> None:
        secret = issue_realtime_client_secret(
            api_key=_API_KEY,
            session={"type": "realtime"},
            issued_at=1_000,
            expires_at=1_600,
        )
        encoded_payload, signature = secret.removeprefix("ek_").split(".", 1)
        replacement = "A" if encoded_payload[0] != "A" else "B"
        tampered = f"ek_{replacement}{encoded_payload[1:]}.{signature}"

        cases = (
            (secret, _API_KEY, 1_600),
            (tampered, _API_KEY, 1_001),
            (secret, "sk-another-deployment-key", 1_001),
        )
        for candidate, key, now in cases:
            with self.subTest(candidate=candidate[:8], now=now), self.assertRaises(RealtimeAuthenticationError):
                verify_realtime_client_secret(candidate, api_key=key, now=now)

    def test_header_and_browser_auth_share_the_same_secret_contract(self) -> None:
        secret = issue_realtime_client_secret(
            api_key=_API_KEY,
            session={"type": "realtime", "instructions": "bound"},
            issued_at=1_000,
            expires_at=1_600,
        )
        server = authenticate_realtime_websocket(
            {"authorization": f"Bearer {secret}"},
            api_key=_API_KEY,
            now=1_001,
        )
        browser_headers = {
            "sec-websocket-protocol": f"openai-insecure-api-key.{secret}, realtime",
        }
        browser = authenticate_realtime_websocket(browser_headers, api_key=_API_KEY, now=1_001)

        self.assertEqual(server.claims, browser.claims)
        self.assertEqual(_select_realtime_subprotocol(SimpleNamespace(headers=browser_headers)), "realtime")
        self.assertIsNone(
            _select_realtime_subprotocol(
                SimpleNamespace(headers={"sec-websocket-protocol": f"openai-insecure-api-key.{secret}"})
            )
        )

    def test_master_header_needs_no_subprotocol_and_conflicts_are_rejected(self) -> None:
        authenticated = authenticate_realtime_websocket(
            {"authorization": f"Bearer {_API_KEY}"},
            api_key=_API_KEY,
        )
        self.assertTrue(authenticated.enabled)
        self.assertIsNone(authenticated.claims)
        self.assertIsNone(_select_realtime_subprotocol(SimpleNamespace(headers={})))

        with self.assertRaises(RealtimeAuthenticationError):
            authenticate_realtime_websocket(
                {
                    "authorization": f"Bearer {_API_KEY}",
                    "sec-websocket-protocol": "realtime, openai-insecure-api-key.ek_conflict",
                },
                api_key=_API_KEY,
            )
        with self.assertRaises(RealtimeAuthenticationError):
            authenticate_realtime_websocket(
                {"sec-websocket-protocol": f"realtime, openai-insecure-api-key.{_API_KEY}"},
                api_key=_API_KEY,
            )

    def test_unconfigured_gateway_preserves_local_development_access(self) -> None:
        result = authenticate_realtime_websocket({}, api_key="")
        self.assertFalse(result.enabled)
        self.assertIsNone(result.claims)

    def test_non_ascii_credentials_fail_without_compare_errors(self) -> None:
        headers = {"authorization": "Bearer é"}
        with self.assertRaises(RealtimeAuthenticationError):
            authenticate_realtime_websocket(headers, api_key=_API_KEY)
        self.assertFalse(authenticate_realtime_master_key(headers, api_key=_API_KEY))

    def test_master_key_is_never_accepted_through_browser_subprotocol(self) -> None:
        prefixed_master = "ek_master-key-that-must-stay-server-side-000000"
        header = authenticate_realtime_websocket(
            {"authorization": f"Bearer {prefixed_master}"},
            api_key=prefixed_master,
        )
        self.assertTrue(header.enabled)
        self.assertIsNone(header.claims)

        with self.assertRaises(RealtimeAuthenticationError):
            authenticate_realtime_websocket(
                {"sec-websocket-protocol": f"realtime, openai-insecure-api-key.{prefixed_master}"},
                api_key=prefixed_master,
            )


class RealtimeClientSecretRequestTests(unittest.TestCase):
    def test_defaults_and_ga_expiry_bounds(self) -> None:
        parsed = parse_realtime_client_secret_request({})
        self.assertEqual(parsed.ttl_seconds, 600)
        self.assertEqual(parsed.session, {"type": "realtime"})

        for seconds in (10, 7_200):
            parsed = parse_realtime_client_secret_request(
                {"expires_after": {"anchor": "created_at", "seconds": seconds}}
            )
            self.assertEqual(parsed.ttl_seconds, seconds)

        for seconds in (9, 7_201, True):
            with self.subTest(seconds=seconds), self.assertRaises(RealtimeProtocolError):
                parse_realtime_client_secret_request({"expires_after": {"seconds": seconds}})

    def test_unsafe_backend_routing_fields_are_not_tokenizable(self) -> None:
        unsafe_fields = {
            "base_url": "http://169.254.169.254/latest/meta-data",
            "asr_server": "attacker.example:443",
            "tts_function_id": "private-function-id",
            "api_key": "must-not-enter-token",
        }
        for field, value in unsafe_fields.items():
            with self.subTest(field=field), self.assertRaises(RealtimeProtocolError) as raised:
                parse_realtime_client_secret_request({"session": {"type": "realtime", "nvidia": {field: value}}})
            self.assertEqual(raised.exception.code, "unknown_parameter")
            self.assertEqual(raised.exception.param, f"session.nvidia.{field}")

    def test_only_realtime_sessions_are_supported(self) -> None:
        for session in ({}, {"type": "transcription"}):
            with self.subTest(session=session), self.assertRaises(RealtimeProtocolError):
                parse_realtime_client_secret_request({"session": session})


class BootstrappedGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_standard_model_selects_a_complete_trusted_route(self) -> None:
        public_model = "nvidia/test-realtime-multilingual"
        ws = FakeWebSocket([json.dumps({"type": "response.create"})])
        ws.query_params["model"] = public_model
        captured: dict[str, Any] = {}

        def resolve_route(model: str | None, selectors: dict[str, Any]) -> RealtimeModelRoute:
            self.assertEqual(model, public_model)
            self.assertEqual(selectors, {})
            return RealtimeModelRoute(
                model=public_model,
                runtime_config={
                    "pipeline_mode": "multilingual-assistant",
                    "prompt_key": "multilingual_voice_assistant",
                    "model_id": "private-provider-model",
                },
            )

        async def start_bot(websocket, config, controller) -> None:  # noqa: ARG001
            captured["config"] = config
            captured["controller"] = controller

        await handle_realtime_websocket(
            ws,  # type: ignore[arg-type]
            sanitize_session_config=_sanitize,
            resolve_model_route=resolve_route,
            start_bot=start_bot,
        )

        self.assertEqual(ws.sent[0]["session"]["model"], public_model)
        self.assertEqual(captured["config"]["pipeline_mode"], "multilingual-assistant")
        self.assertEqual(captured["config"]["model_id"], "private-provider-model")
        self.assertEqual(captured["controller"].session.public_view()["model"], public_model)

    async def test_query_and_client_secret_models_must_match_before_route_resolution(self) -> None:
        ws = FakeWebSocket([])
        ws.query_params["model"] = "nvidia/query-model"
        resolver_called = False

        def resolve_route(model: str | None, selectors: dict[str, Any]) -> RealtimeModelRoute:  # noqa: ARG001
            nonlocal resolver_called
            resolver_called = True
            raise AssertionError("conflicting models must fail before route resolution")

        await handle_realtime_websocket(
            ws,  # type: ignore[arg-type]
            sanitize_session_config=_sanitize,
            initial_session={"type": "realtime", "model": "nvidia/token-model"},
            resolve_model_route=resolve_route,
        )

        self.assertFalse(resolver_called)
        self.assertEqual(ws.sent[0]["error"]["code"], "model_not_available")
        self.assertEqual(ws.sent[0]["error"]["param"], "model")
        self.assertEqual(ws.close_code, 1008)

    async def test_initial_event_rejects_non_finite_json_as_invalid_json(self) -> None:
        for constant in ("NaN", "Infinity", "-Infinity"):
            ws = FakeWebSocket(
                [
                    '{"type":"session.update","event_id":"non_finite","session":{"tools":'
                    f'[{{"type":"function","name":"bad","parameters":{{"minimum":{constant}}}}}]}}'
                ]
            )

            with self.subTest(constant=constant):
                await handle_realtime_websocket(ws, sanitize_session_config=_sanitize)  # type: ignore[arg-type]
                self.assertEqual(
                    [event["type"] for event in ws.sent],
                    ["session.created", "conversation.created", "error"],
                )
                self.assertEqual(ws.sent[-1]["error"]["code"], "invalid_json")

    async def test_token_session_is_visible_in_created_and_can_be_overridden(self) -> None:
        ws = FakeWebSocket(
            [
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {"type": "realtime", "instructions": "Connection override"},
                    }
                )
            ]
        )
        captured: dict[str, Any] = {}

        async def start_bot(websocket, config, controller) -> None:  # noqa: ARG001
            captured["config"] = config
            captured["controller"] = controller

        await handle_realtime_websocket(
            ws,  # type: ignore[arg-type]
            sanitize_session_config=_sanitize,
            initial_session={"type": "realtime", "instructions": "Token instructions"},
            start_bot=start_bot,
        )

        self.assertEqual(ws.sent[0]["type"], "session.created")
        self.assertEqual(ws.sent[0]["session"]["instructions"], "Token instructions")
        self.assertEqual(ws.sent[2]["type"], "session.updated")
        self.assertEqual(ws.sent[2]["session"]["instructions"], "Connection override")
        self.assertEqual(captured["config"]["prompt_content"], "Connection override")

    async def test_created_is_not_delayed_by_backend_readiness(self) -> None:
        ws = FakeWebSocket([json.dumps({"type": "response.create"})])

        async def unavailable(config: dict[str, Any]) -> None:  # noqa: ARG001
            raise RuntimeError("backend unavailable")

        await handle_realtime_websocket(
            ws,  # type: ignore[arg-type]
            sanitize_session_config=_sanitize,
            initial_session={"type": "realtime", "instructions": "Already bound"},
            ensure_services_ready=unavailable,
        )

        self.assertEqual(
            [event["type"] for event in ws.sent],
            ["session.created", "conversation.created", "error"],
        )
        self.assertEqual(ws.sent[0]["session"]["instructions"], "Already bound")
        self.assertEqual(ws.sent[2]["error"]["code"], "services_not_ready")

    async def test_runtime_preparation_is_deferred_until_after_created_events(self) -> None:
        ws = FakeWebSocket([json.dumps({"type": "response.create"})])
        captured: dict[str, Any] = {}

        async def prepare(config: dict[str, Any]) -> dict[str, Any]:
            self.assertEqual(
                [event["type"] for event in ws.sent],
                ["session.created", "conversation.created"],
            )
            return {**config, "tts_voice_id": "prepared-voice"}

        async def start_bot(websocket, config, controller) -> None:  # noqa: ARG001
            captured["config"] = config

        await handle_realtime_websocket(
            ws,  # type: ignore[arg-type]
            sanitize_session_config=_sanitize,
            prepare_initial_runtime=prepare,
            start_bot=start_bot,
        )

        self.assertEqual(
            [event["type"] for event in ws.sent[:3]],
            [
                "session.created",
                "conversation.created",
                "session.updated",
            ],
        )
        self.assertEqual(ws.sent[2]["session"]["audio"]["output"]["voice"], "prepared-voice")
        self.assertEqual(captured["config"]["tts_voice_id"], "prepared-voice")

    async def test_token_and_query_models_must_match_selected_catalog_model(self) -> None:
        for initial_session, query_model in (
            ({"type": "realtime", "model": "wrong-model"}, _MODEL),
            ({"type": "realtime", "model": _MODEL}, "wrong-model"),
        ):
            ws = FakeWebSocket([])
            ws.query_params["model"] = query_model
            with self.subTest(initial_session=initial_session, query_model=query_model):
                await handle_realtime_websocket(
                    ws,  # type: ignore[arg-type]
                    sanitize_session_config=_sanitize,
                    initial_session=initial_session,
                )
                self.assertEqual(ws.sent[0]["type"], "error")
                self.assertIn(ws.sent[0]["error"]["code"], {"immutable_field", "model_not_available"})
                self.assertEqual(ws.close_code, 1008)


class RealtimeServerSurfaceTests(unittest.IsolatedAsyncioTestCase):
    def test_logical_component_override_is_bound_to_one_catalog_source(self) -> None:
        import server

        with patch.dict(os.environ, {"REALTIME_SERVICE_PLATFORM": "server"}):
            route = server._resolve_realtime_model_route(
                "nvidia/nemotron-realtime-client-tools",
                {"llm_id": "nemotron-lightning"},
                "generic-assistant",
            )
        self.assertEqual(route.runtime_config["llm_id"], "self-hosted:server:nemotron-lightning")

        with (
            patch.dict(os.environ, {"REALTIME_SERVICE_PLATFORM": "singlegpu"}),
            self.assertRaises(RealtimeProtocolError) as raised,
        ):
            server._resolve_realtime_model_route(
                "nvidia/nemotron-realtime-client-tools",
                {"llm_id": "self-hosted:server:nemotron-lightning"},
                "generic-assistant",
            )
        self.assertEqual(raised.exception.code, "invalid_value")
        self.assertEqual(raised.exception.param, "session.nvidia.llm_id")

    def test_standalone_route_exposes_only_realtime_surface(self) -> None:
        with _standalone_realtime_test_app() as app:
            self.assertEqual(
                {getattr(route, "path", "") for route in app.routes},
                {"/health", "/v1/realtime/client_secrets", "/v1/realtime"},
            )

    async def test_sdk_client_secret_round_trip_preserves_private_session_state(self) -> None:
        from openai.types.realtime.client_secret_create_response import ClientSecretCreateResponse

        with _standalone_realtime_test_app() as app:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/v1/realtime/client_secrets",
                    headers={"Authorization": f"Bearer {_API_KEY}"},
                    json={
                        "expires_after": {"anchor": "created_at", "seconds": 60},
                        "session": {
                            "type": "realtime",
                            "instructions": "SDK bootstrap",
                            "max_output_tokens": 64,
                            "output_modalities": ["text"],
                            "tools": [
                                {
                                    "type": "mcp",
                                    "server_label": "private-test",
                                    "server_url": "https://mcp.example.test/rpc",
                                    "authorization": "private-bearer",
                                    "headers": {"X-Private-Key": "private-header"},
                                }
                            ],
                        },
                    },
                )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        created = ClientSecretCreateResponse.model_validate(body)
        self.assertTrue(created.value.startswith("ek_"))
        self.assertGreater(created.expires_at, 0)
        self.assertTrue(created.session.id.startswith("sess_"))
        self.assertEqual(created.session.object, "realtime.session")
        self.assertEqual(created.session.type, "realtime")
        self.assertEqual(created.session.instructions, "SDK bootstrap")
        self.assertEqual(created.session.model, "nvidia/nemotron-realtime")
        self.assertIsNone(created.session.expires_at)
        self.assertNotIn("authorization", body["session"]["tools"][0])
        self.assertNotIn("headers", body["session"]["tools"][0])

        claims = verify_realtime_client_secret(created.value, api_key=_API_KEY)
        self.assertNotIn("id", claims.session)
        self.assertNotIn("object", claims.session)
        self.assertEqual(claims.session["instructions"], "SDK bootstrap")
        self.assertEqual(claims.session["max_output_tokens"], 64)
        self.assertEqual(claims.session["output_modalities"], ["text"])
        self.assertEqual(claims.session["audio"]["output"]["voice"], _VOICE)
        self.assertEqual(claims.session["tools"][0]["authorization"], "private-bearer")
        self.assertEqual(claims.session["tools"][0]["headers"], {"X-Private-Key": "private-header"})
        self.assertEqual(claims.session["model"], "nvidia/nemotron-realtime")
        self.assertEqual(claims.session["nvidia"]["pipeline_mode"], "generic-assistant")
        self.assertEqual(claims.session["nvidia"]["prompt_key"], "generic_assistant")
        self.assertEqual(
            claims.session["nvidia"]["llm_id"],
            "self-hosted:singlegpu:nemotron-lightning",
        )

    async def test_client_secret_profiles_resolve_and_reject_conflicts(self) -> None:
        authorization = {"Authorization": f"Bearer {_API_KEY}"}
        with _standalone_realtime_test_app() as app:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                client_tools = await client.post(
                    "/v1/realtime/client_secrets",
                    headers=authorization,
                    json={"session": {"type": "realtime", "model": "nvidia/nemotron-realtime-client-tools"}},
                )
                pipeline_default = await client.post(
                    "/v1/realtime/client_secrets",
                    headers=authorization,
                    json={"session": {"type": "realtime", "nvidia": {"pipeline_mode": "multilingual-assistant"}}},
                )
                unknown_model = await client.post(
                    "/v1/realtime/client_secrets",
                    headers=authorization,
                    json={"session": {"type": "realtime", "model": "nvidia/unknown-realtime"}},
                )
                conflicting_pipeline = await client.post(
                    "/v1/realtime/client_secrets",
                    headers=authorization,
                    json={
                        "session": {
                            "type": "realtime",
                            "model": "nvidia/nemotron-realtime-client-tools",
                            "nvidia": {"pipeline_mode": "multilingual-assistant"},
                        }
                    },
                )
                conflicting_prompt = await client.post(
                    "/v1/realtime/client_secrets",
                    headers=authorization,
                    json={
                        "session": {
                            "type": "realtime",
                            "model": "nvidia/nemotron-realtime-client-tools",
                            "nvidia": {"prompt_key": "generic_assistant"},
                        }
                    },
                )

        self.assertEqual(client_tools.status_code, 200)
        self.assertEqual(client_tools.json()["session"]["model"], "nvidia/nemotron-realtime-client-tools")
        self.assertEqual(
            client_tools.json()["session"]["nvidia"]["prompt_key"],
            "generic_assistant_without_tools",
        )
        self.assertEqual(pipeline_default.status_code, 200)
        self.assertEqual(pipeline_default.json()["session"]["model"], "nvidia/nemotron-realtime-multilingual")
        self.assertEqual(
            pipeline_default.json()["session"]["nvidia"]["pipeline_mode"],
            "multilingual-assistant",
        )
        for response in (unknown_model, conflicting_pipeline, conflicting_prompt):
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["error"]["code"], "model_not_available")
        self.assertEqual(conflicting_prompt.json()["error"]["param"], "session.nvidia.prompt_key")

    async def test_client_secret_endpoint_rejects_unsafe_and_nonfinite_input(self) -> None:
        authorization = {"Authorization": f"Bearer {_API_KEY}"}
        with _standalone_realtime_test_app() as app:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                unsafe = await client.post(
                    "/v1/realtime/client_secrets",
                    headers=authorization,
                    json={
                        "session": {
                            "type": "realtime",
                            "nvidia": {"base_url": "http://169.254.169.254/latest/meta-data"},
                        }
                    },
                )
                non_finite = await client.post(
                    "/v1/realtime/client_secrets",
                    headers={**authorization, "Content-Type": "application/json"},
                    content='{"session":{"type":"realtime","tools":'
                    '[{"type":"function","name":"bad","parameters":{"minimum":NaN}}]}}',
                )

        self.assertEqual(unsafe.status_code, 400)
        self.assertEqual(unsafe.json()["error"]["code"], "unknown_parameter")
        self.assertEqual(non_finite.status_code, 400)
        self.assertEqual(non_finite.json()["error"]["code"], "invalid_json")

    async def test_text_secret_skips_cold_tts_catalog_and_audio_secret_resolves_it_once(self) -> None:
        import server

        runtime = {
            "pipeline_mode": "multilingual-assistant",
            "base_url": "http://llm.example/v1",
            "model_id": "provider-model",
            "prompt_key": "multilingual_voice_assistant",
            "prompt_content": "Default instructions",
            "tts_server": "tts.example:443",
            "tts_function_id": "tts-function",
            "tts_model": "tts-model",
            "tts_voice_id": _VOICE,
            "asr_server": "asr.example:443",
            "asr_function_id": "asr-function",
            "asr_model": "asr-model",
        }
        trusted_entries = {
            "llm": {
                "base_url": runtime["base_url"],
                "model_id": runtime["model_id"],
            },
            "tts": {
                "server": runtime["tts_server"],
                "function_id": runtime["tts_function_id"],
                "model": runtime["tts_model"],
            },
            "asr": {
                "server": runtime["asr_server"],
                "function_id": runtime["asr_function_id"],
                "model": runtime["asr_model"],
            },
        }

        def sanitize(data: dict[str, Any], **_: Any) -> dict[str, Any]:
            return {**runtime, **data}

        def trusted_entry(config: dict[str, Any], category: str) -> dict[str, Any]:  # noqa: ARG001
            return trusted_entries[category]

        tts_catalog = {"voices": [{"id": _VOICE}], "defaultVoiceId": _VOICE}
        run_blocking = AsyncMock(return_value=tts_catalog)
        with (
            patch.dict(os.environ, {"REALTIME_API_KEY": _API_KEY}),
            patch.object(server, "_sanitize_realtime_config", side_effect=sanitize),
            patch.object(
                server,
                "_resolve_realtime_model_route",
                return_value=RealtimeModelRoute(model=_MODEL, runtime_config=runtime),
            ),
            patch.object(server, "_resolve_realtime_server_tools", return_value=[]),
            patch.object(server, "_resolve_realtime_delegate_tools", return_value=[]),
            patch.object(server, "_resolve_realtime_server_tool_schemas", return_value=[]),
            patch.object(server, "_resolve_realtime_delegate_tool_schemas", return_value=[]),
            patch.object(
                server,
                "_bind_example_context_by_key",
                return_value={"key": "multilingual-assistant", "slots": ["llm", "tts", "asr"]},
            ),
            patch.object(server, "_trusted_realtime_service_entry", side_effect=trusted_entry),
            patch.object(server, "peek_cached_tts_config", side_effect=[None, tts_catalog]) as peek_tts,
            patch.object(server, "peek_cached_asr_config", return_value=None) as peek_asr,
            patch.object(server, "_run_blocking", run_blocking),
        ):
            app = server.create_realtime_app()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                text_response = await client.post(
                    "/v1/realtime/client_secrets",
                    headers={"Authorization": f"Bearer {_API_KEY}"},
                    json={"session": {"type": "realtime", "output_modalities": ["text"]}},
                )
                self.assertEqual(text_response.status_code, 200)
                peek_tts.assert_not_called()
                self.assertEqual(peek_asr.call_count, 1)
                run_blocking.assert_not_awaited()

                audio_response = await client.post(
                    "/v1/realtime/client_secrets",
                    headers={"Authorization": f"Bearer {_API_KEY}"},
                    json={"session": {"type": "realtime", "output_modalities": ["audio"]}},
                )

        self.assertEqual(audio_response.status_code, 200)
        self.assertEqual(peek_tts.call_count, 1)
        self.assertEqual(peek_asr.call_count, 3)
        run_blocking.assert_awaited_once_with(
            server.get_tts_config,
            runtime["tts_server"],
            runtime["tts_voice_id"],
            runtime["tts_function_id"],
            runtime["tts_model"],
            timeout=server._CONNECT_PREWARM_TIMEOUT_SECS,
        )

    async def test_mint_is_disabled_without_a_master_key(self) -> None:
        import server

        with patch.dict(os.environ, {"REALTIME_API_KEY": ""}):
            app = server.create_realtime_app()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/v1/realtime/client_secrets", json={})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "realtime_auth_not_configured")


if __name__ == "__main__":
    unittest.main()
