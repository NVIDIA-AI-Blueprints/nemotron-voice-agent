# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Regression tests for Cisco Webex BYOVA adapter boundaries."""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, patch

from cisco_webex_byova_adapter.auth import CiscoJwsValidator, _CachedJwkSet
from cisco_webex_byova_adapter.config import AdapterConfig
from cisco_webex_byova_adapter.generated import byova_common_pb2, voicevirtualagent_pb2
from cisco_webex_byova_adapter.nemotron_bridge import NemotronSession
from cisco_webex_byova_adapter.service import VoiceVirtualAgentServicer, _SessionEntry


class CiscoJwsValidatorTests(unittest.IsolatedAsyncioTestCase):
    """Cover fixed-algorithm verification and JWK cache refresh behavior."""

    async def test_validation_pins_rs256(self) -> None:
        """Ignore the unverified header algorithm and allow only RS256."""
        config = AdapterConfig(
            expected_jwt_issuer="https://idbroker.webex.com/idb",
            expected_jwt_audience="NemotronVoiceAgent",
        )
        validator = CiscoJwsValidator(config)

        with (
            patch(
                "cisco_webex_byova_adapter.auth.jwt.get_unverified_header",
                return_value={"kid": "key-1", "alg": "HS256"},
            ),
            patch.object(validator, "_get_key_for_kid", AsyncMock(return_value={"kid": "key-1"})),
            patch(
                "cisco_webex_byova_adapter.auth.jwt.algorithms.RSAAlgorithm.from_jwk",
                return_value=object(),
            ),
            patch(
                "cisco_webex_byova_adapter.auth.jwt.decode",
                return_value={
                    "iss": config.expected_jwt_issuer,
                    "aud": config.expected_jwt_audience,
                    "sub": "callAudioData",
                    "exp": 1,
                },
            ) as decode,
        ):
            await validator.validate("signed-token")

        self.assertEqual(decode.call_args.kwargs["algorithms"], ["RS256"])

    async def test_missing_kid_refreshes_current_jwk_cache(self) -> None:
        """Refresh a current JWK cache once when a rotated key appears."""
        issuer = "https://idbroker.webex.com/idb"
        config = AdapterConfig(expected_jwt_issuer=issuer, expected_jwt_audience="NemotronVoiceAgent")
        validator = CiscoJwsValidator(config)
        validator._cache[issuer] = _CachedJwkSet(
            keys={"old-key": {"kid": "old-key"}},
            expires_at=time.monotonic() + 3600,
        )
        rotated_key = {"kid": "rotated-key"}
        refreshed = _CachedJwkSet(
            keys={"rotated-key": rotated_key},
            expires_at=time.monotonic() + 3600,
        )

        with patch.object(validator, "_fetch_jwk_set", return_value=refreshed) as fetch:
            result = await validator._get_key_for_kid(issuer, "rotated-key")

        self.assertEqual(result, rotated_key)
        fetch.assert_called_once_with(issuer)


class AdapterConfigTests(unittest.TestCase):
    """Cover startup validation for TLS listener configuration."""

    def test_partial_tls_configuration_is_rejected(self) -> None:
        """Reject startup when only one TLS path is configured."""
        for cert_path, key_path in (("cert.pem", ""), ("", "key.pem")):
            with self.subTest(cert_path=cert_path, key_path=key_path):
                config = AdapterConfig(
                    enable_auth=False,
                    tls_cert_path=cert_path,
                    tls_key_path=key_path,
                )
                with self.assertRaisesRegex(ValueError, "must be configured together"):
                    config.validate()

    def test_complete_tls_configuration_is_accepted(self) -> None:
        """Accept startup when both TLS paths are configured."""
        config = AdapterConfig(
            enable_auth=False,
            tls_cert_path="cert.pem",
            tls_key_path="key.pem",
        )
        config.validate()
        self.assertTrue(config.tls_enabled)


class NemotronSessionConnectionTests(unittest.IsolatedAsyncioTestCase):
    """Cover the stateless adapter-to-backend WebSocket handshake."""

    async def test_start_uses_direct_websocket_without_session_routing(self) -> None:
        """Use the example selected when the backend deployment started."""
        websocket = AsyncMock()
        connect = AsyncMock(return_value=websocket)
        reader_loop = AsyncMock()
        session = NemotronSession(
            config=AdapterConfig(
                enable_auth=False,
                nemotron_voice_agent_ws="ws://127.0.0.1:7860",
            ),
            conversation_id="stateless-connection",
        )

        with (
            patch("cisco_webex_byova_adapter.nemotron_bridge.websockets.connect", connect),
            patch.object(NemotronSession, "_reader_loop", reader_loop),
        ):
            await session.start()
            await session.reader_task

        uri = connect.await_args.args[0]
        self.assertEqual(uri, "ws://127.0.0.1:7860/api/ws")
        self.assertNotIn("?", uri)
        websocket.send.assert_awaited_once()


class _FakeSession:
    """Minimal Nemotron session used by service stream tests."""

    def __init__(self, *outbound_items: dict[str, object]) -> None:
        self.outbound_queue: asyncio.Queue = asyncio.Queue()
        for item in outbound_items:
            self.outbound_queue.put_nowait(item)
        self.caller_resample_state = None
        self.transfer_requested = False
        self.end_session_requested = False
        self.closed = False
        self.sent_audio: list[bytes] = []

    async def send_audio(self, audio: bytes) -> None:
        self.sent_audio.append(audio)

    async def close(self) -> None:
        self.closed = True


async def _request_stream(*requests: voicevirtualagent_pb2.VoiceVARequest):
    for request in requests:
        yield request


def _audio_request(conversation_id: str) -> voicevirtualagent_pb2.VoiceVARequest:
    return voicevirtualagent_pb2.VoiceVARequest(
        conversation_id=conversation_id,
        audio_input=voicevirtualagent_pb2.VoiceInput(
            caller_audio=b"\x00\x00" * 160,
            encoding=voicevirtualagent_pb2.VoiceInput.LINEAR16_FORMAT,
            sample_rate_hertz=16000,
        ),
    )


def _event_request(conversation_id: str, event_type: int) -> voicevirtualagent_pb2.VoiceVARequest:
    return voicevirtualagent_pb2.VoiceVARequest(
        conversation_id=conversation_id,
        event_input=byova_common_pb2.EventInput(event_type=event_type),
    )


class VoiceVirtualAgentLifecycleTests(unittest.IsolatedAsyncioTestCase):
    """Ensure each terminal stream path emits exactly one FINAL response."""

    def _servicer_with_session(self, conversation_id: str, session: _FakeSession) -> VoiceVirtualAgentServicer:
        servicer = VoiceVirtualAgentServicer(AdapterConfig(enable_auth=False))
        servicer._sessions[conversation_id] = _SessionEntry(session)
        servicer._ensure_session_reaper_started = AsyncMock()
        return servicer

    async def test_mid_turn_session_end_is_returned_and_session_is_closed(self) -> None:
        """Return one SESSION_END response when it arrives during audio."""
        conversation_id = "mid-turn-session-end"
        session = _FakeSession()
        servicer = self._servicer_with_session(conversation_id, session)
        requests = _request_stream(
            _audio_request(conversation_id),
            _event_request(conversation_id, byova_common_pb2.EventInput.SESSION_END),
        )

        responses = [response async for response in servicer.ProcessCallerInput(requests, object())]

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].response_type, voicevirtualagent_pb2.VoiceVAResponse.FINAL)
        self.assertEqual(responses[0].output_events[0].event_type, byova_common_pb2.OutputEvent.SESSION_END)
        self.assertTrue(session.closed)
        self.assertNotIn(conversation_id, servicer._sessions)

    async def test_bridge_error_emits_one_final_and_closes_session(self) -> None:
        """Return one error FINAL during a caller turn."""
        conversation_id = "bridge-error"
        session = _FakeSession({"kind": "error", "error": "bridge failed"})
        servicer = self._servicer_with_session(conversation_id, session)

        responses = [
            response
            async for response in servicer.ProcessCallerInput(
                _request_stream(_audio_request(conversation_id)),
                object(),
            )
        ]

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].response_type, voicevirtualagent_pb2.VoiceVAResponse.FINAL)
        self.assertEqual(responses[0].output_events[0].event_type, byova_common_pb2.OutputEvent.CUSTOM_EVENT)
        self.assertEqual(responses[0].output_events[0].name, "nemotron-error")
        self.assertTrue(session.closed)
        self.assertNotIn(conversation_id, servicer._sessions)

    async def test_intro_bridge_error_emits_one_final_and_closes_session(self) -> None:
        """Return one error FINAL during the initial bot turn."""
        conversation_id = "intro-bridge-error"
        session = _FakeSession({"kind": "error", "error": "intro failed"})
        servicer = self._servicer_with_session(conversation_id, session)

        responses = [
            response
            async for response in servicer.ProcessCallerInput(
                _request_stream(
                    _event_request(conversation_id, byova_common_pb2.EventInput.SESSION_START),
                ),
                object(),
            )
        ]

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].response_type, voicevirtualagent_pb2.VoiceVAResponse.FINAL)
        self.assertEqual(responses[0].output_events[0].name, "nemotron-error")
        self.assertTrue(session.closed)
        self.assertNotIn(conversation_id, servicer._sessions)


if __name__ == "__main__":
    unittest.main()
