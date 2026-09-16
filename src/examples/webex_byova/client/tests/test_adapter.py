# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Regression tests for Cisco Webex BYOVA adapter boundaries."""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock, patch

from jwt import InvalidTokenError
from pipecat.frames.protobufs import frames_pb2

from cisco_webex_byova_adapter.auth import CiscoJwsValidator, _CachedJwkSet
from cisco_webex_byova_adapter.config import AdapterConfig
from cisco_webex_byova_adapter.generated import byova_common_pb2, voicevirtualagent_pb2
from cisco_webex_byova_adapter.nemotron_bridge import NemotronSession
from cisco_webex_byova_adapter.service import (
    VoiceVirtualAgentServicer,
    _chunk_response_with_audio,
    _dtmf_symbols,
    _SessionEntry,
)


class CiscoJwsValidatorTests(unittest.IsolatedAsyncioTestCase):
    """Cover fixed-algorithm verification and JWK cache refresh behavior."""

    async def test_validation_rejects_non_rs256(self) -> None:
        """Reject a token whose protected header does not select RS256."""
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
            self.assertRaisesRegex(InvalidTokenError, "RS256"),
        ):
            await validator.validate("Bearer signed-token")

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

    async def test_validation_enforces_configured_datasource_binding(self) -> None:
        """Reject a validly signed token bound to another data source."""
        config = AdapterConfig(
            expected_jwt_issuer="https://idbroker.webex.com/idb",
            expected_jwt_audience="NemotronVoiceAgent",
            expected_datasource_url="nemotron.example:50061",
        )
        validator = CiscoJwsValidator(config)
        with (
            patch(
                "cisco_webex_byova_adapter.auth.jwt.get_unverified_header",
                return_value={"kid": "key-1", "alg": "RS256"},
            ),
            patch.object(
                validator,
                "_get_key_for_kid",
                AsyncMock(return_value={"kid": "key-1", "kty": "RSA", "alg": "RS256"}),
            ),
            patch("cisco_webex_byova_adapter.auth.jwt.algorithms.RSAAlgorithm.from_jwk", return_value=object()),
            patch(
                "cisco_webex_byova_adapter.auth.jwt.decode",
                return_value={"com.cisco.datasource.url": "other.example:50061"},
            ),
            self.assertRaisesRegex(InvalidTokenError, "com.cisco.datasource.url"),
        ):
            await validator.validate("Bearer signed-token")


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

    def test_runtime_bounds_are_validated(self) -> None:
        """Reject ports and timeouts that cannot produce a healthy runtime."""
        invalid = (
            ("NEMOTRON_BYOVA_ADAPTER_GRPC_PORT", {"grpc_port": 0}),
            ("NEMOTRON_BYOVA_ADAPTER_HEALTH_PORT", {"health_http_port": 65536}),
            ("CISCO_JWK_CACHE_TTL_SECS", {"jwk_cache_ttl_secs": 0}),
            ("OUTPUT_IDLE_TIMEOUT_MS", {"output_idle_timeout_ms": 0}),
            ("RESPONSE_IDLE_TIMEOUT_SECS", {"response_idle_timeout_secs": 0}),
            ("FIRST_AUDIO_TIMEOUT_SECS", {"first_audio_timeout_secs": 0}),
            ("DTMF_INTER_DIGIT_TIMEOUT_MS", {"dtmf_inter_digit_timeout_ms": 0}),
            ("ADAPTER_IDLE_SESSION_TIMEOUT_SECS", {"idle_session_timeout_secs": 0}),
        )
        for expected, values in invalid:
            with self.subTest(setting=expected), self.assertRaisesRegex(ValueError, expected):
                AdapterConfig(enable_auth=False, **values).validate()


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


class NemotronCallControlTests(unittest.IsolatedAsyncioTestCase):
    """Cover typed LLM call control and redacted DTMF forwarding."""

    def test_transcript_keywords_do_not_trigger_terminal_actions(self) -> None:
        """Treat transfer-like words as ordinary text without an LLM tool event."""
        session = NemotronSession(config=AdapterConfig(enable_auth=False), conversation_id="no-keywords")
        session._handle_message_payload(
            json.dumps(
                {
                    "type": "server-message",
                    "data": {"type": "user-llm-text", "text": "What does transfer to agent mean?"},
                }
            )
        )
        self.assertIsNone(session.terminal_action)

    def test_terminal_action_is_typed_and_deduplicated(self) -> None:
        """Accept the first typed terminal action and ignore later actions."""
        session = NemotronSession(config=AdapterConfig(enable_auth=False), conversation_id="terminal")
        session._handle_message_payload(
            json.dumps(
                {
                    "type": "server-message",
                    "data": {
                        "type": "webex-call-control",
                        "action": "transfer_to_human",
                        "reason": "caller asked",
                        "metadata": {"route": "billing"},
                    },
                }
            )
        )
        session._handle_message_payload(
            json.dumps(
                {
                    "type": "server-message",
                    "data": {"type": "webex-call-control", "action": "end_call", "reason": "later"},
                }
            )
        )
        self.assertEqual(session.terminal_action, "transfer_to_human")
        self.assertEqual(session.transfer_metadata()["route"], "billing")

    async def test_dtmf_is_forwarded_only_when_complete(self) -> None:
        """Forward one Cisco interaction that completed on its final digit."""
        session = NemotronSession(config=AdapterConfig(enable_auth=False), conversation_id="dtmf")
        session.websocket = AsyncMock()
        self.assertTrue(session.request_keypad_input("phone_number"))

        self.assertEqual(await session.ingest_dtmf(list("4155550123")), "complete")

        serialized = session.websocket.send.await_args.args[0]
        frame = frames_pb2.Frame.FromString(serialized)
        payload = json.loads(frame.message.data)
        self.assertEqual(payload["type"], "webex-dtmf")
        self.assertEqual(
            payload["data"],
            {"field": "phone_number", "value": "4155550123"},
        )

    async def test_trailing_terminator_interaction_is_ignored(self) -> None:
        """Ignore a stray hash the caller presses after the entry completed."""
        session = NemotronSession(config=AdapterConfig(enable_auth=False), conversation_id="termchar")
        session.websocket = AsyncMock()
        session.request_keypad_input("phone_number")

        self.assertEqual(await session.ingest_dtmf(["#"]), "ignored")
        session.websocket.send.assert_not_awaited()
        self.assertEqual(session.pending_dtmf_field, "phone_number")

    async def test_rejected_entry_keeps_same_field_armed_for_retry(self) -> None:
        """Keep the field armed so the caller can immediately re-enter digits."""
        session = NemotronSession(config=AdapterConfig(enable_auth=False), conversation_id="retry")
        session.websocket = AsyncMock()
        session.request_keypad_input("phone_number")

        self.assertEqual(await session.ingest_dtmf(list("41555")), "invalid")
        self.assertEqual(session.pending_dtmf_field, "phone_number")
        self.assertEqual(await session.ingest_dtmf(list("4155550123")), "complete")
        self.assertIsNone(session.pending_dtmf_field)

    async def test_invalid_dtmf_length_requests_retry_without_value(self) -> None:
        """Send only field and error metadata when keypad length is invalid."""
        session = NemotronSession(config=AdapterConfig(enable_auth=False), conversation_id="invalid-dtmf")
        session.websocket = AsyncMock()
        session.request_keypad_input("date_of_birth")

        self.assertEqual(await session.ingest_dtmf(list("0101#")), "invalid")
        serialized = session.websocket.send.await_args.args[0]
        frame = frames_pb2.Frame.FromString(serialized)
        payload = json.loads(frame.message.data)
        self.assertEqual(payload["type"], "webex-dtmf-error")
        self.assertNotIn("value", payload["data"])

    async def test_invalid_calendar_date_requests_retry(self) -> None:
        """Reject an eight-digit DOB that is not a real calendar date."""
        session = NemotronSession(config=AdapterConfig(enable_auth=False), conversation_id="invalid-date")
        session.websocket = AsyncMock()
        session.request_keypad_input("date_of_birth")

        self.assertEqual(await session.ingest_dtmf(list("13322000#")), "invalid")
        serialized = session.websocket.send.await_args.args[0]
        frame = frames_pb2.Frame.FromString(serialized)
        payload = json.loads(frame.message.data)
        self.assertEqual(payload["type"], "webex-dtmf-error")
        self.assertIn("valid date", payload["data"]["reason"])

    async def test_dob_uses_ddmmyyyy_format(self) -> None:
        """Accept a valid day-month-year keypad value."""
        session = NemotronSession(config=AdapterConfig(enable_auth=False), conversation_id="valid-date")
        session.websocket = AsyncMock()
        session.request_keypad_input("date_of_birth")

        self.assertEqual(await session.ingest_dtmf(list("31121990#")), "complete")


class _FakeSession:
    """Minimal Nemotron session used by service stream tests."""

    def __init__(self, *outbound_items: dict[str, object], emit_on_dtmf: bool = False) -> None:
        self.outbound_queue: asyncio.Queue = asyncio.Queue()
        for item in outbound_items:
            self.outbound_queue.put_nowait(item)
        self.caller_resample_state = None
        self.terminal_action = None
        self.pending_dtmf_field = None
        self.dtmf_input_length = 0
        self.closed = False
        self.sent_audio: list[bytes] = []
        self.ingested_dtmf: list[list[str]] = []
        self.emit_on_dtmf = emit_on_dtmf

    async def send_audio(self, audio: bytes) -> None:
        self.sent_audio.append(audio)

    async def ingest_dtmf(self, digits: list[str]) -> str:
        self.ingested_dtmf.append(digits)
        if self.emit_on_dtmf:
            self.outbound_queue.put_nowait({"kind": "audio", "audio": b"\x00\x00" * 200})
            self.outbound_queue.put_nowait({"kind": "final"})
        return "complete"

    async def retry_dtmf(self, reason: str) -> bool:
        del reason
        return False

    def transfer_metadata(self) -> dict[str, str]:
        return {"route": "customer_service"}

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


def _dtmf_request(conversation_id: str, *events: int) -> voicevirtualagent_pb2.VoiceVARequest:
    return voicevirtualagent_pb2.VoiceVARequest(
        conversation_id=conversation_id,
        dtmf_input=byova_common_pb2.DTMFInputs(dtmf_events=events),
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

    async def test_top_level_dtmf_is_forwarded_and_answered(self) -> None:
        """Handle a Cisco turn whose first payload contains keypad input."""
        conversation_id = "top-level-dtmf"
        session = _FakeSession(emit_on_dtmf=True)
        servicer = self._servicer_with_session(conversation_id, session)
        request = _dtmf_request(
            conversation_id,
            byova_common_pb2.DTMF_DIGIT_ONE,
            byova_common_pb2.DTMF_DIGIT_ZERO,
            byova_common_pb2.DTMF_DIGIT_POUND,
        )

        responses = [response async for response in servicer.ProcessCallerInput(_request_stream(request), object())]

        self.assertEqual(session.ingested_dtmf, [["1", "0", "#"]])
        self.assertEqual(
            [response.response_type for response in responses],
            [voicevirtualagent_pb2.VoiceVAResponse.CHUNK, voicevirtualagent_pb2.VoiceVAResponse.FINAL],
        )

    async def test_mid_stream_dtmf_is_merged_with_voice_turn(self) -> None:
        """Consume keypad input while the same Cisco stream carries caller audio."""
        conversation_id = "mid-stream-dtmf"
        session = _FakeSession(emit_on_dtmf=True)
        servicer = self._servicer_with_session(conversation_id, session)
        requests = _request_stream(
            _audio_request(conversation_id),
            _dtmf_request(
                conversation_id,
                byova_common_pb2.DTMF_DIGIT_TWO,
                byova_common_pb2.DTMF_DIGIT_POUND,
            ),
        )

        responses = [response async for response in servicer.ProcessCallerInput(requests, object())]

        self.assertEqual(session.ingested_dtmf, [["2", "#"]])
        self.assertTrue(session.sent_audio)
        self.assertEqual(responses[-1].response_type, voicevirtualagent_pb2.VoiceVAResponse.FINAL)

    def test_turn_final_requests_sensitive_dtmf(self) -> None:
        """Collect keypad-only input that completes on the digit count."""
        session = NemotronSession(config=AdapterConfig(enable_auth=False), conversation_id="input-config")
        session.request_keypad_input("phone_number")
        servicer = VoiceVirtualAgentServicer(AdapterConfig(enable_auth=False, dtmf_inter_digit_timeout_ms=2500))

        response = servicer._build_turn_final(_SessionEntry(session), include_empty_audio=False)

        self.assertEqual(response.input_mode, voicevirtualagent_pb2.INPUT_EVENT_DTMF)
        self.assertTrue(response.input_sensitive)
        config = response.input_handling_config.dtmf_config
        self.assertEqual(config.dtmf_input_length, 10)
        self.assertEqual(config.inter_digit_timeout_msec, 2500)
        self.assertEqual(config.termchar, byova_common_pb2.DTMF_EVENT_UNSPECIFIED)

        chunk = servicer._apply_next_input_config(
            voicevirtualagent_pb2.VoiceVAResponse(
                input_mode=voicevirtualagent_pb2.INPUT_VOICE,
                response_type=voicevirtualagent_pb2.VoiceVAResponse.CHUNK,
            ),
            session,
        )
        self.assertEqual(chunk.input_mode, voicevirtualagent_pb2.INPUT_EVENT_DTMF)
        self.assertTrue(chunk.input_sensitive)
        self.assertEqual(chunk.input_handling_config.dtmf_config.dtmf_input_length, 10)

    def test_audio_chunk_preserves_prompt_text(self) -> None:
        """Keep aligned bot transcript text on streamed Cisco audio."""
        response = _chunk_response_with_audio(b"\x00\x00" * 200, text="Please enter your number.")

        self.assertIsNotNone(response)
        self.assertEqual(response.prompts[0].text, "Please enter your number.")

    def test_audio_chunk_uses_declared_nemotron_sample_rate(self) -> None:
        """Convert one second of 22.05 kHz PCM into one second of 8 kHz mu-law."""
        response = _chunk_response_with_audio(
            b"\x00\x00" * 22_050,
            sample_rate_hz=22_050,
            num_channels=1,
        )

        self.assertIsNotNone(response)
        self.assertEqual(len(response.prompts[0].audio_content), 8_000)

    def test_terminal_action_takes_precedence_over_pending_dtmf(self) -> None:
        """Emit transfer rather than another keypad request once terminal."""
        session = NemotronSession(config=AdapterConfig(enable_auth=False), conversation_id="precedence")
        session.request_keypad_input("date_of_birth")
        session.request_terminal_action(
            "transfer_to_human",
            reason="caller requested",
            metadata={"route": "customer_service"},
        )
        servicer = VoiceVirtualAgentServicer(AdapterConfig(enable_auth=False))

        response = servicer._build_turn_final(_SessionEntry(session), include_empty_audio=False)

        self.assertFalse(response.input_sensitive)
        self.assertEqual(response.output_events[0].event_type, byova_common_pb2.OutputEvent.TRANSFER_TO_AGENT)

    def test_dtmf_enum_mapping_ignores_unsupported_keys(self) -> None:
        """Map decimal digits and pound while ignoring star and unspecified."""
        request = voicevirtualagent_pb2.VoiceVARequest(
            dtmf_input=byova_common_pb2.DTMFInputs(
                dtmf_events=[
                    byova_common_pb2.DTMF_DIGIT_ONE,
                    byova_common_pb2.DTMF_DIGIT_ZERO,
                    byova_common_pb2.DTMF_DIGIT_STAR,
                    byova_common_pb2.DTMF_EVENT_UNSPECIFIED,
                    byova_common_pb2.DTMF_DIGIT_POUND,
                ]
            )
        )
        self.assertEqual(_dtmf_symbols(request), ["1", "0", "#"])


if __name__ == "__main__":
    unittest.main()
