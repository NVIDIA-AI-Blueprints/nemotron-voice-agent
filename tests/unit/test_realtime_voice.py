# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

from realtime.gateway import create_realtime_controller
from realtime.protocol import RealtimeProtocolError
from realtime.session import CanonicalRealtimeSession, RealtimeSessionCapabilities

_ARIA = "Magpie-Multilingual.EN-US.Aria"
_CLAIRE = "Magpie-Multilingual.EN-US.Claire"
_ES_LUNA = "Magpie-Multilingual.ES-US.Luna"


def _session() -> CanonicalRealtimeSession:
    return CanonicalRealtimeSession(
        model="test-realtime-model",
        voice=_ARIA,
        capabilities=RealtimeSessionCapabilities(voices=frozenset({_ARIA, _CLAIRE, _ES_LUNA})),
    )


class StrictVoiceValidationTests(unittest.TestCase):
    def test_known_voice_is_kept_exactly(self) -> None:
        session = _session()
        updated = session.apply_update({"audio": {"output": {"voice": _CLAIRE}}})
        self.assertEqual(updated["audio"]["output"]["voice"], _CLAIRE)

    def test_unknown_voice_is_rejected_without_default_fallback(self) -> None:
        session = _session()
        with self.assertRaises(RealtimeProtocolError) as raised:
            session.apply_update({"audio": {"output": {"voice": "alloy"}}})
        self.assertEqual(raised.exception.code, "unsupported_capability")
        self.assertEqual(raised.exception.param, "session.audio.output.voice")
        self.assertEqual(session.public_view()["audio"]["output"]["voice"], _ARIA)

    def test_binding_a_prepared_voice_preserves_the_resolved_allowlist(self) -> None:
        session = _session()
        session.bind_output_voice(_CLAIRE)

        self.assertEqual(session.public_view()["audio"]["output"]["voice"], _CLAIRE)
        self.assertEqual(session.capabilities.voices, frozenset({_ARIA, _CLAIRE, _ES_LUNA}))
        updated = session.apply_update({"audio": {"output": {"voice": _ARIA}}})
        self.assertEqual(updated["audio"]["output"]["voice"], _ARIA)

    def test_trusted_discovery_can_extend_but_clients_cannot_bypass_voice_allowlist(self) -> None:
        session = CanonicalRealtimeSession(
            model="test-realtime-model",
            voice=_ARIA,
            capabilities=RealtimeSessionCapabilities(voices=frozenset({_ARIA})),
        )

        with self.assertRaises(RealtimeProtocolError):
            session.apply_update({"audio": {"output": {"voice": _ES_LUNA}}})

        session.bind_discovered_output_voice(_ES_LUNA)

        self.assertEqual(session.public_view()["audio"]["output"]["voice"], _ES_LUNA)
        self.assertEqual(session.capabilities.voices, frozenset({_ARIA, _ES_LUNA}))

    def test_voice_cannot_change_after_output_audio_starts(self) -> None:
        session = _session()
        session.begin_response("resp_test")
        session.mark_output_audio_started("resp_test")
        with self.assertRaises(RealtimeProtocolError) as raised:
            session.apply_update({"audio": {"output": {"voice": _CLAIRE}}})
        self.assertEqual(raised.exception.code, "immutable_field")
        self.assertEqual(session.public_view()["audio"]["output"]["voice"], _ARIA)


class MultilingualVoiceRoutingTests(unittest.TestCase):
    def test_preferred_voice_must_match_requested_language(self) -> None:
        from examples.shared import prewarm as prewarm_mod

        cached = {
            "voices": [
                {"id": _ARIA, "language": "en-US"},
                {"id": _CLAIRE, "language": "en-US"},
                {"id": _ES_LUNA, "language": "es-US"},
            ]
        }
        with patch.object(prewarm_mod.config_store, "get", return_value=cached):
            self.assertEqual(
                prewarm_mod.resolve_voice_for_language(
                    "en-US",
                    _CLAIRE,
                    server="tts.example:443",
                    model="magpie",
                ),
                _CLAIRE,
            )
            self.assertEqual(
                prewarm_mod.resolve_voice_for_language(
                    "es-US",
                    _CLAIRE,
                    server="tts.example:443",
                    model="magpie",
                ),
                _ES_LUNA,
            )

    def test_missing_language_returns_empty_for_readiness_rejection(self) -> None:
        from examples.shared import prewarm as prewarm_mod

        cached = {"voices": [{"id": _ARIA, "language": "en-US"}], "defaultVoiceId": _ARIA}
        with patch.object(prewarm_mod.config_store, "get", return_value=cached):
            resolved = prewarm_mod.resolve_voice_for_language(
                "fr-FR",
                _ARIA,
                server="tts.example:443",
                model="magpie",
            )
        self.assertEqual(resolved, "")

    def test_exact_route_never_uses_an_unqualified_voice_cache(self) -> None:
        from examples.shared import prewarm as prewarm_mod

        unqualified = {"voices": [{"id": _ES_LUNA, "language": "es-US"}]}

        def cached(key: str, default=None):
            return unqualified if key == "tts" else default

        with patch.object(prewarm_mod.config_store, "get", side_effect=cached) as get_mock:
            resolved = prewarm_mod.resolve_voice_for_language(
                "es-US",
                server="exact-tts.example:443",
                model="exact-model",
            )

        self.assertEqual(resolved, "")
        self.assertNotIn("tts", [call.args[0] for call in get_mock.call_args_list])


class GetTtsConfigCacheTests(unittest.TestCase):
    def test_cache_hit_skips_prewarm(self) -> None:
        from examples.shared import prewarm as prewarm_mod

        cached = {
            "voices": [{"id": _ARIA}],
            "languages": [{"code": "en-US"}],
            "defaultVoiceId": _ARIA,
            "server": "tts.example:443",
        }
        with (
            patch.object(prewarm_mod.config_store, "get", return_value=cached) as get_mock,
            patch.object(prewarm_mod.config_store, "set") as set_mock,
            patch.object(prewarm_mod, "prewarm_tts") as prewarm_mock,
        ):
            result = prewarm_mod.get_tts_config(
                "tts.example:443",
                _CLAIRE,
                "",
                "magpie",
            )
        prewarm_mock.assert_not_called()
        get_mock.assert_called_once()
        set_mock.assert_called_once_with("tts", result)
        self.assertEqual(result["voices"], cached["voices"])
        self.assertEqual(result["defaultVoiceId"], _CLAIRE)

    def test_cache_miss_calls_prewarm(self) -> None:
        from examples.shared import prewarm as prewarm_mod

        fetched = {
            "voices": [{"id": _ARIA}],
            "defaultVoiceId": _ARIA,
        }
        with (
            patch.object(prewarm_mod.config_store, "get", return_value=None),
            patch.object(prewarm_mod, "prewarm_tts", return_value=fetched) as prewarm_mock,
        ):
            result = prewarm_mod.get_tts_config("tts.example:443", "v", "fid", "model")
        prewarm_mock.assert_called_once_with("tts.example:443", "v", "fid", "model")
        self.assertEqual(result, fetched)


class RealtimeCatalogCapabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_initial_voice_patch_is_rejected_before_deferred_discovery(self) -> None:
        def sanitize(data: dict[str, Any], **_: Any) -> dict[str, Any]:
            return {
                "pipeline_mode": "generic-assistant",
                "model_id": "test-model",
                "tts_voice_id": _ARIA,
                **data,
            }

        resolution_modes: list[bool] = []

        async def resolve(config: dict[str, Any], discover_output_voices: bool) -> RealtimeSessionCapabilities:
            resolution_modes.append(discover_output_voices)
            if discover_output_voices:
                raise RuntimeError("catalog unavailable")
            return RealtimeSessionCapabilities(voices=frozenset({_ARIA}))

        with self.assertRaises(RealtimeProtocolError) as raised:
            await create_realtime_controller(
                sanitize_session_config=sanitize,
                initial_session={
                    "type": "realtime",
                    "audio": {
                        "output": {
                            "voice": _CLAIRE,
                            "unexpected": True,
                        }
                    },
                },
                resolve_session_capabilities=resolve,
            )

        self.assertEqual(raised.exception.code, "unknown_parameter")
        self.assertEqual(raised.exception.param, "session.audio.output.unexpected")
        self.assertEqual(resolution_modes, [False])

    async def test_text_session_defers_voice_discovery_and_hydrates_once(self) -> None:
        def sanitize(data: dict[str, Any], **_: Any) -> dict[str, Any]:
            return {
                "pipeline_mode": "generic-assistant",
                "model_id": "test-model",
                "tts_voice_id": _ARIA,
                **data,
            }

        resolution_modes: list[bool] = []

        async def resolve(config: dict[str, Any], discover_output_voices: bool) -> RealtimeSessionCapabilities:
            self.assertEqual(config["tts_voice_id"], _ARIA)
            resolution_modes.append(discover_output_voices)
            if discover_output_voices:
                await asyncio.sleep(0)
                return RealtimeSessionCapabilities(voices=frozenset({_ARIA, _CLAIRE}))
            return RealtimeSessionCapabilities(voices=frozenset({_ARIA}))

        controller = await create_realtime_controller(
            sanitize_session_config=sanitize,
            initial_session={"type": "realtime", "output_modalities": ["text"]},
            resolve_session_capabilities=resolve,
        )

        self.assertEqual(resolution_modes, [False])
        self.assertEqual(controller.session.capabilities.voices, frozenset({_ARIA}))
        await asyncio.gather(
            controller.ensure_output_voice_capabilities(),
            controller.ensure_output_voice_capabilities(),
        )
        self.assertEqual(resolution_modes, [False, True])
        self.assertEqual(controller.session.capabilities.voices, frozenset({_ARIA, _CLAIRE}))

    async def test_failed_lazy_voice_discovery_is_retryable_and_does_not_publish_partial_state(self) -> None:
        attempts = 0

        async def resolve() -> frozenset[str]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("catalog unavailable")
            return frozenset({_ARIA, _CLAIRE})

        from realtime.controller import RealtimeSessionController

        controller = RealtimeSessionController(
            model="test-model",
            voice=_ARIA,
            runtime_config={},
            capabilities=RealtimeSessionCapabilities(voices=frozenset({_ARIA})),
            output_voice_resolver=resolve,
            output_voices_resolved=False,
        )

        with self.assertRaisesRegex(RuntimeError, "catalog unavailable"):
            await controller.ensure_output_voice_capabilities()
        self.assertEqual(controller.session.capabilities.voices, frozenset({_ARIA}))

        await controller.ensure_output_voice_capabilities()
        self.assertEqual(attempts, 2)
        self.assertEqual(controller.session.capabilities.voices, frozenset({_ARIA, _CLAIRE}))

    async def test_initial_session_can_select_any_voice_from_resolved_route(self) -> None:
        def sanitize(data: dict[str, Any], **_: Any) -> dict[str, Any]:
            config = dict(data)
            config.setdefault("pipeline_mode", "generic-assistant")
            config.setdefault("model_id", "test-model")
            config.setdefault("tts_voice_id", _ARIA)
            return config

        resolution_modes: list[bool] = []

        async def resolve(config: dict[str, Any], discover_output_voices: bool) -> RealtimeSessionCapabilities:
            self.assertEqual(config["tts_voice_id"], _ARIA)
            resolution_modes.append(discover_output_voices)
            voices = frozenset({_ARIA, _CLAIRE}) if discover_output_voices else frozenset({_ARIA})
            return RealtimeSessionCapabilities(voices=voices)

        controller = await create_realtime_controller(
            sanitize_session_config=sanitize,
            initial_session={
                "type": "realtime",
                "audio": {"output": {"voice": _CLAIRE}},
            },
            resolve_session_capabilities=resolve,
        )

        self.assertEqual(controller.public_session()["audio"]["output"]["voice"], _CLAIRE)
        self.assertEqual(controller.session.capabilities.voices, frozenset({_ARIA, _CLAIRE}))
        self.assertEqual(resolution_modes, [False, True])

    async def test_initial_transcription_language_is_canonicalized_and_projected(self) -> None:
        def sanitize(data: dict[str, Any], **_: Any) -> dict[str, Any]:
            config = dict(data)
            config.setdefault("pipeline_mode", "generic-assistant")
            config.setdefault("model_id", "test-model")
            config.setdefault("tts_voice_id", _ARIA)
            config.setdefault("asr_model", "asr-model")
            return config

        async def resolve(
            config: dict[str, Any],
            discover_output_voices: bool,  # noqa: ARG001
        ) -> RealtimeSessionCapabilities:
            return RealtimeSessionCapabilities(
                voices=frozenset({_ARIA}),
                input_transcription_models=frozenset({"asr-model"}),
                input_transcription_language_aliases=(("en", "en-US"), ("en-us", "en-US")),
            )

        controller = await create_realtime_controller(
            sanitize_session_config=sanitize,
            initial_session={
                "type": "realtime",
                "audio": {"input": {"transcription": {"model": "asr-model", "language": "en"}}},
            },
            resolve_session_capabilities=resolve,
        )

        transcription = controller.public_session()["audio"]["input"]["transcription"]
        self.assertEqual(transcription, {"model": "asr-model", "language": "en-US"})
        self.assertEqual(controller.runtime_config["asr_language_code"], "en-US")

    async def test_server_resolver_uses_exact_catalogs_and_unambiguous_language_aliases(self) -> None:
        import server

        config = {
            "pipeline_mode": "generic-assistant",
            "base_url": "http://llm.example/v1",
            "model_id": "test-model",
            "realtime_model_max_output_tokens": 2048,
            "tts_server": "tts.example:443",
            "tts_function_id": "tts-function",
            "tts_model": "tts-model",
            "tts_voice_id": _ARIA,
            "asr_server": "asr.example:443",
            "asr_function_id": "asr-function",
            "asr_model": "asr-model",
        }
        llm_entry = {
            "base_url": config["base_url"],
            "model_id": config["model_id"],
            "supports_tokenize": True,
            "realtime_max_output_tokens": 2048,
        }
        tts_entry = {
            "server": config["tts_server"],
            "function_id": config["tts_function_id"],
            "model": config["tts_model"],
        }
        asr_entry = {
            "server": config["asr_server"],
            "function_id": config["asr_function_id"],
            "model": config["asr_model"],
        }
        run_blocking = AsyncMock(side_effect=AssertionError("capability resolution must not perform I/O"))
        with (
            patch.object(
                server,
                "_bind_example_context_by_key",
                return_value={"key": "generic-assistant", "slots": ["llm", "tts", "asr"]},
            ),
            patch.object(server, "_trusted_realtime_service_entry", side_effect=[llm_entry, tts_entry, asr_entry]),
            patch.object(server, "_run_blocking", run_blocking),
            patch.object(
                server,
                "peek_cached_tts_config",
                return_value={
                    "voices": [{"id": _ARIA}, {"id": _CLAIRE}, {"id": _ES_LUNA}],
                    "languages": ["en-US", "es-US"],
                },
            ),
            patch.object(
                server,
                "peek_cached_asr_config",
                return_value={"languages": ["en-US", "es-ES", "es-US"]},
            ),
        ):
            capabilities = await server._resolve_realtime_session_capabilities(config)

        run_blocking.assert_not_awaited()
        self.assertEqual(capabilities.voices, frozenset({_ARIA, _CLAIRE, _ES_LUNA}))
        self.assertEqual(
            {(audio_format.type, audio_format.rate) for audio_format in capabilities.input_formats},
            {("audio/pcm", 24000), ("audio/pcma", None), ("audio/pcmu", None)},
        )
        self.assertEqual(capabilities.output_formats, capabilities.input_formats)
        self.assertEqual(capabilities.input_transcription_models, frozenset({"asr-model"}))
        self.assertTrue(capabilities.supports_input_transcription_disable)
        self.assertTrue(capabilities.truncation)
        aliases = dict(capabilities.input_transcription_language_aliases)
        self.assertEqual(aliases["en"], "en-US")
        self.assertEqual(aliases["en-us"], "en-US")
        self.assertNotIn("es", aliases)

    async def test_server_resolver_discovers_all_voices_when_exact_route_cache_is_cold(self) -> None:
        import server

        config = {
            "pipeline_mode": "generic-assistant",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "model_id": "test-model",
            "tts_server": "tts.example:443",
            "tts_function_id": "tts-function",
            "tts_model": "tts-model",
            "tts_voice_id": _ARIA,
        }
        llm_entry = {
            "base_url": config["base_url"],
            "model_id": config["model_id"],
        }
        tts_entry = {
            "server": config["tts_server"],
            "function_id": config["tts_function_id"],
            "model": config["tts_model"],
        }
        with (
            patch.object(
                server,
                "_bind_example_context_by_key",
                return_value={"key": "generic-assistant", "slots": ["llm", "tts"]},
            ),
            patch.object(server, "_trusted_realtime_service_entry", side_effect=[llm_entry, tts_entry]),
            patch.object(server, "peek_cached_tts_config", return_value=None),
            patch.object(
                server,
                "_run_blocking",
                AsyncMock(
                    return_value={
                        "voices": [{"id": _ARIA}, {"id": _CLAIRE}, {"id": _ES_LUNA}],
                        "defaultVoiceId": _ARIA,
                    }
                ),
            ) as run_blocking,
        ):
            capabilities = await server._resolve_realtime_session_capabilities(config)

        run_blocking.assert_awaited_once_with(
            server.get_tts_config,
            config["tts_server"],
            config["tts_voice_id"],
            config["tts_function_id"],
            config["tts_model"],
            timeout=server._CONNECT_PREWARM_TIMEOUT_SECS,
        )
        self.assertEqual(capabilities.voices, frozenset({_ARIA, _CLAIRE, _ES_LUNA}))
        session = CanonicalRealtimeSession(
            model="test-realtime-model",
            voice=_ARIA,
            capabilities=capabilities,
        )
        self.assertEqual(
            session.apply_update({"audio": {"output": {"voice": _CLAIRE}}})["audio"]["output"]["voice"],
            _CLAIRE,
        )

    async def test_server_resolver_rejects_failed_cold_voice_discovery(self) -> None:
        import server

        config = {
            "pipeline_mode": "generic-assistant",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "model_id": "test-model",
            "tts_server": "tts.example:443",
            "tts_function_id": "tts-function",
            "tts_model": "tts-model",
            "tts_voice_id": _ARIA,
        }
        llm_entry = {
            "base_url": config["base_url"],
            "model_id": config["model_id"],
        }
        tts_entry = {
            "server": config["tts_server"],
            "function_id": config["tts_function_id"],
            "model": config["tts_model"],
        }
        with (
            patch.object(
                server,
                "_bind_example_context_by_key",
                return_value={"key": "generic-assistant", "slots": ["llm", "tts"]},
            ),
            patch.object(server, "_trusted_realtime_service_entry", side_effect=[llm_entry, tts_entry]),
            patch.object(server, "peek_cached_tts_config", return_value=None),
            patch.object(
                server,
                "_run_blocking",
                AsyncMock(return_value={"voices": [], "error": "unavailable"}),
            ),
            self.assertRaisesRegex(RuntimeError, "voice catalog is unavailable"),
        ):
            await server._resolve_realtime_session_capabilities(config)

    async def test_server_resolver_advertises_truncation_only_for_supported_cascades(self) -> None:
        import server

        config = {
            "base_url": "http://llm.example/v1",
            "model_id": "test-model",
            "tts_server": "tts.example:443",
            "tts_function_id": "tts-function",
            "tts_model": "tts-model",
            "tts_voice_id": _ARIA,
        }
        tts_entry = {
            "server": config["tts_server"],
            "function_id": config["tts_function_id"],
            "model": config["tts_model"],
        }
        for pipeline_mode, supports_tokenize, expected in (
            ("generic-assistant", True, True),
            ("generic-assistant", False, False),
            ("omni-assistant", True, False),
        ):
            llm_entry = {
                "base_url": config["base_url"],
                "model_id": config["model_id"],
                "supports_tokenize": supports_tokenize,
            }
            selected_config = {**config, "pipeline_mode": pipeline_mode}
            if supports_tokenize:
                llm_entry["realtime_max_output_tokens"] = 2048
                selected_config["realtime_model_max_output_tokens"] = 2048
            with (
                self.subTest(pipeline_mode=pipeline_mode, supports_tokenize=supports_tokenize),
                patch.object(
                    server,
                    "_bind_example_context_by_key",
                    return_value={"key": pipeline_mode, "slots": ["llm", "tts"]},
                ),
                patch.object(
                    server,
                    "_trusted_realtime_service_entry",
                    side_effect=[llm_entry, tts_entry],
                ),
                patch.object(
                    server,
                    "peek_cached_tts_config",
                    return_value={"voices": [{"id": _ARIA}], "defaultVoiceId": _ARIA},
                ),
            ):
                capabilities = await server._resolve_realtime_session_capabilities(selected_config)

            self.assertEqual(capabilities.truncation, expected)

    def test_tokenizer_route_requires_a_valid_realtime_model_output_cap(self) -> None:
        import server

        for value in (None, True, 0, 4097, "2048"):
            entry = {"supports_tokenize": True}
            if value is not None:
                entry["realtime_max_output_tokens"] = value
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                server._trusted_realtime_model_max_output_tokens(entry)

        self.assertEqual(
            server._trusted_realtime_model_max_output_tokens(
                {"supports_tokenize": True, "realtime_max_output_tokens": 2048}
            ),
            2048,
        )
        self.assertIsNone(server._trusted_realtime_model_max_output_tokens({}))


if __name__ == "__main__":
    unittest.main()
