# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from realtime.voice import resolve_realtime_tts_voice, tts_routing_changed


class SoftVoiceResolveTests(unittest.TestCase):
    def test_known_voice_kept(self) -> None:
        config = {
            "tts_voice_id": "Magpie-Multilingual.EN-US.Claire",
            "tts_server": "tts.example:443",
        }
        with (
            patch("realtime.voice.load_service_entry", return_value={"voice_id": "Magpie-Multilingual.EN-US.Aria"}),
            patch(
                "realtime.voice.get_tts_config",
                return_value={
                    "voices": [{"id": "Magpie-Multilingual.EN-US.Aria"}, {"id": "Magpie-Multilingual.EN-US.Claire"}],
                    "defaultVoiceId": "Magpie-Multilingual.EN-US.Aria",
                },
            ),
        ):
            resolved = resolve_realtime_tts_voice(config, voice_was_set=True)
        self.assertEqual(resolved, "Magpie-Multilingual.EN-US.Claire")
        self.assertEqual(config["tts_voice_id"], "Magpie-Multilingual.EN-US.Claire")

    def test_unknown_voice_falls_back_to_default(self) -> None:
        config = {
            "tts_voice_id": "alloy",
            "tts_server": "tts.example:443",
        }
        with (
            patch("realtime.voice.load_service_entry", return_value={"voice_id": "Magpie-Multilingual.EN-US.Aria"}),
            patch(
                "realtime.voice.get_tts_config",
                return_value={
                    "voices": [{"id": "Magpie-Multilingual.EN-US.Aria"}, {"id": "Magpie-Multilingual.EN-US.Claire"}],
                    "defaultVoiceId": "Magpie-Multilingual.EN-US.Aria",
                },
            ),
        ):
            resolved = resolve_realtime_tts_voice(config, voice_was_set=True)
        self.assertEqual(resolved, "Magpie-Multilingual.EN-US.Aria")
        self.assertEqual(config["tts_voice_id"], "Magpie-Multilingual.EN-US.Aria")

    def test_empty_catalog_uses_yaml_default(self) -> None:
        config = {
            "tts_voice_id": "alloy",
            "tts_server": "tts.example:443",
        }
        with (
            patch("realtime.voice.load_service_entry", return_value={"voice_id": "Magpie-Multilingual.EN-US.Aria"}),
            patch("realtime.voice.get_tts_config", return_value={"voices": [], "defaultVoiceId": ""}),
        ):
            resolved = resolve_realtime_tts_voice(config, voice_was_set=True)
        self.assertEqual(resolved, "Magpie-Multilingual.EN-US.Aria")

    def test_routing_changed_triggers_resolve(self) -> None:
        config = {
            "tts_voice_id": "Magpie-Multilingual.EN-US.Claire",
            "tts_server": "new-tts:443",
        }
        with (
            patch("realtime.voice.load_service_entry", return_value={"voice_id": "Magpie-Multilingual.EN-US.Aria"}),
            patch(
                "realtime.voice.get_tts_config",
                return_value={
                    "voices": [{"id": "Magpie-Multilingual.EN-US.Aria"}],
                    "defaultVoiceId": "Magpie-Multilingual.EN-US.Aria",
                },
            ),
        ):
            resolved = resolve_realtime_tts_voice(config, voice_was_set=False, tts_routing_changed=True)
        self.assertEqual(resolved, "Magpie-Multilingual.EN-US.Aria")

    def test_tts_routing_changed_detects_server(self) -> None:
        self.assertTrue(tts_routing_changed({"tts_server": "a"}, {"tts_server": "b"}))
        self.assertFalse(tts_routing_changed({"tts_server": "a"}, {"tts_server": "a"}))


class GetTtsConfigCacheTests(unittest.TestCase):
    def test_cache_hit_skips_prewarm(self) -> None:
        from examples.shared import prewarm as prewarm_mod

        cached = {
            "voices": [{"id": "Magpie-Multilingual.EN-US.Aria"}],
            "languages": [{"code": "en-US"}],
            "defaultVoiceId": "Magpie-Multilingual.EN-US.Aria",
            "server": "tts.example:443",
        }
        with (
            patch.object(prewarm_mod.config_store, "get", return_value=cached) as get_mock,
            patch.object(prewarm_mod.config_store, "set") as set_mock,
            patch.object(prewarm_mod, "prewarm_tts") as prewarm_mock,
        ):
            result = prewarm_mod.get_tts_config(
                "tts.example:443",
                "Magpie-Multilingual.EN-US.Claire",
                "",
                "magpie",
            )
        prewarm_mock.assert_not_called()
        get_mock.assert_called_once()
        set_mock.assert_called_once_with("tts", result)
        self.assertEqual(result["voices"], cached["voices"])
        self.assertEqual(result["defaultVoiceId"], "Magpie-Multilingual.EN-US.Claire")

    def test_cache_miss_calls_prewarm(self) -> None:
        from examples.shared import prewarm as prewarm_mod

        fetched = {
            "voices": [{"id": "Magpie-Multilingual.EN-US.Aria"}],
            "defaultVoiceId": "Magpie-Multilingual.EN-US.Aria",
        }
        with (
            patch.object(prewarm_mod.config_store, "get", return_value=None),
            patch.object(prewarm_mod, "prewarm_tts", return_value=fetched) as prewarm_mock,
        ):
            result = prewarm_mod.get_tts_config("tts.example:443", "v", "fid", "model")
        prewarm_mock.assert_called_once_with("tts.example:443", "v", "fid", "model")
        self.assertEqual(result, fetched)


class TtsSynthesisWarmupTests(unittest.TestCase):
    def setUp(self) -> None:
        from examples.shared import prewarm as prewarm_mod

        prewarm_mod._TTS_SYNTHESIS_READY_KEYS.clear()

    def tearDown(self) -> None:
        from examples.shared import prewarm as prewarm_mod

        prewarm_mod._TTS_SYNTHESIS_READY_KEYS.clear()

    def test_successful_route_is_warmed_only_once(self) -> None:
        from examples.shared import prewarm as prewarm_mod

        service = SimpleNamespace(
            _initialize_client=Mock(),
            _settings=SimpleNamespace(voice="voice", language="en-US", quality=20),
            _service=SimpleNamespace(synthesize_online=Mock(return_value=iter([object()]))),
        )
        with patch.object(prewarm_mod, "_create_tts_service", return_value=service) as create_service:
            first = prewarm_mod.warmup_tts_synthesis("tts.example:443", "voice", "fid", "model")
            second = prewarm_mod.warmup_tts_synthesis("tts.example:443", "voice", "fid", "model")

        self.assertTrue(first)
        self.assertTrue(second)
        create_service.assert_called_once_with("tts.example:443", "voice", "fid", "model")


if __name__ == "__main__":
    unittest.main()
