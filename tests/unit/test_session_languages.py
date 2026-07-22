# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D107

import sys
import types
import unittest
from unittest.mock import patch


class _FakeRivaSpeechRecognitionConfigRequest:
    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "")
        self.model_name_was_set = "model_name" in kwargs


if "pipecat.services.nvidia.stt" not in sys.modules:
    pipecat = types.ModuleType("pipecat")
    services = types.ModuleType("pipecat.services")
    nvidia = types.ModuleType("pipecat.services.nvidia")
    stt = types.ModuleType("pipecat.services.nvidia.stt")
    tts = types.ModuleType("pipecat.services.nvidia.tts")
    stt.NvidiaSTTService = object
    tts.NvidiaTTSService = object
    tts.NvidiaTTSSettings = object
    sys.modules["pipecat"] = pipecat
    sys.modules["pipecat.services"] = services
    sys.modules["pipecat.services.nvidia"] = nvidia
    sys.modules["pipecat.services.nvidia.stt"] = stt
    sys.modules["pipecat.services.nvidia.tts"] = tts

if "riva.client.proto.riva_asr_pb2" not in sys.modules:
    riva = types.ModuleType("riva")
    client = types.ModuleType("riva.client")
    proto = types.ModuleType("riva.client.proto")
    riva_asr_pb2 = types.ModuleType("riva.client.proto.riva_asr_pb2")

    riva_asr_pb2.RivaSpeechRecognitionConfigRequest = _FakeRivaSpeechRecognitionConfigRequest
    sys.modules["riva"] = riva
    sys.modules["riva.client"] = client
    sys.modules["riva.client.proto"] = proto
    sys.modules["riva.client.proto.riva_asr_pb2"] = riva_asr_pb2

from examples.shared.prewarm import intersect_session_languages, prewarm_asr


class _FakeModelConfig:
    def __init__(self, language_code: str):
        self.parameters = {"language_code": language_code}


class _FakeRecognitionConfig:
    def __init__(self, language_code: str):
        self.model_config = [_FakeModelConfig(language_code)]


class _FallbackASRStub:
    def __init__(self):
        self.requests: list[tuple[str, bool]] = []

    def GetRivaSpeechRecognitionConfig(self, request, timeout: float):
        self.requests.append((request.model_name, request.model_name_was_set))
        return _FakeRecognitionConfig("en-US,fr-FR,multi")


class _FakeNvidiaSTTService:
    last: "_FakeNvidiaSTTService | None" = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._asr_service = types.SimpleNamespace(stub=_FallbackASRStub())
        _FakeNvidiaSTTService.last = self

    def _initialize_client(self) -> None:
        pass


class SessionLanguageCatalogTests(unittest.TestCase):
    def test_empty_asr_catalog_falls_back_to_spanish_only(self) -> None:
        tts_config = {"languages": ["en-US", "es-US", "vi-VN", "zh-CN"], "voices": []}

        self.assertEqual(intersect_session_languages({"languages": []}, tts_config), ["es-US"])

    def test_empty_asr_catalog_does_not_return_tts_languages_when_spanish_is_unavailable(self) -> None:
        tts_config = {"languages": ["en-US", "vi-VN", "zh-CN"], "voices": []}

        self.assertEqual(intersect_session_languages({"languages": []}, tts_config), [])

    def test_runtime_asr_catalog_drives_intersection(self) -> None:
        tts_config = {"languages": ["en-US", "fr-FR"], "voices": []}

        languages = intersect_session_languages(
            {"languages": ["fr-FR"]},
            tts_config,
        )

        self.assertEqual(languages, ["fr-FR"])

    def test_asr_prewarm_uses_default_config_request_without_model_name(self) -> None:
        _FakeNvidiaSTTService.last = None
        with (
            patch("examples.shared.prewarm.NvidiaSTTService", _FakeNvidiaSTTService),
            patch(
                "examples.shared.prewarm.riva_asr_pb2.RivaSpeechRecognitionConfigRequest",
                _FakeRivaSpeechRecognitionConfigRequest,
            ),
            patch("examples.shared.prewarm.config_store.get", return_value=None),
            patch("examples.shared.prewarm.config_store.set"),
        ):
            config = prewarm_asr(
                "parakeet-rnnt-asr:50052",
                "parakeet-1.1b-rnnt-multilingual-asr",
                "",
            )

        self.assertEqual(config["languages"], ["en-US", "fr-FR", "multi"])
        self.assertEqual(config["config_model"], "")
        self.assertIsNotNone(_FakeNvidiaSTTService.last)
        self.assertEqual(
            _FakeNvidiaSTTService.last._asr_service.stub.requests,
            [("", False)],
        )


if __name__ == "__main__":
    unittest.main()
