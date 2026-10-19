# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import unittest

from utils import PROJECT_ROOT, load_yaml_file


class SingleGpuDiarizationComposeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = PROJECT_ROOT / "docker/docker-compose.nemo-speech-cpp.yaml"
        self.compose = load_yaml_file(self.path)
        self.services = self.compose["services"]

    def test_only_generic_speech_service_runs_diarization(self) -> None:
        speech = self.services["nemo-speech"]

        self.assertEqual(speech["profiles"], ["generic-assistant/single-gpu"])
        self.assertIn("NEMO_SPEECH_ASR_DIAR_MODEL_PATH", speech["environment"])
        for name in ("nemo-speech-frontend-backend", "nemo-speech-multilingual", "nemo-speech-tts"):
            self.assertNotIn("environment", self.services[name])
        frontend_backend = self.services["nemo-speech-frontend-backend"]
        self.assertEqual(frontend_backend["profiles"], ["frontend-backend-agent/single-gpu"])

    def test_reuses_existing_generic_speech_service_and_port(self) -> None:
        speech = self.services["nemo-speech"]

        self.assertNotIn("nemo-speech-diarizer", self.services)
        self.assertIn("generic-assistant/single-gpu", speech["profiles"])
        self.assertEqual(speech["ports"], ["50051:50051"])
        ports = [str(port) for service in self.services.values() for port in service.get("ports", [])]
        host_ports = {port.split(":")[0] for port in ports}
        self.assertNotIn("50053", host_ports)

    def test_diarization_build_uses_matching_models(self) -> None:
        speech = self.services["nemo-speech"]
        command = "\n".join(speech["command"])

        self.assertEqual(
            speech["environment"]["NEMO_SPEECH_ASR_DIAR_MODEL_PATH"],
            "/models/diarization/Nemotron-3-Diarization.q8_0.gguf",
        )
        self.assertEqual(speech["environment"]["NEMO_SPEECH_ASR_VAD_MODEL_PATH"], "/models/vad/silero-v6.2.0.gguf")
        self.assertIn("--asr.model.path=/models/asr/nemotron-speech-streaming-en-0.6b.q8_0.gguf", command)
        self.assertIn("--tts.magpie-model=/models/magpie-tts/magpie_tts_multilingual_357m.v2607.f16.gguf", command)

    def test_single_gpu_catalog_offers_diarization_for_nemo_speech(self) -> None:
        services = load_yaml_file(PROJECT_ROOT / "services.yaml")
        asr = services["single-gpu"]["asr"]["nemotron-asr-streaming-english"]

        self.assertEqual(asr["server"], "nemo-speech:50051")
        self.assertIs(asr["speaker_diarization_supported"], True)
        self.assertEqual(asr["speaker_diarization_max_speakers"], 8)


if __name__ == "__main__":
    unittest.main()
