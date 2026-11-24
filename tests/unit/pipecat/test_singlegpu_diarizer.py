# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import unittest

from nemotron_voice_agent.core.catalog import load_yaml_file
from nemotron_voice_agent.core.paths import REPO_ROOT


class SingleGpuDiarizationComposeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = REPO_ROOT / "docker/sidecars/docker-compose.nemo-speech-cpp.yaml"
        self.compose = load_yaml_file(self.path)
        self.services = self.compose["services"]

    def test_only_generic_speech_service_runs_diarization(self) -> None:
        speech = self.services["nemo-speech-asr"]

        self.assertEqual(speech["profiles"], ["generic-assistant/single-gpu"])
        self.assertIn("NEMO_SPEECH_ASR_DIAR_MODEL_PATH", speech["environment"])
        for name in ("nemo-speech-asr-frontend-backend", "nemo-speech-asr-multilingual", "nemo-speech-tts"):
            self.assertNotIn("environment", self.services[name])
        frontend_backend = self.services["nemo-speech-asr-frontend-backend"]
        self.assertEqual(frontend_backend["profiles"], ["frontend-backend-live/single-gpu"])

    def test_asr_and_tts_run_in_separate_services_so_asr_can_scale_to_zero(self) -> None:
        asr_services = ("nemo-speech-asr", "nemo-speech-asr-frontend-backend", "nemo-speech-asr-multilingual")
        for name in asr_services:
            command = "\n".join(self.services[name]["command"])
            self.assertIn("--asr.model.path=", command)
            self.assertNotIn("--tts.", command)
            self.assertIn("--bind=0.0.0.0:50052", command)
            self.assertEqual(self.services[name]["ports"], ["50052:50052"])
        tts = self.services["nemo-speech-tts"]
        self.assertNotIn("--asr.", "\n".join(tts["command"]))
        self.assertEqual(tts["ports"], ["50051:50051"])
        for recipe in ("generic-assistant", "multilingual-assistant", "frontend-backend-live"):
            self.assertIn(f"{recipe}/single-gpu", tts["profiles"])

    def test_reuses_existing_generic_speech_service_and_port(self) -> None:
        speech = self.services["nemo-speech-asr"]

        self.assertNotIn("nemo-speech-diarizer", self.services)
        self.assertIn("generic-assistant/single-gpu", speech["profiles"])
        self.assertEqual(speech["ports"], ["50052:50052"])
        ports = [str(port) for service in self.services.values() for port in service.get("ports", [])]
        host_ports = {port.split(":")[0] for port in ports}
        self.assertNotIn("50053", host_ports)

    def test_diarization_build_uses_matching_models(self) -> None:
        speech = self.services["nemo-speech-asr"]
        command = "\n".join(speech["command"])
        tts_command = "\n".join(self.services["nemo-speech-tts"]["command"])

        self.assertEqual(
            speech["environment"]["NEMO_SPEECH_ASR_DIAR_MODEL_PATH"],
            "/models/diarization/Nemotron-3-Diarization.q8_0.gguf",
        )
        self.assertEqual(speech["environment"]["NEMO_SPEECH_ASR_VAD_MODEL_PATH"], "/models/vad/silero-v6.2.0.gguf")
        self.assertIn("--asr.model.path=/models/asr/nemotron-speech-streaming-en-0.6b.q8_0.gguf", command)
        self.assertIn("--tts.magpie-model=/models/magpie-tts/magpie_tts_multilingual_357m.v2607.f16.gguf", tts_command)

    def test_single_gpu_catalog_offers_diarization_for_nemo_speech(self) -> None:
        services = load_yaml_file(REPO_ROOT / "services.yaml")
        asr = services["single-gpu"]["asr"]["nemotron-asr-streaming-english"]

        self.assertEqual(asr["server"], "nemo-speech-asr:50052")
        self.assertIs(asr["speaker_diarization_supported"], True)
        self.assertEqual(asr["speaker_diarization_max_speakers"], 8)


if __name__ == "__main__":
    unittest.main()
