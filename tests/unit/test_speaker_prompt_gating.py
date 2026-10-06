# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

from unittest.mock import patch

from fastapi.testclient import TestClient

import examples_registry
from examples.shared.speaker import resolve_asr_speaker_support, speaker_diarization_enabled
from server import _sanitize_session_config, create_app
from utils import load_service_entry


def test_multi_speaker_prompt_falls_back_for_unsupported_asr():
    config = _sanitize_session_config(
        {
            "pipeline_mode": "generic-assistant",
            "asr_id": "cloud-nim:parakeet-rnnt",
            "prompt_key": "multi_speaker_assistant",
        }
    )

    assert config["prompt_key"] == "generic_assistant"


def test_multi_speaker_prompt_remains_for_supported_asr():
    with patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-test"}):
        config = _sanitize_session_config(
            {
                "pipeline_mode": "generic-assistant",
                "asr_id": "cloud-nim:nemotron-asr-streaming-english",
                "prompt_key": "multi_speaker_assistant",
                "asr_speaker_diarization": "true",
            }
        )

    assert config["prompt_key"] == "multi_speaker_assistant"


def test_session_without_asr_id_uses_the_default_asr_for_diarization():
    session = {"pipeline_mode": "generic-assistant", "prompt_key": "multi_speaker_assistant"}
    for recipe in ("server", "cloud"):
        env = {"NVIDIA_API_KEY": "nvapi-test", "SERVICE_RECIPE": recipe, "ENABLE_SPEAKER_DIARIZATION": "true"}
        with patch.dict("os.environ", env):
            config = _sanitize_session_config(session)
            default_asr = load_service_entry("asr", "")

            assert config["prompt_key"] == "multi_speaker_assistant"
            assert "asr_id" not in config
            assert speaker_diarization_enabled(config, default_asr=default_asr)
            assert resolve_asr_speaker_support(config, default_asr) == (True, 8)


def test_custom_asr_does_not_borrow_the_default_asr_support():
    default_asr = {"speaker_diarization_supported": True, "speaker_diarization_max_speakers": 8}
    body = {"asr_id": "custom-asr", "asr_server": "other-asr:50052", "asr_speaker_diarization": "true"}

    assert resolve_asr_speaker_support(body, default_asr) == (False, None)
    assert not speaker_diarization_enabled(body, default_asr=default_asr)


def test_omitted_diarization_field_follows_env_default():
    session = {
        "pipeline_mode": "generic-assistant",
        "asr_id": "cloud-nim:nemotron-asr-streaming-english",
        "prompt_key": "multi_speaker_assistant",
    }
    with patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-test", "ENABLE_SPEAKER_DIARIZATION": "false"}):
        assert _sanitize_session_config(session)["prompt_key"] == "generic_assistant"
    with patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-test", "ENABLE_SPEAKER_DIARIZATION": "true"}):
        assert _sanitize_session_config(session)["prompt_key"] == "multi_speaker_assistant"


def test_prompt_catalog_hides_multi_speaker_for_unsupported_asr():
    with TestClient(create_app()) as client:
        response = client.get(
            "/api/prompts",
            params={
                "pipeline_mode": "generic-assistant",
                "asr_id": "cloud-nim:parakeet-rnnt",
            },
        )

    assert response.status_code == 200
    assert "multi_speaker_assistant" not in {prompt["key"] for prompt in response.json()}


def test_multi_speaker_prompt_falls_back_when_diarization_is_off():
    with patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-test"}):
        config = _sanitize_session_config(
            {
                "pipeline_mode": "generic-assistant",
                "asr_id": "cloud-nim:nemotron-asr-streaming-english",
                "prompt_key": "multi_speaker_assistant",
                "asr_speaker_diarization": "false",
            }
        )

    assert config["prompt_key"] == "generic_assistant"


def test_prompt_catalog_exposes_multi_speaker_for_supported_asr():
    with patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-test"}):
        response = TestClient(create_app()).get(
            "/api/prompts",
            params={
                "pipeline_mode": "generic-assistant",
                "asr_id": "cloud-nim:nemotron-asr-streaming-english",
            },
        )

    assert response.status_code == 200
    assert "multi_speaker_assistant" in {prompt["key"] for prompt in response.json()}


def test_registry_keeps_speaker_labels_on_generic_example():
    generic = examples_registry.find("generic-assistant", ignore_lock=True)

    assert "speaker_labels" in generic["capabilities"]
    assert generic["bot"] == "examples.generic.pipeline:bot"
    assert examples_registry.resolve_bot(generic).__module__ == "examples.generic.pipeline"
