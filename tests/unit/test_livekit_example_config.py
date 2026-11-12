# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

from pathlib import Path

import pytest

from examples.generic_assistant_livekit import config

CATALOG = config.load_catalog(Path(__file__).resolve().parents[2] / "services.yaml")


def test_cloud_recipe_uses_nvcf_endpoints():
    cfg = config.load_config("cloud", env={}, catalog=CATALOG)
    assert cfg.hosted
    assert cfg.llm_base_url == config.NVCF_LLM_URL
    assert cfg.asr_server == config.NVCF_SERVER
    assert cfg.tts_server == config.NVCF_SERVER
    assert cfg.asr_function_id == CATALOG["server"]["asr"]["nemotron-asr-streaming-english"]["nvcf"]["function_id"]
    assert cfg.tts_function_id == CATALOG["server"]["tts"]["magpie-multilingual-tts"]["nvcf"]["function_id"]
    # The Riva client needs the catalog entry model, not the nvcf override that Pipecat uses.
    assert cfg.asr_model == CATALOG["server"]["asr"]["nemotron-asr-streaming-english"]["model"]
    assert cfg.asr_model != "nemotron-asr-streaming"


def test_server_recipe_uses_compose_hostnames():
    cfg = config.load_config("server", env={}, catalog=CATALOG)
    assert not cfg.hosted
    assert cfg.llm_base_url == "http://nvidia-llm:8000/v1"
    assert cfg.asr_server == "nemotron-asr-streaming-english:50052"
    assert cfg.tts_server == "magpie-multilingual-tts-service:50051"
    assert cfg.asr_model == ""
    assert cfg.tts_sample_rate == config.DEFAULT_TTS_SAMPLE_RATE


def test_single_gpu_recipe_uses_nemo_speech():
    cfg = config.load_config("single-gpu", env={}, catalog=CATALOG)
    assert cfg.asr_server == cfg.tts_server == "nemo-speech:50051"
    assert cfg.llm_base_url == "http://nvidia-llm-vllm:8000/v1"
    assert cfg.tts_sample_rate == config.SINGLE_GPU_TTS_SAMPLE_RATE


def test_recipe_defaults_to_environment_then_cloud():
    assert config.load_config(env={"SERVICE_RECIPE": "server"}, catalog=CATALOG).recipe == "server"
    assert config.load_config(env={}, catalog=CATALOG).recipe == "cloud"


def test_llm_keeps_catalog_params_and_disables_reasoning():
    body = config.load_config("server", env={}, catalog=CATALOG).llm_extra_body
    assert body["repetition_penalty"] == 1.05
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_service_keys_can_be_overridden():
    env = {"LLM_SERVICE": "nemotron-bolt-4b"}
    cfg = config.load_config("single-gpu", env=env, catalog=CATALOG)
    assert cfg.llm_model == "nvidia/nemotron-3.5-bolt-e4b"
    assert cfg.llm_base_url == "http://nemotron-35-bolt-e4b-vllm:8000/v1"


def test_cloud_rejects_services_without_nvcf():
    with pytest.raises(ValueError, match="not hosted on NVIDIA Cloud"):
        config.load_config("cloud", env={"ASR_SERVICE": "parakeet-ctc"}, catalog=CATALOG)


def test_unknown_recipe_and_service_are_errors():
    with pytest.raises(ValueError, match="SERVICE_RECIPE"):
        config.load_config("bogus", env={}, catalog=CATALOG)
    with pytest.raises(ValueError, match="not available"):
        config.load_config("server", env={"TTS_SERVICE": "nope"}, catalog=CATALOG)


def test_models_summary_for_the_ui():
    models = config.load_config("cloud", env={}, catalog=CATALOG).models()
    assert [m["kind"] for m in models] == ["ASR", "LLM", "TTS"]
    assert all(m["runs"] == "NVIDIA Cloud" for m in models)
