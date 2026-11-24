# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

from pathlib import Path

import pytest

from nemotron_voice_agent.livekit.examples.generic_assistant import config

CUSTOM_ON = {"LIVEKIT_UI_CUSTOM_ENDPOINTS": "1"}
CATALOG = config.load_catalog(Path(__file__).resolve().parents[3] / "services.yaml")


def test_cloud_recipe_uses_nvcf_endpoints():
    cfg = config.load_config("cloud", env={}, catalog=CATALOG)
    assert cfg.hosted
    assert cfg.llm_base_url == config.NVCF_LLM_BASE_URL
    assert cfg.asr_server == config.NVCF_GRPC_SERVER
    assert cfg.tts_server == config.NVCF_GRPC_SERVER
    assert cfg.asr_function_id == CATALOG["server"]["asr"]["nemotron-asr-streaming-english"]["nvcf"]["function_id"]
    assert cfg.tts_function_id == CATALOG["server"]["tts"]["magpie-multilingual-tts"]["nvcf"]["function_id"]
    assert cfg.asr_model == ""


def test_every_cloud_asr_entry_lets_the_function_pick_its_model():
    for key in config.available_services(CATALOG, "cloud")["asr"]:
        assert config.load_config("cloud", env={"ASR_SERVICE": key}, catalog=CATALOG).asr_model == "", key


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
    assert cfg.asr_server == "nemo-speech-asr:50052"
    assert cfg.tts_server == "nemo-speech:50051"
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
    with pytest.raises(ValueError, match="Cloud serves only entries with an nvcf key"):
        config.load_config("cloud", env={"ASR_SERVICE": "parakeet-ctc"}, catalog=CATALOG)


def test_unknown_recipe_and_service_are_errors():
    with pytest.raises(ValueError, match="SERVICE_RECIPE"):
        config.load_config("bogus", env={}, catalog=CATALOG)
    with pytest.raises(ValueError, match="not available"):
        config.load_config("server", env={"TTS_SERVICE": "nope"}, catalog=CATALOG)


def test_unknown_service_error_lists_the_choices():
    with pytest.raises(ValueError, match="Choose one of: .*nemotron-lightning"):
        config.load_config("single-gpu", env={"LLM_SERVICE": "typo"}, catalog=CATALOG)


def test_available_services_follow_the_recipe():
    cloud = config.available_services(CATALOG, "cloud")
    assert "nemotron-bolt-4b" in cloud["llm"]  # has an nvcf key
    assert "magpie-zeroshot-tts" not in cloud["tts"]  # self-hosted only
    server = config.available_services(CATALOG, "server")
    assert "magpie-zeroshot-tts" in server["tts"]
    single = config.available_services(CATALOG, "single-gpu")
    assert single["asr"] == ["nemotron-asr-streaming-english", "nemotron-asr-streaming-multilingual"]
    for recipe in config.RECIPES:
        services = config.available_services(CATALOG, recipe)
        assert all(config.DEFAULT_SERVICES[slot] in services[slot] for slot in services)


def test_transcribe_asr_is_rejected_because_the_plugin_speaks_riva_grpc():
    assert "nemotron-transcribe" in CATALOG["server"]["asr"]
    assert "nemotron-transcribe" not in config.available_services(CATALOG, "server")["asr"]
    with pytest.raises(ValueError, match="not available"):
        config.load_config("server", env={"ASR_SERVICE": "nemotron-transcribe"}, catalog=CATALOG)


def test_every_listed_combination_resolves():
    for recipe in config.RECIPES:
        for slot, keys in config.available_services(CATALOG, recipe).items():
            for key in keys:
                config.load_config(recipe, env={f"{slot.upper()}_SERVICE": key}, catalog=CATALOG)


def test_models_summary_for_the_ui():
    models = config.load_config("cloud", env={}, catalog=CATALOG).models()
    assert [m["kind"] for m in models] == ["ASR", "LLM", "TTS"]
    assert all(m["runs"] == "NVIDIA Cloud" for m in models)


def test_models_name_the_runtime_for_each_recipe():
    server = config.load_config("server", env={}, catalog=CATALOG).models()
    assert [m["runs"] for m in server] == ["NVIDIA NIM"] * 3
    single = config.load_config("single-gpu", env={}, catalog=CATALOG).models()
    assert {m["kind"]: m["runs"] for m in single} == {"ASR": "NeMo-Speech.cpp", "LLM": "vLLM", "TTS": "NeMo-Speech.cpp"}


def test_deployment_summary_has_a_label_and_the_model_ids():
    deployment = config.load_config("single-gpu", env={}, catalog=CATALOG).deployment()
    assert deployment["recipe"] == "single-gpu"
    assert deployment["label"] == "Single GPU"
    llm = next(m for m in deployment["models"] if m["kind"] == "LLM")
    assert llm["id"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert config.load_config("server", env={}, catalog=CATALOG).deployment()["label"] == "Server (NIM)"
    assert config.load_config("cloud", env={}, catalog=CATALOG).deployment()["label"] == "Cloud"


def test_list_command_prints_choices_and_marks_defaults(capsys, monkeypatch):
    monkeypatch.setenv("SERVICES_PATH", str(Path(__file__).resolve().parents[3] / "services.yaml"))
    assert config.main(["single-gpu"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("single-gpu\n")
    assert "LLM_SERVICE: nemotron-lightning (default)" in out
    assert config.main(["nope"]) == 2


def ids(options):
    return {slot: [entry["id"] for entry in entries] for slot, entries in options.items()}


def test_offered_services_follow_the_registry_order_and_the_recipe():
    cloud = ids(config.service_options("cloud", CATALOG, env={}))
    assert cloud["llm"] == ["nemotron-lightning", "nemotron-super", "nemotron-bolt-4b"]  # bolt-74b has no nvcf key
    assert "magpie-zeroshot-tts" not in cloud["tts"]
    server = ids(config.service_options("server", CATALOG, env={}))
    assert server["tts"] == config.REGISTRY_SERVICES["tts"]
    single = ids(config.service_options("single-gpu", CATALOG, env={}))
    assert single["asr"] == ["nemotron-asr-streaming-english"]


def test_the_environment_default_is_offered_first():
    offered = config.service_options("server", CATALOG, env={"LLM_SERVICE": "nemotron-super"})
    assert [(e["id"], e["default"]) for e in offered["llm"][:2]] == [
        ("nemotron-super", True),
        ("nemotron-lightning", False),
    ]


def test_offered_entries_carry_the_source_and_the_editable_settings():
    llm = config.service_options("server", CATALOG, env={})["llm"][0]
    assert llm["source"] == "self-hosted" and llm["health_path"]
    assert {"temperature", "top_p", "max_tokens"} <= set(llm["settings"])
    assert config.service_options("cloud", CATALOG, env={})["tts"][0]["source"] == "cloud-nim"
    single_asr = config.service_options("single-gpu", CATALOG, env={})["asr"][0]
    assert set(single_asr["settings"]) == {"automatic_punctuation"}
    single_tts = config.service_options("single-gpu", CATALOG, env={})["tts"][0]
    assert single_tts["settings"] == {}  # synthesis_mode is a Pipecat setting


def test_a_session_choice_picks_another_offered_service():
    cfg = config.load_config("server", env={}, catalog=CATALOG, choices={"tts": {"key": "chatterbox-multilingual-tts"}})
    assert cfg.keys["tts"] == "chatterbox-multilingual-tts"
    assert cfg.tts_name == CATALOG["server"]["tts"]["chatterbox-multilingual-tts"]["name"]


def test_llm_settings_split_into_client_arguments_and_the_body():
    choices = {"llm": {"key": "nemotron-super", "settings": {"temperature": 0.2, "max_tokens": 200}}}
    cfg = config.load_config("server", env={}, catalog=CATALOG, choices=choices)
    assert cfg.llm_options["temperature"] == 0.2 and cfg.llm_options["max_tokens"] == 200
    assert not set(config.LLM_ARGUMENTS) & set(cfg.llm_extra_body)
    assert cfg.llm_extra_body["chat_template_kwargs"]["enable_thinking"] is False


def test_asr_punctuation_follows_the_session_setting():
    choices = {"asr": {"key": "nemotron-asr-streaming-english", "settings": {"automatic_punctuation": False}}}
    assert not config.load_config("single-gpu", env={}, catalog=CATALOG, choices=choices).asr_punctuate
    assert config.load_config("single-gpu", env={}, catalog=CATALOG).asr_punctuate


def test_a_custom_llm_gets_only_its_own_parameters_and_no_api_key():
    custom = {"base_url": "http://my-llm:8000/v1", "model_id": "my/model", "extra_params": '{"top_k": 5}'}
    cfg = config.load_config("cloud", env=CUSTOM_ON, catalog=CATALOG, choices={"llm": {"custom": custom}})
    assert (cfg.llm_base_url, cfg.llm_model, cfg.llm_extra_body) == ("http://my-llm:8000/v1", "my/model", {"top_k": 5})
    assert not cfg.slot_hosted("llm") and cfg.slot_hosted("asr")
    assert next(m for m in cfg.models() if m["kind"] == "LLM")["runs"] == config.CUSTOM_RUNTIME


def test_a_custom_endpoint_on_nvidia_cloud_gets_tls_and_the_key():
    nvcf_llm = {"base_url": config.NVCF_LLM_BASE_URL, "model_id": "m"}
    assert config.load_config(
        "server", env=CUSTOM_ON, catalog=CATALOG, choices={"llm": {"custom": nvcf_llm}}
    ).slot_hosted("llm")
    tts = {"server": config.NVCF_GRPC_SERVER, "voice_id": "v", "function_id": "f"}
    cfg = config.load_config("single-gpu", env=CUSTOM_ON, catalog=CATALOG, choices={"tts": {"custom": tts}})
    assert cfg.slot_hosted("tts") and cfg.tts_sample_rate == config.DEFAULT_TTS_SAMPLE_RATE
    asr = {"server": "my-asr:50051", "model": "my-model"}
    cfg = config.load_config("server", env=CUSTOM_ON, catalog=CATALOG, choices={"asr": {"custom": asr}})
    assert not cfg.slot_hosted("asr") and cfg.asr_model == "my-model"


@pytest.mark.parametrize("slot", ["asr", "tts"])
def test_a_function_id_on_another_host_is_not_nvidia_cloud(slot):
    entry = {"server": "attacker.example:443", "function_id": "f", "voice_id": "v"}
    cfg = config.load_config("cloud", env=CUSTOM_ON, catalog=CATALOG, choices={slot: {"custom": entry}})
    assert not cfg.slot_hosted(slot)


@pytest.mark.parametrize(
    ("slot", "entry", "hosted"),
    [
        ("llm", {"base_url": "https://integrate.api.nvidia.com/v1"}, True),
        ("llm", {"base_url": "http://integrate.api.nvidia.com:443/v1"}, False),
        ("llm", {"base_url": "http://integrate.api.nvidia.com/v1"}, False),
        ("llm", {"base_url": "https://integrate.api.nvidia.com.attacker.example/v1"}, False),
        ("tts", {"server": "grpc.nvcf.nvidia.com:443", "function_id": "f"}, True),
        ("tts", {"server": "grpc.nvcf.nvidia.com:443"}, False),
        ("asr", {"server": "grpc.nvcf.nvidia.com:50051", "function_id": "f"}, False),
    ],
)
def test_only_nvidia_cloud_over_tls_gets_the_api_key(slot, entry, hosted):
    assert config.custom_on_nvidia_cloud(slot, entry) is hosted


@pytest.mark.parametrize(
    ("extra_params", "error"),
    [
        ('{"extra_body": 5}', "extra_body must be a JSON object"),
        ('{"chat_template_kwargs": []}', "chat_template_kwargs must be a JSON object"),
        ('{"max_tokens": 1e999}', "with finite numbers"),
        ('{"temperature": NaN}', "with finite numbers"),
        ("[1]", "must be a JSON object"),
    ],
)
def test_unusable_extra_params_are_an_error(extra_params, error):
    with pytest.raises(ValueError, match=error):
        config.custom_entry("llm", {"base_url": "http://x:1/v1", "model_id": "m", "extra_params": extra_params})


def test_custom_endpoints_are_off_unless_turned_on():
    custom = {"llm": {"custom": {"base_url": "http://x:1/v1", "model_id": "m"}}}
    with pytest.raises(ValueError, match="custom endpoints are turned off"):
        config.load_config("cloud", env={}, catalog=CATALOG, choices=custom)
    assert config.load_config("cloud", env=CUSTOM_ON, catalog=CATALOG, choices=custom).keys["llm"] == ""


def test_a_custom_endpoint_without_its_required_fields_is_an_error():
    with pytest.raises(ValueError, match="needs model_id"):
        config.custom_entry("llm", {"base_url": "http://x:1/v1"})
    with pytest.raises(ValueError, match="JSON object"):
        config.custom_entry("llm", {"base_url": "http://x:1/v1", "model_id": "m", "extra_params": "{not json"})
