# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

from pathlib import Path

import pytest
import yaml

from voiceclaw.config import (
    AsrServiceConfig,
    BundledNvaFrontendProfile,
    CascadedServicesConfig,
    ConfigurationError,
    CredentialReference,
    LlmServiceConfig,
    OpenAIRealtimeFrontendProfile,
    TtsServiceConfig,
    load_config,
)
from voiceclaw.frontend_runtime import bind_nva_credential, materialize_frontend_runtime
from voiceclaw.model_contracts import load_model_contract_catalog

_DEFAULT_CREDENTIAL = CredentialReference(env="INFERENCE_API_KEY")
_EXAMPLE_CONFIG = Path(__file__).parents[1] / "src" / "voiceclaw" / "resources" / "voiceclaw.example.yaml"
_MODEL_CONTRACTS = Path(__file__).parents[1] / "src" / "voiceclaw" / "resources" / "model_contracts.v1.yaml"


def _bundled_profile(
    *,
    platform: str = "cloud",
    llm_credential: CredentialReference | None = _DEFAULT_CREDENTIAL,
    asr_credential: CredentialReference | None = _DEFAULT_CREDENTIAL,
    tts_credential: CredentialReference | None = _DEFAULT_CREDENTIAL,
) -> BundledNvaFrontendProfile:
    return BundledNvaFrontendProfile(
        kind="bundled_nva",
        public_model="nvidia/voiceclaw",
        realtime_model="nvidia/voiceclaw-cascade",
        platform=platform,
        pipeline_mode="generic-assistant",
        services=CascadedServicesConfig(
            llm=LlmServiceConfig(
                id="inference-hub",
                name="Inference Hub",
                provider="openai_compatible",
                endpoint="https://inference-api.nvidia.com/v1",
                model="azure/anthropic/claude-opus-5",
                credential=llm_credential,
                max_tokens=2048,
                temperature=0.2,
                extra_params={
                    "extra_body": {
                        "chat_template_kwargs": {"enable_thinking": False},
                        "repetition_penalty": 1.05,
                    }
                },
                supports_tokenize=True,
                realtime_max_output_tokens=4096,
                forced_tool_call_stops=("tool_calls",),
            ),
            asr=AsrServiceConfig(
                id="nemotron-asr",
                name="Nemotron ASR",
                provider="nvidia_grpc",
                endpoint="grpc.nvcf.nvidia.com:443",
                model="nemotron-asr-streaming",
                credential=asr_credential,
                tls=True,
                function_id="asr-function",
                language_code="en-US",
            ),
            tts=TtsServiceConfig(
                id="magpie-tts",
                name="Magpie TTS",
                provider="nvidia_grpc",
                endpoint="grpc.nvcf.nvidia.com:443",
                model="magpie-tts-multilingual",
                voice="Magpie-Multilingual.EN-US.Aria",
                credential=tts_credential,
                tls=True,
                function_id="tts-function",
                synthesis_mode="stitched",
                language_code="en-US",
            ),
        ),
    )


def _load(path: Path) -> dict:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_materializes_bundled_nva_native_catalogs_without_secret_values(tmp_path) -> None:
    credential = CredentialReference(env="INFERENCE_API_KEY")
    plan = materialize_frontend_runtime(
        _bundled_profile(
            llm_credential=credential,
            asr_credential=credential,
            tts_credential=credential,
        ),
        tmp_path / "runtime",
        internal_endpoint="ws://127.0.0.1:7861/v1/realtime",
    )

    assert plan.launch_bundled_nva is True
    assert plan.upstream_endpoint == "ws://127.0.0.1:7861/v1/realtime"
    assert plan.upstream_model == "nvidia/voiceclaw-cascade"
    assert plan.nva_credential == credential
    assert plan.nva_environment == {
        "NVA_RUNTIME_CONFIG_DIR": str(tmp_path / "runtime"),
        "REALTIME_SERVICE_PLATFORM": "cloud",
        "EXAMPLE_SELECTION": "generic-assistant",
        "PROMPT_FILE_PATH": str(tmp_path / "runtime" / "prompts.yaml"),
        "TRANSPORT_SELECTION": "websocket",
    }

    registry = _load(plan.registry_path)  # type: ignore[arg-type]
    assert registry["selection"] == "generic-assistant"
    profile = registry["realtime_models"]["nvidia/voiceclaw-cascade"]
    assert profile["selectors"] == {
        "prompt_key": "voiceclaw_frontend",
        "llm_id": "inference-hub",
        "asr_id": "nemotron-asr",
        "tts_id": "magpie-tts",
    }
    prompts = _load(plan.prompt_catalog_path)  # type: ignore[arg-type]
    assert prompts["voiceclaw_frontend"]["content"]
    assert prompts["voiceclaw_frontend"]["content"] == load_model_contract_catalog().static_instructions
    assert plan.prompt_key == "voiceclaw_frontend"
    assert plan.model_contract_digest == load_model_contract_catalog().digest

    cloud = _load(plan.services_cloud_path)  # type: ignore[arg-type]
    llm = cloud["llm"]["inference-hub"]
    assert llm["model_id"] == "azure/anthropic/claude-opus-5"
    assert llm["base_url"] == "https://inference-api.nvidia.com/v1"
    assert llm["system_prompt"] == ""
    assert llm["extra_params"] == (
        '{"extra_body":{"chat_template_kwargs":{"enable_thinking":false},"repetition_penalty":1.05}}'
    )
    assert llm["forced_tool_call_stops"] == ["tool_calls"]
    assert cloud["asr"]["nemotron-asr"]["language_code"] == "en-US"
    assert cloud["asr"]["nemotron-asr"]["use_ssl"] is True
    assert cloud["tts"]["magpie-tts"]["voice_id"] == "Magpie-Multilingual.EN-US.Aria"
    assert cloud["tts"]["magpie-tts"]["use_ssl"] is True

    local = _load(plan.services_local_path)  # type: ignore[arg-type]
    assert local["server"] == cloud
    assert local["singlegpu"] == cloud
    generated = "\n".join(path.read_text(encoding="utf-8") for path in (tmp_path / "runtime").iterdir())
    assert "INFERENCE_API_KEY" not in generated
    assert "NVIDIA_API_KEY" not in generated


def test_selected_local_platform_is_expressed_only_in_child_environment(tmp_path) -> None:
    plan = materialize_frontend_runtime(
        _bundled_profile(platform="singlegpu"),
        tmp_path,
        internal_endpoint="ws://127.0.0.1:9000/v1/realtime",
    )

    assert plan.nva_environment["REALTIME_SERVICE_PLATFORM"] == "singlegpu"
    assert _load(plan.services_local_path)["singlegpu"] == _load(plan.services_cloud_path)  # type: ignore[arg-type]


def test_bundled_prompt_is_generated_from_selected_model_contract(tmp_path) -> None:
    override = tmp_path / "model-contracts.yaml"
    custom_policy = "You are the reviewed custom VoiceClaw frontend policy."
    override.write_text(
        _MODEL_CONTRACTS.read_text(encoding="utf-8").replace(
            "You are a low-latency conversational voice interface.",
            custom_policy,
            1,
        ),
        encoding="utf-8",
    )
    contracts = load_model_contract_catalog(override)

    plan = materialize_frontend_runtime(
        _bundled_profile(),
        tmp_path / "runtime",
        internal_endpoint="ws://127.0.0.1:7861/v1/realtime",
        model_contracts=contracts,
    )

    prompt_catalog = _load(plan.prompt_catalog_path)  # type: ignore[arg-type]
    assert custom_policy in prompt_catalog["voiceclaw_frontend"]["content"]
    assert plan.model_contract_digest == contracts.digest


def test_example_local_cascade_materializes_deterministic_temperature(tmp_path) -> None:
    configuration = load_config(
        _EXAMPLE_CONFIG,
        environ={"NEMOCLAW_VOICE_GATEWAY_BEARER_FILE": "/run/secrets/nemoclaw_voice_gateway_bearer"},
    )
    profile = configuration.selected_frontend
    assert isinstance(profile, BundledNvaFrontendProfile)

    plan = materialize_frontend_runtime(
        profile,
        tmp_path,
        internal_endpoint="ws://127.0.0.1:7861/v1/realtime",
    )

    cloud = _load(plan.services_cloud_path)  # type: ignore[arg-type]
    assert cloud["llm"]["local-nemotron-lightning"]["temperature"] == 0.0
    assert cloud["asr"]["local-nemotron-asr"]["use_ssl"] is False
    assert cloud["tts"]["local-magpie-tts"]["use_ssl"] is False


def test_materializes_recursively_frozen_generic_provider_options(tmp_path) -> None:
    profile = _bundled_profile()
    object.__setattr__(
        profile.services.llm,
        "extra_params",
        {
            "reasoning_effort": "high",
            "extra_body": {"vendor": {"seed": 7, "labels": ["voice", "agent"]}},
        },
    )
    profile.services.llm.__post_init__()

    plan = materialize_frontend_runtime(
        profile,
        tmp_path,
        internal_endpoint="ws://127.0.0.1:7861/v1/realtime",
    )

    catalog = _load(plan.services_cloud_path)  # type: ignore[arg-type]
    encoded = catalog["llm"]["inference-hub"]["extra_params"]
    assert yaml.safe_load(encoded) == {
        "reasoning_effort": "high",
        "extra_body": {"vendor": {"seed": 7, "labels": ["voice", "agent"]}},
    }


def test_external_realtime_profile_does_not_create_or_launch_nva(tmp_path) -> None:
    destination = tmp_path / "must-not-exist"
    plan = materialize_frontend_runtime(
        OpenAIRealtimeFrontendProfile(
            kind="openai_realtime",
            endpoint="wss://realtime.example.test/v1/realtime",
            model="provider/speech-to-speech",
            public_model="nvidia/voiceclaw",
            credential=CredentialReference(env="EXTERNAL_REALTIME_KEY"),
        ),
        destination,
        internal_endpoint="ws://127.0.0.1:7861/v1/realtime",
    )

    assert plan.launch_bundled_nva is False
    assert plan.upstream_endpoint == "wss://realtime.example.test/v1/realtime"
    assert plan.upstream_model == "provider/speech-to-speech"
    assert plan.nva_environment == {}
    assert plan.registry_path is None
    assert destination.exists() is False


def test_rejects_different_service_credential_sources(tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="same credential source"):
        materialize_frontend_runtime(
            _bundled_profile(
                llm_credential=CredentialReference(env="LLM_KEY"),
                asr_credential=CredentialReference(env="SPEECH_KEY"),
            ),
            tmp_path,
            internal_endpoint="ws://127.0.0.1:7861/v1/realtime",
        )


@pytest.mark.parametrize(
    ("llm_credential", "asr_credential", "tts_credential"),
    [
        (_DEFAULT_CREDENTIAL, None, None),
        (None, _DEFAULT_CREDENTIAL, None),
        (None, None, _DEFAULT_CREDENTIAL),
        (_DEFAULT_CREDENTIAL, _DEFAULT_CREDENTIAL, None),
    ],
)
def test_rejects_partial_shared_service_credentials(
    tmp_path,
    llm_credential: CredentialReference | None,
    asr_credential: CredentialReference | None,
    tts_credential: CredentialReference | None,
) -> None:
    with pytest.raises(ConfigurationError, match="same credential source on llm, asr, and tts"):
        materialize_frontend_runtime(
            _bundled_profile(
                platform="singlegpu",
                llm_credential=llm_credential,
                asr_credential=asr_credential,
                tts_credential=tts_credential,
            ),
            tmp_path,
            internal_endpoint="ws://127.0.0.1:7861/v1/realtime",
        )


def test_cloud_platform_requires_a_shared_service_credential(tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="platform cloud requires"):
        materialize_frontend_runtime(
            _bundled_profile(llm_credential=None, asr_credential=None, tts_credential=None),
            tmp_path,
            internal_endpoint="ws://127.0.0.1:7861/v1/realtime",
        )


def test_rejects_unsupported_pipeline_and_provider(tmp_path) -> None:
    unsupported_pipeline = _bundled_profile()
    object.__setattr__(unsupported_pipeline, "pipeline_mode", "omni-assistant")
    with pytest.raises(ConfigurationError, match="currently supports pipeline_mode"):
        materialize_frontend_runtime(
            unsupported_pipeline,
            tmp_path / "pipeline",
            internal_endpoint="ws://127.0.0.1:7861/v1/realtime",
        )

    unsupported_provider = _bundled_profile()
    object.__setattr__(unsupported_provider.services.llm, "provider", "anthropic_messages")
    with pytest.raises(ConfigurationError, match="llm.provider must be openai_compatible"):
        materialize_frontend_runtime(
            unsupported_provider,
            tmp_path / "provider",
            internal_endpoint="ws://127.0.0.1:7861/v1/realtime",
        )


def test_binds_environment_credential_without_inheriting_an_ambient_key() -> None:
    prepared = bind_nva_credential(
        {"PATH": "/usr/bin", "NVIDIA_API_KEY": "ambient-wrong-key"},
        CredentialReference(env="SELECTED_MODEL_KEY"),
        source_environment={"SELECTED_MODEL_KEY": "selected-secret"},
    )

    assert prepared == {"PATH": "/usr/bin", "NVIDIA_API_KEY": "selected-secret"}
    assert (
        bind_nva_credential(
            {"NVIDIA_API_KEY": "ambient-wrong-key"},
            None,
            source_environment={},
        )
        == {}
    )


def test_resolves_file_credential_only_into_the_private_nva_environment(tmp_path) -> None:
    secret = tmp_path / "model-key"
    secret.write_text(" file-secret\n", encoding="utf-8")
    secret.chmod(0o600)
    with pytest.raises(ConfigurationError, match="without whitespace"):
        bind_nva_credential(
            {},
            CredentialReference(file=str(secret)),
            source_environment={},
        )

    secret.write_text("file-secret\n", encoding="utf-8")
    prepared = bind_nva_credential(
        {},
        CredentialReference(file=str(secret)),
        source_environment={},
    )

    assert prepared == {"NVIDIA_API_KEY": "file-secret"}
    assert "NVIDIA_API_KEY_FILE" not in prepared


def test_rejects_credential_file_accessible_by_others(tmp_path) -> None:
    secret = tmp_path / "model-key"
    secret.write_text("file-secret\n", encoding="utf-8")
    secret.chmod(0o604)

    with pytest.raises(ConfigurationError, match="accessible by others"):
        bind_nva_credential(
            {},
            CredentialReference(file=str(secret)),
            source_environment={},
        )


def test_rejects_missing_environment_credential() -> None:
    with pytest.raises(ConfigurationError, match="is not set"):
        bind_nva_credential(
            {},
            CredentialReference(env="MISSING_MODEL_KEY"),
            source_environment={},
        )
