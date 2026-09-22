# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

import os
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest
import yaml

from voiceclaw.config import (
    BackendProfile,
    ConfigurationError,
    CredentialReference,
    FrontendKind,
    InteractionPolicy,
    OpenAIRealtimeFrontendProfile,
    RealtimeConfig,
    ServerConfig,
    StateConfig,
    VoiceClawConfig,
    parse_config,
)
from voiceclaw.container import (
    _STATE_ISOLATION_CHECK,
    _assert_files_readable_by_identity,
    _assert_not_mutable_by_identity,
    _assert_not_readable_by_identity,
    _assert_owner_only_secret,
    _assert_provider_files_isolated,
    _assert_state_path_usable,
    _facade_command,
    _facade_environment,
    _facade_secret_files_for_nva,
    _frontend_runtime_plan,
    _healthcheck_host,
    _listener_tls_environment,
    _nva_environment,
    _preflight_facade_configuration,
    _preflight_nva_profile,
    _prepare_environment,
    _prepare_managed_volume_layout,
    _read_healthcheck_target,
    _require_container_runtime,
    _run_healthcheck,
    _snapshot_configuration,
    _stage_nva_file_credential,
    _write_healthcheck_target,
)
from voiceclaw.container import _parser as _container_parser
from voiceclaw.frontend_runtime import bind_nva_credential


def _bundled_frontend(*, credential_env: str | None = None, credential_file: str | None = None) -> dict:
    credential: dict[str, str] | None = None
    if credential_env is not None:
        credential = {"env": credential_env}
    elif credential_file is not None:
        credential = {"file": credential_file}
    services: dict[str, dict[str, object]] = {
        "llm": {
            "id": "local-llm",
            "name": "Local LLM",
            "provider": "openai_compatible",
            "endpoint": "http://127.0.0.1:18000/v1",
            "model": "local-llm",
        },
        "asr": {
            "id": "local-asr",
            "name": "Local ASR",
            "provider": "nvidia_grpc",
            "endpoint": "127.0.0.1:50051",
            "model": "local-asr",
        },
        "tts": {
            "id": "local-tts",
            "name": "Local TTS",
            "provider": "nvidia_grpc",
            "endpoint": "127.0.0.1:50051",
            "model": "local-tts",
            "voice": "Local.Voice",
            "synthesis_mode": "stitched",
        },
    }
    if credential is not None:
        for service in services.values():
            service["credential"] = credential
    return {
        "kind": "bundled_nva",
        "public_model": "nvidia/voiceclaw",
        "realtime_model": "nvidia/voiceclaw-local",
        "platform": "singlegpu",
        "services": services,
    }


def test_container_ui_flag_is_opt_in_and_forwarded() -> None:
    config_path = Path("/app/voiceclaw.yaml")

    assert _container_parser().parse_args([]).ui is False
    assert _container_parser().parse_args(["--ui"]).ui is True
    assert _container_parser().parse_args(["--healthcheck"]).healthcheck is True
    assert _container_parser().parse_args(["--config", "/run/config.yaml"]).config == Path("/run/config.yaml")
    assert _facade_command(config_path, "127.0.0.1", ui=False)[-1] == "127.0.0.1"
    assert _facade_command(config_path, "127.0.0.1", ui=True)[-1] == "--ui"


def test_container_runtime_marker_has_clear_package_boundary(tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="image-internal"):
        _require_container_runtime(tmp_path / "missing")

    marker = tmp_path / "marker"
    marker.touch()
    _require_container_runtime(marker)


@pytest.mark.parametrize(
    ("listener", "expected"),
    [("0.0.0.0", "127.0.0.1"), ("::", "::1"), ("[::1]", "::1"), ("10.0.0.8", "10.0.0.8")],
)
def test_healthcheck_host_resolves_wildcard_listeners(listener: str, expected: str) -> None:
    assert _healthcheck_host(listener) == expected


def test_healthcheck_target_tracks_effective_scheme_host_and_port(tmp_path) -> None:
    target = tmp_path / "runtime" / "healthcheck.json"

    _write_healthcheck_target(
        target,
        listener_host="0.0.0.0",
        listener_port=18443,
        tls_enabled=True,
    )

    assert _read_healthcheck_target(target) == ("https", "127.0.0.1", 18443)
    assert target.stat().st_mode & 0o777 == 0o644


def test_container_healthcheck_probes_process_liveness(tmp_path, monkeypatch) -> None:
    target = tmp_path / "healthcheck.json"
    _write_healthcheck_target(
        target,
        listener_host="127.0.0.1",
        listener_port=18790,
        tls_enabled=False,
    )
    requests: list[tuple[str, str, dict[str, str]]] = []

    class Response:
        status = 200

        @staticmethod
        def read() -> bytes:
            return b""

    class Connection:
        def __init__(self, host: str, port: int, *, timeout: float) -> None:
            assert (host, port, timeout) == ("127.0.0.1", 18790, 2.5)

        @staticmethod
        def getresponse() -> Response:
            return Response()

        @staticmethod
        def close() -> None:
            return None

        @staticmethod
        def request(method: str, path: str, *, headers: dict[str, str]) -> None:
            requests.append((method, path, headers))

    monkeypatch.setattr("voiceclaw.container.http.client.HTTPConnection", Connection)

    _run_healthcheck(target)

    assert requests == [("GET", "/livez", {"Connection": "close"})]


def test_container_uses_one_root_owned_configuration_snapshot(tmp_path) -> None:
    source = tmp_path / "source.yaml"
    destination = tmp_path / "runtime" / "snapshot.yaml"
    source.write_text("schema_version: voiceclaw.config.v3\n", encoding="utf-8")

    result = _snapshot_configuration(
        source,
        destination,
        owner_uid=os.geteuid(),
        facade_gid=os.getegid(),
    )
    source.write_text("changed: true\n", encoding="utf-8")

    assert result == destination
    assert destination.read_text(encoding="utf-8") == "schema_version: voiceclaw.config.v3\n"
    assert destination.stat().st_mode & 0o777 == 0o640


def test_container_configuration_snapshot_rejects_symlink_source(tmp_path) -> None:
    source = tmp_path / "source.yaml"
    source.write_text("schema_version: voiceclaw.config.v3\n", encoding="utf-8")
    symlink = tmp_path / "link.yaml"
    symlink.symlink_to(source)

    with pytest.raises(ConfigurationError, match="could not be snapshotted"):
        _snapshot_configuration(
            symlink,
            tmp_path / "snapshot.yaml",
            owner_uid=os.geteuid(),
            facade_gid=os.getegid(),
        )


def test_listener_tls_uses_explicit_paths_or_complete_operator_pair(tmp_path) -> None:
    operator = tmp_path / "operator"
    tls = operator / "tls"
    tls.mkdir(parents=True)
    certificate = tls / "cert.pem"
    private_key = tls / "key.pem"
    certificate.touch()

    assert _listener_tls_environment({"VOICECLAW_OPERATOR_FILES_DIR": str(operator)}) == (
        str(certificate),
        str(private_key),
    )
    assert _listener_tls_environment({"VOICECLAW_TLS_CERTFILE": "/cert", "VOICECLAW_TLS_KEYFILE": "/key"}) == (
        "/cert",
        "/key",
    )


def test_container_generates_one_private_loopback_realtime_key() -> None:
    prepared = _prepare_environment({}, key_factory=lambda: "generated-internal-key")

    assert prepared["REALTIME_API_KEY"] == "generated-internal-key"
    assert prepared["REALTIME_UPSTREAM_API_KEY"] == "generated-internal-key"
    assert prepared["PIPELINE_TLS"] == "false"
    assert prepared["UVICORN_WORKERS"] == "1"


def test_container_rejects_mismatched_internal_keys() -> None:
    with pytest.raises(ConfigurationError, match="must match"):
        _prepare_environment({"REALTIME_API_KEY": "nva-key", "REALTIME_UPSTREAM_API_KEY": "facade-key"})


def test_container_materializes_selected_bundled_frontend(tmp_path) -> None:
    config = parse_config(
        {
            "schema_version": "voiceclaw.config.v3",
            "frontend_profiles": {
                "local": {
                    "kind": "bundled_nva",
                    "public_model": "nvidia/voiceclaw",
                    "realtime_model": "nvidia/voiceclaw-local",
                    "platform": "singlegpu",
                    "services": {
                        "llm": {
                            "id": "local-llm",
                            "name": "Local LLM",
                            "provider": "openai_compatible",
                            "endpoint": "http://127.0.0.1:18000/v1",
                            "model": "nvidia/local-model",
                        },
                        "asr": {
                            "id": "local-asr",
                            "name": "Local ASR",
                            "provider": "nvidia_grpc",
                            "endpoint": "127.0.0.1:50051",
                            "model": "local-asr",
                        },
                        "tts": {
                            "id": "local-tts",
                            "name": "Local TTS",
                            "provider": "nvidia_grpc",
                            "endpoint": "127.0.0.1:50051",
                            "model": "local-tts",
                            "voice": "Local.Voice",
                            "synthesis_mode": "stitched",
                        },
                    },
                }
            },
            "default_frontend": "local",
            "backend_profiles": {"none": {"kind": "none"}},
            "default_backend": "none",
        }
    )

    plan = _frontend_runtime_plan(
        config,
        {
            "VOICECLAW_FRONTEND_RUNTIME_DIR": str(tmp_path),
            "PROMPT_FILE_PATH": "/ambient/untrusted-prompts.yaml",
            "PROMPT_SELECTOR": "ambient_prompt",
        },
        internal_endpoint="ws://127.0.0.1:7861/v1/realtime",
    )

    assert plan is not None and plan.launch_bundled_nva is True
    assert plan.nva_environment["NVA_RUNTIME_CONFIG_DIR"] == str(tmp_path)
    assert (tmp_path / "examples_registry.yaml").is_file()
    assert (tmp_path / "services.cloud.yaml").is_file()
    assert (tmp_path / "services.local.yaml").is_file()
    assert (tmp_path / "prompts.yaml").is_file()
    assert plan.nva_environment["PROMPT_FILE_PATH"] == str(tmp_path / "prompts.yaml")
    child_environment = _nva_environment(
        {
            "PROMPT_FILE_PATH": "/ambient/untrusted-prompts.yaml",
            "PROMPT_SELECTOR": "ambient_prompt",
        },
        config,
    )
    child_environment.update(plan.nva_environment)
    assert child_environment["PROMPT_FILE_PATH"] == str(tmp_path / "prompts.yaml")
    generated_prompts = yaml.safe_load((tmp_path / "prompts.yaml").read_text(encoding="utf-8"))
    assert "conversational voice interface" in generated_prompts["voiceclaw_frontend"]["content"]


def test_container_external_realtime_frontend_skips_nva_materialization(tmp_path) -> None:
    config = parse_config(
        {
            "schema_version": "voiceclaw.config.v3",
            "frontend_profiles": {
                "hosted": {
                    "kind": "openai_realtime",
                    "endpoint": "wss://realtime.example.test/v1/realtime",
                    "model": "provider/speech-to-speech",
                }
            },
            "default_frontend": "hosted",
            "backend_profiles": {"none": {"kind": "none"}},
            "default_backend": "none",
        }
    )
    destination = tmp_path / "unused"

    plan = _frontend_runtime_plan(
        config,
        {"VOICECLAW_FRONTEND_RUNTIME_DIR": str(destination)},
        internal_endpoint="ws://127.0.0.1:7861/v1/realtime",
    )

    assert plan is not None and plan.launch_bundled_nva is False
    assert plan.upstream_endpoint == "wss://realtime.example.test/v1/realtime"
    assert destination.exists() is False


def test_unselected_frontend_credential_env_does_not_cross_into_nva() -> None:
    base = parse_config(
        {
            "schema_version": "voiceclaw.config.v3",
            "frontend_profiles": {"local": _bundled_frontend()},
            "default_frontend": "local",
            "backend_profiles": {"none": {"kind": "none"}},
            "default_backend": "none",
        }
    )
    config = replace(
        base,
        frontend_profiles={
            **base.frontend_profiles,
            "unused_external": OpenAIRealtimeFrontendProfile(
                kind=FrontendKind.OPENAI_REALTIME,
                endpoint="wss://127.0.0.1:9443/v1/realtime",
                model="provider/realtime",
                public_model="provider/realtime",
                credential=CredentialReference(env="HTTP_PROXY"),
            ),
        },
    )

    prepared = _nva_environment({"HTTP_PROXY": "sentinel-unselected-secret"}, config)

    assert "HTTP_PROXY" not in prepared


def test_facade_receives_only_selected_external_frontend_credential_env(tmp_path) -> None:
    raw = {
        "schema_version": "voiceclaw.config.v3",
        "frontend_profiles": {
            "selected": {
                "kind": "openai_realtime",
                "endpoint": "wss://127.0.0.1:9443/v1/realtime",
                "model": "provider/realtime",
                "credential": {"env": "SELECTED_REALTIME_KEY"},
            },
            "unused_external": {
                "kind": "openai_realtime",
                "endpoint": "wss://127.0.0.1:9555/v1/realtime",
                "model": "provider/unused",
                "credential": {"env": "UNUSED_REALTIME_KEY"},
            },
        },
        "default_frontend": "selected",
        "backend_profiles": {"none": {"kind": "none"}},
        "default_backend": "none",
    }
    base = parse_config(raw)
    unused = base.frontend_profiles["unused_external"]
    assert isinstance(unused, OpenAIRealtimeFrontendProfile)
    config = replace(
        base,
        frontend_profiles={
            **base.frontend_profiles,
            "unused_external": replace(unused, credential=CredentialReference(env="SSL_CERT_FILE")),
        },
    )
    config_path = tmp_path / "voiceclaw.yaml"
    config_path.write_text("schema_version: voiceclaw.config.v3\n", encoding="utf-8")

    prepared = _facade_environment(
        {
            "SELECTED_REALTIME_KEY": "selected-secret",
            "SSL_CERT_FILE": "sentinel-unselected-secret",
        },
        config,
        config_path,
    )

    assert prepared["SELECTED_REALTIME_KEY"] == "selected-secret"
    assert "SSL_CERT_FILE" not in prepared


@pytest.mark.parametrize("credential_name", ["NVIDIA_API_KEY", "REALTIME_API_KEY", "REALTIME_UPSTREAM_API_KEY"])
def test_facade_accepts_common_provider_key_names_for_selected_external_frontend(
    tmp_path,
    credential_name: str,
) -> None:
    raw = {
        "schema_version": "voiceclaw.config.v3",
        "frontend_profiles": {
            "hosted": {
                "kind": "openai_realtime",
                "endpoint": "wss://realtime.example.test/v1/realtime",
                "model": "provider/speech-to-speech",
                "credential": {"env": credential_name},
            }
        },
        "default_frontend": "hosted",
        "backend_profiles": {"none": {"kind": "none"}},
        "default_backend": "none",
    }
    config = parse_config(raw)
    config_path = tmp_path / "voiceclaw.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    prepared = _facade_environment(
        {credential_name: "selected-external-secret"},
        config,
        config_path,
    )

    assert prepared[credential_name] == "selected-external-secret"


def test_unselected_frontend_credential_name_cannot_remove_active_public_credential(tmp_path) -> None:
    raw = {
        "schema_version": "voiceclaw.config.v3",
        "server": {"auth_mode": "ephemeral", "api_key_env": "ACTIVE_PUBLIC_KEY"},
        "frontend_profiles": {
            "selected": {
                "kind": "openai_realtime",
                "endpoint": "wss://realtime.example.test/v1/realtime",
                "model": "provider/realtime",
            },
            "unused": {
                "kind": "openai_realtime",
                "endpoint": "wss://unused.example.test/v1/realtime",
                "model": "provider/unused",
                "credential": {"env": "ACTIVE_PUBLIC_KEY"},
            },
        },
        "default_frontend": "selected",
        "backend_profiles": {"none": {"kind": "none"}},
        "default_backend": "none",
    }
    config = parse_config(raw)
    config_path = tmp_path / "voiceclaw.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    prepared = _facade_environment(
        {"ACTIVE_PUBLIC_KEY": "active-public-secret"},
        config,
        config_path,
    )

    assert prepared["ACTIVE_PUBLIC_KEY"] == "active-public-secret"


def test_selected_external_credential_cannot_reuse_operational_env() -> None:
    raw = {
        "schema_version": "voiceclaw.config.v3",
        "frontend_profiles": {
            "selected": {
                "kind": "openai_realtime",
                "endpoint": "wss://127.0.0.1:9443/v1/realtime",
                "model": "provider/realtime",
                "credential": {"env": "HTTP_PROXY"},
            }
        },
        "default_frontend": "selected",
        "backend_profiles": {"none": {"kind": "none"}},
        "default_backend": "none",
    }
    with pytest.raises(ConfigurationError, match="reserved operational environment variable"):
        parse_config(raw)


def test_nva_child_does_not_receive_facade_or_backend_credentials() -> None:
    config = VoiceClawConfig(
        schema_version="voiceclaw.config.v2",
        server=ServerConfig(auth_mode="ephemeral", api_key_env="PUBLIC_REALTIME_KEY"),
        backend_profiles={"nemoclaw": BackendProfile(kind="nemoclaw_committed_turn", credential_env="BACKEND_BEARER")},
        default_backend="nemoclaw",
        realtime=RealtimeConfig(
            upstream_endpoint="ws://127.0.0.1:7861/v1/realtime",
            upstream_model="nvidia/nemotron-realtime-client-tools",
            credential_env="REALTIME_UPSTREAM_API_KEY",
        ),
        interaction=InteractionPolicy(),
        state=StateConfig(),
    )
    prepared = _nva_environment(
        {
            "PUBLIC_REALTIME_KEY": "public-secret",
            "BACKEND_BEARER": "backend-secret",
            "NEMOCLAW_VOICE_GATEWAY_BEARER_FILE": "/run/secrets/backend",
            "REALTIME_UPSTREAM_API_KEY": "facade-to-nva-secret",
            "REALTIME_API_KEY": "facade-to-nva-secret",
            "NVIDIA_API_KEY": "model-secret",
            "HF_TOKEN": "unrelated-model-registry-secret",
            "TURN_PASSWORD": "unrelated-turn-secret",
            "CHAT_HISTORY_RECENT_TURNS": "12",
        },
        config,
    )

    assert "PUBLIC_REALTIME_KEY" not in prepared
    assert "BACKEND_BEARER" not in prepared
    assert "NEMOCLAW_VOICE_GATEWAY_BEARER_FILE" not in prepared
    assert "REALTIME_UPSTREAM_API_KEY" not in prepared
    assert prepared["REALTIME_API_KEY"] == "facade-to-nva-secret"
    assert prepared["NVIDIA_API_KEY"] == "model-secret"
    assert prepared["CHAT_HISTORY_RECENT_TURNS"] == "12"
    assert "HF_TOKEN" not in prepared
    assert "TURN_PASSWORD" not in prepared
    assert prepared["HOME"] == "/home/voiceclaw-nva"
    assert prepared["XDG_CACHE_HOME"] == "/var/cache/voiceclaw-nva"


def test_nva_child_does_not_receive_stale_public_key_when_auth_is_disabled() -> None:
    config = VoiceClawConfig(
        schema_version="voiceclaw.config.v2",
        server=ServerConfig(auth_mode="none"),
        backend_profiles={"none": BackendProfile(kind="none")},
        default_backend="none",
        realtime=RealtimeConfig(
            upstream_endpoint="ws://127.0.0.1:7861/v1/realtime",
            upstream_model="nvidia/nemotron-realtime-client-tools",
            credential_env="REALTIME_UPSTREAM_API_KEY",
        ),
        interaction=InteractionPolicy(),
        state=StateConfig(),
    )

    prepared = _nva_environment(
        {
            "VOICECLAW_REALTIME_API_KEY": "stale-public-secret",
            "REALTIME_UPSTREAM_API_KEY": "facade-to-nva-secret",
            "REALTIME_API_KEY": "facade-to-nva-secret",
        },
        config,
    )

    assert "VOICECLAW_REALTIME_API_KEY" not in prepared
    assert "REALTIME_UPSTREAM_API_KEY" not in prepared
    assert prepared["REALTIME_API_KEY"] == "facade-to-nva-secret"


def test_nva_child_drops_ambient_mcp_product_configuration() -> None:
    config = parse_config(
        {
            "schema_version": "voiceclaw.config.v3",
            "frontend_profiles": {"local": _bundled_frontend()},
            "default_frontend": "local",
            "backend_profiles": {"none": {"kind": "none"}},
            "default_backend": "none",
        }
    )
    ambient_mcp = {
        "REALTIME_MCP_ALLOWED_SERVER_URLS": '["https://mcp.example.com/mcp"]',
        "REALTIME_MCP_APPROVAL_TIMEOUT_SECONDS": "30",
        "REALTIME_MCP_CALL_ANNOUNCE_TIMEOUT_SECONDS": "30",
        "REALTIME_MCP_CALL_TIMEOUT_SECONDS": "120",
        "REALTIME_MCP_CONNECT_TIMEOUT_SECONDS": "10",
        "REALTIME_MCP_DISCOVERY_TIMEOUT_SECONDS": "30",
        "REALTIME_MCP_MAX_BINDINGS": "512",
        "REALTIME_MCP_MAX_OUTPUT_BYTES": "262144",
        "REALTIME_MCP_MAX_SERVERS": "8",
        "REALTIME_MCP_MAX_TOOLS_PER_SERVER": "128",
        "REALTIME_MCP_READ_TIMEOUT_SECONDS": "120",
    }

    prepared = _nva_environment(ambient_mcp, config)

    assert ambient_mcp.keys().isdisjoint(prepared)


def test_facade_child_receives_only_explicit_configuration_environment(tmp_path) -> None:
    config = VoiceClawConfig(
        schema_version="voiceclaw.config.v2",
        server=ServerConfig(auth_mode="ephemeral", api_key_env="PUBLIC_REALTIME_KEY"),
        backend_profiles={"nemoclaw": BackendProfile(kind="nemoclaw_committed_turn", credential_env="BACKEND_BEARER")},
        default_backend="nemoclaw",
        realtime=RealtimeConfig(
            upstream_endpoint="ws://127.0.0.1:7861/v1/realtime",
            upstream_model="nvidia/nemotron-realtime-client-tools",
            credential_env="REALTIME_UPSTREAM_API_KEY",
        ),
        interaction=InteractionPolicy(),
        state=StateConfig(),
    )
    config_path = tmp_path / "voiceclaw.yaml"
    config_path.write_text(
        "# ${COMMENT_ONLY_SECRET}\nstate:\n  path: ${CONFIG_REFERENCED_VALUE}\n",
        encoding="utf-8",
    )

    prepared = _facade_environment(
        {
            "PATH": "/usr/bin",
            "PUBLIC_REALTIME_KEY": "public-secret",
            "BACKEND_BEARER": "backend-secret",
            "NEMOCLAW_VOICE_GATEWAY_ORIGIN": "http://127.0.0.1:19000",
            "REALTIME_UPSTREAM_API_KEY": "facade-to-nva-secret",
            "CONFIG_REFERENCED_VALUE": "operator-selected-value",
            "COMMENT_ONLY_SECRET": "must-not-cross-from-a-comment",
            "VOICECLAW_TLS_CERTFILE": "/run/tls/cert.pem",
            "VOICECLAW_TLS_KEYFILE": "/run/tls/key.pem",
            "NVIDIA_API_KEY": "frontend-model-secret",
            "HF_TOKEN": "unrelated-model-registry-secret",
            "TURN_PASSWORD": "unrelated-turn-secret",
        },
        config,
        config_path,
    )

    assert prepared["PATH"] == "/usr/bin"
    assert prepared["PUBLIC_REALTIME_KEY"] == "public-secret"
    assert prepared["BACKEND_BEARER"] == "backend-secret"
    assert "NEMOCLAW_VOICE_GATEWAY_ORIGIN" not in prepared
    assert prepared["REALTIME_UPSTREAM_API_KEY"] == "facade-to-nva-secret"
    assert prepared["CONFIG_REFERENCED_VALUE"] == "operator-selected-value"
    assert prepared["VOICECLAW_TLS_CERTFILE"] == "/run/tls/cert.pem"
    assert prepared["VOICECLAW_TLS_KEYFILE"] == "/run/tls/key.pem"
    assert prepared["VOICECLAW_CONFIG"] == str(config_path)
    assert "NVIDIA_API_KEY" not in prepared
    assert "COMMENT_ONLY_SECRET" not in prepared
    assert "HF_TOKEN" not in prepared
    assert "TURN_PASSWORD" not in prepared


def test_facade_child_receives_only_selected_backend_credential_environment(tmp_path) -> None:
    config = VoiceClawConfig(
        schema_version="voiceclaw.config.v2",
        server=ServerConfig(auth_mode="none"),
        backend_profiles={
            "selected": BackendProfile(kind="nemoclaw_committed_turn", credential_env="SELECTED_BACKEND_KEY"),
            "unused": BackendProfile(kind="nemoclaw_committed_turn", credential_env="UNUSED_BACKEND_KEY"),
        },
        default_backend="selected",
        realtime=RealtimeConfig(
            upstream_endpoint="ws://127.0.0.1:7861/v1/realtime",
            upstream_model="nvidia/nemotron-realtime-client-tools",
            credential_env="REALTIME_UPSTREAM_API_KEY",
        ),
        interaction=InteractionPolicy(),
        state=StateConfig(),
    )
    config_path = tmp_path / "voiceclaw.yaml"
    config_path.write_text("schema_version: voiceclaw.config.v2\n", encoding="utf-8")

    prepared = _facade_environment(
        {
            "SELECTED_BACKEND_KEY": "selected",
            "UNUSED_BACKEND_KEY": "unused",
            "REALTIME_UPSTREAM_API_KEY": "upstream",
        },
        config,
        config_path,
    )

    assert prepared["SELECTED_BACKEND_KEY"] == "selected"
    assert "UNUSED_BACKEND_KEY" not in prepared


def test_container_rejects_secret_readable_by_nva_group(tmp_path) -> None:
    secret = tmp_path / "secret"
    secret.write_text("x" * 32, encoding="ascii")
    os.chmod(secret, 0o640)
    metadata = secret.stat()
    identity = MappingProxyType({"user": metadata.st_uid + 1, "group": metadata.st_gid, "extra_groups": ()})

    with pytest.raises(ConfigurationError, match="readable by the realtime-model identity"):
        _assert_not_readable_by_identity(str(secret), identity, "backend credential")


def test_container_accepts_secret_hidden_from_nva_identity(tmp_path) -> None:
    secret = tmp_path / "secret"
    secret.write_text("x" * 32, encoding="ascii")
    os.chmod(secret, 0o640)
    metadata = secret.stat()
    identity = MappingProxyType(
        {"user": metadata.st_uid + 10_000, "group": metadata.st_gid + 10_000, "extra_groups": ()}
    )

    _assert_not_readable_by_identity(str(secret), identity, "backend credential")


def test_container_rejects_secret_readable_by_supplemental_group(tmp_path) -> None:
    secret = tmp_path / "secret"
    secret.write_text("x" * 32, encoding="ascii")
    os.chmod(secret, 0o640)
    metadata = secret.stat()
    identity = MappingProxyType(
        {
            "user": metadata.st_uid + 10_000,
            "group": metadata.st_gid + 10_000,
            "extra_groups": (metadata.st_gid,),
        }
    )

    with pytest.raises(ConfigurationError, match="facade identity"):
        _assert_not_readable_by_identity(
            str(secret),
            identity,
            "provider credential",
            identity_label="facade identity",
        )


def test_container_provider_secret_requires_owner_only_permissions(tmp_path) -> None:
    secret = tmp_path / "provider-secret"
    secret.write_text("x" * 32, encoding="ascii")
    os.chmod(secret, 0o640)

    with pytest.raises(ConfigurationError, match="owner-only permissions"):
        _assert_owner_only_secret(str(secret), "provider credential")

    os.chmod(secret, 0o600)
    _assert_owner_only_secret(str(secret), "provider credential")


def test_container_provider_secret_parent_is_not_mutable_by_facade(tmp_path) -> None:
    credential_directory = tmp_path / "credentials"
    credential_directory.mkdir(mode=0o700)
    secret = credential_directory / "speech"
    secret.write_text("x" * 32, encoding="ascii")
    secret.chmod(0o400)
    identity = MappingProxyType({"user": os.getuid(), "group": os.getgid(), "extra_groups": ()})

    with pytest.raises(ConfigurationError, match="mutable by the facade identity"):
        _assert_not_mutable_by_identity(
            str(secret),
            identity,
            "provider credential",
            identity_label="facade identity",
        )

    credential_directory.chmod(0o500)
    _assert_not_mutable_by_identity(
        str(secret),
        identity,
        "provider credential",
        identity_label="facade identity",
    )


def test_container_prepares_empty_managed_volume(tmp_path) -> None:
    volume_root = tmp_path / "voiceclaw"
    identity = MappingProxyType({"user": os.getuid(), "group": os.getgid(), "extra_groups": ()})

    _prepare_managed_volume_layout(
        volume_root,
        identity,
        root_uid=os.getuid(),
        root_gid=os.getgid(),
    )

    expected_layout = {
        volume_root: 0o755,
        volume_root / "config": 0o700,
        volume_root / "credentials": 0o710,
        volume_root / "state": 0o700,
    }
    for path, expected_mode in expected_layout.items():
        metadata = path.lstat()
        assert stat.S_ISDIR(metadata.st_mode)
        assert not path.is_symlink()
        assert stat.S_IMODE(metadata.st_mode) == expected_mode
        assert metadata.st_uid == os.getuid()
        assert metadata.st_gid == os.getgid()


def test_container_prepares_managed_volume_with_prepopulated_credentials(tmp_path) -> None:
    volume_root = tmp_path / "voiceclaw"
    credential_directory = volume_root / "credentials"
    credential_directory.mkdir(parents=True, mode=0o755)
    credential_file = credential_directory / "selected-agent"
    credential_file.write_text("opaque-credential", encoding="utf-8")
    credential_file.chmod(0o400)
    identity = MappingProxyType({"user": os.getuid(), "group": os.getgid(), "extra_groups": ()})

    _prepare_managed_volume_layout(
        volume_root,
        identity,
        root_uid=os.getuid(),
        root_gid=os.getgid(),
    )

    assert credential_file.read_text(encoding="utf-8") == "opaque-credential"
    assert stat.S_IMODE(credential_file.stat().st_mode) == 0o400
    assert stat.S_IMODE(volume_root.stat().st_mode) == 0o755
    assert credential_directory.stat().st_gid == os.getgid()
    assert stat.S_IMODE(credential_directory.stat().st_mode) == 0o710
    assert stat.S_IMODE((volume_root / "config").stat().st_mode) == 0o700
    assert stat.S_IMODE((volume_root / "state").stat().st_mode) == 0o700


def test_container_rejects_symlink_in_managed_volume_layout(tmp_path) -> None:
    volume_root = tmp_path / "voiceclaw"
    volume_root.mkdir()
    external_config = tmp_path / "external-config"
    external_config.mkdir()
    (volume_root / "config").symlink_to(external_config, target_is_directory=True)
    identity = MappingProxyType({"user": os.getuid(), "group": os.getgid(), "extra_groups": ()})

    with pytest.raises(ConfigurationError, match="managed volume path must be a directory"):
        _prepare_managed_volume_layout(
            volume_root,
            identity,
            root_uid=os.getuid(),
            root_gid=os.getgid(),
        )


def test_container_stages_provider_secret_as_an_nva_only_file(tmp_path) -> None:
    source = tmp_path / "source-key"
    source.write_text("provider-secret\n", encoding="utf-8")
    source.chmod(0o600)
    destination = tmp_path / "runtime" / "nvidia-api-key"
    identity = MappingProxyType({"user": os.getuid(), "group": os.getgid(), "extra_groups": ()})

    staged = _stage_nva_file_credential(
        CredentialReference(file=str(source)),
        destination,
        identity,
    )
    prepared = bind_nva_credential(
        {"NVIDIA_API_KEY": "ambient-secret"},
        staged,
        source_environment={},
    )

    assert staged == CredentialReference(file=str(destination))
    assert destination.read_text(encoding="utf-8") == "provider-secret"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o400
    assert prepared == {"NVIDIA_API_KEY": "provider-secret"}
    assert "NVIDIA_API_KEY_FILE" not in prepared


def test_external_selected_profile_still_enforces_provider_file_isolation(tmp_path) -> None:
    provider_secret = tmp_path / "provider-secret"
    provider_secret.write_text("provider-secret", encoding="utf-8")
    os.chmod(provider_secret, 0o600)
    config = parse_config(
        {
            "schema_version": "voiceclaw.config.v3",
            "frontend_profiles": {
                "selected_external": {
                    "kind": "openai_realtime",
                    "endpoint": "wss://127.0.0.1:9443/v1/realtime",
                    "model": "provider/realtime",
                },
                "unused_bundled": _bundled_frontend(credential_file=str(provider_secret)),
            },
            "default_frontend": "selected_external",
            "backend_profiles": {"none": {"kind": "none"}},
            "default_backend": "none",
        }
    )
    metadata = provider_secret.stat()
    facade_identity = MappingProxyType({"user": metadata.st_uid, "group": metadata.st_gid + 10_000, "extra_groups": ()})

    with pytest.raises(ConfigurationError, match="readable by the facade identity"):
        _assert_provider_files_isolated(config, facade_identity)


def test_unselected_missing_backend_secret_does_not_block_bundled_startup(tmp_path) -> None:
    selected = tmp_path / "selected"
    unselected_existing = tmp_path / "unselected-existing"
    unselected_missing = tmp_path / "unselected-missing"
    private_key = tmp_path / "tls-key"
    public_master = tmp_path / "public-master"
    unselected_existing.touch()
    config = VoiceClawConfig(
        schema_version="voiceclaw.config.v2",
        server=ServerConfig(auth_mode="ephemeral", api_key_file=str(public_master)),
        backend_profiles={
            "selected": BackendProfile(kind="nemoclaw", credential_file=str(selected)),
            "existing": BackendProfile(kind="nemoclaw", credential_file=str(unselected_existing)),
            "missing": BackendProfile(kind="nemoclaw", credential_file=str(unselected_missing)),
        },
        default_backend="selected",
        realtime=RealtimeConfig(
            upstream_endpoint="ws://127.0.0.1:7861/v1/realtime",
            upstream_model="nvidia/nemotron-realtime-client-tools",
        ),
        interaction=InteractionPolicy(),
        state=StateConfig(),
    )

    assert _facade_secret_files_for_nva(config, str(private_key)) == {
        str(selected),
        str(unselected_existing),
        str(private_key),
        str(public_master),
    }


def test_container_state_path_is_absolute_and_preflighted_as_facade(tmp_path, monkeypatch) -> None:
    identity = MappingProxyType({"user": 10001, "group": 10001, "extra_groups": (1000,)})
    nva_identity = MappingProxyType({"user": 10002, "group": 10002, "extra_groups": ()})
    with pytest.raises(ConfigurationError, match="must be absolute"):
        _assert_state_path_usable("state.db", identity)
    with pytest.raises(ConfigurationError, match="parent directory does not exist"):
        _assert_state_path_usable(str(tmp_path / "missing" / "state.db"), identity)

    calls: list[dict[str, object]] = []

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(kwargs)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("voiceclaw.container.subprocess.run", run)
    _assert_state_path_usable(str(tmp_path / "state.db"), identity, forbidden_identity=nva_identity)

    assert calls[0]["user"] == 10001
    assert calls[0]["group"] == 10001
    assert calls[0]["extra_groups"] == (1000,)
    assert calls[1]["user"] == 10002
    assert calls[1]["group"] == 10002
    assert calls[1]["extra_groups"] == ()


def test_container_rejects_state_accessible_to_the_bundled_frontend(tmp_path, monkeypatch) -> None:
    identity = MappingProxyType({"user": 10001, "group": 10001, "extra_groups": (1000,)})
    nva_identity = MappingProxyType({"user": 10002, "group": 10002, "extra_groups": ()})
    return_codes = iter((0, 1))

    def run(*args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args, next(return_codes))

    monkeypatch.setattr("voiceclaw.container.subprocess.run", run)
    with pytest.raises(ConfigurationError, match="accessible to the bundled realtime-model identity"):
        _assert_state_path_usable(str(tmp_path / "state.db"), identity, forbidden_identity=nva_identity)


def test_state_isolation_probe_accepts_a_nontraversable_parent(tmp_path) -> None:
    parent = tmp_path / "private-state"
    parent.mkdir(mode=0o700)
    state_path = parent / "state.db"
    state_path.touch(mode=0o600)
    original_mode = stat.S_IMODE(parent.stat().st_mode)
    try:
        # A non-root test runner cannot switch to the image's NVA UID. Removing
        # its own traversal permission exercises the same EACCES boundary as a
        # different UID encountering the production 0700 directory.
        parent.chmod(0)
        completed = subprocess.run(
            [sys.executable, "-c", _STATE_ISOLATION_CHECK, str(state_path)],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    finally:
        parent.chmod(original_mode)

    assert completed.returncode == 0
    assert completed.stderr == ""


def test_container_preflights_operational_files_as_facade_identity(tmp_path, monkeypatch) -> None:
    operational_file = tmp_path / "catalog.yaml"
    operational_file.touch()
    identity = MappingProxyType({"user": 10001, "group": 10001, "extra_groups": (1000,)})
    calls: list[tuple[object, dict[str, object]]] = []

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("voiceclaw.container.subprocess.run", run)
    _assert_files_readable_by_identity(
        [str(operational_file), str(operational_file)],
        identity,
        identity_label="voiceclaw facade identity",
    )

    command = calls[0][0][0]
    assert isinstance(command, list)
    assert command.count(str(operational_file)) == 1
    assert calls[0][1]["user"] == 10001
    assert calls[0][1]["extra_groups"] == (1000,)


def test_facade_configuration_preflight_drops_privilege_and_uses_allowlisted_environment(
    tmp_path,
    monkeypatch,
) -> None:
    config = VoiceClawConfig(
        schema_version="voiceclaw.config.v2",
        server=ServerConfig(auth_mode="none"),
        backend_profiles={
            "selected": BackendProfile(kind="nemoclaw_committed_turn", credential_env="SELECTED_BACKEND_KEY"),
            "unused": BackendProfile(kind="nemoclaw_committed_turn", credential_env="UNUSED_BACKEND_KEY"),
        },
        default_backend="selected",
        realtime=RealtimeConfig(
            upstream_endpoint="ws://127.0.0.1:7861/v1/realtime",
            upstream_model="nvidia/nemotron-realtime-client-tools",
            credential_env="REALTIME_UPSTREAM_API_KEY",
        ),
        interaction=InteractionPolicy(),
        state=StateConfig(),
    )
    config_path = tmp_path / "voiceclaw.yaml"
    config_path.write_text("schema_version: voiceclaw.config.v2\n", encoding="utf-8")
    environment = _facade_environment(
        {
            "PATH": "/usr/bin",
            "HTTP_PROXY": "http://proxy.invalid",
            "SELECTED_BACKEND_KEY": "selected-secret",
            "UNUSED_BACKEND_KEY": "unused-secret",
            "REALTIME_UPSTREAM_API_KEY": "private-loopback-secret",
            "AMBIENT_PARENT_SECRET": "must-not-cross",
        },
        config,
        config_path,
    )
    identity = MappingProxyType({"user": 10001, "group": 10001, "extra_groups": (1000,)})
    calls: list[tuple[object, dict[str, object]]] = []

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("voiceclaw.container.subprocess.run", run)
    _preflight_facade_configuration(
        config_path,
        "127.0.0.1",
        environment,
        identity,
        bundled_nva_supervised=True,
    )

    command = calls[0][0][0]
    assert isinstance(command, list)
    assert command[-3:] == [str(config_path), "127.0.0.1", "1"]
    assert calls[0][1]["user"] == 10001
    assert calls[0][1]["group"] == 10001
    assert calls[0][1]["extra_groups"] == (1000,)
    assert calls[0][1]["env"] == environment
    assert "HTTP_PROXY" not in environment
    assert environment["SELECTED_BACKEND_KEY"] == "selected-secret"
    assert "UNUSED_BACKEND_KEY" not in environment
    assert "AMBIENT_PARENT_SECRET" not in environment


def test_facade_configuration_preflight_rejects_child_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "voiceclaw.container.subprocess.run",
        lambda *args, **_kwargs: subprocess.CompletedProcess(args, 1),
    )

    with pytest.raises(ConfigurationError, match="facade identity rejected"):
        _preflight_facade_configuration(
            tmp_path / "voiceclaw.yaml",
            "127.0.0.1",
            {"PATH": "/usr/bin"},
            MappingProxyType({"user": 10001, "group": 10001, "extra_groups": ()}),
            bundled_nva_supervised=False,
        )


def test_nva_profile_preflight_uses_effective_prompt_catalog(tmp_path, monkeypatch) -> None:
    prompts = tmp_path / "prompts.yaml"
    prompts.write_text("operator_prompt:\n  content: Be helpful.\n", encoding="utf-8")
    identity = MappingProxyType({"user": 10002, "group": 10002, "extra_groups": ()})
    calls: list[tuple[object, ...]] = []

    def run(*args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("voiceclaw.container.subprocess.run", run)
    _preflight_nva_profile(
        Path("/nva/python"),
        Path("/nva/src/realtime_server.py"),
        "operator/realtime-model",
        "operator_prompt",
        {"PROMPT_FILE_PATH": str(prompts)},
        identity,
    )

    command = calls[0][0]
    assert isinstance(command, list)
    assert command[-3] == "operator/realtime-model"
    assert command[-2] == str(prompts)
    assert command[-1] == "operator_prompt"


def test_nva_profile_preflight_rejects_internal_or_missing_prompt(tmp_path, monkeypatch) -> None:
    prompts = tmp_path / "prompts.yaml"
    prompts.write_text("private_prompt:\n  content: Hidden.\n  internal: true\n", encoding="utf-8")
    monkeypatch.setattr(
        "voiceclaw.container.subprocess.run",
        lambda *args, **_kwargs: subprocess.CompletedProcess(args, 1),
    )

    with pytest.raises(ConfigurationError, match="NVA rejected"):
        _preflight_nva_profile(
            Path("/nva/python"),
            Path("/nva/src/realtime_server.py"),
            "operator/realtime-model",
            "private_prompt",
            {"PROMPT_FILE_PATH": str(prompts)},
            MappingProxyType({"user": 10002, "group": 10002, "extra_groups": ()}),
        )


def test_container_rejects_secret_with_posix_access_acl(tmp_path, monkeypatch) -> None:
    secret = tmp_path / "secret"
    secret.write_text("x" * 32, encoding="ascii")
    os.chmod(secret, 0o600)
    metadata = secret.stat()
    identity = MappingProxyType(
        {"user": metadata.st_uid + 10_000, "group": metadata.st_gid + 10_000, "extra_groups": ()}
    )
    monkeypatch.setattr(os, "getxattr", lambda *_args, **_kwargs: b"extended-acl")

    with pytest.raises(ConfigurationError, match="must not have a POSIX access ACL"):
        _assert_not_readable_by_identity(str(secret), identity, "backend credential")
