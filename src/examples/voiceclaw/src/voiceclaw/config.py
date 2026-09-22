# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Strict YAML configuration for the VoiceClaw facade and backend adapters."""

from __future__ import annotations

import ipaddress
import math
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

import yaml

from voiceclaw.interaction_profiles import (
    INTERACTION_PROFILE_SCHEMA,
    InteractionProfileCatalog,
    InteractionProfileError,
    parse_interaction_profile_catalog,
)

_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SEMANTIC_NAME = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+")
_PROPERTY_NAME = re.compile(r"[a-z][a-z0-9_]*")
_CONFIG_SCHEMA_V2 = "voiceclaw.config.v2"
_CONFIG_SCHEMA_V3 = "voiceclaw.config.v3"
_SUPPORTED_CONFIG_SCHEMAS = frozenset({_CONFIG_SCHEMA_V2, _CONFIG_SCHEMA_V3})
_PRIVATE_REALTIME_ENV_NAMES = frozenset({"REALTIME_API_KEY", "REALTIME_UPSTREAM_API_KEY"})
_MAX_MODEL_COPY_CHARACTERS = 4096
_MAX_PROVIDER_OPTION_DEPTH = 12
_MAX_PROVIDER_OPTION_ITEMS = 512
_MAX_PROVIDER_OPTION_KEY_CHARACTERS = 128
_MAX_PROVIDER_OPTION_STRING_CHARACTERS = 16_384
_SERVICE_ID = re.compile(r"[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*")
_BUNDLED_REALTIME_ENDPOINT = "ws://127.0.0.1:7861/v1/realtime"
_BUNDLED_REALTIME_CREDENTIAL_ENV = "REALTIME_UPSTREAM_API_KEY"
_NVA_PIPELINE_MODE = "generic-assistant"
_RESERVED_OPERATIONAL_ENV_NAMES = frozenset(
    {
        "ALL_PROXY",
        "CURL_CA_BUNDLE",
        "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH",
        "HOME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "LIBRARY_PATH",
        "LOGNAME",
        "NO_PROXY",
        "NEMOCLAW_AGENT",
        "NEMOCLAW_AGENT_CREDENTIAL_FILE",
        "NEMOCLAW_AGENT_GATEWAY_URL",
        "NEMOCLAW_DEPLOYMENT_CREDENTIAL_FILE",
        "NEMOCLAW_GATEWAY_PORT",
        "NEMOCLAW_RUNTIME_IDENTITY",
        "NEMOCLAW_RUNTIME_PROFILE",
        "NEMOCLAW_SANDBOX",
        "NEMOCLAW_SOURCE_DIR",
        "NEMOCLAW_VOICE_GATEWAY_BEARER_FILE",
        "NEMOCLAW_VOICE_GATEWAY_PORT",
        "NVA_RUNTIME_CONFIG_DIR",
        "NVIDIA_API_KEY_FILE",
        "OLDPWD",
        "PATH",
        "PIPELINE_TLS",
        "PROMPT_FILE_PATH",
        "PROMPT_SELECTOR",
        "PWD",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED",
        "REQUESTS_CA_BUNDLE",
        "SERVICES_CLOUD_PATH",
        "SERVICES_LOCAL_PATH",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TOOLS_FILE_PATH",
        "TZ",
        "USER",
        "UVICORN_WORKERS",
        "VOICECLAW_CONFIG",
        "VOICECLAW_CONFIG_FILE_HOST",
        "VOICECLAW_CLIENT_SECRET_FILE",
        "VOICECLAW_FILE_GID",
        "VOICECLAW_FRONTEND_RUNTIME_DIR",
        "VOICECLAW_HOST_IP",
        "VOICECLAW_INTERNAL_READY_TIMEOUT",
        "VOICECLAW_NVA_PYTHON",
        "VOICECLAW_NVA_SERVER",
        "VOICECLAW_OPERATOR_FILES_DIR",
        "VOICECLAW_OPERATOR_FILES_HOST",
        "VOICECLAW_RUNTIME_ROOT",
        "VOICECLAW_SERVER_HOST",
        "VOICECLAW_STATE_PATH",
        "VOICECLAW_TLS_CERTFILE",
        "VOICECLAW_TLS_KEYFILE",
        "XDG_CACHE_HOME",
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)
_RESERVED_OPERATIONAL_ENV_PREFIXES = ("NVA_", "OTEL_")
_PROVIDER_SECRET_OPTION_PARTS = frozenset(
    {
        "apikey",
        "auth",
        "authorization",
        "bearer",
        "credential",
        "credentials",
        "header",
        "headers",
        "password",
        "secret",
        "token",
    }
)
_BACKEND_SECRET_SETTING_PARTS = _PROVIDER_SECRET_OPTION_PARTS - {"header", "headers"}
_PROVIDER_CONTROL_OPTION_NAMES = frozenset(
    {
        "base_url",
        "developer_message",
        "developer_prompt",
        "endpoint",
        "extra_headers",
        "forced_tool_call_stops",
        "grammar",
        "function_call",
        "functions",
        "http_headers",
        "input",
        "input_messages",
        "instructions",
        "messages",
        "max_tokens",
        "model",
        "parallel_tool_calls",
        "prompt",
        "proxy",
        "response_format",
        "response_schema",
        "stop",
        "stop_sequences",
        "stop_token_ids",
        "stream",
        "stream_options",
        "structured_output",
        "structured_outputs",
        "system",
        "system_message",
        "system_prompt",
        "temperature",
        "tool_choice",
        "tools",
    }
)


class ConfigurationError(ValueError):
    """Raised when VoiceClaw configuration is incomplete or inconsistent."""


class FrontendKind(StrEnum):
    """Closed frontend implementations supported by this schema version."""

    BUNDLED_NVA = "bundled_nva"
    OPENAI_REALTIME = "openai_realtime"


class ListenerSecurity(StrEnum):
    """Explicit trust boundary for the public facade listener."""

    LOOPBACK = "loopback"
    PRIVATE_NETWORK = "private_network"
    TLS = "tls"


class NvaPlatform(StrEnum):
    """NVA service-catalog placement selected for a bundled cascade."""

    CLOUD = "cloud"
    SERVER = "server"
    SINGLEGPU = "singlegpu"


class ProviderKind(StrEnum):
    """Provider protocols currently understood by the bundled NVA runtime."""

    OPENAI_COMPATIBLE = "openai_compatible"
    NVIDIA_GRPC = "nvidia_grpc"


class TtsSynthesisMode(StrEnum):
    """NVA TTS streaming modes available to a registered service."""

    STITCHED = "stitched"
    PER_SENTENCE = "per_sentence"


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a VoiceClaw configuration mapping",
                node.start_mark,
                "mapping keys must be scalar values",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a VoiceClaw configuration mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be an object")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigurationError(f"unknown keys in {path}: {', '.join(unknown)}")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{path} must be a non-empty string")
    return value.strip()


def _bounded_string(value: object, path: str, maximum: int) -> str:
    text = _string(value, path)
    if len(text) > maximum:
        raise ConfigurationError(f"{path} must be at most {maximum} characters")
    return text


def _optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _credential_environment_name(value: object, path: str) -> str:
    name = _string(value, path)
    if _ENV_NAME.fullmatch(name) is None:
        raise ConfigurationError(f"{path} must be an environment-variable name")
    if name in _RESERVED_OPERATIONAL_ENV_NAMES or name.startswith(_RESERVED_OPERATIONAL_ENV_PREFIXES):
        raise ConfigurationError(f"{path} must not use reserved operational environment variable {name!r}")
    return name


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{path} must be a positive integer")
    return value


def _positive_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f"{path} must be a finite positive number")
    return float(value)


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{path} must be a boolean")
    return value


def _deep_freeze_json(value: Any) -> Any:
    """Return a detached recursively immutable JSON-compatible value."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """VoiceClaw service listener configuration."""

    host: str = "127.0.0.1"
    port: int = 7860
    listener_security: ListenerSecurity = ListenerSecurity.LOOPBACK
    auth_mode: str = "none"
    api_key_env: str | None = None
    api_key_file: str | None = None
    client_secret_lifetime_seconds: int = 600


@dataclass(frozen=True, slots=True)
class RealtimeConfig:
    """Public facade to private OpenAI Realtime-compatible upstream binding."""

    upstream_endpoint: str
    upstream_model: str
    public_model: str = "nvidia/voiceclaw"
    credential_env: str | None = None
    credential_file: str | None = None
    connect_timeout_seconds: float = 15.0
    bootstrap_timeout_seconds: float = 15.0
    max_event_bytes: int = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CredentialReference:
    """Reference to a credential supplied outside the configuration file."""

    env: str | None = None
    file: str | None = None


@dataclass(frozen=True, slots=True)
class LlmServiceConfig:
    """One registered OpenAI-compatible LLM used by a bundled cascade."""

    id: str
    name: str
    provider: ProviderKind
    endpoint: str
    model: str
    credential: CredentialReference | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    extra_params: Mapping[str, Any] = field(default_factory=dict)
    supports_tokenize: bool = False
    realtime_max_output_tokens: int | None = None
    forced_tool_call_stops: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Detach provider parameters from the mutable YAML mapping."""
        _validate_provider_options(self.extra_params, "llm.extra_params")
        object.__setattr__(self, "extra_params", _deep_freeze_json(self.extra_params))


@dataclass(frozen=True, slots=True)
class AsrServiceConfig:
    """One registered streaming ASR service used by a bundled cascade."""

    id: str
    name: str
    provider: ProviderKind
    endpoint: str
    model: str
    credential: CredentialReference | None = None
    tls: bool = False
    function_id: str | None = None
    language_code: str | None = None


@dataclass(frozen=True, slots=True)
class TtsServiceConfig:
    """One registered streaming TTS service used by a bundled cascade."""

    id: str
    name: str
    provider: ProviderKind
    endpoint: str
    model: str
    voice: str
    credential: CredentialReference | None = None
    tls: bool = False
    function_id: str | None = None
    synthesis_mode: TtsSynthesisMode = TtsSynthesisMode.STITCHED
    language_code: str | None = None


@dataclass(frozen=True, slots=True)
class CascadedServicesConfig:
    """The three provider registrations required by a cascaded frontend."""

    llm: LlmServiceConfig
    asr: AsrServiceConfig
    tts: TtsServiceConfig


def shared_bundled_service_credential(
    services: CascadedServicesConfig,
    *,
    path: str = "bundled_nva.services",
) -> CredentialReference | None:
    """Return the one credential NVA may expose, rejecting partial isolation."""
    credentials = {
        "llm": services.llm.credential,
        "asr": services.asr.credential,
        "tts": services.tts.credential,
    }
    configured = {name: credential for name, credential in credentials.items() if credential is not None}
    if not configured:
        return None
    if len(configured) != len(credentials):
        raise ConfigurationError(
            f"{path} must configure the same credential source on llm, asr, and tts because "
            "the current private NVA runtime exposes one shared model key to all three services"
        )

    def identity(credential: CredentialReference) -> tuple[str, str]:
        if credential.env is not None:
            return ("env", credential.env)
        if credential.file is None:  # pragma: no cover - references are constructed by the strict parser
            raise ConfigurationError(f"{path} credential must select env or file")
        return ("file", os.path.realpath(credential.file))

    selected = services.llm.credential
    assert selected is not None
    selected_identity = identity(selected)
    if any(identity(credential) != selected_identity for credential in configured.values()):
        raise ConfigurationError(
            f"{path} must configure the same credential source on llm, asr, and tts because "
            "the current private NVA runtime exposes one shared model key to all three services"
        )
    return selected


@dataclass(frozen=True, slots=True)
class BundledNvaFrontendProfile:
    """A cascaded frontend hosted by the NVA runtime bundled with VoiceClaw."""

    kind: FrontendKind
    public_model: str
    realtime_model: str
    platform: NvaPlatform
    pipeline_mode: str
    services: CascadedServicesConfig
    connect_timeout_seconds: float = 15.0
    bootstrap_timeout_seconds: float = 15.0
    max_event_bytes: int = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class OpenAIRealtimeFrontendProfile:
    """An externally hosted OpenAI Realtime-compatible frontend."""

    kind: FrontendKind
    endpoint: str
    model: str
    public_model: str
    credential: CredentialReference | None = None
    connect_timeout_seconds: float = 15.0
    bootstrap_timeout_seconds: float = 15.0
    max_event_bytes: int = 16 * 1024 * 1024


type FrontendProfile = BundledNvaFrontendProfile | OpenAIRealtimeFrontendProfile


@dataclass(frozen=True, slots=True)
class ModelContractsConfig:
    """Versioned model-contract profile and optional operator override."""

    profile: str = "default"
    path: str | None = None


@dataclass(frozen=True, slots=True)
class InteractionProfilesConfig:
    """Versioned interaction-profile catalog selected by the operator."""

    path: str | None = None
    inline_catalog: InteractionProfileCatalog | None = None


@dataclass(frozen=True, slots=True)
class ToolCopyOverride:
    """Trusted prose-only override for one protected semantic tool."""

    description: str | None = None
    properties: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Detach property copy from the mutable configuration mapping."""
        object.__setattr__(self, "properties", MappingProxyType(dict(self.properties)))


@dataclass(frozen=True, slots=True)
class BackendProfile:
    """Operator-selected backend adapter binding."""

    kind: str
    credential_env: str | None = None
    credential_file: str | None = None
    settings: Mapping[str, Any] = field(default_factory=dict)
    interaction_profile: str = "stateless"
    tool_copy: Mapping[str, ToolCopyOverride] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Detach adapter-owned settings from the mutable YAML mapping."""
        object.__setattr__(self, "settings", _deep_freeze_json(self.settings))
        object.__setattr__(self, "tool_copy", MappingProxyType(dict(self.tool_copy)))


@dataclass(frozen=True, slots=True)
class InteractionPolicy:
    """Active speech-queue and context bounds independent of providers."""

    max_pending_speech: int = 32
    context_character_budget: int = 24_000
    request_summary_character_limit: int = 512
    retained_request_limit: int = 8
    turn_routing_mode: str = "model"


@dataclass(frozen=True, slots=True)
class StateConfig:
    """Local recovery-state configuration."""

    kind: str = "sqlite"
    path: str = "/var/lib/voiceclaw/state/state.db"


@dataclass(frozen=True, slots=True)
class VoiceClawConfig:
    """Validated VoiceClaw facade and backend composition."""

    schema_version: str
    server: ServerConfig
    backend_profiles: Mapping[str, BackendProfile]
    default_backend: str
    realtime: RealtimeConfig | None
    interaction: InteractionPolicy
    state: StateConfig
    model_contracts: ModelContractsConfig = field(default_factory=ModelContractsConfig)
    interaction_profiles: InteractionProfilesConfig = field(default_factory=InteractionProfilesConfig)
    frontend_profiles: Mapping[str, FrontendProfile] = field(default_factory=dict)
    default_frontend: str | None = None

    def __post_init__(self) -> None:
        """Expose immutable registry mappings after validation."""
        object.__setattr__(self, "backend_profiles", MappingProxyType(dict(self.backend_profiles)))
        object.__setattr__(self, "frontend_profiles", MappingProxyType(dict(self.frontend_profiles)))

    @property
    def selected_frontend(self) -> FrontendProfile | None:
        """Return the selected typed v3 profile, or ``None`` for legacy v2."""
        if self.default_frontend is None:
            return None
        return self.frontend_profiles[self.default_frontend]


def load_config(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> VoiceClawConfig:
    """Load and strictly validate one YAML configuration file."""
    source = Path(path)
    try:
        payload = source.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(f"cannot read VoiceClaw config: {source}") from error
    try:
        raw = yaml.load(payload, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ConfigurationError(f"invalid YAML in VoiceClaw config {source}: {error}") from error
    expanded = _expand_environment(raw, os.environ if environ is None else environ)
    return parse_config(_mapping(expanded, "config"))


def parse_config(raw: Mapping[str, Any]) -> VoiceClawConfig:
    """Validate parsed configuration data."""
    schema_version = _string(raw.get("schema_version"), "schema_version")
    if schema_version not in _SUPPORTED_CONFIG_SCHEMAS:
        supported = ", ".join(sorted(_SUPPORTED_CONFIG_SCHEMAS))
        raise ConfigurationError(f"schema_version must be one of: {supported}")

    allowed = {
        "schema_version",
        "server",
        "backend_profiles",
        "default_backend",
        "interaction",
        "state",
        "model_contracts",
        "interaction_profiles",
    }
    if schema_version == _CONFIG_SCHEMA_V2:
        allowed.add("realtime")
    else:
        allowed.update({"frontend_profiles", "default_frontend"})
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigurationError(f"unknown top-level configuration keys: {', '.join(unknown)}")

    server_raw = _mapping(raw.get("server", {}), "server")
    _reject_unknown(
        server_raw,
        {
            "host",
            "port",
            "listener_security",
            "auth_mode",
            "api_key_env",
            "api_key_file",
            "client_secret_lifetime_seconds",
        },
        "server",
    )
    try:
        listener_security = ListenerSecurity(
            _string(server_raw.get("listener_security", ListenerSecurity.LOOPBACK), "server.listener_security")
        )
    except ValueError as error:
        supported = ", ".join(item.value for item in ListenerSecurity)
        raise ConfigurationError(f"server.listener_security must be one of: {supported}") from error
    auth_mode = _string(server_raw.get("auth_mode", "none"), "server.auth_mode")
    if auth_mode not in {"none", "ephemeral"}:
        raise ConfigurationError("server.auth_mode must be none or ephemeral")
    raw_api_key_env = server_raw.get("api_key_env")
    api_key_env = (
        None if raw_api_key_env is None else _credential_environment_name(raw_api_key_env, "server.api_key_env")
    )
    api_key_file = _optional_string(server_raw.get("api_key_file"), "server.api_key_file")
    if api_key_file is not None and not Path(api_key_file).is_absolute():
        raise ConfigurationError("server.api_key_file must be an absolute path")
    configured_public_credentials = sum(value is not None for value in (api_key_env, api_key_file))
    if auth_mode == "none" and configured_public_credentials:
        raise ConfigurationError(
            "server.api_key_env and server.api_key_file must be omitted when server.auth_mode is none"
        )
    if auth_mode == "ephemeral" and configured_public_credentials != 1:
        raise ConfigurationError(
            "exactly one of server.api_key_env or server.api_key_file is required when server.auth_mode is ephemeral"
        )
    server = ServerConfig(
        host=_string(server_raw.get("host", "127.0.0.1"), "server.host"),
        port=_positive_int(server_raw.get("port", 7860), "server.port"),
        listener_security=listener_security,
        auth_mode=auth_mode,
        api_key_env=api_key_env,
        api_key_file=api_key_file,
        client_secret_lifetime_seconds=_positive_int(
            server_raw.get("client_secret_lifetime_seconds", 600),
            "server.client_secret_lifetime_seconds",
        ),
    )
    if server.port > 65_535:
        raise ConfigurationError("server.port must be at most 65535")
    if not 10 <= server.client_secret_lifetime_seconds <= 3600:
        raise ConfigurationError("server.client_secret_lifetime_seconds must be between 10 and 3600")

    backends = _parse_backends(_mapping(raw.get("backend_profiles"), "backend_profiles"))

    default_backend = _string(raw.get("default_backend"), "default_backend")
    if default_backend not in backends:
        raise ConfigurationError(f"default_backend references unknown profile: {default_backend}")
    if schema_version == _CONFIG_SCHEMA_V2:
        frontend_profiles: dict[str, FrontendProfile] = {}
        default_frontend = None
        realtime = _parse_realtime(raw.get("realtime"))
    else:
        frontend_profiles = _parse_frontends(_mapping(raw.get("frontend_profiles"), "frontend_profiles"))
        default_frontend = _string(raw.get("default_frontend"), "default_frontend")
        if default_frontend not in frontend_profiles:
            raise ConfigurationError(f"default_frontend references unknown profile: {default_frontend}")
        realtime = _frontend_realtime(frontend_profiles[default_frontend])

    model_contracts_raw = _mapping(raw.get("model_contracts", {}), "model_contracts")
    _reject_unknown(model_contracts_raw, {"profile", "path"}, "model_contracts")
    model_contract_path = _optional_string(model_contracts_raw.get("path"), "model_contracts.path")
    if model_contract_path is not None and not Path(model_contract_path).is_absolute():
        raise ConfigurationError("model_contracts.path must be absolute")
    model_contracts = ModelContractsConfig(
        profile=_string(model_contracts_raw.get("profile", "default"), "model_contracts.profile"),
        path=model_contract_path,
    )

    interaction_profiles_raw = _mapping(raw.get("interaction_profiles", {}), "interaction_profiles")
    _reject_unknown(interaction_profiles_raw, {"path", "profiles"}, "interaction_profiles")
    if "path" in interaction_profiles_raw and "profiles" in interaction_profiles_raw:
        raise ConfigurationError("interaction_profiles.path and interaction_profiles.profiles are mutually exclusive")
    interaction_profiles_path = _optional_string(
        interaction_profiles_raw.get("path"),
        "interaction_profiles.path",
    )
    if interaction_profiles_path is not None and not Path(interaction_profiles_path).is_absolute():
        raise ConfigurationError("interaction_profiles.path must be absolute")
    inline_interaction_profiles = None
    if "profiles" in interaction_profiles_raw:
        profiles_raw = _mapping(interaction_profiles_raw["profiles"], "interaction_profiles.profiles")
        try:
            inline_interaction_profiles = parse_interaction_profile_catalog(
                {
                    "schema_version": INTERACTION_PROFILE_SCHEMA,
                    "profiles": profiles_raw,
                }
            )
        except InteractionProfileError as error:
            raise ConfigurationError(str(error)) from error
    interaction_profiles = InteractionProfilesConfig(
        path=interaction_profiles_path,
        inline_catalog=inline_interaction_profiles,
    )

    _validate_credential_boundaries(
        schema_version=schema_version,
        server=server,
        backends=backends,
        default_backend=default_backend,
        frontends=frontend_profiles,
        default_frontend=default_frontend,
        realtime=realtime,
    )

    interaction_raw = _mapping(raw.get("interaction", {}), "interaction")
    _reject_unknown(
        interaction_raw,
        {
            "max_pending_speech",
            "context_character_budget",
            "request_summary_character_limit",
            "retained_request_limit",
            "turn_routing_mode",
        },
        "interaction",
    )
    max_pending_speech = _positive_int(
        interaction_raw.get("max_pending_speech", 32),
        "interaction.max_pending_speech",
    )
    if max_pending_speech > 1024:
        raise ConfigurationError("interaction.max_pending_speech must be at most 1024")
    context_character_budget = _positive_int(
        interaction_raw.get("context_character_budget", 24_000),
        "interaction.context_character_budget",
    )
    if context_character_budget > 64_000:
        raise ConfigurationError("interaction.context_character_budget must be at most 64000")
    request_summary_character_limit = _positive_int(
        interaction_raw.get("request_summary_character_limit", 512),
        "interaction.request_summary_character_limit",
    )
    if request_summary_character_limit > 512:
        raise ConfigurationError("interaction.request_summary_character_limit must be at most 512")
    retained_request_limit = _positive_int(
        interaction_raw.get("retained_request_limit", 8),
        "interaction.retained_request_limit",
    )
    if retained_request_limit > 128:
        raise ConfigurationError("interaction.retained_request_limit must be at most 128")
    turn_routing_mode = _string(
        interaction_raw.get("turn_routing_mode", "model"),
        "interaction.turn_routing_mode",
    )
    if turn_routing_mode != "model":
        raise ConfigurationError("interaction.turn_routing_mode must be model")
    interaction = InteractionPolicy(
        max_pending_speech=max_pending_speech,
        context_character_budget=context_character_budget,
        request_summary_character_limit=request_summary_character_limit,
        retained_request_limit=retained_request_limit,
        turn_routing_mode=turn_routing_mode,
    )

    state_raw = _mapping(raw.get("state", {}), "state")
    _reject_unknown(state_raw, {"kind", "path"}, "state")
    state_kind = _string(state_raw.get("kind", "sqlite"), "state.kind")
    if state_kind != "sqlite":
        raise ConfigurationError("state.kind currently supports only sqlite")
    state = StateConfig(
        kind=state_kind,
        path=_string(state_raw.get("path", "/var/lib/voiceclaw/state/state.db"), "state.path"),
    )

    return VoiceClawConfig(
        schema_version=schema_version,
        server=server,
        backend_profiles=backends,
        default_backend=default_backend,
        realtime=realtime,
        interaction=interaction,
        state=state,
        model_contracts=model_contracts,
        interaction_profiles=interaction_profiles,
        frontend_profiles=frontend_profiles,
        default_frontend=default_frontend,
    )


@dataclass(frozen=True, slots=True)
class _CredentialUse:
    role: str
    path: str
    identities: tuple[tuple[str, str], ...]


def _credential_identities(credential: CredentialReference) -> tuple[tuple[str, str], ...]:
    if credential.env is not None:
        return (("env", credential.env),)
    if credential.file is None:  # pragma: no cover - CredentialReference is constructed by the strict parser
        raise ConfigurationError("credential reference must select env or file")
    canonical_path = os.path.realpath(credential.file)
    identities = [("file", canonical_path)]
    try:
        metadata = os.stat(canonical_path)
    except OSError:
        pass
    else:
        identities.append(("file inode", f"{metadata.st_dev}:{metadata.st_ino}"))
    return tuple(identities)


def _validate_credential_boundaries(
    *,
    schema_version: str,
    server: ServerConfig,
    backends: Mapping[str, BackendProfile],
    default_backend: str,
    frontends: Mapping[str, FrontendProfile],
    default_frontend: str | None,
    realtime: RealtimeConfig | None,
) -> None:
    """Reject one credential source reused across concurrently active roles."""
    uses = [
        _CredentialUse(
            role="private_realtime",
            path=f"VoiceClaw private Realtime transport ({name})",
            identities=(("env", name),),
        )
        for name in sorted(_PRIVATE_REALTIME_ENV_NAMES)
    ]
    selected_frontend = frontends.get(default_frontend) if default_frontend is not None else None
    if isinstance(selected_frontend, BundledNvaFrontendProfile):
        selected_credential = shared_bundled_service_credential(selected_frontend.services)
        if selected_credential is not None:
            # The selected bundled child receives either a developer env value
            # or a managed file path through one fixed process-local name.
            binding_name = "NVIDIA_API_KEY_FILE" if selected_credential.file is not None else "NVIDIA_API_KEY"
            uses.append(
                _CredentialUse(
                    role="frontend_model",
                    path=f"selected bundled frontend model ({binding_name})",
                    identities=(("env", binding_name),),
                )
            )
    if server.api_key_env is not None or server.api_key_file is not None:
        credential = CredentialReference(env=server.api_key_env, file=server.api_key_file)
        suffix = "env" if credential.env is not None else "file"
        uses.append(
            _CredentialUse(
                role="public_realtime",
                path=f"server.api_key_{suffix}",
                identities=_credential_identities(credential),
            )
        )
    profile_name = default_backend
    profile = backends[profile_name]
    if profile.credential_env is not None or profile.credential_file is not None:
        credential = CredentialReference(env=profile.credential_env, file=profile.credential_file)
        suffix = "env" if credential.env is not None else "file"
        uses.append(
            _CredentialUse(
                role="backend",
                path=f"backend_profiles.{profile_name}.credential.{suffix}",
                identities=_credential_identities(credential),
            )
        )

    if schema_version == _CONFIG_SCHEMA_V2:
        if realtime is not None and (realtime.credential_env is not None or realtime.credential_file is not None):
            credential = CredentialReference(env=realtime.credential_env, file=realtime.credential_file)
            suffix = "env" if credential.env is not None else "file"
            uses.append(
                _CredentialUse(
                    role="private_realtime",
                    path=f"realtime.credential.{suffix}",
                    identities=_credential_identities(credential),
                )
            )
    elif selected_frontend is not None and default_frontend is not None:
        if isinstance(selected_frontend, OpenAIRealtimeFrontendProfile):
            if selected_frontend.credential is not None:
                suffix = "env" if selected_frontend.credential.env is not None else "file"
                uses.append(
                    _CredentialUse(
                        role="private_realtime",
                        path=f"frontend_profiles.{default_frontend}.credential.{suffix}",
                        identities=_credential_identities(selected_frontend.credential),
                    )
                )
        else:
            service_credentials = {
                "llm": selected_frontend.services.llm.credential,
                "asr": selected_frontend.services.asr.credential,
                "tts": selected_frontend.services.tts.credential,
            }
            for service_name, credential in service_credentials.items():
                if credential is None:
                    continue
                suffix = "env" if credential.env is not None else "file"
                uses.append(
                    _CredentialUse(
                        role="frontend_model",
                        path=f"frontend_profiles.{default_frontend}.services.{service_name}.credential.{suffix}",
                        identities=_credential_identities(credential),
                    )
                )

    owners: dict[tuple[str, str], _CredentialUse] = {}
    for use in uses:
        for identity in use.identities:
            previous = owners.get(identity)
            if previous is None:
                owners[identity] = use
                continue
            if previous.role == use.role:
                continue
            source_kind, source_name = identity
            boundary = {
                "public_realtime": "the public Realtime credential",
                "private_realtime": "the private Realtime transport credential",
                "backend": "a backend credential",
                "frontend_model": "a frontend model credential",
            }[previous.role]
            raise ConfigurationError(
                f"{use.path} must not reuse {boundary} "
                f"({source_kind} source {source_name!r}) assigned to {previous.path}"
            )


def _parse_frontends(raw: Mapping[str, Any]) -> dict[str, FrontendProfile]:
    profiles: dict[str, FrontendProfile] = {}
    for raw_name, value in raw.items():
        name = _string(raw_name, "frontend_profiles key")
        if _SERVICE_ID.fullmatch(name) is None:
            raise ConfigurationError("frontend_profiles keys must be stable lowercase identifiers")
        path = f"frontend_profiles.{name}"
        profile_raw = _mapping(value, path)
        kind = _string(profile_raw.get("kind"), f"{path}.kind")
        if kind == FrontendKind.BUNDLED_NVA:
            profiles[name] = _parse_bundled_nva_frontend(profile_raw, path)
        elif kind == FrontendKind.OPENAI_REALTIME:
            profiles[name] = _parse_openai_realtime_frontend(profile_raw, path)
        else:
            raise ConfigurationError(f"{path}.kind must be bundled_nva or openai_realtime")
    if not profiles:
        raise ConfigurationError("frontend_profiles must define at least one profile")
    return profiles


def _parse_bundled_nva_frontend(raw: Mapping[str, Any], path: str) -> BundledNvaFrontendProfile:
    _reject_unknown(
        raw,
        {
            "kind",
            "public_model",
            "realtime_model",
            "platform",
            "pipeline_mode",
            "services",
            "connect_timeout_seconds",
            "bootstrap_timeout_seconds",
            "max_event_bytes",
        },
        path,
    )
    platform_value = _string(raw.get("platform"), f"{path}.platform")
    try:
        platform = NvaPlatform(platform_value)
    except ValueError as error:
        raise ConfigurationError(f"{path}.platform must be cloud, server, or singlegpu") from error
    pipeline_mode = _string(raw.get("pipeline_mode", _NVA_PIPELINE_MODE), f"{path}.pipeline_mode")
    if pipeline_mode != _NVA_PIPELINE_MODE:
        raise ConfigurationError(f"{path}.pipeline_mode currently supports only generic-assistant")
    services_path = f"{path}.services"
    services_raw = _mapping(raw.get("services"), services_path)
    _reject_unknown(services_raw, {"llm", "asr", "tts"}, services_path)
    services = CascadedServicesConfig(
        llm=_parse_llm_service(_mapping(services_raw.get("llm"), f"{services_path}.llm"), f"{services_path}.llm"),
        asr=_parse_asr_service(_mapping(services_raw.get("asr"), f"{services_path}.asr"), f"{services_path}.asr"),
        tts=_parse_tts_service(_mapping(services_raw.get("tts"), f"{services_path}.tts"), f"{services_path}.tts"),
    )
    shared_bundled_service_credential(services, path=services_path)
    validate_bundled_service_transports(services, path=services_path)
    connect_timeout, bootstrap_timeout, max_event_bytes = _realtime_bounds(raw, path)
    return BundledNvaFrontendProfile(
        kind=FrontendKind.BUNDLED_NVA,
        public_model=_bounded_string(raw.get("public_model", "nvidia/voiceclaw"), f"{path}.public_model", 512),
        realtime_model=_bounded_string(raw.get("realtime_model"), f"{path}.realtime_model", 512),
        platform=platform,
        pipeline_mode=pipeline_mode,
        services=services,
        connect_timeout_seconds=connect_timeout,
        bootstrap_timeout_seconds=bootstrap_timeout,
        max_event_bytes=max_event_bytes,
    )


def _parse_openai_realtime_frontend(raw: Mapping[str, Any], path: str) -> OpenAIRealtimeFrontendProfile:
    _reject_unknown(
        raw,
        {
            "kind",
            "endpoint",
            "model",
            "public_model",
            "credential",
            "connect_timeout_seconds",
            "bootstrap_timeout_seconds",
            "max_event_bytes",
        },
        path,
    )
    connect_timeout, bootstrap_timeout, max_event_bytes = _realtime_bounds(raw, path)
    return OpenAIRealtimeFrontendProfile(
        kind=FrontendKind.OPENAI_REALTIME,
        endpoint=_websocket_endpoint(raw.get("endpoint"), f"{path}.endpoint"),
        model=_bounded_string(raw.get("model"), f"{path}.model", 512),
        public_model=_bounded_string(raw.get("public_model", "nvidia/voiceclaw"), f"{path}.public_model", 512),
        credential=_parse_credential_reference(raw.get("credential"), f"{path}.credential"),
        connect_timeout_seconds=connect_timeout,
        bootstrap_timeout_seconds=bootstrap_timeout,
        max_event_bytes=max_event_bytes,
    )


def _parse_llm_service(raw: Mapping[str, Any], path: str) -> LlmServiceConfig:
    _reject_unknown(
        raw,
        {
            "id",
            "name",
            "provider",
            "endpoint",
            "model",
            "credential",
            "max_tokens",
            "temperature",
            "extra_params",
            "supports_tokenize",
            "realtime_max_output_tokens",
            "forced_tool_call_stops",
        },
        path,
    )
    provider_value = _string(raw.get("provider"), f"{path}.provider")
    if provider_value != ProviderKind.OPENAI_COMPATIBLE:
        raise ConfigurationError(f"{path}.provider currently supports only openai_compatible")
    provider = ProviderKind.OPENAI_COMPATIBLE
    max_tokens = _optional_bounded_positive_int(raw.get("max_tokens"), f"{path}.max_tokens", 1_000_000)
    realtime_max_output_tokens = _optional_bounded_positive_int(
        raw.get("realtime_max_output_tokens"),
        f"{path}.realtime_max_output_tokens",
        4096,
    )
    temperature = _optional_temperature(raw.get("temperature"), f"{path}.temperature")
    supports_tokenize = raw.get("supports_tokenize", False)
    if not isinstance(supports_tokenize, bool):
        raise ConfigurationError(f"{path}.supports_tokenize must be a boolean")
    if supports_tokenize and realtime_max_output_tokens is None:
        raise ConfigurationError(f"{path}.realtime_max_output_tokens is required when supports_tokenize is true")
    stops_raw = raw.get("forced_tool_call_stops", [])
    if not isinstance(stops_raw, list) or len(stops_raw) > 32:
        raise ConfigurationError(f"{path}.forced_tool_call_stops must be a list with at most 32 entries")
    stops: list[str] = []
    for index, stop in enumerate(stops_raw):
        parsed_stop = _string(stop, f"{path}.forced_tool_call_stops[{index}]")
        if len(parsed_stop) > 256:
            raise ConfigurationError(f"{path}.forced_tool_call_stops[{index}] must be at most 256 characters")
        stops.append(parsed_stop)
    extra_params = _mapping(raw.get("extra_params", {}), f"{path}.extra_params")
    _validate_provider_options(extra_params, f"{path}.extra_params")
    common = _parse_service_common(raw, path, allowed_provider=provider, endpoint_kind="http")
    return LlmServiceConfig(
        **common,
        credential=_parse_credential_reference(raw.get("credential"), f"{path}.credential"),
        max_tokens=max_tokens,
        temperature=temperature,
        extra_params=extra_params,
        supports_tokenize=supports_tokenize,
        realtime_max_output_tokens=realtime_max_output_tokens,
        forced_tool_call_stops=tuple(stops),
    )


def _parse_asr_service(raw: Mapping[str, Any], path: str) -> AsrServiceConfig:
    _reject_unknown(
        raw,
        {"id", "name", "provider", "endpoint", "model", "credential", "tls", "function_id", "language_code"},
        path,
    )
    provider_value = _string(raw.get("provider"), f"{path}.provider")
    if provider_value != ProviderKind.NVIDIA_GRPC:
        raise ConfigurationError(f"{path}.provider currently supports only nvidia_grpc")
    provider = ProviderKind.NVIDIA_GRPC
    common = _parse_service_common(raw, path, allowed_provider=provider, endpoint_kind="grpc")
    return AsrServiceConfig(
        **common,
        credential=_parse_credential_reference(raw.get("credential"), f"{path}.credential"),
        tls=_boolean(raw.get("tls", False), f"{path}.tls"),
        function_id=_optional_string(raw.get("function_id"), f"{path}.function_id"),
        language_code=_optional_string(raw.get("language_code"), f"{path}.language_code"),
    )


def _parse_tts_service(raw: Mapping[str, Any], path: str) -> TtsServiceConfig:
    _reject_unknown(
        raw,
        {
            "id",
            "name",
            "provider",
            "endpoint",
            "model",
            "credential",
            "tls",
            "function_id",
            "voice",
            "synthesis_mode",
            "language_code",
        },
        path,
    )
    provider_value = _string(raw.get("provider"), f"{path}.provider")
    if provider_value != ProviderKind.NVIDIA_GRPC:
        raise ConfigurationError(f"{path}.provider currently supports only nvidia_grpc")
    provider = ProviderKind.NVIDIA_GRPC
    synthesis_mode_value = _string(raw.get("synthesis_mode"), f"{path}.synthesis_mode")
    try:
        synthesis_mode = TtsSynthesisMode(synthesis_mode_value)
    except ValueError as error:
        raise ConfigurationError(f"{path}.synthesis_mode must be stitched or per_sentence") from error
    common = _parse_service_common(raw, path, allowed_provider=provider, endpoint_kind="grpc")
    return TtsServiceConfig(
        **common,
        credential=_parse_credential_reference(raw.get("credential"), f"{path}.credential"),
        tls=_boolean(raw.get("tls", False), f"{path}.tls"),
        function_id=_optional_string(raw.get("function_id"), f"{path}.function_id"),
        voice=_bounded_string(raw.get("voice"), f"{path}.voice", 256),
        synthesis_mode=synthesis_mode,
        language_code=_optional_string(raw.get("language_code"), f"{path}.language_code"),
    )


def _parse_service_common(
    raw: Mapping[str, Any],
    path: str,
    *,
    allowed_provider: ProviderKind,
    endpoint_kind: str,
) -> dict[str, Any]:
    service_id = _string(raw.get("id"), f"{path}.id")
    if _SERVICE_ID.fullmatch(service_id) is None:
        raise ConfigurationError(f"{path}.id must be a stable lowercase identifier")
    name = _bounded_string(raw.get("name"), f"{path}.name", 256)
    endpoint = (
        _http_endpoint(raw.get("endpoint"), f"{path}.endpoint")
        if endpoint_kind == "http"
        else _grpc_endpoint(raw.get("endpoint"), f"{path}.endpoint")
    )
    return {
        "id": service_id,
        "name": name,
        "provider": allowed_provider,
        "endpoint": endpoint,
        "model": _bounded_string(raw.get("model"), f"{path}.model", 512),
    }


def _parse_credential_reference(value: object, path: str) -> CredentialReference | None:
    if value is None:
        return None
    raw = _mapping(value, path)
    _reject_unknown(raw, {"env", "file"}, path)
    raw_env = raw.get("env")
    env = None if raw_env is None else _credential_environment_name(raw_env, f"{path}.env")
    file = _optional_string(raw.get("file"), f"{path}.file")
    if (env is None) == (file is None):
        raise ConfigurationError(f"{path} must select exactly one of env or file")
    if file is not None and not Path(file).is_absolute():
        raise ConfigurationError(f"{path}.file must be an absolute path")
    return CredentialReference(env=env, file=file)


def _frontend_realtime(profile: FrontendProfile) -> RealtimeConfig:
    if isinstance(profile, BundledNvaFrontendProfile):
        return RealtimeConfig(
            upstream_endpoint=_BUNDLED_REALTIME_ENDPOINT,
            upstream_model=profile.realtime_model,
            public_model=profile.public_model,
            credential_env=_BUNDLED_REALTIME_CREDENTIAL_ENV,
            connect_timeout_seconds=profile.connect_timeout_seconds,
            bootstrap_timeout_seconds=profile.bootstrap_timeout_seconds,
            max_event_bytes=profile.max_event_bytes,
        )
    credential = profile.credential
    return RealtimeConfig(
        upstream_endpoint=profile.endpoint,
        upstream_model=profile.model,
        public_model=profile.public_model,
        credential_env=credential.env if credential is not None else None,
        credential_file=credential.file if credential is not None else None,
        connect_timeout_seconds=profile.connect_timeout_seconds,
        bootstrap_timeout_seconds=profile.bootstrap_timeout_seconds,
        max_event_bytes=profile.max_event_bytes,
    )


def _realtime_bounds(raw: Mapping[str, Any], path: str) -> tuple[float, float, int]:
    connect_timeout = _positive_float(raw.get("connect_timeout_seconds", 15.0), f"{path}.connect_timeout_seconds")
    bootstrap_timeout = _positive_float(
        raw.get("bootstrap_timeout_seconds", 15.0),
        f"{path}.bootstrap_timeout_seconds",
    )
    max_event_bytes = _positive_int(raw.get("max_event_bytes", 16 * 1024 * 1024), f"{path}.max_event_bytes")
    if not 1024 <= max_event_bytes <= 16 * 1024 * 1024:
        raise ConfigurationError(f"{path}.max_event_bytes must be between 1024 and 16777216")
    return connect_timeout, bootstrap_timeout, max_event_bytes


def _optional_bounded_positive_int(value: object, path: str, maximum: int) -> int | None:
    if value is None:
        return None
    parsed = _positive_int(value, path)
    if parsed > maximum:
        raise ConfigurationError(f"{path} must be at most {maximum}")
    return parsed


def _optional_temperature(value: object, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigurationError(f"{path} must be a finite number between 0 and 2")
    parsed = float(value)
    if not 0 <= parsed <= 2:
        raise ConfigurationError(f"{path} must be between 0 and 2")
    return parsed


def _provider_option_name(value: object, path: str) -> str:
    value = _bounded_option_name(value, path)
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    parts = frozenset(part for part in normalized.split("_") if part)
    if (
        normalized in _PROVIDER_CONTROL_OPTION_NAMES
        or normalized.startswith("guided_")
        or "api_key" in normalized
        or parts.intersection(_PROVIDER_SECRET_OPTION_PARTS)
        or normalized.endswith(("_cert", "_certificate", "_private_key"))
    ):
        raise ConfigurationError(
            f"{path}.{value} is credential-, prompt-, transport-, or protocol-sensitive; "
            "use the typed service, credential, and tool fields"
        )
    return value


def _bounded_option_name(value: object, path: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_PROVIDER_OPTION_KEY_CHARACTERS:
        raise ConfigurationError(
            f"{path} keys must be non-empty strings of at most {_MAX_PROVIDER_OPTION_KEY_CHARACTERS} characters"
        )
    return value


def _backend_setting_name(value: object, path: str) -> str:
    value = _bounded_option_name(value, path)
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    parts = frozenset(part for part in normalized.split("_") if part)
    if (
        "api_key" in normalized
        or parts.intersection(_BACKEND_SECRET_SETTING_PARTS)
        or normalized.endswith(("_private_key",))
    ):
        raise ConfigurationError(
            f"{path}.{value} must use the backend credential reference instead of an inline setting"
        )
    return value


def _validate_bounded_json_object(
    value: object,
    path: str,
    *,
    key_validator: Callable[[object, str], str],
) -> None:
    """Validate one bounded JSON object with a caller-owned key policy."""
    item_count = 0

    def visit(item: object, item_path: str, depth: int) -> None:
        nonlocal item_count
        item_count += 1
        if item_count > _MAX_PROVIDER_OPTION_ITEMS:
            raise ConfigurationError(f"{path} must contain at most {_MAX_PROVIDER_OPTION_ITEMS} JSON values")
        if depth > _MAX_PROVIDER_OPTION_DEPTH:
            raise ConfigurationError(f"{path} must be at most {_MAX_PROVIDER_OPTION_DEPTH} levels deep")
        if item is None or isinstance(item, bool):
            return
        if isinstance(item, str):
            if len(item) > _MAX_PROVIDER_OPTION_STRING_CHARACTERS:
                raise ConfigurationError(
                    f"{item_path} must be at most {_MAX_PROVIDER_OPTION_STRING_CHARACTERS} characters"
                )
            return
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            if isinstance(item, float) and not math.isfinite(item):
                raise ConfigurationError(f"{item_path} must be a finite JSON number")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{item_path}[{index}]", depth + 1)
            return
        if isinstance(item, Mapping):
            for raw_key, child in item.items():
                key = key_validator(raw_key, item_path)
                visit(child, f"{item_path}.{key}", depth + 1)
            return
        raise ConfigurationError(f"{item_path} must contain only JSON-compatible values")

    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be an object")
    visit(value, path, 0)


def _validate_provider_options(value: object, path: str) -> None:
    """Validate a bounded provider-owned JSON tree without fixing model knobs."""
    _validate_bounded_json_object(value, path, key_validator=_provider_option_name)


def _validate_backend_settings(value: object, path: str) -> None:
    """Validate bounded plugin settings while keeping credentials out of YAML values."""
    _validate_bounded_json_object(value, path, key_validator=_backend_setting_name)


def _http_endpoint(value: object, path: str) -> str:
    endpoint = _bounded_string(value, path, 2048)
    parsed = urlsplit(endpoint)
    try:
        port = parsed.port
    except ValueError as error:
        raise ConfigurationError(f"{path} contains an invalid port") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or port == 0
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ConfigurationError(f"{path} must be an http/https URL without credentials, query, or fragment")
    return endpoint


def _grpc_endpoint(value: object, path: str) -> str:
    endpoint = _bounded_string(value, path, 2048)
    if any(character.isspace() for character in endpoint) or "://" in endpoint:
        raise ConfigurationError(f"{path} must be a host:port endpoint without a URL scheme")
    parsed = urlsplit(f"//{endpoint}")
    try:
        port = parsed.port
    except ValueError as error:
        raise ConfigurationError(f"{path} must be a host:port endpoint") from error
    if (
        not parsed.hostname
        or port is None
        or port == 0
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
    ):
        raise ConfigurationError(f"{path} must be a host:port endpoint")
    return endpoint


def _is_literal_loopback(hostname: str | None) -> bool:
    if hostname is None:
        return False
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def validate_bundled_service_transports(
    services: CascadedServicesConfig,
    *,
    path: str = "bundled_nva.services",
) -> None:
    """Require encryption for model traffic outside literal loopback."""
    for category, service in (("asr", services.asr), ("tts", services.tts)):
        if not isinstance(service.tls, bool):
            raise ConfigurationError(f"{path}.{category}.tls must be a boolean")

    llm_endpoint = _http_endpoint(services.llm.endpoint, f"{path}.llm.endpoint")
    llm_url = urlsplit(llm_endpoint)
    if llm_url.scheme != "https" and not _is_literal_loopback(llm_url.hostname):
        raise ConfigurationError(f"{path}.llm.endpoint requires https outside literal loopback")

    for category, service in (("asr", services.asr), ("tts", services.tts)):
        endpoint = _grpc_endpoint(service.endpoint, f"{path}.{category}.endpoint")
        hostname = urlsplit(f"//{endpoint}").hostname
        if not _is_literal_loopback(hostname) and not service.tls:
            raise ConfigurationError(f"{path}.{category}.tls must be true outside literal loopback")


def _websocket_endpoint(value: object, path: str) -> str:
    endpoint = _bounded_string(value, path, 2048)
    parsed = urlsplit(endpoint)
    try:
        port = parsed.port
    except ValueError as error:
        raise ConfigurationError(f"{path} contains an invalid port") from error
    if (
        parsed.scheme not in {"ws", "wss"}
        or not parsed.hostname
        or port == 0
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ConfigurationError(f"{path} must be a ws/wss URL without credentials, query, or fragment")
    if parsed.scheme == "ws":
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise ConfigurationError(f"{path} requires wss outside literal loopback")
    return endpoint


def _parse_realtime(value: object) -> RealtimeConfig | None:
    if value is None:
        return None
    raw = _mapping(value, "realtime")
    _reject_unknown(
        raw,
        {
            "upstream_endpoint",
            "upstream_model",
            "public_model",
            "credential",
            "connect_timeout_seconds",
            "bootstrap_timeout_seconds",
            "max_event_bytes",
        },
        "realtime",
    )
    endpoint = _websocket_endpoint(raw.get("upstream_endpoint"), "realtime.upstream_endpoint")
    credential_raw = _mapping(raw.get("credential", {}), "realtime.credential")
    _reject_unknown(credential_raw, {"env"}, "realtime.credential")
    raw_credential_env = credential_raw.get("env")
    credential_env = (
        None
        if raw_credential_env is None
        else _credential_environment_name(raw_credential_env, "realtime.credential.env")
    )
    max_event_bytes = _positive_int(raw.get("max_event_bytes", 16 * 1024 * 1024), "realtime.max_event_bytes")
    if not 1024 <= max_event_bytes <= 16 * 1024 * 1024:
        raise ConfigurationError("realtime.max_event_bytes must be between 1024 and 16777216")
    return RealtimeConfig(
        upstream_endpoint=endpoint,
        upstream_model=_string(raw.get("upstream_model"), "realtime.upstream_model"),
        public_model=_string(raw.get("public_model", "nvidia/voiceclaw"), "realtime.public_model"),
        credential_env=credential_env,
        connect_timeout_seconds=_positive_float(
            raw.get("connect_timeout_seconds", 15.0),
            "realtime.connect_timeout_seconds",
        ),
        bootstrap_timeout_seconds=_positive_float(
            raw.get("bootstrap_timeout_seconds", 15.0),
            "realtime.bootstrap_timeout_seconds",
        ),
        max_event_bytes=max_event_bytes,
    )


def _parse_backends(raw: Mapping[str, Any]) -> dict[str, BackendProfile]:
    profiles: dict[str, BackendProfile] = {}
    for name, value in raw.items():
        path = f"backend_profiles.{name}"
        profile_raw = _mapping(value, path)
        _reject_unknown(profile_raw, {"kind", "credential", "settings", "interaction"}, path)
        kind = _string(profile_raw.get("kind"), f"{path}.kind")
        credential_raw = _mapping(profile_raw.get("credential", {}), f"{path}.credential")
        _reject_unknown(credential_raw, {"env", "file"}, f"{path}.credential")
        raw_credential_env = credential_raw.get("env")
        credential_env = (
            None
            if raw_credential_env is None
            else _credential_environment_name(raw_credential_env, f"{path}.credential.env")
        )
        credential_file = _optional_string(credential_raw.get("file"), f"{path}.credential.file")
        if credential_env is not None and credential_file is not None:
            raise ConfigurationError(f"{path}.credential must select exactly one of env or file")
        if credential_file is not None and not Path(credential_file).is_absolute():
            raise ConfigurationError(f"{path}.credential.file must be an absolute path")
        interaction_raw = _mapping(profile_raw.get("interaction", {}), f"{path}.interaction")
        _reject_unknown(interaction_raw, {"profile", "tool_copy"}, f"{path}.interaction")
        interaction_profile = _string(
            interaction_raw.get("profile", "stateless"),
            f"{path}.interaction.profile",
        )
        tool_copy = _parse_tool_copy(
            _mapping(interaction_raw.get("tool_copy", {}), f"{path}.interaction.tool_copy"),
            f"{path}.interaction.tool_copy",
        )
        settings = _mapping(profile_raw.get("settings", {}), f"{path}.settings")
        _validate_backend_settings(settings, f"{path}.settings")
        if kind in {"none", "disabled"}:
            if credential_env is not None or credential_file is not None:
                raise ConfigurationError(f"{path}.credential must be omitted when {path}.kind is {kind}")
            if settings:
                raise ConfigurationError(f"{path}.settings must be empty when {path}.kind is {kind}")
            if tool_copy:
                raise ConfigurationError(f"{path}.interaction.tool_copy must be empty when {path}.kind is {kind}")
        profiles[_string(name, "backend_profiles key")] = BackendProfile(
            kind=kind,
            credential_env=credential_env,
            credential_file=credential_file,
            settings=settings,
            interaction_profile=interaction_profile,
            tool_copy=tool_copy,
        )
    if not profiles:
        raise ConfigurationError("backend_profiles must define at least one profile")
    return profiles


def _parse_tool_copy(raw: Mapping[str, Any], path: str) -> dict[str, ToolCopyOverride]:
    overrides: dict[str, ToolCopyOverride] = {}
    for logical_name, raw_override in raw.items():
        tool_path = f"{path}.{logical_name}"
        if not isinstance(logical_name, str) or _SEMANTIC_NAME.fullmatch(logical_name) is None:
            raise ConfigurationError(f"{path} keys must be canonical dotted tool names")
        override = _mapping(raw_override, tool_path)
        _reject_unknown(override, {"description", "properties"}, tool_path)
        description = _optional_model_copy(override.get("description"), f"{tool_path}.description")
        properties_raw = _mapping(override.get("properties", {}), f"{tool_path}.properties")
        properties: dict[str, str] = {}
        for property_name, property_description in properties_raw.items():
            property_path = f"{tool_path}.properties.{property_name}"
            if not isinstance(property_name, str) or _PROPERTY_NAME.fullmatch(property_name) is None:
                raise ConfigurationError(f"{tool_path}.properties keys must be canonical property names")
            properties[property_name] = _model_copy(property_description, property_path)
        if description is None and not properties:
            raise ConfigurationError(f"{tool_path} must override description or properties")
        overrides[logical_name] = ToolCopyOverride(description=description, properties=properties)
    return overrides


def _optional_model_copy(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _model_copy(value, path)


def _model_copy(value: object, path: str) -> str:
    text = _string(value, path)
    if len(text) > _MAX_MODEL_COPY_CHARACTERS or any(
        ord(character) < 32 and character not in "\n\t" for character in text
    ):
        raise ConfigurationError(f"{path} must be bounded printable text")
    return text


def _environment_reference_allowed(path: tuple[str, ...]) -> bool:
    if path in {
        ("server", "api_key_file"),
        ("realtime", "upstream_endpoint"),
        ("model_contracts", "path"),
        ("interaction_profiles", "path"),
        ("state", "path"),
    }:
        return True
    if len(path) == 3 and path[0] == "frontend_profiles" and path[2] == "endpoint":
        return True
    if len(path) == 5 and path[0] == "frontend_profiles" and path[2] == "services" and path[4] == "endpoint":
        return True
    if (
        len(path) == 4
        and path[0] in {"frontend_profiles", "backend_profiles"}
        and path[2:]
        == (
            "credential",
            "file",
        )
    ):
        return True
    if (
        len(path) == 6
        and path[0] == "frontend_profiles"
        and path[2] == "services"
        and path[4:] == ("credential", "file")
    ):
        return True
    # Endpoint interpolation belongs to the provider-neutral backend boundary,
    # not to any one built-in adapter.  Plugin factories still own semantic
    # validation of the expanded value; credentials remain confined to the
    # typed credential reference above and cannot be interpolated into settings.
    return len(path) == 4 and path[0] == "backend_profiles" and path[2:] == ("settings", "endpoint")


def _configuration_path(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "config"


def configuration_environment_references(value: object) -> frozenset[str]:
    """Return environment names used only by approved URL and path fields."""
    references: set[str] = set()

    def visit(item: object, path: tuple[str, ...]) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, (*path, str(key)))
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, (*path, f"[{index}]"))
            return
        if not isinstance(item, str):
            return
        names = set(_ENV_REFERENCE.findall(item))
        if not names:
            return
        if not _environment_reference_allowed(path):
            raise ConfigurationError(
                f"environment references are not allowed in {_configuration_path(path)}; "
                "use credential.env for secrets and literal trusted text for this field"
            )
        references.update(names)

    visit(value, ())
    return frozenset(references)


def _substitute_environment(value: Any, environ: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {key: _substitute_environment(item, environ) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute_environment(item, environ) for item in value]
    if not isinstance(value, str):
        return value
    return _ENV_REFERENCE.sub(lambda match: environ[match.group(1)], value)


def _expand_environment(value: Any, environ: Mapping[str, str]) -> Any:
    references = configuration_environment_references(value)
    missing = sorted(name for name in references if name not in environ)
    if missing:
        raise ConfigurationError(f"missing required environment variables: {', '.join(missing)}")
    return _substitute_environment(value, environ)
