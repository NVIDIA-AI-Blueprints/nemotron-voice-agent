# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Compile a selected frontend profile into an isolated NVA runtime plan.

The public configuration model is owned by :mod:`voiceclaw.config`.  This
module is deliberately the one-way adapter from that provider-neutral model
to the file formats and environment names understood by the bundled NVA
Realtime process.  External OpenAI Realtime-compatible frontends do not
materialize NVA files and do not launch the bundled child.
"""

from __future__ import annotations

import errno
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import yaml

from voiceclaw.config import (
    ConfigurationError,
    shared_bundled_service_credential,
    validate_bundled_service_transports,
)
from voiceclaw.model_contracts import ModelContractCatalog, load_model_contract_catalog

if TYPE_CHECKING:
    from voiceclaw.config import (
        BundledNvaFrontendProfile,
        CredentialReference,
        OpenAIRealtimeFrontendProfile,
    )

_SUPPORTED_PIPELINE = "generic-assistant"
_SUPPORTED_LLM_PROVIDER = "openai_compatible"
_SUPPORTED_SPEECH_PROVIDER = "nvidia_grpc"
_SUPPORTED_PLATFORMS = frozenset({"cloud", "server", "singlegpu"})
_SERVICE_ID = re.compile(r"[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*")
_REALTIME_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}")
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MAX_SECRET_BYTES = 16 * 1024
_VOICECLAW_PROMPT_KEY = "voiceclaw_frontend"


@dataclass(frozen=True, slots=True)
class FrontendRuntimePlan:
    """Resolved startup plan for one selected frontend profile."""

    kind: str
    launch_bundled_nva: bool
    public_model: str
    upstream_endpoint: str
    upstream_model: str
    nva_environment: Mapping[str, str] = field(default_factory=dict)
    nva_credential: CredentialReference | None = None
    registry_path: Path | None = None
    services_cloud_path: Path | None = None
    services_local_path: Path | None = None
    prompt_catalog_path: Path | None = None
    prompt_key: str | None = None
    model_contract_digest: str | None = None

    def __post_init__(self) -> None:
        """Detach runtime environment data from mutable caller mappings."""
        object.__setattr__(self, "nva_environment", MappingProxyType(dict(self.nva_environment)))


def materialize_frontend_runtime(
    profile: BundledNvaFrontendProfile | OpenAIRealtimeFrontendProfile,
    destination: str | Path,
    *,
    internal_endpoint: str,
    model_contracts: ModelContractCatalog | None = None,
) -> FrontendRuntimePlan:
    """Compile ``profile`` and return the exact child-process startup plan.

    ``destination`` is touched only for ``kind: bundled_nva``.  Catalogs never
    contain credential values; they refer only to server-owned endpoints and
    model identifiers.  The returned credential reference can subsequently be
    resolved directly into the child environment with
    :func:`bind_nva_credential`.
    """
    kind = str(profile.kind)
    if kind == "openai_realtime":
        return _external_realtime_plan(profile)
    if kind != "bundled_nva":
        raise ConfigurationError(f"unsupported frontend kind: {kind}")
    return _materialize_bundled_nva(
        profile,
        Path(destination),
        internal_endpoint=internal_endpoint,
        model_contracts=model_contracts or load_model_contract_catalog(),
    )


def bind_nva_credential(
    environment: Mapping[str, str],
    credential: CredentialReference | None,
    *,
    source_environment: Mapping[str, str],
) -> dict[str, str]:
    """Bind the selected model credential to NVA's single credential input.

    NVA consumes one credential for all three cascaded services. VoiceClaw
    resolves either trusted reference only at child launch and exposes the
    value solely as the private NVA child's existing ``NVIDIA_API_KEY`` input.
    A missing reference clears both supported names.
    """
    prepared = dict(environment)
    prepared.pop("NVIDIA_API_KEY", None)
    prepared.pop("NVIDIA_API_KEY_FILE", None)
    if credential is None:
        return prepared
    value = resolve_credential_value(
        credential,
        source_environment=source_environment,
        label="frontend service credential",
    )
    assert value is not None
    prepared["NVIDIA_API_KEY"] = value
    return prepared


def resolve_credential_value(
    credential: CredentialReference | None,
    *,
    source_environment: Mapping[str, str],
    label: str = "frontend credential",
) -> str | None:
    """Resolve one trusted env/file reference without persisting its value."""
    if credential is None:
        return None

    env_name = getattr(credential, "env", None)
    file_name = getattr(credential, "file", None)
    if env_name is not None:
        if not isinstance(env_name, str) or _ENVIRONMENT_NAME.fullmatch(env_name) is None:
            raise ConfigurationError(f"{label}.env must be an environment-variable name")
        value = source_environment.get(env_name, "")
        if not value:
            raise ConfigurationError(f"{label} environment is not set: {env_name}")
    elif file_name is not None:
        value = _read_secret(Path(file_name), label=label)
    else:
        raise ConfigurationError(f"{label} must select env or file")

    return _normalize_secret(value, label=label)


def _external_realtime_plan(profile: OpenAIRealtimeFrontendProfile) -> FrontendRuntimePlan:
    endpoint = str(profile.endpoint)
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
        raise ConfigurationError("openai_realtime frontend endpoint must be a ws/wss URL")
    return FrontendRuntimePlan(
        kind="openai_realtime",
        launch_bundled_nva=False,
        public_model=str(profile.public_model),
        upstream_endpoint=endpoint,
        upstream_model=str(profile.model),
    )


def _materialize_bundled_nva(
    profile: BundledNvaFrontendProfile,
    destination: Path,
    *,
    internal_endpoint: str,
    model_contracts: ModelContractCatalog,
) -> FrontendRuntimePlan:
    if str(profile.pipeline_mode) != _SUPPORTED_PIPELINE:
        raise ConfigurationError(
            f"bundled_nva currently supports pipeline_mode {_SUPPORTED_PIPELINE!r}; got {profile.pipeline_mode!r}"
        )
    platform = str(profile.platform)
    if platform not in _SUPPORTED_PLATFORMS:
        raise ConfigurationError(f"bundled_nva platform must be one of: {', '.join(sorted(_SUPPORTED_PLATFORMS))}")
    if _REALTIME_MODEL_ID.fullmatch(str(profile.realtime_model)) is None:
        raise ConfigurationError("bundled_nva realtime_model is not a valid Realtime model identifier")
    _validate_internal_endpoint(internal_endpoint)

    llm = profile.services.llm
    asr = profile.services.asr
    tts = profile.services.tts
    _require_provider(llm.provider, _SUPPORTED_LLM_PROVIDER, "llm")
    _require_provider(asr.provider, _SUPPORTED_SPEECH_PROVIDER, "asr")
    _require_provider(tts.provider, _SUPPORTED_SPEECH_PROVIDER, "tts")
    credential = shared_bundled_service_credential(profile.services)
    if platform == "cloud" and credential is None:
        raise ConfigurationError("bundled_nva platform cloud requires one shared service credential")
    validate_bundled_service_transports(profile.services)

    service_entries = {
        "llm": {_service_id(llm.id, "llm"): _llm_entry(llm)},
        "asr": {_service_id(asr.id, "asr"): _asr_entry(asr)},
        "tts": {_service_id(tts.id, "tts"): _tts_entry(tts)},
    }
    registry = _registry_document(profile)
    prompt_catalog = {
        _VOICECLAW_PROMPT_KEY: {
            "description": "VoiceClaw frontend policy generated from the selected model contract.",
            "content": model_contracts.static_instructions,
        }
    }
    # NVA validates a registered public model against every supported catalog
    # platform at process startup.  Duplicate this one operator-selected route
    # into each non-secret catalog while REALTIME_SERVICE_PLATFORM below pins
    # the route that can actually be selected for sessions.
    cloud_catalog: dict[str, object] = deepcopy(service_entries)
    local_catalog: dict[str, object] = {
        "server": deepcopy(service_entries),
        "singlegpu": deepcopy(service_entries),
    }

    _prepare_destination(destination)
    registry_path = destination / "examples_registry.yaml"
    services_cloud_path = destination / "services.cloud.yaml"
    services_local_path = destination / "services.local.yaml"
    prompt_catalog_path = destination / "prompts.yaml"
    _write_yaml(registry_path, registry)
    _write_yaml(services_cloud_path, cloud_catalog)
    _write_yaml(services_local_path, local_catalog)
    _write_yaml(prompt_catalog_path, prompt_catalog)

    child_environment = {
        "NVA_RUNTIME_CONFIG_DIR": str(destination),
        "REALTIME_SERVICE_PLATFORM": platform,
        "EXAMPLE_SELECTION": _SUPPORTED_PIPELINE,
        "PROMPT_FILE_PATH": str(prompt_catalog_path),
        "TRANSPORT_SELECTION": "websocket",
    }
    return FrontendRuntimePlan(
        kind="bundled_nva",
        launch_bundled_nva=True,
        public_model=str(profile.public_model),
        upstream_endpoint=internal_endpoint,
        upstream_model=str(profile.realtime_model),
        nva_environment=child_environment,
        nva_credential=credential,
        registry_path=registry_path,
        services_cloud_path=services_cloud_path,
        services_local_path=services_local_path,
        prompt_catalog_path=prompt_catalog_path,
        prompt_key=_VOICECLAW_PROMPT_KEY,
        model_contract_digest=model_contracts.digest,
    )


def _registry_document(profile: BundledNvaFrontendProfile) -> dict[str, object]:
    llm_id = _service_id(profile.services.llm.id, "llm")
    asr_id = _service_id(profile.services.asr.id, "asr")
    tts_id = _service_id(profile.services.tts.id, "tts")
    prompt_key = _VOICECLAW_PROMPT_KEY
    selectors = {
        "prompt_key": prompt_key,
        "llm_id": llm_id,
        "asr_id": asr_id,
        "tts_id": tts_id,
    }
    return {
        "selection": _SUPPORTED_PIPELINE,
        "transports": "websocket",
        "realtime_models": {
            str(profile.realtime_model): {
                "label": str(profile.public_model),
                "pipeline_mode": _SUPPORTED_PIPELINE,
                "default": True,
                "selectors": selectors,
            }
        },
        "examples": {
            _SUPPORTED_PIPELINE: {
                "label": str(profile.public_model),
                "slots": ["llm", "asr", "tts"],
                "defaults": {
                    "prompt": [prompt_key],
                    "llm": [llm_id],
                    "asr": [asr_id],
                    "tts": [tts_id],
                },
                "welcome_message": False,
                "bot": "examples.generic.pipeline:bot",
            }
        },
    }


def _llm_entry(service: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "name": str(service.name),
        "model_id": str(service.model),
        "base_url": _http_endpoint(service.endpoint, "llm"),
        # VoiceClaw's selected model-contract catalog is the sole behavior
        # authority.  A provider-level system prompt would create a second,
        # independently mutable policy layer in the generic NVA pipeline.
        "system_prompt": "",
    }
    if service.max_tokens is not None:
        entry["max_tokens"] = int(service.max_tokens)
    if service.temperature is not None:
        entry["temperature"] = float(service.temperature)
    entry["supports_tokenize"] = bool(service.supports_tokenize)
    if service.realtime_max_output_tokens is not None:
        entry["realtime_max_output_tokens"] = int(service.realtime_max_output_tokens)
    if service.forced_tool_call_stops:
        entry["forced_tool_call_stops"] = list(service.forced_tool_call_stops)
    if service.extra_params:
        entry["extra_params"] = json.dumps(
            _plain_json_value(service.extra_params),
            separators=(",", ":"),
            sort_keys=True,
        )
    return entry


def _plain_json_value(value: object) -> object:
    """Copy a recursively frozen provider option tree into JSON containers."""
    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json_value(item) for item in value]
    return value


def _asr_entry(service: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "name": str(service.name),
        "server": _grpc_endpoint(service.endpoint, "asr"),
        "model": str(service.model),
        "use_ssl": bool(service.tls),
    }
    if service.function_id is not None:
        entry["function_id"] = str(service.function_id)
    if service.language_code is not None:
        entry["language_code"] = str(service.language_code)
    return entry


def _tts_entry(service: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "name": str(service.name),
        "server": _grpc_endpoint(service.endpoint, "tts"),
        "model": str(service.model),
        "voice_id": str(service.voice),
        "synthesis_mode": str(service.synthesis_mode),
        "use_ssl": bool(service.tls),
    }
    if service.function_id is not None:
        entry["function_id"] = str(service.function_id)
    if service.language_code is not None:
        entry["language_code"] = str(service.language_code)
    return entry


def _require_provider(actual: object, expected: str, category: str) -> None:
    if str(actual) != expected:
        raise ConfigurationError(f"bundled_nva {category}.provider must be {expected}")


def _service_id(value: object, category: str) -> str:
    identifier = str(value)
    if _SERVICE_ID.fullmatch(identifier) is None:
        raise ConfigurationError(f"bundled_nva {category}.id is not a valid catalog identifier")
    return identifier


def _http_endpoint(value: object, category: str) -> str:
    endpoint = str(value)
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ConfigurationError(f"bundled_nva {category}.endpoint must be an http/https URL without credentials")
    return endpoint.rstrip("/")


def _grpc_endpoint(value: object, category: str) -> str:
    endpoint = str(value).strip()
    parsed = urlsplit(f"//{endpoint}")
    try:
        port = parsed.port
    except ValueError:
        port = None
    if (
        not endpoint
        or any(character.isspace() for character in endpoint)
        or parsed.hostname is None
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ConfigurationError(f"bundled_nva {category}.endpoint must be host[:port] without credentials")
    return endpoint


def _validate_internal_endpoint(endpoint: str) -> None:
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "ws"
        or parsed.hostname not in {"127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ConfigurationError("bundled NVA internal endpoint must be a literal loopback ws URL")


def _prepare_destination(destination: Path) -> None:
    if destination.exists() and (destination.is_symlink() or not destination.is_dir()):
        raise ConfigurationError("frontend runtime destination must be a directory, not a symlink")
    try:
        destination.mkdir(mode=0o755, parents=True, exist_ok=True)
    except OSError as error:
        raise ConfigurationError(f"could not create frontend runtime directory: {destination}") from error


def _write_yaml(path: Path, document: Mapping[str, object]) -> None:
    payload = yaml.safe_dump(dict(document), sort_keys=False, allow_unicode=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
    except OSError as error:
        if temporary_name is not None:
            with suppress(OSError):
                Path(temporary_name).unlink()
        raise ConfigurationError(f"could not materialize frontend runtime file: {path.name}") from error


def _read_secret(path: Path, *, label: str) -> str:
    if not path.is_absolute():
        raise ConfigurationError(f"{label}.file must be an absolute path")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ConfigurationError(f"{label} file could not be opened") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ConfigurationError(f"{label} file must be a regular file")
        if metadata.st_mode & stat.S_IWGRP or metadata.st_mode & stat.S_IRWXO:
            raise ConfigurationError(f"{label} file must not be group-writable or accessible by others")
        getxattr = getattr(os, "getxattr", None)
        if getxattr is not None:
            try:
                access_acl = getxattr(descriptor, "system.posix_acl_access")
            except OSError as error:
                no_acl_errors = {
                    errno.ENODATA,
                    errno.ENOTSUP,
                    getattr(errno, "ENOATTR", errno.ENODATA),
                }
                if error.errno not in no_acl_errors:
                    raise ConfigurationError(f"{label} file POSIX ACL could not be inspected") from error
            else:
                if access_acl:
                    raise ConfigurationError(f"{label} file must not have a POSIX access ACL")
        raw = os.read(descriptor, _MAX_SECRET_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(raw) > _MAX_SECRET_BYTES:
        raise ConfigurationError(f"{label} file exceeds the supported size")
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ConfigurationError(f"{label} file must contain UTF-8 text") from error
    return value


def _normalize_secret(value: str, *, label: str) -> str:
    if value.endswith("\r\n"):
        value = value[:-2]
    elif value.endswith("\n"):
        value = value[:-1]
    if not value or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value):
        raise ConfigurationError(f"{label} must be non-empty text without whitespace or controls")
    if len(value.encode("utf-8")) > _MAX_SECRET_BYTES:
        raise ConfigurationError(f"{label} exceeds the supported size")
    return value
