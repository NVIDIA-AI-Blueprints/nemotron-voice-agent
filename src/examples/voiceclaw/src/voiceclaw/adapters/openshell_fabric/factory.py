# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Composition factory for a deployed Fabric agent reached through OpenShell."""

from __future__ import annotations

import ipaddress
import math
import re
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit

from voiceclaw.adapters.openshell_fabric.client import (
    OpenShellAuthenticationMode,
    OpenShellClientError,
    SdkSandboxExecutor,
)
from voiceclaw.adapters.openshell_fabric.committed_turn import (
    DEFAULT_CHECK_TIMEOUT_SECONDS,
    DEFAULT_INVOCATION_TIMEOUT_SECONDS,
    OpenShellFabricAdapter,
    validate_fabric_binding_identity,
)
from voiceclaw.adapters.openshell_fabric.protocol import FabricProtocolError, adapter_codec, parse_json_pointer
from voiceclaw.adapters.openshell_fabric.security import client_secret_loader
from voiceclaw.adapters.result_envelope import (
    DEFAULT_RESULT_SPEECH_BUDGET_BYTES,
    MAX_RESULT_DISPLAY_BYTES,
    MAX_RESULT_SPEECH_BYTES,
)
from voiceclaw.backends import BackendComposition
from voiceclaw.config import BackendProfile, ConfigurationError
from voiceclaw.model_contracts import ModelContractCatalog, load_model_contract_catalog

_ALLOWED_SETTINGS = frozenset(
    {
        "endpoint",
        "authentication",
        "workspace",
        "sandbox",
        "fabric_agent",
        "adapter_id",
        "native_agent",
        "label",
        "issuer",
        "client_id",
        "scopes",
        "audience",
        "tls",
        "tls_ca_file",
        "rpc_timeout_seconds",
        "invoke_timeout_seconds",
        "check_timeout_seconds",
        "result_pointer",
        "result_display_budget_bytes",
        "result_speech_budget_bytes",
    }
)
_OAUTH_SCOPE = re.compile(r"^[\x21\x23-\x5B\x5D-\x7E]{1,256}$")
_ENDPOINT_FORBIDDEN_CHARACTERS = frozenset("$%{}\\")


def build_openshell_fabric_backend(
    profile: BackendProfile,
    environ: Mapping[str, str],
    model_contracts: ModelContractCatalog | None = None,
) -> BackendComposition:
    """Validate one service binding and construct its response-only adapter."""
    unknown = sorted(set(profile.settings) - _ALLOWED_SETTINGS)
    if unknown:
        raise ConfigurationError(f"unknown openshell_fabric settings: {', '.join(unknown)}")
    endpoint, tls = _endpoint(
        _required(profile, "endpoint"),
        configured_tls=_optional_boolean(profile, "tls"),
    )
    if not tls and not _endpoint_is_loopback(endpoint):
        raise ConfigurationError("openshell_fabric permits plaintext only for a loopback endpoint")
    tls_ca_file = _tls_ca_file(profile, tls=tls)
    authentication = _authentication(profile)
    if authentication is OpenShellAuthenticationMode.ANONYMOUS:
        if tls or not _endpoint_is_loopback(endpoint):
            raise ConfigurationError("openshell_fabric anonymous authentication requires a plaintext loopback endpoint")
        if profile.credential_env is not None or profile.credential_file is not None:
            raise ConfigurationError("openshell_fabric anonymous authentication cannot include a credential")
        oauth_settings = sorted(set(profile.settings) & {"issuer", "client_id", "scopes", "audience"})
        if oauth_settings:
            raise ConfigurationError(
                "openshell_fabric anonymous authentication cannot include OAuth settings: " + ", ".join(oauth_settings)
            )
        secret_loader = None
        issuer = None
        client_id = None
        scopes: tuple[str, ...] = ()
        audience = None
    else:
        secret_loader = client_secret_loader(profile, environ)
        issuer = _issuer(_required(profile, "issuer"))
        client_id = _required(profile, "client_id")
        scopes = _scopes(profile.settings.get("scopes", ()))
        audience = _optional(profile, "audience")
    result_pointer = _optional(profile, "result_pointer")
    if result_pointer is not None:
        try:
            parse_json_pointer(result_pointer)
        except ValueError as error:
            raise ConfigurationError("openshell_fabric result_pointer is invalid") from error
    adapter_id = _required(profile, "adapter_id")
    try:
        adapter_codec(adapter_id)
    except FabricProtocolError as error:
        raise ConfigurationError("openshell_fabric adapter_id has no installed codec") from error
    try:
        workspace = validate_fabric_binding_identity(_required(profile, "workspace"), name="workspace")
        sandbox = validate_fabric_binding_identity(_required(profile, "sandbox"), name="sandbox")
        fabric_agent = validate_fabric_binding_identity(_required(profile, "fabric_agent"), name="fabric_agent")
    except ValueError as error:
        raise ConfigurationError(str(error)) from error
    native_agent = _optional(profile, "native_agent")
    try:
        if native_agent is not None:
            native_agent = validate_fabric_binding_identity(native_agent, name="native_agent")
        label = validate_fabric_binding_identity(
            _optional(profile, "label") or "OpenShell / Fabric",
            name="label",
            maximum_characters=128,
        )
    except ValueError as error:
        raise ConfigurationError(str(error)) from error
    rpc_timeout_seconds = _positive_number(profile, "rpc_timeout_seconds", 30.0, maximum=30.0)
    invoke_timeout_seconds = _bounded_integer(
        profile,
        "invoke_timeout_seconds",
        DEFAULT_INVOCATION_TIMEOUT_SECONDS,
        minimum=1,
        maximum=3600,
    )
    check_timeout_seconds = _bounded_integer(
        profile,
        "check_timeout_seconds",
        DEFAULT_CHECK_TIMEOUT_SECONDS,
        minimum=1,
        maximum=120,
    )
    result_display_budget_bytes = _bounded_integer(
        profile,
        "result_display_budget_bytes",
        MAX_RESULT_DISPLAY_BYTES,
        minimum=1,
        maximum=MAX_RESULT_DISPLAY_BYTES,
    )
    result_speech_budget_bytes = _bounded_integer(
        profile,
        "result_speech_budget_bytes",
        DEFAULT_RESULT_SPEECH_BUDGET_BYTES,
        minimum=1,
        maximum=MAX_RESULT_SPEECH_BYTES,
    )
    contracts = model_contracts or load_model_contract_catalog()
    try:
        executor = SdkSandboxExecutor(
            endpoint=endpoint,
            workspace=workspace,
            sandbox=sandbox,
            authentication=authentication,
            client_secret=secret_loader,
            issuer=issuer,
            client_id=client_id,
            scopes=scopes,
            audience=audience,
            tls=tls,
            tls_ca_file=tls_ca_file,
            rpc_timeout_seconds=rpc_timeout_seconds,
        )
    except OpenShellClientError as error:
        raise ConfigurationError("OpenShell client could not be initialized") from error
    try:
        adapter = OpenShellFabricAdapter(
            executor=executor,
            workspace=workspace,
            sandbox=sandbox,
            fabric_agent=fabric_agent,
            adapter_id=adapter_id,
            native_agent=native_agent,
            label=label,
            invoke_timeout_seconds=invoke_timeout_seconds,
            check_timeout_seconds=check_timeout_seconds,
            result_pointer=result_pointer,
            result_display_budget_bytes=result_display_budget_bytes,
            result_speech_budget_bytes=result_speech_budget_bytes,
            model_contracts=contracts,
        )
        return BackendComposition(
            turn_backend=adapter,
            turn_status="response_only",
            selected_agent_readiness=adapter,
        )
    except (TypeError, ValueError) as error:
        with suppress(OpenShellClientError):
            executor.close()
        raise ConfigurationError("openshell_fabric adapter settings are invalid") from error
    except BaseException:
        with suppress(OpenShellClientError):
            executor.close()
        raise


def _required(profile: BackendProfile, name: str) -> str:
    value = profile.settings.get(name)
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ConfigurationError(f"openshell_fabric requires a non-empty {name!r} setting")
    return value.strip()


def _authentication(profile: BackendProfile) -> OpenShellAuthenticationMode:
    value = profile.settings.get("authentication", OpenShellAuthenticationMode.CLIENT_CREDENTIALS)
    if not isinstance(value, str):
        raise ConfigurationError("openshell_fabric authentication must be a string")
    try:
        return OpenShellAuthenticationMode(value)
    except ValueError as error:
        choices = ", ".join(mode.value for mode in OpenShellAuthenticationMode)
        raise ConfigurationError(f"openshell_fabric authentication must be one of: {choices}") from error


def _optional(profile: BackendProfile, name: str) -> str | None:
    value = profile.settings.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ConfigurationError(f"openshell_fabric setting {name!r} must be a non-empty string")
    return value.strip()


def _optional_boolean(profile: BackendProfile, name: str) -> bool | None:
    if name not in profile.settings:
        return None
    value = profile.settings[name]
    if not isinstance(value, bool):
        raise ConfigurationError(f"openshell_fabric setting {name!r} must be a boolean")
    return value


def _positive_number(profile: BackendProfile, name: str, default: float, *, maximum: float) -> float:
    value = profile.settings.get(name, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= maximum
    ):
        raise ConfigurationError(f"openshell_fabric setting {name!r} must be in (0, {maximum}]")
    return float(value)


def _bounded_integer(
    profile: BackendProfile,
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = profile.settings.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ConfigurationError(
            f"openshell_fabric setting {name!r} must be an integer from {minimum} through {maximum}"
        )
    return value


def _scopes(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ConfigurationError("openshell_fabric scopes must be a list of strings")
    scopes = tuple(value)
    if len(scopes) > 32 or any(not isinstance(scope, str) or _OAUTH_SCOPE.fullmatch(scope) is None for scope in scopes):
        raise ConfigurationError("openshell_fabric scopes contain an invalid OAuth scope")
    if len(scopes) != len(set(scopes)):
        raise ConfigurationError("openshell_fabric scopes must be unique")
    return scopes


def _endpoint(value: str, *, configured_tls: bool | None) -> tuple[str, bool]:
    if any(
        character.isspace()
        or ord(character) < 0x20
        or ord(character) == 0x7F
        or character in _ENDPOINT_FORBIDDEN_CHARACTERS
        for character in value
    ):
        raise ConfigurationError("openshell_fabric endpoint is invalid")
    has_scheme = "://" in value
    try:
        parsed = urlsplit(value if has_scheme else f"//{value}")
    except ValueError as error:
        raise ConfigurationError("openshell_fabric endpoint is invalid") from error
    if has_scheme:
        if parsed.scheme not in {"http", "https"}:
            raise ConfigurationError("openshell_fabric endpoint must use http or https")
        tls = parsed.scheme == "https"
        if configured_tls is not None and configured_tls != tls:
            raise ConfigurationError("openshell_fabric tls setting conflicts with endpoint scheme")
        if parsed.path not in {"", "/"}:
            raise ConfigurationError("openshell_fabric endpoint URL must not contain a path")
    else:
        tls = True if configured_tls is None else configured_tls
    try:
        port = parsed.port
        host = parsed.hostname
    except ValueError as error:
        raise ConfigurationError("openshell_fabric endpoint is invalid") from error
    if (
        not host
        or port == 0
        or (has_scheme and parsed.netloc.endswith(":"))
        or parsed.username is not None
        or parsed.password is not None
        or (not has_scheme and parsed.path)
        or "?" in value
        or "#" in value
        or parsed.query
        or parsed.fragment
    ):
        raise ConfigurationError("openshell_fabric endpoint is invalid")
    if not has_scheme and port is None:
        raise ConfigurationError("openshell_fabric bare endpoint must be host:port")
    if port is None:
        port = 443 if tls else 80
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if not re.fullmatch(r"[A-Za-z0-9.-]{1,253}", host) or any(
            not label or len(label) > 63 or label.startswith("-") or label.endswith("-")
            for label in host.rstrip(".").split(".")
        ):
            raise ConfigurationError("openshell_fabric endpoint host is invalid") from None
        rendered_host = host.lower()
    else:
        rendered_host = f"[{address.compressed}]" if address.version == 6 else address.compressed
    return f"{rendered_host}:{port}", tls


def _endpoint_is_loopback(endpoint: str) -> bool:
    host = urlsplit(f"//{endpoint}").hostname
    return _host_is_loopback(host)


def _host_is_loopback(host: str | None) -> bool:
    if host is None:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _issuer(value: str) -> str:
    parsed = urlsplit(value)
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ConfigurationError("openshell_fabric issuer is invalid")
    try:
        _ = parsed.port
    except ValueError as error:
        raise ConfigurationError("openshell_fabric issuer is invalid") from error
    if parsed.scheme == "https":
        return value.rstrip("/")
    if parsed.scheme == "http" and _host_is_loopback(parsed.hostname):
        return value.rstrip("/")
    raise ConfigurationError("openshell_fabric issuer must use HTTPS except on loopback")


def _tls_ca_file(profile: BackendProfile, *, tls: bool) -> Path | None:
    value = _optional(profile, "tls_ca_file")
    if value is None:
        return None
    if not tls:
        raise ConfigurationError("openshell_fabric tls_ca_file requires tls: true")
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise ConfigurationError("openshell_fabric tls_ca_file must be an existing absolute file")
    return path


__all__ = ["build_openshell_fabric_backend"]
