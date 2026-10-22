# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Container Service Contract v1 settings read from the process environment."""

from __future__ import annotations

import ipaddress
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit
from uuid import UUID

from voiceclaw.adapters.openshell_fabric.protocol import FABRIC_ADAPTER_KINDS

DEFAULT_PORT = 18790
CREDENTIAL_LIMIT = 16 * 1024
INVOKE_SECONDS = 330
_NAME = re.compile(r"[a-z][a-z0-9-]{0,62}\Z")
_WORKSPACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_FABRIC_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_BEARER = re.compile(rb"[A-Za-z0-9._~+/-]+=*\Z")


class ContainerInputError(ValueError):
    """An input is absent, unsafe, or incompatible, without content."""

    def __init__(self) -> None:
        """Expose one fixed diagnostic without input names or values."""
        super().__init__("container-input-invalid")


def validate_bearer(payload: bytes) -> str:
    """Accept a bounded RFC 6750 bearer without header-breaking characters."""
    if not 1 <= len(payload) <= CREDENTIAL_LIMIT or _BEARER.fullmatch(payload) is None:
        raise ContainerInputError
    return payload.decode("ascii")


def validate_speech_key(payload: bytes) -> str:
    """Validate one NVIDIA API key value, independently of bearer framing."""
    if not 1 <= len(payload) <= CREDENTIAL_LIMIT or any(c < 33 or c > 126 for c in payload):
        raise ContainerInputError
    return payload.decode("ascii")


def strict_json(payload: bytes, maximum: int) -> dict:
    """Decode exactly one bounded UTF-8 JSON object with no duplicate keys."""

    def collect(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ContainerInputError
            result[key] = value
        return result

    def invalid_constant(_value):
        raise ContainerInputError

    try:
        if not 0 < len(payload) <= maximum:
            raise ContainerInputError
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=collect, parse_constant=invalid_constant)
        if not isinstance(value, dict):
            raise ContainerInputError
        return value
    except Exception:
        raise ContainerInputError from None


def _identity(value: object, pattern: re.Pattern) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ContainerInputError
    return value


def _uuid(value: object) -> str:
    try:
        if not isinstance(value, str) or str(UUID(value)) != value or UUID(value).int == 0:
            raise ContainerInputError
    except (ValueError, TypeError, AttributeError):
        raise ContainerInputError from None
    return value


def _adapter_kind(value: object) -> str:
    kind = _identity(value, _FABRIC_ID)
    if kind not in FABRIC_ADAPTER_KINDS:
        raise ContainerInputError
    return kind


def _json_pointer(value: object) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 512 or not value.startswith("/"):
        raise ContainerInputError
    index = 0
    while index < len(value):
        character = value[index]
        if ord(character) < 33 or ord(character) > 126:
            raise ContainerInputError
        if character == "~":
            index += 1
            if index == len(value) or value[index] not in "01":
                raise ContainerInputError
        index += 1
    return value


def _port(value: object) -> int:
    if value is None:
        return DEFAULT_PORT
    if not isinstance(value, str) or not value.isascii() or not value.isdigit() or value != str(int(value)):
        raise ContainerInputError
    port = int(value)
    if not 1 <= port <= 65535:
        raise ContainerInputError
    return port


def _endpoint(value: object, *, no_auth: bool = False) -> str:
    if not isinstance(value, str) or not value or any(ord(c) < 33 or ord(c) > 126 for c in value):
        raise ContainerInputError
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
        if (
            parsed.scheme != ("http" if no_auth else "https")
            or not host
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or parsed.query
            or parsed.path not in {"", "/"}
            or "#" in value
            or "?" in value
            or "%" in host
            or "\\" in value
            or port is None
            or not 1 <= port <= 65535
        ):
            raise ContainerInputError
        if no_auth:
            # A literal excludes DNS rebinding and alternate URL/IP spellings.
            # This is only an address filter: the operator must keep the gateway
            # on a trusted private network without LAN/public publication.
            address = ipaddress.IPv4Address(host)
            if value not in {f"http://{address}:{port}", f"http://{address}:{port}/"} or not any(
                address in ipaddress.IPv4Network(network)
                for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
            ):
                raise ContainerInputError
            return value
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            labels = host.split(".")
            if (
                host.lower() == "localhost"
                or host.lower().endswith(".localhost")
                or len(host) > 253
                or not any(c.isalpha() for c in host)
                or all(re.fullmatch(r"(?:0[xX][0-9A-Fa-f]+|[0-9]+)", label) for label in labels)
                or any(
                    re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label) is None for label in labels
                )
            ):
                raise ContainerInputError from None
        else:
            if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
                address = address.ipv4_mapped
            if address.is_loopback or address.is_unspecified or address.is_multicast or address.is_link_local:
                raise ContainerInputError
    except (ValueError, TypeError):
        raise ContainerInputError from None
    return value


@dataclass(frozen=True, slots=True)
class AgentConnection:
    """One explicit target and auth mode; no inferred credentials or fallback."""

    endpoint: str
    authentication_mode: str
    workspace: str
    sandbox: str
    sandbox_id: str
    agent: str
    fabric_runtime_id: str
    fabric_adapter_id: str
    fabric_harness: str
    fabric_adapter_kind: str
    fabric_result_pointer: str
    bearer: str | None = field(default=None, repr=False)
    invoke_seconds: int = INVOKE_SECONDS


@dataclass(frozen=True, slots=True)
class ContainerSettings:
    """The complete application environment; credentials are hidden from repr."""

    port: int
    connection: AgentConnection
    speech_key: str = field(repr=False)

    @classmethod
    def from_environment(cls, environ: Mapping[str, str]) -> ContainerSettings:
        """Validate every documented input before activation; never fall back."""
        mode = environ.get("VOICECLAW_AUTH_MODE")
        token = environ.get("VOICECLAW_GATEWAY_TOKEN")
        if mode == "oidcBearer" and token is not None:
            bearer = validate_bearer(token.encode("utf-8"))
        elif mode == "none" and token is None:
            # Never ignore a supplied credential or downgrade an authenticated
            # deployment: plaintext mode must be selected without a token.
            bearer = None
        else:
            raise ContainerInputError
        speech = environ.get("VOICECLAW_SPEECH_KEY")
        if speech is None:
            raise ContainerInputError
        connection = AgentConnection(
            endpoint=_endpoint(environ.get("VOICECLAW_GATEWAY_URL"), no_auth=mode == "none"),
            authentication_mode=mode,
            workspace=_identity(environ.get("VOICECLAW_WORKSPACE"), _WORKSPACE),
            sandbox=_identity(environ.get("VOICECLAW_SANDBOX_NAME"), _NAME),
            sandbox_id=_uuid(environ.get("VOICECLAW_SANDBOX_ID")),
            agent=_identity(environ.get("VOICECLAW_AGENT_NAME"), _NAME),
            fabric_runtime_id=_identity(environ.get("VOICECLAW_FABRIC_RUNTIME_ID"), _FABRIC_ID),
            fabric_adapter_id=_identity(environ.get("VOICECLAW_FABRIC_ADAPTER_ID"), _FABRIC_ID),
            fabric_harness=_identity(environ.get("VOICECLAW_FABRIC_HARNESS"), _FABRIC_ID),
            fabric_adapter_kind=_adapter_kind(environ.get("VOICECLAW_FABRIC_ADAPTER_KIND")),
            fabric_result_pointer=_json_pointer(environ.get("VOICECLAW_FABRIC_RESULT_POINTER")),
            bearer=bearer,
        )
        return cls(
            port=_port(environ.get("VOICECLAW_PORT")),
            connection=connection,
            speech_key=validate_speech_key(speech.encode("utf-8")),
        )
