# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Composition factory for the configured NemoClaw connection."""

from __future__ import annotations

import errno
import math
import os
import stat
from collections.abc import Mapping
from enum import StrEnum

from voiceclaw.adapters.nemoclaw.committed_turn import (
    DEFAULT_RESULT_DISPLAY_BUDGET_BYTES,
    EndpointPolicy,
    NemoClawCommittedTurnAdapter,
)
from voiceclaw.adapters.result_envelope import MAX_RESULT_DISPLAY_BYTES
from voiceclaw.config import BackendProfile, ConfigurationError
from voiceclaw.model_contracts import ModelContractCatalog, load_model_contract_catalog
from voiceclaw.ports.turns import EphemeralCommittedTurnPort

_DEFAULT_ENDPOINT = "http://127.0.0.1:18800"
_ALLOWED_SETTINGS = frozenset(
    {
        "mode",
        "endpoint",
        "endpoint_policy",
        "exchange_deadline_seconds",
        "result_display_budget_bytes",
    }
)


class NemoClawConnectionMode(StrEnum):
    """Connection modes whose semantics are implemented by this adapter."""

    RESPONSE_ONLY = "response_only"


def _credential(profile: BackendProfile, environ: Mapping[str, str]) -> str | None:
    del environ
    if profile.credential_env is not None:
        raise ConfigurationError("NemoClaw committed-turn credentials must use credential.file")
    if profile.credential_file is None:
        return None
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ConfigurationError("secure no-follow credential reads are unavailable")
    descriptor: int | None = None
    try:
        descriptor = os.open(profile.credential_file, flags | no_follow)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 1 <= metadata.st_size <= 4097:
            raise ConfigurationError("NemoClaw credential file must be a bounded regular file")
        if metadata.st_mode & (stat.S_IWGRP | stat.S_IXGRP | stat.S_IRWXO):
            raise ConfigurationError(
                "NemoClaw credential file must be owner-readable, optionally group-readable, and inaccessible to others"
            )
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
                    raise ConfigurationError("NemoClaw credential file POSIX ACL could not be inspected") from error
            else:
                if access_acl:
                    raise ConfigurationError("NemoClaw credential file must not have a POSIX access ACL")
        raw = os.read(descriptor, 4098)
    except OSError as error:
        raise ConfigurationError("NemoClaw credential file could not be read securely") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(raw) > 4097:
        raise ConfigurationError("NemoClaw credential file is too large")
    try:
        value = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ConfigurationError("NemoClaw credential file must contain visible ASCII") from error
    if value.endswith("\n"):
        value = value[:-1]
    if not 32 <= len(value) <= 4096 or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value):
        raise ConfigurationError("NemoClaw credential file must contain one 32..4096 byte visible ASCII bearer")
    return value


def _setting(profile: BackendProfile, name: str) -> str:
    value = profile.settings.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"backend profile {profile.kind!r} requires a non-empty {name!r} setting")
    return value.strip()


def _endpoint(profile: BackendProfile) -> str:
    if "endpoint" in profile.settings:
        return _setting(profile, "endpoint")
    return _DEFAULT_ENDPOINT


def _validate_settings(profile: BackendProfile) -> None:
    unknown = sorted(set(profile.settings) - _ALLOWED_SETTINGS)
    if unknown:
        raise ConfigurationError(f"unknown NemoClaw settings: {', '.join(unknown)}")
    try:
        NemoClawConnectionMode(profile.settings.get("mode", NemoClawConnectionMode.RESPONSE_ONLY.value))
    except (TypeError, ValueError) as error:
        raise ConfigurationError(
            "NemoClaw mode must be 'response_only'; the durable Agent Session connection is not implemented yet"
        ) from error


def _positive_float_setting(profile: BackendProfile, name: str, default: float) -> float:
    value = profile.settings.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f"backend profile {profile.kind!r} requires {name!r} to be a finite positive number")
    return float(value)


def _bounded_int_setting(profile: BackendProfile, name: str, default: int, maximum: int) -> int:
    value = profile.settings.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ConfigurationError(
            f"backend profile {profile.kind!r} requires {name!r} to be an integer from 1 through {maximum}"
        )
    return value


def build_nemoclaw_backend(
    profile: BackendProfile,
    environ: Mapping[str, str],
    model_contracts: ModelContractCatalog | None = None,
) -> tuple[EphemeralCommittedTurnPort, str]:
    """Validate and construct the selected NemoClaw connection mode."""
    _validate_settings(profile)
    credential = _credential(profile, environ)
    if credential is None:
        source = profile.credential_env or profile.credential_file or "<unset>"
        raise ConfigurationError(f"missing required NemoClaw committed-turn credential: {source}")
    policy_value = profile.settings.get("endpoint_policy", EndpointPolicy.LOOPBACK_ONLY.value)
    try:
        policy = EndpointPolicy(policy_value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError("nemoclaw committed-turn endpoint_policy is invalid") from error
    adapter = NemoClawCommittedTurnAdapter(
        origin=_endpoint(profile),
        deployment_bearer=credential,
        endpoint_policy=policy,
        exchange_deadline_seconds=_positive_float_setting(profile, "exchange_deadline_seconds", 125.0),
        result_display_budget_bytes=_bounded_int_setting(
            profile,
            "result_display_budget_bytes",
            DEFAULT_RESULT_DISPLAY_BUDGET_BYTES,
            MAX_RESULT_DISPLAY_BYTES,
        ),
        model_contracts=model_contracts or load_model_contract_catalog(),
    )
    return adapter, "response_only"


def build_committed_turn_backend(
    profile: BackendProfile,
    environ: Mapping[str, str],
    model_contracts: ModelContractCatalog | None = None,
) -> tuple[EphemeralCommittedTurnPort, str]:
    """Build the legacy-named response-only connection for compatibility."""
    return build_nemoclaw_backend(profile, environ, model_contracts)


__all__ = ["NemoClawConnectionMode", "build_committed_turn_backend", "build_nemoclaw_backend"]
