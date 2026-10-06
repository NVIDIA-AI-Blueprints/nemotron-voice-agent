# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Credential loading for the OpenShell service identity."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable, Mapping
from pathlib import Path

from voiceclaw.config import BackendProfile, ConfigurationError

_MAX_SECRET_BYTES = 4096


def client_secret_loader(profile: BackendProfile, environ: Mapping[str, str]) -> Callable[[], str]:
    """Return one bounded env- or file-backed OAuth secret supplier."""
    configured = sum(value is not None for value in (profile.credential_env, profile.credential_file))
    if configured != 1:
        raise ConfigurationError("openshell_fabric requires exactly one credential.env or credential.file")
    if profile.credential_env is not None:
        raw = environ.get(profile.credential_env)
        if raw is None:
            raise ConfigurationError(f"missing OpenShell client secret: {profile.credential_env}")
        secret = _validate_secret(raw)
        return lambda: secret
    assert profile.credential_file is not None
    path = Path(profile.credential_file)
    if not path.is_absolute():
        raise ConfigurationError("OpenShell credential.file must be absolute")

    def load() -> str:
        return _read_secret(path)

    load()
    return load


def _read_secret(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ConfigurationError("secure no-follow credential reads are unavailable")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags | no_follow)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 1 <= metadata.st_size <= _MAX_SECRET_BYTES + 1:
            raise ConfigurationError("OpenShell credential.file must be a bounded regular file")
        if metadata.st_mode & (stat.S_IWGRP | stat.S_IXGRP | stat.S_IRWXO):
            raise ConfigurationError("OpenShell credential.file permissions are too broad")
        chunks: list[bytes] = []
        remaining = _MAX_SECRET_BYTES + 2
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    except OSError as error:
        raise ConfigurationError("OpenShell credential.file could not be read securely") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(raw) > _MAX_SECRET_BYTES + 1:
        raise ConfigurationError("OpenShell credential.file is too large")
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ConfigurationError("OpenShell credential.file must contain UTF-8") from error
    if value.endswith("\n"):
        value = value[:-1]
    return _validate_secret(value)


def _validate_secret(value: str) -> str:
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ConfigurationError("OpenShell client secret must be valid UTF-8") from error
    if not encoded or len(encoded) > _MAX_SECRET_BYTES or any(ord(character) < 0x20 for character in value):
        raise ConfigurationError("OpenShell client secret is invalid")
    return value


__all__ = ["client_secret_loader"]
