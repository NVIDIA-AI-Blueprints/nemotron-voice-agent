# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

from __future__ import annotations

import os
from pathlib import Path

import pytest

from voiceclaw.adapters.openshell_fabric import security
from voiceclaw.adapters.openshell_fabric.security import client_secret_loader
from voiceclaw.config import BackendProfile, ConfigurationError


def _secret_file(path: Path, content: bytes = b"secret-value\n", *, mode: int = 0o600) -> Path:
    path.write_bytes(content)
    path.chmod(mode)
    return path


def _file_profile(path: Path) -> BackendProfile:
    return BackendProfile(kind="openshell_fabric", credential_file=str(path))


def test_file_secret_loader_reloads_rotated_credentials(tmp_path: Path) -> None:
    path = _secret_file(tmp_path / "client-secret", b"first-secret\n")
    loader = client_secret_loader(_file_profile(path), {})

    assert loader() == "first-secret"
    _secret_file(path, b"second-secret\n")
    assert loader() == "second-secret"


@pytest.mark.parametrize("mode", [0o400, 0o600, 0o640])
def test_secret_file_accepts_current_owner_and_group_read_policy(tmp_path: Path, mode: int) -> None:
    path = _secret_file(tmp_path / "client-secret", mode=mode)

    assert security._read_secret(path) == "secret-value"


@pytest.mark.parametrize("mode", [0o620, 0o610, 0o604, 0o601])
def test_secret_file_rejects_group_mutation_and_all_other_access(tmp_path: Path, mode: int) -> None:
    path = _secret_file(tmp_path / "client-secret", mode=mode)

    with pytest.raises(ConfigurationError, match="permissions are too broad"):
        security._read_secret(path)


def test_secret_file_rejects_symlinks_and_nonregular_files(tmp_path: Path) -> None:
    target = _secret_file(tmp_path / "target")
    link = tmp_path / "link"
    link.symlink_to(target)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo, mode=0o600)

    with pytest.raises(ConfigurationError, match="could not be read securely"):
        security._read_secret(link)
    with pytest.raises(ConfigurationError, match="bounded regular file"):
        security._read_secret(fifo)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "bounded regular file"),
        (b"\n", "client secret is invalid"),
        (b"two\nlines", "client secret is invalid"),
        (b"value\x00suffix", "client secret is invalid"),
        (b"\xff", "must contain UTF-8"),
        (b"x" * 4098, "bounded regular file"),
    ],
    ids=("empty", "newline-only", "embedded-newline", "nul", "invalid-utf8", "oversized"),
)
def test_secret_file_rejects_invalid_content(tmp_path: Path, content: bytes, message: str) -> None:
    path = _secret_file(tmp_path / "client-secret", content)

    with pytest.raises(ConfigurationError, match=message):
        security._read_secret(path)


def test_secret_file_accepts_exact_utf8_limit_with_one_trailing_newline(tmp_path: Path) -> None:
    path = _secret_file(tmp_path / "client-secret", b"x" * 4096 + b"\n")

    assert security._read_secret(path) == "x" * 4096


@pytest.mark.parametrize("content", [b"secret-value\n", b"\xff"])
def test_secret_file_descriptor_is_closed_on_success_and_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: bytes,
) -> None:
    path = _secret_file(tmp_path / "client-secret", content)
    real_close = security.os.close
    closed: list[int] = []

    def close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(security.os, "close", close)
    if content == b"\xff":
        with pytest.raises(ConfigurationError, match="must contain UTF-8"):
            security._read_secret(path)
    else:
        assert security._read_secret(path) == "secret-value"

    assert len(closed) == 1


def test_file_secret_configuration_requires_one_absolute_source(tmp_path: Path) -> None:
    absolute = _secret_file(tmp_path / "client-secret")

    with pytest.raises(ConfigurationError, match="exactly one"):
        client_secret_loader(BackendProfile(kind="openshell_fabric"), {})
    with pytest.raises(ConfigurationError, match="exactly one"):
        client_secret_loader(
            BackendProfile(
                kind="openshell_fabric",
                credential_env="SECRET",
                credential_file=str(absolute),
            ),
            {"SECRET": "secret-value"},
        )
    with pytest.raises(ConfigurationError, match="must be absolute"):
        client_secret_loader(
            BackendProfile(kind="openshell_fabric", credential_file="relative-secret"),
            {},
        )


@pytest.mark.parametrize("value", ["", "line\nbreak", "tab\tvalue", "x" * 4097])
def test_environment_secret_uses_the_same_bounded_content_validation(value: str) -> None:
    profile = BackendProfile(kind="openshell_fabric", credential_env="SECRET")

    with pytest.raises(ConfigurationError, match="client secret is invalid"):
        client_secret_loader(profile, {"SECRET": value})
