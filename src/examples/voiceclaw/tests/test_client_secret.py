# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

import pytest

import voiceclaw.client_secret as client_secret
from voiceclaw.config import ConfigurationError


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, amount: int) -> bytes:
        return self._payload[:amount]


def _master_key(tmp_path: Path) -> Path:
    path = tmp_path / "voiceclaw-public-master"
    path.write_text("voiceclaw-test-public-master-key-0001\n", encoding="ascii")
    path.chmod(0o600)
    return path


def test_issuer_keeps_master_out_of_argv_and_writes_only_owner_readable_client_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, **kwargs: Any) -> _Response:
        captured["authorization"] = request.get_header("Authorization")
        captured.update(kwargs)
        return _Response(b'{"value":"ek_test-browser-credential","expires_at":9999999999}')

    monkeypatch.setattr(client_secret, "urlopen", fake_urlopen)
    master_key = _master_key(tmp_path)
    output = tmp_path / "client-secrets" / "browser"

    client_secret.issue_client_secret(
        url="https://127.0.0.1:7860/v1/realtime/client_secrets",
        master_key_file=master_key,
        output=output,
        ca_file=None,
        insecure=True,
        timeout=3.0,
    )

    assert captured["authorization"] == "Bearer voiceclaw-test-public-master-key-0001"
    assert output.read_text(encoding="ascii") == "ek_test-browser-credential"
    assert stat.S_IMODE(output.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    options = {option for action in client_secret._parser()._actions for option in action.option_strings}
    assert "--master-key" not in options
    assert "--master-key-file" in options


def test_issuer_rejects_insecure_tls_for_a_nonloopback_listener(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="literal loopback"):
        client_secret.issue_client_secret(
            url="https://192.0.2.10:7860/v1/realtime/client_secrets",
            master_key_file=_master_key(tmp_path),
            output=tmp_path / "private" / "browser",
            ca_file=None,
            insecure=True,
            timeout=3.0,
        )


def test_issuer_accepts_a_verified_https_hostname(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, **kwargs: Any) -> _Response:
        captured["url"] = request.full_url
        captured.update(kwargs)
        return _Response(b'{"value":"ek_verified-host-credential"}')

    monkeypatch.setattr(client_secret, "urlopen", fake_urlopen)
    output = tmp_path / "private" / "browser"

    client_secret.issue_client_secret(
        url="https://voiceclaw.example.test/v1/realtime/client_secrets",
        master_key_file=_master_key(tmp_path),
        output=output,
        ca_file=None,
        insecure=False,
        timeout=3.0,
    )

    assert captured["url"] == "https://voiceclaw.example.test/v1/realtime/client_secrets"
    assert output.read_text(encoding="ascii") == "ek_verified-host-credential"


@pytest.mark.parametrize(
    "url",
    (
        "http://127.0.0.1:7860/v1/realtime/client_secrets",
        "https://user:password@voiceclaw.example.test/v1/realtime/client_secrets",
        "https://voiceclaw.example.test/v1/realtime/client_secrets?master=secret",
        "https://voiceclaw.example.test/v1/realtime/client_secrets#fragment",
    ),
)
def test_issuer_rejects_ambiguous_or_non_https_endpoints(tmp_path: Path, url: str) -> None:
    with pytest.raises(ValueError, match="HTTPS origin"):
        client_secret.issue_client_secret(
            url=url,
            master_key_file=_master_key(tmp_path),
            output=tmp_path / "private" / "browser",
            ca_file=None,
            insecure=False,
            timeout=3.0,
        )


def test_issuer_rejects_a_master_key_accessible_by_other_users(tmp_path: Path) -> None:
    master_key = _master_key(tmp_path)
    master_key.chmod(0o604)

    with pytest.raises(ConfigurationError, match="must not be group-writable or accessible by others"):
        client_secret.issue_client_secret(
            url="https://127.0.0.1:7860/v1/realtime/client_secrets",
            master_key_file=master_key,
            output=tmp_path / "private" / "browser",
            ca_file=None,
            insecure=True,
            timeout=3.0,
        )


def test_issuer_rejects_a_master_key_with_a_posix_access_acl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("voiceclaw.frontend_runtime.os.getxattr", lambda *_args, **_kwargs: b"acl")

    with pytest.raises(ConfigurationError, match="must not have a POSIX access ACL"):
        client_secret.issue_client_secret(
            url="https://127.0.0.1:7860/v1/realtime/client_secrets",
            master_key_file=_master_key(tmp_path),
            output=tmp_path / "private" / "browser",
            ca_file=None,
            insecure=True,
            timeout=3.0,
        )


def test_issuer_rejects_an_unbounded_or_non_client_credential_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client_secret, "urlopen", lambda *_args, **_kwargs: _Response(b'{"value":"master-key"}'))

    with pytest.raises(ValueError, match="invalid credential"):
        client_secret.issue_client_secret(
            url="https://127.0.0.1:7860/v1/realtime/client_secrets",
            master_key_file=_master_key(tmp_path),
            output=tmp_path / "private" / "browser",
            ca_file=None,
            insecure=True,
            timeout=3.0,
        )
