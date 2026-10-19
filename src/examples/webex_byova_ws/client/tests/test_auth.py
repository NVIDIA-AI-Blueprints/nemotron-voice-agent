"""Cisco bearer JWS validation tests."""

from __future__ import annotations

import os
import time
from unittest.mock import AsyncMock

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt import InvalidTokenError

from webex_byova_ws_adapter.auth import CiscoJwsValidator, DatasourceNonce
from webex_byova_ws_adapter.config import SCHEMA_UUID, AdapterConfig


def config() -> AdapterConfig:
    """Return strict test authentication settings."""
    return AdapterConfig(
        allow_plaintext_local=True,
        auth_enabled=True,
        valid_issuers=("https://idbroker.webex.com/idb",),
        audience="audience",
        subject="subject",
        nonce="nonce",
        datasource_url="https://voice.example.test",
        schema_uuid=SCHEMA_UUID,
        org_uuid="org-1",
    )


@pytest.mark.asyncio
async def test_valid_rs256_token_checks_all_bindings() -> None:
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk = jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key(), as_dict=True)
    public_jwk.update({"kid": "key-1", "alg": "RS256"})
    now = int(time.time())
    claims = {
        "iss": "https://idbroker.webex.com/idb",
        "aud": "audience",
        "sub": "subject",
        "jti": "nonce",
        "iat": now,
        "exp": now + 300,
        "com.cisco.datasource.url": "https://voice.example.test",
        "com.cisco.datasource.schema.uuid": SCHEMA_UUID,
        "com.cisco.org.uuid": "org-1",
    }
    token = jwt.encode(claims, private, algorithm="RS256", headers={"kid": "key-1"})
    validator = CiscoJwsValidator(config())
    validator._key = AsyncMock(return_value=public_jwk)
    assert (await validator.validate(f"Bearer {token}"))["jti"] == "nonce"


@pytest.mark.asyncio
async def test_org_binding_mismatch_is_rejected() -> None:
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk = jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key(), as_dict=True)
    public_jwk.update({"kid": "key-1", "alg": "RS256"})
    now = int(time.time())
    token = jwt.encode(
        {
            "iss": "https://idbroker.webex.com/idb",
            "aud": "audience",
            "sub": "subject",
            "jti": "nonce",
            "iat": now,
            "exp": now + 300,
            "com.cisco.datasource.url": "https://voice.example.test",
            "com.cisco.datasource.schema.uuid": SCHEMA_UUID,
            "com.cisco.org.uuid": "wrong-org",
        },
        private,
        algorithm="RS256",
        headers={"kid": "key-1"},
    )
    validator = CiscoJwsValidator(config())
    validator._key = AsyncMock(return_value=public_jwk)
    with pytest.raises(InvalidTokenError, match="org"):
        await validator.validate(f"Bearer {token}")


def test_unsafe_modes_require_explicit_local_flags() -> None:
    with pytest.raises(ValueError, match="TLS"):
        AdapterConfig(auth_enabled=False, allow_auth_disabled_local=True).validate()
    with pytest.raises(ValueError, match="disabled auth"):
        AdapterConfig(allow_plaintext_local=True, auth_enabled=False).validate()
    AdapterConfig(
        allow_plaintext_local=True,
        auth_enabled=False,
        allow_auth_disabled_local=True,
    ).validate()


@pytest.mark.parametrize(
    "override",
    [
        {"port": 0},
        {"queue_size": 0},
        {"max_message_bytes": 100},
        {"send_timeout_seconds": 0},
        {"first_audio_timeout_seconds": 0},
        {"completion_timeout_seconds": 0},
        {"jwks_ttl_seconds": 0},
        {"auth_clock_skew_seconds": -1},
    ],
)
def test_runtime_bounds_are_validated_in_local_mode(override: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        AdapterConfig(
            allow_plaintext_local=True,
            auth_enabled=False,
            allow_auth_disabled_local=True,
            **override,
        ).validate()


def test_nonce_file_reload_tracks_rotation_without_restart(tmp_path) -> None:
    """Pick up a rotated nonce from disk the next time a token is validated."""
    path = tmp_path / "byova-nonce"
    path.write_text("first\n", encoding="utf-8")
    nonce = DatasourceNonce(AdapterConfig(nonce="static", nonce_file=str(path)))
    assert nonce.current() == "first"

    path.write_text("second\n", encoding="utf-8")
    os.utime(path, (0, 0))
    assert nonce.current() == "second"


def test_missing_or_empty_nonce_file_fails_closed(tmp_path) -> None:
    """Reject tokens when rotation state cannot be read."""
    missing = DatasourceNonce(AdapterConfig(nonce_file=str(tmp_path / "absent")))
    with pytest.raises(InvalidTokenError):
        missing.current()

    path = tmp_path / "byova-nonce"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(InvalidTokenError):
        DatasourceNonce(AdapterConfig(nonce_file=str(path))).current()


def test_auth_accepts_a_nonce_file_instead_of_a_literal_nonce(tmp_path) -> None:
    """Treat a configured nonce file as satisfying the nonce requirement."""
    settings = config()
    settings.nonce = ""
    settings.nonce_file = str(tmp_path / "byova-nonce")
    settings.validate()
