# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Authentication and stateless client secrets for the Realtime gateway."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_CLIENT_SECRET_PREFIX = "ek_"
_CLIENT_SECRET_VERSION = 2
_CLIENT_SECRET_AUDIENCE = "nemotron-voice-agent-realtime"
_CLIENT_SECRET_KEY_CONTEXT = b"nemotron-voice-agent/realtime/client-secret/v2"
_CLIENT_SECRET_ASSOCIATED_DATA = b"nemotron-voice-agent/realtime/client-secret"
_CLIENT_SECRET_NONCE_BYTES = 12
_CLIENT_SECRET_MAX_CHARS = 16_384
_CLIENT_SECRET_MIN_LIFETIME_SECONDS = 10
_CLIENT_SECRET_MAX_LIFETIME_SECONDS = 7_200
_CLIENT_SECRET_CLOCK_SKEW_SECONDS = 30
_BROWSER_KEY_PREFIX = "openai-insecure-api-key."


class RealtimeAuthenticationError(ValueError):
    """A Realtime credential is absent, malformed, invalid, or expired."""


@dataclass(frozen=True, slots=True)
class RealtimeClientSecretClaims:
    """Verified configuration bound to a short-lived client secret."""

    session: dict[str, Any]
    issued_at: int
    expires_at: int


@dataclass(frozen=True, slots=True)
class RealtimeAuthentication:
    """Authentication result for one WebSocket handshake."""

    enabled: bool
    claims: RealtimeClientSecretClaims | None = None


def configured_realtime_api_key() -> str | None:
    """Return the configured master key, or ``None`` for local-development mode."""
    value = os.getenv("REALTIME_API_KEY", "").strip()
    return value or None


def _urlsafe_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _urlsafe_decode(value: str) -> bytes:
    if not value or any(character.isspace() for character in value):
        raise RealtimeAuthenticationError("Invalid Realtime client secret")
    padding = "=" * (-len(value) % 4)
    try:
        return base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RealtimeAuthenticationError("Invalid Realtime client secret") from exc


def _encryption_key(api_key: str) -> bytes:
    """Derive a domain-separated authenticated-encryption key from the master key."""
    return hmac.new(api_key.encode("utf-8"), _CLIENT_SECRET_KEY_CONTEXT, hashlib.sha256).digest()


def _constant_time_equal(left: str, right: str) -> bool:
    """Compare possibly non-ASCII credentials without ``compare_digest`` type errors."""
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def issue_realtime_client_secret(
    *,
    api_key: str,
    session: dict[str, Any],
    expires_at: int,
    issued_at: int | None = None,
) -> str:
    """Issue one opaque, reusable-until-expiry, authenticated ``ek_`` credential."""
    now = int(time.time()) if issued_at is None else issued_at
    if not api_key:
        raise ValueError("A Realtime API key is required to issue client secrets")
    if not isinstance(session, dict):
        raise TypeError("The bound Realtime session must be an object")
    lifetime = expires_at - now
    if lifetime < _CLIENT_SECRET_MIN_LIFETIME_SECONDS or lifetime > _CLIENT_SECRET_MAX_LIFETIME_SECONDS:
        raise ValueError("The client secret expiration is outside the supported lifetime")

    claims = {
        "aud": _CLIENT_SECRET_AUDIENCE,
        "exp": expires_at,
        "iat": now,
        "session": session,
        "v": _CLIENT_SECRET_VERSION,
    }
    payload = json.dumps(claims, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    nonce = os.urandom(_CLIENT_SECRET_NONCE_BYTES)
    ciphertext = AESGCM(_encryption_key(api_key)).encrypt(
        nonce,
        payload,
        _CLIENT_SECRET_ASSOCIATED_DATA,
    )
    secret = f"{_CLIENT_SECRET_PREFIX}{_urlsafe_encode(nonce)}.{_urlsafe_encode(ciphertext)}"
    if len(secret) > _CLIENT_SECRET_MAX_CHARS:
        raise ValueError("The bound Realtime session is too large for a WebSocket client secret")
    return secret


def verify_realtime_client_secret(
    secret: str,
    *,
    api_key: str,
    now: int | None = None,
) -> RealtimeClientSecretClaims:
    """Verify a stateless client secret and return its immutable claims."""
    if not api_key or not isinstance(secret, str) or len(secret) > _CLIENT_SECRET_MAX_CHARS:
        raise RealtimeAuthenticationError("Invalid Realtime client secret")
    if not secret.startswith(_CLIENT_SECRET_PREFIX):
        raise RealtimeAuthenticationError("Invalid Realtime client secret")
    encoded = secret[len(_CLIENT_SECRET_PREFIX) :]
    if encoded.count(".") != 1:
        raise RealtimeAuthenticationError("Invalid Realtime client secret")
    encoded_nonce, encoded_ciphertext = encoded.split(".", 1)
    nonce = _urlsafe_decode(encoded_nonce)
    ciphertext = _urlsafe_decode(encoded_ciphertext)
    if len(nonce) != _CLIENT_SECRET_NONCE_BYTES:
        raise RealtimeAuthenticationError("Invalid Realtime client secret")
    try:
        payload = AESGCM(_encryption_key(api_key)).decrypt(
            nonce,
            ciphertext,
            _CLIENT_SECRET_ASSOCIATED_DATA,
        )
    except (InvalidTag, ValueError) as exc:
        raise RealtimeAuthenticationError("Invalid Realtime client secret") from exc

    try:
        claims = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RealtimeAuthenticationError("Invalid Realtime client secret") from exc
    if not isinstance(claims, dict) or set(claims) != {"aud", "exp", "iat", "session", "v"}:
        raise RealtimeAuthenticationError("Invalid Realtime client secret")
    issued_at = claims.get("iat")
    expires_at = claims.get("exp")
    session = claims.get("session")
    if (
        claims.get("aud") != _CLIENT_SECRET_AUDIENCE
        or claims.get("v") != _CLIENT_SECRET_VERSION
        or isinstance(issued_at, bool)
        or not isinstance(issued_at, int)
        or isinstance(expires_at, bool)
        or not isinstance(expires_at, int)
        or not isinstance(session, dict)
        or expires_at <= issued_at
        or expires_at - issued_at < _CLIENT_SECRET_MIN_LIFETIME_SECONDS
        or expires_at - issued_at > _CLIENT_SECRET_MAX_LIFETIME_SECONDS
    ):
        raise RealtimeAuthenticationError("Invalid Realtime client secret")

    current_time = int(time.time()) if now is None else now
    if issued_at > current_time + _CLIENT_SECRET_CLOCK_SKEW_SECONDS or current_time >= expires_at:
        raise RealtimeAuthenticationError("Invalid or expired Realtime client secret")
    return RealtimeClientSecretClaims(session=session, issued_at=issued_at, expires_at=expires_at)


def _bearer_credential(headers: Mapping[str, str]) -> str | None:
    raw = headers.get("authorization")
    if raw is None:
        return None
    parts = raw.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
        raise RealtimeAuthenticationError("Invalid Realtime authorization header")
    return parts[1]


def _browser_subprotocol_credential(headers: Mapping[str, str]) -> str | None:
    raw = headers.get("sec-websocket-protocol") or ""
    credentials = [
        offered[len(_BROWSER_KEY_PREFIX) :]
        for value in raw.split(",")
        if (offered := value.strip()).startswith(_BROWSER_KEY_PREFIX)
    ]
    if not credentials:
        return None
    if len(credentials) != 1 or not credentials[0]:
        raise RealtimeAuthenticationError("Invalid Realtime WebSocket credential")
    return credentials[0]


def authenticate_realtime_websocket(
    headers: Mapping[str, str],
    *,
    api_key: str | None = None,
    now: int | None = None,
) -> RealtimeAuthentication:
    """Authenticate a server or browser WebSocket handshake.

    When no deployment master key is configured, the gateway intentionally
    remains open for local development and ignores optional client credentials.
    """
    configured_key = configured_realtime_api_key() if api_key is None else api_key
    if not configured_key:
        return RealtimeAuthentication(enabled=False)

    bearer = _bearer_credential(headers)
    browser = _browser_subprotocol_credential(headers)
    if bearer and browser and not _constant_time_equal(bearer, browser):
        raise RealtimeAuthenticationError("Conflicting Realtime credentials")
    if browser is not None and not browser.startswith(_CLIENT_SECRET_PREFIX):
        raise RealtimeAuthenticationError("Browser WebSocket authentication requires a client secret")
    if browser is not None:
        claims = verify_realtime_client_secret(browser, api_key=configured_key, now=now)
        return RealtimeAuthentication(enabled=True, claims=claims)

    credential = bearer
    if not credential:
        raise RealtimeAuthenticationError("A Realtime credential is required")
    if _constant_time_equal(credential, configured_key):
        return RealtimeAuthentication(enabled=True)
    claims = verify_realtime_client_secret(credential, api_key=configured_key, now=now)
    return RealtimeAuthentication(enabled=True, claims=claims)


def authenticate_realtime_master_key(headers: Mapping[str, str], *, api_key: str) -> bool:
    """Return whether an HTTP request supplied the exact deployment master key."""
    try:
        credential = _bearer_credential(headers)
    except RealtimeAuthenticationError:
        return False
    return bool(credential and _constant_time_equal(credential, api_key))
