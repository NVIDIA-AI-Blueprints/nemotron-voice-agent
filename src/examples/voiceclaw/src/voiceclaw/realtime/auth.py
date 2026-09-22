# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Small stateless authentication boundary for the public Realtime facade."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass

_AUDIENCE = "voiceclaw-realtime"
_BROWSER_PREFIX = "openai-insecure-api-key."
_CLIENT_PREFIX = "ek_"
_MAX_SECRET_CHARACTERS = 4096


class RealtimeAuthenticationError(ValueError):
    """A public Realtime credential is absent, invalid, or expired."""


@dataclass(frozen=True, slots=True)
class ClientSecret:
    """One short-lived browser credential and its expiration epoch."""

    value: str
    expires_at: int


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    if not value or len(value) > _MAX_SECRET_CHARACTERS or any(character.isspace() for character in value):
        raise RealtimeAuthenticationError("invalid realtime credential")
    try:
        return base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise RealtimeAuthenticationError("invalid realtime credential") from error


class ClientSecretIssuer:
    """Issue and verify HMAC-authenticated ephemeral browser credentials."""

    def __init__(self, master_key: str) -> None:
        """Retain the deployment key only inside the server process."""
        if not isinstance(master_key, str) or len(master_key) < 24 or len(master_key) > 4096:
            raise ValueError("the VoiceClaw Realtime API key must contain 24 to 4096 characters")
        self._key = hashlib.sha256(b"voiceclaw/realtime/client-secret/v1\x00" + master_key.encode()).digest()
        self._master_digest = hmac.new(self._key, master_key.encode(), hashlib.sha256).digest()

    def issue(self, *, lifetime_seconds: int = 600, now: int | None = None) -> ClientSecret:
        """Create one opaque credential with a bounded lifetime."""
        if isinstance(lifetime_seconds, bool) or not 10 <= lifetime_seconds <= 3600:
            raise ValueError("client secret lifetime must be between 10 and 3600 seconds")
        issued_at = int(time.time()) if now is None else now
        expires_at = issued_at + lifetime_seconds
        payload = json.dumps(
            {
                "aud": _AUDIENCE,
                "exp": expires_at,
                "iat": issued_at,
                "nonce": secrets.token_urlsafe(18),
                "v": 1,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        signature = hmac.new(self._key, payload, hashlib.sha256).digest()
        return ClientSecret(value=f"{_CLIENT_PREFIX}{_encode(payload)}.{_encode(signature)}", expires_at=expires_at)

    def verify(self, credential: str, *, now: int | None = None) -> None:
        """Verify one ephemeral credential without returning secret-bearing state."""
        if not isinstance(credential, str) or not credential.startswith(_CLIENT_PREFIX):
            raise RealtimeAuthenticationError("invalid realtime credential")
        encoded = credential[len(_CLIENT_PREFIX) :]
        if encoded.count(".") != 1:
            raise RealtimeAuthenticationError("invalid realtime credential")
        payload_encoded, signature_encoded = encoded.split(".", 1)
        payload = _decode(payload_encoded)
        signature = _decode(signature_encoded)
        expected = hmac.new(self._key, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise RealtimeAuthenticationError("invalid realtime credential")
        try:
            claims = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RealtimeAuthenticationError("invalid realtime credential") from error
        if not isinstance(claims, dict) or set(claims) != {"aud", "exp", "iat", "nonce", "v"}:
            raise RealtimeAuthenticationError("invalid realtime credential")
        issued_at = claims.get("iat")
        expires_at = claims.get("exp")
        if (
            claims.get("aud") != _AUDIENCE
            or claims.get("v") != 1
            or isinstance(issued_at, bool)
            or not isinstance(issued_at, int)
            or isinstance(expires_at, bool)
            or not isinstance(expires_at, int)
            or not isinstance(claims.get("nonce"), str)
            or expires_at <= issued_at
            or expires_at - issued_at > 3600
        ):
            raise RealtimeAuthenticationError("invalid realtime credential")
        current = int(time.time()) if now is None else now
        if issued_at > current + 30 or current >= expires_at:
            raise RealtimeAuthenticationError("invalid realtime credential")

    def is_master(self, credential: str) -> bool:
        """Compare a supplied master credential without storing its cleartext twin."""
        if not isinstance(credential, str):
            return False
        candidate = hmac.new(self._key, credential.encode(), hashlib.sha256).digest()
        return hmac.compare_digest(candidate, self._master_digest)

    def authenticate_websocket(self, headers: Mapping[str, str]) -> None:
        """Accept a master bearer or browser ``ek_`` subprotocol credential."""
        credential = realtime_credential_from_headers(headers)
        if credential is None:
            raise RealtimeAuthenticationError("realtime credential required")
        if _browser_credential(headers) is not None:
            self.verify(credential)
            return
        if self.is_master(credential):
            return
        self.verify(credential)


def _bearer(headers: Mapping[str, str]) -> str | None:
    value = headers.get("authorization")
    if value is None:
        return None
    parts = value.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
        raise RealtimeAuthenticationError("invalid realtime authorization")
    return parts[1]


def _browser_credential(headers: Mapping[str, str]) -> str | None:
    raw = headers.get("sec-websocket-protocol", "")
    values = [
        offered[len(_BROWSER_PREFIX) :]
        for item in raw.split(",")
        if (offered := item.strip()).startswith(_BROWSER_PREFIX)
    ]
    if not values:
        return None
    if len(values) != 1 or not values[0].startswith(_CLIENT_PREFIX):
        raise RealtimeAuthenticationError("browser requires one ephemeral realtime credential")
    return values[0]


def bearer_from_headers(headers: Mapping[str, str]) -> str | None:
    """Expose strict bearer parsing for the client-secret HTTP endpoint."""
    return _bearer(headers)


def realtime_credential_from_headers(headers: Mapping[str, str]) -> str | None:
    """Return one unambiguous public Realtime credential, if supplied."""
    bearer = _bearer(headers)
    browser = _browser_credential(headers)
    if bearer is not None and browser is not None and not hmac.compare_digest(bearer.encode(), browser.encode()):
        raise RealtimeAuthenticationError("conflicting realtime credentials")
    return browser or bearer
