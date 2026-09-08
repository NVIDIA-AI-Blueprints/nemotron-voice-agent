"""Pre-upgrade validation of Cisco RS256 bearer JWS credentials."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import aiohttp
import jwt
from jwt import InvalidTokenError

from webex_byova_ws_adapter.config import AdapterConfig

DATASOURCE_URL = "com.cisco.datasource.url"
DATASOURCE_SCHEMA = "com.cisco.datasource.schema.uuid"
ORG_UUID = "com.cisco.org.uuid"
JWKS_PATH = "/oauth2/v2/keys/verificationjwk"


@dataclass(slots=True)
class _KeySet:
    keys: dict[str, dict[str, Any]]
    expires: float


class CiscoJwsValidator:
    """Validate signature, registered claims, nonce, and datasource binding."""

    def __init__(self, config: AdapterConfig) -> None:
        """Create a validator for one adapter configuration."""
        self.config = config
        self._cache: dict[str, _KeySet] = {}
        self._lock = asyncio.Lock()

    async def validate(self, authorization: str) -> dict[str, Any]:
        """Validate one Authorization header and return trusted claims."""
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise InvalidTokenError("missing bearer token")
        header = jwt.get_unverified_header(token)
        if header.get("alg") != "RS256":
            raise InvalidTokenError("only RS256 is accepted")
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            raise InvalidTokenError("token is missing kid")
        untrusted = jwt.decode(token, options={"verify_signature": False})
        issuer = untrusted.get("iss")
        if issuer not in self.config.valid_issuers:
            raise InvalidTokenError("issuer is not allowlisted")
        jwk = await self._key(issuer, kid)
        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
        claims = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            issuer=issuer,
            audience=self.config.audience,
            leeway=self.config.auth_clock_skew_seconds,
            options={"require": ["exp", "iat", "iss", "aud", "sub", "jti"]},
        )
        expected = {
            "sub": self.config.subject,
            "jti": self.config.nonce,
            DATASOURCE_URL: self.config.datasource_url,
            DATASOURCE_SCHEMA: self.config.schema_uuid,
            ORG_UUID: self.config.org_uuid,
        }
        for claim, value in expected.items():
            if claims.get(claim) != value:
                raise InvalidTokenError(f"{claim} did not match configured value")
        if (
            not isinstance(claims["iat"], (int, float))
            or claims["iat"] > time.time() + self.config.auth_clock_skew_seconds
        ):
            raise InvalidTokenError("iat is in the future")
        return claims

    async def _key(self, issuer: str, kid: str) -> dict[str, Any]:
        async with self._lock:
            cached = self._cache.get(issuer)
            now = time.monotonic()
            if cached is None or cached.expires <= now:
                cached = await self._fetch(issuer)
                self._cache[issuer] = cached
            key = cached.keys.get(kid)
            if key is None:
                cached = await self._fetch(issuer)
                self._cache[issuer] = cached
                key = cached.keys.get(kid)
            if key is None or key.get("kty") != "RSA":
                raise InvalidTokenError("kid does not identify an RSA JWK")
            if key.get("alg") not in (None, "RS256"):
                raise InvalidTokenError("JWK algorithm is not RS256")
            return key

    async def _fetch(self, issuer: str) -> _KeySet:
        timeout = aiohttp.ClientTimeout(total=10)
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.get(f"{issuer.rstrip('/')}{JWKS_PATH}") as response,
        ):
            response.raise_for_status()
            body = await response.json(content_type=None)
        keys = {
            item["kid"]: item
            for item in body.get("keys", [])
            if isinstance(item, dict) and isinstance(item.get("kid"), str)
        }
        return _KeySet(keys, time.monotonic() + self.config.jwks_ttl_seconds)
