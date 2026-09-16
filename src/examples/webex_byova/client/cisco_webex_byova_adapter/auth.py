# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Cisco JWS validation for the BYOVA adapter."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

import grpc
import jwt
from jwt import InvalidTokenError

from cisco_webex_byova_adapter.config import AdapterConfig

logger = logging.getLogger(__name__)

_JWK_PATH = "/oauth2/v2/keys/verificationjwk"
_DATASOURCE_URL_CLAIM = "com.cisco.datasource.url"
_DATASOURCE_SCHEMA_CLAIM = "com.cisco.datasource.schema.uuid"
_ORG_UUID_CLAIM = "com.cisco.org.uuid"


@dataclass(slots=True)
class _CachedJwkSet:
    keys: dict[str, dict[str, Any]]
    expires_at: float


class CiscoJwsValidator:
    """Validate Cisco-issued JWS bearer tokens against the issuer JWK set."""

    def __init__(self, config: AdapterConfig) -> None:
        """Create a validator backed by the adapter configuration."""
        self._config = config
        self._cache: dict[str, _CachedJwkSet] = {}
        self._lock = asyncio.Lock()

    async def validate(self, token: str) -> dict[str, Any]:
        """Validate a bearer token and return its decoded claims."""
        scheme, separator, token = token.partition(" ")
        if not separator or scheme.lower() != "bearer" or not token.strip():
            raise InvalidTokenError("missing bearer token")
        token = token.strip()

        header = jwt.get_unverified_header(token)
        if header.get("alg") != "RS256":
            raise InvalidTokenError("only RS256 is accepted")
        issuer = self._config.expected_jwt_issuer.rstrip("/")
        audience = self._config.expected_jwt_audience.strip()
        kid = str(header.get("kid", "")).strip()

        if not issuer or not audience:
            raise InvalidTokenError("CISCO_JWT_ISSUER and CISCO_JWT_AUDIENCE are required when auth is enabled")
        if not kid:
            raise InvalidTokenError("token missing required Cisco key id")

        jwk = await self._get_key_for_kid(issuer, kid)
        if jwk is None:
            raise InvalidTokenError(f"no public key found for kid={kid}")
        if jwk.get("kty") != "RSA" or jwk.get("alg") not in (None, "RS256"):
            raise InvalidTokenError("Cisco key is not valid for RS256")

        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
        decoded = jwt.decode(
            token,
            key=public_key,
            algorithms=["RS256"],
            audience=audience,
            issuer=issuer,
            leeway=self._config.auth_clock_skew_secs,
            options={"require": ["exp", "iss", "sub", "aud"]},
        )
        expected_claims = {
            "sub": self._config.expected_jwt_subject,
            "jti": self._config.expected_jwt_nonce,
            _DATASOURCE_URL_CLAIM: self._config.expected_datasource_url,
            _DATASOURCE_SCHEMA_CLAIM: self._config.expected_schema_uuid,
            _ORG_UUID_CLAIM: self._config.expected_org_uuid,
        }
        for claim, expected in expected_claims.items():
            if expected and decoded.get(claim) != expected:
                raise InvalidTokenError(f"{claim} did not match configured value")
        return decoded

    async def _get_key_for_kid(self, issuer: str, kid: str) -> dict[str, Any] | None:
        async with self._lock:
            cached = self._cache.get(issuer)
            now = time.monotonic()
            cache_is_current = cached is not None and cached.expires_at > now
            if not cache_is_current:
                cached = await asyncio.to_thread(self._fetch_jwk_set, issuer)
                self._cache[issuer] = cached
            jwk = cached.keys.get(kid)
            if jwk is None and cache_is_current:
                cached = await asyncio.to_thread(self._fetch_jwk_set, issuer)
                self._cache[issuer] = cached
                jwk = cached.keys.get(kid)
            return jwk

    def _fetch_jwk_set(self, issuer: str) -> _CachedJwkSet:
        url = f"{issuer}{_JWK_PATH}"
        logger.info("Fetching Cisco JWK set from %s", url)
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=15) as response:
            body = json.load(response)
        keys = {str(key.get("kid", "")): key for key in body.get("keys", []) if key.get("kid")}
        return _CachedJwkSet(keys=keys, expires_at=time.monotonic() + self._config.jwk_cache_ttl_secs)


class CiscoAuthInterceptor(grpc.aio.ServerInterceptor):
    """gRPC interceptor that enforces Cisco JWS validation when enabled."""

    def __init__(self, config: AdapterConfig) -> None:
        """Create an auth interceptor from adapter configuration."""
        self._config = config
        self._validator = CiscoJwsValidator(config)

    async def intercept_service(self, continuation, handler_call_details):
        """Validate auth metadata before dispatching BYOVA RPC handlers."""
        method = handler_call_details.method or ""
        if method.endswith("/Check") or method.endswith("/Watch"):
            return await continuation(handler_call_details)

        metadata = {key: value for key, value in (handler_call_details.invocation_metadata or [])}
        token = metadata.get("authorization", "")
        if self._config.enable_auth:
            try:
                claims = await self._validator.validate(token)
                logger.info("Cisco JWS validated for sub=%s aud=%s", claims.get("sub"), claims.get("aud"))
            except Exception as exc:
                logger.warning("Cisco JWS validation failed for method=%s: %s", method, exc)
                return self._unauthenticated_handler(method)
        return await continuation(handler_call_details)

    @staticmethod
    def _unauthenticated_handler(method: str):
        if method.endswith("/ProcessCallerInput"):

            async def abort_stream(request_iterator, context):
                await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid or missing Cisco JWS token")
                if False:
                    yield None

            return grpc.stream_stream_rpc_method_handler(abort_stream)

        async def abort_unary(request, context):
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid or missing Cisco JWS token")

        return grpc.unary_unary_rpc_method_handler(abort_unary)
