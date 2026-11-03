# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Who may use the live endpoints: the bearer token, the browser origin, the host name and a bounded request body."""

from __future__ import annotations

import hmac
import ipaddress
import json

from fastapi import HTTPException, Request

from live.gateway.settings import LiveSettings

MAX_BODY_BYTES = 2**20
LOCAL_NAMES = frozenset({"localhost"})


def authorize(settings: LiveSettings, connection) -> None:
    """Check the bearer token (when configured) and reject cross-site browser origins."""
    if settings.api_token:
        authorization = connection.headers.get("authorization", "")
        if not hmac.compare_digest(authorization.encode(), ("Bearer " + settings.api_token).encode()):
            raise HTTPException(401, "Invalid live bearer token")
    origin = connection.headers.get("origin")
    host = connection.headers.get("host")
    if origin and origin not in {f"http://{host}", f"https://{host}"}:
        raise HTTPException(403, "Unexpected browser origin")


def host_name(host: str) -> str:
    """The name of a ``Host`` header, without its port: ``[::1]:8000`` is ``::1``, ``app:8000`` is ``app``."""
    if host.startswith("["):
        return host[1 : host.find("]")] if "]" in host else host
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def check_host(settings: LiveSettings, connection) -> None:
    """With no token, answer only an IP literal, ``localhost`` or a name in ``allowed_hosts`` (DNS rebinding).

    A browser page on another site whose name resolves to this server sends its own name as ``Host``; a token, which no
    such page has, makes the check needless.
    """
    if settings.api_token:
        return
    name = host_name(connection.headers.get("host", "")).lower()
    try:
        ipaddress.ip_address(name)
        return
    except ValueError:
        pass
    if name not in LOCAL_NAMES and name not in settings.allowed_hosts:
        raise HTTPException(403, f"Host {name!r} is not allowed: add it to LIVE_ALLOWED_HOSTS, or set LIVE_API_TOKEN")


async def read_json_body(settings: LiveSettings, request: Request, limit: int = MAX_BODY_BYTES) -> dict:
    """Authorize the request and read its JSON object body, at most ``limit`` bytes."""
    authorize(settings, request)
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > limit:
            raise HTTPException(413, f"Request body exceeds {limit // 1024} KiB")
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise HTTPException(400, "Expected a JSON object") from exc
    if not isinstance(data, dict):
        raise HTTPException(400, "Expected a JSON object")
    return data
