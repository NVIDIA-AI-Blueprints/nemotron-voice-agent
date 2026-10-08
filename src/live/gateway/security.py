# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Who may use the live endpoints: the bearer token, the browser origin, and a bounded request body."""

from __future__ import annotations

import hmac
import json

from fastapi import HTTPException, Request

from live.gateway.settings import LiveSettings

MAX_BODY_BYTES = 2**20


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


async def read_json_body(settings: LiveSettings, request: Request) -> dict:
    """Authorize the request and read its bounded JSON object body."""
    authorize(settings, request)
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > MAX_BODY_BYTES:
            raise HTTPException(413, "Request body exceeds 1 MiB")
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise HTTPException(400, "Expected a JSON object") from exc
    if not isinstance(data, dict):
        raise HTTPException(400, "Expected a JSON object")
    return data
