# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Signed, self-contained session tickets.

``POST /api/session-config`` validates a session config once and returns a ticket that
carries the sanitized config and a server-chosen session id. The client presents the ticket
when it opens the WebRTC offer or WebSocket, so whichever uvicorn worker (or replica)
receives that request can rebuild the session without any state shared with the worker that
issued it. The HMAC key must therefore be shared: ``SESSION_TICKET_SECRET`` across
replicas, and a key generated before the workers are spawned for one host.

Ticket format: ``base64url(zlib(json{sid, cfg, exp})).base64url(hmac_sha256)``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
import zlib
from dataclasses import dataclass

SECRET_ENV = "SESSION_TICKET_SECRET"
TICKET_TTL_SECS = 120
# Tickets travel in a URL query string; stay under common 8 KB request-line limits.
MAX_TICKET_CHARS = 6000
_MAX_DECOMPRESSED_BYTES = 256 * 1024

_process_secret: bytes | None = None


class TicketError(ValueError):
    """The ticket could not be issued or redeemed; the text is safe to show the client."""


@dataclass(frozen=True)
class SessionTicket:
    """A redeemed (or freshly issued) ticket's contents."""

    session_id: str
    config: dict


def share_secret_with_workers() -> None:
    """Export a signing key to the environment so forked uvicorn workers inherit it."""
    os.environ.setdefault(SECRET_ENV, secrets.token_hex(32))


def _secret() -> bytes:
    global _process_secret
    configured = os.environ.get(SECRET_ENV, "")
    if configured:
        return configured.encode()
    if _process_secret is None:
        _process_secret = secrets.token_bytes(32)
    return _process_secret


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(payload: str) -> str:
    return _b64(hmac.new(_secret(), payload.encode("ascii"), hashlib.sha256).digest())


def issue(config: dict, *, ttl_secs: int = TICKET_TTL_SECS, now: float | None = None) -> tuple[str, SessionTicket]:
    """Return ``(token, ticket)`` for a sanitized ``config``, with a new session id."""
    session_id = uuid.uuid4().hex[:12]
    body = {"sid": session_id, "cfg": config, "exp": int(now if now is not None else time.time()) + ttl_secs}
    payload = _b64(zlib.compress(json.dumps(body, separators=(",", ":")).encode()))
    token = f"{payload}.{_sign(payload)}"
    if len(token) > MAX_TICKET_CHARS:
        raise TicketError("Session config is too large to start a session; shorten the prompt or its overrides")
    return token, SessionTicket(session_id, config)


def redeem(token: str, *, now: float | None = None) -> SessionTicket:
    """Verify ``token`` and return its session; raise :class:`TicketError` if invalid or expired."""
    payload, _, signature = token.partition(".")
    if not payload or not signature or not hmac.compare_digest(signature, _sign(payload)):
        raise TicketError("Invalid session ticket")
    try:
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(_unb64(payload), _MAX_DECOMPRESSED_BYTES)
        body = json.loads(raw)
    except (zlib.error, binascii.Error, ValueError) as exc:
        raise TicketError("Invalid session ticket") from exc
    if not isinstance(body, dict) or not isinstance(body.get("cfg"), dict) or not isinstance(body.get("sid"), str):
        raise TicketError("Invalid session ticket")
    if (now if now is not None else time.time()) > float(body.get("exp", 0)):
        raise TicketError("Session ticket expired; start the session again")
    return SessionTicket(body["sid"], body["cfg"])
