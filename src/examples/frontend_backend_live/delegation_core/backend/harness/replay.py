# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Once-only HTTP admission for the handshake-era MCP bridge, before the SDK sees a repeated RPC ID.

The authenticated door and a server-issued MCP session define the namespace. This recognizes retransmission of
an existing operation; it does not recognize a native agent invoking the same tool again under a fresh RPC ID.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from examples.shared.ids import approx_tokens, uid

OPERATION_SCOPE = "nvidia.voice_agent/native_operation"
SESSION_HEADER = b"mcp-session-id"
VERSION_HEADER = b"mcp-protocol-version"


@dataclass(frozen=True)
class Operation:
    """Immutable first-arrival identity and handler binding, kept out of the model's business arguments."""

    id: str
    door: Any
    call: Callable[[str, dict], Awaitable[Any]]
    context: Any

    @classmethod
    def capture(cls, door: Any) -> Operation:
        """Bind a new request to the door and foreground it reached at admission."""
        return cls(uid("native_operation"), door, door.capture_call(), door.context.copy())


@dataclass(frozen=True)
class HttpReply:
    """One completed SDK HTTP response; replaying it cannot dispatch another SDK request."""

    status: int
    headers: tuple[tuple[bytes, bytes], ...]
    body: bytes

    async def send(self, send: Callable) -> None:
        """Send the retained HTTP response through ASGI."""
        await send({"type": "http.response.start", "status": self.status, "headers": list(self.headers)})
        await send({"type": "http.response.body", "body": self.body})


@dataclass
class _Receipt:
    signature: bytes
    operation_id: str
    owner: asyncio.Task[HttpReply] | None = None
    reply: HttpReply | None = None


@dataclass
class _Session:
    door: Any
    version: str
    receipts: dict[tuple[type, str | int], _Receipt] = field(default_factory=dict)
    aliases: dict[str | int, tuple[type, str | int]] = field(default_factory=dict)
    closed: bool = False


@dataclass
class _Retention:
    receipt_tokens: int = 0
    reply_tokens: int = 0


def _rpc_error(identity: str | int | None, message: str, status: int = 409) -> HttpReply:
    body = json.dumps({"jsonrpc": "2.0", "id": identity, "error": {"code": -32600, "message": message}}).encode()
    headers = ((b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()))
    return HttpReply(status, headers, body)


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON member")
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"Invalid JSON constant: {value}")


class NativeHttpReplay:
    """Bind SDK-minted sessions to doors and retain exact request/result identity until that door is forgotten."""

    def __init__(self, context_budget: int):
        """Bound each door's request receipts and cached replies independently."""
        if type(context_budget) is not int or context_budget <= 0:
            raise ValueError("Native replay retention needs the existing positive Core context budget")
        # This is the existing host context estimate, not exact model-token or Python-memory accounting.
        self.context_budget = context_budget
        self.sessions: dict[str, _Session] = {}
        self.retention: dict[Any, _Retention] = {}
        self.pending: set[asyncio.Task] = set()

    def forget(self, door: Any) -> None:
        """Expire the complete namespace; its credential must already have been revoked by the owning bridge."""
        self.sessions = {key: session for key, session in self.sessions.items() if session.door is not door}
        self.retention.pop(door, None)

    async def drain(self) -> None:
        """Join original HTTP owners after SDK shutdown; no business operation is cancelled or retried here."""
        if self.pending:
            await asyncio.gather(*(asyncio.shield(task) for task in tuple(self.pending)), return_exceptions=True)

    async def handle(
        self, app: Callable, stateless: Callable, door: Any, scope: dict, receive: Callable, send: Callable
    ) -> None:
        """Preserve stateless requests and replay exact operations in a known MCP session."""
        from mcp.server.transport_security import DEFAULT_MAX_REQUEST_BODY_SIZE
        from mcp.shared.dispatcher import coerce_request_id
        from mcp_types.version import HANDSHAKE_PROTOCOL_VERSIONS

        raw_headers = scope.get("headers") or []
        for header in (b"authorization", SESSION_HEADER, VERSION_HEADER):
            if sum(key == header for key, _ in raw_headers) > 1:
                await _rpc_error(None, "Duplicate MCP authentication or session header", 400).send(send)
                return
        headers = dict(raw_headers)
        version = headers.get(VERSION_HEADER, b"").decode("latin-1")
        session_id = headers.get(SESSION_HEADER, b"").decode("latin-1")
        if not session_id and version and version not in HANDSHAKE_PROTOCOL_VERSIONS:
            await stateless({**scope, OPERATION_SCOPE: Operation.capture(door)}, receive, send)
            return
        body = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            body.extend(event.get("body", b""))
            if len(body) > DEFAULT_MAX_REQUEST_BODY_SIZE:
                await _rpc_error(None, "MCP request body exceeds the transport limit", 413).send(send)
                return
            if not event.get("more_body", False):
                break
        try:
            request = json.loads(body, object_pairs_hook=_pairs, parse_constant=_invalid_constant)
            if (
                not isinstance(request, dict)
                or request.get("jsonrpc") != "2.0"
                or not isinstance(request.get("method"), str)
            ):
                raise ValueError("Expected one JSON-RPC request or notification")
            identity = request.get("id")
            if "id" in request and type(identity) not in (str, int):
                raise ValueError("A request ID must be a non-null string or integer")
            signature = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).digest()
        except (ValueError, TypeError, UnicodeError) as exc:
            await _rpc_error(None, str(exc), 400).send(send)
            return
        session = self.sessions.get(session_id)
        if session_id and (session is None or session.door is not door):
            await _rpc_error(identity, "Unknown MCP session for this authenticated door", 404).send(send)
            return
        if session is not None and version and version != session.version:
            await _rpc_error(identity, "The MCP protocol version differs from the initialized session", 400).send(send)
            return
        if session is None:
            params = request.get("params")
            offered = params.get("protocolVersion") if isinstance(params, dict) else None
            if (
                door.retired
                or request["method"] != "initialize"
                or "id" not in request
                or offered not in HANDSHAKE_PROTOCOL_VERSIONS
            ):
                buffered = body

                async def receive_buffered() -> dict:
                    nonlocal buffered
                    if buffered is None:
                        return await receive()
                    payload, buffered = buffered, None
                    return {"type": "http.request", "body": bytes(payload), "more_body": False}

                await stateless({**scope, OPERATION_SCOPE: Operation.capture(door)}, receive_buffered, send)
                return
        elif "id" not in request:
            # Notifications have no response/replay identity and cannot invoke tool handlers. They are delivered
            # only into their already-authenticated namespace; the SDK validates the notification method.
            if session.closed:
                await _rpc_error(None, "MCP session is no longer available", 404).send(send)
                return
            await (await self._exchange(app, scope, bytes(body))).send(send)
            return
        else:
            key = type(identity), identity
            prior = session.receipts.get(key)
            if prior is not None:
                if prior.signature != signature:
                    await _rpc_error(identity, "An MCP request ID cannot be reused for different contents").send(send)
                elif prior.reply is not None:
                    await prior.reply.send(send)
                elif prior.owner is not None:
                    await (await asyncio.shield(prior.owner)).send(send)
                else:
                    await _rpc_error(
                        identity, "The original operation result is unavailable; it was not executed again"
                    ).send(send)
                return
            alias = coerce_request_id(identity)
            if alias in session.aliases:
                await _rpc_error(identity, "This typed request ID collides with an earlier SDK correlation ID").send(
                    send
                )
                return
            if session.closed or (door.retired and request["method"] != "ping"):
                await _rpc_error(identity, "This MCP session accepts no new tool operations").send(send)
                return
        retention = self.retention.setdefault(door, _Retention())
        operation = Operation.capture(door)
        metadata = json.dumps(
            {
                "session_id": session_id,
                "request_id_type": type(identity).__name__,
                "request_id": identity,
                "operation_id": operation.id,
                "signature": signature.hex(),
            }
        )
        charge = approx_tokens(metadata)
        if door.replay_exhausted or retention.receipt_tokens + charge > self.context_budget:
            door.replay_exhausted = True
            await _rpc_error(identity, "Native replay retention is full; no new operation was executed", 503).send(send)
            return
        receipt = _Receipt(signature, operation.id)
        retention.receipt_tokens += charge
        if session is not None:
            session.receipts[type(identity), identity] = receipt
            session.aliases[coerce_request_id(identity)] = type(identity), identity
        bound_scope = {**scope, OPERATION_SCOPE: operation}

        async def exchange() -> HttpReply:
            reply = await self._exchange(app, bound_scope, bytes(body))
            target = session
            retained = self.retention.get(door) is retention
            if target is None and retained:
                minted = dict(reply.headers).get(SESSION_HEADER, b"").decode("latin-1")
                if 200 <= reply.status < 300 and minted:
                    initialized = json.loads(reply.body)
                    negotiated = initialized.get("result", {}).get("protocolVersion")
                    if negotiated not in HANDSHAKE_PROTOCOL_VERSIONS:
                        raise RuntimeError("The SDK initialized an unsupported native bridge protocol")
                    target = _Session(door, negotiated)
                    target.receipts[type(identity), identity] = receipt
                    target.aliases[coerce_request_id(identity)] = type(identity), identity
                    self.sessions[minted] = target
            elif target is not None and reply.status == 404:
                target.closed = True
            # Keep the receipt when a reply exceeds the cache budget so a retry cannot execute it again.
            header_text = json.dumps([(key.decode("latin-1"), value.decode("latin-1")) for key, value in reply.headers])
            charge = approx_tokens(header_text + reply.body.decode("latin-1"))
            if retained and retention.reply_tokens + charge <= self.context_budget:
                receipt.reply = reply
                retention.reply_tokens += charge
            return reply

        owner = receipt.owner = asyncio.create_task(exchange(), name="native-http-operation")
        self.pending.add(owner)

        def finished(done: asyncio.Task) -> None:
            self.pending.discard(done)
            receipt.owner = None
            if not done.cancelled():
                done.exception()

        owner.add_done_callback(finished)
        await (await asyncio.shield(owner)).send(send)

    async def _exchange(self, app: Callable, scope: dict, body: bytes) -> HttpReply:
        """Run one SDK request independently of HTTP waiter cancellation and collect its single JSON response."""
        received = False
        complete = asyncio.Event()
        status = 500
        headers: tuple[tuple[bytes, bytes], ...] = ()
        parts: list[bytes] = []

        async def receive() -> dict:
            nonlocal received
            if not received:
                received = True
                return {"type": "http.request", "body": body, "more_body": False}
            await complete.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict) -> None:
            nonlocal status, headers
            if message["type"] == "http.response.start":
                status, headers = message["status"], tuple(message.get("headers") or ())
            elif message["type"] == "http.response.body":
                parts.append(message.get("body", b""))
                if not message.get("more_body", False):
                    complete.set()

        try:
            await app(scope, receive, send)
        except Exception:
            # A failed transport is not proof that its handler did nothing. The original identity remains
            # admitted and this uncertain result is replayed; a second arrival never executes it again.
            return _rpc_error(None, "The original MCP transport failed; operation outcome may be unknown", 500)
        finally:
            complete.set()
        return HttpReply(status, headers, b"".join(parts))
