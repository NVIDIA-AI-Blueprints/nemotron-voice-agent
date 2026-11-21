# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A short process-local logical-call lease, independent of attachment-bound media."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import json
import secrets
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger

from examples.shared.observe import log_observation
from gateways.live.events import event
from gateways.live.protocol import ProtocolError

if TYPE_CHECKING:
    from gateways.live.gateway.gateway import LiveGateway
    from gateways.live.gateway.websocket import PrimarySocketView
    from gateways.live.session.session import LiveProtocolSession

# ASGI transports can report established TCP loss as 1005 (no close status), rather than 1006.
TRANSIENT_CLOSE_CODES = frozenset({1005, 1006, 1012, 1013})


def unavailable() -> ProtocolError:
    """Use one refusal for unknown, expired, foreign-scope and stale resume authority."""
    return ProtocolError(
        "This call cannot be resumed; start a new call without replaying uncertain actions", "resume_unavailable"
    )


def token_key(token: str) -> str:
    """Keep resume authority out of registry keys and diagnostic representations."""
    return hashlib.sha256(token.encode()).hexdigest()


def auth_scope(gateway: LiveGateway, connection) -> str:
    """Bind to this endpoint and its actual bearer/origin/host policy, without inventing a user identity."""
    settings = gateway.settings
    value = repr(
        (
            settings.api_token,
            settings.allowed_hosts,
            connection.headers.get("host", "").lower(),
            connection.headers.get("origin", ""),
        )
    )
    return hashlib.sha256(value.encode()).hexdigest()


async def _call(part, name: str, *args):
    method = getattr(part, name, None)
    if method is None:
        raise unavailable()
    result = method(*args)
    return await result if inspect.isawaitable(result) else result


class ReconnectLease:
    """One logical call's sole attachment claim, fixed grace deadline, and exact resume receipt."""

    def __init__(self, session: LiveProtocolSession, gateway: LiveGateway, scope: str, seconds: float):
        """Retain no media; register protected authority separately from the public logical session id."""
        self.session, self.gateway, self.scope, self.seconds = session, gateway, scope, seconds
        self.lock = asyncio.Lock()
        self.generation = 1
        self.token = secrets.token_urlsafe(32)
        self.previous_key: str | None = None
        self.last_request: str | None = None
        self.deadline: float | None = None
        self.expiry: asyncio.Task | None = None
        self.state = "active"
        self.snapshot: dict | None = None
        self.cleanup: set[asyncio.Task] = set()
        # Resume restores the original frontend settings and reads current backend settings.
        self.frontend_config = session.config
        self.operations: dict[str, tuple[str, str]] = {}
        try:
            self.fingerprint = session.frontend.attachment_fingerprint()
        except ValueError:
            raise ProtocolError("The frontend reconnect snapshot is unavailable.", "resume_unavailable") from None
        gateway.resume_leases[token_key(self.token)] = self

    def receipt(self) -> dict:
        """Return authority only to the authenticated primary socket, never the fan-out or history endpoint."""
        return {"version": 1, "token": self.token, "generation": self.generation, "grace_seconds": self.seconds}

    async def operation(self, message: dict, connection: str, prepare: Callable[[], None] | None = None) -> bool:
        """Return whether this new logical input may run; exact retries retrieve evidence without dispatching again."""
        operation_id = message.get("operation_id")
        if not isinstance(operation_id, str) or not 1 <= len(operation_id) <= 128:
            raise ProtocolError("Reconnectable call inputs require a stable operation_id", param="operation_id")
        payload = {key: value for key, value in message.items() if key not in {"event_id", "operation_id"}}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        previous = self.operations.get(operation_id)
        if previous is not None:
            if previous[0] != digest:
                raise ProtocolError("operation_id already names different input", "invalid_value", "operation_id")
            await self.session.fanout.send_to(
                connection, event("session.operation.receipt", operation_id=operation_id, status=previous[1])
            )
            return False
        if len(self.operations) >= 512:
            raise ProtocolError("The call's operation receipt limit was reached", "rate_limit_exceeded")
        # Preparation is synchronous and side-effect free, under the command lock. A definite rejection
        # consumes no operation. Reserve before the first accepted mutation; any later failure stays unknown.
        if prepare is not None:
            prepare()
        self.operations[operation_id] = digest, "unknown"
        return True

    async def operation_recorded(self, message: dict, connection: str) -> None:
        """Record gateway admission separately from any claim of backend completion or external execution."""
        operation_id = message["operation_id"]
        digest, _ = self.operations[operation_id]
        self.operations[operation_id] = digest, "recorded"
        await self.session.fanout.send_to(
            connection, event("session.operation.receipt", operation_id=operation_id, status="recorded")
        )

    async def _owned_cleanup(self, coroutine) -> None:
        """Bound attachment waiting while retaining ownership of external/native cleanup."""
        task = asyncio.create_task(coroutine, name=f"{self.session.id}:attachment-cleanup")
        self.cleanup.add(task)

        def done(completed):
            self.cleanup.discard(completed)
            if not completed.cancelled() and completed.exception() is not None:
                logger.error("Retained attachment cleanup failed: {}", type(completed.exception()).__name__)

        task.add_done_callback(done)
        async with asyncio.timeout(max(0, (self.deadline or time.monotonic()) - time.monotonic())):
            await asyncio.shield(task)

    def _fail_recovery(self, detail: str, *, reason: str = "connection_lost") -> None:
        """Withdraw attachment authority before letting the session's close owner finish cleanup."""
        if self.state not in {"ending", "ended"}:
            session = self.session
            self.state = session.attachment_state = "ending"
            session.spawn(session.close(reason, asked_by=detail), "reconnect-failed")

    def mailbox_full(self) -> None:
        """End a call with an incomplete recovery journal and preserve the capacity outcome on the wire."""
        self._fail_recovery("the reconnect result mailbox is full", reason="rate_limit_exceeded")

    async def lost(self, socket: PrimarySocketView | None = None) -> None:
        """Fence an old generation before closing its frontend; keep its core lease and accepted readers."""
        async with self.lock:
            session = self.session
            if (
                self.state != "active"
                or session.status != "active"
                or (socket is not None and socket is not session.socket)
            ):
                return
            self.state = session.attachment_state = "reconnecting"
            self.deadline = time.monotonic() + self.seconds
            self.expiry = asyncio.create_task(self._expire(self.generation), name=f"{session.id}:reconnect-expiry")
            session.attachment_generation += 1
            log_observation(
                "reconnect.detached",
                session_id=session.id,
                generation=self.generation,
                attachment_generation=session.attachment_generation,
            )
            old_socket, session.socket = session.socket, None
            old_subscription = None
            if old_socket is not None:
                # This exact socket/generation is retained; later replacement state cannot rewrite its close.
                old_socket.retained_loss = True
                old_socket.left.set()
                if old_socket.connection is not None:
                    old_subscription = session.fanout.detach(old_socket.connection)
            try:
                # Close caller authority first. Neither a closed frontend nor a socket close releases the core.
                await self._owned_cleanup(session.delegation.suspend_attachment())
                if old_subscription is not None:
                    await self._owned_cleanup(old_subscription.close())
                async with asyncio.timeout(max(0, self.deadline - time.monotonic())):
                    self.snapshot = await _call(session.frontend, "export_attachment")
                await self._owned_cleanup(session.frontend.close())
            except Exception:
                logger.error("A reconnect attachment could not be fenced or closed")
                log_observation("reconnect.cleanup_failed", session_id=session.id, generation=self.generation)
                self._fail_recovery("reconnect attachment cleanup failed")
                return

    async def _expire(self, generation: int) -> None:
        try:
            await asyncio.sleep(max(0, (self.deadline or 0) - time.monotonic()))
            async with self.lock:
                if self.state not in {"reconnecting", "attaching"} or self.generation != generation:
                    return
                self.state = "ending"
                log_observation("reconnect.expired", session_id=self.session.id, generation=generation)
            await self.session.close("connection_lost", asked_by="the reconnect grace expired")
        except asyncio.CancelledError:
            pass

    async def attach(self, socket: PrimarySocketView, resume: dict, scope: str) -> dict:
        """Claim once, construct fresh media, restore context, then enable new input under the same core lease."""
        async with self.lock:
            session = self.session
            token, generation, request_id = resume.get("token"), resume.get("generation"), resume.get("request_id")
            if (
                not isinstance(token, str)
                or not isinstance(request_id, str)
                or not 1 <= len(request_id) <= 128
                or type(resume.get("version")) is not int
                or resume.get("version") != 1
                or type(generation) is not int
                or not hmac.compare_digest(scope, self.scope)
            ):
                raise unavailable()
            key = token_key(token)
            if session.status != "active" or self.state in {"ending", "ended"}:
                raise unavailable()
            if self.previous_key == key and self.last_request == request_id and generation == self.generation - 1:
                # The receipt of this exact successful request is safe to retrieve, but never creates a second owner.
                return {"receipt_only": True, "reconnect": self.receipt(), "request_id": request_id}
            if (
                key != token_key(self.token)
                or generation != self.generation
                or self.state != "reconnecting"
                or session.status != "active"
                or self.deadline is None
                or time.monotonic() >= self.deadline
            ):
                raise unavailable()
            deadline = self.deadline
            session.attachment_state = self.state = "attaching"
            parts = None
            try:
                async with asyncio.timeout(max(0, deadline - time.monotonic())):
                    parts = await self.gateway.factory(self.frontend_config, "websocket", None)
                    if self.state != "attaching" or session.status != "active":
                        raise unavailable()
                    if parts.connection is not None or parts.frontend.attachment_fingerprint() != self.fingerprint:
                        raise unavailable()
                    session.parts, session.frontend, session.socket = parts, parts.frontend, socket
                    session.new_adapter_view()
                    socket.attach(session)
                    await session.frontend.start(session.adapter_view, session.delegation_port)
                    if self.state != "attaching" or session.status != "active":
                        raise unavailable()
                    await _call(session.frontend, "restore_attachment", self.snapshot)
                    current = await session.delegation.resume_attachment()
                    if callable(getattr(session.frontend, "prepare_resume", None)):
                        await _call(session.frontend, "prepare_resume")
                    if self.state != "attaching" or session.status != "active" or time.monotonic() >= deadline:
                        raise unavailable()
            except BaseException:
                session.attachment_generation += 1
                if session.socket is socket:
                    session.socket = None

                async def clean_failed_attach():
                    try:
                        if socket.connection is not None:
                            await session.unsubscribe(socket.connection)
                    finally:
                        try:
                            await session.delegation.suspend_attachment()
                        finally:
                            if parts is not None:
                                await parts.frontend.close()

                cleaned = False
                try:
                    await self._owned_cleanup(clean_failed_attach())
                    cleaned = True
                except (TimeoutError, asyncio.CancelledError):
                    pass
                except Exception:
                    logger.error("A failed resume could not restore the caller fence")
                log_observation("reconnect.resume_failed", session_id=session.id, generation=self.generation)
                if cleaned and self.state == "attaching" and session.status == "active" and time.monotonic() < deadline:
                    self.state = session.attachment_state = "reconnecting"
                else:
                    self._fail_recovery("reconnect could not restore a clean attachment before the grace expired")
                raise
            old_key = token_key(self.token)
            if self.previous_key is not None:
                self.gateway.resume_leases.pop(self.previous_key, None)
            self.previous_key, self.last_request = old_key, request_id
            self.token = secrets.token_urlsafe(32)
            self.gateway.resume_leases[token_key(self.token)] = self
            self.generation += 1
            self.state = session.attachment_state = "active"
            log_observation(
                "reconnect.resumed",
                session_id=session.id,
                generation=self.generation,
                attachment_generation=session.attachment_generation,
            )
            try:
                async with asyncio.timeout(max(0, deadline - time.monotonic())):
                    await session.fanout.send_to(
                        socket.connection,
                        event(
                            "session.resumed",
                            session=session.snapshot(),
                            reconnect=self.receipt(),
                            pending=session.delegation.mailbox.snapshot(),
                            request_id=request_id,
                        ),
                    )
                    await session.delegation.relay_pending_calls(current)
                    if self.state != "active" or session.status != "active":
                        raise unavailable()
                    if callable(getattr(session.frontend, "resume_pending", None)):
                        await _call(session.frontend, "resume_pending")
                    if self.state != "active" or session.status != "active" or time.monotonic() >= deadline:
                        raise unavailable()
            except asyncio.CancelledError:
                self._fail_recovery("reconnect activation was interrupted")
                raise
            except Exception:
                # The receipt or an app call may already be on the wire. This attachment cannot be retried.
                self._fail_recovery("reconnect activation failed or exceeded the grace")
                raise unavailable() from None
            self.deadline = None
            if self.expiry is not None:
                self.expiry.cancel()
            return {"receipt_only": False}

    def revoke(self) -> None:
        """End authority on explicit hang-up, expiry, shutdown or logical close."""
        self.state = "ended"
        if self.expiry is not None and self.expiry is not asyncio.current_task():
            self.expiry.cancel()
        self.gateway.resume_leases.pop(token_key(self.token), None)
        if self.previous_key is not None:
            self.gateway.resume_leases.pop(self.previous_key, None)
