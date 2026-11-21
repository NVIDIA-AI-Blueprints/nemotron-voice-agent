# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verified native guidance for an ACP worker on an observer-enabled Hermes profile."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Literal
from uuid import uuid4

from examples.frontend_backend_live.delegation_core.backend.port import Delivery

from ...protocol import MessageChunk, NativeSteerReceipt, PromptEnded
from .evidence import INPUT, INVOCATION, PRIVATE_ORIGIN, ModelOrigin, OriginStore
from .gateway_control import GatewayProcess


@dataclass
class _Update:
    input_id: str
    marker: str
    receipt: asyncio.Future[NativeSteerReceipt]
    phase: Literal["unsent", "sending", "finished"] = "unsent"


@dataclass
class _Foreground:
    prompt: int
    invocation: str
    marker: str
    text: str
    required: list[str] = field(default_factory=list)
    pending: _Update | None = None
    messages: dict[str, list[str]] = field(default_factory=dict)
    used_calls: set[tuple[str, float, str]] = field(default_factory=set)
    native: tuple[str, str] | None = None


class HermesSteering:
    """Own steering and effect evidence for one local Hermes worker."""

    def __init__(self, scope: OriginStore, session_id: str, home: Path, env: Mapping[str, str], timeout: float):
        """Bind control and proof to a session's private scope."""
        self.scope, self.session_id, self.home, self.env, self.timeout = scope, session_id, home, env, timeout
        self.foreground: _Foreground | None = None
        self.commands: set[asyncio.Task] = set()

    def prompt(self, number: int, invocation: str, text: str) -> str:
        """Give the active prompt one host-owned marker before ACP sees it."""
        marker = uuid4().hex
        self.scope.begin(marker)
        authored = f"{INVOCATION}{marker}\n{text}"
        self.foreground = _Foreground(number, invocation, marker, authored)
        return authored

    def submit(
        self, prompt: int, invocation_id: str, input_id: str, text: str
    ) -> asyncio.Future[NativeSteerReceipt] | None:
        """Offer a correction only when the signed observer is ready for the current invocation."""
        current = self.foreground
        if (
            current is None
            or (current.prompt, current.invocation) != (prompt, invocation_id)
            or current.pending is not None
            or (ready := self.scope.started(current.marker)) is None
        ):
            return None
        identity = ready[0].session, ready[0].turn
        if current.native is not None and current.native != identity:
            return None
        current.native = identity
        marker = uuid4().hex
        self.scope.issue_input(current.marker, marker)
        update = _Update(input_id, marker, asyncio.get_running_loop().create_future())
        current.pending = update
        current.required.append(marker)
        sending = asyncio.create_task(self._send(current, update, ready[1], text), name="hermes-native-steer")
        self.commands.add(sending)
        sending.add_done_callback(self.commands.discard)
        return update.receipt

    async def _send(self, current: _Foreground, update: _Update, interpreter: Path, text: str) -> None:
        try:
            async with GatewayProcess.connect(
                interpreter, self.home, self.session_id, env=self.env, timeout=self.timeout
            ) as controller:
                turn = await controller.bind(current.text)
                if (
                    turn is None
                    or self.foreground is not current
                    or update.receipt.cancelled()
                    or self.scope.started(current.marker) is None
                ):
                    self._settle(current, update, "not_applied")
                    return
                update.phase = "sending"
                try:
                    reply = await controller.steer(turn, f"{INPUT}{update.marker}\n{text}", redirect=True)
                    if reply.status == "not_sent":
                        self._settle(current, update, "not_applied")
                finally:
                    update.phase = "finished"
                    if self.foreground is not current:
                        self._settle(current, update, "unknown")
        except (OSError, ValueError, TypeError, TimeoutError):
            if update.phase == "unsent":
                self._settle(current, update, "not_applied")

    def _settle(
        self, current: _Foreground, update: _Update, result: Delivery, model: ModelOrigin | None = None
    ) -> None:
        if current.pending is not update:
            return
        current.pending = None
        if result == "not_applied":
            current.required.remove(update.marker)
        if not update.receipt.done():
            request = f"{model.request}@{model.started_at!r}" if model is not None else None
            update.receipt.set_result(NativeSteerReceipt(current.invocation, update.input_id, result, request))

    def admit_tool(self, name: str, arguments: dict) -> tuple[dict, Callable[[], bool]] | None:
        """Verify one tool and return its original args and a guard for delayed admission."""
        current = self.foreground
        if current is None:
            return None
        if current.native is None:
            if (ready := self.scope.started(current.marker)) is not None:
                current.native = (ready[0].session, ready[0].turn)
            elif PRIVATE_ORIGIN not in arguments and not current.required:
                return arguments, lambda: self.foreground is current and current.native is None and not current.required
        args = dict(arguments)
        reference = args.pop(PRIVATE_ORIGIN, None)
        try:
            proof = self.scope.tool_receipt(reference, name, args)
        except (OSError, ValueError, TypeError):
            return None
        model = proof.model

        def current_origin() -> bool:
            return (
                self.foreground is current
                and model.invocation == current.marker
                and (model.session, model.turn) == current.native
                and all(marker in model.inputs for marker in current.required)
            )

        identity = model.request, model.started_at, proof.call
        if not current_origin() or identity in current.used_calls:
            return None
        current.used_calls.add(identity)
        if (update := current.pending) is not None and update.marker in model.inputs:
            self._settle(current, update, "applied", model)
        return args, current_origin

    def capture(self, event: MessageChunk) -> bool:
        """Hold text after steering until the original ACP turn supplies its terminal."""
        current = self.foreground
        if current is None or event.prompt != current.prompt or not (current.required or current.messages):
            return False
        if event.message_id:
            current.messages.setdefault(event.message_id, []).append(event.text)
        return True

    def finish(self, ended: PromptEnded) -> MessageChunk | None:
        """Retire the turn, verify its output, and leave any outstanding gateway reply to settle."""
        current = self.foreground
        if current is None or ended.prompt != current.prompt:
            return None
        self.foreground = None
        if not current.required:
            return None
        identity = current.required[-1]
        proof = self.scope.model_receipt(current.marker, identity)
        model, digest, length = proof if proof is not None else (None, None, None)
        if model is not None and (
            (model.session, model.turn) != current.native
            or not all(marker in model.inputs for marker in current.required)
        ):
            model, digest = None, None
        if (update := current.pending) is not None:
            if model is not None:
                self._settle(current, update, "applied", model)
            elif update.phase == "unsent":
                self._settle(current, update, "not_applied")
            elif update.phase == "finished":
                self._settle(current, update, "unknown")
        if (
            digest is None
            or length == 0
            or not ended.terminal_confirmed
            or ended.stop_reason != "end_turn"
            or ended.error
        ):
            return None
        for identifier, chunks in reversed(current.messages.items()):
            text = "".join(chunks).strip()
            # Hermes can put the final reply after stale text under the same ACP message ID.
            if length is not None:
                text = text[-length:]
            if sha256(text.encode()).hexdigest() == digest:
                return MessageChunk(identifier, text, current.prompt)
        return None

    async def close(self) -> None:
        """Join locally owned helper requests when ACP closes the session."""
        self.foreground = None
        for command in self.commands:
            command.cancel()
        if self.commands:
            await asyncio.gather(*self.commands, return_exceptions=True)
