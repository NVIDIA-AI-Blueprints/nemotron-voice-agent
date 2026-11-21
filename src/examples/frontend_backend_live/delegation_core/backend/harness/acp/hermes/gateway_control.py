# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation-bound control of an existing Hermes gateway.

The helper runs in the Hermes environment. ACP continues to own the prompts
and the terminal outcomes. A successful gateway command only queues guidance.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class GatewayTurn:
    """Identity of a started ACP admission on one gateway authority."""

    admission_id: str
    authority_epoch: int
    execution_generation: int


@dataclass(frozen=True)
class GatewaySteer:
    """Transmission status from the gateway, independent of model delivery."""

    status: Literal["queued", "not_sent", "unknown"]
    reason: str | None = None
    turn: GatewayTurn | None = None


class GatewayControl:
    """Use one authenticated connection and one exact native session."""

    def __init__(self, client: Any, session_id: str):
        """Bind this controller to the native identity supplied by ACP."""
        self.client, self.session_id = client, session_id

    async def bind(self, submitted_text: str) -> GatewayTurn | None:
        """Match the host-marked ACP prompt to a single running admission."""
        snapshot = await self.client.rpc("session.resume", session_id=self.session_id, omit_messages=True)
        if snapshot.get("running") is not True:
            return None
        matches = [
            admission
            for admission in snapshot.get("pending", ())
            if admission.get("status") == "started"
            and admission.get("text") == submitted_text
            and admission.get("steering") is False
        ]
        if len(matches) != 1:
            return None
        row = matches[0]
        admission_id = row.get("admission_id")
        epoch, generation = row.get("authority_epoch"), row.get("execution_generation")
        if (
            not isinstance(admission_id, str)
            or not admission_id
            or type(epoch) is not int
            or type(generation) is not int
            or epoch != snapshot.get("authority_epoch")
            or generation != snapshot.get("execution_generation")
        ):
            return None
        selected = GatewayTurn(admission_id, epoch, generation)
        return selected if await self._owns_running(selected) else None

    async def _owns_running(self, turn: GatewayTurn) -> bool:
        receipt = await self.client.rpc("prompt.receipt", session_id=self.session_id, admission_id=turn.admission_id)
        return receipt.get("status") == "started" and all(
            receipt.get(field) == value for field, value in asdict(turn).items()
        )

    async def steer(self, turn: GatewayTurn, text: str, *, redirect: bool = False) -> GatewaySteer:
        """Submit guidance only if it still belongs to the same running invocation."""
        from hermes_cli import gateway_client

        # No control request has been sent if the admission check fails or raises.
        try:
            running = await self._owns_running(turn)
        except gateway_client.GatewayClientError:
            return GatewaySteer("not_sent", "binding_unavailable", turn)
        if not running:
            return GatewaySteer("not_sent", "stale_generation", turn)
        method = "session.redirect" if redirect else "session.steer"
        try:
            reply = await self.client.rpc(
                method, session_id=self.session_id, text=text, execution_generation=turn.execution_generation
            )
        except gateway_client.GatewayClientError as error:
            if isinstance(error, gateway_client.GatewayRPCError) and str(error) == "stale_generation":
                return GatewaySteer("not_sent", "stale_generation", turn)
            not_sent = getattr(gateway_client, "NOT_SENT", None)
            if (
                type(error) is gateway_client.GatewayClientError
                and isinstance(not_sent, str)
                and error.args == (not_sent,)
            ):
                return GatewaySteer("not_sent", "transport_not_sent", turn)
            return GatewaySteer("unknown", "control_rpc_failed", turn)
        if reply.get("authority_epoch") != turn.authority_epoch or (
            reply.get("execution_generation") != turn.execution_generation
        ):
            return GatewaySteer("unknown", "different_generation", turn)
        if reply.get("status") == ("redirected" if redirect else "queued"):
            return GatewaySteer("queued", turn=turn)
        if reply.get("status") == "rejected" and reply.get("delivered") is False:
            return GatewaySteer("not_sent", "runtime_refused", turn)
        return GatewaySteer("unknown", "control_outcome_unconfirmed", turn)


class GatewayProcess:
    """The example's pipe to the control helper in the Hermes interpreter."""

    def __init__(self, process: asyncio.subprocess.Process, timeout: float):
        """Own request serialization for the lifetime of this process."""
        self.process, self.timeout = process, timeout
        self._lock = asyncio.Lock()
        self._failed = False

    @classmethod
    @asynccontextmanager
    async def connect(
        cls,
        interpreter: Path,
        home: Path,
        session_id: str,
        *,
        env: Mapping[str, str],
        timeout: float,
    ) -> AsyncIterator[GatewayProcess]:
        """Start and join one controller without taking ownership of a gateway daemon."""
        process = await asyncio.create_subprocess_exec(
            str(interpreter),
            "-B",
            str(Path(__file__)),
            str(home),
            session_id,
            env={**env, "HERMES_HOME": str(home)},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            yield cls(process, timeout)
        except asyncio.CancelledError:
            with suppress(ProcessLookupError):
                process.terminate()
            raise
        finally:
            process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    process.kill()
                await process.wait()

    async def _request(self, action: str, *args: Any) -> Any:
        async with self._lock:
            if self._failed:
                raise ConnectionError("Hermes gateway control pipe has an unconfirmed reply")
            try:
                async with asyncio.timeout(self.timeout):
                    stdin, stdout = self.process.stdin, self.process.stdout
                    stdin.write(json.dumps([action, args]).encode() + b"\n")
                    await stdin.drain()
                    line = await stdout.readline()
                    if not line:
                        raise ConnectionError("Hermes gateway control pipe ended")
                    return json.loads(line)
            except BaseException:
                # A later reply must never be read as the next operation's result.
                self._failed = True
                raise

    async def bind(self, submitted_text: str) -> GatewayTurn | None:
        """Resolve the admission of the exact marked prompt."""
        reply = await self._request("bind", submitted_text)
        return GatewayTurn(**reply) if reply is not None else None

    async def steer(self, turn: GatewayTurn, text: str, *, redirect: bool = False) -> GatewaySteer:
        """Return gateway transmission evidence while leaving model delivery unconfirmed."""
        try:
            reply = await self._request("redirect" if redirect else "steer", asdict(turn), text)
        except (OSError, TimeoutError, ValueError):
            return GatewaySteer("unknown", "control_pipe_failed", turn)
        if not isinstance(reply, dict) or reply.get("turn") != asdict(turn):
            return GatewaySteer("unknown", "different_generation", turn)
        status = reply.get("status")
        if status not in ("queued", "not_sent"):
            return GatewaySteer("unknown", "control_outcome_unconfirmed", turn)
        reason = reply.get("reason")
        known_reasons = ("stale_generation", "binding_unavailable", "runtime_refused", "transport_not_sent")
        return GatewaySteer(status, reason if reason in known_reasons else None, turn)


async def serve(home: Path, session_id: str) -> None:
    """Serve the adapter's local control pipe without starting another gateway."""
    from hermes_cli.gateway_client import GatewayClientError, connect_gateway
    from hermes_cli.gateway_runtime import discover_gateway_endpoint

    discovered = await asyncio.to_thread(discover_gateway_endpoint, home)
    if discovered.state != "ready" or discovered.endpoint is None:
        raise GatewayClientError("Hermes steering requires an already running gateway")
    async with connect_gateway(endpoint=discovered.endpoint) as client:
        controller = GatewayControl(client, session_id)
        while line := await asyncio.to_thread(sys.stdin.buffer.readline):
            action, args = json.loads(line)
            if action == "bind":
                selected = await controller.bind(*args)
                response = asdict(selected) if selected is not None else None
            elif action in ("steer", "redirect"):
                binding, text = args
                result = await controller.steer(GatewayTurn(**binding), text, redirect=action == "redirect")
                response = asdict(result)
            else:
                raise ValueError("Unknown Hermes control operation")
            print(json.dumps(response), flush=True)


if __name__ == "__main__":
    asyncio.run(serve(Path(sys.argv[1]), sys.argv[2]))
