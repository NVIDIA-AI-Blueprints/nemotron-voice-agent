# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker process entry point (plan section 9.2).

``python -m examples.frontend_delegation_hermes.core.worker.worker_main --socket PATH
--agent-kind hermes|fake --session-id ID``

Connects to the gateway's Unix socket and speaks JSON lines through one
:class:`QueueWriter`. stdout/stderr are never used for the protocol (Hermes and its
libraries may print); the gateway redirects them to a per-worker log file. The
process exits when ``close`` finished or the socket closes.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
from typing import Any

from examples.frontend_delegation_hermes.core.backend import protocol as proto
from examples.frontend_delegation_hermes.core.backend.writer import QueueWriter
from examples.frontend_delegation_hermes.core.worker.tool_futures import ToolFutures
from examples.frontend_delegation_hermes.core.worker.worker_core import (
    ActivityEmit,
    AgentHost,
    WorkerCore,
    ready_message,
)

logger = logging.getLogger("fdh.worker")

#: Frames may carry the whole settled history.
STREAM_LIMIT = 64 * 1024 * 1024


def make_host_factory(agent_kind: str, session_id: str) -> Any:
    """Host factory for ``agent_kind``; Hermes is imported only for ``hermes``."""
    if agent_kind == "hermes":
        from examples.frontend_delegation_hermes.core.worker.hermes_adapter import HermesAgentHost  # noqa: PLC0415

        def factory(futures: ToolFutures, on_activity: ActivityEmit) -> AgentHost:
            return HermesAgentHost(futures, on_activity, session_id=session_id)

        return factory
    if agent_kind == "fake":
        from examples.frontend_delegation_hermes.core.worker.fake_host import FakeAgentHost  # noqa: PLC0415

        return lambda futures, on_activity: FakeAgentHost(futures, on_activity)
    raise ValueError(f"unknown agent kind {agent_kind!r}")


async def serve(socket_path: str, agent_kind: str, session_id: str) -> None:
    """Connect, announce ``ready``, serve commands until closed."""
    reader, writer = await asyncio.open_unix_connection(socket_path, limit=STREAM_LIMIT)

    async def send(frame: str) -> None:
        writer.write(frame.encode("utf-8") + b"\n")
        await writer.drain()

    out = QueueWriter(send, name="worker-writer")
    out.start()
    done = asyncio.Event()

    def emit(data: dict[str, Any]) -> None:
        fields = dict(data)
        kind = fields.pop("type")
        out.put_threadsafe(proto.encode(proto.WORKER_TO_GATEWAY, kind, **fields))

    core = WorkerCore(
        session_id=session_id,
        host_factory=make_host_factory(agent_kind, session_id),
        emit=emit,
        on_exit=done.set,
    )
    emit(ready_message(core.host))
    try:
        while not done.is_set():
            line = await reader.readline()
            if not line:
                break
            try:
                msg = proto.decode(proto.GATEWAY_TO_WORKER, line)
            except proto.ProtocolError as exc:
                logger.error("bad frame from gateway: %s", exc)
                continue
            await core.handle(msg)
    finally:
        if not core.exited:
            await core.close("link_lost")
        await asyncio.sleep(0)  # let call_soon_threadsafe puts land before draining
        await out.close(drain=True, timeout=5.0)
        with contextlib.suppress(Exception):
            writer.close()


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="fdh Hermes worker")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--agent-kind", choices=("hermes", "fake"), default="hermes")
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("FDH_WORKER_LOG_LEVEL", "INFO"),
        format=f"%(asctime)s %(levelname)s [worker {args.session_id} pid={os.getpid()}] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        asyncio.run(serve(args.socket, args.agent_kind, args.session_id))
    except Exception:  # noqa: BLE001 - logged; the gateway sees the socket close
        logger.exception("worker failed")
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        # Hermes may leave non-daemon threads (a hung tool); never let them keep the process alive.
        os._exit(0)


if __name__ == "__main__":
    main()
