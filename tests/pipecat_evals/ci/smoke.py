# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke checks for production Pipecat Eval integration used by CI."""

from __future__ import annotations

import asyncio
import json
from argparse import Namespace
from pathlib import Path

from pipecat.evals.fastapi_transport import EvalFastAPIWebsocketTransport
from pipecat.runner.types import WebSocketRunnerArguments

import eval_bot
from attachment_store import clear_session_attachments, latest_attachment
from examples.frontend_backend_agent.pipeline import create_transport as create_frontend_backend_transport
from examples.omni_assistant.pipeline import create_transport as create_omni_transport
from examples.shared.pipeline_utils import create_transport as create_shared_transport

ROOT = Path(__file__).resolve().parents[3]


class _AcceptedEvalWebSocket:
    def iter_text(self):
        async def messages():
            yield json.dumps(
                {
                    "label": "rtvi-ai",
                    "type": "client-message",
                    "data": {"t": "eval-connect", "d": {"skip_tts": True}},
                }
            )

        return messages()


def _assert_eval_transport(name: str, transport) -> None:
    if not isinstance(transport, EvalFastAPIWebsocketTransport):
        raise AssertionError(f"{name} did not return EvalFastAPIWebsocketTransport: {type(transport).__name__}")


async def _check_transport_factories() -> None:
    for name, factory in (
        ("shared", create_shared_transport),
        ("omni_assistant", create_omni_transport),
        ("frontend_backend_agent", create_frontend_backend_transport),
    ):
        args = WebSocketRunnerArguments(websocket=_AcceptedEvalWebSocket())
        _assert_eval_transport(name, await factory(args))


def _check_eval_bot_uploaded_attachment_body() -> None:
    body_path = ROOT / "tests/pipecat_evals/service/runner_bodies/omni_uploaded_image.json"
    body = json.loads(body_path.read_text(encoding="utf-8"))
    args = WebSocketRunnerArguments(
        websocket=_AcceptedEvalWebSocket(),
        body=body,
        session_id=str(body.get("session_id") or ""),
    )
    args.cli_args = Namespace(runner_body=str(body_path))
    example = eval_bot._select_example(body)
    prepared = eval_bot._prepare_body(body, example, args)

    try:
        eval_bot._preload_eval_attachment(body, prepared["session_id"], args)
        attachment = latest_attachment(prepared["session_id"])
        if attachment is None:
            raise AssertionError("eval attachment was not stored")
        if attachment.kind != "image" or attachment.content_type != "image/png" or not attachment.data:
            raise AssertionError(f"unexpected eval attachment metadata: {attachment.metadata()}")
    finally:
        clear_session_attachments(prepared["session_id"])


async def main() -> None:
    """Run fast smoke checks for eval-ready production entrypoints."""
    await _check_transport_factories()
    _check_eval_bot_uploaded_attachment_body()
    print("Pipecat eval smoke checks passed")


if __name__ == "__main__":
    asyncio.run(main())
