# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Smoke checks for production Pipecat Eval integration used by CI."""

from __future__ import annotations

import json
from pathlib import Path

from pipecat.evals.transport import EvalTransport
from pipecat.runner.types import EvalRunnerArguments

import eval_bot
from attachment_store import clear_session_attachments, latest_attachment
from examples.frontend_backend_agent.pipeline import _create_transport as create_frontend_backend_transport
from examples.omni_assistant.pipeline import _create_transport as create_omni_transport
from examples.shared.pipeline_utils import create_transport as create_shared_transport

ROOT = Path(__file__).resolve().parents[3]


def _assert_eval_transport(name: str, transport) -> None:
    if not isinstance(transport, EvalTransport):
        raise AssertionError(f"{name} did not return EvalTransport: {type(transport).__name__}")


def _check_transport_factories() -> None:
    args = EvalRunnerArguments(host="127.0.0.1", port=7999)
    _assert_eval_transport("shared", create_shared_transport(args))
    _assert_eval_transport("omni_assistant", create_omni_transport(args))
    _assert_eval_transport("frontend_backend_agent", create_frontend_backend_transport(args))


def _check_eval_bot_uploaded_attachment_body() -> None:
    body_path = ROOT / "tests/pipecat_evals/service/runner_bodies/omni_uploaded_image.json"
    body = json.loads(body_path.read_text(encoding="utf-8"))
    args = EvalRunnerArguments(body=body, session_id=str(body.get("session_id") or ""))
    example = eval_bot._select_example(body)
    prepared = eval_bot._prepare_body(body, example, args)

    try:
        eval_bot._preload_eval_attachment(body, prepared["session_id"])
        attachment = latest_attachment(prepared["session_id"])
        if attachment is None:
            raise AssertionError("eval attachment was not stored")
        if attachment.kind != "image" or attachment.content_type != "image/png" or not attachment.data:
            raise AssertionError(f"unexpected eval attachment metadata: {attachment.metadata()}")
    finally:
        clear_session_attachments(prepared["session_id"])


def main() -> None:
    """Run fast smoke checks for eval-ready production entrypoints."""
    _check_transport_factories()
    _check_eval_bot_uploaded_attachment_body()
    print("Pipecat eval smoke checks passed")


if __name__ == "__main__":
    main()
