#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Submit one arbitrary committed turn through the configured NemoClaw adapter."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
from pathlib import Path

from voiceclaw.adapters.nemoclaw.factory import build_nemoclaw_backend
from voiceclaw.composition import compose_backends
from voiceclaw.config import BackendProfile, load_config
from voiceclaw.model_contracts import load_model_contract_catalog
from voiceclaw.ports.turns import CommittedTurnRequest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Arbitrary finalized user request to delegate")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="VoiceClaw YAML configuration (or set VOICECLAW_CONFIG)",
    )
    parser.add_argument(
        "--origin",
        help="Legacy direct gateway origin; use with --credential-file instead of --config",
    )
    parser.add_argument(
        "--credential-file",
        type=Path,
        help="Legacy direct bearer file; use instead of --config",
    )
    parser.add_argument("--conversation-id", default="", help="Stable VoiceClaw conversation identifier")
    return parser


async def _run(arguments: argparse.Namespace) -> None:
    config_path = arguments.config
    legacy_requested = arguments.origin is not None or arguments.credential_file is not None
    if config_path is not None and legacy_requested:
        raise ValueError("--config cannot be combined with --origin or --credential-file")
    if config_path is None and not legacy_requested:
        configured_path = os.getenv("VOICECLAW_CONFIG", "").strip()
        config_path = Path(configured_path) if configured_path else None

    if config_path is not None:
        config = load_config(config_path)
        adapter = compose_backends(config, environ=os.environ).turn_backend
    else:
        if arguments.credential_file is None:
            raise ValueError("--config, VOICECLAW_CONFIG, or --credential-file is required")
        settings = {"mode": "response_only", "endpoint_policy": "loopback_only"}
        if arguments.origin is not None:
            settings["endpoint"] = arguments.origin
        adapter, _status = build_nemoclaw_backend(
            BackendProfile(
                kind="nemoclaw",
                credential_file=str(arguments.credential_file.absolute()),
                settings=settings,
            ),
            os.environ,
            load_model_contract_catalog(),
        )
    if adapter is None:
        raise ValueError("the selected backend does not support committed turns")
    backend = await adapter.inspect()
    conversation_id = arguments.conversation_id.strip() or f"voiceclaw-smoke-{secrets.token_hex(8)}"
    result = await adapter.commit_turn(
        CommittedTurnRequest(
            runtime_conversation_id=conversation_id,
            commit_id=f"commit-{secrets.token_hex(8)}",
            text=arguments.query,
        )
    )
    print(
        json.dumps(
            {
                "backend": backend.label,
                "mode": backend.mode,
                "conversation_id": conversation_id,
                "backend_session_id": result.backend_session_id,
                "turn_id": result.turn_id,
                "response_id": result.response_id,
                "display_text": result.display_text,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    """Parse CLI arguments and execute one response-only delegation."""
    arguments = _parser().parse_args()
    try:
        asyncio.run(_run(arguments))
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"NemoClaw delegation failed: {error}") from error


if __name__ == "__main__":
    main()
